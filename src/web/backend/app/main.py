from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Any
import os
import sys
import json
import tempfile
import asyncio
import uuid

# Reduce random native runtime crashes in mixed NumPy/PyTorch OpenMP environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np
import pandas as pd
import xarray as xr
import uuid

# Ensure src and root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # adds src/
sys.path.insert(0, PROJECT_ROOT) # adds root

from src.element_forecasting.predictor import ElementForecastPredictor
from src.element_forecasting.dataset import ElementForecastWindowDataset
from src.anomaly_detection.dataset import AnomalyFrameDataset
from src.eddy_detection.dataset import EddySegmentationDataset
from src.eddy_detection.model import EddyUNet
from src.eddy_detection.predictor import infer_batch_to_objects, load_checkpoint
from src.eddy_detection.postprocess import extract_eddy_objects

app = FastAPI(title="OceanRace Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root_status():
    return {
        "service": "OceanRace Backend API",
        "status": "ok",
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

try:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    # Keep backend boot robust even if thread controls are unavailable.
    pass

def resolve_path(p: str) -> str:
    """Resolve a path relative to the project root."""
    if os.path.isabs(p):
        return p
    return os.path.join(PROJECT_ROOT, p)

class DatasetInfoRequest(BaseModel):
    data_path: str

class PredictRequest(BaseModel):
    model_path: str
    data_path: str
    start_idx: int


class EddyDatasetInfoRequest(BaseModel):
    data_path: str
    input_steps: int = 1


class EddyPredictRequest(BaseModel):
    model_path: str
    data_path: str
    start_idx: int
    input_steps: int = 1
    base_channels: int = 32
    min_region_pixels: int = 16
    horizon_steps: int = 1


class EddyPredictDayRequest(BaseModel):
    model_path: str
    data_path: str
    day_index: int
    input_steps: int = 1
    base_channels: int = 32
    min_region_pixels: int = 16


class EddyDateIndexRequest(BaseModel):
    data_path: str
    date: str
    input_steps: int = 1


class AnomalyInspectRequest(BaseModel):
    labels_json: str
    events_json: str
    manifest_path: str = "data/processed/splits/anomaly_detection_competition.json"
    processed_dir: str = "data/processed/anomaly_detection"
    norm_stats_path: str = "data/processed/normalization/anomaly_detection_norm.json"
    split: str = "test"
    open_file_lru_size: int = 32
    max_points: int = 200

# In-memory store for the last prediction to serve slices efficiently
prediction_cache = {}
eddy_prediction_cache = {}
anomaly_timestamps_cache: dict[tuple[str, str, str], list[int]] = {}


def _to_epoch_seconds(ts_raw: int) -> int:
    t = int(ts_raw)
    at = abs(t)
    if at >= 10**17:
        return t // 10**9
    if at >= 10**14:
        return t // 10**6
    if at >= 10**11:
        return t // 10**3
    return t


def _build_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Build a 0/1 edge map from a label mask where foreground is (mask > 0)."""
    fg = (mask > 0).astype(np.uint8)
    if fg.ndim != 2:
        return np.zeros_like(fg, dtype=np.uint8)

    padded = np.pad(fg, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    interior = (center == 1) & (up == 1) & (down == 1) & (left == 1) & (right == 1)
    boundary = (center == 1) & (~interior)
    return boundary.astype(np.uint8)


def _default_eddy_paths() -> tuple[str, str]:
    split_manifest = resolve_path("data/processed/splits/eddy_merged_time.json")
    if os.path.exists(split_manifest):
        try:
            with open(split_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            clean_path = data.get("clean_nc", "")
            label_path = data.get("label_nc", "")
            clean_abs = resolve_path(clean_path) if clean_path and not os.path.isabs(clean_path) else clean_path
            label_abs = resolve_path(label_path) if label_path and not os.path.isabs(label_path) else label_path
            if clean_abs and os.path.exists(clean_abs):
                if label_abs and os.path.exists(label_abs):
                    return clean_path, label_path
                return clean_path, ""
        except Exception:
            pass
    clean_candidates = [
        "data/processed/eddy_detection/19930101_20241231_clean.nc",
    ]
    label_candidates = [
        "data/processed/eddy_detection/labels/19930101_20241231_label_meta4_mask_bg0.nc",
    ]

    clean_path = clean_candidates[0]
    for cand in clean_candidates:
        abs_path = resolve_path(cand) if not os.path.isabs(cand) else cand
        if os.path.exists(abs_path):
            clean_path = cand
            break

    label_path = label_candidates[0]
    for cand in label_candidates:
        abs_path = resolve_path(cand) if not os.path.isabs(cand) else cand
        if os.path.exists(abs_path):
            label_path = cand
            break

    return (clean_path, label_path)


def _default_eddy_model_path() -> str:
    candidates = [
        "outputs/final_results/eddy_detection/meta4_mask_retrain_20260413_bg/checkpoints/best.pt",
        "outputs/eddy_detection/runs/global_boundary_full_20260403/checkpoints/best.pt",
        "outputs/eddy_detection/checkpoints/best.pt",
        "models/eddy_model.pth",
    ]
    for rel in candidates:
        abs_path = resolve_path(rel)
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            return rel
    return candidates[0]


def _resolve_optional_path(p: str | None) -> str | None:
    if p is None:
        return None
    s = str(p).strip().strip('"\'')
    if not s:
        return None
    return resolve_path(s)


def _infer_eddy_run_tag_from_model(model_path: str) -> str:
    p = Path(model_path)
    parts = p.parts
    if "final_results" in parts and "eddy_detection" in parts:
        i = parts.index("eddy_detection")
        if i + 1 < len(parts):
            return parts[i + 1]
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "meta4_mask_retrain_20260413_bg"


def _find_eddy_reference_figure(run_tag: str, time_index: int) -> Path | None:
    figures_dir = Path(resolve_path(f"outputs/final_results/eddy_detection/{run_tag}/figures"))
    if not figures_dir.exists():
        return None

    candidates = sorted(figures_dir.glob(f"sample_*_t{int(time_index)}.png"))
    if candidates:
        return candidates[0]
    return None


def _load_eddy_norm_variables() -> dict[str, dict[str, float]]:
    norm_path = resolve_path("data/processed/normalization/eddy_norm.json")
    if not os.path.exists(norm_path):
        return {}
    try:
        with open(norm_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        variables = payload.get("variables", {})
        return variables if isinstance(variables, dict) else {}
    except Exception:
        return {}


def _standardize_eddy(arr: np.ndarray, var_name: str, stats: dict[str, dict[str, float]]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    meta = stats.get(var_name, {}) if isinstance(stats, dict) else {}
    mean = float(meta.get("mean", 0.0))
    std = float(meta.get("std", 1.0))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    out = (out - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _build_unlabeled_eddy_batch(
    *,
    data_path: str,
    start_idx: int,
    input_steps: int,
    horizon_steps: int,
) -> tuple[torch.Tensor, list[int], list[np.ndarray], int]:
    ds = xr.open_dataset(data_path)
    try:
        for v in ("adt", "ugos", "vgos"):
            if v not in ds:
                raise HTTPException(status_code=400, detail=f"Dataset missing variable: {v}")

        tlen = int(ds["adt"].shape[0])
        if tlen <= 0:
            raise HTTPException(status_code=400, detail="Dataset time axis is empty")

        max_index = max(0, tlen - input_steps)
        if start_idx < 0 or start_idx > max_index:
            raise HTTPException(status_code=400, detail="Start index out of range")

        end_idx = min(max_index + 1, start_idx + horizon_steps)
        stats = _load_eddy_norm_variables()

        x_list: list[torch.Tensor] = []
        time_indices: list[int] = []
        adt_maps: list[np.ndarray] = []

        for t in range(start_idx, end_idx):
            t0 = t
            t1 = t + input_steps
            adt_seq = np.asarray(ds["adt"].values[t0:t1], dtype=np.float32)
            ugos_seq = np.asarray(ds["ugos"].values[t0:t1], dtype=np.float32)
            vgos_seq = np.asarray(ds["vgos"].values[t0:t1], dtype=np.float32)

            adt_std = _standardize_eddy(adt_seq, "adt", stats)
            ugos_std = _standardize_eddy(ugos_seq, "ugos", stats)
            vgos_std = _standardize_eddy(vgos_seq, "vgos", stats)

            stacked = np.stack([adt_std, ugos_std, vgos_std], axis=1)  # (T,3,H,W)
            x_np = stacked.reshape(-1, stacked.shape[-2], stacked.shape[-1])
            x_list.append(torch.from_numpy(x_np))

            time_indices.append(int(t + input_steps - 1))
            adt_maps.append(np.asarray(adt_seq[-1], dtype=np.float32))

        if not x_list:
            raise HTTPException(status_code=400, detail="No samples available for selected date")

        x_batch = torch.stack(x_list, dim=0)
        return x_batch, time_indices, adt_maps, max_index
    finally:
        ds.close()


def _field_2d(step_data: list[dict[str, Any]], var_name: str) -> np.ndarray | None:
    for item in step_data:
        if str(item.get("var", "")).upper() == var_name.upper():
            arr = np.asarray(item.get("data", []), dtype=np.float32)
            if arr.ndim == 2:
                return arr
    return None


def _json_safe_2d(arr: np.ndarray) -> list[list[float | None]]:
    """Convert 2D float array to JSON-safe nested list (NaN/Inf -> None)."""
    a = np.asarray(arr)
    if a.ndim != 2:
        return []
    if np.issubdtype(a.dtype, np.floating):
        a = np.where(np.isfinite(a), a, None)
    return a.tolist()


def _render_eddy_boundary_image(step_payload: dict[str, Any], output_file: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_data = step_payload.get("data", [])
    centers = step_payload.get("centers", []) if isinstance(step_payload, dict) else []
    adt = _field_2d(step_data, "ADT")
    pred_mask = _field_2d(step_data, "PRED_MASK")
    pred_boundary = _field_2d(step_data, "PRED_BOUNDARY")

    if adt is None:
        raise HTTPException(status_code=500, detail="Missing ADT field in cached step")

    coords = step_payload.get("coords", {}) if isinstance(step_payload, dict) else {}
    lat = np.asarray(coords.get("latitude", []), dtype=np.float32)
    lon = np.asarray(coords.get("longitude", []), dtype=np.float32)
    has_geo = lat.ndim == 1 and lon.ndim == 1 and lat.size == adt.shape[0] and lon.size == adt.shape[1]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    finite = np.isfinite(adt)
    vmin = 0.2
    if np.any(finite):
        vmax = float(np.nanpercentile(adt[finite], 99.0))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmax = 1.2

    # Required visual rule: 0.2 is blue; larger values gradually turn red.
    if has_geo:
        extent = (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))
        ax.imshow(adt, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=extent, origin="lower", alpha=0.90)
    else:
        ax.imshow(adt, cmap="coolwarm", vmin=vmin, vmax=vmax, origin="lower", alpha=0.90)

    cyc_boundary = None
    anti_boundary = None
    if pred_mask is not None:
        cyc_boundary = _build_boundary_mask((pred_mask == 1).astype(np.uint8))
        anti_boundary = _build_boundary_mask((pred_mask == 2).astype(np.uint8))
    elif pred_boundary is not None:
        # Fallback for old cached payloads.
        anti_boundary = pred_boundary

    if cyc_boundary is not None:
        if has_geo:
            ax.contour(lon, lat, cyc_boundary, levels=[0.5], colors=["#ef4444"], linewidths=0.8)
        else:
            ax.contour(cyc_boundary, levels=[0.5], colors=["#ef4444"], linewidths=0.8)
    if anti_boundary is not None:
        if has_geo:
            ax.contour(lon, lat, anti_boundary, levels=[0.5], colors=["#3b82f6"], linewidths=0.8)
        else:
            ax.contour(anti_boundary, levels=[0.5], colors=["#3b82f6"], linewidths=0.8)

    if isinstance(centers, list) and centers:
        cyc_lon, cyc_lat, anti_lon, anti_lat = [], [], [], []
        for c in centers:
            if not isinstance(c, dict):
                continue
            lo = c.get("lon")
            la = c.get("lat")
            if not (isinstance(lo, (int, float)) and isinstance(la, (int, float))):
                continue
            if int(c.get("class_id", 0)) == 1:
                cyc_lon.append(float(lo))
                cyc_lat.append(float(la))
            elif int(c.get("class_id", 0)) == 2:
                anti_lon.append(float(lo))
                anti_lat.append(float(la))

        if cyc_lon:
            ax.scatter(cyc_lon, cyc_lat, s=18, c="#f87171", marker="x", linewidths=0.9)
        if anti_lon:
            ax.scatter(anti_lon, anti_lat, s=18, c="#60a5fa", marker="x", linewidths=0.9)

    summary = step_payload.get("summary", {})
    cyc = int(summary.get("cyclonic_count", 0))
    anti = int(summary.get("anticyclonic_count", 0))
    ax.set_title(f"Eddy Boundary | cyclonic={cyc}, anticyclonic={anti}")
    ax.text(0.98, 0.96, "cyclonic", color="#ef4444", fontsize=9, ha="right", va="top", transform=ax.transAxes)
    ax.text(0.98, 0.91, "anticyclonic", color="#3b82f6", fontsize=9, ha="right", va="top", transform=ax.transAxes)
    ax.set_xlabel("longitude" if has_geo else "longitude index")
    ax.set_ylabel("latitude" if has_geo else "latitude index")
    ax.grid(alpha=0.25, color="#94a3b8", linewidth=0.5)
    ax.set_facecolor("#030712")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_file


@app.get("/api/default-data-path")
def get_default_data_path():
    path_file = resolve_path("data/processed/element_forecasting/path.txt")
    if os.path.exists(path_file):
        with open(path_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return {"path": content}
    return {"path": ""}


@app.get("/api/eddy/default-paths")
def get_default_eddy_paths():
    clean_path, _ = _default_eddy_paths()
    return {
        "model_path": _default_eddy_model_path(),
        "data_path": clean_path,
    }


@app.get("/api/eddy/default-data-path")
def get_eddy_default_data_path():
    clean_path, _ = _default_eddy_paths()
    return {"path": clean_path}


@app.post("/api/eddy/dataset-info")
def get_eddy_dataset_info(req: EddyDatasetInfoRequest):
    try:
        data_path = resolve_path(str(req.data_path).strip().strip('"\''))
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

        ds = xr.open_dataset(data_path)
        try:
            for v in ("adt", "ugos", "vgos"):
                if v not in ds:
                    raise HTTPException(status_code=400, detail=f"Dataset missing variable: {v}")

            tlen = int(ds["adt"].shape[0])
            if tlen <= 0:
                raise HTTPException(status_code=400, detail="Dataset time axis is empty")

            input_steps = max(1, int(req.input_steps))
            max_index = max(0, tlen - input_steps)

            lat_size = int(ds["adt"].shape[-2])
            lon_size = int(ds["adt"].shape[-1])
            if "time" in ds:
                tvals = pd.to_datetime(ds["time"].values)
                if len(tvals) > 0:
                    time_range = f"{str(tvals[0])[:10]} ~ {str(tvals[-1])[:10]}"
                    dates = [str(x)[:10] for x in tvals]
                else:
                    time_range = "N/A"
                    dates = []
            else:
                time_range = "N/A"
                dates = []

            info = (
                f"Eddy dataset ready\n"
                f"data_path: {data_path}\n"
                f"input_steps: {input_steps}\n"
                f"shape: time={tlen}, lat={lat_size}, lon={lon_size}\n"
                f"原始时间范围: {time_range}"
            )
        finally:
            ds.close()

        return {"info": info, "max_index": max_index, "dates": dates}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to inspect eddy dataset: {e}")


def _nearest_index(values: np.ndarray, target: float) -> int:
    if values.ndim != 1 or values.size == 0:
        return 0
    return int(np.argmin(np.abs(values - float(target))))


def _extract_centers_from_mask(mask: np.ndarray, min_region_pixels: int = 16) -> list[list[float]]:
    if mask.ndim != 2:
        return []

    mask_int = np.asarray(mask, dtype=np.int32)
    visited = np.zeros(mask_int.shape, dtype=bool)
    centers: list[list[float]] = []
    rows, cols = mask_int.shape

    for class_id in (1, 2):
        class_mask = mask_int == class_id
        if not np.any(class_mask):
            continue

        for start_r, start_c in np.argwhere(class_mask):
            if visited[start_r, start_c]:
                continue

            stack = [(int(start_r), int(start_c))]
            component: list[tuple[int, int]] = []
            visited[start_r, start_c] = True

            while stack:
                r, c = stack.pop()
                component.append((r, c))

                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[rr, cc] or not class_mask[rr, cc]:
                        continue
                    visited[rr, cc] = True
                    stack.append((rr, cc))

            if len(component) < int(min_region_pixels):
                continue

            comp_arr = np.asarray(component, dtype=np.float32)
            row_center = float(np.mean(comp_arr[:, 0]))
            col_center = float(np.mean(comp_arr[:, 1]))
            centers.append([row_center, col_center, float(class_id)])

    return centers


@app.post("/api/eddy/predict-day")
async def run_eddy_prediction_day(req: EddyPredictDayRequest):
    try:
        model_path = resolve_path(req.model_path)
        data_path = resolve_path(str(req.data_path).strip().strip('"\''))
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

        input_steps = max(1, int(req.input_steps))
        x_batch, time_indices, adt_maps, _ = await asyncio.to_thread(
            _build_unlabeled_eddy_batch,
            data_path=data_path,
            start_idx=int(req.day_index),
            input_steps=input_steps,
            horizon_steps=1,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EddyUNet(in_channels=x_batch.shape[1], num_classes=3, base_channels=req.base_channels)
        load_checkpoint(model, model_path, map_location=device)
        model = model.to(device)
        results = infer_batch_to_objects(model, x_batch, device, min_region_pixels=req.min_region_pixels)
        del model, x_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not results:
            raise HTTPException(status_code=500, detail="Empty model output for selected day")

        ds = xr.open_dataset(data_path)
        try:
            tvals = pd.to_datetime(ds["time"].values) if "time" in ds else []
            lat = np.asarray(ds["latitude"].values, dtype=np.float32) if "latitude" in ds else np.array([])
            lon = np.asarray(ds["longitude"].values, dtype=np.float32) if "longitude" in ds else np.array([])
        finally:
            ds.close()

        res0 = results[0]
        pred_mask = np.asarray(res0.get("mask"), dtype=np.int32)
        adt = np.asarray(adt_maps[0], dtype=np.float32)
        cyc_count = int(res0.get("cyclonic_count", 0))
        anti_count = int(res0.get("anticyclonic_count", 0))

        centers: list[list[float]] = []
        for obj in res0.get("objects", []):
            try:
                c_lat = obj.get("center_lat")
                c_lon = obj.get("center_lon")
                if c_lat is None or c_lon is None:
                    continue
                r = _nearest_index(lat, float(c_lat)) if lat.size else 0
                c = _nearest_index(lon, float(c_lon)) if lon.size else 0
                centers.append([float(r), float(c), float(obj.get("class_id", 0))])
            except Exception:
                continue

        if not centers:
            centers = _extract_centers_from_mask(pred_mask, min_region_pixels=req.min_region_pixels)

        t_index = int(time_indices[0]) if time_indices else int(req.day_index)
        if isinstance(tvals, pd.DatetimeIndex) and len(tvals) > 0 and 0 <= t_index < len(tvals):
            day_label = str(tvals[t_index])[:10]
        else:
            day_label = f"idx_{int(req.day_index)}"

        return {
            "day_index": int(req.day_index),
            "day_label": day_label,
            "cyclonic_count": cyc_count,
            "anticyclonic_count": anti_count,
            "adt": _json_safe_2d(adt),
            "pred_mask": pred_mask.tolist(),
            "centers": centers,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/eddy/date-index")
def get_eddy_date_index(req: EddyDateIndexRequest):
    try:
        data_path = resolve_path(str(req.data_path).strip().strip('"\''))
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

        ds = xr.open_dataset(data_path)
        try:
            if "time" not in ds:
                raise HTTPException(status_code=400, detail="Dataset missing time coordinate")

            times = pd.to_datetime(ds["time"].values)
            if len(times) == 0:
                raise HTTPException(status_code=400, detail="Dataset time axis is empty")

            input_steps = max(1, int(req.input_steps))
            target = pd.to_datetime(req.date)

            # Match selected day to the closest end-frame, then convert to window start index.
            deltas = np.abs(times - target)
            end_idx = int(np.argmin(deltas))
            start_idx = max(0, end_idx - (input_steps - 1))
            max_index = max(0, len(times) - input_steps)
            if start_idx > max_index:
                start_idx = max_index
                end_idx = min(len(times) - 1, start_idx + input_steps - 1)

            resolved_time = times[end_idx]
            exact_match = str(resolved_time)[:10] == str(target)[:10]
        finally:
            ds.close()

        return {
            "start_idx": start_idx,
            "max_index": max_index,
            "resolved_time": str(resolved_time),
            "exact_match": exact_match,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to find date index: {e}")


@app.post("/api/predict")
def run_prediction(req: PredictRequest):
    data_path = resolve_path(req.data_path.strip('\"\''))
    model_path = resolve_path(req.model_path.strip('\"\''))
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")
        
    try:
        norm_path = resolve_path("data/processed/normalization/element_forecasting_norm.json")
        dataset = ElementForecastWindowDataset(
            data_file=data_path, 
            input_steps=24, 
            output_steps=72, 
            split=None,
            norm_stats_path=norm_path
        )
        if req.start_idx < 0 or req.start_idx >= len(dataset):
            raise HTTPException(status_code=400, detail="Start index out of range")

        sample = dataset[req.start_idx]
        x_tensor = sample["x"].unsqueeze(0)
        y_tensor = sample["y"].numpy()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = ElementForecastPredictor(checkpoint_path=model_path, device=device, norm_stats_path=norm_path)

        result = predictor.predict_long_horizon(
            x=x_tensor, 
            target_steps=72,
            overlap_steps=4,
            enable_overlap_blend=True,
            denormalize=True, 
            return_cpu=True
        )
        pred_numpy = result["pred"][0].numpy()
        var_names = result.get("var_names", ["SST", "SSS", "SSU", "SSV"])
        
        valid_mask_tensor = sample.get("y_valid", None)
        mask_numpy = valid_mask_tensor.numpy() if valid_mask_tensor is not None else None
        
        # Save to cache to avoid sending massive payload, client can request steps
        session_id = "default_session"
        prediction_cache[session_id] = {
            "pred": pred_numpy,
            "true": y_tensor,
            "mask": mask_numpy,
            "vars": var_names
        }
        
        return {"message": "Prediction successful", "session_id": session_id, "steps": pred_numpy.shape[0]}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/eddy/predict")
async def run_eddy_prediction(req: EddyPredictRequest):
    try:
        model_path = resolve_path(req.model_path)
        data_path = resolve_path(str(req.data_path).strip().strip('"\''))
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

        x_batch, time_indices, adt_maps, max_index = await asyncio.to_thread(
            _build_unlabeled_eddy_batch,
            data_path=data_path,
            start_idx=req.start_idx,
            input_steps=req.input_steps,
            horizon_steps=req.horizon_steps,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EddyUNet(in_channels=x_batch.shape[1], num_classes=3, base_channels=req.base_channels)
        load_checkpoint(model, model_path, map_location=device)
        model = model.to(device)

        results = infer_batch_to_objects(model, x_batch, device, min_region_pixels=req.min_region_pixels)
        del model, x_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ds_for_coords = xr.open_dataset(data_path)
        lat_coords = ds_for_coords["latitude"].values.tolist()
        lon_coords = ds_for_coords["longitude"].values.tolist()
        ds_for_coords.close()

        steps = []
        for i, res in enumerate(results):
            pred_mask = res["mask"]
            cyc_boundary = _build_boundary_mask((pred_mask == 1).astype(np.uint8))
            anti_boundary = _build_boundary_mask((pred_mask == 2).astype(np.uint8))
            cyc_count = int(res.get("cyclonic_count", 0))
            anti_count = int(res.get("anticyclonic_count", 0))

            steps.append({
                "time_index": time_indices[i],
                "summary": {
                    "cyclonic_count": cyc_count,
                    "anticyclonic_count": anti_count,
                    "total_count": cyc_count + anti_count,
                },
                "coords": {
                    "latitude": lat_coords,
                    "longitude": lon_coords,
                },
                "data": [
                    {"var": "ADT", "data": _json_safe_2d(adt_maps[i])},
                    {"var": "CYCLONIC_BOUNDARY", "data": _json_safe_2d(cyc_boundary)},
                    {"var": "ANTICYCLONIC_BOUNDARY", "data": _json_safe_2d(anti_boundary)},
                ],
                "centers": [
                    {
                        "lon": obj.get("center_lon"),
                        "lat": obj.get("center_lat"),
                        "class_id": obj.get("class_id"),
                    }
                    for obj in res.get("objects", [])
                ],
            })

        session_id = str(uuid.uuid4())
        run_tag = Path(req.model_path).parent.parent.name
        eddy_prediction_cache[session_id] = {
            "steps": steps,
            "run_tag": run_tag,
        }

        return {
            "message": "Eddy detection successful",
            "session_id": session_id,
            "steps": len(steps),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def extract_mask(mask_numpy, t_idx, c_idx, H, W):
    if mask_numpy is None: return None
    if mask_numpy.shape == (H, W): return mask_numpy
    elif mask_numpy.ndim == 3:
        if mask_numpy.shape[0] == 4: return mask_numpy[min(c_idx, 3)]
        else: return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1)]
    elif mask_numpy.ndim == 4:
        return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1), min(c_idx, mask_numpy.shape[1] - 1)]
    return mask_numpy


@app.get("/api/eddy/predict/{session_id}/step/{step_idx}")
def get_eddy_prediction_step(session_id: str, step_idx: int):
    state = eddy_prediction_cache.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = state["steps"]
    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(status_code=400, detail="Step out of range")
    return {"step": step_idx, **steps[step_idx]}


@app.get("/api/eddy/predict/{session_id}/curve")
def get_eddy_prediction_curve(session_id: str):
    state = eddy_prediction_cache.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    state = prediction_cache[session_id]
    pred_numpy = state["pred"]
    mask_numpy = state["mask"]
    var_names = state["vars"]
    
    num_steps, _, H, W = pred_numpy.shape
    response_data = []
    
    for i in range(min(4, pred_numpy.shape[1])):
        mean_vals = []
        for t in range(num_steps):
            data_slice = pred_numpy[t, i]
            mask_slice = extract_mask(mask_numpy, t, i, H, W)
            if mask_slice is not None and mask_slice.shape == (H, W):
                valid_data = data_slice[mask_slice >= 0.5]
            else:
                valid_data = data_slice[~np.isnan(data_slice)]
            mean_vals.append(float(np.mean(valid_data)) if len(valid_data) > 0 else None)
            
        response_data.append({
            "var": var_names[i],
            "means": mean_vals
        })
        
    return {"data": response_data}


@app.post("/api/anomaly/inspect")
def inspect_anomaly(req: AnomalyInspectRequest):
    split = str(req.split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise HTTPException(status_code=400, detail="split must be one of train/val/test")

    labels_path = resolve_path(req.labels_json.strip('"\''))
    events_path = resolve_path(req.events_json.strip('"\''))
    manifest_path = resolve_path(req.manifest_path.strip('"\''))
    processed_dir = resolve_path(req.processed_dir.strip('"\''))
    norm_stats_path = resolve_path(req.norm_stats_path.strip('"\''))

    if not os.path.exists(labels_path):
        raise HTTPException(status_code=404, detail=f"labels file not found: {labels_path}")
    if not os.path.exists(events_path):
        raise HTTPException(status_code=404, detail=f"events file not found: {events_path}")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail=f"manifest file not found: {manifest_path}")

    try:
        labels_obj = json.loads(open(labels_path, "r", encoding="utf-8").read())
        events_obj = json.loads(open(events_path, "r", encoding="utf-8").read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid json: {e}")

    if not isinstance(labels_obj, dict):
        raise HTTPException(status_code=400, detail="labels_json must be an object with split keys")

    split_labels_raw = labels_obj.get(split)
    if not isinstance(split_labels_raw, list):
        raise HTTPException(status_code=400, detail=f"labels_json missing list for split={split}")

    labels = [1 if int(v) == 1 else 0 for v in split_labels_raw]

    events: list[dict] = []
    if isinstance(events_obj, list):
        for ev in events_obj:
            if not isinstance(ev, dict):
                continue
            if "start" not in ev or "end" not in ev:
                continue
            try:
                start = int(ev["start"])
                end = int(ev["end"])
            except Exception:
                continue
            if end < start:
                continue
            events.append({"name": str(ev.get("name", "event")), "start": start, "end": end})

    cache_key = (processed_dir, manifest_path, split)
    cached_timestamps = anomaly_timestamps_cache.get(cache_key)
    if cached_timestamps is None:
        ds = AnomalyFrameDataset(
            processed_anomaly_dir=processed_dir,
            split=split,
            manifest_path=manifest_path,
            norm_stats_path=norm_stats_path if os.path.exists(norm_stats_path) else None,
            open_file_lru_size=max(0, int(req.open_file_lru_size)),
        )
        try:
            timestamps = [_to_epoch_seconds(ts) for ts in ds.get_timestamps()]
        finally:
            ds.close()
        anomaly_timestamps_cache[cache_key] = timestamps
    else:
        timestamps = cached_timestamps

    raw_ts_len = len(timestamps)
    timestamp_mode = "dataset"
    fallback_warning = ""

    if raw_ts_len == 0 and len(labels) > 0:
        # Fallback for environments without local processed anomaly files.
        timestamps = [-1] * len(labels)
        timestamp_mode = "labels_only_fallback"
        fallback_warning = (
            "processed anomaly files unavailable; using labels-only fallback without event-time matching"
        )

    n_pair = min(len(labels), len(timestamps))
    if n_pair == 0:
        raise HTTPException(status_code=400, detail="empty split after loading labels/timestamps")

    if len(labels) != len(timestamps):
        # Keep service usable even when lengths mismatch.
        labels = labels[:n_pair]
        timestamps = timestamps[:n_pair]

    def _hit_events(ts: int) -> list[str]:
        if ts < 0:
            return []
        return [ev["name"] for ev in events if ev["start"] <= ts <= ev["end"]]

    positive_points: list[dict] = []
    matched_positive = 0
    matched_event_names: set[str] = set()

    for idx, (y, ts) in enumerate(zip(labels, timestamps)):
        if y != 1:
            continue
        hits = _hit_events(ts)
        if hits:
            matched_positive += 1
            matched_event_names.update(hits)
        positive_points.append(
            {
                "index": idx,
                "timestamp": ts,
                "event_hits": hits,
                "matched": bool(hits),
            }
        )

    max_points = max(1, int(req.max_points))
    preview = positive_points[:max_points]

    curve = state["curve"]
    has_truth = bool(state.get("has_truth", False))
    return {
        "split": split,
        "num_samples": n_pair,
        "num_positive": int(sum(labels)),
        "positive_ratio": float(sum(labels) / n_pair),
        "num_events": len(events),
        "matched_positive": matched_positive,
        "matched_positive_ratio": float(matched_positive / max(1, sum(labels))),
        "matched_event_count": len(matched_event_names),
        "points": preview,
        "truncated": len(positive_points) > len(preview),
        "labels_timestamps_aligned": len(split_labels_raw) == raw_ts_len,
        "timestamp_mode": timestamp_mode,
        "warning": fallback_warning,
    }


@app.get("/api/eddy/predict/{session_id}/boundary-image/{step_idx}")
def export_eddy_boundary_image(session_id: str, step_idx: int):
    state = eddy_prediction_cache.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = state.get("steps", [])
    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(status_code=400, detail="Step out of range")

    tmp_dir = Path(tempfile.gettempdir()) / "oceanrace_eddy_exports"
    out_file = tmp_dir / f"{session_id}_step_{step_idx}.png"
    _render_eddy_boundary_image(steps[step_idx], out_file)
    return FileResponse(path=out_file, media_type="image/png", filename=out_file.name)


@app.get("/api/eddy/predict/{session_id}/reference-image/{step_idx}")
def get_eddy_reference_image(session_id: str, step_idx: int):
    state = eddy_prediction_cache.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = state.get("steps", [])
    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(status_code=400, detail="Step out of range")

    run_tag = str(state.get("run_tag", "global_boundary_full_20260403"))
    time_index = int(steps[step_idx].get("time_index", -1))
    ref = _find_eddy_reference_figure(run_tag, time_index)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"Reference image not found for run={run_tag}, time_index={time_index}")

    return FileResponse(path=ref, media_type="image/png", filename=ref.name)
