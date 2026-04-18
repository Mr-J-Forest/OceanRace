"""Eddy detection helpers: caches, batching, boundary rendering."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from fastapi import HTTPException

from src.eddy_detection.model import EddyUNet
from src.eddy_detection.predictor import load_checkpoint

from . import state
from .paths import read_path_txt, resolve_path



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
    path_txt_dir = read_path_txt("data/processed/eddy_detection/path.txt")
    if path_txt_dir:
        base_dir = Path(resolve_path(path_txt_dir))
        if base_dir.exists() and base_dir.is_dir():
            direct_clean = base_dir / "19930101_20241231_clean.nc"
            if direct_clean.exists():
                return (str(direct_clean), "")
            candidates = sorted(base_dir.rglob("*_clean.nc"))
            if candidates:
                return (str(candidates[0]), "")

        split_manifest = resolve_path("data/processed/splits/eddy_merged_time_competition.json")
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
        "models/eddy_model.pt",
        "outputs/eddy_detection/meta4_mask_retrain_20260413_bg/checkpoints/best.pt",
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
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    if "eddy_detection" in parts:
        i = parts.index("eddy_detection")
        if i + 1 < len(parts):
            seg = parts[i + 1]
            if seg not in ("checkpoints", "figures", "runs"):
                return seg
    return "meta4_mask_retrain_20260413_bg"


def _find_eddy_reference_figure(run_tag: str, time_index: int) -> Path | None:
    figures_dir = Path(resolve_path(f"outputs/eddy_detection/{run_tag}/figures"))
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
    if state.eddy_norm_cache.get("path") == norm_path:
        cached = state.eddy_norm_cache.get("variables", {})
        if isinstance(cached, dict):
            return cached
    try:
        with open(norm_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        variables = payload.get("variables", {})
        variables = variables if isinstance(variables, dict) else {}
        state.eddy_norm_cache["path"] = norm_path
        state.eddy_norm_cache["variables"] = variables
        return variables
    except Exception:
        return {}


def _file_signature(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return (int(st.st_mtime_ns), int(st.st_size))


def _get_cached_eddy_dataset_handle(data_path: str) -> dict[str, Any]:
    sig = _file_signature(data_path)
    with state.eddy_cache_lock:
        cached = state.eddy_dataset_handle_cache.get(data_path)
        if cached is not None and cached.get("sig") == sig:
            return cached

        # File changed (or first load): close stale handle first.
        if cached is not None:
            try:
                cached.get("ds").close()
            except Exception:
                pass

        ds = xr.open_dataset(data_path)
        payload = {
            "sig": sig,
            "ds": ds,
        }
        state.eddy_dataset_handle_cache[data_path] = payload

        # Keep cache bounded: close and evict older paths.
        if len(state.eddy_dataset_handle_cache) > 3:
            for k in list(state.eddy_dataset_handle_cache.keys())[:-3]:
                if k == data_path:
                    continue
                old = state.eddy_dataset_handle_cache.pop(k, None)
                if old is not None:
                    try:
                        old.get("ds").close()
                    except Exception:
                        pass
        return payload


def _get_cached_eddy_model(
    *,
    model_path: str,
    in_channels: int,
    base_channels: int,
    device: str,
) -> EddyUNet:
    model_sig = _file_signature(model_path)
    cache_key = (
        model_path,
        int(in_channels),
        int(base_channels),
        str(device),
        model_sig[0],
        model_sig[1],
    )
    cached = state.eddy_model_cache.get(cache_key)
    if cached is not None:
        return cached

    # Drop stale cache entries for the same logical model/device to keep memory bounded.
    stale_keys = [
        k
        for k in state.eddy_model_cache.keys()
        if k[0] == model_path and k[1] == int(in_channels) and k[2] == int(base_channels) and k[3] == str(device)
    ]
    for k in stale_keys:
        state.eddy_model_cache.pop(k, None)

    model = EddyUNet(in_channels=int(in_channels), num_classes=3, base_channels=int(base_channels))
    load_checkpoint(model, model_path, map_location=device)
    model = model.to(device)
    model.eval()
    state.eddy_model_cache[cache_key] = model
    return model


def _get_cached_eddy_dataset_meta(data_path: str) -> dict[str, Any]:
    handle = _get_cached_eddy_dataset_handle(data_path)
    sig = handle["sig"]
    with state.eddy_cache_lock:
        cached = state.eddy_dataset_meta_cache.get(data_path)
        if cached is not None and cached.get("sig") == sig:
            return cached

    ds = handle["ds"]
    tvals = pd.to_datetime(ds["time"].values) if "time" in ds else pd.DatetimeIndex([])
    lat = np.asarray(ds["latitude"].values, dtype=np.float32) if "latitude" in ds else np.array([], dtype=np.float32)
    lon = np.asarray(ds["longitude"].values, dtype=np.float32) if "longitude" in ds else np.array([], dtype=np.float32)

    payload = {
        "sig": sig,
        "tvals": tvals,
        "lat": lat,
        "lon": lon,
        "lat_list": lat.tolist(),
        "lon_list": lon.tolist(),
    }
    with state.eddy_cache_lock:
        state.eddy_dataset_meta_cache[data_path] = payload
    return payload


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
    handle = _get_cached_eddy_dataset_handle(data_path)
    ds = handle["ds"]
    sig = handle["sig"]

    for v in ("adt", "ugos", "vgos"):
        if v not in ds:
            raise HTTPException(status_code=400, detail=f"Dataset missing variable: {v}")

    tlen = int(ds["adt"].shape[0])
    if tlen <= 0:
        raise HTTPException(status_code=400, detail="Dataset time axis is empty")

    max_index = max(0, tlen - input_steps)
    if start_idx < 0 or start_idx > max_index:
        raise HTTPException(status_code=400, detail="Start index out of range")

    cache_key = (data_path, sig[0], sig[1], int(start_idx), int(input_steps))
    with state.eddy_cache_lock:
        cached = state.eddy_window_cache.get(cache_key)
    if cached is not None:
        return (
            cached["x_batch"].clone(),
            list(cached["time_indices"]),
            [np.asarray(m, dtype=np.float32).copy() for m in cached["adt_maps"]],
            int(cached["max_index"]),
        )

    end_idx = min(max_index + 1, start_idx + horizon_steps)
    stats = _load_eddy_norm_variables()

    x_list: list[torch.Tensor] = []
    time_indices: list[int] = []
    adt_maps: list[np.ndarray] = []

    for t in range(start_idx, end_idx):
        t0 = t
        t1 = t + input_steps
        # Slice first, then materialize values; avoids loading full variable into memory.
        adt_seq = np.asarray(ds["adt"].isel(time=slice(t0, t1)).values, dtype=np.float32)
        ugos_seq = np.asarray(ds["ugos"].isel(time=slice(t0, t1)).values, dtype=np.float32)
        vgos_seq = np.asarray(ds["vgos"].isel(time=slice(t0, t1)).values, dtype=np.float32)

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

    x_batch = torch.stack(x_list, dim=0).contiguous()
    with state.eddy_cache_lock:
        state.eddy_window_cache[cache_key] = {
            "x_batch": x_batch,
            "time_indices": list(time_indices),
            "adt_maps": [np.asarray(m, dtype=np.float32).copy() for m in adt_maps],
            "max_index": int(max_index),
        }
        # Bound cache size to avoid uncontrolled memory growth.
        if len(state.eddy_window_cache) > 64:
            for k in list(state.eddy_window_cache.keys())[:-64]:
                state.eddy_window_cache.pop(k, None)

    return x_batch.clone(), time_indices, adt_maps, max_index


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
