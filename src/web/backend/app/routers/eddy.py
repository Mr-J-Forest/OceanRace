"""涡旋：默认路径、数据集信息、推理与缓存会话。"""
from __future__ import annotations

import asyncio
import os
import tempfile
import traceback
import uuid
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xarray as xr
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.eddy_detection.predictor import infer_batch_to_objects
from utils.logger import get_logger

from .. import eddy_core as ec
from .. import state
from ..paths import resolve_path
from ..schemas import (
    EddyDatasetInfoRequest,
    EddyDateIndexRequest,
    EddyPredictDayRequest,
    EddyPredictRequest,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/eddy", tags=["eddy"])


@router.get("/default-paths")
def get_default_eddy_paths():
    return {
        "model_path": ec._default_eddy_model_path(),
        "data_path": "data/processed/eddy_detection/path.txt",
    }


@router.get("/default-data-path")
def get_eddy_default_data_path():
    return {"path": "data/processed/eddy_detection/path.txt"}


@router.post("/dataset-info")
def get_eddy_dataset_info(req: EddyDatasetInfoRequest):
    try:
        data_path = ec.resolve_eddy_clean_nc_path(req.data_path)
        if not data_path:
            raise HTTPException(
                status_code=404,
                detail="无法解析涡旋数据文件：path.txt 若指向目录，请确保其下存在 *_clean.nc（或 19930101_20241231_clean.nc）。",
            )
        if not os.path.isfile(data_path):
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


@router.post("/predict-day")
async def run_eddy_prediction_day(req: EddyPredictDayRequest):
    t0_total = perf_counter()
    try:
        model_path = resolve_path(req.model_path)
        data_path = ec.resolve_eddy_clean_nc_path(req.data_path)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        if not data_path or not os.path.isfile(data_path):
            raise HTTPException(
                status_code=404,
                detail="无法解析涡旋数据文件：path.txt 若指向目录，请确保其下存在 *_clean.nc。",
            )

        input_steps = max(1, int(req.input_steps))
        t0_read = perf_counter()
        x_batch, time_indices, adt_maps, _ = await asyncio.to_thread(
            ec._build_unlabeled_eddy_batch,
            data_path=data_path,
            start_idx=int(req.day_index),
            input_steps=input_steps,
            horizon_steps=1,
        )
        t_read = perf_counter() - t0_read

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0_infer = perf_counter()
        model = ec._get_cached_eddy_model(
            model_path=model_path,
            in_channels=int(x_batch.shape[1]),
            base_channels=int(req.base_channels),
            device=device,
        )
        results = infer_batch_to_objects(model, x_batch, device, min_region_pixels=req.min_region_pixels)
        del x_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_infer = perf_counter() - t0_infer

        if not results:
            raise HTTPException(status_code=500, detail="Empty model output for selected day")

        t0_post = perf_counter()
        meta = ec._get_cached_eddy_dataset_meta(data_path)
        tvals = meta.get("tvals", pd.DatetimeIndex([]))
        lat = np.asarray(meta.get("lat", np.array([], dtype=np.float32)), dtype=np.float32)
        lon = np.asarray(meta.get("lon", np.array([], dtype=np.float32)), dtype=np.float32)

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
                r = ec._nearest_index(lat, float(c_lat)) if lat.size else 0
                c = ec._nearest_index(lon, float(c_lon)) if lon.size else 0
                centers.append([float(r), float(c), float(obj.get("class_id", 0))])
            except Exception:
                continue

        if not centers:
            centers = ec._extract_centers_from_mask(pred_mask, min_region_pixels=req.min_region_pixels)

        t_index = int(time_indices[0]) if time_indices else int(req.day_index)
        if isinstance(tvals, pd.DatetimeIndex) and len(tvals) > 0 and 0 <= t_index < len(tvals):
            day_label = str(tvals[t_index])[:10]
        else:
            day_label = f"idx_{int(req.day_index)}"
        t_post = perf_counter() - t0_post
        t_total = perf_counter() - t0_total

        logger.info(
            "eddy predict-day timing | day_index=%s device=%s read=%.3fs infer=%.3fs post=%.3fs total=%.3fs",
            int(req.day_index),
            device,
            t_read,
            t_infer,
            t_post,
            t_total,
        )

        return {
            "day_index": int(req.day_index),
            "day_label": day_label,
            "cyclonic_count": cyc_count,
            "anticyclonic_count": anti_count,
            "adt": ec._json_safe_2d(adt),
            "pred_mask": pred_mask.tolist(),
            "centers": centers,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/date-index")
def get_eddy_date_index(req: EddyDateIndexRequest):
    try:
        data_path = ec.resolve_eddy_clean_nc_path(req.data_path)
        if not data_path or not os.path.isfile(data_path):
            raise HTTPException(
                status_code=404,
                detail="无法解析涡旋数据文件：path.txt 若指向目录，请确保其下存在 *_clean.nc。",
            )

        ds = xr.open_dataset(data_path)
        try:
            if "time" not in ds:
                raise HTTPException(status_code=400, detail="Dataset missing time coordinate")

            times = pd.to_datetime(ds["time"].values)
            if len(times) == 0:
                raise HTTPException(status_code=400, detail="Dataset time axis is empty")

            input_steps = max(1, int(req.input_steps))
            target = pd.to_datetime(req.date)

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


@router.post("/predict")
async def run_eddy_prediction(req: EddyPredictRequest):
    try:
        model_path = resolve_path(req.model_path)
        data_path = ec.resolve_eddy_clean_nc_path(req.data_path)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        if not data_path or not os.path.isfile(data_path):
            raise HTTPException(
                status_code=404,
                detail="无法解析涡旋数据文件：path.txt 若指向目录，请确保其下存在 *_clean.nc。",
            )

        x_batch, time_indices, adt_maps, _max_index = await asyncio.to_thread(
            ec._build_unlabeled_eddy_batch,
            data_path=data_path,
            start_idx=req.start_idx,
            input_steps=req.input_steps,
            horizon_steps=req.horizon_steps,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ec._get_cached_eddy_model(
            model_path=model_path,
            in_channels=int(x_batch.shape[1]),
            base_channels=int(req.base_channels),
            device=device,
        )

        results = infer_batch_to_objects(model, x_batch, device, min_region_pixels=req.min_region_pixels)
        del x_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        meta = ec._get_cached_eddy_dataset_meta(data_path)
        lat_coords = meta.get("lat_list", [])
        lon_coords = meta.get("lon_list", [])

        steps = []
        for i, res in enumerate(results):
            pred_mask = res["mask"]
            cyc_boundary = ec._build_boundary_mask((pred_mask == 1).astype(np.uint8))
            anti_boundary = ec._build_boundary_mask((pred_mask == 2).astype(np.uint8))
            cyc_count = int(res.get("cyclonic_count", 0))
            anti_count = int(res.get("anticyclonic_count", 0))

            steps.append(
                {
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
                        {"var": "ADT", "data": ec._json_safe_2d(adt_maps[i])},
                        {"var": "CYCLONIC_BOUNDARY", "data": ec._json_safe_2d(cyc_boundary)},
                        {"var": "ANTICYCLONIC_BOUNDARY", "data": ec._json_safe_2d(anti_boundary)},
                    ],
                    "centers": [
                        {
                            "lon": obj.get("center_lon"),
                            "lat": obj.get("center_lat"),
                            "class_id": obj.get("class_id"),
                        }
                        for obj in res.get("objects", [])
                    ],
                }
            )

        session_id = str(uuid.uuid4())
        run_tag = Path(req.model_path).parent.parent.name
        state.eddy_prediction_cache[session_id] = {
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/predict/{session_id}/step/{step_idx}")
def get_eddy_prediction_step(session_id: str, step_idx: int):
    st = state.eddy_prediction_cache.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = st["steps"]
    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(status_code=400, detail="Step out of range")
    return {"step": step_idx, **steps[step_idx]}


@router.get("/predict/{session_id}/curve")
def get_eddy_prediction_curve(session_id: str):
    st = state.eddy_prediction_cache.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Session not found")

    curve = st["curve"]
    has_truth = bool(st.get("has_truth", False))
    return {
        "data": [
            {"var": "cyclonic", "means": curve["cyclonic"]},
            {"var": "anticyclonic", "means": curve["anticyclonic"]},
            {"var": "total", "means": curve["total"]},
            {"var": "true_cyclonic", "means": curve.get("true_cyclonic", [])},
            {"var": "true_anticyclonic", "means": curve.get("true_anticyclonic", [])},
            {"var": "true_total", "means": curve.get("true_total", [])},
        ],
        "has_truth": has_truth,
    }


@router.get("/predict/{session_id}/boundary-image/{step_idx}")
def export_eddy_boundary_image(session_id: str, step_idx: int):
    st = state.eddy_prediction_cache.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = st.get("steps", [])
    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(status_code=400, detail="Step out of range")

    tmp_dir = Path(tempfile.gettempdir()) / "oceanrace_eddy_exports"
    out_file = tmp_dir / f"{session_id}_step_{step_idx}.png"
    ec._render_eddy_boundary_image(steps[step_idx], out_file)
    return FileResponse(path=out_file, media_type="image/png", filename=out_file.name)


@router.get("/predict/{session_id}/reference-image/{step_idx}")
def get_eddy_reference_image(session_id: str, step_idx: int):
    st = state.eddy_prediction_cache.get(session_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = st.get("steps", [])
    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(status_code=400, detail="Step out of range")

    run_tag = str(st.get("run_tag", "global_boundary_full_20260403"))
    time_index = int(steps[step_idx].get("time_index", -1))
    ref = ec._find_eddy_reference_figure(run_tag, time_index)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"Reference image not found for run={run_tag}, time_index={time_index}")

    return FileResponse(path=ref, media_type="image/png", filename=ref.name)
