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

        centers = []
        for obj in res0.get("objects", []):
            try:
                center_yx = obj.get("center_yx")
                if center_yx is None:
                    continue
                r = float(center_yx[0])
                c = float(center_yx[1])
                
                real_lat = float(lat[int(round(r))]) if lat.size > 0 and 0 <= int(round(r)) < lat.size else 0.0
                real_lon = float(lon[int(round(c))]) if lon.size > 0 and 0 <= int(round(c)) < lon.size else 0.0
                
                centers.append({
                    "r": float(r),
                    "c": float(c),
                    "lat": real_lat,
                    "lon": real_lon,
                    "class_id": int(obj.get("class_id", 0)),
                    "area": int(obj.get("area", 0)),
                    "bbox": obj.get("bbox_yx", [])
                })
            except Exception:
                continue

        if not centers:
            # Fallback to the old mask extraction, format it properly
            raw_centers = ec._extract_centers_from_mask(pred_mask, min_region_pixels=req.min_region_pixels)
            for rc in raw_centers:
                r, c, cid = rc
                real_lat = float(lat[int(round(r))]) if lat.size > 0 and 0 <= int(round(r)) < lat.size else 0.0
                real_lon = float(lon[int(round(c))]) if lon.size > 0 and 0 <= int(round(c)) < lon.size else 0.0
                centers.append({
                    "r": float(r),
                    "c": float(c),
                    "lat": real_lat,
                    "lon": real_lon,
                    "class_id": int(cid),
                    "area": 0,
                    "bbox": []
                })

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

from pydantic import BaseModel

class EddyTrackRequest(BaseModel):
    data_path: str
    model_path: str
    start_day_index: int
    r: float
    c: float
    class_id: int

@router.post("/track")
async def generate_eddy_track(req: EddyTrackRequest):
    """
    基于深度学习模型进行实时的多时相物理回溯与演变追踪 (±15 天)。
    """
    import math
    from src.eddy_detection.predictor import infer_batch_to_objects
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 我们往前追溯 15 天，往后追踪 15 天
    TRACK_RADIUS = 15
    start_idx = max(0, req.start_day_index - TRACK_RADIUS)
    anchor_batch_idx = req.start_day_index - start_idx
    
    horizon = TRACK_RADIUS * 2 + 1
    
    x_batch, time_indices, adt_maps, max_index = await asyncio.to_thread(
        ec._build_unlabeled_eddy_batch,
        data_path=ec.resolve_eddy_clean_nc_path(req.data_path),
        start_idx=start_idx,
        input_steps=1,
        horizon_steps=horizon,
    )
    
    model_path = resolve_path(req.model_path)
    model = ec._get_cached_eddy_model(
        model_path=model_path,
        in_channels=int(x_batch.shape[1]),
        base_channels=32,
        device=device,
    )
    
    results = infer_batch_to_objects(model, x_batch, device, min_region_pixels=16)
    del x_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    meta = ec._get_cached_eddy_dataset_meta(ec.resolve_eddy_clean_nc_path(req.data_path))
    lat_coords = meta.get("lat_list", [])
    lon_coords = meta.get("lon_list", [])
    
    def _get_phys_lat_lon(r: float, c: float):
        real_lat = float(lat_coords[int(round(r))]) if len(lat_coords) > 0 and 0 <= int(round(r)) < len(lat_coords) else 0.0
        real_lon = float(lon_coords[int(round(c))]) if len(lon_coords) > 0 and 0 <= int(round(c)) < len(lon_coords) else 0.0
        return real_lat, real_lon

    def find_nearest_eddy(day_result, target_r, target_c, target_class_id, max_dist=15.0):
        best_obj = None
        best_dist = float('inf')
        for obj in day_result.get("objects", []):
            if int(obj.get("class_id", 0)) != target_class_id:
                continue
            cy, cx = obj.get("center_yx", [None, None])
            if cy is None or cx is None:
                continue
            dist = math.hypot(cy - target_r, cx - target_c)
            if dist < best_dist and dist < max_dist:
                best_dist = dist
                best_obj = obj
        return best_obj, best_dist

    # 向前追踪 (Forward)
    forward_track = []
    curr_r, curr_c = req.r, req.c
    patience = 2
    missing_count = 0
    for i in range(anchor_batch_idx + 1, len(results)):
        if i >= len(results):
            break
        match, dist = find_nearest_eddy(results[i], curr_r, curr_c, req.class_id, max_dist=15.0 + missing_count * 5.0)
        if not match:
            missing_count += 1
            if missing_count > patience:
                break
            continue
            
        missing_count = 0
        curr_r, curr_c = match["center_yx"]
        rlat, rlon = _get_phys_lat_lon(curr_r, curr_c)
        area = match.get("area", 0)
        adt_val = float(adt_maps[i][int(curr_r), int(curr_c)])
        intensity = abs(adt_val) * 100.0  # scale for visualization
        forward_track.append({
            "day_index": time_indices[i],
            "r": float(curr_r),
            "c": float(curr_c),
            "lat": rlat,
            "lon": rlon,
            "class_id": req.class_id,
            "area": area,
            "intensity": intensity,
            "shift": float(dist)
        })

    # 向后回溯 (Backward)
    backward_track = []
    curr_r, curr_c = req.r, req.c
    missing_count = 0
    for i in range(anchor_batch_idx - 1, -1, -1):
        if i >= len(results):
            continue
        match, dist = find_nearest_eddy(results[i], curr_r, curr_c, req.class_id, max_dist=15.0 + missing_count * 5.0)
        if not match:
            missing_count += 1
            if missing_count > patience:
                break
            continue
            
        missing_count = 0
        curr_r, curr_c = match["center_yx"]
        rlat, rlon = _get_phys_lat_lon(curr_r, curr_c)
        area = match.get("area", 0)
        adt_val = float(adt_maps[i][int(curr_r), int(curr_c)])
        intensity = abs(adt_val) * 100.0
        backward_track.append({
            "day_index": time_indices[i],
            "r": float(curr_r),
            "c": float(curr_c),
            "lat": rlat,
            "lon": rlon,
            "class_id": req.class_id,
            "area": area,
            "intensity": intensity,
            "shift": float(dist)
        })

    backward_track.reverse()
    
    # Anchor 节点
    anchor_rlat, anchor_rlon = _get_phys_lat_lon(req.r, req.c)
    anchor_adt = float(adt_maps[anchor_batch_idx][int(req.r), int(req.c)]) if anchor_batch_idx < len(adt_maps) else 0.0
    anchor_node = {
        "day_index": req.start_day_index,
        "r": req.r,
        "c": req.c,
        "lat": anchor_rlat,
        "lon": anchor_rlon,
        "class_id": req.class_id,
        "area": 0,
        "intensity": abs(anchor_adt) * 100.0,
        "shift": 0.0
    }
    if anchor_batch_idx < len(results):
        anchor_match, _ = find_nearest_eddy(results[anchor_batch_idx], req.r, req.c, req.class_id, max_dist=2.0)
        if anchor_match:
            anchor_node["area"] = anchor_match.get("area", 0)

    nodes = backward_track + [anchor_node] + forward_track
    
    intensities = [n["intensity"] for n in nodes]

    return {
        "start_day": req.start_day_index,
        "nodes": nodes,
        "min_intensity": min(intensities) if intensities else 0.0,
        "max_intensity": max(intensities) if intensities else 0.0
    }
