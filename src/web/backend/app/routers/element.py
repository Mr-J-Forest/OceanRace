"""要素预报：默认路径、数据集信息、长时预测与按步查询。"""
from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd
import torch
import xarray as xr
from fastapi import APIRouter, HTTPException

from src.element_forecasting.dataset import ElementForecastWindowDataset
from src.element_forecasting.predictor import ElementForecastPredictor

from .. import state
from ..paths import resolve_data_path_or_path_txt
from ..schemas import DatasetInfoRequest, PredictRequest
from ..services.element_mask import extract_mask

router = APIRouter(prefix="/api", tags=["element"])


def _element_forecast_dataset(data_path: str, norm_path: str) -> ElementForecastWindowDataset:
    """进程内复用数据集与 NetCDF 句柄；含 data/norm 的 mtime 以便文件更新后自动换新。"""
    try:
        dm = float(os.path.getmtime(data_path))
    except OSError:
        dm = 0.0
    try:
        nm = float(os.path.getmtime(norm_path)) if os.path.isfile(norm_path) else 0.0
    except OSError:
        nm = 0.0
    ap_data = os.path.abspath(data_path)
    ap_norm = os.path.abspath(norm_path)
    key = (ap_data, ap_norm, dm, nm, 24, 72)
    with state.element_predictor_lock:
        cached = state.element_forecast_dataset_cache.get(key)
        if cached is not None:
            return cached
        ds = ElementForecastWindowDataset(
            data_file=data_path,
            input_steps=24,
            output_steps=72,
            split=None,
            norm_stats_path=norm_path,
        )
        state.element_forecast_dataset_cache[key] = ds
        stale = [k for k in state.element_forecast_dataset_cache if k[0] == ap_data and k != key]
        for k in stale:
            old = state.element_forecast_dataset_cache.pop(k, None)
            if old is not None:
                try:
                    old._close_all_open_ds()
                except Exception:
                    pass
        return ds


@router.get("/default-data-path")
def get_default_data_path():
    return {"path": "data/processed/element_forecasting/path.txt"}


@router.post("/dataset-info")
def get_dataset_info(req: DatasetInfoRequest):
    data_path = resolve_data_path_or_path_txt(req.data_path)
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

    try:
        norm_path = resolve_data_path_or_path_txt("data/processed/normalization/element_forecasting_norm.json")
        dataset = _element_forecast_dataset(data_path, norm_path)

        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty or insufficient steps")

        ds = xr.open_dataset(data_path)
        try:
            times = pd.to_datetime(ds["time"].values)
            first_time = times[dataset._windows[0]].strftime("%Y-%m-%d %H:%M:%S")
            last_time = times[dataset._windows[-1]].strftime("%Y-%m-%d %H:%M:%S")
        finally:
            ds.close()

        return {
            "max_index": len(dataset) - 1,
            "info": f"可用预测窗口: {len(dataset)}\n第一步: {first_time}\n最后一步: {last_time}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def run_prediction(req: PredictRequest):
    data_path = resolve_data_path_or_path_txt(req.data_path)
    model_path = resolve_data_path_or_path_txt(req.model_path)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")

    try:
        norm_path = resolve_data_path_or_path_txt("data/processed/normalization/element_forecasting_norm.json")
        dataset = _element_forecast_dataset(data_path, norm_path)
        if req.start_idx < 0 or req.start_idx >= len(dataset):
            raise HTTPException(status_code=400, detail="Start index out of range")

        sample = dataset[req.start_idx]
        x_tensor = sample["x"].unsqueeze(0)
        y_tensor = sample["y"].numpy()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ap_model = os.path.normcase(os.path.abspath(model_path))
        try:
            mtime = float(os.path.getmtime(model_path))
        except OSError:
            mtime = 0.0
        cache_key = (ap_model, device, mtime)
        with state.element_predictor_lock:
            predictor = state.element_predictor_cache.get(cache_key)
            if predictor is None:
                predictor = ElementForecastPredictor(
                    checkpoint_path=model_path, device=device, norm_stats_path=norm_path
                )
                state.element_predictor_cache[cache_key] = predictor
                # 防止无限增长：只保留同一模型路径下最新 mtime 的一条
                stale = [k for k in state.element_predictor_cache if k[0] == ap_model and k != cache_key]
                for k in stale:
                    del state.element_predictor_cache[k]

        result = predictor.predict_long_horizon(
            x=x_tensor,
            target_steps=72,
            overlap_steps=4,
            enable_overlap_blend=True,
            denormalize=True,
            return_cpu=True,
        )
        pred_numpy = result["pred"][0].numpy()
        var_names = result.get("var_names", ["SST", "SSS", "SSU", "SSV"])

        valid_mask_tensor = sample.get("y_valid", None)
        mask_numpy = valid_mask_tensor.numpy() if valid_mask_tensor is not None else None

        session_id = "default_session"
        state.prediction_cache[session_id] = {
            "pred": pred_numpy,
            "true": y_tensor,
            "mask": mask_numpy,
            "vars": var_names,
        }

        return {"message": "Prediction successful", "session_id": session_id, "steps": pred_numpy.shape[0]}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/predict/{session_id}/step/{step_idx}")
def get_prediction_step(session_id: str, step_idx: int):
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")

    st = state.prediction_cache[session_id]
    pred = st["pred"]
    mask = st["mask"]
    vars_names = st["vars"]

    if step_idx < 0 or step_idx >= pred.shape[0]:
        raise HTTPException(status_code=400, detail="Step out of range")

    step_data = pred[step_idx]
    h, w = step_data.shape[1], step_data.shape[2]

    response_data = []
    for i in range(min(4, step_data.shape[0])):
        data_slice = step_data[i].copy()
        mask_slice = extract_mask(mask, step_idx, i, h, w)
        if mask_slice is not None and mask_slice.shape == (h, w):
            data_slice[mask_slice < 0.5] = np.nan

        data_slice = np.where(np.isnan(data_slice), None, data_slice)
        response_data.append(
            {
                "var": vars_names[i],
                "data": data_slice.tolist(),
            }
        )

    return {"step": step_idx, "data": response_data}


@router.get("/predict/{session_id}/curve")
def get_prediction_curve(session_id: str, r: int | None = None, c: int | None = None):
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")

    st = state.prediction_cache[session_id]
    pred_numpy = st["pred"]
    mask_numpy = st["mask"]
    var_names = st["vars"]

    num_steps, _, h, w = pred_numpy.shape
    response_data = []

    use_point = False
    if r is not None and c is not None and 0 <= r < h and 0 <= c < w:
        use_point = True

    for i in range(min(4, pred_numpy.shape[1])):
        mean_vals = []
        for t in range(num_steps):
            data_slice = pred_numpy[t, i]
            if use_point:
                val = data_slice[r, c]
                mean_vals.append(float(val) if not np.isnan(val) else None)
            else:
                mask_slice = extract_mask(mask_numpy, t, i, h, w)
                if mask_slice is not None and mask_slice.shape == (h, w):
                    valid_data = data_slice[mask_slice >= 0.5]
                else:
                    valid_data = data_slice[~np.isnan(data_slice)]
                mean_vals.append(float(np.mean(valid_data)) if len(valid_data) > 0 else None)

        response_data.append(
            {
                "var": var_names[i],
                "means": mean_vals,
            }
        )

    return {"data": response_data, "point": {"r": r, "c": c} if use_point else None}
