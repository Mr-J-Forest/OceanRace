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
from ..services.element_historical import element_metrics_masked, load_historical_same_period
from ..services.element_analysis import calculate_warnings, calculate_correlation
from ..services.element_mask import extract_mask

router = APIRouter(prefix="/api", tags=["element"])


def _grid_to_json_list(arr: np.ndarray) -> list:
    out = np.asarray(arr, dtype=np.float64)
    out = np.where(np.isfinite(out), out, np.nan)
    return np.where(np.isnan(out), None, out).tolist()


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
            lon_arr = np.asarray(ds["longitude"].values) if "longitude" in ds else (
                np.asarray(ds["lon"].values) if "lon" in ds else np.array([])
            )
            lat_arr = np.asarray(ds["latitude"].values) if "latitude" in ds else (
                np.asarray(ds["lat"].values) if "lat" in ds else np.array([])
            )
            lon_info = f"{float(lon_arr[0]):.2f}°E ~ {float(lon_arr[-1]):.2f}°E" if len(lon_arr) else "未知"
            lat_info = f"{float(lat_arr[0]):.2f}°N ~ {float(lat_arr[-1]):.2f}°N" if len(lat_arr) else "未知"
        finally:
            ds.close()

        return {
            "max_index": len(dataset) - 1,
            "info": f"可用预测窗口: {len(dataset)}\n第一步: {first_time}\n最后一步: {last_time}\n经度: {lon_info}\n纬度: {lat_info}",
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
        var_names = list(result.get("var_names", ["SST", "SSS", "SSU", "SSV"]))

        y_t = torch.from_numpy(y_tensor.astype(np.float32)).unsqueeze(0)
        true_phys = predictor._destandardize_pred(y_t)[0].numpy()

        valid_mask_tensor = sample.get("y_valid", None)
        mask_numpy = valid_mask_tensor.numpy() if valid_mask_tensor is not None else None

        nc_vars = tuple(str(v) for v in sample.get("var_names", ("sst", "sss", "ssu", "ssv")))
        t0 = int(sample.get("t0", 0))
        in_steps = int(dataset.input_steps)
        out_steps = int(pred_numpy.shape[0])
        hist_arr: np.ndarray | None = None
        if bool(req.include_historical):
            hist_arr = load_historical_same_period(
                data_path, t0, in_steps, out_steps, nc_vars, max_time_delta_hours=72.0
            )

        session_id = "default_session"
        state.prediction_cache[session_id] = {
            "pred": pred_numpy,
            "true": true_phys,
            "historical": hist_arr,
            "mask": mask_numpy,
            "vars": var_names,
            "step_hours": float(req.step_hours),
            "t0": t0,
            "input_steps": in_steps,
            "data_path": data_path,
            "nc_var_names": nc_vars,
        }

        # 获取经纬度边界
        ds = xr.open_dataset(data_path)
        try:
            lon_arr = np.asarray(ds["longitude"].values) if "longitude" in ds else (
                np.asarray(ds["lon"].values) if "lon" in ds else np.array([])
            )
            lat_arr = np.asarray(ds["latitude"].values) if "latitude" in ds else (
                np.asarray(ds["lat"].values) if "lat" in ds else np.array([])
            )
            lon_list = lon_arr.astype(float).tolist() if len(lon_arr) else []
            lat_list = lat_arr.astype(float).tolist() if len(lat_arr) else []
        finally:
            ds.close()

        return {
            "message": "Prediction successful",
            "session_id": session_id,
            "steps": pred_numpy.shape[0],
            "step_hours": float(req.step_hours),
            "has_historical": bool(hist_arr is not None and np.any(np.isfinite(hist_arr))),
            "lon": lon_list,
            "lat": lat_list,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/predict/{session_id}/step/{step_idx}")
def get_prediction_step(session_id: str, step_idx: int, compare: int = 0):
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")

    st = state.prediction_cache[session_id]
    pred = st["pred"]
    true = st["true"]
    hist = st.get("historical")
    mask = st["mask"]
    vars_names = st["vars"]

    if step_idx < 0 or step_idx >= pred.shape[0]:
        raise HTTPException(status_code=400, detail="Step out of range")

    step_data = pred[step_idx]
    true_step = true[step_idx]
    hist_step = hist[step_idx] if isinstance(hist, np.ndarray) and hist.shape[0] > step_idx else None
    h, w = step_data.shape[1], step_data.shape[2]

    response_data = []
    for i in range(min(4, step_data.shape[0])):
        data_slice = step_data[i].copy()
        mask_slice = extract_mask(mask, step_idx, i, h, w)
        if mask_slice is not None and mask_slice.shape == (h, w):
            data_slice[mask_slice < 0.5] = np.nan

        item: dict = {
            "var": vars_names[i],
            "data": _grid_to_json_list(data_slice),
        }
        if int(compare) == 1:
            ts = true_step[i].copy()
            if mask_slice is not None and mask_slice.shape == (h, w):
                ts = ts.copy()
                ts[mask_slice < 0.5] = np.nan
            diff = data_slice - ts
            item["true"] = _grid_to_json_list(ts)
            item["diff"] = _grid_to_json_list(diff)
            if hist_step is not None and i < hist_step.shape[0]:
                hs = hist_step[i].copy()
                if mask_slice is not None and mask_slice.shape == (h, w):
                    hs[mask_slice < 0.5] = np.nan
                item["historical"] = _grid_to_json_list(hs)
            else:
                item["historical"] = None

        response_data.append(item)

    return {"step": step_idx, "compare": bool(int(compare) == 1), "data": response_data}


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


@router.get("/predict/{session_id}/curve-compare")
def get_prediction_curve_compare(session_id: str, r: int | None = None, c: int | None = None):
    """单点或区域平均：预测 / 实况 / 历史同期 同步序列为对比曲线。"""
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")

    st = state.prediction_cache[session_id]
    pred_numpy = st["pred"]
    true_numpy = st["true"]
    hist_numpy = st.get("historical")
    mask_numpy = st["mask"]
    var_names = st["vars"]

    num_steps, _, h, w = pred_numpy.shape
    use_point = False
    if r is not None and c is not None and 0 <= r < h and 0 <= c < w:
        use_point = True

    response_data = []
    for i in range(min(4, pred_numpy.shape[1])):
        pred_s, true_s, hist_s = [], [], []
        for t in range(num_steps):
            ps = pred_numpy[t, i]
            ts = true_numpy[t, i]
            mask_slice = extract_mask(mask_numpy, t, i, h, w)
            if use_point:
                pv = float(ps[r, c]) if np.isfinite(ps[r, c]) else None
                tv = float(ts[r, c]) if np.isfinite(ts[r, c]) else None
                if isinstance(hist_numpy, np.ndarray) and t < hist_numpy.shape[0]:
                    hv = hist_numpy[t, i]
                    hv = float(hv[r, c]) if np.isfinite(hv[r, c]) else None
                else:
                    hv = None
            else:
                if mask_slice is not None and mask_slice.shape == (h, w):
                    vm = mask_slice >= 0.5
                    pv = float(np.mean(ps[vm])) if np.any(vm) else None
                    tv = float(np.mean(ts[vm])) if np.any(vm) else None
                else:
                    pv = float(np.nanmean(ps))
                    tv = float(np.nanmean(ts))
                if isinstance(hist_numpy, np.ndarray) and t < hist_numpy.shape[0]:
                    hs = hist_numpy[t, i]
                    if mask_slice is not None and mask_slice.shape == (h, w):
                        vm = mask_slice >= 0.5
                        hv = float(np.mean(hs[vm])) if np.any(vm) else None
                    else:
                        hv = float(np.nanmean(hs)) if np.any(np.isfinite(hs)) else None
                else:
                    hv = None
            pred_s.append(pv)
            true_s.append(tv)
            hist_s.append(hv)

        response_data.append(
            {
                "var": var_names[i],
                "pred": pred_s,
                "true": true_s,
                "historical": hist_s,
            }
        )

    return {"data": response_data, "point": {"r": r, "c": c} if use_point else None}


@router.get("/predict/{session_id}/section")
def get_forecast_section(
    session_id: str,
    axis: str = "row",
    pos: int = 0,
    var_index: int = 0,
    field: str = "pred",
):
    """
    断面：无垂向深度时，返回「预报时效 × 沿格点剖线」的二维断面（lat 或 lon 方向一条线 × 时间）。
    axis=row 固定行 pos，列方向为横轴；axis=col 固定列，行方向为横轴。
    field: pred | true | diff | historical
    """
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")

    st = state.prediction_cache[session_id]
    pred = st["pred"]
    true = st["true"]
    hist = st.get("historical")
    _, _, h, w = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
    vi = int(np.clip(var_index, 0, pred.shape[1] - 1))
    ax = str(axis).strip().lower()
    if ax not in ("row", "col"):
        raise HTTPException(status_code=400, detail="axis must be row or col")
    fld = str(field).strip().lower()
    if fld not in ("pred", "true", "diff", "historical"):
        raise HTTPException(status_code=400, detail="field must be pred/true/diff/historical")

    z_stack = []
    for t in range(pred.shape[0]):
        p = pred[t, vi]
        tr = true[t, vi]
        if fld == "pred":
            z_stack.append(p)
        elif fld == "true":
            z_stack.append(tr)
        elif fld == "diff":
            z_stack.append(p - tr)
        else:
            if isinstance(hist, np.ndarray) and t < hist.shape[0]:
                z_stack.append(hist[t, vi])
            else:
                z_stack.append(np.full_like(p, np.nan))

    if ax == "row":
        pos = int(np.clip(pos, 0, h - 1))
        mat = np.stack([z[pos, :] for z in z_stack], axis=0)
        x_label = "沿剖面距离 (横向格点 / 经度方向)"
    else:
        pos = int(np.clip(pos, 0, w - 1))
        mat = np.stack([z[:, pos] for z in z_stack], axis=0)
        x_label = "沿剖面距离 (纵向格点 / 纬度方向)"

    step_h = float(st.get("step_hours", 1.0))
    y_hours = [float((i + 1) * step_h) for i in range(mat.shape[0])]

    return {
        "axis": ax,
        "pos": pos,
        "var_index": vi,
        "field": fld,
        "x_label": x_label,
        "y_label": "预报时效 (h)",
        "y_hours": y_hours,
        "z": _grid_to_json_list(mat),
    }


@router.get("/predict/{session_id}/evaluation")
def get_forecast_evaluation(
    session_id: str,
    var_index: int = 0,
    step_idx: int | None = None,
    error_thresh: float | None = None,
):
    """
    全时段每变量 MSE/MAE/R2；可选单步空间 |误差| 与超阈高亮。
    """
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")

    st = state.prediction_cache[session_id]
    pred = st["pred"]
    true = st["true"]
    mask = st["mask"]
    vars_names = st["vars"]

    metrics = element_metrics_masked(pred, true, mask)
    named: dict[str, dict[str, float]] = {}
    for k, v in metrics["per_var_index"].items():
        idx = int(k)
        name = vars_names[idx] if idx < len(vars_names) else str(idx)
        named[name] = v

    out: dict = {
        "per_variable": named,
        "per_step_mse": metrics["per_step_mse"],
        "step_hours": float(st.get("step_hours", 1.0)),
    }

    vi = int(np.clip(var_index, 0, pred.shape[1] - 1))
    if step_idx is not None:
        s = int(np.clip(step_idx, 0, pred.shape[0] - 1))
        err = np.abs(pred[s, vi] - true[s, vi])
        h, w = err.shape
        mask_slice = extract_mask(mask, s, vi, h, w)
        if mask_slice is not None and mask_slice.shape == (h, w):
            err = err.copy()
            err[mask_slice < 0.5] = np.nan
        out["step"] = s
        out["var"] = vars_names[vi] if vi < len(vars_names) else str(vi)
        out["spatial_abs_error"] = _grid_to_json_list(err)
        if error_thresh is not None and float(error_thresh) > 0:
            thr = float(error_thresh)
            highlight = np.where(np.isfinite(err) & (err >= thr), 1.0, np.nan)
            out["error_threshold"] = thr
            out["spatial_highlight"] = _grid_to_json_list(highlight)
        else:
            out["error_threshold"] = None
            out["spatial_highlight"] = None

    return out


from pydantic import BaseModel
class WarningRequest(BaseModel):
    thresholds: dict[str, float]

@router.post("/predict/{session_id}/warnings")
def get_prediction_warnings(session_id: str, req: WarningRequest):
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")
        
    st = state.prediction_cache[session_id]
    pred_numpy = st["pred"]
    mask_numpy = st["mask"]
    var_names = st["vars"]
    
    lons = st.get("longitude", [])
    lats = st.get("latitude", [])
    
    res = calculate_warnings(pred_numpy, mask_numpy, var_names, req.thresholds, lons, lats)
    return res

@router.get("/predict/{session_id}/correlation")
def get_prediction_correlation(session_id: str, var1: str, var2: str):
    if session_id not in state.prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")
        
    st = state.prediction_cache[session_id]
    pred_numpy = st["pred"]
    mask_numpy = st["mask"]
    var_names = st["vars"]
    
    res = calculate_correlation(pred_numpy, mask_numpy, var_names, var1, var2)
    if res is None:
        raise HTTPException(status_code=400, detail="Invalid variables for correlation")
        
    return res
