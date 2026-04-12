from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import xarray as xr

# Ensure src and root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # adds src/
sys.path.insert(0, PROJECT_ROOT) # adds root

from src.element_forecasting.predictor import ElementForecastPredictor
from src.element_forecasting.dataset import ElementForecastWindowDataset
from src.anomaly_detection.dataset import AnomalyFrameDataset

app = FastAPI(title="OceanRace Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/api/default-data-path")
def get_default_data_path():
    path_file = resolve_path("data/processed/element_forecasting/path.txt")
    if os.path.exists(path_file):
        with open(path_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return {"path": content}
    return {"path": ""}

@app.post("/api/dataset-info")
def get_dataset_info(req: DatasetInfoRequest):
    data_path = resolve_path(req.data_path.strip('\"\''))
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
        
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="Dataset is empty or insufficient steps")
            
        ds = xr.open_dataset(data_path)
        times = pd.to_datetime(ds['time'].values)
        first_time = times[dataset._windows[0]].strftime("%Y-%m-%d %H:%M:%S")
        last_time = times[dataset._windows[-1]].strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "max_index": len(dataset) - 1,
            "info": f"可用预测窗口: {len(dataset)}\n第一步: {first_time}\n最后一步: {last_time}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))

def extract_mask(mask_numpy, t_idx, c_idx, H, W):
    if mask_numpy is None: return None
    if mask_numpy.shape == (H, W): return mask_numpy
    elif mask_numpy.ndim == 3:
        if mask_numpy.shape[0] == 4: return mask_numpy[min(c_idx, 3)]
        else: return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1)]
    elif mask_numpy.ndim == 4:
        return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1), min(c_idx, mask_numpy.shape[1] - 1)]
    return mask_numpy

@app.get("/api/predict/{session_id}/step/{step_idx}")
def get_prediction_step(session_id: str, step_idx: int):
    if session_id not in prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")
        
    state = prediction_cache[session_id]
    pred = state["pred"]
    mask = state["mask"]
    vars_names = state["vars"]
    
    if step_idx < 0 or step_idx >= pred.shape[0]:
        raise HTTPException(status_code=400, detail="Step out of range")
        
    step_data = pred[step_idx]
    H, W = step_data.shape[1], step_data.shape[2]
    
    response_data = []
    for i in range(min(4, step_data.shape[0])):
        data_slice = step_data[i].copy()
        mask_slice = extract_mask(mask, step_idx, i, H, W)
        if mask_slice is not None and mask_slice.shape == (H, W):
            data_slice[mask_slice < 0.5] = np.nan
            
        # Replace NaN with null for JSON serialization
        data_slice = np.where(np.isnan(data_slice), None, data_slice)
        response_data.append({
            "var": vars_names[i],
            "data": data_slice.tolist()
        })
        
    return {"step": step_idx, "data": response_data}

@app.get("/api/predict/{session_id}/curve")
def get_prediction_curve(session_id: str, r: int = None, c: int = None):
    if session_id not in prediction_cache:
        raise HTTPException(status_code=404, detail="Session not found")
        
    state = prediction_cache[session_id]
    pred_numpy = state["pred"]
    mask_numpy = state["mask"]
    var_names = state["vars"]
    
    num_steps, _, H, W = pred_numpy.shape
    response_data = []
    
    # Validation for specific point coordinates
    use_point = False
    if r is not None and c is not None:
        if 0 <= r < H and 0 <= c < W:
            use_point = True

    for i in range(min(4, pred_numpy.shape[1])):
        mean_vals = []
        for t in range(num_steps):
            data_slice = pred_numpy[t, i]
            
            if use_point:
                val = data_slice[r, c]
                mean_vals.append(float(val) if not np.isnan(val) else None)
            else:
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
        
    return {"data": response_data, "point": {"r": r, "c": c} if use_point else None}


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
        "labels_timestamps_aligned": len(split_labels_raw) == len(timestamps),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
