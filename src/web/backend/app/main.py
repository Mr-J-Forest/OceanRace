from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
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

# In-memory store for the last prediction to serve slices efficiently
prediction_cache = {}

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
def get_prediction_curve(session_id: str):
    if session_id not in prediction_cache:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
