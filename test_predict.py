import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('src'))
from utils.logger import setup_logging, get_logger
from utils.visualization_defaults import apply_matplotlib_defaults, standard_savefig_kwargs, DEFAULT_CMAP
from element_forecasting.predictor import ElementForecastPredictor
from element_forecasting.dataset import ElementForecastWindowDataset

logger = get_logger(__name__)

def main():
    setup_logging()
    apply_matplotlib_defaults()
    
    with open("data/processed/element_forecasting/path.txt", "r") as f:
        nc_path = f.read().strip()
        
    logger.info(f"Loading data from {nc_path}")
    
    input_steps = 24
    output_steps = 72
    norm_path = "data/processed/normalization/element_forecasting_norm.json"
    
    dataset = ElementForecastWindowDataset(
        data_file=nc_path,
        input_steps=input_steps,
        output_steps=output_steps,
        split=None,
        norm_stats_path=norm_path
    )
    
    if len(dataset) == 0:
        logger.error("Dataset has no available samples.")
        return
        
    last_idx = len(dataset) - 1
    logger.info(f"Selecting the last window in the dataset (idx={last_idx})")
    
    sample = dataset[last_idx]
    x_tensor = sample["x"].unsqueeze(0)
    
    # Check if a mask is available
    valid_mask = None
    if "y_valid" in sample and sample["y_valid"] is not None:
        valid_mask = sample["y_valid"].numpy()
    
    model_path = "outputs/element_forecasting/checkpoints/hybrid_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {model_path} onto {device}")
    
    predictor = ElementForecastPredictor(
        checkpoint_path=model_path, 
        device=device,
        norm_stats_path=norm_path
    )
    
    logger.info("Running prediction with predict_long_horizon...")
    res = predictor.predict_long_horizon(
        x=x_tensor, 
        target_steps=output_steps,
        overlap_steps=4,
        enable_overlap_blend=True,
        denormalize=True, 
        return_cpu=True
    )
    
    pred_data = res["pred"][0].numpy() # shape: (output_steps, C, H, W)
    var_names = res.get("var_names", ["sst", "sss", "ssu", "ssv"])
    
    # We will visualize the prediction for the final step (e.g. +24h)
    step_to_plot = output_steps - 1
    pred_step = pred_data[step_to_plot] # shape: (C, H, W)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        data_2d = pred_step[i]
        
        # Apply mask to not show land as weird values (set to NaN)
        if valid_mask is not None:
            # valid_mask usually matches output shape or applies dynamically
            # If valid_mask is shaped (output_steps, C, H, W):
            if valid_mask.ndim == 4:
                m = valid_mask[step_to_plot, i]
            elif valid_mask.ndim == 3:
                m = valid_mask[i]
            elif valid_mask.ndim == 2:
                m = valid_mask
            else:
                m = valid_mask
            
            data_2d = np.where(m, data_2d, np.nan)
        
        # Default colormap handling (viridus is good, RdBu_r is often used for velocity/temperature anomaly)
        cmap = DEFAULT_CMAP if var_name.lower() != 'sss' else "viridis"
        if var_name.lower() in ["ssu", "ssv"]:
            cmap = "RdBu_r"
            limit = max(abs(np.nanmin(data_2d)), abs(np.nanmax(data_2d)))
            im = ax.imshow(data_2d, origin='lower', cmap=cmap, vmin=-limit, vmax=limit)
        else:
            im = ax.imshow(data_2d, origin='lower', cmap=cmap)
            
        ax.set_title(f"{var_name.upper()} Forecast (+{step_to_plot+1}h)", fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
    plt.suptitle(f"Element Forecasting Model Prediction - Last {output_steps}h Input", fontsize=16)
    plt.tight_layout()
    
    out_dir = os.path.join("outputs", "element_forecasting", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_prediction_last_{output_steps}h.png")
    
    fig.savefig(out_path, **standard_savefig_kwargs())
    logger.info(f"Saved plotting to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
