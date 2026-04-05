import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_mask(mask_numpy, t_idx, c_idx, H, W):
    return None

def draw_spatial_plot(pred_numpy, mask_numpy, var_names, step_idx):
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=[f"{v} (预测步: {step_idx+1}, +{(step_idx+1)*6}H)" for v in var_names[:4]],
                        vertical_spacing=0.1, horizontal_spacing=0.1)
    
    step_pred = pred_numpy[step_idx]
    H, W = step_pred.shape[1], step_pred.shape[2]
        
    for i in range(min(4, step_pred.shape[0])):
        var_name = var_names[i]
        data_slice = step_pred[i].copy()
        
        mask_slice = extract_mask(mask_numpy, step_idx, i, H, W)
        if mask_slice is not None and mask_slice.shape == (H, W):
            data_slice[mask_slice < 0.5] = np.nan
        
        if var_name in ["SST", "海温"]:
            cmap_name = "RdBu_r"
        elif var_name in ["SSS", "盐度"]:
            cmap_name = "Viridis"
        else:
            cmap_name = "RdBu_r"
            
        valid_data = data_slice[~np.isnan(data_slice)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [1, 99])
        else:
            vmin, vmax = None, None

        row = (i // 2) + 1
        col = (i % 2) + 1
        
        hm = go.Heatmap(
            z=data_slice,
            colorscale=cmap_name,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            colorbar=dict(
                thickness=15, 
                len=0.45, 
                # Place colorbar near the plot dynamically.
                # We can place it at coordinates.
                y=0.75 if row==1 else 0.25, 
                x=0.46 if col==1 else 1.0
            ),
            hoverinfo="z"
        )
        fig.add_trace(hm, row=row, col=col)
        
        # Hide axes and set grid background to gray to mimic bad point masking
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, 
                         scaleanchor=f"x{i+1}" if i>0 else "x", scaleratio=1, row=row, col=col)

    fig.update_layout(
        height=800, 
        paper_bgcolor='white', 
        plot_bgcolor='#dddddd', 
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

pred_numpy = np.random.rand(1, 4, 10, 10)
fig = draw_spatial_plot(pred_numpy, None, ["SST", "SSS", "SSU", "SSV"], 0)
print("Spatial plot generated successfully.")

