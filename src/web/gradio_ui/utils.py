import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def extract_mask(mask_numpy, t_idx, c_idx, H, W):
    if mask_numpy is None:
        return None
    if mask_numpy.shape == (H, W):
        return mask_numpy
    elif mask_numpy.ndim == 3:
        if mask_numpy.shape[0] == 4:
            return mask_numpy[min(c_idx, 3)]
        else:
            return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1)]
    elif mask_numpy.ndim == 4:
        return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1), min(c_idx, mask_numpy.shape[1] - 1)]
    return mask_numpy

def create_heatmap(data, vmin, vmax, cmap_name, row, col, x_axis_id="x"):
    return go.Heatmap(
        z=data, colorscale=cmap_name, zmin=vmin, zmax=vmax, showscale=True, zsmooth="best",
        colorbar=dict(thickness=12, len=0.45, y=0.79 if row==1 else 0.21, 
                      x=0.455 if col==1 else 1.0, outlinewidth=0, tickfont=dict(size=11), title=""),
        hoverinfo="z+x+y"
    )

def draw_spatial_plot(pred_numpy, mask_numpy, var_names, step_idx):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{v.upper()} (+{(step_idx+1)*6}H)" for v in var_names[:4]],
                        vertical_spacing=0.08, horizontal_spacing=0.08)
    step_pred = pred_numpy[step_idx]
    H, W = step_pred.shape[1], step_pred.shape[2]
    
    for i in range(min(4, step_pred.shape[0])):
        var_name = var_names[i]
        data_slice = step_pred[i].copy() 
        mask_slice = extract_mask(mask_numpy, step_idx, i, H, W)
        if mask_slice is not None and mask_slice.shape == (H, W):
            data_slice[mask_slice < 0.5] = np.nan
        
        valid_data = data_slice[np.isfinite(data_slice)]
        vmin, vmax = (float(np.min(valid_data)), float(np.max(valid_data))) if valid_data.size > 0 else (-1, 1)

        row, col = (i // 2) + 1, (i % 2) + 1
        cmap_name = "Viridis" if var_name.upper() in ["SSS", "盐度"] else "RdBu_r"

        fig.add_trace(create_heatmap(data_slice, vmin, vmax, cmap_name, row, col, f"x{i+1}" if i>0 else "x"), row=row, col=col)
        x_axis_id = f"x{i+1}" if i>0 else "x"
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain", row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed", scaleanchor=x_axis_id, scaleratio=1, constrain="domain", row=row, col=col)

    fig.update_layout(height=750, width=1000, paper_bgcolor='white', plot_bgcolor='#e2e6ea', margin=dict(l=20, r=20, t=50, b=20))
    return fig

def draw_curve_plot(pred_numpy, mask_numpy, var_names):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{v} 预测期海区均值变化" for v in var_names[:4]],
                        horizontal_spacing=0.1, vertical_spacing=0.15)
    num_steps, _, H, W = pred_numpy.shape
    x_axis = np.arange(1, num_steps + 1) * 6
    
    for i in range(min(4, pred_numpy.shape[1])):
        var_name = var_names[i]
        mean_vals = []
        for t in range(num_steps):
            data_slice = pred_numpy[t, i]
            mask_slice = extract_mask(mask_numpy, t, i, H, W)
            valid_data = data_slice[mask_slice >= 0.5] if (mask_slice is not None and mask_slice.shape == (H, W)) else data_slice[~np.isnan(data_slice)]
            mean_vals.append(np.mean(valid_data) if len(valid_data) > 0 else np.nan)
            
        row, col = (i // 2) + 1, (i % 2) + 1
        fig.add_trace(go.Scatter(x=x_axis, y=mean_vals, mode='lines+markers', name=var_name, showlegend=False), row=row, col=col)
        fig.update_xaxes(title_text="预报时间 (Hours)", showgrid=True, gridcolor='lightgray', row=row, col=col)
        fig.update_yaxes(title_text=f"平均 {var_name}", showgrid=True, gridcolor='lightgray', row=row, col=col)
        
    fig.update_layout(height=600, paper_bgcolor='white', plot_bgcolor='white')
    return fig

def draw_rmse_plot(pred_numpy, true_numpy, mask_numpy, var_names, step_idx):
    if true_numpy is None: return go.Figure()
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{v.upper()} RMSE (Pred vs True)" for v in var_names[:4]],
                        vertical_spacing=0.08, horizontal_spacing=0.08)
    # limit step to match true size
    step_true = min(step_idx, true_numpy.shape[0]-1)
    p_step = pred_numpy[step_idx]
    t_step = true_numpy[step_true]
    H, W = p_step.shape[1], p_step.shape[2]
    
    for i in range(min(4, p_step.shape[0])):
        diff = np.abs(p_step[i] - t_step[i])
        data_slice = diff.copy()
        mask_slice = extract_mask(mask_numpy, step_idx, i, H, W)
        if mask_slice is not None and mask_slice.shape == (H, W):
            data_slice[mask_slice < 0.5] = np.nan
        row, col = (i // 2) + 1, (i % 2) + 1
        valid_data = data_slice[np.isfinite(data_slice)]
        vmax = float(np.max(valid_data)) if valid_data.size > 0 else 1
        fig.add_trace(create_heatmap(data_slice, 0, vmax, "Reds", row, col, f"x{i+1}" if i>0 else "x"), row=row, col=col)
        x_axis_id = f"x{i+1}" if i>0 else "x"
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain", row=row, col=col)
        fig.update_yaxes(autorange="reversed", scaleanchor=x_axis_id, scaleratio=1, showticklabels=False, showgrid=False, zeroline=False, constrain="domain", row=row, col=col)
    fig.update_layout(height=750, width=1000, paper_bgcolor='white', plot_bgcolor='#e2e6ea', margin=dict(l=20, r=20, t=50, b=20))
    return fig

def draw_single_point_curve(pred_numpy, true_numpy, var_names, x_idx, y_idx):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{v} @({x_idx},{y_idx}) 时序变化" for v in var_names[:4]],
                        horizontal_spacing=0.1, vertical_spacing=0.15)
    num_steps = pred_numpy.shape[0]
    x_axis = np.arange(1, num_steps + 1) * 6
    
    for i in range(min(4, pred_numpy.shape[1])):
        row, col = (i // 2) + 1, (i % 2) + 1
        vals = pred_numpy[:, i, y_idx, x_idx]
        fig.add_trace(go.Scatter(x=x_axis, y=vals, mode='lines+markers', name="预测", showlegend=(i==0)), row=row, col=col)
        
        if true_numpy is not None:
            t_steps = true_numpy.shape[0]
            t_vals = true_numpy[:, i, y_idx, x_idx]
            fig.add_trace(go.Scatter(x=np.arange(1, t_steps + 1) * 6, y=t_vals, mode='lines', line=dict(dash='dash'), name="真实观测", showlegend=(i==0)), row=row, col=col)
            
        fig.update_xaxes(title_text="预报时间 (Hours)", showgrid=True, gridcolor='lightgray', row=row, col=col)
        fig.update_yaxes(title_text=f"要素值", showgrid=True, gridcolor='lightgray', row=row, col=col)
    fig.update_layout(height=600, paper_bgcolor='white', plot_bgcolor='white')
    return fig
