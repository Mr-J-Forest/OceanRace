import os
import sys
import torch
import numpy as np
import pandas as pd
import xarray as xr

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: pip install gradio>=3.0")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Please install plotly for frontend rendering: pip install plotly")
    sys.exit(1)

# 将 src 目录加入模块搜索路径，确保可以导入相关模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from element_forecasting.predictor import ElementForecastPredictor
from element_forecasting.dataset import ElementForecastWindowDataset

def load_dataset_info(data_path):
    """
    加载 .nc 文件，提取总有效切片数，更新 Slider 范围并展示首尾真实时间供用户参考，避免渲染几万个选项导致浏览器卡死。
    """
    data_path = data_path.strip('\"\'')
    if not os.path.exists(data_path):
        return gr.update(interactive=False), "错误：未找到数据文件"
        
    try:
        norm_path = "data/processed/normalization/element_forecasting_norm.json"
        dataset = ElementForecastWindowDataset(
            data_file=data_path,
            input_steps=24,
            output_steps=72,
            split=None,
            norm_stats_path=norm_path
        )
        
        if len(dataset) == 0:
            return gr.update(interactive=False), "错误：数据集为空或步长不足"
            
        ds = xr.open_dataset(data_path)
        times = pd.to_datetime(ds['time'].values)
        
        first_time = times[dataset._windows[0]].strftime("%Y-%m-%d %H:%M:%S")
        last_time = times[dataset._windows[-1]].strftime("%Y-%m-%d %H:%M:%S")
        
        info = f"共提取到 {len(dataset)} 个可用预测窗口(Index 0 ~ {len(dataset)-1})。\n切片包含真实起点范围:\n第一步: {first_time}\n最后一步: {last_time}"
        
        # 将滑动条极值更新到位
        return gr.update(maximum=len(dataset)-1, value=0, interactive=True), info
    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update(interactive=False), f"解析失败: {str(e)}"

def extract_mask(mask_numpy, t_idx, c_idx, H, W):
    if mask_numpy is None:
        return None
    # 判断掩码矩阵形状以兼容各种情况 (Static, T_only, C_only, or full T-C-H-W)
    if mask_numpy.shape == (H, W):
        return mask_numpy
    elif mask_numpy.ndim == 3:
        # (X, H, W) 这里的 X 可能是 T 也可能是 C. 因为我们有4个通道
        # 若为 (4, H, W)，就当做各通道掩码
        if mask_numpy.shape[0] == 4:
            return mask_numpy[min(c_idx, 3)]
        else:
            return mask_numpy[min(t_idx, mask_numpy.shape[0] - 1)]
    elif mask_numpy.ndim == 4:
        # (T, C, H, W)
        t_safe = min(t_idx, mask_numpy.shape[0] - 1)
        c_safe = min(c_idx, mask_numpy.shape[1] - 1)
        return mask_numpy[t_safe, c_safe]
    return mask_numpy # 备用兜底

def draw_spatial_plot(pred_numpy, mask_numpy, var_names, step_idx):
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=[f"{v.upper()} (+{(step_idx+1)*6}H)" for v in var_names[:4]],
        vertical_spacing=0.08, horizontal_spacing=0.08
    )
    
    # 获取指定步长的数据: [variable, H, W]
    step_pred = pred_numpy[step_idx]
    H, W = step_pred.shape[1], step_pred.shape[2]
        
    for i in range(min(4, step_pred.shape[0])):
        var_name = var_names[i]
        data_slice = step_pred[i].copy() 
        
        # 兼容性提取掩码片段
        mask_slice = extract_mask(mask_numpy, step_idx, i, H, W)
        if mask_slice is not None and mask_slice.shape == (H, W):
            data_slice[mask_slice < 0.5] = np.nan
        
        var_upper = var_name.upper()
        if var_upper in ["SSS", "盐度"]:
            cmap_name = "Viridis"
        else:
            cmap_name = "RdBu_r"
            
        valid_data = data_slice[np.isfinite(data_slice)]
        if valid_data.size > 0:
            vmin = float(np.min(valid_data))
            vmax = float(np.max(valid_data))
        else:
            vmin, vmax = -1.0, 1.0

        row = (i // 2) + 1
        col = (i % 2) + 1

        hm = go.Heatmap(
            z=data_slice,
            colorscale=cmap_name,
            zmin=vmin,
            zmax=vmax,
            showscale=True,
            zsmooth="best",  # 开启双线性平滑插值，消除马赛克方块感
            colorbar=dict(
                thickness=12, 
                len=0.45, 
                y=0.79 if row==1 else 0.21, 
                x=0.455 if col==1 else 1.0,
                outlinewidth=0,
                tickfont=dict(size=11),
                title=""
            ),
            hoverinfo="z+x+y" 
        )
        fig.add_trace(hm, row=row, col=col)
        
        # 隐藏坐标轴并紧凑贴合画面 Domain
        x_axis_id = f"x{i+1}" if i>0 else "x"
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain", row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, 
                         autorange="reversed", scaleanchor=x_axis_id, scaleratio=1, constrain="domain", row=row, col=col)

    fig.update_layout(
        height=750, 
        width=1000,
        paper_bgcolor='white', 
        plot_bgcolor='#e2e6ea', # 柔和淡灰蓝作为陆地/无数据区域背景
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def draw_curve_plot(pred_numpy, mask_numpy, var_names):
    # 绘制随时间变化的区域平均曲线
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=[f"{v} 预测期海区均值变化" for v in var_names[:4]],
        horizontal_spacing=0.1, vertical_spacing=0.15
    )
    
    num_steps = pred_numpy.shape[0]
    x_axis = np.arange(1, num_steps + 1) * 6  # 转换为小时
    H, W = pred_numpy.shape[2], pred_numpy.shape[3]
    
    for i in range(min(4, pred_numpy.shape[1])):
        var_name = var_names[i]
        
        mean_vals = []
        for t in range(num_steps):
            data_slice = pred_numpy[t, i]
            
            mask_slice = extract_mask(mask_numpy, t, i, H, W)
            if mask_slice is not None and mask_slice.shape == (H, W):
                valid_data = data_slice[mask_slice >= 0.5]
            else:
                valid_data = data_slice[~np.isnan(data_slice)]
                
            mean_vals.append(np.mean(valid_data) if len(valid_data) > 0 else np.nan)
            
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        sc = go.Scatter(
            x=x_axis,
            y=mean_vals,
            mode='lines+markers',
            line=dict(width=2, color='royalblue'),
            marker=dict(size=6),
            name=var_name,
            showlegend=False
        )
        fig.add_trace(sc, row=row, col=col)
        
        fig.update_xaxes(title_text="预报时间 (Hours)", showgrid=True, gridcolor='lightgray', row=row, col=col)
        fig.update_yaxes(title_text=f"平均 {var_name}", showgrid=True, gridcolor='lightgray', row=row, col=col)
        
    fig.update_layout(height=600, paper_bgcolor='white', plot_bgcolor='white')
    return fig

def element_forecasting_logic(model_path_from_ui, data_path, start_idx):
    """
    执行推理的核心逻辑，计算所有步长，并返回状态（以供滑动条切换）以及默认初始视图。
    """
    data_path = data_path.strip('\"\'')
    model_path = "models/forecast_model.pt"
    
    empty_fig = go.Figure()
    if not os.path.exists(model_path):
        return None, f"Error: 模型路径 {model_path} 不存在", empty_fig, empty_fig
    if not os.path.exists(data_path):
        return None, f"Error: 数据路径 {data_path} 不存在", empty_fig, empty_fig
    
    try:
        norm_path = "data/processed/normalization/element_forecasting_norm.json"
        dataset = ElementForecastWindowDataset(
            data_file=data_path, 
            input_steps=24, 
            output_steps=72, 
            split=None,
            norm_stats_path=norm_path
        )

        idx = int(start_idx)
        if idx < 0 or idx >= len(dataset):
            return None, f"Error: 起始步超出范围", empty_fig, empty_fig

        sample = dataset[idx]
        x_tensor = sample["x"].unsqueeze(0)  

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
        pred_numpy = result["pred"][0].numpy()  # shape: (output_steps, Channels, H, W)
        var_names = result.get("var_names", ["SST", "SSS", "SSU", "SSV"])

        valid_mask_tensor = sample.get("y_valid", None)
        mask_numpy = valid_mask_tensor.numpy() if valid_mask_tensor is not None else None

        state_dict = {
            "pred": pred_numpy,
            "mask": mask_numpy,
            "vars": var_names
        }
        
        # 初始视图，默认绘制第 1 步 (step_idx = 0)
        spatial_fig = draw_spatial_plot(pred_numpy, mask_numpy, var_names, 0)
        # 绘制整个预报期的 72 小时变化曲线
        curve_fig = draw_curve_plot(pred_numpy, mask_numpy, var_names)
        
        msg = f"预测成功！生成了未来 {pred_numpy.shape[0]} 步的多变量预报结果。"
        return state_dict, msg, spatial_fig, curve_fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", go.Figure(), go.Figure()

def update_spatial_plot(state_dict, step_val):
    if not state_dict:
        return go.Figure()
    
    pred_numpy = state_dict["pred"]
    mask_numpy = state_dict["mask"]
    var_names = state_dict["vars"]
    
    step_idx = int(step_val) - 1 # Slider 从 1 开始，数组索引从 0 开始
    return draw_spatial_plot(pred_numpy, mask_numpy, var_names, step_idx)

def create_gui():
    with gr.Blocks(title="OceanRace 智能海洋分析系统 UI") as app:
        gr.Markdown("# 🌊 OceanRace 面向海洋环境智能分析系统")
        gr.Markdown("包含三大核心模块：海洋要素短临预报、海洋中尺度涡旋检测、极端异常事件检测")
        
        with gr.Tab("要素预测 (Element Forecasting)"):
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=1):
                        model_input = gr.Textbox(value="models/forecast_model.pt", label="模型路径 (Checkpoint)", interactive=False)
                        with gr.Row():
                            data_input = gr.Textbox(value="data/processed/element_forecasting/示例数据.nc", label="测试输入数据序列路径 (.nc)")
                            load_btn = gr.Button("加载数据信息", size="sm")
                        
                        time_idx_input = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="拉动选择时间序列起点 (Index)", interactive=False)
                        dataset_info = gr.Textbox(label="起点真实时间跨度参考", interactive=False, lines=3)
                        predict_btn = gr.Button("生成预测", variant="primary")
                        status_output = gr.Textbox(label="运行状态", interactive=False)
                        
                    with gr.Column(scale=2):
                        # 添加隐藏状态保存推理输出的数值
                        prediction_state = gr.State()
                        
                        with gr.Tabs():
                            with gr.Tab("空间分布预报 (多要素 2D 场)"):
                                step_slider = gr.Slider(minimum=1, maximum=12, step=1, value=1, label="滑动切换预测步长 (1步 = 6小时, 最大12步即72小时)", interactive=True)
                                plot_output = gr.Plot(label="要素场预测可视化")
                            with gr.Tab("区域平均趋势 (72小时变化线)"):
                                curve_plot = gr.Plot(label="海区平均物理量变化趋势")
            
            # 点击"加载时间供选"，由更新逻辑读取 NC 文件并重绘选项
            load_btn.click(
                fn=load_dataset_info,
                inputs=[data_input],
                outputs=[time_idx_input, dataset_info]
            )
            
            # 主生成逻辑，产生 state 和第一张图及趋势图
            predict_btn.click(
                fn=element_forecasting_logic,
                inputs=[model_input, data_input, time_idx_input],
                outputs=[prediction_state, status_output, plot_output, curve_plot]
            )
            
            # 当滑动条的值改变时，免运行推理，直接使用现有的 state 重新进行绘制返回
            step_slider.change(
                fn=update_spatial_plot,
                inputs=[prediction_state, step_slider],
                outputs=[plot_output]
            )
            
        with gr.Tab("涡旋检测 (Eddy Detection)"):
            gr.Markdown("### 🌀 涡旋识别功能建设中 ... 该界面保留为入口空位")
            gr.Textbox(value="待接入 Eddy 分支逻辑...", label="备用占位框", interactive=False)
            gr.Button("运行涡旋检测 (暂不支持)")
            
        with gr.Tab("异常检测 (Anomaly Detection)"):
            gr.Markdown("### ⚠️ 极端异常事件检测功能建设中 ... 该界面保留为入口空位")
            gr.Textbox(value="待接入 Anomaly 分支逻辑...", label="备用占位框", interactive=False)
            gr.Button("运行异常检测 (暂不支持)")
            
    return app

if __name__ == "__main__":
    app = create_gui()
    # server_name 设为 0.0.0.0 支持网络访问或 Docker 映射
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
