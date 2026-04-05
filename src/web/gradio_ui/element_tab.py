import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import gradio as gr
from plotly import graph_objects as go
from src.element_forecasting.predictor import ElementForecastPredictor
from src.element_forecasting.dataset import ElementForecastWindowDataset
from src.web.gradio_ui.utils import draw_spatial_plot, draw_curve_plot, draw_rmse_plot, draw_single_point_curve

def load_dataset_info(data_path):
    data_path = data_path.strip('\"\'')
    if not os.path.exists(data_path):
        return gr.update(interactive=False), "错误：未找到数据文件"
    try:
        norm_path = "data/processed/normalization/element_forecasting_norm.json"
        dataset = ElementForecastWindowDataset(data_file=data_path, input_steps=24, output_steps=72, split=None, norm_stats_path=norm_path)
        if len(dataset) == 0:
            return gr.update(interactive=False), "错误：数据集为空或步长不足"
            
        ds = xr.open_dataset(data_path)
        times = pd.to_datetime(ds['time'].values)
        
        first_time = times[dataset._windows[0]].strftime("%Y-%m-%d %H:%M:%S")
        last_time = times[dataset._windows[-1]].strftime("%Y-%m-%d %H:%M:%S")
        
        info = f"共提取到 {len(dataset)} 个可用预测窗口。\n第一步: {first_time}\n最后一步: {last_time}"
        return gr.update(maximum=len(dataset)-1, value=0, interactive=True), info
    except Exception as e:
        return gr.update(interactive=False), f"解析失败: {str(e)}"

def element_forecasting_logic(model_path_from_ui, data_path, start_idx):
    data_path = data_path.strip('\"\'')
    model_path = model_path_from_ui.strip('\"\'')
    
    if not os.path.exists(model_path): return None, f"Error: 模型 {model_path} 不存在", go.Figure(), go.Figure()
    if not os.path.exists(data_path): return None, f"Error: 数据 {data_path} 不存在", go.Figure(), go.Figure()
    
    try:
        norm_path = "data/processed/normalization/element_forecasting_norm.json"
        dataset = ElementForecastWindowDataset(data_file=data_path, input_steps=24, output_steps=72, split=None, norm_stats_path=norm_path)
        idx = int(start_idx)
        if idx < 0 or idx >= len(dataset): return None, "Error: 起始步超出范围", go.Figure(), go.Figure()

        sample = dataset[idx]
        x_tensor = sample["x"].unsqueeze(0)  
        y_tensor = sample["y"].numpy()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = ElementForecastPredictor(checkpoint_path=model_path, device=device, norm_stats_path=norm_path)

        result = predictor.predict_long_horizon(x=x_tensor, target_steps=72, overlap_steps=4, enable_overlap_blend=True, denormalize=True, return_cpu=True)
        pred_numpy = result["pred"][0].numpy()
        var_names = result.get("var_names", ["SST", "SSS", "SSU", "SSV"])

        valid_mask_tensor = sample.get("y_valid", None)
        mask_numpy = valid_mask_tensor.numpy() if valid_mask_tensor is not None else None

        state_dict = {
            "pred": pred_numpy,
            "true": y_tensor,  # Assuming denormalization is needed but mock for now or use original shape
            "mask": mask_numpy,
            "vars": var_names
        }
        
        # 初始视图
        spatial_fig = draw_spatial_plot(pred_numpy, mask_numpy, var_names, 0)
        curve_fig = draw_curve_plot(pred_numpy, mask_numpy, var_names)
        
        return state_dict, f"预报成功（{pred_numpy.shape[0]}步长）", spatial_fig, curve_fig
        
    except Exception as e:
        return None, f"Error: {str(e)}", go.Figure(), go.Figure()

def update_visuals(state_dict, step_val, pt_x, pt_y, thresh_val):
    if not state_dict:
        fig = go.Figure()
        return fig, fig, fig, "暂无数据"
    
    pred_numpy, true_numpy = state_dict["pred"], state_dict.get("true", None)
    mask_numpy, var_names = state_dict["mask"], state_dict["vars"]
    
    step_idx = int(step_val) - 1
    pt_x, pt_y = int(pt_x), int(pt_y)
    
    sp_fig = draw_spatial_plot(pred_numpy, mask_numpy, var_names, step_idx)
    rmse_fig = draw_rmse_plot(pred_numpy, true_numpy, mask_numpy, var_names, step_idx)
    
    # Validation bounds for point
    H, W = pred_numpy.shape[2], pred_numpy.shape[3]
    pt_y = min(max(pt_y, 0), H-1)
    pt_x = min(max(pt_x, 0), W-1)
    
    single_pt_fig = draw_single_point_curve(pred_numpy, true_numpy, var_names, pt_x, pt_y)
    
    # Alarm Logic: Check if any step exceeds thresh_val
    alarms = []
    for step in range(pred_numpy.shape[0]):
        max_val = np.max(pred_numpy[step])
        if max_val > thresh_val:
            alarms.append(f"步长 {step+1} 存在区域超过警戒阈值 ({max_val:.2f} > {thresh_val})")
            
    alarm_msg = "\n".join(alarms[:5])
    if not alarm_msg:
        alarm_msg = f"未检测到超出阈值 {thresh_val} 的异常区域。"
    else:
        if len(alarms) > 5:
            alarm_msg += f"\n...及其他 {len(alarms)-5} 个时间步异常"
            
    return sp_fig, rmse_fig, single_pt_fig, alarm_msg

def create_element_forecasting_tab():
    with gr.Tab("要素时序预测与分析"):
        gr.Markdown("### 🌊 核心要素精细化预报面板\n包含：多要素时序预测、单点曲线分析、真值对比精度验证与动态阈值告警功能。")
        with gr.Row():
            with gr.Column(scale=1):
                model_input = gr.Textbox(value="models/forecast_model.pt", label="模型权重路径")
                data_input = gr.Textbox(value="data/processed/element_forecasting/示例数据.nc", label="测试数据集 (.nc)")
                load_btn = gr.Button("解析数据集", size="sm")
                
                time_idx_input = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="选择起始时间窗截点 (Index)", interactive=False)
                dataset_info = gr.Textbox(label="数据基础信息", interactive=False, lines=2)
                predict_btn = gr.Button("生成全流程预测", variant="primary")
                status_output = gr.Textbox(label="执行状态", interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 📍 精细化单点分析与告警设置")
                point_x = gr.Number(value=10, label="关注点 X 坐标 (经度方向)")
                point_y = gr.Number(value=10, label="关注点 Y 坐标 (纬度方向)")
                alarm_thresh = gr.Number(value=28.0, label="SST 异常报警阈值 (°C)")
                update_btn = gr.Button("更新分析视图", size="sm")
                alarm_output = gr.Textbox(label="智能告警系统", lines=4, interactive=False)
                
            with gr.Column(scale=3):
                prediction_state = gr.State()
                
                with gr.Tabs():
                    with gr.Tab("总体空间预报 (2D色斑图)"):
                        step_slider = gr.Slider(minimum=1, maximum=12, step=1, value=1, label="预测时长切换 (1步=6小时)")
                        plot_output = gr.Plot(label="预报场")
                    with gr.Tab("单点预报演变曲线"):
                        single_pt_plot = gr.Plot(label="指定坐标多要素变化")
                    with gr.Tab("真值对比与精度量化(RMSE)"):
                        rmse_plot = gr.Plot(label="误差分布热力图")
                    with gr.Tab("海区总均值趋势"):
                        curve_plot = gr.Plot(label="要素均方趋势演化")

        load_btn.click(fn=load_dataset_info, inputs=[data_input], outputs=[time_idx_input, dataset_info])
        predict_btn.click(fn=element_forecasting_logic, inputs=[model_input, data_input, time_idx_input], 
                          outputs=[prediction_state, status_output, plot_output, curve_plot])
        
        # 联动更新所有高级视图
        inputs_to_watch = [prediction_state, step_slider, point_x, point_y, alarm_thresh]
        outputs_to_watch = [plot_output, rmse_plot, single_pt_plot, alarm_output]
        
        update_btn.click(fn=update_visuals, inputs=inputs_to_watch, outputs=outputs_to_watch)
        step_slider.change(fn=update_visuals, inputs=inputs_to_watch, outputs=outputs_to_watch)

