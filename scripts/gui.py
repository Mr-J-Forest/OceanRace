import os
import sys
import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: pip install gradio>=3.0")
    sys.exit(1)

# 将 src 目录加入模块搜索路径，确保可以导入相关模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from element_forecasting.predictor import ElementForecastPredictor
from element_forecasting.dataset import ElementForecastWindowDataset
from utils.visualization_defaults import apply_matplotlib_defaults

def load_dataset_info(data_path):
    """
    加载 .nc 文件，提取总有效切片数，更新 Slider 范围并展示首尾真实时间供用户参考，避免渲染几万个选项导致浏览器卡死。
    """
    if not os.path.exists(data_path):
        return gr.update(interactive=False), "错误：未找到数据文件"
        
    try:
        dataset = ElementForecastWindowDataset(
            data_file=data_path,
            input_steps=12,
            output_steps=12,
            split=None
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

def draw_spatial_plot(pred_numpy, mask_numpy, var_names, step_idx):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor('white') # 确保背景是干净的纯白
    axs = axs.flatten()
    
    # 获取指定步长的数据: [variable, H, W]
    step_pred = pred_numpy[step_idx]
    if mask_numpy is not None:
        step_mask = mask_numpy[step_idx]
    else:
        step_mask = None
        
    for i in range(min(4, step_pred.shape[0])):
        ax = axs[i]
        var_name = var_names[i]
        data_slice = step_pred[i].copy() # 拷贝以便修改
        
        # 如果存在掩码，将被掩码屏蔽的陆地/缺测点设为 NaN
        if step_mask is not None:
            mask_slice = step_mask[i]
            data_slice[mask_slice < 0.5] = np.nan
        
        if var_name in ["SST", "海温"]:
            cmap_name = "coolwarm"
        elif var_name in ["SSS", "盐度"]:
            cmap_name = "viridis"
        else:
            cmap_name = "RdBu_r"
            
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color='#dddddd')
        
        valid_data = data_slice[~np.isnan(data_slice)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [1, 99])
        else:
            vmin, vmax = None, None

        im = ax.imshow(
            data_slice, 
            cmap=cmap, 
            origin="lower", 
            interpolation="bilinear", 
            vmin=vmin, 
            vmax=vmax
        )
        
        # 将步长转化为大约的小时数 (假设 dt=6h，预报12步即 72h)
        ax.set_title(f"{var_name} (预测步: {step_idx+1}, +{(step_idx+1)*6}H)", pad=10)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def draw_curve_plot(pred_numpy, mask_numpy, var_names):
    # 绘制随时间变化的区域平均曲线
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    axs = axs.flatten()
    
    num_steps = pred_numpy.shape[0]
    x_axis = np.arange(1, num_steps + 1) * 6  # 转换为小时
    
    for i in range(min(4, pred_numpy.shape[1])):
        ax = axs[i]
        var_name = var_names[i]
        
        mean_vals = []
        for t in range(num_steps):
            data_slice = pred_numpy[t, i]
            if mask_numpy is not None:
                mask_slice = mask_numpy[t, i]
                valid_data = data_slice[mask_slice >= 0.5]
            else:
                valid_data = data_slice[~np.isnan(data_slice)]
                
            mean_vals.append(np.mean(valid_data) if len(valid_data) > 0 else np.nan)
            
        ax.plot(x_axis, mean_vals, marker='o', linestyle='-', linewidth=2, color='tab:blue')
        ax.set_title(f"{var_name} 预测期海区均值变化", pad=10)
        ax.set_xlabel("预报时间 (Hours)")
        ax.set_ylabel(f"平均 {var_name}")
        ax.grid(True, linestyle="--", alpha=0.6)
        
    plt.tight_layout()
    return fig

def element_forecasting_logic(model_path, data_path, start_idx):
    """
    执行推理的核心逻辑，计算所有步长，并返回状态（以供滑动条切换）以及默认初始视图。
    """
    empty_fig = plt.figure()
    if not os.path.exists(model_path):
        return None, f"Error: 模型路径 {model_path} 不存在", empty_fig, empty_fig
    if not os.path.exists(data_path):
        return None, f"Error: 数据路径 {data_path} 不存在", empty_fig, empty_fig
    
    try:
        apply_matplotlib_defaults()
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        dataset = ElementForecastWindowDataset(
            data_file=data_path, input_steps=12, output_steps=12, split=None
        )

        idx = int(start_idx)
        if idx < 0 or idx >= len(dataset):
            return None, f"Error: 起始步超出范围", empty_fig, empty_fig

        sample = dataset[idx]
        x_tensor = sample["x"].unsqueeze(0)  

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = ElementForecastPredictor(checkpoint_path=model_path, device=device)

        result = predictor.predict(x_tensor, denormalize=True, return_cpu=True)
        pred_numpy = result["pred"][0].numpy()  # shape: (output_steps, Channels, H, W)
        var_names = result.get("var_names", ["SST", "SSS", "SSU", "SSV"])

        valid_mask_tensor = sample.get("y_valid", None)
        mask_numpy = valid_mask_tensor[0].numpy() if valid_mask_tensor is not None else None

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
        return None, f"Error: {str(e)}", empty_fig, empty_fig

def update_spatial_plot(state_dict, step_val):
    if not state_dict:
        return plt.figure()
    
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
                        model_input = gr.Textbox(value="models/forecast_model.pt", label="模型路径 (Checkpoint)")
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
