import sys
import os

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: pip install gradio>=3.0")
    sys.exit(1)

from src.web.gradio_ui.element_tab import create_element_forecasting_tab
from src.web.gradio_ui.eddy_tab import create_eddy_tab
from src.web.gradio_ui.anomaly_tab import create_anomaly_tab

def create_gui():
    with gr.Blocks(title="OceanRace 智能海洋分析系统 UI") as app:
        gr.Markdown("# 🌊 OceanRace 面向海洋环境智能分析系统")
        gr.Markdown("包含三大核心模块：海洋要素短临预报、海洋中尺度涡旋检测、极端异常事件检测")
        
        create_element_forecasting_tab()
        create_eddy_tab()
        create_anomaly_tab()
            
    return app
