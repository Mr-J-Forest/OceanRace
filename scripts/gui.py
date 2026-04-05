import os
import sys

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.gradio_ui.app import create_gui

if __name__ == "__main__":
    app = create_gui()
    # server_name 设为 0.0.0.0 支持网络访问或 Docker 映射
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
