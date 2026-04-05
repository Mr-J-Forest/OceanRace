import gradio as gr

def create_anomaly_tab():
    with gr.Tab("异常检测 (Anomaly Detection)"):
        gr.Markdown("### ⚠️ 极端异常事件与灾害预警 (规划中)")
        gr.Markdown("- 异常信号智能检测\n- 历史台风路径融合联动\n- 四级红黄蓝风险划定与弹窗推送。")
        gr.Textbox(value="待接入 Anomaly 核心算法层与台风库...", label="系统状态", interactive=False)
        gr.Button("运行异常检测 (暂不支持)")
