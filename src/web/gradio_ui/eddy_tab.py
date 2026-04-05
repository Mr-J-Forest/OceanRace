import gradio as gr

def create_eddy_tab():
    with gr.Tab("涡旋检测 (Eddy Detection)"):
        gr.Markdown("### 🌀 涡旋识别功能 (规划中)")
        gr.Markdown("- 批量自动化识别\n- 涡旋生命周期追踪\n- 人工交互修正结果\n- 历史与多维时空统计关联。")
        gr.Textbox(value="待接入 Eddy 核心算法层 API...", label="系统状态", interactive=False)
        gr.Button("运行涡旋检测 (暂不支持)")
