"""前端用户界面."""
import os

import gradio as gr


def create_interface():
    with gr.Blocks(title="电厂百小生", fill_height=True, fill_width=True) as demo:
        gr.Markdown("# <center>电厂百小生</center>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙ 设置")
                temperature = gr.Slider(0, 2, 0, step=0.1, label="模型温度")
                timestamp = gr.Dropdown(
                    [f for f in os.listdir("output") if os.path.isdir(f"output/{f}")],
                    label="时间戳"
                )
                query_type = gr.Radio(["自动", "全局搜索", "局部搜索", "不查询"], value="自动", type="index", label="查询方式")
                community_level = gr.Slider(0, 3, 2, step=1, label="社区层级")
                response_type = gr.Textbox("多个段落", label="响应类型", placeholder="例如：“一句话”、“几个要点”、“多页报告”等")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话框", placeholder="请在下方输入问题")
                query_input = gr.Textbox(placeholder="输入问题", container=False, show_copy_button=True)
                query_btn = gr.Button("发送", min_width=10)
            with gr.Column(scale=2):
                gr.HTML("请在对话框中点击对应的引用")
    return demo


async def main():
    create_interface().launch()


if __name__ == "__main__":
    create_interface().launch()
