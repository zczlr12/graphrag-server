"""前端用户界面."""
import json
import os
import time

import gradio as gr
import requests

from router import auto_select

API_BASE = "http://localhost:20213"
QUERY_TYPES = ["auto", "local", "global", "direct"]
QUERY_TYPES_CN = ["自动", "局部", "全局", "无"]
INDEX_FOLDERS = sorted([f for f in os.listdir("output") if os.path.isdir(f"output/{f}")])


def send_message(temperature, timestamp, index, community_level, response_type, stream, history):
    try:
        model = QUERY_TYPES[index]
        if not index:
            model = auto_select(history[-1][0])
        messages = [
            {
                "role": "system",
                "content": "你叫电厂百晓生，你将会学习核电厂经验反馈，工作负责人会对你进行提问，你需要给出针对性建议。"
            }
        ]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            json={
                "messages": messages,
                "model": f"{timestamp}-{model}",
                "temperature": temperature,
                "community_level": community_level,
                "response_type": response_type,
                "stream": stream
            },
            stream=stream
        )
        history[-1][-1] = f"***数据库：{timestamp}，搜索方法：{QUERY_TYPES_CN[QUERY_TYPES.index(model)]}，社区层级：{community_level}。***\n"
        if stream:
            for chunk in response.iter_content(None):
                choice = json.loads(chunk)["choices"][0]
                if choice["finish_reason"]:
                    history[-1][-1] = f"***数据库：{timestamp}，搜索方法：{QUERY_TYPES_CN[QUERY_TYPES.index(model)]}，社区层级：{community_level}。***\n"
                history[-1][-1] += choice["delta"]["content"]
                time.sleep(0.02)
                yield history
        else:
            history[-1][-1] += response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        history[-1][-1] += f"请求失败: {e}"
    yield history


def create_interface():
    with gr.Blocks(gr.themes.Base(), title="电厂百晓生", fill_height=True, fill_width=True) as demo:
        gr.Markdown("# <center>电厂百晓生</center>")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## ⚙ 设置")
                temperature = gr.Slider(0, 2, 0, step=0.1, label="模型温度")
                timestamp = gr.Dropdown(INDEX_FOLDERS, value=INDEX_FOLDERS[-1], label="数据库")
                query_type = gr.Radio(QUERY_TYPES_CN, value="自动", type="index", label="搜索方法")
                community_level = gr.Slider(0, 3, 2, step=1, label="社区层级")
                response_type = gr.Textbox(label="响应类型", placeholder="例如：“一句话”、“几个要点”、“多页报告”等")
                stream = gr.Checkbox(True, label="流式输出")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话框", layout="panel", show_copy_button=True, placeholder="请在下方输入问题")
                query_input = gr.Textbox(placeholder="输入问题", label="输入框", show_copy_button=True)
                with gr.Row():
                    with gr.Column():
                        query_btn = gr.Button("发送", variant="primary")
                    with gr.Column():
                        gr.ClearButton(chatbot, value="清空聊天记录")
        
        query_input.submit(
            lambda x, y: ("", [*y, [x, None]]),
            [query_input, chatbot],
            [query_input, chatbot]
        ).then(
            send_message,
            [
                temperature,
                timestamp,
                query_type,
                community_level,
                response_type,
                stream,
                chatbot
            ],
            chatbot
        )
        query_btn.click(
            lambda x, y: ("", [*y, [x, None]]),
            [query_input, chatbot],
            [query_input, chatbot]
        ).then(
            send_message,
            [
                temperature,
                timestamp,
                query_type,
                community_level,
                response_type,
                stream,
                chatbot
            ],
            chatbot
        )
    
    return demo


async def main():
    create_interface().launch()


if __name__ == "__main__":
    create_interface().launch()
