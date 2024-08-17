"""前端用户界面."""
import os
import requests
import gradio as gr

API_BASE = "http://localhost:20213"
QUERY_TYPES = ["auto", "global", "local", "direct"]


def send_message(temperature, timestamp, index, community_level, response_type, query, history):
    try:
        messages = [
            {
                "role": "system",
                "content": "你将会学习核电厂经验反馈，工作负责人会对你进行提问，你需要给出针对性建议。"
            }
        ]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": query})
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            json={
                "messages": messages,
                "model": f"{timestamp}-{QUERY_TYPES[index]}",
                "temperature": temperature,
                "community_level": community_level,
                "response_type": response_type
            }
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        history.append((query, result))
    except requests.RequestException as e:
        history.append((query, f"请求失败：{e}"))
    return gr.update(), history


def create_interface():
    with gr.Blocks(title="电厂百晓生", fill_height=True, fill_width=True) as demo:
        gr.Markdown("# <center>电厂百晓生</center>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙ 设置")
                temperature = gr.Slider(0, 2, 0, step=0.1, label="模型温度")
                timestamp = gr.Dropdown(
                    [f for f in os.listdir("output") if os.path.isdir(f"output/{f}")],
                    label="时间戳"
                )
                query_type = gr.Radio(["自动", "全局搜索", "局部搜索", "不查询"], value="局部搜索", type="index", label="查询方式")
                community_level = gr.Slider(0, 3, 2, step=1, label="社区层级")
                response_type = gr.Textbox(label="响应类型", placeholder="例如：“一句话”、“几个要点”、“多页报告”等")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话框", placeholder="请在下方输入问题")
                query_input = gr.Textbox(placeholder="输入问题", container=False, show_copy_button=True)
                query_btn = gr.Button("发送", min_width=10)
        
        query_input.submit(
            fn=send_message,
            inputs=[
                temperature,
                timestamp,
                query_type,
                community_level,
                response_type,
                query_input,
                chatbot
            ],
            outputs=[query_input, chatbot]
        )
    
        query_btn.click(
            fn=send_message,
            inputs=[
                temperature,
                timestamp,
                query_type,
                community_level,
                response_type,
                query_input,
                chatbot
            ],
            outputs=[query_input, chatbot]
        )
    
    return demo


async def main():
    create_interface().launch(share=True)


if __name__ == "__main__":
    create_interface().launch(share=True)
