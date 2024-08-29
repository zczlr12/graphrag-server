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
SYSTEM_MESSAGE = """
你叫电厂百晓生，你将会学习核电厂经验反馈，工作负责人会对你进行提问，你需要给出针对性建议。

该电厂经验反馈是日本政府的最终报告，由IAEA主导，分为五个技术卷，详细介绍了事故的背景、技术评估、应急响应和事故后的管理措施。各卷的主要内容为：
1.事故描述和背景：包括对事故的详细描述和背景信息。
2.安全评估：对事故的技术和安全方面进行了详细评估。
3.应急准备和响应：讨论了事故期间的应急措施和响应。
4.事故后的恢复和管理：涉及环境恢复和核材料管理。
5.总结和建议
"""


def send_message(temperature, timestamp, index, community_level, response_type, stream, history):
    try:
        model = QUERY_TYPES[index]
        if not index:
            model = auto_select(history[-1][0])
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        for human, assistant in history:
            if human:
                messages.append({"content": human})
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
            for chunk in response.iter_lines(None):
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


def get_questions(temperature, timestamp, community_level, history):
    return requests.post(
        f"{API_BASE}/v1/advice_questions",
        json={
            "messages": history,
            "model": f"{timestamp}-local",
            "temperature": temperature,
            "community_level": community_level
        },
    ).json()["questions"]


def create_interface(g):
    with gr.Blocks(gr.themes.Base(), title="电厂百晓生", fill_height=True, fill_width=True) as demo:
        gr.Markdown("# <center>电厂百晓生</center>")
        question_history = gr.State([{"content": SYSTEM_MESSAGE}])
        with gr.Row():
            with gr.Column():
                gr.Markdown("## ⚙ 设置")
                temperature = gr.Slider(0, 2, 0, step=0.1, label="模型温度", info="模型生成文本的随机程度")
                timestamp = gr.Dropdown(INDEX_FOLDERS, value=INDEX_FOLDERS[-1], label="数据库")
                query_type = gr.Radio(QUERY_TYPES_CN, value="自动", type="index", label="搜索方法", info="自动：自动判断，局部：图元素+原文本，全局：社区报告，无：仅历史记录")
                community_level = gr.Slider(0, 3, 2, step=1, label="社区层级", info="社区的细粒程度")
                response_type = gr.Textbox(label="响应类型", placeholder="例如：“一句话”、“几个要点”、“多页报告”等")
                stream = gr.Checkbox(True, label="流式输出")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话框", show_copy_button=True, layout="panel", placeholder="请在下方输入问题")
                gr.ClearButton(chatbot, value="清空聊天记录")
                with gr.Group():
                    with gr.Row():
                        query_input = gr.Textbox(placeholder="输入问题", label="输入框", scale=7, show_copy_button=True)
                        query_btn = gr.Button("发送", variant="primary", scale=1)
                gr.Markdown("### 你可能想问")
                q1 = gr.DuplicateButton(g[0])
                q2 = gr.DuplicateButton(g[1])
                q3 = gr.DuplicateButton(g[2])
                q4 = gr.DuplicateButton(g[-2])
                q5 = gr.DuplicateButton(g[-1])

        timestamp.change(
            get_questions,
            [temperature, timestamp, community_level, question_history],
            [q1, q2, q3, q4, q5]
        )
        q1.click(lambda x: x, q1, query_input)
        q2.click(lambda x: x, q2, query_input)
        q3.click(lambda x: x, q3, query_input)
        q4.click(lambda x: x, q4, query_input)
        q5.click(lambda x: x, q5, query_input)
        query_input.submit(
            lambda x, y: ("", [*y, [x, None]]),
            [query_input, chatbot],
            [query_input, chatbot]
        ).then(
            lambda x, y: [*x, {"content": y[-1][0]}],
            [question_history, chatbot],
            question_history
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
            lambda x, y: [*x, {"content": y[-1][0]}],
            [question_history, chatbot],
            question_history
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
        question_history.change(
            get_questions,
            [temperature, timestamp, community_level, question_history],
            [q1, q2, q3, q4, q5]
        )

    return demo


async def main():
    g = get_questions(0, INDEX_FOLDERS[-1], 2, [{"content": SYSTEM_MESSAGE}])
    create_interface(g).launch()


if __name__ == "__main__":
    g = get_questions(0, INDEX_FOLDERS[-1], 2, [{"content": SYSTEM_MESSAGE}])
    create_interface(g).launch()
