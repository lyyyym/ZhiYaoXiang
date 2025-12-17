import gradio as gr
from inference import generate_rag_response


def respond(user_message, chat_history):
    if not user_message:
        return "", (chat_history or [])
    answer = generate_rag_response(user_message)
    messages = chat_history or []
    messages = messages + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return "", messages


with gr.Blocks() as demo:
    gr.Markdown("# 药店智能问答助手")
    gr.Markdown("像和药店药剂师聊天一样描述你的情况，系统会基于药品知识库给出智能用药建议。")
    chatbot = gr.Chatbot(height=450, label="对话")
    with gr.Row():
        msg = gr.Textbox(
            label="输入你的问题",
            placeholder="例如：最近发烧咳嗽，想买点药。",
            lines=3,
        )
        submit_btn = gr.Button("发送", variant="primary")
    clear_btn = gr.Button("清空对话")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
