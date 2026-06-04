import gradio as gr
from answer import answer_question

def format_context(chunks):
    out = "<h3 style='color:#ff7800;'>Retrieved Context</h3>\n\n"
    for c in chunks:
        out += f"<b style='color:#ff7800;'>Source: {c.metadata['source']}</b>\n\n{c.page_content}\n\n---\n\n"
    return out

def chat(history):
    answer, chunks = answer_question(history[-1]["content"], history[:-1])
    history.append({"role": "assistant", "content": answer})
    return history, format_context(chunks)

def main():
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# Insurellm Expert Assistant\n")
        with gr.Row():
            with gr.Column():
                bot = gr.Chatbot(label="Conversation", height=600)
                box = gr.Textbox(placeholder="Ask anything about Insurellm...", show_label=False)
            ctx = gr.Markdown(value="*Retrieved context will appear here*", height=600)

        box.submit(lambda m, h: ("", h + [{"role": "user", "content": m}]),
                   [box, bot], [box, bot]).then(chat, bot, [bot, ctx])
    ui.launch(inbrowser=True, theme=theme)

if __name__ == "__main__":
    main()
