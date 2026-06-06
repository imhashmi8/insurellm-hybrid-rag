import gradio as gr
from answer import answer_question

# Render each source with its content so users can verify which documents backed the answer
def format_context(chunks):
    out = "<h3 style='color:#ff7800;'>Retrieved Context</h3>\n\n"
    for c in chunks:
        out += f"<b style='color:#ff7800;'>Source: {c.metadata['source']}</b>\n\n{c.page_content}\n\n---\n\n"
    return out

# Handle a full turn in one function to avoid Gradio 6 issues with chained .then() event handlers
def respond(message, history):
    print(f"Question: {message}")
    history = history + [{"role": "user", "content": message}]
    answer, chunks = answer_question(message, history[:-1])
    print(f"Answer: {answer[:100]}")
    history.append({"role": "assistant", "content": answer})
    return "", history, format_context(chunks)

def main():
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# Insurellm Expert Assistant")
        # Two-column layout: conversation on the left, retrieved context on the right
        with gr.Row():
            with gr.Column():
                bot = gr.Chatbot(label="Conversation", height=600)
                box = gr.Textbox(placeholder="Ask anything about Insurellm...", show_label=False)
            ctx = gr.Markdown(value="*Retrieved context will appear here*", height=600)

        # Wire submit to respond so the chatbot and context panel update together in one event
        box.submit(respond, [box, bot], [box, bot, ctx])
    ui.launch(inbrowser=True, theme=theme)

if __name__ == "__main__":
    main()
