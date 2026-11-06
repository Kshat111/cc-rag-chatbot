"""Gradio chat UI: answer on the left, search query and cited sources on the right (D-014).

Run: uv run python -m ragbot.ui
One Chatbot (history) per browser session; the index and embedder are loaded once.
"""

import time

import gradio as gr

from ragbot import config
from ragbot.pipeline import Chatbot


def format_sources(ans):
    """Markdown list of the chunks that went into the prompt; cited ones in bold with a tick."""
    if not ans.chunks:
        return "_No passages retrieved._"
    lines = []
    for c in ans.chunks:
        cited = c["chunk_id"] in ans.cited
        snippet = " ".join(c["text"].split())[: config.UI_SNIPPET_CHARS]
        title = f"**✓ {c['chunk_id']}**" if cited else c["chunk_id"]
        lines.append(f"{title} · [{c['heading_path_str']}]({c['permalink']})\n\n> {snippet}…")
    return "\n\n".join(lines)


def format_status(ans):
    """One line about how the answer was produced: rewrite, refusal, warnings."""
    parts = []
    if ans.rewrite_note in ("", "no LLM") or ans.rewrite_note.startswith("rewrite"):
        parts.append(f"**Search query:** {ans.query}")
        if ans.rewrite_note.startswith("rewrite"):
            parts.append(f"_(kept your original question: {ans.rewrite_note})_")
    else:
        parts.append(f"**Search query:** {ans.query}")
    if ans.refused:
        parts.append("🚫 The model said the documents don't contain the answer.")
    if ans.invalid_citations:
        parts.append(f"⚠️ Cited ids that were not in the context: {', '.join(ans.invalid_citations)}")
    if not ans.used_llm:
        parts.append("ℹ️ No LLM answer (see message); passages shown below.")
    return "\n\n".join(parts)


def build_app(retriever=None, make_bot=None, clock=time.monotonic):
    """make_bot() -> Chatbot-like; defaults to one Chatbot per session over a shared retriever."""
    if make_bot is None:
        if retriever is None:
            from ragbot.retrieve import Retriever

            retriever = Retriever()
        make_bot = lambda: Chatbot(retriever=retriever)  # noqa: E731

    def ask(message, chat, bot, last_ts):
        bot = bot or make_bot()
        message = (message or "").strip()
        chat = chat or []
        wait = config.UI_MIN_SECONDS_BETWEEN_QUESTIONS - (clock() - last_ts)
        if not message:
            return chat, "", bot, last_ts, gr.skip(), gr.skip()
        if wait > 0:
            chat = chat + [{"role": "assistant", "content": f"⚠️ Please wait {wait:.0f}s between questions."}]
            return chat, message, bot, last_ts, gr.skip(), gr.skip()
        try:
            ans = bot.ask(message)
        except ValueError as e:  # empty or over-long question
            chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": f"⚠️ {e}"}]
            return chat, "", bot, clock(), gr.skip(), gr.skip()
        chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": ans.text}]
        return chat, "", bot, clock(), format_status(ans), format_sources(ans)

    def reset():
        return [], "", None, -1e9, "", ""

    with gr.Blocks(title="Shinkansen RAG chatbot") as demo:
        gr.Markdown("# Shinkansen & Japanese railways: RAG chatbot\n"
                    "Answers only from 15 Wikipedia articles (CC BY-SA 4.0), with citations. "
                    "Ask follow-ups with pronouns; start a new conversation to switch topics.")
        bot_state, ts_state = gr.State(None), gr.State(-1e9)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=460, label="Conversation")
                box = gr.Textbox(placeholder="Ask a question…", show_label=False, max_length=config.MAX_QUERY_CHARS)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    new = gr.Button("New conversation")
                gr.Examples(config.UI_EXAMPLES, inputs=box)
            with gr.Column(scale=2):
                status = gr.Markdown(label="How this answer was made")
                gr.Markdown("### Sources (✓ = cited)")
                sources = gr.Markdown()
        io = dict(inputs=[box, chat, bot_state, ts_state], outputs=[chat, box, bot_state, ts_state, status, sources])
        box.submit(ask, api_name="ask", **io)
        send.click(ask, api_name=False, **io)
        new.click(reset, outputs=[chat, box, bot_state, ts_state, status, sources], api_name=False)
    return demo


def main():
    build_app().queue(default_concurrency_limit=2).launch(server_name=config.UI_HOST, server_port=config.UI_PORT)


if __name__ == "__main__":
    main()
