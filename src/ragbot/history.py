"""Conversation history and follow-up rewriting (D-012).

Each follow-up is rewritten into a standalone search query using the last few
turns, because the retriever sees no history: "Why did he resign?" alone
retrieves nothing useful. The original question is still what the answering
prompt receives, next to the rewrite.
"""

from dataclasses import dataclass

from ragbot import config


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


REWRITE_SYSTEM = (
    "You rewrite the user's last question into one standalone search query for a "
    "knowledge base about Japanese railways and the Shinkansen.\n"
    "- Replace pronouns and vague references (he, it, that line, the same train) with the "
    "names they refer to in the conversation.\n"
    "- If the question is already standalone, or switches to a new topic, return it unchanged.\n"
    "- Do not answer the question. Output only the query, on one line."
)


def trim_history(history, max_turns=config.MAX_HISTORY_TURNS, max_chars=config.MAX_HISTORY_CHARS):
    """Keep the most recent turns within both limits; the oldest are dropped first."""
    kept, used = [], 0
    for turn in reversed(history[-max_turns:] if max_turns else []):
        used += len(turn.question) + len(turn.answer)
        if kept and used > max_chars:
            break
        kept.append(turn)
    return list(reversed(kept))


def rewrite_query(question, history, client=None):
    """Return (query, note). note is "" when the rewrite was used, else why the original was kept.

    Falls back to the original question whenever the rewrite looks wrong: a
    broken rewrite must never make retrieval worse than no rewrite.
    """
    history = trim_history(history)
    if not history:
        return question, "no history"
    if client is None:
        return question, "no LLM"
    convo = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in history)
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLast question: {question}\n\nStandalone query:"},
    ]
    try:
        out = client.chat(messages, max_tokens=config.LLM_MAX_REWRITE_TOKENS)
    except Exception as e:  # any LLM failure: keep going with the raw question
        return question, f"rewrite failed ({type(e).__name__})"
    out = out.strip().strip('"')
    if not out or "\n" in out or len(out) > 3 * len(question) + 80:
        return question, "rewrite looked invalid"
    return out, ""
