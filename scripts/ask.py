"""Query pipeline entry point (CLI).

  uv run python scripts/ask.py "question"        one question
  uv run python scripts/ask.py                   interactive chat with history ("/reset", "/quit")
  uv run python scripts/ask.py --provider ollama "question"

Provider and model: see config.LLM_PROVIDERS and the RAGBOT_LLM_* variables (README).
Without an API key it still runs and prints the retrieved passages.
"""

import argparse
import sys

from ragbot import config
from ragbot.index import StaleIndexError
from ragbot.pipeline import Chatbot


def show(ans):
    if ans.query and ans.rewrite_note != "no history":
        print(f"  [search query: {ans.query}]" + (f"  (kept original: {ans.rewrite_note})" if ans.rewrite_note else ""))
    print(f"\n{ans.text}")
    if ans.refused:
        print("  [refusal]")
    if ans.invalid_citations:
        print(f"  [WARNING: cited ids not in the context: {ans.invalid_citations}]")
    print("\nSources:")
    for c in ans.chunks:
        mark = "*" if c["chunk_id"] in ans.cited else " "
        print(f" {mark} {c['chunk_id']:<32} {c['heading_path_str']}   {c['permalink']}")
    print("  (* = cited in the answer)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("question", nargs="*")
    p.add_argument("--provider", choices=sorted(config.LLM_PROVIDERS), default=None)
    p.add_argument("--method", choices=config.RETRIEVAL_METHODS, default=config.RETRIEVAL_METHOD)
    p.add_argument("--show-prompt", action="store_true", help="print the messages sent to the LLM and exit")
    args = p.parse_args()
    try:
        bot = Chatbot(provider=args.provider, method=args.method)
    except StaleIndexError as e:
        print(f"Index not usable: {e}", file=sys.stderr)
        return 1
    if bot.client is None:
        print(f"Note: {bot.llm_hint}\n", file=sys.stderr)

    if args.show_prompt:
        for m in bot.prompt_for(" ".join(args.question)):
            print(f"## {m['role']}\n{m['content']}\n")
        return 0
    if args.question:
        show(bot.ask(" ".join(args.question)))
        return 0
    print("Ask about the Shinkansen and Japanese railways. /reset starts a new conversation, /quit exits.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q in ("/quit", "/exit"):
            return 0
        if q == "/reset":
            bot.reset()
            print("(history cleared)")
        elif q:
            try:
                show(bot.ask(q))
            except ValueError as e:
                print(f"! {e}")


if __name__ == "__main__":
    sys.exit(main())
