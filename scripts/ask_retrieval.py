"""Retrieval-only CLI: print the top hits for a query, no LLM.

Usage:
  uv run python scripts/ask_retrieval.py "N700S" --method bm25
  uv run python scripts/ask_retrieval.py "N700S" --method all -k 3
"""

import argparse
import sys

from ragbot import config
from ragbot.index import StaleIndexError
from ragbot.retrieve import Retriever


def show(retriever, query, method, k):
    print(f"[{method}]")
    hits = retriever.search(query, method, k)
    if not hits:
        print("  (no hits)")
    for h in hits:
        c = retriever.by_id[h.chunk_id]
        snippet = " ".join(c["text"].split())[:150]
        print(f"  {h.rank}. {h.chunk_id:<34} {h.score:8.4f}  {c['heading_path_str']}")
        print(f"     {snippet}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("query")
    p.add_argument("--method", choices=[*config.RETRIEVAL_METHODS, "all"], default="hybrid")
    p.add_argument("-k", type=int, default=3)
    args = p.parse_args()
    try:
        retriever = Retriever()
    except StaleIndexError as e:
        print(f"Index not usable: {e}", file=sys.stderr)
        return 1
    print(f"Query: {args.query}")
    methods = config.RETRIEVAL_METHODS if args.method == "all" else [args.method]
    for m in methods:
        show(retriever, args.query, m, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
