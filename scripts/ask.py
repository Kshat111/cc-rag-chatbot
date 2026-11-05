"""Query pipeline entry point (CLI). Usage: uv run python scripts/ask.py "question" """

import sys

from ragbot import pipeline


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: ask.py "your question"')
    print(pipeline.answer(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
