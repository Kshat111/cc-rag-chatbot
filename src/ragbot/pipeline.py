"""Query pipeline: rewrite -> retrieve -> generate, keeping chat history."""

from ragbot import config  # noqa: F401  (all tunables come from here)


def answer(question, history=None):
    raise NotImplementedError
