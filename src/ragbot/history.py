"""Rewrite follow-up questions into standalone queries using chat history."""

from ragbot import config  # noqa: F401  (all tunables come from here)


def rewrite_query(question, history):
    """Return a standalone query; returns `question` unchanged if there is no history."""
    raise NotImplementedError
