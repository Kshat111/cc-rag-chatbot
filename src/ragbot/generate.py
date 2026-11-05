"""Build a grounded prompt from retrieved chunks and call the LLM."""

from ragbot import config  # noqa: F401  (all tunables come from here)


def build_prompt(question, chunks):
    raise NotImplementedError


def generate_answer(question, chunks):
    """Return an answer grounded only in `chunks`, with source citations."""
    raise NotImplementedError
