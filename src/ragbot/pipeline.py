"""Query pipeline: rewrite -> retrieve -> select within budget -> generate (D-011..D-013)."""

from dataclasses import dataclass, field

from ragbot import config
from ragbot.generate import build_messages, generate_answer, is_refusal, select_chunks
from ragbot.history import Turn, rewrite_query, trim_history
from ragbot.llm import ChatClient, LLMError, LLMNotConfigured


@dataclass
class Answer:
    text: str
    query: str  # what was searched (the rewrite, or the question itself)
    rewrite_note: str
    chunks: list  # chunk records passed to the prompt, best first
    hits: list  # retrieval hits (chunk_id, score, rank)
    cited: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)  # cited ids that were not in the context
    refused: bool = False
    used_llm: bool = True
    error: str = ""


class Chatbot:
    """One conversation. Holds the history; the retriever and LLM client are shared/injected."""

    def __init__(self, retriever=None, client=None, provider=None, method=config.RETRIEVAL_METHOD):
        if retriever is None:
            from ragbot.retrieve import Retriever

            retriever = Retriever()
        self.retriever, self.method = retriever, method
        self.llm_hint = ""
        if client is None:
            try:
                client = ChatClient(provider)
            except LLMNotConfigured as e:
                self.llm_hint = str(e)  # run retrieval-only instead of failing
        self.client = client
        self.history: list[Turn] = []

    def reset(self):
        self.history = []

    def ask(self, question, k=config.TOP_K):
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        if len(question) > config.MAX_QUERY_CHARS:
            raise ValueError(f"question is {len(question)} characters; the limit is {config.MAX_QUERY_CHARS}")

        history = trim_history(self.history)
        query, note = rewrite_query(question, history, self.client)
        hits = self.retriever.search(query, self.method, k)
        chunks = select_chunks(hits, self.retriever.by_id)
        ans = Answer(text="", query=query, rewrite_note=note, chunks=chunks, hits=hits)

        if self.client is None:
            ans.used_llm = False
            ans.text = config.NO_LLM_NOTICE.format(hint=self.llm_hint)
            return ans  # not stored in history: there is no real answer to condition on
        try:
            ans.text, ans.cited, ans.invalid_citations = generate_answer(
                question, chunks, self.client, history, rewritten=query)
        except LLMError as e:
            ans.error, ans.text, ans.used_llm = str(e), f"The LLM call failed: {e}", False
            return ans
        ans.refused = is_refusal(ans.text)
        self.history.append(Turn(question, ans.text))
        self.history = trim_history(self.history)
        return ans

    def prompt_for(self, question, k=config.TOP_K):
        """The exact messages that would be sent (for inspection and the docs)."""
        history = trim_history(self.history)
        hits = self.retriever.search(question, self.method, k)
        return build_messages(question, select_chunks(hits, self.retriever.by_id), history)
