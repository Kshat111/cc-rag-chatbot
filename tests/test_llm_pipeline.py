"""Tests for the LLM client, history rewrite, grounded prompt and pipeline (D-011..D-013).
No network: the HTTP layer and the LLM are fakes."""

import io
import json
import urllib.error

import pytest
from fakes import FakeEmbedder

from ragbot import config
from ragbot.chunking import ChunkConfig
from ragbot.generate import SYSTEM_PROMPT, build_messages, cited_ids, is_refusal, select_chunks
from ragbot.history import Turn, rewrite_query, trim_history
from ragbot.index import build_dense_index
from ragbot.llm import ChatClient, LLMError, LLMNotConfigured
from ragbot.pipeline import Chatbot
from ragbot.retrieve import Hit, Retriever


def reply(text):
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(json.dumps({"error": {"message": "nope"}}).encode()))


# --- Client ----------------------------------------------------------------------------


def test_keyless_provider_needs_no_key_and_sends_no_auth_header(monkeypatch):
    sent = {}

    def post(url, headers, body, timeout):
        sent.update(url=url, headers=headers, body=json.loads(body))
        return reply(" hi ")

    c = ChatClient("ollama", post=post)
    assert c.chat([{"role": "user", "content": "x"}], 10) == "hi"
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in sent["headers"] and sent["body"]["temperature"] == 0.0


def test_keyed_provider_without_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMNotConfigured, match="OPENAI_API_KEY"):
        ChatClient("openai")


def test_key_is_sent_as_bearer_and_env_overrides_apply(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    monkeypatch.setenv("RAGBOT_LLM_BASE_URL", "http://gateway/v1/")
    monkeypatch.setenv("RAGBOT_LLM_MODEL", "my-model")
    monkeypatch.setenv("RAGBOT_LLM_API_KEY_ENV", "MY_KEY")
    seen = {}
    c = ChatClient("openai", post=lambda u, h, b, t: seen.update(u=u, h=h, b=json.loads(b)) or reply("ok"))
    c.chat([], 5)
    assert seen["u"] == "http://gateway/v1/chat/completions" and seen["b"]["model"] == "my-model"
    assert seen["h"]["Authorization"] == "Bearer secret"


def test_retries_transient_errors_then_succeeds():
    calls = []

    def post(*a):
        calls.append(1)
        if len(calls) < 3:
            raise http_error(429)
        return reply("done")

    sleeps = []
    assert ChatClient("ollama", post=post, sleep=sleeps.append).chat([], 5) == "done"
    assert len(calls) == 3 and sleeps == [2, 4]


def test_client_errors_are_not_retried_and_hide_nothing_useful():
    calls = []

    def post(*a):
        calls.append(1)
        raise http_error(401)

    with pytest.raises(LLMError, match="HTTP 401.*nope"):
        ChatClient("ollama", post=post, sleep=lambda s: None).chat([], 5)
    assert len(calls) == 1


def test_gives_up_after_retries_and_bad_shape_is_an_error():
    def down(*a):
        raise urllib.error.URLError("refused")

    with pytest.raises(LLMError, match="after 4 attempts"):
        ChatClient("ollama", post=down, sleep=lambda s: None).chat([], 5)
    with pytest.raises(LLMError, match="unexpected response"):
        ChatClient("ollama", post=lambda *a: b'{"oops": 1}').chat([], 5)


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        config.llm_settings("nope")


# --- History rewrite ---------------------------------------------------------------------


class ScriptedClient:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def chat(self, messages, max_tokens, temperature=0.0):
        self.calls.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


H = [Turn("When did the Tokaido Shinkansen open?", "1 October 1964."), Turn("Who was its chief engineer?", "Hideo Shima.")]


def test_no_history_means_no_llm_call():
    c = ScriptedClient()
    assert rewrite_query("Why did he resign?", [], c) == ("Why did he resign?", "no history") and not c.calls


def test_rewrite_uses_history_and_returns_clean_query():
    c = ScriptedClient('"Why did Hideo Shima resign?"')
    q, note = rewrite_query("Why did he resign?", H, c)
    assert (q, note) == ("Why did Hideo Shima resign?", "")
    assert "Hideo Shima." in c.calls[0][1]["content"] and "Last question: Why did he resign?" in c.calls[0][1]["content"]


@pytest.mark.parametrize("bad", ["", "line one\nline two", "x" * 500, RuntimeError("down")])
def test_bad_rewrite_falls_back_to_original(bad):
    q, note = rewrite_query("Why did he resign?", H, ScriptedClient(bad))
    assert q == "Why did he resign?" and note


def test_trim_history_drops_oldest_first_by_turns_and_chars():
    many = [Turn(f"q{i}", "a" * 10) for i in range(8)]
    assert [t.question for t in trim_history(many, max_turns=3, max_chars=10_000)] == ["q5", "q6", "q7"]
    big = [Turn("old", "x" * 5000), Turn("new", "y" * 100)]
    assert [t.question for t in trim_history(big, max_turns=5, max_chars=1000)] == ["new"]
    assert trim_history(many, max_turns=0) == []


# --- Prompt --------------------------------------------------------------------------------


def chunk(i, text, n=10):
    return {"chunk_id": f"doc#{i:04d}", "text": text, "embed_text": f"Doc > A\n\n{text}", "heading_path_str": "Doc > A",
            "n_tokens": n, "permalink": "http://x"}


def test_prompt_numbers_passages_cites_ids_and_marks_them_untrusted():
    msgs = build_messages("Q?", [chunk(0, "Alpha."), chunk(1, "Beta.")], history=H[:1], rewritten="Q standalone?")
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT
    assert "untrusted" in SYSTEM_PROMPT and config.REFUSAL_TEXT in SYSTEM_PROMPT and "ONLY" in SYSTEM_PROMPT
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    user = msgs[-1]["content"]
    assert "--- passage 1 ---\n[doc#0000] (Doc > A)\nAlpha." in user and "[doc#0001]" in user
    assert "Question: Q?" in user and "Standalone form used for the search: Q standalone?" in user


def test_budget_keeps_best_first_and_always_at_least_one():
    by_id = {f"doc#{i:04d}": chunk(i, "t", n=400) for i in range(5)}
    hits = [Hit(f"doc#{i:04d}", 1.0, i + 1) for i in range(5)]
    assert len(select_chunks(hits, by_id, budget=1000)) == 2
    assert [c["chunk_id"] for c in select_chunks(hits, by_id, budget=100)] == ["doc#0000"]


def test_citation_and_refusal_parsing():
    assert cited_ids("Yes [shinkansen#0006] and [n700-series-shinkansen#0001], again [shinkansen#0006].") == [
        "shinkansen#0006", "n700-series-shinkansen#0001"]
    assert is_refusal(config.REFUSAL_TEXT) and is_refusal("  i don't know based on the provided documents")
    assert not is_refusal("It was built in 1964 [a#0001].")


# --- Pipeline ------------------------------------------------------------------------------


def bot(client, chunks=None):
    chunks = chunks or [chunk(0, "Hideo Shima resigned in 1963 over cost overruns."), chunk(1, "The 700T was built by Kawasaki.")]
    emb = FakeEmbedder(dim=64, max_tokens=100, query_prefix="")
    idx = build_dense_index(chunks, emb, ChunkConfig(size=512, overlap=0, min_size=0), manifest_path=__file__)
    return Chatbot(retriever=Retriever(embedder=emb, index=idx), client=client, method="bm25")


def test_answer_with_valid_and_invalid_citations():
    b = bot(ScriptedClient("Shima resigned [doc#0000]. Also [other#0009]."))
    a = b.ask("Shima resigned")
    assert a.cited == ["doc#0000"] and a.invalid_citations == ["other#0009"] and not a.refused
    assert b.history == [Turn("Shima resigned", a.text)]


def test_refusal_detected_and_followup_is_rewritten_with_history():
    c = ScriptedClient(config.REFUSAL_TEXT, "Why did Shima resign?", "Because of cost overruns [doc#0000].")
    b = bot(c)
    assert b.ask("What is the TGV speed?").refused
    a = b.ask("Why did he resign?")
    assert a.query == "Why did Shima resign?" and a.cited == ["doc#0000"]
    assert [m["role"] for m in c.calls[-1]] == ["system", "user", "assistant", "user"]  # history reaches the answer prompt


def test_no_client_runs_retrieval_only_and_keeps_no_history():
    b = bot(None)
    b.llm_hint = "set X"
    a = b.ask("Shima resigned")
    assert not a.used_llm and "set X" in a.text and a.chunks and b.history == []


def test_llm_failure_is_reported_not_raised_and_history_unchanged():
    b = bot(ScriptedClient(LLMError("HTTP 500")))
    a = b.ask("Shima resigned")
    assert not a.used_llm and "HTTP 500" in a.error and b.history == []


def test_input_limits():
    b = bot(ScriptedClient())
    with pytest.raises(ValueError, match="limit"):
        b.ask("x" * (config.MAX_QUERY_CHARS + 1))
    with pytest.raises(ValueError, match="empty"):
        b.ask("   ")


def test_reset_clears_history():
    b = bot(ScriptedClient("ok [doc#0000]"))
    b.ask("Shima")
    b.reset()
    assert b.history == []
