"""UI helpers and wiring (D-014). The app is built, not launched."""

from types import SimpleNamespace

from ragbot import config
from ragbot.ui import build_app, format_sources, format_status


def ans(**kw):
    base = dict(text="t", query="q", rewrite_note="", chunks=[], hits=[], cited=[], invalid_citations=[],
                refused=False, used_llm=True)
    base.update(kw)
    return SimpleNamespace(**base)


CH = [{"chunk_id": "a#0001", "heading_path_str": "A > B", "permalink": "http://p", "text": "x " * 500},
      {"chunk_id": "b#0002", "heading_path_str": "C", "permalink": "http://q", "text": "short"}]


def test_sources_mark_cited_chunks_link_to_the_revision_and_truncate():
    md = format_sources(ans(chunks=CH, cited=["a#0001"]))
    assert "**✓ a#0001**" in md and "b#0002" in md and "**✓ b#0002**" not in md
    assert "(http://p)" in md
    assert max(len(line) for line in md.splitlines()) < config.UI_SNIPPET_CHARS + 10
    assert "No passages" in format_sources(ans())


def test_status_reports_query_refusal_warnings_and_no_llm():
    s = format_status(ans(query="Why did Shima resign?", refused=True, invalid_citations=["z#0009"], used_llm=False))
    assert "Why did Shima resign?" in s and "don't contain" in s and "z#0009" in s and "No LLM answer" in s
    assert "kept your original" in format_status(ans(rewrite_note="rewrite looked invalid"))


def test_app_builds_with_an_injected_bot_factory():
    app = build_app(make_bot=lambda: None)
    assert app.title == "Shinkansen RAG chatbot"
