"""The experiment's guards (scripts/experiment_chunking.py): no model needed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment_chunking as ex  # noqa: E402


def test_scorable_ids_requires_every_evidence_quote_inside_one_chunk():
    chunks = [{"chunk_id": "a", "text": "It opened in 1964."}, {"chunk_id": "b", "text": "Then more."}]
    qs = [{"id": "q1", "evidence": ["opened in 1964"]},
          {"id": "q2", "evidence": ["opened in 1964", "Then more"]},
          {"id": "q3", "evidence": ["opened in 1964. Then more"]},  # split across two chunks
          {"id": "q4", "evidence": ["nowhere"]},
          {"id": "q5", "evidence": [], "answerable_now": True},
          {"id": "q6", "evidence": ["opened in 1964"], "answerable_now": False}]
    assert ex.scorable_ids(qs, chunks) == {"q1", "q2"}


def test_run_stops_with_a_clear_message_when_no_question_is_comparable(monkeypatch):
    chunk = {"chunk_id": "a", "text": "x", "n_tokens": 1}
    cache = {(64, 0): (object(), [chunk]), (128, 0): (object(), [chunk])}
    with pytest.raises(SystemExit, match="cannot compare"):
        ex.run([], None, [{"id": "q1", "evidence": ["missing"]}], [(64, 0), (128, 0)], "t", cache)


def test_cache_means_a_shared_configuration_is_built_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ex, "build", lambda docs, emb, size, ov: calls.append((size, ov)) or (object(), []))
    cache = {}
    with pytest.raises(SystemExit):  # no questions at all: stops after building
        ex.run([], None, [], [(512, 64)], "a", cache)
    with pytest.raises(SystemExit):
        ex.run([], None, [], [(512, 0), (512, 64)], "b", cache)
    assert calls == [(512, 64), (512, 0)]
