"""Tests for the retrieval metrics in scripts/evaluate.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import evidence_groups, mrr, query_for, score  # noqa: E402


def test_evidence_matches_across_whitespace_and_both_overlapping_chunks():
    chunks = [{"chunk_id": "a", "text": "It was built\nby   Kawasaki."}, {"chunk_id": "b", "text": "built by Kawasaki. More."},
              {"chunk_id": "c", "text": "unrelated"}]
    assert evidence_groups({"evidence": ["built by Kawasaki"]}, chunks) == [{"a", "b"}]


def test_recall_counts_groups_and_mrr_uses_first_gold():
    groups = [{"x"}, {"y", "z"}]
    assert score(["x", "q", "w"], groups, 3) == 0.5  # only the first group is covered
    assert score(["x", "z"], groups, 3) == 1.0
    assert mrr(["q", "z", "x"], groups) == pytest.approx(0.5)
    assert mrr(["q", "w"], groups) == 0.0


def test_multi_turn_uses_last_expected_rewrite():
    q = {"question": "Why did he resign?", "turns": [{"expected_rewrite": "a"}, {"expected_rewrite": "Why did Shima resign?"}]}
    assert query_for(q) == "Why did Shima resign?"
