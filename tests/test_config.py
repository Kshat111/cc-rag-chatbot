"""Smoke test: config imports and core tunables are sane."""

from ragbot import config


def test_overlap_smaller_than_chunk():
    assert 0 <= config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_top_k_positive():
    assert config.TOP_K > 0


def test_hybrid_alpha_in_range():
    assert 0.0 <= config.HYBRID_ALPHA <= 1.0
