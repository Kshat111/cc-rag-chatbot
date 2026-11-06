"""Tests for the dense index (D-006, D-007). No model download: a fake embedder
stands in for fastembed. The real model is exercised only when
RAGBOT_MODEL_TESTS=1 (see the last test)."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from fakes import FakeEmbedder

from ragbot.chunking import ChunkConfig
from ragbot.index import (
    StaleIndexError,
    TokenLimitError,
    build_dense_index,
    check_token_limits,
    l2_normalize,
    load_dense_index,
    save_index,
)


def chunk(i, text, path="Doc > A"):
    return {"chunk_id": f"doc#{i:04d}", "text": text, "embed_text": f"{path}\n\n{text}",
            "n_tokens": max(1, len(text) // 4)}


CFG = ChunkConfig(size=512, overlap=64, min_size=64)


@pytest.fixture
def paths(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"documents": []}')
    return {
        "manifest_path": manifest,
        "chunks_path": tmp_path / "index" / "chunks.jsonl",
        "vectors_path": tmp_path / "index" / "dense.npy",
        "meta_path": tmp_path / "index" / "index_meta.json",
    }


def build_and_save(chunks, emb, paths, field="embed_text", cfg=CFG):
    idx = build_dense_index(chunks, emb, cfg, manifest_path=paths["manifest_path"], dense_field=field)
    save_index(idx, paths["chunks_path"], paths["vectors_path"], paths["meta_path"])
    return idx


def load(emb, paths, field="embed_text", cfg=CFG):
    return load_dense_index(emb, cfg, paths["manifest_path"], field,
                            paths["chunks_path"], paths["vectors_path"], paths["meta_path"])


CHUNKS = [chunk(0, "the line opened in 1964"), chunk(1, "maglev test track in Yamanashi")]


# --- Token check ---------------------------------------------------------------


def test_chunk_over_limit_fails_loudly_and_names_it():
    emb = FakeEmbedder(max_tokens=8)
    long = chunk(1, "one two three four five six seven")  # path(3) + 7 words + 2 specials = 12
    with pytest.raises(TokenLimitError, match=r"doc#0001: 12 tokens"):
        check_token_limits([chunk(0, "short"), long], emb, "embed_text")


def test_exactly_at_limit_passes_but_prefix_pushes_it_over():
    text = "a b c d e"  # embed_text = "Doc > A" (3 words) + 5 words + 2 specials = 10
    check_token_limits([chunk(0, text)], FakeEmbedder(max_tokens=10), "embed_text")
    with pytest.raises(TokenLimitError):
        check_token_limits([chunk(0, text)], FakeEmbedder(max_tokens=10, passage_prefix="passage: "), "embed_text")


def test_token_check_runs_on_the_configured_field():
    c = chunk(0, "a b c d e")  # text: 5 + 2 = 7 tokens; embed_text: 10
    check_token_limits([c], FakeEmbedder(max_tokens=7), "text")
    with pytest.raises(TokenLimitError):
        check_token_limits([c], FakeEmbedder(max_tokens=7), "embed_text")


def test_failed_token_check_writes_nothing(paths):
    with pytest.raises(TokenLimitError):
        build_and_save([chunk(0, " ".join(["w"] * 50))], FakeEmbedder(max_tokens=12), paths)
    assert not any(paths[k].exists() for k in ("chunks_path", "vectors_path", "meta_path"))


def test_token_stats_report_estimate_undercount():
    _, stats = check_token_limits(CHUNKS, FakeEmbedder(max_tokens=50), "embed_text")
    assert stats["max"] <= 50 and stats["limit"] == 50
    assert set(stats) >= {"min", "median", "max", "chars_per_token", "estimate_undercounted"}


# --- Build, save, load ---------------------------------------------------------


def test_vectors_are_normalised_and_rows_follow_chunk_order(paths):
    emb = FakeEmbedder(max_tokens=50)
    idx = build_and_save(CHUNKS, emb, paths)
    assert idx.vectors.shape == (2, emb.dim) and idx.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(idx.vectors, axis=1), 1.0)
    # Row 1 is the maglev chunk: it is the nearest row to a maglev query.
    q = l2_normalize(emb.embed_query("maglev Yamanashi track"))
    assert int(np.argmax(idx.vectors @ q)) == 1


def test_round_trip(paths):
    emb = FakeEmbedder(max_tokens=50)
    built = build_and_save(CHUNKS, emb, paths)
    loaded = load(emb, paths)
    assert [c["chunk_id"] for c in loaded.chunks] == ["doc#0000", "doc#0001"]
    assert np.array_equal(loaded.vectors, built.vectors)
    assert loaded.meta["n_chunks"] == 2 and loaded.meta["chunks_sha256"]


def test_l2_normalize_keeps_zero_vector_finite():
    out = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.all(np.isfinite(out)) and np.all(out == 0)


# --- Staleness -----------------------------------------------------------------


@pytest.mark.parametrize(
    "change, key",
    [
        (dict(emb=FakeEmbedder(name="other-model", max_tokens=50)), "embedding_model"),
        (dict(emb=FakeEmbedder(max_tokens=40)), "max_tokens"),
        (dict(emb=FakeEmbedder(max_tokens=50, passage_prefix="passage: ")), "passage_prefix"),
        (dict(field="text"), "dense_field"),
        (dict(cfg=replace(CFG, size=256)), "chunk_config"),
    ],
)
def test_config_or_model_change_is_stale(paths, change, key):
    build_and_save(CHUNKS, FakeEmbedder(max_tokens=50), paths)
    kwargs = {"emb": FakeEmbedder(max_tokens=50), "field": "embed_text", "cfg": CFG, **change}
    with pytest.raises(StaleIndexError, match=key):
        load(kwargs["emb"], paths, kwargs["field"], kwargs["cfg"])


def test_query_instruction_toggle_does_not_require_rebuild(paths):
    build_and_save(CHUNKS, FakeEmbedder(max_tokens=50, query_prefix="q: "), paths)
    load(FakeEmbedder(max_tokens=50, query_prefix=""), paths)  # no error


def test_refetched_corpus_is_stale(paths):
    build_and_save(CHUNKS, FakeEmbedder(max_tokens=50), paths)
    paths["manifest_path"].write_text('{"documents": ["changed"]}')
    with pytest.raises(StaleIndexError, match="manifest_sha256"):
        load(FakeEmbedder(max_tokens=50), paths)


def test_overwritten_chunks_file_is_stale(paths):
    build_and_save(CHUNKS, FakeEmbedder(max_tokens=50), paths)
    with open(paths["chunks_path"], "a") as f:
        f.write(json.dumps(chunk(2, "extra")) + "\n")
    with pytest.raises(StaleIndexError, match="changed since the index was built"):
        load(FakeEmbedder(max_tokens=50), paths)


def test_corrupt_vectors_file_is_stale(paths):
    build_and_save(CHUNKS, FakeEmbedder(max_tokens=50), paths)
    paths["vectors_path"].write_bytes(paths["vectors_path"].read_bytes()[:40])
    with pytest.raises(StaleIndexError):
        load(FakeEmbedder(max_tokens=50), paths)


def test_missing_file_is_stale(paths):
    with pytest.raises(StaleIndexError, match="missing"):
        load(FakeEmbedder(max_tokens=50), paths)


def test_silently_changed_model_weights_caught_by_probe(paths):
    build_and_save(CHUNKS, FakeEmbedder(max_tokens=50), paths)
    # Same name, dims and settings, different weights (salted hashing).
    with pytest.raises(StaleIndexError, match="probe embedding changed"):
        load(FakeEmbedder(max_tokens=50, salt="v2"), paths)


# --- Real model (opt-in) ---------------------------------------------------------


@pytest.mark.skipif(os.environ.get("RAGBOT_MODEL_TESTS") != "1", reason="set RAGBOT_MODEL_TESTS=1 to run")
def test_real_model_tokenizer_counts_specials_and_ignores_truncation():
    from ragbot.index import FastEmbedEmbedder

    emb = FastEmbedEmbedder()
    assert emb.dim == 384 and emb.max_tokens == 512
    assert emb.count_tokens(["hello"]) == [3]  # [CLS] hello [SEP]
    assert emb.count_tokens([" ".join(["word"] * 600)])[0] > 512  # truncation is off
    v = l2_normalize(emb.embed_passages(["Tōkaidō Shinkansen"]))
    assert v.shape == (1, 384)
