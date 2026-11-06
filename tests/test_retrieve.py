"""Tests for BM25, dense and hybrid retrieval (D-008, D-009). No model download."""

import pytest
from fakes import FakeEmbedder

from ragbot.chunking import ChunkConfig
from ragbot.index import TokenLimitError, build_dense_index
from ragbot.retrieve import BM25, Hit, Retriever, rrf_fuse, tokenize


def chunk(i, text, path="Doc > A"):
    return {"chunk_id": f"doc#{i:04d}", "text": text, "embed_text": f"{path}\n\n{text}",
            "heading_path_str": path, "n_tokens": len(text.split())}


def retriever(chunks, field="embed_text", **emb_kw):
    emb = FakeEmbedder(dim=256, max_tokens=200, query_prefix="", **emb_kw)
    idx = build_dense_index(chunks, emb, ChunkConfig(size=512, overlap=0, min_size=0),
                            manifest_path=__file__, dense_field="embed_text")
    return Retriever(embedder=emb, index=idx, bm25_field=field)


def ids(hits):
    return [h.chunk_id for h in hits]


# --- Tokenizer (D-008 rules) -----------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tōkaidō Shinkansen", ["tokaido", "shinkansen"]),  # accents + case
        ("San'yō and San’yō", ["sanyo", "and", "sanyo"]),  # straight and curly apostrophes
        ("the Shinkansen's network", ["the", "shinkansen", "network"]),  # possessive dropped
        ("1,435 mm and 5.3 km", ["1435", "mm", "and", "5.3", "km"]),  # thousands, decimals
        ("in 1964.", ["in", "1964"]),  # sentence-final period is not a decimal
        ("Shin-Osaka at 320 km/h", ["shin", "osaka", "at", "320", "km", "h"]),  # hyphen, slash
        ("東海道 Shinkansen", ["東", "海", "道", "shinkansen"]),  # one token per CJK char
        ("ガ", ["ガ"]),  # katakana voicing mark kept (no accent stripping on Japanese)
        ("", []),
    ],
)
def test_tokenizer_rules(text, expected):
    assert tokenize(text) == expected


def test_model_name_stems_added_on_index_side_only():
    assert tokenize("N700A and 700T in the 1960s", index_side=True) == [
        "n700a", "n700", "and", "700t", "700", "in", "the", "1960s", "1960"]
    assert tokenize("N700A") == ["n700a"]  # query keeps the token as typed
    assert tokenize("1,067mm", index_side=True) == ["1067mm", "1067"]


# --- BM25 scoring ------------------------------------------------------------------------


def test_rare_term_outranks_common_term():
    bm = BM25([["train", "fast"], ["train", "maglev"], ["train", "slow"]])
    s = bm.scores(["train", "maglev"])
    assert s.argmax() == 1 and s[1] > s[0] == s[2]


def test_idf_never_negative_even_for_term_in_every_doc():
    bm = BM25([["train"]] * 5)
    assert bm.idf["train"] > 0


def test_length_normalisation_prefers_shorter_doc():
    bm = BM25([["maglev"] + ["filler"] * 20, ["maglev", "filler"]])
    s = bm.scores(["maglev"])
    assert s[1] > s[0]


def test_repeated_query_word_counts_once():
    bm = BM25([["maglev", "train"], ["train"]])
    assert list(bm.scores(["maglev"])) == list(bm.scores(["maglev", "maglev"]))


def test_empty_or_unknown_query_returns_nothing():
    r = retriever([chunk(0, "the line opened"), chunk(1, "maglev test track")])
    assert r.bm25_search("") == [] and r.bm25_search("zeppelin") == []


def test_ties_break_by_chunk_order():
    r = retriever([chunk(i, "identical text") for i in range(4)])
    assert ids(r.bm25_search("identical", k=4)) == ["doc#0000", "doc#0001", "doc#0002", "doc#0003"]


# --- The cases from the task -------------------------------------------------------------


def test_n700_matches_variants_but_not_unrelated_numbers():
    chunks = [
        chunk(0, "The N700A has improved brakes."),
        chunk(1, "The N700S entered service in 2020."),
        chunk(2, "The 700 Series ran at 285 km/h."),
        chunk(3, "Trains reached 700 passengers per set."),
        chunk(4, "The N7000 is a fictional model."),
    ]
    r = retriever(chunks)
    assert set(ids(r.bm25_search("N700"))) == {"doc#0000", "doc#0001"}
    assert ids(r.bm25_search("N700S")) == ["doc#0001"]  # a variant does not pull in its siblings


def test_japanese_query_does_not_flood_results():
    # 新幹線 appears in every chunk (like the corpus's parenthesised names); 東海道 in one.
    chunks = [chunk(i, f"Line {i} (新幹線) details.") for i in range(9)]
    chunks.append(chunk(9, "Tokaido line (東海道新幹線) details."))
    r = retriever(chunks)
    hits = r.bm25_search("東海道新幹線", k=10)
    assert hits[0].chunk_id == "doc#0009"
    # Characters shared by every chunk carry almost no weight: the others
    # score a small fraction of the real match instead of tying with it.
    assert all(h.score < 0.1 * hits[0].score for h in hits[1:])


def test_bm25_field_switch_heading_only_word():
    c = chunk(0, "It opened in 1964.", path="Line > Ridership")
    assert ids(retriever([c, chunk(1, "other")], field="embed_text").bm25_search("ridership")) == ["doc#0000"]
    assert retriever([c, chunk(1, "other")], field="text").bm25_search("ridership") == []


# --- Dense -------------------------------------------------------------------------------


def test_dense_exact_match_ranks_first_and_ranks_are_1_to_k():
    chunks = [chunk(i, f"topic{i} alpha{i} beta{i}") for i in range(6)]
    r = retriever(chunks)
    hits = r.dense_search("Doc > A topic3 alpha3 beta3", k=3)
    assert hits[0].chunk_id == "doc#0003"
    assert [h.rank for h in hits] == [1, 2, 3]


def test_dense_k_larger_than_corpus_returns_all():
    r = retriever([chunk(0, "a"), chunk(1, "b")])
    assert len(r.dense_search("a", k=10)) == 2


def test_overlong_query_fails_loudly():
    r = retriever([chunk(0, "a")])
    with pytest.raises(TokenLimitError):
        r.dense_search(" ".join(["word"] * 300))


# --- RRF (D-009) -------------------------------------------------------------------------


def hits(*cids):
    return [Hit(c, 1.0 / (i + 1), i + 1) for i, c in enumerate(cids)]


def test_rrf_hand_computed():
    out = rrf_fuse(hits("a", "b"), hits("b", "c"), k=3, alpha=0.5, rrf_k=60)
    # b: 0.5/62 + 0.5/61; a: 0.5/61; c: 0.5/62
    assert ids(out) == ["b", "a", "c"]
    assert out[0].score == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert out[2].score == pytest.approx(0.5 / 62)


def test_rrf_alpha_one_and_zero_reproduce_single_methods():
    dense, bm25 = hits("a", "b", "c", "d"), hits("d", "x", "a", "y")
    assert ids(rrf_fuse(dense, bm25, k=4, alpha=1.0)) == ["a", "b", "c", "d"]
    assert ids(rrf_fuse(dense, bm25, k=4, alpha=0.0)) == ["d", "x", "a", "y"]


def test_rrf_alpha_extremes_on_real_retriever():
    chunks = [chunk(i, t) for i, t in enumerate(
        ["maglev test track", "the line opened in 1964", "maglev speed record 603 km/h",
         "tokaido line ridership", "n700s entered service"])]
    r = retriever(chunks)
    q = "maglev speed"
    assert ids(r.hybrid_search(q, k=5, alpha=1.0)) == ids(r.dense_search(q, k=5))
    assert ids(r.hybrid_search(q, k=5, alpha=0.0)) == ids(r.bm25_search(q, k=5))


def test_rrf_keeps_chunk_found_by_one_method_and_breaks_ties_by_order():
    out = rrf_fuse(hits("b"), hits("a"), k=2, alpha=0.5, order={"a": 0, "b": 1})
    assert ids(out) == ["a", "b"]  # equal scores, a comes first in the corpus


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        retriever([chunk(0, "a")]).search("a", method="tfidf")
