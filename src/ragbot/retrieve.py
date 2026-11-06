"""Dense, BM25 and hybrid search over the built index (D-007, D-008, D-009).

All three methods return list[Hit] ranked 1..k, so the evaluation and the
pipeline can swap them freely. Scores are only comparable within one method
and one query.

The dense index is loaded with its staleness checks (index.load_dense_index);
BM25 is rebuilt in memory from the same chunks, so both always agree on the
chunk set.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ragbot import config
from ragbot.index import TokenLimitError, l2_normalize, load_dense_index


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float  # cosine (dense), BM25 score, or fused RRF score
    rank: int  # 1-based


# --- BM25 tokenizer (D-008) ---------------------------------------------------------

_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"  # hiragana, katakana, ideographs
_POSSESSIVE = re.compile(r"['’ʼ]s\b")  # "Shinkansen's" -> "Shinkansen", not "shinkansens"
_APOSTROPHE = re.compile(r"['’ʼ]")  # "San'yō" -> "Sanyō", so it matches "Sanyo"
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # "1,435" -> "1435"; "3, 4" untouched
_TOKEN = re.compile(rf"[{_CJK}]|\d+(?:\.\d+)+|[a-z0-9]+")  # one CJK char | decimal | word
_STEM = re.compile(r"^([a-z]*\d+)[a-z]+$")  # n700a -> n700, 700t -> 700, 1960s -> 1960


@lru_cache(maxsize=4096)
def _fold(ch):
    # Strip accents from Latin letters only. NFKD on Japanese would split
    # ガ into カ + a combining mark and drop it, merging different characters.
    return "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))


def tokenize(text, index_side=False):
    """BM25 tokens. index_side=True also adds model-name stems (rule 6 of D-008):
    chunks containing "N700A" index both n700a and n700, while a query keeps its
    token as typed, so "N700" finds every N700 variant and "N700S" only N700S."""
    text = re.sub(r"[\u0080-῿]", lambda m: _fold(m.group()), text).lower()
    text = _POSSESSIVE.sub("", text)
    text = _APOSTROPHE.sub("", text)
    text = _THOUSANDS.sub("", text)
    tokens = _TOKEN.findall(text)
    if not index_side:
        return tokens
    out = []
    for t in tokens:
        out.append(t)
        m = _STEM.match(t)
        if m:
            out.append(m.group(1))
    return out


# --- BM25 index -------------------------------------------------------------------------


class BM25:
    """Okapi BM25 over pre-tokenized documents, as an inverted index:
    term -> (doc positions, term frequencies)."""

    def __init__(self, docs_tokens, k1=config.BM25_K1, b=config.BM25_B):
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        postings = {}
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                postings.setdefault(term, ([], []))
                postings[term][0].append(i)
                postings[term][1].append(tf)
        self.postings = {t: (np.array(d), np.array(f, dtype=np.float32)) for t, (d, f) in postings.items()}
        # Lucene's IDF: never negative, so a term in every document adds ~0 instead of penalising.
        self.idf = {t: math.log(1 + (self.n_docs - len(d) + 0.5) / (len(d) + 0.5)) for t, (d, _) in self.postings.items()}

    def scores(self, query_tokens):
        out = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        # Unique terms: repeating a word in the query should not multiply its weight.
        for term in dict.fromkeys(query_tokens):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            out[docs] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[docs])
        return out


def _rank(scores, k, ids, keep=lambda s: True):
    """Top-k by score, ties broken by chunk order (deterministic)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    hits = []
    for i in order:
        if len(hits) == k or not keep(scores[i]):
            break
        hits.append(Hit(ids[i], float(scores[i]), len(hits) + 1))
    return hits


# --- Fusion (D-009) ---------------------------------------------------------------------


def rrf_fuse(dense_hits, bm25_hits, k, alpha=config.HYBRID_ALPHA, rrf_k=config.RRF_K, order=None):
    """Weighted reciprocal rank fusion. A chunk absent from one list gets 0 from it;
    chunks with a total of 0 are dropped (so alpha=1 or 0 reproduces one method).
    order: chunk_id -> position, used to break ties deterministically."""
    fused = {}
    for weight, hits in ((alpha, dense_hits), (1 - alpha, bm25_hits)):
        for h in hits:
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + weight / (rrf_k + h.rank)
    order = order or {}
    ranked = sorted((cid for cid, s in fused.items() if s > 0), key=lambda c: (-fused[c], order.get(c, 0), c))
    return [Hit(cid, fused[cid], r) for r, cid in enumerate(ranked[:k], start=1)]


# --- Retriever ------------------------------------------------------------------------------


class Retriever:
    """Loads the index once and serves all three methods."""

    def __init__(self, embedder=None, index=None, bm25_field=config.BM25_INDEX_FIELD):
        if embedder is None:
            from ragbot.index import FastEmbedEmbedder

            embedder = FastEmbedEmbedder()
        self.embedder = embedder
        self.index = index if index is not None else load_dense_index(embedder)
        self.chunks = self.index.chunks
        self.ids = [c["chunk_id"] for c in self.chunks]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.order = {cid: i for i, cid in enumerate(self.ids)}
        self.bm25_field = bm25_field
        self.bm25 = BM25([tokenize(c[bm25_field], index_side=True) for c in self.chunks])

    def dense_search(self, query, k=config.TOP_K):
        text = self.embedder.query_prefix + query
        n = self.embedder.count_tokens([text])[0]
        if n > self.embedder.max_tokens:
            # fastembed would silently truncate the query; refuse instead.
            raise TokenLimitError(f"query is {n} tokens (with instruction); limit {self.embedder.max_tokens}")
        q = l2_normalize(self.embedder.embed_query(query))
        return _rank(self.index.vectors @ q, k, self.ids)

    def bm25_search(self, query, k=config.TOP_K):
        # Only chunks sharing at least one term; an unknown-word query returns [].
        return _rank(self.bm25.scores(tokenize(query)), k, self.ids, keep=lambda s: s > 0)

    def hybrid_search(self, query, k=config.TOP_K, alpha=config.HYBRID_ALPHA):
        n = config.HYBRID_CANDIDATES
        return rrf_fuse(self.dense_search(query, n), self.bm25_search(query, n), k, alpha, config.RRF_K, self.order)

    def search(self, query, method="hybrid", k=config.TOP_K):
        if method not in config.RETRIEVAL_METHODS:
            raise ValueError(f"method must be one of {config.RETRIEVAL_METHODS}")
        return getattr(self, f"{method}_search")(query, k)


_default = None


def default_retriever():
    """Shared Retriever, loaded on first use (model load + staleness checks)."""
    global _default
    if _default is None:
        _default = Retriever()
    return _default


def dense_search(query, k=config.TOP_K):
    return default_retriever().dense_search(query, k)


def bm25_search(query, k=config.TOP_K):
    return default_retriever().bm25_search(query, k)


def hybrid_search(query, k=config.TOP_K):
    """Fuse dense and BM25 results, weighted by config.HYBRID_ALPHA."""
    return default_retriever().hybrid_search(query, k)
