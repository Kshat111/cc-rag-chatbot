"""Dense, BM25 and hybrid search over the built indexes."""

from ragbot import config  # noqa: F401  (all tunables come from here)


def dense_search(query, k=None):
    raise NotImplementedError


def bm25_search(query, k=None):
    raise NotImplementedError


def hybrid_search(query, k=None):
    """Fuse dense and BM25 results, weighted by config.HYBRID_ALPHA."""
    raise NotImplementedError
