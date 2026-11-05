"""Build and persist the dense and BM25 indexes into config.INDEX_DIR."""

from ragbot import config  # noqa: F401  (all tunables come from here)


def build_dense_index(chunks):
    raise NotImplementedError


def build_bm25_index(chunks):
    raise NotImplementedError


def load_indexes():
    raise NotImplementedError
