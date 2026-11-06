"""Test doubles shared by the index and retrieval tests."""

import hashlib

import numpy as np


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors; 1 token per word plus 2 specials."""

    def __init__(self, name="fake-model", dim=16, max_tokens=12, passage_prefix="", query_prefix="q: ", salt=""):
        self.name, self.dim, self.max_tokens = name, dim, max_tokens
        self.passage_prefix, self.query_prefix, self.salt = passage_prefix, query_prefix, salt

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            h = int(hashlib.md5((self.salt + w).encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed_passages(self, texts):
        return np.stack([self._vec(self.passage_prefix + t) for t in texts])

    def embed_query(self, text):
        return self._vec(self.query_prefix + text)

    def count_tokens(self, texts):
        return [len(t.split()) + 2 for t in texts]
