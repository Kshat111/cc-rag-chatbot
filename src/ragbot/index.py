"""Build, save and load the dense index (D-006, D-007).

Flow: chunks -> token check (real tokenizer, fail loudly) -> embed ->
L2-normalise -> save chunks.jsonl + dense.npy + index_meta.json.
Loading re-checks everything that could make the stored vectors disagree
with the current corpus, chunk settings or model, and raises StaleIndexError.

BM25 is built in stage 3b (retrieve.py) from the same chunks.jsonl.
"""

import hashlib
import io
import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from ragbot import config
from ragbot.chunking import ChunkConfig, get_token_counter

SCHEMA_VERSION = 1


class TokenLimitError(Exception):
    """A chunk is longer than the embedding model's real input limit."""


class StaleIndexError(Exception):
    """The saved index no longer matches the corpus, config or model."""


# --- Embedders -----------------------------------------------------------------


class Embedder(Protocol):
    """What the index needs from a model. Tests use a fake implementation."""

    name: str
    dim: int
    max_tokens: int
    passage_prefix: str
    query_prefix: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Token counts with special tokens included and no truncation."""
        ...


class FastEmbedEmbedder:
    """bge-small-en-v1.5 through fastembed (onnxruntime, no PyTorch)."""

    def __init__(
        self,
        model_name=config.EMBEDDING_MODEL,
        tokenizer_repo=config.EMBED_TOKENIZER_REPO,
        cache_dir=config.EMBED_CACHE_DIR,
        use_query_instruction=config.EMBED_USE_QUERY_INSTRUCTION,
    ):
        # Imported here so tests and tools that never embed don't load onnxruntime.
        from fastembed import TextEmbedding

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.name = model_name
        self.max_tokens = config.EMBED_MAX_TOKENS
        self.passage_prefix = config.EMBED_PASSAGE_PREFIX
        self.query_prefix = config.EMBED_QUERY_INSTRUCTION if use_query_instruction else ""
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        self.dim = self._model.embedding_size

        # Same tokenizer.json fastembed loads, but with truncation off (fastembed's
        # own copy truncates at 512, hiding exactly what we check for). Shared with
        # the chunker, so chunk sizes and this check count identically.
        self._count = get_token_counter(tokenizer_repo)

    def embed_passages(self, texts):
        return _stack(self._model.embed([self.passage_prefix + t for t in texts],
                                        batch_size=config.EMBED_BATCH_SIZE))

    def embed_query(self, text):
        return _stack(self._model.embed([self.query_prefix + text]))[0]

    def count_tokens(self, texts):
        return [self._count(t, True) for t in texts]


def _stack(vectors):
    return np.asarray(list(vectors), dtype=np.float32)


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0  # an all-zero vector stays zero instead of becoming NaN
    return (matrix / norms).astype(np.float32)


# --- Token check (fail loudly) ----------------------------------------------------


def check_token_limits(chunks, embedder, field=config.DENSE_INDEX_FIELD):
    """Count real tokens for every chunk as the model will see it.

    Raises TokenLimitError listing every chunk over the limit. Returns per-chunk
    counts and summary stats, including how often chars/4 undercounted.
    """
    texts = [embedder.passage_prefix + c[field] for c in chunks]
    counts = embedder.count_tokens(texts)
    over = [(c["chunk_id"], n) for c, n in zip(chunks, counts) if n > embedder.max_tokens]
    if over:
        listed = "\n".join(f"  {cid}: {n} tokens" for cid, n in over)
        raise TokenLimitError(
            f"{len(over)} chunk(s) exceed {embedder.name}'s limit of "
            f"{embedder.max_tokens} tokens; nothing was written:\n{listed}"
        )
    chars = sum(len(t) for t in texts)
    # Compare against the chars/4 estimate, whatever unit the chunker used.
    est = [c.get("n_tokens_est", c["n_tokens"]) for c in chunks]
    undercounted = sum(1 for e, n in zip(est, counts) if n > e)
    stats = {
        "min": min(counts),
        "median": statistics.median(counts),
        "max": max(counts),
        "limit": embedder.max_tokens,
        "chars_per_token": round(chars / sum(counts), 3),
        "estimate_undercounted": undercounted,
        "max_estimate_error_pct": round(max((n - e) / e * 100 for e, n in zip(est, counts)), 1),
    }
    return counts, stats


# --- Fingerprint and staleness ------------------------------------------------------


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_fingerprint(embedder, chunk_cfg, manifest_path, dense_field):
    """Everything that changes the stored vectors. The query instruction is left
    out on purpose: it only affects query vectors, so toggling it needs no rebuild."""
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "max_tokens": embedder.max_tokens,
        "passage_prefix": embedder.passage_prefix,
        "dense_field": dense_field,
        "chunk_config": asdict(chunk_cfg),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _versions():
    out = {}
    for pkg in ("fastembed", "onnxruntime", "numpy", "tokenizers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


# --- Build and save -------------------------------------------------------------------


@dataclass
class DenseIndex:
    chunks: list
    vectors: np.ndarray  # [n_chunks, dim], L2-normalised
    meta: dict


def build_dense_index(chunks, embedder, chunk_cfg, manifest_path=None, dense_field=None):
    """Check tokens, embed, and return an in-memory DenseIndex (not yet saved)."""
    manifest_path = manifest_path or config.MANIFEST_PATH
    dense_field = dense_field or config.DENSE_INDEX_FIELD
    timings = {}

    t = time.perf_counter()
    counts, token_stats = check_token_limits(chunks, embedder, dense_field)
    timings["token_check_s"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    vectors = l2_normalize(embedder.embed_passages([c[dense_field] for c in chunks]))
    timings["embed_s"] = round(time.perf_counter() - t, 3)
    if vectors.shape != (len(chunks), embedder.dim):
        raise ValueError(f"embedder returned {vectors.shape}, expected {(len(chunks), embedder.dim)}")

    probe = l2_normalize(embedder.embed_passages([config.EMBED_PROBE_TEXT]))[0]
    meta = {
        **expected_fingerprint(embedder, chunk_cfg, manifest_path, dense_field),
        "query_prefix": embedder.query_prefix,
        "n_chunks": len(chunks),
        "token_stats": token_stats,
        "timings": timings,
        "probe_text": config.EMBED_PROBE_TEXT,
        "probe_vector": [round(float(x), 6) for x in probe],
        "versions": _versions(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return DenseIndex(chunks=chunks, vectors=vectors, meta=meta)


def _atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def save_index(index, chunks_path=None, vectors_path=None, meta_path=None):
    """Write chunks, vectors, then meta. Meta goes last: if the process dies
    midway, the old meta no longer matches the new files and load fails loudly."""
    chunks_path = Path(chunks_path or config.INDEX_CHUNKS_PATH)
    vectors_path = Path(vectors_path or config.DENSE_VECTORS_PATH)
    meta_path = Path(meta_path or config.INDEX_META_PATH)

    chunk_bytes = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in index.chunks).encode("utf-8")
    _atomic_write_bytes(chunks_path, chunk_bytes)

    buf = io.BytesIO()
    np.save(buf, index.vectors, allow_pickle=False)
    _atomic_write_bytes(vectors_path, buf.getvalue())

    meta = {**index.meta, "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest()}
    _atomic_write_bytes(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    index.meta = meta
    return {p.name: p.stat().st_size for p in (chunks_path, vectors_path, meta_path)}


# --- Load with staleness checks ---------------------------------------------------------


def load_dense_index(
    embedder,
    chunk_cfg=None,
    manifest_path=None,
    dense_field=None,
    chunks_path=None,
    vectors_path=None,
    meta_path=None,
):
    """Load the saved index, raising StaleIndexError if anything disagrees."""
    chunk_cfg = chunk_cfg or ChunkConfig.from_config()
    manifest_path = manifest_path or config.MANIFEST_PATH
    dense_field = dense_field or config.DENSE_INDEX_FIELD
    chunks_path = Path(chunks_path or config.INDEX_CHUNKS_PATH)
    vectors_path = Path(vectors_path or config.DENSE_VECTORS_PATH)
    meta_path = Path(meta_path or config.INDEX_META_PATH)
    hint = "run: uv run python scripts/build_index.py"

    for p in (meta_path, chunks_path, vectors_path):
        if not p.exists():
            raise StaleIndexError(f"missing {p}; {hint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    expected = expected_fingerprint(embedder, chunk_cfg, manifest_path, dense_field)
    diffs = [k for k in expected if meta.get(k) != expected[k]]
    if diffs:
        detail = "\n".join(f"  {k}: index has {meta.get(k)!r}, now {expected[k]!r}" for k in diffs)
        raise StaleIndexError(f"index is stale ({', '.join(diffs)}); {hint}\n{detail}")

    chunk_bytes = chunks_path.read_bytes()
    if hashlib.sha256(chunk_bytes).hexdigest() != meta.get("chunks_sha256"):
        # e.g. `python -m ragbot.chunking --dump` with other settings overwrote it
        raise StaleIndexError(f"{chunks_path.name} changed since the index was built; {hint}")
    chunks = [json.loads(line) for line in chunk_bytes.decode("utf-8").splitlines() if line]

    try:
        vectors = np.load(vectors_path, allow_pickle=False)
    except ValueError as e:
        raise StaleIndexError(f"cannot read {vectors_path.name} ({e}); {hint}") from e
    if vectors.shape != (len(chunks), meta["dim"]) or len(chunks) != meta.get("n_chunks"):
        raise StaleIndexError(
            f"shape mismatch: vectors {vectors.shape}, chunks {len(chunks)}, "
            f"meta n_chunks {meta.get('n_chunks')}; {hint}"
        )

    # Same name and settings, but did the model weights change underneath?
    probe_now = l2_normalize(embedder.embed_passages([meta["probe_text"]]))[0]
    cosine = float(np.dot(probe_now, np.asarray(meta["probe_vector"], dtype=np.float32)))
    if cosine < config.EMBED_PROBE_MIN_COSINE:
        raise StaleIndexError(
            f"probe embedding changed (cosine {cosine:.5f} < {config.EMBED_PROBE_MIN_COSINE}); "
            f"the model differs from build time; {hint}"
        )
    return DenseIndex(chunks=chunks, vectors=vectors, meta=meta)
