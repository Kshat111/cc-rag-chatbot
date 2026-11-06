"""Indexing entry point (stage 3a): load docs -> chunk -> token check -> embed -> save.

Usage: uv run python scripts/build_index.py

Re-chunks from data/raw/ on every run, so the index always reflects the
current config. Exits 1 without writing anything if any chunk exceeds the
embedding model's real token limit. BM25 is not saved: retrieve.Retriever
rebuilds it from chunks.jsonl at load (D-008).
"""

import sys
import time

from ragbot import config
from ragbot.chunking import ChunkConfig, chunk_documents
from ragbot.index import FastEmbedEmbedder, TokenLimitError, build_dense_index, save_index
from ragbot.ingest import load_documents


def main():
    t0 = time.perf_counter()
    docs = load_documents()
    cfg = ChunkConfig.from_config()
    chunks = chunk_documents(docs, cfg)
    t_chunk = time.perf_counter() - t0
    print(f"Documents: {len(docs)}   Chunks: {len(chunks)}   (chunk config {cfg.size}/{cfg.overlap}/{cfg.min_size})")

    t = time.perf_counter()
    embedder = FastEmbedEmbedder()
    t_load = time.perf_counter() - t
    print(f"Model: {embedder.name}   dim {embedder.dim}   limit {embedder.max_tokens} tokens   "
          f"dense field: {config.DENSE_INDEX_FIELD}")
    print(f"Query instruction: {'on' if embedder.query_prefix else 'off'}")

    try:
        index = build_dense_index(chunks, embedder, cfg)
    except TokenLimitError as e:
        print(f"\nFAILED token check: {e}", file=sys.stderr)
        return 1

    sizes = save_index(index)
    m = index.meta
    ts = m["token_stats"]
    print(f"\nReal tokens (incl. [CLS]/[SEP]): min {ts['min']}  median {ts['median']:.0f}  "
          f"max {ts['max']} / limit {ts['limit']}")
    print(f"Measured chars per token: {ts['chars_per_token']} (estimate assumes {config.CHARS_PER_TOKEN})")
    print(f"chars/4 estimate undercounted {ts['estimate_undercounted']}/{m['n_chunks']} chunks "
          f"(worst by {ts['max_estimate_error_pct']}%)")
    print(f"\nVectors: {index.vectors.shape[0]} x {index.vectors.shape[1]} {index.vectors.dtype}")
    for name, size in sizes.items():
        print(f"  {name:<18} {size / 1024:8.1f} KB")
    total = time.perf_counter() - t0
    print(f"\nTimings: chunking {t_chunk:.2f}s   model load {t_load:.2f}s   "
          f"token check {m['timings']['token_check_s']:.2f}s   embedding {m['timings']['embed_s']:.2f}s   "
          f"total {total:.2f}s")
    print(f"Versions: {m['versions']}")
    print(f"Wrote index to {config.INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
