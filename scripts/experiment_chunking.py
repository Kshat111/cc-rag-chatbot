"""Chunk size and overlap experiments (D-015). Builds every configuration in memory only:
data/index/ is never touched.

Usage: uv run python scripts/experiment_chunking.py [--no-save]

1. Size sweep (config.CHUNK_SIZE_EXPERIMENTS) at the configured overlap.
2. Overlap sweep (config.CHUNK_OVERLAP_EXPERIMENTS) at the configured size and at the best size of step 1.

Gold chunks are found by evidence quotes (see evaluate.py). A quote that no single chunk contains
(the sentence was split across chunks) cannot be scored, so every table is computed on the
questions whose evidence fits in one chunk in ALL compared configurations; the number of
questions lost per configuration is reported, because that loss is itself a cost of small chunks.
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import KS, evaluate, evidence_groups, load_questions, summarize  # noqa: E402

from ragbot import config  # noqa: E402
from ragbot.chunking import ChunkConfig, chunk_documents  # noqa: E402
from ragbot.index import FastEmbedEmbedder, build_dense_index  # noqa: E402
from ragbot.ingest import load_documents  # noqa: E402
from ragbot.retrieve import Retriever  # noqa: E402

METHODS = {"dense": lambda r: r.dense_search, "bm25": lambda r: r.bm25_search,
           "hybrid": lambda r: r.hybrid_search}


def build(docs, embedder, size, overlap):
    cfg = ChunkConfig.from_config(size=size, overlap=overlap)
    chunks = chunk_documents(docs, cfg)
    index = build_dense_index(chunks, embedder, cfg)  # in memory; never saved
    return Retriever(embedder=embedder, index=index), chunks


def scorable_ids(questions, chunks):
    ok = set()
    for q in questions:
        if q.get("answerable_now") is False or not q.get("evidence"):
            continue
        if all(evidence_groups(q, chunks)):
            ok.add(q["id"])
    return ok


def run(docs, embedder, questions, configs, label):
    """configs: list of (size, overlap). Returns (results, common_n)."""
    built = {c: build(docs, embedder, *c) for c in configs}
    per_cfg_ok = {c: scorable_ids(questions, built[c][1]) for c in configs}
    common = set.intersection(*per_cfg_ok.values())
    subset = [q for q in questions if q["id"] in common]
    results = []
    for c in configs:
        r, chunks = built[c]
        rows, _ = evaluate(r, subset, {m: f(r) for m, f in METHODS.items()})
        lengths = sorted(ch["n_tokens"] for ch in chunks)
        results.append({
            "size": c[0], "overlap": c[1], "n_chunks": len(chunks), "median_tokens": lengths[len(lengths) // 2],
            "questions_lost": len(set().union(*per_cfg_ok.values()) - per_cfg_ok[c]),
            "metrics": {m: summarize(rows, m) for m in METHODS},
        })
    return {"label": label, "n_questions": len(subset), "common_ids": sorted(common), "configs": results}


def table(block):
    head = f"{'size/overlap':<13}{'chunks':>7}{'lost':>5} " + "".join(
        f"{m + ' R@5':>11}{m + ' MRR':>11}" for m in METHODS) + f"{'hybrid R@10':>13}"
    lines = [f"{block['label']} (scored on {block['n_questions']} questions common to all configs)", head, "-" * len(head)]
    for r in block["configs"]:
        m = r["metrics"]
        lines.append(f"{str(r['size']) + '/' + str(r['overlap']):<13}{r['n_chunks']:>7}{r['questions_lost']:>5} " + "".join(
            f"{m[x]['recall@5']:>11.3f}{m[x]['mrr']:>11.3f}" for x in METHODS) + f"{m['hybrid']['recall@10']:>13.3f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    docs, questions, embedder = load_documents(), load_questions(), FastEmbedEmbedder()

    size_cfgs = [(s, config.CHUNK_OVERLAP) for s in config.CHUNK_SIZE_EXPERIMENTS]
    blocks = [run(docs, embedder, questions, size_cfgs, "Size sweep, overlap fixed")]
    print(table(blocks[0]), flush=True)

    best = max(blocks[0]["configs"], key=lambda r: r["metrics"]["hybrid"]["mrr"])["size"]
    for size in dict.fromkeys([config.CHUNK_SIZE, best]):
        ov = [(size, o) for o in config.CHUNK_OVERLAP_EXPERIMENTS if o < size]
        blocks.append(run(docs, embedder, questions, ov, f"Overlap sweep at size {size}"))
        print("\n" + table(blocks[-1]), flush=True)

    if not args.no_save:
        config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = config.EVAL_RESULTS_DIR / f"{date.today().isoformat()}-chunking"
        stem.with_suffix(".json").write_text(json.dumps(
            {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": embedder.name,
             "min_size": config.CHUNK_MIN_SIZE, "blocks": blocks}, indent=2) + "\n")
        stem.with_suffix(".md").write_text(f"# Chunking experiments {date.today().isoformat()}\n\n```\n" +
                                          "\n\n".join(table(b) for b in blocks) + "\n```\n")
        print(f"\nSaved {stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
