"""Retrieval evaluation (Stage 4): recall@k and MRR for dense, BM25 and hybrid.

Usage: uv run python scripts/evaluate.py [--no-save]

Gold chunks are found by matching each question's `evidence` quotes against
chunk text (whitespace-normalised), not by chunk id, so the eval survives
re-chunking and re-fetching. Each quote forms one "evidence group": the set of
chunks containing it (overlap can put a quote in two chunks).

  recall@k = share of a question's evidence groups with a chunk in the top k
  MRR      = 1 / rank of the first chunk that belongs to any evidence group

Questions without evidence or with answerable_now == false (tables,
unanswerable) are skipped here; they belong to the generation metrics.
Multi-turn questions are retrieved with the last turn's expected_rewrite, which
isolates retrieval from the rewrite step (Stage 5 tests the rewrite itself).
"""

import argparse
import json
import statistics
import sys
from datetime import date, datetime, timezone

from ragbot import config
from ragbot.index import StaleIndexError
from ragbot.retrieve import Retriever

KS = (3, 5, 10)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def norm(s):
    return " ".join(s.split())


def load_questions():
    with open(config.EVAL_QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def query_for(q):
    return q["turns"][-1]["expected_rewrite"] if q.get("turns") else q["question"]


def evidence_groups(q, chunks):
    groups = []
    for quote in q.get("evidence", []):
        groups.append({c["chunk_id"] for c in chunks if norm(quote) in norm(c["text"])})
    return groups


def score(ranked_ids, groups, k):
    top = set(ranked_ids[:k])
    return sum(1 for g in groups if g & top) / len(groups)


def mrr(ranked_ids, groups):
    gold = set().union(*groups)
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1 / i
    return 0.0


def evaluate(retriever, questions, searchers):
    """searchers: name -> fn(query, k) -> list[Hit]. Returns (rows, skipped)."""
    rows, skipped = [], []
    for q in questions:
        groups = evidence_groups(q, retriever.chunks) if q.get("evidence") else []
        if q.get("answerable_now") is False or not groups:
            skipped.append((q["id"], q["type"], "not answerable_now" if q.get("answerable_now") is False else "no evidence"))
            continue
        if any(not g for g in groups):
            raise SystemExit(f"{q['id']}: an evidence quote matches no chunk; re-check the quote or the chunking")
        row = {"id": q["id"], "type": q["type"], "query": query_for(q), "gold": sorted(set().union(*groups)), "methods": {}}
        for name, fn in searchers.items():
            ids = [h.chunk_id for h in fn(row["query"], max(KS))]
            first = next((i for i, c in enumerate(ids, 1) if c in row["gold"]), None)
            row["methods"][name] = {
                "first_gold_rank": first,
                "mrr": mrr(ids, groups),
                **{f"recall@{k}": score(ids, groups, k) for k in KS},
            }
        rows.append(row)
    return rows, skipped


def summarize(rows, name):
    out = {"n": len(rows), "mrr": statistics.mean(r["methods"][name]["mrr"] for r in rows)}
    for k in KS:
        out[f"recall@{k}"] = statistics.mean(r["methods"][name][f"recall@{k}"] for r in rows)
    return out


def table(rows, names):
    head = f"{'method':<14}" + "".join(f"{f'recall@{k}':>10}" for k in KS) + f"{'MRR':>8}"
    lines = [head, "-" * len(head)]
    for n in names:
        s = summarize(rows, n)
        lines.append(f"{n:<14}" + "".join(f"{s[f'recall@{k}']:>10.3f}" for k in KS) + f"{s['mrr']:>8.3f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    try:
        r = Retriever()
    except StaleIndexError as e:
        print(f"Index not usable: {e}", file=sys.stderr)
        return 1

    questions = load_questions()
    searchers = {"dense": r.dense_search, "bm25": r.bm25_search}
    for a in ALPHAS:
        searchers[f"hybrid a={a}"] = lambda q, k, a=a: r.hybrid_search(q, k, alpha=a)
    rows, skipped = evaluate(r, questions, searchers)
    names = list(searchers)

    print(f"{len(questions)} questions; {len(rows)} scored, {len(skipped)} skipped "
          f"({', '.join(f'{i} {why}' for i, _, why in skipped)})")
    print(f"Index: {r.index.meta['n_chunks']} chunks, {r.index.meta['embedding_model']}, "
          f"query instruction {'on' if r.embedder.query_prefix else 'off'}, BM25 field {r.bm25_field}, "
          f"RRF k={config.RRF_K}, candidates {config.HYBRID_CANDIDATES}\n")
    print(table(rows, names))
    types = sorted({x["type"] for x in rows})
    print("\nBy question type, MRR (n in brackets):")
    cols = ["dense", "bm25", "hybrid a=0.5"]
    print(f"{'type':<16}{'n':>3}" + "".join(f"{c:>15}" for c in cols))
    for t in types:
        sub = [x for x in rows if x["type"] == t]
        print(f"{t:<16}{len(sub):>3}" + "".join(f"{summarize(sub, c)['mrr']:>15.3f}" for c in cols))
    print("\nRank of the first gold chunk per question (- = not in top 10):")
    print(f"{'id':<6}{'type':<16}" + "".join(f"{c:>15}" for c in cols))
    for x in rows:
        print(f"{x['id']:<6}{x['type']:<16}" + "".join(
            f"{(x['methods'][c]['first_gold_rank'] or '-'):>15}" for c in cols))

    if not args.no_save:
        config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = config.EVAL_RESULTS_DIR / f"{date.today().isoformat()}-retrieval"
        result = {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {k: v for k, v in r.index.meta.items() if k in ("embedding_model", "chunk_config", "dense_field", "n_chunks")}
            | {"bm25_field": r.bm25_field, "bm25_k1": config.BM25_K1, "bm25_b": config.BM25_B, "rrf_k": config.RRF_K,
               "hybrid_candidates": config.HYBRID_CANDIDATES, "query_prefix": r.embedder.query_prefix},
            "n_scored": len(rows), "skipped": skipped,
            "summary": {n: summarize(rows, n) for n in names},
            "per_question": rows,
        }
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        stem.with_suffix(".md").write_text(
            f"# Retrieval results {date.today().isoformat()}\n\n{len(rows)} scored questions, "
            f"{r.index.meta['n_chunks']} chunks.\n\n```\n{table(rows, names)}\n```\n", encoding="utf-8")
        print(f"\nSaved {stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
