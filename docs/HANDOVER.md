# Handover

_Last updated: 2026-11-17. Code is on GitHub (Kshat111/cc-rag-chatbot, `main`, 121 tests pass, README verified from a fresh clone).
`docs/` is currently **gitignored** (commit 2b038ca) and was restored from history on 2026-11-17; decide whether to track it._

## Done
- **Stages 1-2, corpus and chunking (D-002 to D-005):** 15 Wikipedia articles (218,602 chars) fetched with urllib, cleaned
  (reference sections, orphan labels, template artifacts). Heading-aware recursive chunks, sized in real bge-small tokens over
  exactly what is embedded: 248 chunks, 23 / 174 / 510 tokens (min / median / max, limit 512), size 512, overlap 64, min 64.
- **Stage 3, indexes (D-006 to D-009):** fastembed + BAAI/bge-small-en-v1.5 (384 d, model cached in `data/models/`);
  NumPy exact cosine index with staleness checks (settings, manifest hash, chunks hash, probe vector); own BM25
  (accent/apostrophe/number rules, model-name stems, single Japanese characters); weighted RRF hybrid (k 60, alpha 0.75,
  top-50 candidates). `retrieve.py`: `dense_search`, `bm25_search`, `hybrid_search` -> `Hit(chunk_id, score, rank)`.
- **Stage 4, evaluation (D-010, D-015):** 25 questions, 19 scored for retrieval (recall@3/5/10, MRR, gold from evidence quotes).
  MRR: dense .810, BM25 .634, hybrid(.75) .802; recall@10: hybrid 1.000, dense .947, BM25 .895. Chunk size 512 beats 256 and 128
  on every method; overlap 0/32/64/128 within noise. Saved in `eval/results/`.
- **Stages 5-6, generation (D-011 to D-013):** OpenAI-compatible client over urllib (9 provider presets in config, env overrides),
  follow-up rewriting from the last 5 turns, grounded prompt with chunk-id citations and one fixed refusal sentence,
  warnings for `empty`, `uncited` and `invalid_citations` answers, no-key fallback that prints the retrieved passages.
- **Stage 8, UI and README (D-014):** Gradio chat with search query, flags and cited sources; README verified from a fresh clone of
  the pushed `main` (sync, build 37 s, tests, ask without key, evaluate, UI HTTP 200).
- **Live testing, llama3.2 via Ollama (FAILURE F15-F24):** works: multi-turn rewrites, opening dates, refusals (TGV, Shanghai maglev,
  fare, Indian train). Fails: answer from memory with a fake supporting citation (F20), "ignore your rules" injection (F21),
  merged sections (F22), correct answer with the wrong chunk cited (F19, F24). Planted instruction inside a passage: not followed
  (2 of 2).
- **Git:** branch `main` = `stage-3-retrieval`, 7 tags (`stage-3a` ... `chunk-experiments`). History was rewritten once to remove a
  Claude co-author line from the first commit (commit messages only, file contents identical), so commit hashes in older notes no longer exist.

## In progress
- Nothing is half-built. Remaining before submission: the owner's pass over `DEFENSE.md`, and tracking and pushing `docs/` (see Next).

## Broken / known issues
- **Tables and infoboxes are dropped (D-002)**; q005/q006 are not scorable and ~20 chunks are lead-ins to missing tables (F1).
  The planned REST HTML fetch (D-002 revisit) was not done.
- **Small models ignore the grounding rules** in some cases (F19-F22, F24). The code only checks that cited ids exist; it cannot
  check that the cited chunk supports the sentence. No faithfulness or refusal-rate metric is built (Stage 7 of the plan).
- **Hybrid does not beat dense** on this eval (MRR .802 vs .810); alpha 0.75 was tuned on the same 19 questions. The pipeline default
  is hybrid; flipping it is a one-line config change (`RETRIEVAL_METHOD`). Live example where hybrid lost an answer dense had: F11, F16.
- **Eval limits:** 19 scored questions (one moves MRR by ~0.05), drafted partly after seeing retrieval output, q011 label too narrow (F14),
  q014-q025 drafted by Claude and approved by the owner 2026-11-17; q025 (Japan Rail Pass price) is a weak unanswerable question (the corpus mentions the pass, only the price is absent).
- **Only Ollama has been tested live.** OpenAI: key valid, account out of credit (HTTP 429). Groq preset model retired (HTTP 404).
  The other presets' model names are unverified.
- The app does not read `.env`; keys must be exported in the shell.
- Embedding takes ~27 s for 248 chunks on CPU; the index must be rebuilt after any chunk or corpus change.

## Next (in order)
1. **You:** go through `DEFENSE.md` (a draft exists, numbers checked against the repo on 2026-11-17) and make it your own words; be able to explain every line without notes. (D-010, D-012, D-013, D-015 and q014-q025 were approved by the owner on 2026-11-17.)
2. Decide whether `docs/` is tracked (remove the `docs/` line from `.gitignore`, commit, normal push).
3. Optional, with a working key or a bigger local model: re-run the same questions (the N700S, W7, "when did Shinkansen start?",
   injection, Chuo opening) on a stronger model for a before/after.
4. Optional: faithfulness / refusal-rate metric; fix q011's label and re-run D-010 and D-015; REST HTML tables (D-002 revisit);
   an app-side `.env` loader.

## Do not touch
- `.env` (never read or edit it from tooling).
- `uv.lock` by hand (regenerate with `uv lock`).
- `data/index/*` and `data/models/*` by hand (regenerate with `scripts/build_index.py`).
- `data/raw/*.txt` by hand (regenerate with `scripts/fetch_corpus.py`; manifest hashes will not match edits).
- `eval/questions.jsonl` evidence quotes: they must stay exact substrings of `data/raw/`.
