# Handover

_Last updated: 2026-11-05 (Stage 3a, branch `stage-3-retrieval`, uncommitted)_

## Done
- Project scaffold: directory layout, stub modules, CLAUDE.md, docs templates.
- Stage 1a: Wikipedia corpus fetch (D-002).
  - `scripts/fetch_corpus.py` (`--check` resolves titles only; full run overwrites files and rebuilds the manifest).
  - `ragbot.ingest.clean_wikipedia_extract` + 5 tests in `tests/test_ingest_clean.py`.
  - Titles and fetch settings in `config.py` (`CORPUS_TITLES`, `WIKI_*`, `FETCH_*`).
  - 15 documents, 219,107 chars in `data/raw/`, plus `manifest.json`.
  - Redirects followed: Sanyo → San'yō, Tohoku → Tōhoku, Chuo → Chūō Shinkansen. No missing or disambiguation pages.
- D-003: trailing bare labels ("Legend:", "Service column legend:") stripped; sections left empty are removed.
  Removed 133 chars across 4 files; stations sections of Tokaido, Tōhoku and Hokuriku are gone. 7 tests in `test_ingest_clean.py`.
- Stage 2: chunking (D-004), heading-aware recursive.
  - `ragbot.chunking`: `ChunkConfig` (from config.py, overridable per experiment), `chunk_documents()`.
  - `ingest.load_documents()` reads manifest.json + files.
  - Stats: `uv run python -m ragbot.chunking [--dump]`; dump at `data/index/chunks.jsonl` (gitignored).
  - At 512/64/64: 250 chunks, est. tokens min 28 / median 196 / max 512. No sentence lost.
    At 256: 368 chunks; at 1024: 229 chunks.
  - 18 tests in `tests/test_chunking.py`.
- Chunk review recorded: `docs/FAILURE.md` F1–F8.
- `eval/questions.jsonl`: 13 questions. Types: conflict, near_duplicate, recency, keyword_list,
  table (q005–q006, blocked on tables), number, keyword, paraphrase, two_chunk, multi_turn
  (q011, with `turns` and `expected_rewrite` per turn), partial (q012), unanswerable (q013).
  q007–q013 carry `evidence`: exact quotes verified against data/raw/ (re-check after any re-fetch).
- D-005 text hygiene in `ingest.py`: zero-width chars removed, odd spaces normalised, coordinate
  duplicate dropped, "×1012" → "×10^12", table footnote lines dropped (`WIKI_FOOTNOTE_MARKERS`).
  Corpus now 218,602 chars (only shinkansen −462 and chuo −43 changed, same revisions);
  "### Passenger-kilometres" vanished because its only line was a footnote. 249 chunks. 32 tests passing.
- Initial commit `2cf5dac` on `main` (Stages 1–2). Stage 3 work is on branch `stage-3-retrieval`.
- **Stage 3a: dense index (D-006, D-007).**
  - Dependency `fastembed` 0.9.1 (Qdrant; onnxruntime, no PyTorch). Model `BAAI/bge-small-en-v1.5`
    (512 tokens, 384 dims, MIT), cached in `data/models/` (64 MB, gitignored).
  - `ragbot.index`: `Embedder` protocol + `FastEmbedEmbedder`; real-tokenizer check (`TokenLimitError`,
    nothing written on failure); `build_dense_index`, atomic `save_index`, `load_dense_index` with
    `StaleIndexError` (model, dims, max tokens, passage prefix, dense field, chunk_config,
    manifest sha256, chunks.jsonl sha256, shape, probe vector). Query instruction on by default
    (`EMBED_USE_QUERY_INSTRUCTION`); toggling it needs no rebuild.
  - `scripts/build_index.py`: 248 chunks → 248 × 384 float32; chunks.jsonl 624 KB, dense.npy 372 KB,
    meta 7 KB; ~30 s total, of which embedding ~28 s.
  - **D-004 result:** chars/4 undercounted up to +92%; one chunk was 552 real tokens, so the build
    failed loudly. Chunk size is now counted in `model_tokens` (bge tokenizer) over exactly what is
    embedded: passage prefix + heading path + text + [CLS]/[SEP]. `approx_tokens` stays as a fallback.
    Real tokens now min 23 / median 174 / max 510. Size experiments now 128/256/512 (1024 dropped).
  - FAILURE.md chunk ids refreshed (+ F9: silent truncation, mitigated). 56 tests pass, 1 opt-in
    real-model test (`RAGBOT_MODEL_TESTS=1`) passes.

## In progress
- Stage 3a done, **waiting for review**. Do not start 3b (BM25, retrieve.py, hybrid; D-008, D-009)
  until approved.

## Broken / known issues
- Pipeline modules other than ingest and chunking still raise NotImplementedError (by design).
- **Tables and infoboxes are dropped (D-002), and the eval depends on table data.** ~20 lead-in
  chunks point at missing tables (FAILURE.md F1); italics-dependent sentences lose meaning (F4).
  Eval q005/q006 are `answerable_now: false` until tables are fetched.
- 8 chunks stay under CHUNK_MIN_SIZE: single `##` sections with no neighbour in the same
  top-level section (merging never crosses `##`), plus the JNR lead.
- Overlap is whole sentences only: some split points get no overlap because the boundary sentence
  is longer than 64 tokens (counted at 12 of 32 under the old estimator; not re-measured).
- Embedding takes ~28 s for 248 chunks on CPU (Qdrant's ONNX file is optimised for GPU). Fine at
  this size; revisit if rebuilds become frequent (e.g. the size/overlap experiments).
- `python -m ragbot.chunking --dump` overwrites the index's chunks.jsonl; the index then reports
  itself stale until `build_index.py` is rerun (by design, D-007).
- Corpus is skewed: `shinkansen.txt` is 64.5k chars (~29% of the corpus), 18× the smallest (Hideo Shima, 3.6k).

## Next
- **Next change: revisit D-002, fetch REST HTML and convert tables to text (decided 2026-11-04).**
  Needs its own plan + DECISIONS entry. When it lands: remove the D-005 footnote rule and replace
  the exponent rule with `<sup>` rendering (D-005 "revisit if"); fill q005/q006 expected answers;
  re-check FAILURE.md F1, F2, F4; chunk ids in FAILURE.md will shift.
- Decide whether to rebalance the corpus (missing JR Hokkaido/Shikoku/Kyushu, Kyushu/Hokkaido Shinkansen; Chūō Shinkansen vs SCMaglev overlap).
- **Stage 3b (after 3a review):** own BM25 (D-008; tokenization: accent stripping, apostrophe
  removal, number normalisation, model-name stems, Japanese single characters), `BM25_INDEX_FIELD`
  switch (default `embed_text`), `retrieve.py` with `Hit(chunk_id, score, rank)`, weighted RRF
  hybrid (D-009, k=60, alpha 0.5). Then five example queries per method.
- Experiments planned in D-004: size 128/256/512 (384 optional); overlap 0/32/64/128 at best
  size; heading prepend on/off; query instruction on/off; fixed-size baseline. Fixed overlap 64
  is 50% of a 128-token chunk and 12.5% of 512, so the size experiment is not "size only".
- Write the test set in eval/questions.jsonl. Avoid questions whose answers live only in tables.
- Choose and approve the remaining runtime dependencies (LLM client, UI) and log each in DECISIONS.md.

## Do not touch
- .env
- uv.lock by hand (regenerate with `uv lock`).
- `data/index/*` and `data/models/*` by hand (regenerate with `scripts/build_index.py`).
- `data/raw/*.txt` by hand (regenerate with `scripts/fetch_corpus.py`; manifest hashes will not match edits).
