# Handover

_Last updated: 2026-11-04 (Stage 2 + review fixes)_

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

## In progress
- Nothing.

## Broken / known issues
- Pipeline modules other than ingest and chunking still raise NotImplementedError (by design).
- **Tables and infoboxes are dropped (D-002), and the eval depends on table data.** ~20 lead-in
  chunks point at missing tables (FAILURE.md F1); italics-dependent sentences lose meaning (F4).
  Eval q005/q006 are `answerable_now: false` until tables are fetched.
- 8 chunks stay under CHUNK_MIN_SIZE: single `##` sections with no neighbour in the same
  top-level section (merging never crosses `##`), plus the JNR lead.
- Overlap is whole sentences only: 12 of 32 split points have no overlap because the sentence
  at the boundary is longer than 64 est. tokens.
- Corpus is skewed: `shinkansen.txt` is 64.5k chars (~29% of the corpus), 18× the smallest (Hideo Shima, 3.6k).

## Next
- **Next change: revisit D-002, fetch REST HTML and convert tables to text (decided 2026-11-04).**
  Needs its own plan + DECISIONS entry. When it lands: remove the D-005 footnote rule and replace
  the exponent rule with `<sup>` rendering (D-005 "revisit if"); fill q005/q006 expected answers;
  re-check FAILURE.md F1, F2, F4; chunk ids in FAILURE.md will shift.
- Decide whether to rebalance the corpus (missing JR Hokkaido/Shikoku/Kyushu, Kyushu/Hokkaido Shinkansen; Chūō Shinkansen vs SCMaglev overlap).
- Stage 3 (index.py): **fail loudly if any chunk exceeds the embedding model's real input limit**
  (D-004 safeguard). Check the model's documented max length before choosing it:
  all-MiniLM-L6-v2 truncates at 256, which would cut 91/249 chunks (37%); prefer a 512-token
  BGE or E5 variant. The check must tokenize every chunk's `embed_text` with the model's own
  tokenizer, including special tokens and any required prefix (E5: "passage: "), not just
  compare CHUNK_SIZE: 12 chunks are ≥480 est. tokens and chars/4 may undercount Japanese text.
  Then measure the real chars-per-token ratio and update CHARS_PER_TOKEN or plug in the tokenizer.
  Decide whether BM25 indexes `text` or `embed_text`.
- Experiments planned in D-004: size 256/512/1024; overlap 0/32/64/128 at best size;
  heading prepend on/off; fixed-size baseline. Note: most sections fit in one chunk at 512,
  so size mainly affects the ~21 long sections, and fixed overlap is 25% at 256 vs 6% at 1024.
- Write the test set in eval/questions.jsonl. Avoid questions whose answers live only in tables.
- Choose and approve runtime dependencies (embeddings, BM25, LLM client, UI) and log each in DECISIONS.md.

## Do not touch
- .env
- uv.lock by hand (regenerate with `uv lock`).
- `data/raw/*.txt` by hand (regenerate with `scripts/fetch_corpus.py`; manifest hashes will not match edits).
