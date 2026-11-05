# Decisions

One entry per decision. Newest at the bottom.

## Template

### D-000: <title>
- **Date:**
- **Context:** what problem forced a choice
- **Options:** A / B / C with trade-offs
- **Choice:**
- **Why:**
- **Model:** which model/assistant helped (or "human")

---

### D-001: Use uv for dependency management
- **Date:** 2026-11-02
- **Context:** Need reproducible installs from a clean clone.
- **Options:** pip + requirements.txt / Poetry / uv
- **Choice:** uv, with uv.lock committed.
- **Why:** Fast, single tool for venv + lock + run; lockfile gives exact reproducibility.
- **Model:** Claude Opus 5.5 (scaffold)

### D-002: Wikipedia fetch method for the corpus
- **Date:** 2026-11-04
- **Context:** Need 12–15 Wikipedia articles as plain text with section
  headings preserved (the chunker uses them as context) and per-document
  provenance (page id, revision id, URL) for citations and reproducibility.
- **Options:**
  - A. MediaWiki Action API, TextExtracts (`prop=extracts&explaintext`):
    clean text, headings via `exsectionformat=wiki`, metadata and
    redirect/disambiguation detection in the same request. Drops tables
    and infoboxes.
  - B. REST API Parsoid HTML: full content including tables, but needs a
    hand-written HTML-to-text converter (~200 lines, more failure modes).
  - C. HTML scrape: fragile, against Wikipedia guidance. Rejected.
  - Client: stdlib urllib vs httpx/requests. 15 sequential GETs do not
    need a dependency.
- **Choice:** A, using stdlib `urllib.request`.
- **Why:** The official API returns clean text with headings and metadata
  in one request, so I avoided writing an HTML parser. The cost is that
  tables and infoboxes are dropped.
- **Trade-off:** Simplicity and an official, stable contract in exchange
  for losing table and infobox facts (e.g. train specs).
- **Cleaning:** Drop sections References, Notes, Citations, Sources,
  Bibliography, Further reading, External links, and **See also**.
  "See also" is a bare list of links with no prose, so it only adds noise
  to retrieval. Also drop headings with no text under them.
- **Revisit if:** eval questions depend on facts that appear only in
  tables or infoboxes, or the corpus grows past ~100 pages (then consider
  batching and an HTTP client with connection pooling).
- **Model:** Claude Opus 5.5 (proposed), human (approved)
- **Status (2026-11-04): revisit condition met.** Chunk review found ~20
  lead-in chunks whose tables were dropped (FAILURE.md F1), and the eval
  will depend on table data. Decision: move to option B (REST HTML with
  tables converted to text). To be designed as its own decision entry.

### D-003: Strip orphan table labels (consequence of D-002)
- **Date:** 2026-11-04
- **Context:** D-002 (TextExtracts) drops tables but keeps the label that
  introduced them. Four stations sections were left containing only
  "Legend:" or "Service column legend:", which would become chunks with a
  heading and no content.
- **Options:**
  - A. Strip a *trailing* line of 1–3 alphabetic words ending in ":" from
    each section; then the existing empty-section pass removes any heading
    left with nothing under it.
  - B. Strip *any* trailing line ending in ":", including lead-in sentences
    such as "The sets were formed as follows:".
- **Choice:** A. Limit set by `WIKI_BARE_LABEL_MAX_WORDS = 3` in config.py.
- **Why:** A bare label carries no information once its table is gone. A
  lead-in sentence can (e.g. "There are two standard-gauge lines not
  classified as Shinkansen"), so B would delete facts. A label followed by
  text (e.g. "Notes:" then real notes) is kept because it still introduces
  something.
- **Trade-off:** 7 orphan lead-in sentences and table footnotes
  ("† Indicates…", "* The sum of…") remain in the corpus.
- **Revisit if:** retrieval returns chunks dominated by those leftovers, or
  we switch to REST HTML (D-002 alternative), which would keep the tables.
- **Model:** Claude Opus 5.5 (proposed), human (approved)

### D-004: Chunking strategy
- **Date:** 2026-11-04
- **Context:** Cleaned Wikipedia text with Markdown headings, one paragraph
  per line. 309 sections: median 435 chars, 99 under 200 chars, max 5,181.
  Need chunks with correct heading paths for citations, and configurable
  sizes for the 256/512/1024 experiments.
- **Options:**
  - A. Fixed-size sliding window with overlap: simplest, predictable sizes,
    but cuts mid-sentence and mixes sections, so heading paths are wrong.
  - B. Heading-aware recursive (section, then line, then sentence, then
    word), greedy packing, overlap within a section only, small chunks
    merged within a top-level section.
  - C. Semantic chunking by embedding similarity: needs an embedding model
    not yet chosen; cuts are hard to explain.
- **Choice:** B. Size measured in estimated tokens, ceil(chars / 4), through
  a pluggable length function; no tokenizer dependency. Heading path is
  prepended to `embed_text`; raw `text` is kept separately. Overlap is a
  fixed count (64 tokens) of whole sentences, never across a heading.
- **Why:** The cleaned text already has headings and one paragraph per
  line, so I used them to keep each chunk under one correct heading path.
  Fixed-size (A) is my baseline. The estimator overcounts tokens, so
  chunks stay under the limit.
- **Trade-off:** Uneven chunk sizes and ~150 lines of code vs ~30; token
  counts are estimates (about ±25%) until a real tokenizer is plugged in.
- **Caveat for experiments:** with a fixed overlap of 64, overlap is 25% of
  a 256-token chunk but only 6% of a 1024-token one. The 256/512/1024 size
  experiment therefore changes size *and* relative overlap; do not read it
  as "size only". Overlap gets its own experiment (0, 32, 64, 128) at the
  best size.
- **Planned experiments (recall@k on eval/questions.jsonl):**
  1. Size 256 / 512 / 1024 (overlap 64).
  2. Overlap 0 / 32 / 64 / 128 at the best size.
  3. Heading-path prepend on vs off (`CHUNK_PREPEND_HEADINGS`).
  4. Baseline: fixed-size window (option A) at the best size.
- **Stage 3 safeguard:** `index.py` must fail loudly if `CHUNK_SIZE`
  exceeds the chosen embedding model's max input length (some models
  silently truncate at 256 tokens).
- **Revisit if:** retrieval misses answers that span sections (consider
  small-to-big retrieval), the embedding model's max input is below
  CHUNK_SIZE, or the estimator is off by more than 25% against the real
  tokenizer.
- **Model:** Claude Opus 5.5 (proposed), human (approved)

### D-005: Text hygiene for template artifacts
- **Date:** 2026-11-04
- **Context:** Chunk review (FAILURE.md F2, F3) found template artifacts that
  add noise to embeddings and BM25 terms: U+FEFF and duplicated decimals
  from the coordinate template, a flattened exponent ("×1012" for ×10^12),
  non-standard spaces (U+2009, U+202F), and table footnote lines whose
  table was dropped. A corpus survey found 2 U+FEFF, 11 odd spaces, 1
  coordinate, 1 exponent and exactly 3 footnote lines (all Shinkansen
  "Ridership").
- **Options:**
  - A. Narrow, pattern-specific rules in `clean_wikipedia_extract`, each
    matched to an observed artifact and tested.
  - B. Broad Unicode normalisation (NFKC) plus dropping any line that
    starts with a symbol. Catches more, but NFKC also rewrites real text
    (e.g. full-width Japanese punctuation, "km²"), and symbol-led lines may
    be real content.
  - C. Wait for the REST HTML fetch (D-002 revisit), which renders
    `<sup>` and tables properly.
- **Choice:** A. Rules: remove zero-width characters (U+FEFF, U+200B–U+200D,
  U+2060, U+00AD); map U+00A0, U+2009, U+202F to a normal space; keep the
  readable coordinate "35.583°N 138.927°E" but drop its duplicated decimal
  form and label; rewrite a decimal mantissa followed by "×10NN" as
  "×10^NN"; drop lines starting with a footnote marker listed in
  `WIKI_FOOTNOTE_MARKERS`.
- **Why:** Each rule fixes a measured artifact and nothing else, so its
  effect can be checked character by character. C would leave the noise in
  place for the Stage 3 experiments in the meantime.
- **Trade-off:** Narrow rules miss artifacts we have not seen yet. The
  exponent rule only fires after a decimal mantissa, so "3×100" is never
  touched.
- **Revisit if:** the REST HTML fetch lands. Footnotes become meaningful
  once their tables are back, so the footnote rule should then be removed,
  and the exponent rule replaced by rendering `<sup>`.
- **Model:** Claude Opus 5.5 (proposed), human (approved)
