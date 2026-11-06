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
  1. Size 128 / 256 / 512 (384 optional), overlap 64. *Changed 2026-11-05:*
     1024 dropped because bge-small-en-v1.5 (D-006) is capped at 512 input
     tokens, so a 1024-token chunk cannot be embedded; the token check would
     stop the build. A long-context comparison (e.g. nomic-embed-text-v1.5)
     is future work and would change the model as well as the size. With a
     fixed overlap of 64, overlap is 50% of a 128-token chunk and 12.5% of 512.
  2. Overlap 0 / 32 / 64 / 128 at the best size (128 only valid for sizes > 128).
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
- **Result (2026-11-05):** revisit condition met. Measured against the
  bge-small tokenizer: 4.30 chars/token overall (so chars/4 overcounts on
  average, median chunk −9%), but the worst case is +92% (Japanese-heavy
  lists in JR Central affiliates), 64 of 249 chunks were undercounted, and
  1 chunk (shinkansen#0020, Proposed lines) was 552 tokens, over the 512
  limit; the D-007 token check stopped the build. Fix: new unit
  `model_tokens` counts with the embedding model's own tokenizer, and
  CHUNK_SIZE now bounds exactly what is embedded (passage prefix + heading
  path + text + [CLS]/[SEP]). chars/4 stays as the fallback unit
  `approx_tokens`. After: 248 chunks, real tokens min 23 / median 174 /
  max 510, none over 512. Model: Claude Opus 5.5 (claude-opus-5-5).

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

### D-006: Embedding model and runtime
- **Date:** 2026-11-05
- **Context:** Chunks reach 512 estimated tokens (12 are ≥480), and chars/4
  may undercount Japanese text. Need a model with a ≥512-token input limit,
  an English corpus with Japanese names in parentheses, CPU only, small install.
- **Options (model):**
  - A. BAAI/bge-small-en-v1.5: 512 tokens, 384 dims, MIT, 0.067 GB ONNX.
  - B. BAAI/bge-base-en-v1.5: 512 tokens, 768 dims, MIT, 0.21 GB ONNX.
  - C. intfloat/multilingual-e5-small: 512 tokens, 384 dims, MIT; handles
    Japanese, but needs PyTorch (not in fastembed) and required prefixes.
  - Rejected: all-MiniLM-L6-v2 (max_seq_length 256 would cut 91/249 chunks);
    nomic-embed-text-v1.5 (8192 tokens, but 0.52 GB and custom architecture,
    not needed while chunks are ≤512).
- **Options (runtime):**
  - fastembed (Qdrant, Apache-2.0): onnxruntime, no PyTorch, ~32 packages /
    ~52 MB download.
  - sentence-transformers (Hugging Face, Apache-2.0): PyTorch, ~41 packages /
    ~166 MB download; original fp32 weights, every Hub model, fine-tuning.
- **Choice:** A on fastembed. Query instruction "Represent this sentence for
  searching relevant passages: " on by default (`EMBED_USE_QUERY_INSTRUCTION`),
  no passage prefix. Model files cached in `data/models/` (gitignored); the
  one-time download from huggingface.co is approved.
- **Why:** Smallest model that meets the 512-token requirement, MIT licence,
  no PyTorch. The corpus and questions are English; exact matches on Japanese
  names are BM25's job.
- **Trade-off:** Qdrant's optimised ONNX conversion differs slightly from the
  original weights. fastembed truncates at 512 silently (guarded by the
  real-tokenizer check in D-007) and pulls in unused pillow, sympy and mmh3.
  Its default cache is the system temp dir, so the cache dir is set explicitly.
- **Revisit if:** paraphrase questions have low recall@k (try B, one config
  line), Japanese queries are needed (C), or chunks must exceed 512 tokens.
- **Model:** Claude Opus 5.5 (claude-opus-5-5) (proposed), human (approved)

### D-007: Dense index, persistence and staleness
- **Date:** 2026-11-05
- **Context:** 249 chunks × 384 dims. The index must not silently disagree
  with the corpus, chunk settings or model that produced it, and must never
  hold a vector for a truncated chunk.
- **Options:**
  - A. NumPy float32 matrix, L2-normalised, exact cosine by dot product.
  - B. FAISS IndexFlatIP: same exact search, extra dependency.
  - C. Approximate search (HNSW/IVF): only pays off around 10⁵–10⁶ vectors.
- **Choice:** A. Saved to `data/index/`: `chunks.jsonl` (row order),
  `dense.npy`, `index_meta.json` (written last, all files written atomically).
  Before embedding, every chunk's indexed text (passage prefix + field) is
  tokenized with the model's own tokenizer, special tokens included and
  truncation off; any chunk over 512 aborts the build and nothing is written.
  At load, a stale index raises `StaleIndexError` naming what differs:
  model, dims, max tokens, passage prefix, dense field, chunk_config,
  manifest.json sha256, chunks.jsonl sha256, row count, or a probe sentence
  whose embedding no longer matches (catches a silently updated model, since
  fastembed does not pin a model revision).
- **Why:** At this size exact search is sub-millisecond and needs nothing
  beyond numpy (already pulled in by fastembed). The checks turn "silently
  wrong" into "loud error".
- **Trade-off:** No approximate search; any change to config or corpus
  requires a rebuild (seconds at this size). The query instruction is
  recorded but not part of the staleness check, because it does not change
  stored vectors, so the on/off experiment needs no rebuild.
- **Revisit if:** more than ~100k vectors or query latency over budget.
- **Model:** Claude Opus 5.5 (claude-opus-5-5) (proposed), human (approved)

### D-008: BM25 implementation and tokenization
- **Date:** 2026-11-06
- **Context:** Dense search (D-006) is weak on exact identifiers: model names
  (`N700A` in 14 chunks, `N700S` in 7, `700T` in 1), years, speeds, and
  romanised names written two ways ("Tōkaidō"/"Tokaido", "San'yō"/"Sanyo").
  36 of 248 chunks contain Japanese characters (902 in total), almost all in
  parenthesised names. 95 chunks contain apostrophes. No new dependency.
- **Options:**
  - A. Own Okapi BM25 (~100 lines, numpy only, already installed).
  - B. `rank-bm25` (Apache-2.0): last release 2022; still needs our tokenizer.
  - C. `bm25s`: active, but PyPI shows no licence; still needs our tokenizer.
  - Japanese text: C1. one token per character; C2. character bigrams
    (Lucene CJKAnalyzer style); C3. a morphological analyser (new dependency).
  - Model-name stems: S1. added on both index and query side; S2. added on
    the index side only.
- **Choice:** A with C1 and S2.
  - Okapi BM25, `k1=1.5`, `b=0.75`, Lucene IDF `ln(1 + (N − df + 0.5)/(df + 0.5))`
    (never negative). Built in memory from `chunks.jsonl` at load, not saved.
  - Indexed field set by `BM25_INDEX_FIELD` (default `embed_text`, same as dense).
  - Tokenizer, applied the same way to chunks and queries except for stems:
    1. NFKD, drop combining marks, lowercase (`Tōkaidō` → `tokaido`). Latin
       characters only: on Japanese, NFKD would turn ガ into カ + a mark.
    2. Drop a possessive `'s` (`Shinkansen's` → `shinkansen`), then remove other
       apostrophes `'` `’` `ʼ` before splitting (`San'yō` → `sanyo`).
    3. Remove commas between digits (`1,435` → `1435`); keep decimals (`5.3`).
    4. Split on everything else that is not a letter or digit
       (`Shin-Osaka` → `shin`, `osaka`; `km/h` → `km`, `h`).
    5. Hiragana, katakana and CJK ideographs: one token per character.
    6. Index side only: a token of optional letters + digits + letters also
       adds its letters-plus-digits stem (`n700a` → `n700a`, `n700`;
       `700t` → `700t`, `700`; `1960s` → `1960s`, `1960`). A query keeps
       its token as typed, so "N700" finds N700A and N700S, but "N700S" does
       not find N700A, and "N700" never matches a bare "700".
    7. No stopwords, no stemming.
    8. Each distinct query term counts once ("N700 N700" = "N700").
- **Why:** The tokenizer is the real work and a library would still need it.
  Stems on the index side only keep queries precise: a query for one variant
  doesn't pull in its siblings, and a query for the family finds all of them.
  Single characters need no dependency; IDF makes characters common across
  the Japanese names (新 appears in 16 chunks, 線 in 14) cheap, and the
  rare ones carry the match.
- **Trade-off:** Single characters ignore word boundaries, so 東海道 also
  partly matches any chunk containing 東 or 海. A query "700T" won't match
  a chunk that only says "700 Series". "1st"/"5th" add bare-number stems that
  can match unrelated numbers in a query (e.g. "5 minutes" vs "5th"). No
  stemming, so "train" doesn't match "trains" in BM25 (dense covers that).
- **Implementation notes (2026-11-06):** rules 1, 2 and 8 refined while coding
  (Latin-only accent stripping, possessive handling, unique query terms); 27 tests
  in `tests/test_retrieve.py`.
- **First observation (2026-11-06, 5 hand queries, not the eval):** BM25 alone
  ranks the answer first for exact-term questions ("actual construction cost" →
  shinkansen#0006 "actual cost was about 380 billion yen"; "700T" →
  shinkansen#0053; "N700S" → all top 3 contain N700S), and misses the paraphrase
  "long pointed noses" (the text says "elongated … nose profiles").
- **Revisit if:** Japanese queries become a requirement or return noisy hits
  (try C2), the eval shows misses on inflected forms (add a light stemmer),
  or the corpus passes ~50k chunks (scipy sparse matrices or B/C).
- **Model:** Claude Opus 5.5 (claude-opus-5-5) (proposed), human (approved)

### D-009: Hybrid fusion
- **Date:** 2026-11-06
- **Context:** Dense (bge-small-en-v1.5 cosine, bounded, typically a narrow
  band; to be measured on real queries) and BM25 (unbounded, grows with query
  length) scores are not on the same scale. The experiments need a dense-only to BM25-only dial.
- **Options:**
  - A. Weighted reciprocal rank fusion:
    `score = α/(k + rank_dense) + (1 − α)/(k + rank_bm25)`.
  - B. Weighted sum of per-query min-max normalised scores.
  - C. Plain RRF (no weight).
- **Choice:** A, with `RRF_K = 60` (Cormack et al., 2009), `HYBRID_ALPHA = 0.5`,
  over the top `HYBRID_CANDIDATES = 50` from each method. A chunk missing from
  one method's candidates gets 0 from that method. Chunks with a total score
  of 0 are dropped. Ties break by chunk order, so results are deterministic.
  At α = 0.5 the ranking equals plain RRF; α = 1 reproduces dense-only order
  and α = 0 BM25-only order.
- **Why:** Ranks need no normalisation, so α means the same thing for every
  query. One constant, the standard value.
- **Trade-off:** Throws away how confident each method is: a BM25 hit far
  ahead of the rest counts the same as a narrow win. At α = 0.5 hybrid scores
  run from 0.0045 (rank 50 in one method only) to 0.0164 (rank 1 in both) and
  are only comparable within one query.
- **First observation (2026-11-06, 5 hand queries, not the eval):** hybrid kept
  the right chunk in its top 3 for the cost and 700T questions, but **lost two
  answers dense had at rank 2**: "pointed noses" (shinkansen#0001, BM25 rank >3)
  and "When will the Chuo Shinkansen open?" (chuo-shinkansen#0000). RRF favours
  chunks that are moderately good in both lists over one method's strong hit.
  Whether that costs recall overall is for the Stage 4 eval to decide (alpha sweep).
- **Revisit if:** the eval shows one method confidently right while RRF
  dilutes it (try B), or the best α found by the experiments sits at 0 or 1
  (then hybrid isn't paying for itself).
- **Model:** Claude Opus 5.5 (claude-opus-5-5) (proposed), human (approved)

### D-010: Retrieval evaluation metrics and gold labels
- **Date:** 2026-11-06
- **Context:** The brief needs at least one metric and a comparison of two retrieval
  approaches. The eval set has 25 questions; 19 are scorable for retrieval.
- **Options:**
  - A. Gold = chunk ids written by hand per question.
  - B. Gold = chunks whose text contains the question's `evidence` quote.
  - C. LLM judge decides relevance.
- **Choice:** B, with recall@3/5/10 and MRR (`scripts/evaluate.py`). One evidence quote =
  one group of chunks (a quote can sit in two overlapping chunks); recall@k is the share
  of groups hit in the top k; MRR uses the first chunk in any group. Multi-turn questions
  are retrieved with the last turn's `expected_rewrite`, so retrieval is measured apart
  from rewriting. Skipped: `answerable_now: false` (tables, unanswerable), which belong to
  generation metrics. Results saved to `eval/results/<date>-retrieval.{json,md}`.
- **Why:** Ids change whenever chunking changes (size, overlap, tables), quotes do not,
  so the same eval compares chunk configurations. No judge to be wrong.
- **Trade-off:** A quote that is paraphrased in another chunk is not counted as gold
  (understates recall); two quotes in one chunk count as separate groups.
  19 questions: one question moves MRR by about 0.05, so differences under ~0.05 are noise.
  Questions q014–q025 and the evidence for q001–q004 were drafted by Claude **after** seeing
  five hand-run retrievals (q007–q011), so the set is not blind to the system; quotes are
  machine-checked against data/raw, answers still need the owner's review.
- **Result (2026-11-06, 248 chunks, bge-small, query instruction on):**
  dense MRR 0.810, recall@3/5/10 = .798/.868/.947; BM25 0.634, .684/.816/.895;
  hybrid α=0.5 0.765, .816/.816/1.000; hybrid α=0.75 0.802, .746/.868/1.000.
  Dense wins MRR; hybrid reaches recall@10 = 1.0 (nothing lost entirely) but pushes
  exact hits down. BM25 wins the two-chunk question (q010, rank 1 vs dense 7) and loses
  paraphrases (q009, q020 not in its top 10).
- **Revisit if:** the eval grows past ~40 questions, quotes stop matching after re-fetching
  tables, or generation metrics need chunk-level gold ids.
- **Model:** Claude Sonnet 5.5 (claude-sonnet-5-5) (proposed), human (approved)
