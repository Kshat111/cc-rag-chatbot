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
- **Change (2026-11-06, after the first eval, D-010):** `HYBRID_ALPHA` 0.5 -> 0.75. On 19
  questions α=0.75 matches dense on recall@5 (.868) and MRR within noise (.802 vs .810) and
  keeps recall@10 = 1.0, while α=0.5 loses MRR (.765). **Tuned on the same questions it is
  judged on**, so the gain may not generalise; the eval still sweeps 0, .25, .5, .75, 1.
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

### D-011: LLM provider and client
- **Date:** 2026-11-06
- **Context:** The reviewers run the project from the README with whatever LLM access they
  have; the brief names no provider. The user wants OpenAI as the default, with local models
  and other providers supported.
- **Options:**
  - A. A thin client for the OpenAI-compatible `/chat/completions` API over `urllib`, with
    named provider presets in config.
  - B. The official `openai` SDK (also works with other base URLs).
  - C. One SDK per provider (`openai`, `anthropic`, `ollama`), or LiteLLM as an adapter.
- **Choice:** A. Presets in `config.LLM_PROVIDERS`: `openai` (default, `gpt-4o-mini`,
  `OPENAI_API_KEY`), `ollama` (local, no key), `groq`, `openrouter`. Switch with
  `RAGBOT_LLM_PROVIDER`; override `RAGBOT_LLM_BASE_URL`, `RAGBOT_LLM_MODEL`,
  `RAGBOT_LLM_API_KEY_ENV` for anything else (vLLM, LM Studio, Azure-style gateways). Retries 429/5xx
  and network errors (2/4/8 s), no retry on other 4xx. Temperature 0. With a keyed provider
  and no key, the pipeline does not fail: it returns the retrieved passages with a notice.
- **Why:** About 60 lines, no dependency to approve or lock, and the same code serves every
  OpenAI-compatible server, including local ones. Keys are read from the environment by name
  only; nothing in config.py holds a secret.
- **Trade-off:** No streaming, no native Anthropic/Gemini protocol (use their OpenAI-compatible
  endpoints, or OpenRouter). Model names in presets can go stale and are not validated; a wrong
  name shows up as an HTTP 404 with the provider's message. Not tested against a live
  endpoint in the build environment (no key, no Ollama): the HTTP layer is tested with a
  fake transport only.
- **Revisit if:** streaming or tool calling is needed, a provider deviates from the
  OpenAI schema, or the retry logic grows beyond what 60 lines can hold (then adopt the SDK).
- **Model:** Claude Sonnet 5.5 (claude-sonnet-5-5) (proposed), human (approved)

### D-012: Multi-turn: rewrite follow-ups into standalone queries
- **Date:** 2026-11-06
- **Context:** The retriever has no memory: "Why did he resign?" retrieves nothing useful.
- **Options:**
  - A. An LLM call rewrites the last question into a standalone query using the last turns.
  - B. Send the raw history (or the last N user turns) to the retriever as the query.
  - C. A running summary of the conversation.
- **Choice:** A. Last `MAX_HISTORY_TURNS` = 5 turns, oldest dropped first beyond
  `MAX_HISTORY_CHARS` = 4000. The rewrite prompt says: replace references with names, and
  return the question unchanged if it is already standalone or changes topic. The rewrite is
  only used for **search**; the answering prompt gets the original question, the rewrite as a
  note, and the prior turns as chat messages. A rewrite that is empty, multi-line, or more
  than 3× the question length is discarded, and any LLM error falls back to the raw question.
  The rewritten query is shown in the CLI.
- **Why:** B drags old topics into the query (a topic switch retrieves the previous topic) and
  dilutes the embedding; C adds a second thing to keep correct. A costs one short LLM call
  per follow-up (none for the first question).
- **Trade-off:** One extra LLM call and latency per follow-up. The rewriter can resolve a
  pronoun wrongly (it picks the wrong entity from the history) and nothing detects that; the
  original question stays in the answer prompt as a partial guard. History is stored only for
  answered turns: a failed or no-LLM turn is not remembered. History lives in memory per
  `Chatbot` instance, not on disk.
- **Revisit if:** eval multi-turn cases fail on rewrites (then show the rewrite for approval,
  or send both the raw and rewritten queries and fuse), or conversations need to span sessions.
- **Model:** Claude Sonnet 5.5 (claude-sonnet-5-5) (proposed), human (approved)

### D-013: Grounded generation prompt and context budget
- **Date:** 2026-11-06
- **Context:** The answer must rely on the retrieved text, cite it, refuse when it is
  insufficient, and surface conflicts (FAILURE F6).
- **Options:**
  - A. Strict system prompt: numbered passages tagged with chunk ids, "only these passages",
    citations by chunk id, one fixed refusal sentence, passages declared untrusted.
  - B. No grounding instruction (free-form answer).
  - C. A. plus a retrieval-score threshold that refuses before calling the LLM.
- **Choice:** A. The exact text is `generate.SYSTEM_PROMPT` (and is printed by
  `ask.py --show-prompt "question"`). Passages sit in the user message in delimited blocks;
  best-first, up to `CONTEXT_TOKEN_BUDGET` = 3000 real bge tokens (at least one), with
  `TOP_K` = 5. Refusal is detected by the fixed sentence. After the call, cited ids are checked
  against the ids in the prompt: an id that was not in the context is reported as an invalid
  citation (`Answer.invalid_citations`, a warning in the CLI). Temperature 0, answer capped at
  500 tokens. The model has no tools and its output is displayed as text.
- **Why:** A fixed refusal string and chunk-id citations make grounding checkable by code
  (refusal rate, citation validity) rather than by reading. Best-first ordering puts the
  likeliest answer first (lost-in-the-middle).
- **Trade-off:** The prompt reduces, never removes, use of outside knowledge: a model can still
  cite a real id and add a fact the chunk does not contain, and the citation check cannot see that
  (that needs a faithfulness judge, Stage 7). No score threshold yet (C): weak retrieval still
  produces an answer attempt, and the model must refuse on its own. The 3000-token budget is
  generous for 5 chunks (max 5 × 512 = 2560) and only starts to matter if TOP_K is raised or a
  small-context local model is used (then lower it).
- **Revisit if:** a real run shows outside-knowledge answers or weak-retrieval answers that a
  threshold would have stopped (record it in FAILURE.md); a local model has a context under ~4k
  tokens; or answers need to be longer than a few sentences.
- **Model:** Claude Sonnet 5.5 (claude-sonnet-5-5) (proposed), human (approved)

### D-014: UI: Gradio chat with sources panel
- **Date:** 2026-11-06
- **Context:** The brief needs "a simple user interface"; the plan asks that it shows the answer
  and the retrieved sources.
- **Options:**
  - A. Gradio `Blocks`: chat on the left; search query, status and cited sources on the right.
  - B. Gradio `ChatInterface` (least code, no side panel without extra work).
  - C. Streamlit; D. CLI only (already exists).
- **Choice:** A with `gradio` (maintained by Hugging Face, Apache-2.0; added with `uv add`, version 6.31.0
  in uv.lock; the user approved it, 2026-11-06). One `Chatbot` per browser session (history lives
  in session state), the index and embedder are loaded once per process. Shows: the **rewritten
  search query**, whether the model refused, invalid citations, and each retrieved chunk with its
  heading path, a link to the exact Wikipedia revision, a text snippet, and a tick for cited ones.
  Per-session limit of one question per `UI_MIN_SECONDS_BETWEEN_QUESTIONS` (2 s), input capped at
  `MAX_QUERY_CHARS`, bound to 127.0.0.1 by default (`UI_HOST`). Launch: `uv run python -m ragbot.ui`.
- **Why:** Showing the query and sources makes grounding and rewrite mistakes visible in the demo,
  which is what the presentation needs to explain. B hides that behind the chat bubble.
- **Trade-off:** gradio is a large dependency (pandas, pillow, fastapi, uvicorn, ...) and its API
  changed between major versions (the code targets 6.x). The rate limit is per session, so a
  script that opens many sessions bypasses it; no authentication. Retrieved text is rendered as
  Markdown (Gradio sanitises HTML); Wikipedia text contains no links, but a different corpus could.
  The browser rendering was not checked by eye in the build environment; the app was started and
  driven through its API (HTTP 200, answer, sources, rate-limit message).
- **Revisit if:** the UI is exposed beyond localhost (add auth and a global limit), or Gradio's
  Markdown rendering of document text becomes an injection path.
- **Model:** Claude Sonnet 5.5 (claude-sonnet-5-5) (proposed), human (approved)
