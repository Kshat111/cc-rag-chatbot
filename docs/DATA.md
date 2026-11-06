# Data

The data structures that flow through the system, in pipeline order.

## Source documents (`data/raw/`)
- 15 Wikipedia articles as plain text (`<slug>.txt`), cleaned by `ingest.clean_wikipedia_extract`
  (D-002, D-003, D-005). Headings kept as Markdown (`#`, `##`, …).
- `manifest.json`: one record per document with `title`, `requested_title`, `file`, `page_id`,
  `revision_id`, `source_url`, `permalink` (exact revision), `license`, `fetched_at`,
  `char_count`, `sha256`. The manifest's own sha256 goes into the index fingerprint, so a
  re-fetch makes the index stale.

## Chunk (one line of `data/index/chunks.jsonl`)
| Field | Meaning |
|---|---|
| `chunk_id` | `<doc slug>#<4-digit index>`, e.g. `shinkansen#0006`. Stable for a given corpus + chunk config |
| `doc_title`, `file`, `source_url`, `permalink`, `revision_id`, `page_id` | Provenance, used for citations |
| `heading_path`, `heading_path_str` | e.g. `["Shinkansen", "History", "Construction"]` / `Shinkansen > History > Construction` |
| `text` | The chunk text (whole sentences) |
| `embed_text` | `heading_path_str + "\n\n" + text`: what dense and BM25 index by default |
| `n_tokens` | Real bge tokens of exactly what is embedded, incl. [CLS]/[SEP] (≤ 512) |
| `n_tokens_est`, `n_chars` | chars/4 estimate and length, kept for comparison (D-004 result) |
| `chunk_config` | The settings that produced it |

Why: line *i* of the file is row *i* of the vector matrix, so the file is the join key
between the two indexes and the citations.

## Dense index (`data/index/dense.npy` + `index_meta.json`)
- `float32` matrix `[248, 384]`, each row L2-normalised, so cosine similarity = dot product.
  Search is one matrix-vector product over all rows (exact search, ~372 KB).
- `index_meta.json`: the fingerprint checked at load (schema version, model, dims, max tokens,
  passage prefix, dense field, chunk_config, manifest sha256), plus chunks.jsonl sha256,
  token stats, timings, library versions, and a probe sentence with its vector (D-007).

## BM25 inverted index (in memory, `retrieve.BM25`)
- Built at load from `chunks.jsonl` (`BM25_INDEX_FIELD`, default `embed_text`), never saved.
- `postings: term -> (array of chunk positions, array of term frequencies)`; `idf: term -> float`;
  `doc_len` per chunk and `avgdl`. A query only touches the postings of its own terms.
- Terms come from `retrieve.tokenize` (D-008): accent-folded lowercase words, decimals,
  single Japanese characters, plus model-name stems on the index side (`n700a` → also `n700`).

## Search result (`retrieve.Hit`)
`Hit(chunk_id, score, rank)`, rank 1-based. Same type for `dense_search`, `bm25_search` and
`hybrid_search`. Scores are comparable only within one method and one query
(cosine, BM25 score, or weighted RRF score ≤ 1/61).

## Eval question record (`eval/questions.jsonl`)
| Field | Meaning |
|---|---|
| `id`, `type` | e.g. `q007`, `number` / `paraphrase` / `two_chunk` / `multi_turn` / `unanswerable` … |
| `question` | The user question (for `multi_turn`, see `turns`) |
| `turns` | For multi-turn: list of `{question, expected_rewrite}` |
| `expected_answer` | Hand-written reference answer |
| `expected_sources` | Files that contain the answer |
| `evidence` | Exact quotes from `data/raw/` that contain the answer; matched against chunk text, so evaluation survives re-chunking |
| `answerable_now` | `false` while the answer lives only in a dropped table (q005, q006) |
| `notes` | Why the question exists (links to FAILURE.md) |

## Conversation history
_Stage 5: to be filled._
