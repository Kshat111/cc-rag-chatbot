# Failure modes

How the system fails, and how we notice. Filled in as we learn.

| # | Failure | Symptom | Detection | Mitigation | Status |
|---|---|---|---|---|---|
| F1 | Lead-in sentence whose table was dropped (D-002) | Chunk like "The sets were formed as follows:" ranks high for "how are the N700A sets formed?" and gives the LLM nothing to answer with | Manual review of chunks.jsonl (0-series #0002–#0006; N700 #0004, #0007, #0009, #0012, #0014, #0016, #0018; scmaglev #0008–#0009; shinkansen #0015); eval type `table` | Fetch tables via REST HTML (D-002 revisit, planned) | Open |
| F2 | Table footnotes without their table | "* The sum of…", "** Only refers to…", "† Indicates…" in shinkansen Ridership (now #0040) | Grep for lines starting with `*`, `†` | D-005 drops them; revisit when tables return | Mitigated (D-005) |
| F3 | Template artifacts | "¥1.239294631884×1012" (now "×10^12", shinkansen #0005); coordinates printed twice with U+FEFF (chuo #0001) | Character survey of data/raw | D-005 normalisation | Mitigated (D-005) |
| F4 | Formatting carries meaning but is lost | west-japan #0002: "Those in italics are announcement names", but italics are gone | Manual review | Only fixable by keeping markup (REST HTML) | Open |
| F5 | Keyword-only list chunks | JR Central affiliates (#0003–#0011), JR West line lists (#0001–#0005): names with no prose; embeddings likely weak | Eval type `keyword_list`; compare dense vs BM25 | Hybrid retrieval (Stage 3b) | Open, measure |
| F6 | Conflicting facts across documents | Chūō Shinkansen opening: "at least 2034" (shinkansen #0008/#0018/#0046, scmaglev #0000/#0005, JR Central #0000) vs "no earlier than 2035" (chuo #0000) and "as early as 2036" (chuo #0008) | Eval type `conflict` | Retrieve from several documents; prompt must surface disagreement and cite both | Open |
| F7 | Near-duplicate paragraphs across documents | Shizuoka Airport station paragraph in tokaido #0009 and chuo #0006 compete for top-k slots | Eval type `near_duplicate`; check top-k diversity | Consider de-duplication or per-document caps (Stage 3b) | Open |
| F8 | LLM answers from its own memory instead of context | Correct answer to a fact newer than the model's training data, with no supporting chunk (Katsuragawa: hokuriku #0000/#0012, shinkansen #0043) | Eval type `recency` | Grounded prompt; check answer cites a retrieved chunk | Open |
| F9 | Chunk longer than the embedding model's input | fastembed silently truncates at 512 tokens, so the end of the chunk is never embedded | `build_index` token check with the model's own tokenizer (D-007) | Chunk sizes counted in real model tokens (D-004 result, 2026-11-05) | Mitigated; the check stays as a guard |

Chunk ids refer to `data/index/chunks.jsonl` built 2026-11-05 (CHUNK_SIZE 512 model_tokens, overlap 64, min 64; 248 chunks). They shift whenever cleaning or chunk settings change; the eval set uses file names and evidence quotes instead.
