# Failure modes

How the system fails, and how we notice. Filled in as we learn.

| # | Failure | Symptom | Detection | Mitigation | Status |
|---|---|---|---|---|---|
| F1 | Lead-in sentence whose table was dropped (D-002) | Chunk like "The sets were formed as follows:" ranks high for "how are the N700A sets formed?" and gives the LLM nothing to answer with | Manual review of chunks.jsonl (0-series #0003–#0006, N700 #0004…#0015, scmaglev #0008–#0009, shinkansen #0015); eval type `table` | Fetch tables via REST HTML (D-002 revisit, planned) | Open |
| F2 | Table footnotes without their table | "* The sum of…", "** Only refers to…", "† Indicates…" in shinkansen #0040–#0041 | Grep for lines starting with `*`, `†` | D-005 drops them; revisit when tables return | Mitigated (D-005) |
| F3 | Template artifacts | "¥1.239294631884×1012" (means ×10^12); coordinates printed twice with U+FEFF in chuo #0002 | Character survey of data/raw | D-005 normalisation | Mitigated (D-005) |
| F4 | Formatting carries meaning but is lost | west-japan #0002: "Those in italics are announcement names", but italics are gone | Manual review | Only fixable by keeping markup (REST HTML) | Open |
| F5 | Keyword-only list chunks | JR Central affiliates (#0003–#0009), JR West line lists: names with no prose; embeddings likely weak | Eval type `keyword_list`; compare dense vs BM25 | Hybrid retrieval (Stage 3) | Open, measure |
| F6 | Conflicting facts across documents | Chūō Shinkansen opening: "at least 2034" (shinkansen, scmaglev, JR Central) vs "no earlier than 2035" and "as early as 2036" (chuo) | Eval type `conflict` | Retrieve from several documents; prompt must surface disagreement and cite both | Open |
| F7 | Near-duplicate paragraphs across documents | Shizuoka Airport station paragraph in tokaido #0009 and chuo #0007 compete for top-k slots | Eval type `near_duplicate`; check top-k diversity | Consider de-duplication or per-document caps (Stage 3) | Open |
| F8 | LLM answers from its own memory instead of context | Correct answer to a fact newer than the model's training data, with no supporting chunk | Eval type `recency` (July 2026 Katsuragawa decision) | Grounded prompt; check answer cites a retrieved chunk | Open |

Chunk ids refer to `data/index/chunks.jsonl` at CHUNK_SIZE 512 and may shift when cleaning or chunk settings change.
