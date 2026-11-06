# Alternatives

Rejected options, and the condition under which to reconsider them.

| Decision | Rejected option | Why rejected | Revisit if |
|---|---|---|---|
| D-001 | Poetry | Slower; separate tool for Python versions | uv becomes unmaintained |
| D-002 | REST API Parsoid HTML | Needs a hand-written HTML-to-text converter | Eval needs table/infobox facts |
| D-002 | httpx / requests | Not needed for 15 sequential GETs | Corpus grows past ~100 pages or needs concurrency |
| D-003 | Strip any trailing line ending in ":" | Deletes real facts in lead-in sentences | Orphan lead-ins hurt retrieval quality |
| D-004 | Fixed-size sliding window | Mixes sections, wrong heading paths, mid-sentence cuts | Kept as the eval baseline; adopt if it matches B on recall@k |
| D-004 | Semantic chunking | Needs embedding model; cuts hard to explain | B misses answers that span sections |
| D-004 | Content-hash chunk ids | Unreadable; collides on repeated text | Chunks must keep ids across reorderings |
| D-005 | NFKC + drop symbol-led lines | Rewrites real text; may drop content | Many new artifact types appear |
| D-006 | bge-base-en-v1.5 | 3× larger download, not needed yet | Paraphrase questions have low recall@k |
| D-006 | multilingual-e5-small | Needs PyTorch; required prefixes | Japanese queries become a requirement |
| D-006 | all-MiniLM-L6-v2 | 256-token limit, would cut 37% of chunks | Never at CHUNK_SIZE > 256 |
| D-006 | sentence-transformers runtime | PyTorch, ~3× install size | Need fine-tuning, GPU, or a model fastembed lacks |
| D-007 | FAISS flat index | Same exact search, extra dependency | Need ANN or GPU search |
| D-007 | Approximate search (HNSW/IVF) | No benefit at 249 vectors; costs recall | >~100k vectors or latency over budget |
| D-004 result | Keep chars/4, shrink CHUNK_SIZE to ~400 or CHARS_PER_TOKEN to ~2.1 | Shrinks every chunk to cover a +92% worst case; 75% of chunks were already overestimated | No access to the model's tokenizer |
| D-004 result | Split oversized chunks inside build_index | Stored chunks would differ from chunker output (breaks D-007) | Never |
| D-008 | rank-bm25 | Unmaintained since 2022; still needs our tokenizer | Corpus passes ~50k chunks |
| D-008 | bm25s | No licence shown on PyPI; still needs our tokenizer | Licence confirmed and corpus passes ~50k chunks |
| D-008 | Japanese character bigrams | Not needed: Japanese only appears in parenthesised names | Japanese queries become a requirement or return noisy hits |
| D-008 | Morphological analyser (e.g. MeCab) | New dependency for 902 characters | Japanese text becomes a real part of the corpus |
| D-008 | Model-name stems on the query side too | "N700S" would also match N700A | Users expect a variant to find its siblings |
| D-009 | Weighted sum of normalised scores | Min-max is unstable per query; α means different things per query | RRF dilutes a confidently right method |
| D-009 | Plain RRF | No dial for the dense vs BM25 experiments | Never (α = 0.5 already equals it) |
| D-010 | Hand-written gold chunk ids | Break on every re-chunk | Never at this corpus size |
| D-010 | LLM judge for retrieval relevance | Costs a key, can be wrong, not needed for recall@k | Gold quotes stop being findable (heavy paraphrase) |
| D-011 | Official `openai` SDK | A dependency for ~60 lines of HTTP | Streaming or tool calling is needed |
| D-011 | Per-provider SDKs / LiteLLM | Several dependencies, heavier install | A provider that is not OpenAI-compatible is required |
| D-012 | Raw history as the retrieval query | A topic switch retrieves the old topic | Never with an LLM available (use it if there is no LLM) |
| D-012 | Running summary memory | Extra component to keep correct | Conversations regularly exceed 5 turns |
| D-013 | No grounding instruction | Model answers from its own knowledge | Never |
| D-013 | Score threshold before the LLM (refuse early) | Needs calibration on the eval; none done yet | A real run answers from weak retrieval |
| D-014 | Streamlit | Second UI stack, reruns the script on each input | Gradio is unavailable on the target machine |
| D-014 | Gradio ChatInterface | No side panel for query and sources | A plain chat is enough |
