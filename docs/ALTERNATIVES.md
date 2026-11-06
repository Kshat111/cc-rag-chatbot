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
