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
