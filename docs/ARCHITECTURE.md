# Architecture

One-page system map. Decision ids refer to DECISIONS.md.

## Indexing (offline): `uv run python scripts/build_index.py`

```
Wikipedia API --fetch_corpus.py--> data/raw/*.txt + manifest.json        (D-002, D-003, D-005)
data/raw --ingest.load_documents--> chunking.chunk_documents              (D-004: heading-aware,
                                     |                                     sized in real bge tokens)
                                     v
               index.check_token_limits  (fail loudly if > 512 tokens)    (D-007)
                                     v
               fastembed bge-small-en-v1.5 -> L2-normalise                 (D-006)
                                     v
data/index/ chunks.jsonl + dense.npy + index_meta.json (written last)     (D-007)
```

## Query (online)

```
user --> ui.py / ask.py --> pipeline
                              |-- history.rewrite_query    (LLM)            Stage 5
                              |-- retrieve.Retriever                         D-008, D-009
                              |     load_dense_index (staleness checks)
                              |     BM25 rebuilt in memory from chunks.jsonl
                              |     dense_search | bm25_search | hybrid_search (weighted RRF)
                              '-- generate.generate_answer (LLM)            Stage 6
                            <-- answer + cited chunk ids + sources
```

Retrieval only, no LLM: `uv run python scripts/ask_retrieval.py "query" --method all`.

## Trust boundaries

| # | Boundary | Untrusted side | Notes |
|---|---|---|---|
| 1 | User -> UI/CLI | User input | May contain prompt injection; query length is checked against the embedder limit |
| 2 | Wikipedia -> data/raw -> prompt | Document content | Retrieved text is data, not instructions |
| 3 | Build machine -> huggingface.co | Model download (once) | Cached in data/models/; probe vector detects a changed model (D-007) |
| 4 | App -> LLM API | Network / provider | API key from env only; prompts leave the machine |
| 5 | LLM -> user | Model output | May hallucinate; must cite retrieved chunk ids |
