# Architecture

One-page system map. Keep it to one page.

## Indexing (offline)

```
data/raw/ --> ingest --> chunking --> index --> data/index/ (dense + BM25)
```

## Query (online)

```
user --> ui / ask.py --> pipeline
                           |-- history.rewrite_query   (LLM)
                           |-- retrieve.hybrid_search  (data/index/)
                           '-- generate.generate_answer (LLM)
                         <-- answer + sources
```

## Trust boundaries

| # | Boundary | Untrusted side | Notes |
|---|---|---|---|
| 1 | User -> UI/CLI | User input | May contain prompt injection |
| 2 | data/raw -> ingest | Document content | Retrieved text is data, not instructions |
| 3 | App -> LLM API | Network / provider | API key from env only; prompts leave the machine |
| 4 | LLM -> user | Model output | May hallucinate; must cite retrieved sources |
