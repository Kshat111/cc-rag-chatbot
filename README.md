# rag-chatbot

A retrieval-augmented chatbot over a small set (10–20) of source documents.
Retrieval is hybrid: dense embeddings + BM25. Follow-up questions are rewritten
using chat history before retrieval.

> Status: complete pipeline (ingest, chunk, index, hybrid retrieval, evaluation, CLI and Gradio chat). See docs/HANDOVER.md.

## Install

Requires [uv](https://docs.astral.sh/uv/) (it installs the right Python itself).

```bash
git clone <repo-url> rag-chatbot
cd rag-chatbot
uv sync
```

The 15 Wikipedia articles are already committed in `data/raw/`, so no fetch is needed.
The first index build downloads the embedding model (BAAI/bge-small-en-v1.5, ~67 MB) from
huggingface.co into `data/models/` and takes about 40 s on a laptop CPU; later runs are offline.

## Usage

```bash
# 1. Build the dense + BM25 data into data/index/ (BM25 itself is rebuilt in memory at load)
uv run python scripts/build_index.py

# 2. Ask a question from the CLI. Needs an LLM (see "Choosing an LLM" below);
#    without one it still runs and prints the retrieved passages.
uv run python scripts/ask.py "your question"
uv run python scripts/ask.py                      # interactive chat with history (/reset, /quit)
uv run python scripts/ask_retrieval.py "N700S" --method all   # retrieval only, no LLM

# 3. Launch the UI, then open http://127.0.0.1:7860
uv run python -m ragbot.ui

# 4. Evaluate retrieval (eval/questions.jsonl): recall@k and MRR, saved to eval/results/
uv run python scripts/evaluate.py

# 4b. Chunk size / overlap experiments (in memory, ~3 min; does not touch data/index)
uv run python scripts/experiment_chunking.py

# 5. Unit tests (no model download, no network)
uv run --with pytest pytest -q

# Optional: re-download the corpus (overwrites data/raw; the index must then be rebuilt)
uv run python scripts/fetch_corpus.py
```

If the index is missing or out of date (corpus, chunk settings or model changed), the tools stop
and tell you to run `scripts/build_index.py`.

## Choosing an LLM

The chatbot talks to any OpenAI-compatible chat endpoint. Export the key in your shell
(never commit it) and pick a provider:

```bash
export OPENAI_API_KEY=...                                  # default provider: openai (gpt-4o-mini)
RAGBOT_LLM_PROVIDER=ollama uv run python scripts/ask.py "..."      # local model, no key (ollama serve; ollama pull llama3.2)
export GROQ_API_KEY=...; RAGBOT_LLM_PROVIDER=groq uv run python scripts/ask.py "..."
# anything else that speaks the OpenAI API:
RAGBOT_LLM_BASE_URL=http://localhost:8000/v1 RAGBOT_LLM_MODEL=my-model RAGBOT_LLM_API_KEY_ENV=MY_KEY ...
```

Presets (openai, ollama, groq, openrouter) are in `src/ragbot/config.py`.

## Troubleshooting

- `Index not usable ... stale`: run `uv run python scripts/build_index.py`.
- `HTTP 429 ... no credits` / `quota`: the key is valid but the account has no API credit; add billing or use another provider.
- `HTTP 401`: wrong key. `HTTP 404`: the model name is not available; set `RAGBOT_LLM_MODEL`.
- Ollama `Connection refused`: start it (`ollama serve`; if it says the address is in use, it is already running).
- The UI says "No LLM is configured": the key must be exported in the same terminal that starts the UI.
- Small local models may return a citation only or ignore the rules; the CLI/UI show a warning. Use a stronger model.

## Layout

| Path | Purpose |
|---|---|
| `src/ragbot/config.py` | All tunables: chunk size, overlap, models, k, paths |
| `src/ragbot/ingest.py` | Load + clean documents |
| `src/ragbot/chunking.py` | Chunking strategy |
| `src/ragbot/index.py` | Build dense + BM25 indexes |
| `src/ragbot/retrieve.py` | Dense, BM25, hybrid search |
| `src/ragbot/history.py` | Follow-up query rewriting |
| `src/ragbot/generate.py` | Grounded prompt + LLM call |
| `src/ragbot/pipeline.py` | Query pipeline loop |
| `src/ragbot/ui.py` | Gradio or Streamlit app |
| `scripts/` | Entry points: build_index, ask, evaluate |
| `eval/` | Hand-written test set and saved metric outputs |
| `docs/` | Paper trail: handover, decisions, architecture, constraints |

## Credits

Corpus text in `data/raw/` is from [Wikipedia](https://en.wikipedia.org/), licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Article URLs and exact
revision permalinks (whose history pages list the authors) are in `data/raw/manifest.json`.
Text was cleaned (references, link lists and tables removed).

## Docs

Start with `docs/HANDOVER.md`, `docs/CONSTRAINTS.md`, `docs/ARCHITECTURE.md`.
