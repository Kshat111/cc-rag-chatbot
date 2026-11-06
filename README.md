# rag-chatbot

A retrieval-augmented chatbot over a small set (10–20) of source documents.
Retrieval is hybrid: dense embeddings + BM25. Follow-up questions are rewritten
using chat history before retrieval.

> Status: indexing, retrieval, evaluation, and the CLI chat work; the Gradio UI is not written yet. See docs/HANDOVER.md.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> rag-chatbot
cd rag-chatbot
uv sync
cp .env.example .env   # then fill in values locally
```

## Usage

```bash
# 1. Download the Wikipedia corpus into data/raw/ (titles live in src/ragbot/config.py)
uv run python scripts/fetch_corpus.py --check   # resolve titles only, writes nothing
uv run python scripts/fetch_corpus.py           # fetch; re-runs overwrite

# 2. Build the dense + BM25 indexes into data/index/
uv run python scripts/build_index.py

# 3. Ask a question from the CLI (needs an LLM, see "Choosing an LLM"; without one it prints the retrieved passages)
uv run python scripts/ask.py "your question"
uv run python scripts/ask.py                      # interactive chat with history
uv run python scripts/ask_retrieval.py "N700S" --method all   # retrieval only, no LLM

# 4. Run the evaluation set (eval/questions.jsonl), results go to eval/results/
uv run python scripts/evaluate.py

# 5. Launch the UI
uv run python -m ragbot.ui
```

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
