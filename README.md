# rag-chatbot

A chatbot that answers questions about Japanese high-speed rail (Shinkansen) using
Wikipedia articles as its knowledge base. It finds the most relevant passages, then
asks an LLM to write a grounded answer from them and cite its sources.

> **Status:** complete pipeline — ingest, chunk, index, hybrid retrieval, evaluation, CLI and Gradio chat UI.

---

## What can I ask?

- *"When did the Tokaido Shinkansen open?"*
- *"Who built the N700S?"*
- *"Why do Shinkansen trains have such long pointed noses?"*
- *"When will the Chuo Shinkansen be finished?"*
- *"What is the top speed of the E5 series?"*
- *"Who was Hideo Shima?"*

The bot answers only from its Wikipedia corpus and says "I don't know" if the answer
isn't there. Follow-up questions work: it uses the conversation history to find the
right passages.

---

## What you need before starting

| Requirement | How to get it |
|---|---|
| **Git** | [git-scm.com](https://git-scm.com) |
| **uv** (installs Python for you) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **An API key** for one LLM | See the table in [Supported LLM providers](#supported-llm-providers) below |

> **No account?** Use [Ollama](https://ollama.ai) to run a model on your own machine — it's free and needs no API key.

---

## Quick start

```bash
# 1. Download the code
git clone <repo-url> rag-chatbot
cd rag-chatbot

# 2. Install dependencies (uv installs the right Python version automatically)
uv sync

# 3. Build the index
#    First run downloads the embedding model (~67 MB) and takes about 40 s.
#    Later runs are offline and instant.
uv run python scripts/build_index.py

# 4. Set your API key (pick any provider from the table below)
export OPENAI_API_KEY=sk-...

# 5. Ask a question
uv run python scripts/ask.py "When did the Tokaido Shinkansen open?"

# Or open the chat UI in your browser
uv run python -m ragbot.ui        # then visit http://127.0.0.1:7860
```

The 15 Wikipedia articles are already committed in `data/raw/` — no separate download needed.

---

## Supported LLM providers

The chatbot works with any OpenAI-compatible chat endpoint. Set **one** API key, then
run the command.

| Provider | Get a key | Key variable | Provider flag |
|---|---|---|---|
| **OpenAI / ChatGPT** | [platform.openai.com](https://platform.openai.com) | `OPENAI_API_KEY` | `openai` *(default)* |
| **Anthropic / Claude** | [openrouter.ai](https://openrouter.ai) ¹ | `OPENROUTER_API_KEY` | `anthropic` |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | `gemini` |
| **GitHub Copilot** | [github.com/marketplace/models](https://github.com/marketplace/models) | `GITHUB_TOKEN` | `github` |
| **xAI Grok** | [console.x.ai](https://console.x.ai) | `XAI_API_KEY` | `grok` |
| **Alibaba Qwen** | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) | `DASHSCOPE_API_KEY` | `qwen` |
| **Groq** (fast) | [console.groq.com](https://console.groq.com) | `GROQ_API_KEY` | `groq` |
| **Ollama** (free, local) | [ollama.ai](https://ollama.ai) — no key | *(none)* | `ollama` |

¹ Claude is routed through [OpenRouter](https://openrouter.ai) because Anthropic's API uses
a different protocol. Create a free OpenRouter account, add credits, and use your OpenRouter key.
To pick a specific Claude model: `RAGBOT_LLM_MODEL=anthropic/claude-opus-4`.

**How to use any provider:**

```bash
# Export your key, then set RAGBOT_LLM_PROVIDER to the flag in the table above.

# Gemini example
export GEMINI_API_KEY=...
RAGBOT_LLM_PROVIDER=gemini uv run python scripts/ask.py "Why do Shinkansen have long noses?"

# Claude example (via OpenRouter)
export OPENROUTER_API_KEY=...
RAGBOT_LLM_PROVIDER=anthropic uv run python scripts/ask.py "Who was Hideo Shima?"

# Grok example
export XAI_API_KEY=...
RAGBOT_LLM_PROVIDER=grok uv run python scripts/ask.py "When will the Chuo Shinkansen open?"

# Ollama example (free, runs on your machine)
ollama pull llama3.2       # download once (~2 GB)
RAGBOT_LLM_PROVIDER=ollama uv run python scripts/ask.py "What is the N700S?"

# Use a custom endpoint or model (overrides any preset)
RAGBOT_LLM_BASE_URL=http://localhost:8000/v1 \
RAGBOT_LLM_MODEL=my-model \
RAGBOT_LLM_API_KEY_ENV=MY_KEY_VAR \
uv run python scripts/ask.py "..."
```

To avoid exporting the key in every new terminal, put it in a `.env` file (gitignored, never
committed) and load it into the shell yourself. The app does **not** read `.env` on its own:

```bash
cp .env.example .env         # then edit .env: OPENAI_API_KEY=your-key (no quotes, no spaces)
set -a; source .env; set +a  # run this in each new terminal before starting the chatbot
```

### Using Ollama (free, local, no key, no account)

1. **Install Ollama** from [ollama.com/download](https://ollama.com/download) (macOS, Linux, Windows).
   On macOS and Windows the app starts the server for you; on Linux run `ollama serve` in a
   separate terminal. (If `ollama serve` says the address is already in use, it is already running.)
2. **Download a model once** (about 2 GB for the default):
   ```bash
   ollama pull llama3.2
   ollama list                # llama3.2 should appear
   ```
3. **Run the chatbot with it.** Set the provider in the same command, no key needed:
   ```bash
   RAGBOT_LLM_PROVIDER=ollama uv run python scripts/ask.py              # chat in the terminal
   RAGBOT_LLM_PROVIDER=ollama uv run python -m ragbot.ui                # web UI at http://127.0.0.1:7860
   ```
4. **Another model?** Pull it and name it: `RAGBOT_LLM_MODEL=<name from ollama list>`.

The first answer can take a while because Ollama loads the model into memory. Small local
models follow the citation and "I don't know" rules less reliably than hosted ones: the CLI and UI
show a warning when an answer has no valid citation, or is only a citation. For cleaner
answers use a larger model or a hosted provider from the table above.

---

## All commands

```bash
# Ask a single question
uv run python scripts/ask.py "your question"

# Interactive chat with follow-up questions  (/reset to start over, /quit to exit)
uv run python scripts/ask.py

# Retrieval only — see which passages the bot finds, without calling an LLM
uv run python scripts/ask_retrieval.py "N700S" --method all

# Launch the web UI
uv run python -m ragbot.ui        # open http://127.0.0.1:7860

# Measure retrieval quality (recall@k and MRR on the eval set)
uv run python scripts/evaluate.py

# Chunk size / overlap experiments (in memory, ~3 min; does not change the index)
uv run python scripts/experiment_chunking.py

# Unit tests — no model download, no network required
uv run --with pytest pytest -q

# Re-download the Wikipedia corpus (overwrites data/raw/; rebuild the index after)
uv run python scripts/fetch_corpus.py
```

If the index is out of date (corpus, chunk settings or embedding model changed),
every tool stops and tells you to run `scripts/build_index.py`.

---

## How it works (brief)

1. **Index (offline, one time):** 15 Wikipedia articles are split into ~250 short
   passages. Each passage is converted to a vector by a small embedding model
   ([bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), ~67 MB)
   and saved to `data/index/`.

2. **Retrieve:** when you ask a question, the system finds the most relevant
   passages in two ways — dense vector similarity (semantic) and BM25 keyword
   matching — then merges the results.

3. **Generate:** the top passages are sent to the LLM with a strict instruction
   to answer only from them and to cite which passage each fact comes from.

4. **Follow-ups:** if you ask a follow-up ("what about the E5?"), the previous
   conversation is used to rewrite your question before retrieval.

---

## File layout

| Path | What it does |
|---|---|
| `src/ragbot/config.py` | All tunables: chunk size, models, retrieval k, paths |
| `src/ragbot/ingest.py` | Load and clean Wikipedia articles |
| `src/ragbot/chunking.py` | Split articles into passages |
| `src/ragbot/index.py` | Build the dense vector index |
| `src/ragbot/retrieve.py` | Dense, BM25, and hybrid search |
| `src/ragbot/history.py` | Rewrite follow-up questions using chat history |
| `src/ragbot/generate.py` | Build the prompt and call the LLM |
| `src/ragbot/pipeline.py` | Full query pipeline (retrieve → generate) |
| `src/ragbot/ui.py` | Gradio chat UI |
| `scripts/` | Entry points: build_index, ask, evaluate |
| `eval/` | Test questions and saved retrieval metrics |
| `data/raw/` | 15 Wikipedia articles (committed to the repo) |
| `data/index/` | Built index — gitignored; rebuild with `build_index.py` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Index not usable ... stale` | Run `uv run python scripts/build_index.py` |
| `HTTP 429` / `no credits` / `quota` | Add billing to your account, or switch provider |
| `HTTP 401 Unauthorized` | Wrong API key — double-check the value |
| `HTTP 404` | Model name not available — set `RAGBOT_LLM_MODEL=<name>` |
| Ollama `Connection refused` | Start it: `ollama serve` (if the port is in use, it is already running) |
| UI: *"No LLM is configured"* | The key must be exported **in the same terminal** that starts the UI |
| Answer is only a citation with no text | The model is too small; switch to a stronger one |
| Answer is confident but wrong | Check the cited passage — the bot answers from Wikipedia only |

---

## Credits

Corpus text in `data/raw/` is from [Wikipedia](https://en.wikipedia.org/), licensed
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Article URLs
and exact revision permalinks (whose history pages list the authors) are in
`data/raw/manifest.json`. Tables and reference sections were removed during cleaning.
