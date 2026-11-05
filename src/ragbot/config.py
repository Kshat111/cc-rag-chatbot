"""Single source of truth for every tunable.

No other module may hardcode paths, model names, keys or retrieval parameters.
Values below are placeholders until chosen and recorded in docs/DECISIONS.md.
"""

import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
INDEX_DIR = ROOT_DIR / "data" / "index"
EVAL_QUESTIONS = ROOT_DIR / "eval" / "questions.jsonl"
EVAL_RESULTS_DIR = ROOT_DIR / "eval" / "results"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# --- Corpus (Wikipedia, see D-002) ---
# Titles as requested; redirects are followed and reported, never guessed.
CORPUS_TITLES = [
    "Shinkansen",
    "Tokaido Shinkansen",
    "Sanyo Shinkansen",
    "Tohoku Shinkansen",
    "Hokuriku Shinkansen",
    "Chuo Shinkansen",
    "SCMaglev",
    "Japanese National Railways",
    "Central Japan Railway Company",
    "East Japan Railway Company",
    "West Japan Railway Company",
    "Hideo Shima",
    "0 Series Shinkansen",
    "N700 Series Shinkansen",
    "Rail transport in Japan",
]
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_LICENSE = "CC BY-SA 4.0"
# Wikimedia asks for a contact in the User-Agent.
HTTP_USER_AGENT = (
    "rag-chatbot-corpus/0.1 (https://github.com/Kshat111) python-urllib"
)
FETCH_DELAY_S = 1.0  # pause between requests, to be polite
FETCH_TIMEOUT_S = 30
FETCH_MAX_RETRIES = 3  # backoff 2s, 4s, 8s
WIKI_MAXLAG_S = 5  # server sheds automated load when replication lag exceeds this
# Matched case-insensitively against heading text. Whole section (with subsections) is removed.
WIKI_DROP_SECTIONS = [
    "References",
    "Notes",
    "Citations",
    "Sources",
    "Bibliography",
    "Further reading",
    "External links",
    "See also",
]
# A trailing line of at most this many words ending in ":" (e.g. "Legend:") is
# an orphan label of a dropped table and is removed (D-003). Sentences are kept.
WIKI_BARE_LABEL_MAX_WORDS = 3
# Lines starting with one of these, then a space, are footnotes of a dropped
# table and are removed (D-005). Remove this rule once tables are fetched.
WIKI_FOOTNOTE_MARKERS = ["*", "**", "***", "†", "‡"]

# --- Chunking ---
# See D-004. Sizes are in CHUNK_LENGTH_UNIT and apply to embed_text
# (heading path + text), so the prepended path counts against the budget.
CHUNK_LENGTH_UNIT = "approx_tokens"  # or "chars"
CHARS_PER_TOKEN = 4.0  # deliberately low, so token counts are overestimated
CHUNK_SIZE = 512  # max length of a chunk
CHUNK_OVERLAP = 64  # whole sentences carried over, within one section only
CHUNK_MIN_SIZE = 64  # smaller chunks are merged with a neighbour
CHUNK_PREPEND_HEADINGS = True
# Fixed overlap means relative overlap differs per size (25% at 256, 6% at 1024).
CHUNK_SIZE_EXPERIMENTS = [256, 512, 1024]
CHUNK_OVERLAP_EXPERIMENTS = [0, 32, 64, 128]
CHUNKS_DUMP_PATH = INDEX_DIR / "chunks.jsonl"

# --- Models ---
EMBEDDING_MODEL = "TODO"  # TODO(decision)
LLM_MODEL = "TODO"  # TODO(decision)
REWRITE_MODEL = LLM_MODEL  # model used for follow-up query rewriting

# --- Retrieval ---
TOP_K = 5
HYBRID_ALPHA = 0.5  # weight of dense vs BM25 in hybrid fusion; 1.0 = dense only

# --- Generation ---
TEMPERATURE = 0.0
MAX_HISTORY_TURNS = 5

# --- Secrets ---
# Read from the environment only; never hardcode. Name must match .env.example.
LLM_API_KEY_ENV = "LLM_API_KEY"


def llm_api_key() -> str | None:
    return os.environ.get(LLM_API_KEY_ENV)
