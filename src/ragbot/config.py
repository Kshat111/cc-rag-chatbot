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
# See D-004. CHUNK_SIZE bounds exactly what the embedding model receives:
# passage prefix + heading path + text, plus [CLS]/[SEP] for "model_tokens".
# "model_tokens" counts with the embedding model's own tokenizer
# (CHUNK_TOKENIZER_REPO, below); "approx_tokens" (chars / CHARS_PER_TOKEN) and
# "chars" are fallbacks that need no model files.
CHUNK_LENGTH_UNIT = "model_tokens"
CHARS_PER_TOKEN = 4.0  # fallback estimate; measured 4.30 overall, but up to 92% low on Japanese-heavy chunks
CHUNK_SIZE = 512  # = EMBED_MAX_TOKENS: the whole chunk as embedded must fit the model
CHUNK_OVERLAP = 64  # whole sentences carried over, within one section only
CHUNK_MIN_SIZE = 64  # smaller chunks are merged with a neighbour
CHUNK_PREPEND_HEADINGS = True
# bge-small caps input at 512 tokens, so 1024 is not testable with this model
# (D-004 note). 384 is an optional extra point. Fixed overlap 64 is 50% of a
# 128-token chunk but 12.5% of 512, so the size experiment is not "size only".
CHUNK_SIZE_EXPERIMENTS = [128, 256, 512]
CHUNK_OVERLAP_EXPERIMENTS = [0, 32, 64, 128]
CHUNKS_DUMP_PATH = INDEX_DIR / "chunks.jsonl"

# --- Models ---
# Embeddings (D-006). EMBEDDING_MODEL is fastembed's name for the model; the
# tokenizer comes from the repo fastembed actually downloads, so the token
# check counts exactly what the model sees.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_TOKENIZER_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
CHUNK_TOKENIZER_REPO = EMBED_TOKENIZER_REPO  # chunk sizes are counted with the embedding tokenizer
EMBED_DIM = 384
EMBED_MAX_TOKENS = 512  # including [CLS]/[SEP]; fastembed silently truncates beyond this
EMBED_USE_QUERY_INSTRUCTION = True  # switch off for the on/off experiment
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBED_PASSAGE_PREFIX = ""  # BGE v1.5 uses no passage prefix
EMBED_BATCH_SIZE = 32
EMBED_CACHE_DIR = ROOT_DIR / "data" / "models"  # fastembed defaults to the OS temp dir
# Fixed sentence embedded at build time and again at load time; a different
# vector means the model changed under us (D-007).
EMBED_PROBE_TEXT = "The Tokaido Shinkansen opened in 1964 between Tokyo and Osaka."
EMBED_PROBE_MIN_COSINE = 0.999

LLM_MODEL = "TODO"  # TODO(decision)
REWRITE_MODEL = LLM_MODEL  # model used for follow-up query rewriting

# --- Index (D-007) ---
# Which chunk field each index reads: "embed_text" (heading path + text) or "text".
DENSE_INDEX_FIELD = "embed_text"
INDEX_CHUNKS_PATH = CHUNKS_DUMP_PATH  # row i of the vectors = line i of this file
DENSE_VECTORS_PATH = INDEX_DIR / "dense.npy"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

# --- Retrieval (D-008 BM25, D-009 hybrid) ---
TOP_K = 5
# BM25 is rebuilt in memory from INDEX_CHUNKS_PATH at load (milliseconds), never saved.
BM25_INDEX_FIELD = "embed_text"  # "embed_text" (heading path + text) or "text"
BM25_K1 = 1.5  # term-frequency saturation
BM25_B = 0.75  # length normalisation (0 = none, 1 = full)
# Weighted reciprocal rank fusion: alpha/(RRF_K + rank_dense) + (1 - alpha)/(RRF_K + rank_bm25)
HYBRID_ALPHA = 0.5  # 1.0 = dense order only, 0.0 = BM25 order only
RRF_K = 60  # standard constant (Cormack et al., 2009)
HYBRID_CANDIDATES = 50  # top-n taken from each method before fusing
RETRIEVAL_METHODS = ("dense", "bm25", "hybrid")

# --- Generation ---
TEMPERATURE = 0.0
MAX_HISTORY_TURNS = 5

# --- Secrets ---
# Read from the environment only; never hardcode. Name must match .env.example.
LLM_API_KEY_ENV = "LLM_API_KEY"


def llm_api_key() -> str | None:
    return os.environ.get(LLM_API_KEY_ENV)
