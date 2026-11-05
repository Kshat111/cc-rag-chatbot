"""Heading-aware recursive chunking (D-004).

Each document is split into sections by its Markdown headings. A section
that fits becomes one chunk; a longer one is split at line (paragraph)
boundaries, then sentences, then words, and packed greedily. Overlap is
whole sentences and only between chunks of the same section, so text never
appears under the wrong heading path. Chunks smaller than min_size are
merged with a neighbour inside the same top-level (##) section.

Run `uv run python -m ragbot.chunking [--dump]` for corpus statistics.
"""

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from ragbot import config

_HEADING = re.compile(r"^(#{1,6}) (.*)$")
# Sentence end: . ! ? then whitespace, then anything that is not lowercase.
# Lowercase after a period is usually an abbreviation ("approx. km").
# Known miss: "U.S. Army" splits after "U.S.".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[^\sa-z])")


@dataclass(frozen=True)
class ChunkConfig:
    size: int
    overlap: int
    min_size: int
    unit: str = "approx_tokens"
    chars_per_token: float = 4.0
    prepend_headings: bool = True

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be >= 0 and < size")
        if not 0 <= self.min_size < self.size:
            raise ValueError("min_size must be >= 0 and < size")
        if self.unit not in ("approx_tokens", "chars"):
            raise ValueError(f"unknown unit {self.unit!r}")

    @classmethod
    def from_config(cls, **overrides):
        values = dict(
            size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            min_size=config.CHUNK_MIN_SIZE,
            unit=config.CHUNK_LENGTH_UNIT,
            chars_per_token=config.CHARS_PER_TOKEN,
            prepend_headings=config.CHUNK_PREPEND_HEADINGS,
        )
        values.update(overrides)
        return cls(**values)

    def length(self, text):
        # The pluggable length function. A real tokenizer can replace this
        # once the embedding model is chosen (D-004).
        if self.unit == "chars":
            return len(text)
        return math.ceil(len(text) / self.chars_per_token)


def estimate_tokens(text, chars_per_token=config.CHARS_PER_TOKEN):
    return math.ceil(len(text) / chars_per_token)


# --- Parsing -----------------------------------------------------------------


def parse_sections(text):
    """Return [(heading_path, lines)] in document order; headings excluded.

    A heading's own lines are the text before its first subheading. Headings
    with no lines of their own produce no section but stay in children's paths.
    """
    stack, sections = [], []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            sections.append(([h for _, h in stack], []))
        elif line.strip():
            if not sections:  # text before any heading; should not happen with our cleaner
                sections.append(([], []))
            sections[-1][1].append(line.strip())
    return [(path, lines) for path, lines in sections if lines]


# --- Splitting and packing -----------------------------------------------------

# An atom is (text, ends_line). Atoms of one line are joined with a space,
# and a line break follows an atom that ends its line.


def _join(atoms):
    out = []
    for i, (text, _) in enumerate(atoms):
        if i:
            out.append("\n" if atoms[i - 1][1] else " ")
        out.append(text)
    return "".join(out)


def _split_words(text, budget, cfg):
    """Split an oversized sentence at word boundaries (hard-cut a giant word)."""
    pieces, cur = [], ""
    for word in text.split():
        while cfg.length(word) > budget:
            k = len(word)
            while cfg.length(word[:k]) > budget:
                k -= 1
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(word[:k])
            word = word[k:]
        candidate = f"{cur} {word}" if cur else word
        if cfg.length(candidate) <= budget:
            cur = candidate
        else:
            pieces.append(cur)
            cur = word
    if cur:
        pieces.append(cur)
    return pieces


def _atomize(lines, budget, cfg):
    """Lines that fit stay whole; longer ones become sentences, then words."""
    atoms = []
    for line in lines:
        if cfg.length(line) <= budget:
            atoms.append((line, True))
            continue
        pieces = []
        for sentence in _SENTENCE_END.split(line):
            if cfg.length(sentence) <= budget:
                pieces.append(sentence)
            else:
                pieces.extend(_split_words(sentence, budget, cfg))
        atoms.extend((p, i == len(pieces) - 1) for i, p in enumerate(pieces))
    return atoms


def _sentences(atoms):
    """Re-split atoms into sentence atoms, for choosing overlap."""
    out = []
    for text, ends_line in atoms:
        parts = _SENTENCE_END.split(text)
        out.extend((p, ends_line and i == len(parts) - 1) for i, p in enumerate(parts))
    return out


def _overlap_tail(atoms, next_atom, budget, cfg):
    """Trailing whole sentences of `atoms`, at most cfg.overlap long, that
    still leave room for next_atom within budget."""
    if cfg.overlap == 0:
        return []
    sentences = _sentences(atoms)
    tail = []
    for sentence in reversed(sentences[1:]):  # never the whole previous chunk
        candidate = [sentence] + tail
        if cfg.length(_join(candidate)) > cfg.overlap:
            break
        if cfg.length(_join(candidate + [next_atom])) > budget:
            break
        tail = candidate
    return tail


def _pack(atoms, budget, cfg):
    chunks, cur = [], []
    for atom in atoms:
        if cur and cfg.length(_join(cur + [atom])) > budget:
            chunks.append(cur)
            cur = _overlap_tail(cur, atom, budget, cfg)
        cur.append(atom)
    if cur:
        chunks.append(cur)
    return [_join(c) for c in chunks]


# --- Records -------------------------------------------------------------------


def _path_str(path):
    return " > ".join(path)


def _embed_text(path, text, cfg):
    return f"{_path_str(path)}\n\n{text}" if cfg.prepend_headings else text


def _common_prefix(a, b):
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def _merge(first, second):
    """Merge two adjacent raw chunks; the deeper parts of each path move inline
    as heading lines so no context is lost."""
    path = _common_prefix(first["path"], second["path"])
    parts = []
    for c in (first, second):
        rest = c["path"][len(path):]
        if rest:
            parts.append("#" * (len(path) + 1) + " " + _path_str(rest) + "\n" + c["text"])
        else:
            parts.append(c["text"])
    return {"path": path, "top": first["top"], "text": "\n".join(parts)}


def _merge_small(raw, cfg):
    def fits(c):
        return cfg.length(_embed_text(c["path"], c["text"], cfg)) <= cfg.size

    def small(c):
        return cfg.length(_embed_text(c["path"], c["text"], cfg)) < cfg.min_size

    i = 0
    while i < len(raw):
        if small(raw[i]):
            # Prefer merging forward; fall back to the previous chunk (e.g. a
            # small last chunk of a section). Never across a ## boundary.
            if i + 1 < len(raw) and raw[i + 1]["top"] == raw[i]["top"]:
                merged = _merge(raw[i], raw[i + 1])
                if fits(merged):
                    raw[i : i + 2] = [merged]
                    continue
            if i > 0 and raw[i - 1]["top"] == raw[i]["top"]:
                merged = _merge(raw[i - 1], raw[i])
                if fits(merged):
                    raw[i - 1 : i + 1] = [merged]
                    i -= 1
                    continue
        i += 1
    return raw


def chunk_document(doc, cfg):
    """Chunk one document dict (from ingest.load_documents) into records."""
    raw = []
    for path, lines in parse_sections(doc["text"]):
        budget = cfg.size - (cfg.length(_path_str(path) + "\n\n") if cfg.prepend_headings else 0)
        if budget <= 0:
            raise ValueError(f"heading path longer than chunk size: {_path_str(path)!r}")
        # Title plus ## heading; the lead section (path of length 1) is its own group.
        top = tuple(path[:2])
        for text in _pack(_atomize(lines, budget, cfg), budget, cfg):
            raw.append({"path": path, "top": top, "text": text})

    raw = _merge_small(raw, cfg)
    stem = Path(doc["file"]).stem
    records = []
    for i, c in enumerate(raw):
        embed = _embed_text(c["path"], c["text"], cfg)
        records.append(
            {
                "chunk_id": f"{stem}#{i:04d}",
                "chunk_index": i,
                "doc_title": doc["title"],
                "heading_path": c["path"],
                "heading_path_str": _path_str(c["path"]),
                "text": c["text"],
                "embed_text": embed,
                "n_tokens": estimate_tokens(embed),
                "n_chars": len(c["text"]),
                "source_url": doc["source_url"],
                "permalink": doc["permalink"],
                "revision_id": doc["revision_id"],
                "page_id": doc["page_id"],
                "file": doc["file"],
                "chunk_config": asdict(cfg),
            }
        )
    return records


def chunk_documents(documents, cfg=None):
    """Return chunk records for all documents, each tied to its source."""
    cfg = cfg or ChunkConfig.from_config()
    return [r for doc in documents for r in chunk_document(doc, cfg)]


# --- Stats CLI -----------------------------------------------------------------


def _describe(values):
    return f"min {min(values):,}  median {statistics.median(values):,.0f}  max {max(values):,}"


def main():
    from ragbot.ingest import load_documents

    ap = argparse.ArgumentParser(description="Chunk the corpus and print statistics.")
    ap.add_argument("--dump", action="store_true", help=f"also write {config.CHUNKS_DUMP_PATH}")
    args = ap.parse_args()

    cfg = ChunkConfig.from_config()
    docs = load_documents()
    chunks = chunk_documents(docs, cfg)

    print(f"Config: {asdict(cfg)}")
    print(f"Documents: {len(docs)}   Chunks: {len(chunks)}")
    print(f"Tokens (est., embed_text): {_describe([c['n_tokens'] for c in chunks])}")
    print(f"Chars (text):              {_describe([c['n_chars'] for c in chunks])}")
    under = sum(1 for c in chunks if cfg.length(c["embed_text"]) < cfg.min_size)
    print(f"Chunks still under min_size ({cfg.min_size}): {under}")
    print("Chunks per document:")
    for doc in docs:
        n = sum(1 for c in chunks if c["file"] == doc["file"])
        print(f"  {n:>4}  {doc['title']}")

    if args.dump:
        config.CHUNKS_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.CHUNKS_DUMP_PATH, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Wrote {config.CHUNKS_DUMP_PATH}")


if __name__ == "__main__":
    main()
