"""Load and clean source documents from config.RAW_DIR."""

import json
import re

from ragbot import config  # noqa: F401  (all tunables come from here)

# TextExtracts with exsectionformat=wiki emits headings as "== Title ==".
# Level 2 (==) is a top-level section; the page title itself is level 1.
_WIKI_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")

# --- D-005 text hygiene. Each rule matches an artifact seen in the corpus. ---
_ZERO_WIDTH = dict.fromkeys(map(ord, "﻿​‌‍⁠­"))
_ODD_SPACES = {ord(c): " " for c in "   "}
# The coord template prints "35.583°N 138.927°E / 35.583; 138.927 (Label)".
# Keep the readable form; drop the decimal duplicate and the optional label.
_COORD_DUPLICATE = re.compile(
    r"(\d+(?:\.\d+)?°[NS] \d+(?:\.\d+)?°[EW]) ?/ ?-?\d+(?:\.\d+)?; ?-?\d+(?:\.\d+)?(?: ?\([^()]*\))?"
)
# The inflation template renders 1.23×10^12 as "1.23×1012". Only rewrite after
# a decimal mantissa, so ordinary products like "3×100" are left alone.
_FLAT_EXPONENT = re.compile(r"(\d\.\d+)×10(\d{1,3})(?!\d)")


def normalize_text(text):
    """Fix template artifacts that survive TextExtracts (D-005)."""
    text = text.translate(_ZERO_WIDTH).translate(_ODD_SPACES)
    text = _COORD_DUPLICATE.sub(r"\1", text)
    return _FLAT_EXPONENT.sub(r"\1×10^\2", text)


def _is_footnote(line, markers):
    stripped = line.lstrip()
    return any(stripped.startswith(m + " ") for m in markers)


def _is_bare_label(line, max_words):
    """True for lines like "Legend:" or "Service column legend:" (see D-003)."""
    stripped = line.strip()
    if not stripped.endswith(":"):
        return False
    words = stripped[:-1].split()
    return 1 <= len(words) <= max_words and all(w.isalpha() for w in words)


def clean_wikipedia_extract(
    title,
    text,
    drop_sections,
    bare_label_max_words=config.WIKI_BARE_LABEL_MAX_WORDS,
    footnote_markers=config.WIKI_FOOTNOTE_MARKERS,
):
    """Turn a TextExtracts plain-text extract into heading-preserving text.

    Output starts with "# <title>", then the lead, then sections as
    "## X" / "### Y" with levels kept, so the chunker can rebuild heading paths.
    Removes dropped sections (and everything nested under them), trailing
    bare labels left behind by dropped tables (D-003), template artifacts
    and table footnotes (D-005), and any heading with no text beneath it.
    A heading whose only content is non-empty subsections is kept.
    """
    drop = {s.casefold() for s in drop_sections}
    text = normalize_text(text)

    # Parse into a flat list of (level, heading, body_lines). Level 1 = lead.
    sections = [(1, title, [])]
    for line in text.splitlines():
        m = _WIKI_HEADING.match(line)
        if m:
            sections.append((len(m.group(1)), m.group(2), []))
        elif _is_footnote(line, footnote_markers):
            continue
        else:
            sections[-1][2].append(line.rstrip())

    # Drop listed sections plus their subsections: skip until a heading at
    # the same or a higher level appears.
    kept, skip_level = [], None
    for level, heading, body in sections:
        if skip_level is not None and level > skip_level:
            continue
        skip_level = None
        if level > 1 and heading.casefold() in drop:
            skip_level = level
            continue
        kept.append((level, heading, body))

    # TextExtracts drops tables but keeps the label that introduced them
    # ("Legend:"). Strip such labels only at the end of a section: one that
    # is followed by text is still introducing something real. Runs before
    # the empty-heading pass, so a section left with nothing is removed.
    for _, _, body in kept:
        while body and (not body[-1].strip() or _is_bare_label(body[-1], bare_label_max_words)):
            body.pop()

    # Remove empty headings. Walk backwards so we know whether any deeper
    # section that follows survived; if so, this heading still has content.
    result, child_has_content = [], {}
    for level, heading, body in reversed(kept):
        has_text = any(line.strip() for line in body)
        has_children = any(v for lvl, v in child_has_content.items() if lvl > level)
        # Children only count for the nearest enclosing heading.
        for lvl in [k for k in child_has_content if k > level]:
            del child_has_content[lvl]
        if level == 1 or has_text or has_children:
            result.append((level, heading, body))
            child_has_content[level] = True
        else:
            child_has_content.setdefault(level, False)
    result.reverse()

    out = []
    for level, heading, body in result:
        out.append("#" * level + " " + heading)
        out.append("")
        out.extend(body)
        out.append("")
    # Collapse runs of blank lines and trim.
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cleaned + "\n"


def load_documents(manifest_path=None):
    """Return cleaned documents as dicts: manifest metadata plus "text".

    The manifest is the list of record: a .txt file in data/raw/ that it
    does not name is ignored (CONSTRAINTS: only data/raw/ is indexed, and
    only what the fetch produced).
    """
    manifest_path = manifest_path or config.MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; run scripts/fetch_corpus.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = []
    for meta in manifest["documents"]:
        text = (manifest_path.parent / meta["file"]).read_text(encoding="utf-8")
        docs.append({**meta, "text": text})
    return docs
