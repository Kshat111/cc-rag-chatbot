"""Tests for heading-aware chunking (D-004). Synthetic documents only."""

import re

import pytest

from ragbot.chunking import ChunkConfig, chunk_document, parse_sections


def doc(text, title="Line"):
    return {
        "title": title,
        "text": text,
        "file": "line.txt",
        "source_url": "https://en.wikipedia.org/wiki/Line",
        "permalink": "https://en.wikipedia.org/w/index.php?oldid=1",
        "revision_id": 1,
        "page_id": 10,
    }


def cfg(**kw):
    base = dict(size=64, overlap=16, min_size=0, unit="approx_tokens", chars_per_token=4.0)
    base.update(kw)
    return ChunkConfig(**base)


def sentence(i, words=8):
    return f"Sentence {i} " + " ".join(f"word{i}x{j}" for j in range(words)) + "."


def test_heading_paths_nested_skipped_level_and_lead():
    text = "# Line\n\nLead.\n\n## History\n\nOld.\n\n#### Deep\n\nDeeper.\n\n## Tech\n\nNew.\n"
    paths = [p for p, _ in parse_sections(text)]
    assert paths == [
        ["Line"],
        ["Line", "History"],
        ["Line", "History", "Deep"],
        ["Line", "Tech"],
    ]


def test_heading_with_no_own_text_produces_no_section_but_stays_in_path():
    text = "# Line\n\nLead.\n\n## History\n\n### Construction\n\nBuilt.\n"
    sections = parse_sections(text)
    assert [p for p, _ in sections] == [["Line"], ["Line", "History", "Construction"]]


def test_no_text_lost():
    body = "\n".join(sentence(i) for i in range(30))
    text = f"# Line\n\nLead.\n\n## History\n\n{body}\n\n## Tech\n\nShort line.\n"
    chunks = chunk_document(doc(text), cfg(size=128, overlap=32))
    joined = "\n".join(c["text"] for c in chunks)
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            assert line in joined


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_every_chunk_within_size(size):
    body = "\n".join(sentence(i, words=20) for i in range(300))
    c = cfg(size=size, overlap=64, min_size=64)
    chunks = chunk_document(doc(f"# Line\n\n## History\n\n{body}\n"), c)
    assert len(chunks) > 1
    assert all(c.length(ch["embed_text"]) <= size for ch in chunks)


def test_long_paragraph_splits_on_sentences_and_giant_sentence_on_words():
    long_para = " ".join(sentence(i) for i in range(20))  # one line, many sentences
    giant = " ".join(f"w{j}" for j in range(400)) + "."  # one sentence, too long
    text = f"# Line\n\n## A\n\n{long_para}\n\n## B\n\n{giant}\n"
    chunks = chunk_document(doc(text), cfg(size=64, overlap=0))
    a_chunks = [c for c in chunks if c["heading_path"] == ["Line", "A"]]
    assert len(a_chunks) > 1
    # Every A chunk starts at a sentence boundary.
    assert all(c["text"].startswith("Sentence ") for c in a_chunks)
    b_chunks = [c for c in chunks if c["heading_path"] == ["Line", "B"]]
    assert len(b_chunks) > 1
    # Word split: no word is cut in half.
    words = {w for c in b_chunks for w in c["text"].replace(".", "").split()}
    assert words == {f"w{j}" for j in range(400)}


def test_overlap_within_section_only():
    body = "\n".join(sentence(i) for i in range(12))
    text = f"# Line\n\n## A\n\n{body}\n\n## B\n\n{sentence(99)}\n"
    # Each sentence is ~19 est. tokens, so overlap=24 fits exactly one.
    chunks = chunk_document(doc(text), cfg(size=64, overlap=24))
    a = [c for c in chunks if c["heading_path"] == ["Line", "A"]]
    assert len(a) > 1
    # Consecutive A chunks share whole sentences.
    for prev, nxt in zip(a, a[1:]):
        first_sentence = re.split(r"(?<=\.)\s+", nxt["text"])[0]
        assert first_sentence in prev["text"]
    b = [c for c in chunks if c["heading_path"] == ["Line", "B"]]
    assert b and "Sentence 11" not in b[0]["text"]


def test_overlap_smaller_than_a_sentence_gives_no_overlap():
    # Overlap is whole sentences only, so a sentence longer than the overlap
    # budget is never carried over (documented trade-off, D-004).
    body = "\n".join(sentence(i) for i in range(12))
    chunks = chunk_document(doc(f"# Line\n\n## A\n\n{body}\n"), cfg(size=64, overlap=16))
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["text"].split("\n")[0] not in prev["text"]


def test_small_sections_merge_forward_within_top_level_only():
    text = (
        "# Line\n\nLead sentence here.\n\n"
        "## Service\n\n### Stops\n\nTiny.\n\n### Speed\n\nAlso tiny.\n\n"
        "## Tech\n\nTiny too.\n"
    )
    chunks = chunk_document(doc(text), cfg(size=128, overlap=0, min_size=30))
    paths = [c["heading_path_str"] for c in chunks]
    # Stops + Speed merge under their common parent; Tech and Lead stay separate.
    assert paths == ["Line", "Line > Service", "Line > Tech"]
    service = chunks[1]["text"]
    assert "### Stops\nTiny." in service and "### Speed\nAlso tiny." in service


def test_small_last_chunk_merges_backward():
    text = "# Line\n\n## Service\n\nA normal sized line of text about trains.\n\n### Coda\n\nEnd.\n"
    chunks = chunk_document(doc(text), cfg(size=128, overlap=0, min_size=20))
    assert [c["heading_path_str"] for c in chunks] == ["Line > Service"]
    assert chunks[0]["text"].endswith("### Coda\nEnd.")


def test_ids_unique_and_stable():
    body = "\n".join(sentence(i) for i in range(40))
    d = doc(f"# Line\n\n## A\n\n{body}\n")
    first = chunk_document(d, cfg())
    second = chunk_document(d, cfg())
    ids = [c["chunk_id"] for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c["chunk_id"] for c in second]
    assert ids[0] == "line#0000"


def test_prepend_toggle():
    d = doc("# Line\n\n## A\n\nText.\n")
    on = chunk_document(d, cfg(prepend_headings=True))[0]
    off = chunk_document(d, cfg(prepend_headings=False))[0]
    assert on["embed_text"] == "Line > A\n\nText."
    assert off["embed_text"] == "Text."


@pytest.mark.parametrize(
    "kw", [dict(overlap=64), dict(overlap=100), dict(min_size=64), dict(size=0), dict(unit="words")]
)
def test_config_validation(kw):
    with pytest.raises(ValueError):
        cfg(**kw)
