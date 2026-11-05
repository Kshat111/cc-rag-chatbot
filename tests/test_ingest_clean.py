"""Tests for clean_wikipedia_extract: heading conversion and section removal."""

from ragbot.ingest import clean_wikipedia_extract

DROP = ["References", "See also", "External links"]


def clean(text):
    return clean_wikipedia_extract("Page", text, DROP)


def test_headings_converted_with_levels_kept():
    text = "Lead.\n\n== History ==\nOld.\n\n=== Construction ===\nBuilt.\n"
    assert clean(text) == (
        "# Page\n\nLead.\n\n## History\n\nOld.\n\n### Construction\n\nBuilt.\n"
    )


def test_dropped_section_removes_its_subsections_too():
    text = (
        "Lead.\n== See also ==\nLink A\n=== More ===\nLink B\n"
        "== Legacy ==\nKept.\n"
    )
    out = clean(text)
    assert "See also" not in out and "Link A" not in out and "Link B" not in out
    assert "## Legacy\n\nKept." in out


def test_drop_match_is_case_insensitive():
    out = clean("Lead.\n== external links ==\nhttp://x\n")
    assert "external links" not in out and "http://x" not in out


def test_empty_heading_removed_but_parent_of_content_kept():
    text = (
        "Lead.\n== Empty ==\n\n== Parent ==\n=== Child ===\nText.\n"
        "=== Empty child ===\n\n"
    )
    out = clean(text)
    assert "Empty" not in out
    assert "## Parent\n\n### Child\n\nText." in out


def test_section_with_only_bare_label_is_removed():
    text = "Lead.\n== Stations ==\nLegend:\n\n== History ==\nOpened.\nService column legend:\n"
    out = clean(text)
    assert "Stations" not in out and "Legend:" not in out and "legend:" not in out
    assert out.endswith("## History\n\nOpened.\n")


def test_label_followed_by_text_and_trailing_sentence_are_kept():
    text = (
        "Lead.\n== Ridership ==\nNotes:\nData in italics is estimated.\n"
        "== Sets ==\nThe sets were formed as follows:\n"
    )
    out = clean(text)
    assert "Notes:\nData in italics is estimated." in out
    assert out.endswith("## Sets\n\nThe sets were formed as follows:\n")


def test_zero_width_removed_and_odd_spaces_normalised():
    out = clean("Lead﻿ text.­\n== A ==\n18.4 km, lit. 'new'\n")
    assert "Lead text." in out and "18.4 km, lit. 'new'" in out
    assert not any(c in out for c in "﻿­  ")


def test_coordinate_duplicate_dropped_readable_form_kept():
    text = "Lead.\n== A ==\nBetween Ōtsuki and Tsuru (35.583°N 138.927°E﻿ / 35.583; 138.927﻿ (Yamanashi Test Track)). Next.\n"
    out = clean(text)
    assert "Between Ōtsuki and Tsuru (35.583°N 138.927°E). Next." in out


def test_flat_exponent_rewritten_only_after_decimal_mantissa():
    out = clean("Lead.\n== A ==\nAbout ¥1.239294631884×1012 in 2024; a 3×100 m relay.\n")
    assert "¥1.239294631884×10^12 in 2024" in out
    assert "3×100 m relay" in out


def test_footnote_lines_dropped_but_symbols_mid_line_kept():
    text = (
        "Lead.\n== Ridership ==\nTable prose.\n* The sum of lines.\n** Only six days.\n"
        "† Indicates sources differ.\nSpeed is 5 * 2 † units.\n*Emphasis not a footnote\n"
    )
    out = clean(text)
    assert "The sum" not in out and "Only six" not in out and "Indicates" not in out
    assert "Speed is 5 * 2 † units." in out and "*Emphasis not a footnote" in out


def test_blank_line_runs_collapsed_and_single_trailing_newline():
    out = clean("Lead.\n\n\n\n\nMore.\n\n\n")
    assert out == "# Page\n\nLead.\n\nMore.\n"
