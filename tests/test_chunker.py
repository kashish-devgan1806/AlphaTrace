"""Offline tests for app/chunker.py.

FAKE_10K_HTML below deliberately mimics the real shape that makes this hard:
a table of contents (repeating "PART I" / "Item 1." / "Item 1A." / "PART II"
/ "Item 7." a second time, with a trailing page number) *before* the real
sections — exercising the TOC-dedup logic split_into_sections relies on,
not just the happy path of one heading per item.
"""
from __future__ import annotations

from app.chunker import chunk_filing, chunk_section_text, split_into_sections, strip_html_to_text

FAKE_10K_HTML = """
<html><body>
<div>TABLE OF CONTENTS</div>
<div>PART I</div>
<div>Item 1. Business ... 3</div>
<div>Item 1A. Risk Factors ... 10</div>
<div>PART II</div>
<div>Item 7. Management's Discussion and Analysis ... 40</div>

<div>PART I</div>
<div>Item 1. Business</div>
<p>We design, manufacture and market widgets.</p>
<p>Our widgets are sold worldwide through retail and online channels.</p>

<div>Item 1A. Risk Factors</div>
<p>Our business is subject to numerous risks, including competition and supply chain disruption.</p>

<div>PART II</div>
<div>Item 7. Management's Discussion and Analysis</div>
<p>Revenue increased year over year, driven by higher unit sales.</p>
</body></html>
"""


def test_strip_html_to_text_puts_block_content_on_its_own_line():
    text = strip_html_to_text("<div>Item 1. Business</div><p>We design widgets.</p>")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines == ["Item 1. Business", "We design widgets."]


def test_split_into_sections_dedupes_table_of_contents():
    sections = split_into_sections(strip_html_to_text(FAKE_10K_HTML))
    labels = [label for label, _ in sections]

    # Exactly one section per real (Part, Item) pair — the TOC's earlier
    # occurrences of the same headings must not produce extra sections.
    assert labels == [
        "Preamble",
        "Part I — Item 1. Business",
        "Part I — Item 1A. Risk Factors",
        "Part II — Item 7. Management's Discussion and Analysis",
    ]


def test_split_into_sections_preamble_holds_the_toc_text():
    sections = dict(split_into_sections(strip_html_to_text(FAKE_10K_HTML)))
    assert "TABLE OF CONTENTS" in sections["Preamble"]


def test_split_into_sections_real_section_excludes_toc_page_numbers():
    sections = dict(split_into_sections(strip_html_to_text(FAKE_10K_HTML)))
    business_section = sections["Part I — Item 1. Business"]

    assert "We design, manufacture and market widgets." in business_section
    assert "... 3" not in business_section  # that's the TOC line's trailing page number


def test_split_into_sections_no_item_headings_returns_full_document():
    sections = split_into_sections("Just some plain filing text with no Item headings at all.")
    assert len(sections) == 1
    assert sections[0][0] == "Full Document"


def test_split_into_sections_empty_text_returns_nothing():
    assert split_into_sections("") == []


def test_chunk_section_text_packs_paragraphs_under_the_limit():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_section_text(text, max_chars=1000)
    assert chunks == [text]  # all three paragraphs fit in one chunk


def test_chunk_section_text_splits_when_over_the_limit():
    para_a = "A" * 60
    para_b = "B" * 60
    chunks = chunk_section_text(f"{para_a}\n\n{para_b}", max_chars=100)
    assert len(chunks) == 2
    assert chunks[0] == para_a
    assert chunks[1] == para_b


def test_chunk_section_text_hard_splits_an_oversized_paragraph():
    huge_para = "X" * 250
    chunks = chunk_section_text(huge_para, max_chars=100)
    assert len(chunks) == 3
    assert "".join(chunks) == huge_para
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_filing_attaches_metadata_and_doc_id_to_every_chunk():
    records = chunk_filing("0000320193-25-000100", FAKE_10K_HTML, metadata={"ticker": "AAPL", "form": "10-K"})

    assert len(records) > 0
    assert all(r.doc_id == "0000320193-25-000100" for r in records)
    assert all(r.metadata == {"ticker": "AAPL", "form": "10-K"} for r in records)

    sections_seen = {r.section for r in records}
    assert "Part I — Item 1A. Risk Factors" in sections_seen


def test_chunk_filing_metadata_dicts_are_independent_copies():
    """Each ChunkRecord must get its own metadata dict — mutating one row's
    metadata (e.g. adding a chunk-index later) must not leak into others."""
    records = chunk_filing("doc-1", FAKE_10K_HTML, metadata={"ticker": "AAPL"})
    records[0].metadata["chunk_index"] = 0

    assert "chunk_index" not in records[1].metadata


def test_chunk_filing_defaults_metadata_to_empty_dict():
    records = chunk_filing("doc-1", "<p>Item 1. Business</p><p>Some text.</p>")
    assert all(r.metadata == {} for r in records)
