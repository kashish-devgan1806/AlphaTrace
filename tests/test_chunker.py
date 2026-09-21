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


def test_split_into_sections_ignores_repeated_page_headers():
    """Some filers (Microsoft) print "PART I" / "Item 1A" at the top of every
    page. Those bare lines must not win over the real "Item 1A. ..." heading,
    or every page before the last header gets folded into the prior section."""
    text = (
        "PART I\nItem 1.\nBusiness\nItem 1A.\nRisk Factors\n"  # TOC
        "PART I\nITEM 1. BUSINESS\nbusiness body\n"
        "PART I\nItem 1\nmore business body\n"  # page header
        "PART I\nITEM 1A. RISK FACTORS\nrisk body one\n"
        "PART I\nItem 1A\nrisk body two\n"  # page header
    )

    sections = dict(split_into_sections(text))

    labels = [label for label in sections if label != "Preamble"]
    assert labels == ["Part I — ITEM 1. BUSINESS", "Part I — ITEM 1A. RISK FACTORS"]
    assert "more business body" in sections["Part I — ITEM 1. BUSINESS"]
    assert "risk body two" in sections["Part I — ITEM 1A. RISK FACTORS"]


def test_split_into_sections_normalizes_label_whitespace():
    text = "PART I\nItem 1A.   Risk Factors\nrisk body\n"

    (label, _), = [s for s in split_into_sections(text) if s[0] != "Preamble"]

    assert label == "Part I — Item 1A. Risk Factors"


def test_split_into_sections_bare_item_heading_does_not_absorb_next_line():
    text = "PART I\nItem 1A\nRisk Factors\nrisk body\n"

    (label, _), = [s for s in split_into_sections(text) if s[0] != "Preamble"]

    assert label == "Part I — Item 1A"


def test_strip_html_to_text_drops_hidden_inline_xbrl_header():
    """Modern filings open with a hidden ix:header block (taxonomy URLs,
    context ids). It must not reach the chunks, or it gets embedded."""
    html = (
        "<html><body>"
        "<div style='display:none'><ix:header><ix:hidden>"
        "<ix:nonnumeric>http://fasb.org/us-gaap/2025#LongTermDebt</ix:nonnumeric>"
        "</ix:hidden></ix:header></div>"
        "<div>FORM 10-K</div>"
        "</body></html>"
    )

    text = strip_html_to_text(html)

    assert "fasb.org" not in text
    assert "FORM 10-K" in text


def test_strip_html_to_text_drops_script_and_style_but_keeps_following_text():
    html = "<style>p {color: red}</style><script>var x = 1;</script><p>real text</p>"

    text = strip_html_to_text(html)

    assert text == "real text"


def _words(text: str) -> int:
    return len(text.split())


def test_chunk_filing_splits_pieces_over_the_token_limit_without_losing_text():
    """The char budget is only a proxy: dense text can blow the model's token
    limit inside it. Every piece must be held to max_tokens."""
    body = " ".join(f"w{i}" for i in range(400))  # ~1.6k chars, 400 "tokens"
    html = f"<div>Item 1A. Risk Factors</div><p>{body}</p>"

    records = chunk_filing("doc", html, max_chars=5000, max_tokens=100, count_tokens=_words)

    risk = [r for r in records if "w0" in r.text or r.section.endswith("Risk Factors")]
    assert risk
    assert all(_words(r.text) <= 100 for r in records if len(r.text) > 600)
    rebuilt = " ".join(r.text for r in records if r.section.endswith("Risk Factors"))
    assert all(f"w{i}" in rebuilt.split() for i in range(400))


def test_chunk_filing_never_tokenizes_short_chunks():
    calls = []

    def spy(text: str) -> int:
        calls.append(text)
        return 10_000

    chunk_filing("doc", "<div>Item 1A. Risk Factors</div><p>short text</p>", count_tokens=spy)

    assert calls == []  # under the length floor: the model is never consulted


def test_strip_html_to_text_normalizes_nbsp_and_space_runs():
    text = strip_html_to_text("<p>Item 8.   Financial   Statements $60.0 billion</p>")

    assert text == "Item 8. Financial Statements $60.0 billion"


def test_chunk_filing_strips_running_page_headers_from_chunk_text():
    html = (
        "<div>PART I</div><div>ITEM 1A. RISK FACTORS</div>"
        "<p>First page of risk text that is long enough to stand on its own as a real paragraph here.</p>"
        "<div>24</div><div>PART I</div><div>Item 1A</div>"
        "<p>Second page of risk text that is also long enough to stand on its own as a real paragraph.</p>"
    )

    (record,) = [r for r in chunk_filing("doc", html, count_tokens=_words) if "risk text" in r.text]

    assert "PART I" not in record.text
    assert "\nItem 1A\n" not in record.text
    assert "First page" in record.text and "Second page" in record.text


def test_chunk_filing_keeps_a_lone_page_number_that_is_table_data():
    html = (
        "<div>ITEM 8. FINANCIAL STATEMENTS</div>"
        "<p>Revenue for the year, in millions of dollars, by reporting segment as shown in the table below:</p>"
        "<div>24</div><div>31</div>"
    )

    text = " ".join(r.text for r in chunk_filing("doc", html, count_tokens=_words))

    assert "24" in text and "31" in text


def test_merge_tiny_pieces_folds_fragments_into_neighbours_but_keeps_a_lone_piece():
    from app.chunker import _merge_tiny_pieces

    big = "x" * 100
    assert _merge_tiny_pieces(["tiny"]) == ["tiny"]
    assert _merge_tiny_pieces([big, "-", big]) == [big + "\n\n-", big]
    assert _merge_tiny_pieces(["-", big]) == ["-\n\n" + big]

