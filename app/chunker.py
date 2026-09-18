"""Section-aware chunker for filing documents (Session 5).

Turns a filing's raw primary-document HTML (10-K/10-Q) into ChunkRecord
objects ready for app.chunks.batch_insert_chunks() — splitting first by
Part/Item section (the `section` column, db/init/02_create_chunks_table.sql,
is NOT NULL specifically anticipating this), then packing each section's
text into pieces sized for bge-small-en-v1.5 (app/embeddings.py, 512-token
max_seq_length).
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional

from app.chunks import ChunkRecord

# ~4 chars/token is the standard rule of thumb for English prose under a
# BPE/wordpiece tokenizer. 1600 chars ≈ 400 tokens, leaving ~22% headroom
# under bge-small-en-v1.5's 512-token max_seq_length — margin for the
# average running shorter than 4 chars/token (numbers, punctuation-heavy
# financial text), without relying on embeddings._warn_on_truncation as the
# only backstop.
DEFAULT_MAX_CHUNK_CHARS = 1600

# EDGAR filings mark Part/Item boundaries as plain-text headings ("PART I",
# "Item 1A. Risk Factors") — not semantic HTML (no <h1>/<section> tag to key
# off), so matched text is the only reliable signal, once HTML is stripped
# to one block element's content per line below.
_PART_RE = re.compile(r"^\s*PART\s+(I{1,3}V?)\b", re.IGNORECASE | re.MULTILINE)
_ITEM_RE = re.compile(r"^\s*Item\s+(\d{1,2}[A-C]?)\.?\s*[-–—:]*\s*(.{0,120})$", re.IGNORECASE | re.MULTILINE)


class _TextExtractor(HTMLParser):
    """Strips tags, keeps block-level elements on their own line.

    Why the stdlib parser instead of a proper HTML library (BeautifulSoup,
    lxml): the only need here is "get item headings onto their own line so
    the regexes above can find them" — not DOM traversal, table layout, or
    malformed-markup recovery. Adding a parsing dependency for that would be
    more than this session needs.
    """

    _BLOCK_TAGS = {"p", "div", "tr", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html_to_text(html: str) -> str:
    """Filing HTML -> plain text, one block element's content per line."""
    parser = _TextExtractor()
    parser.feed(html)
    text = parser.get_text()

    # Block-tag boundaries produce runs of blank lines (e.g. a <div> end
    # immediately followed by a <p> start). Collapse each run to a single
    # blank line — enough to mark a paragraph break for chunk_section_text
    # below, without losing the separation entirely.
    lines = [line.strip() for line in text.splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if line == "" and (not collapsed or collapsed[-1] == ""):
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Plain filing text -> [(section_label, section_text), ...], in filing order.

    Two EDGAR-specific wrinkles this handles:

    1. Every 10-K/10-Q has a table of contents near the top that lists every
       Item heading a second time, before the real section. A naive "split
       on every Item match" would treat each TOC line as its own
       (near-empty) section. Since the TOC always precedes the real
       section, keeping only the *last* match for a given (Part, Item) key
       discards the TOC occurrence and keeps the real one.
    2. 10-Qs reuse Item numbers across parts (Item 1 = "Financial
       Statements" in Part I, "Legal Proceedings" in Part II) — so the
       dedup key is (current Part, Item number), tracked by walking Part
       headings and Item headings together in document order.

    Known limitation: the discarded TOC text isn't dropped, just folded
    into whichever section precedes the real first Item match (usually a
    "Preamble" section covering the cover page + TOC together). Precisely
    excising the TOC would need real document structure (page breaks, a
    <table> boundary) that a plain-text regex pass doesn't have — out of
    scope for this session.
    """
    part_events = [(m.start(), m.group(1).upper()) for m in _PART_RE.finditer(text)]
    item_events = [(m.start(), m.group(1).upper(), m.group(0).strip()) for m in _ITEM_RE.finditer(text)]

    if not item_events:
        stripped = text.strip()
        return [("Full Document", stripped)] if stripped else []

    current_part = ""
    part_iter = iter(sorted(part_events))
    next_part = next(part_iter, None)

    keyed: dict[str, tuple[int, str]] = {}
    for start_pos, item_num, heading_line in sorted(item_events):
        while next_part is not None and next_part[0] < start_pos:
            current_part = f"Part {next_part[1]}"
            next_part = next(part_iter, None)
        key = f"{current_part}|{item_num}"
        label = f"{current_part} — {heading_line}" if current_part else heading_line
        keyed[key] = (start_pos, label)  # last occurrence of this key wins (drops earlier TOC entries)

    ordered = sorted(keyed.values(), key=lambda pair: pair[0])

    sections: list[tuple[str, str]] = []
    preamble = text[: ordered[0][0]].strip()
    if preamble:
        sections.append(("Preamble", preamble))

    for i, (start_pos, label) in enumerate(ordered):
        end_pos = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        section_text = text[start_pos:end_pos].strip()
        if section_text:
            sections.append((label, section_text))

    return sections


def chunk_section_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Pack a section's paragraphs into <= max_chars pieces.

    Paragraph-aligned (splits on blank lines), not fixed-width: greedily
    fills each chunk with whole paragraphs so a chunk boundary lands between
    ideas rather than mid-sentence, whenever the section's paragraphs allow
    it. A single paragraph longer than max_chars (not unusual in a dense
    MD&A section) still needs a hard split, since there's no smaller
    natural unit to fall back to within one paragraph.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(current))

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            current.clear()
            current_len = 0
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue

        added_len = len(para) + (2 if current else 0)  # +2 for the "\n\n" join
        if current and current_len + added_len > max_chars:
            flush()
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += added_len

    flush()
    return chunks


def chunk_filing(
    doc_id: str,
    html: str,
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    metadata: Optional[dict] = None,
) -> list[ChunkRecord]:
    """Filing HTML -> ChunkRecord list, ready for batch_insert_chunks().

    metadata is copied onto every chunk from this filing (e.g. {"ticker":
    "AAPL", "form": "10-K", "filing_date": "2025-11-01"}) — the chunks
    table has no join back to a documents table (out of scope for Month 1,
    per db/init/02_create_chunks_table.sql), so this is the only place that
    context can attach.
    """
    text = strip_html_to_text(html)
    sections = split_into_sections(text)

    records: list[ChunkRecord] = []
    for section_name, section_text in sections:
        for piece in chunk_section_text(section_text, max_chars=max_chars):
            records.append(
                ChunkRecord(doc_id=doc_id, section=section_name, text=piece, metadata=dict(metadata or {}))
            )
    return records
