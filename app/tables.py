"""Structure-preserving table extraction for filing financial statements.

app/chunker.py's strip_html_to_text() flattens <table>/<tr> content into
plain lines with no cell boundary (<td>/<th> aren't in its _BLOCK_TAGS) --
fine for prose chunking, useless for a financial-statement table where
which number sits in which column is the whole point. This module walks
<table> DOM directly instead, keeping each table's row/cell shape, then
serializes it to a Markdown table (kept embeddable/legible as a
ChunkRecord's text, reusing the existing chunks/embedding/search path
unchanged) with the raw rows kept alongside for anything that wants the
structure back verbatim.

Which (Part, Item) section a table belongs to is decided the same way
app.chunker.split_into_sections() decides it for prose: a single parse
pass produces plain text with a marker in place of each table, that text
is split into sections with the existing function, and each table is
attributed to whichever section its marker landed in. This runs as its
own, separate extraction pass rather than changing strip_html_to_text
itself, so nothing about existing chunking behavior/tests changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

from app.chunker import split_into_sections

_BLOCK_TAGS = {"p", "div", "tr", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"ix:header", "script", "style"}
_TABLE_MARKER_RE = re.compile(r"\x00TABLE_(\d+)\x00")

# A layout-only table (one cell, a spacer row) isn't worth indexing as a
# financial-statement table -- skip anything under this many non-empty
# cells rather than embedding noise.
_MIN_NON_EMPTY_CELLS = 4


@dataclass
class TableRecord:
    """One extracted <table>, ready to become a table-type ChunkRecord."""

    doc_id: str
    section: str
    text: str  # Markdown-serialized table
    rows: list[list[str]]
    metadata: dict = field(default_factory=dict)


class _TableAwareExtractor(HTMLParser):
    """Single pass over filing HTML: flattened text (each top-level <table>
    replaced by a "\\x00TABLE_N\\x00" marker) plus every table's row/cell
    text, both in document order. A table nested inside another table's
    cell (rare, layout-only markup) is folded into that enclosing cell's
    text rather than captured as its own separate table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_tag: Optional[str] = None
        self._skip_depth = 0
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: Optional[list[list[str]]] = None
        self._current_row: Optional[list[str]] = None
        self._cell_parts: Optional[list[str]] = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _SKIP_TAGS:
            self._skip_tag = tag
            self._skip_depth = 1
            return
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
                self.tables.append(self._current_table)
                self._parts.append(f"\n\x00TABLE_{len(self.tables) - 1}\x00\n")
            return
        if self._table_depth >= 1:
            if self._table_depth == 1:
                if tag == "tr":
                    self._current_row = []
                    self._current_table.append(self._current_row)
                elif tag in ("td", "th"):
                    self._in_cell = True
                    self._cell_parts = []
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag == "table":
            if self._table_depth >= 1:
                self._table_depth -= 1
            return
        if self._table_depth >= 1:
            if self._table_depth == 1 and tag in ("td", "th") and self._in_cell:
                text = " ".join("".join(self._cell_parts).split())
                self._current_row.append(text)
                self._in_cell = False
                self._cell_parts = None
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_tag is not None:
            return
        if self._in_cell:
            self._cell_parts.append(data)
            return
        if self._table_depth >= 1:
            return  # whitespace/text between table-internal tags, outside any cell
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _to_markdown(rows: list[list[str]]) -> str:
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]

    def esc(cell: str) -> str:
        return cell.strip().replace("|", "\\|")

    lines = ["| " + " | ".join(esc(c) for c in padded[0]) + " |", "| " + " | ".join("---" for _ in range(width)) + " |"]
    for row in padded[1:]:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    return "\n".join(lines)


_TABLE_TAG_RE = re.compile(r"<(/?)table\b[^>]*>", re.IGNORECASE)


def strip_tables(html: str) -> str:
    """Remove every top-level <table>...</table> subtree (nested tables
    included) from filing HTML. Meant to run before chunk_filing() on HTML
    that's also going through extract_tables(): without this, the same
    table's cell text shows up twice -- once garbled (app.chunker's
    _TextExtractor has no cell separator between <td>s) inside a prose
    chunk, once cleanly as its own table chunk here."""
    pieces: list[str] = []
    depth = 0
    last_end = 0
    for match in _TABLE_TAG_RE.finditer(html):
        is_close = bool(match.group(1))
        if not is_close:
            if depth == 0:
                pieces.append(html[last_end : match.start()])
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                last_end = match.end()
    pieces.append(html[last_end:])
    return "".join(pieces)


def extract_tables(doc_id: str, html: str, metadata: Optional[dict] = None) -> list[TableRecord]:
    """Filing HTML -> TableRecord list, one per real <table> element with
    at least _MIN_NON_EMPTY_CELLS of content, tagged with the (Part, Item)
    section it falls under.

    No token/char budget is applied here the way chunk_section_text() caps
    prose chunks -- splitting a table mid-row would destroy the structure
    this function exists to preserve. An unusually large table still goes
    through embed_texts() -> app.embeddings._warn_on_truncation(), same
    safety net every chunk already relies on.
    """
    parser = _TableAwareExtractor()
    parser.feed(html)
    text = parser.get_text().replace("\xa0", " ")
    sections = split_into_sections(text)

    records: list[TableRecord] = []
    for section_label, section_text in sections:
        for match in _TABLE_MARKER_RE.finditer(section_text):
            index = int(match.group(1))
            rows = parser.tables[index]
            non_empty_cells = sum(1 for row in rows for cell in row if cell.strip())
            if non_empty_cells < _MIN_NON_EMPTY_CELLS:
                continue
            markdown = _to_markdown(rows)
            if not markdown:
                continue
            records.append(
                TableRecord(
                    doc_id=doc_id,
                    section=section_label,
                    text=markdown,
                    rows=rows,
                    metadata=dict(metadata or {}),
                )
            )
    return records
