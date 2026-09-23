"""Section-aware chunker for filing documents.

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
from typing import Callable, Optional

from app.chunks import ChunkRecord

# ~4 chars/token is the standard rule of thumb for English prose under a
# BPE/wordpiece tokenizer. 1600 chars ≈ 400 tokens, leaving ~22% headroom
# under bge-small-en-v1.5's 512-token max_seq_length — margin for the
# average running shorter than 4 chars/token (numbers, punctuation-heavy
# financial text), without relying on embeddings._warn_on_truncation as the
# only backstop.
DEFAULT_MAX_CHUNK_CHARS = 1600

# The char budget above is a proxy; this is the real limit, checked with the
# model's own tokenizer. 480 leaves headroom under bge-small's 512 for the
# special tokens and for count drift. Chunks shorter than
# _TOKEN_CHECK_MIN_CHARS are never tokenized (they can't come near 512 tokens
# even at ~1 char/token), which keeps the model unloaded for tiny inputs.
DEFAULT_MAX_CHUNK_TOKENS = 480
_TOKEN_CHECK_MIN_CHARS = 600

# EDGAR filings mark Part/Item boundaries as plain-text headings ("PART I",
# "Item 1A. Risk Factors") — not semantic HTML (no <h1>/<section> tag to key
# off), so matched text is the only reliable signal, once HTML is stripped
# to one block element's content per line below.
_PART_RE = re.compile(r"^\s*PART\s+(I{1,3}V?)\b", re.IGNORECASE | re.MULTILINE)
# Whitespace between tokens is `[^\S\n]` (any whitespace except a newline,
# NBSP included): a plain `\s` would let a bare "Item 1" line run on into the
# next line's text and swallow it into the heading. Group 2 captures the
# period after the item number, which real headings ("Item 1A. Risk
# Factors") have and running page headers / cross-references ("Item 1A") don't.
_ITEM_RE = re.compile(
    r"^[^\S\n]*Item[^\S\n]+(\d{1,2}[A-C]?)(\.?)[^\S\n]*[-–—:]*[^\S\n]*(.{0,120})$",
    re.IGNORECASE | re.MULTILINE,
)


class _TextExtractor(HTMLParser):
    """Strips tags, keeps block-level elements on their own line.

    Why the stdlib parser instead of a proper HTML library (BeautifulSoup,
    lxml): the only need here is "get item headings onto their own line so
    the regexes above can find them" — not DOM traversal, table layout, or
    malformed-markup recovery. Adding a parsing dependency for that would be
    more than this session needs.
    """

    _BLOCK_TAGS = {"p", "div", "tr", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    # Elements whose content is never filing prose. `ix:header` is the hidden
    # inline-XBRL block every modern filing opens with (context ids, taxonomy
    # URLs, dei facts) — left in, it becomes garbage Preamble chunks that get
    # embedded and show up in search results. script/style are plain noise.
    _SKIP_TAGS = {"ix:header", "script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_tag: Optional[str] = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in self._SKIP_TAGS:
            self._skip_tag = tag
            self._skip_depth = 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_tag is None:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html_to_text(html: str) -> str:
    """Filing HTML -> plain text, one block element's content per line."""
    parser = _TextExtractor()
    parser.feed(html)
    # Non-breaking spaces (filings use them everywhere: "$60.0&nbsp;billion",
    # "Item&nbsp;8.") become plain spaces so text and labels compare and
    # embed consistently.
    text = parser.get_text().replace("\xa0", " ")

    # Block-tag boundaries produce runs of blank lines (e.g. a <div> end
    # immediately followed by a <p> start). Collapse each run to a single
    # blank line — enough to mark a paragraph break for chunk_section_text
    # below, without losing the separation entirely.
    lines = [" ".join(line.split()) for line in text.splitlines()]
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
    item_events = [
        (m.start(), m.group(1).upper(), " ".join(m.group(0).split()), bool(m.group(2)))
        for m in _ITEM_RE.finditer(text)
    ]

    if not item_events:
        stripped = text.strip()
        return [("Full Document", stripped)] if stripped else []

    current_part = ""
    part_iter = iter(sorted(part_events))
    next_part = next(part_iter, None)

    candidates: dict[str, list[tuple[int, str, bool]]] = {}
    for start_pos, item_num, heading_line, has_period in sorted(item_events):
        while next_part is not None and next_part[0] < start_pos:
            current_part = f"Part {next_part[1]}"
            next_part = next(part_iter, None)
        key = f"{current_part}|{item_num}"
        label = f"{current_part} — {heading_line}" if current_part else heading_line
        candidates.setdefault(key, []).append((start_pos, label, has_period))

    # Last occurrence of a key wins (drops the earlier TOC entry) — but only
    # among real headings, i.e. those with a period after the item number.
    # Some filers (Microsoft) repeat a running page header ("PART I" / "Item
    # 1A") on every page; without this preference the *last page header*
    # would win, and every page before it would be folded into whatever
    # section came before. A key with no period-style occurrence at all
    # falls back to its last occurrence.
    keyed: dict[str, tuple[int, str]] = {}
    for key, occurrences in candidates.items():
        pool = [o for o in occurrences if o[2]] or occurrences
        start_pos, label, _ = pool[-1]
        keyed[key] = (start_pos, label)

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


# A running page header as it appears once flattened to text: an optional
# page number, a bare "PART II" line, and an optional bare "Item 7" line
# (no period, no title — real headings have both). Run only on section
# bodies, after split_into_sections() has used the PART lines to track the
# current part. A lone page number is deliberately not stripped on its own:
# table cells also sit alone on a line, and "24" there is data.
_PAGE_HEADER_RE = re.compile(
    r"^(?:\d{1,3}[ \t]*\n(?:[ \t]*\n)*)?"
    r"PART[ \t]+(?:I{1,3}|IV)[ \t]*(?:\n|\Z)(?:[ \t]*\n)*"
    r"(?:Item[ \t]+\d{1,2}[A-C]?[ \t]*(?:\n|\Z)(?:[ \t]*\n)*)?",
    re.IGNORECASE | re.MULTILINE,
)

# Pieces shorter than this that share a section with a neighbour are merged
# into it: a stray "•" or a heading fragment embeds as noise and matches
# queries on nothing. A section that is a single short piece ("Item 1B ...
# None.") is left alone — that is the whole section, not a fragment.
_MIN_CHUNK_CHARS = 80


def _strip_page_headers(section_text: str) -> str:
    return _PAGE_HEADER_RE.sub("", section_text).strip()


def _merge_tiny_pieces(pieces: list[str], min_chars: int = _MIN_CHUNK_CHARS) -> list[str]:
    """Fold each piece under min_chars into its neighbour (the previous one,
    or the next if it is first). Never touches a lone piece."""
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        if carry:
            piece = f"{carry}\n\n{piece}"
            carry = ""
        if len(piece) < min_chars and not merged:
            carry = piece  # first piece is tiny: prepend it to the next one
        elif len(piece) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    if carry:  # every piece was tiny (or only the first was, with no next)
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{carry}"
        else:
            merged.append(carry)
    return merged


def _default_count_tokens(text: str) -> int:
    # Imported on first use so the chunker (and its tests) don't load the
    # embedding model unless a chunk is actually big enough to need checking.
    from app.embeddings import count_tokens

    return count_tokens(text)


def _split_to_token_limit(
    chunk: str, count_tokens: Callable[[str], int], max_tokens: int
) -> list[str]:
    """Split `chunk` until every piece is <= max_tokens under the model's
    tokenizer. The 1600-char budget is only a proxy: number-heavy tables run
    ~2 chars/token, so a "1600-char" chunk can be 700+ tokens, and the
    embedding model silently drops everything past token 512.

    Splits at the separator nearest the middle (paragraph break, then line
    break, then space) and recurses, so pieces stay balanced and boundaries
    land on natural breaks. A chunk with no separator at all is halved.
    """
    if len(chunk) <= _TOKEN_CHECK_MIN_CHARS or count_tokens(chunk) <= max_tokens:
        return [chunk]

    mid = len(chunk) // 2
    cut = None
    for sep in ("\n\n", "\n", " "):
        positions = [i for i in range(len(chunk)) if chunk.startswith(sep, i)]
        if positions:
            cut = min(positions, key=lambda i: abs(i - mid))
            left, right = chunk[:cut], chunk[cut + len(sep) :]
            break
    if cut is None or not left.strip() or not right.strip():
        left, right = chunk[:mid], chunk[mid:]

    return _split_to_token_limit(left, count_tokens, max_tokens) + _split_to_token_limit(
        right, count_tokens, max_tokens
    )


def chunk_filing(
    doc_id: str,
    html: str,
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    count_tokens: Optional[Callable[[str], int]] = None,
    metadata: Optional[dict] = None,
) -> list[ChunkRecord]:
    """Filing HTML -> ChunkRecord list, ready for batch_insert_chunks().

    metadata is copied onto every chunk from this filing (e.g. {"ticker":
    "AAPL", "form": "10-K", "filing_date": "2025-11-01"}) — the chunks
    table has no join back to a documents table (out of scope for Month 1,
    per db/init/02_create_chunks_table.sql), so this is the only place that
    context can attach.

    Every piece is also held to `max_tokens` (measured with `count_tokens`,
    default: the embedding model's tokenizer) so nothing is truncated at
    embedding time.
    """
    counter = count_tokens or _default_count_tokens
    text = strip_html_to_text(html)
    sections = split_into_sections(text)

    records: list[ChunkRecord] = []
    for section_name, section_text in sections:
        section_text = _strip_page_headers(section_text)
        pieces = _merge_tiny_pieces(chunk_section_text(section_text, max_chars=max_chars))
        for piece in pieces:
            for sub_piece in _split_to_token_limit(piece, counter, max_tokens):
                records.append(
                    ChunkRecord(
                        doc_id=doc_id, section=section_name, text=sub_piece, metadata=dict(metadata or {})
                    )
                )
    return records
