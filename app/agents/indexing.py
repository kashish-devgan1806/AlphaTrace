"""Indexing Agent: chunks, tables, and visual pages, all produced from
Ingestion's document_bundle for pgvector to be embedded from.

Grown up from f5/f6's single-form-only index_node the same way a1d1 grew
ingest_node up from its own v0: every filing form present in
document_bundle["filings"] is chunked now (text via app.chunker, tables via
app.tables), not just state["form"]'s primary form. The slide deck, when
present, is either rasterized to page images for a later visual embedding
(a PDF exhibit) or chunked as plain text (an HTML exhibit -- it has no
natural page boundary to rasterize).

Never raises: a missing document_bundle is recorded in `errors` and
returned early; a single form's or the slide deck's own failure is
recorded but doesn't stop the other pieces from being indexed. Embedding
and writing to pgvector are a separate step (app.chunks.batch_insert_chunks,
app.visual.batch_insert_page_embeddings) -- this node only produces the
not-yet-embedded records, mirroring how `ingest` only fetches and never
touches the database either.
"""
from __future__ import annotations

from app.chunker import chunk_filing
from app.chunks import ChunkRecord
from app.state import AgentState
from app.tables import extract_tables, strip_tables
from app.visual import PageImage, rasterize_pdf

FORMS = ("10-K", "10-Q", "8-K")


def _chunk_form(doc_id: str, form: str, entry: dict, errors: list[str]) -> list[ChunkRecord]:
    """One filing form's text + table chunks. `entry` is
    document_bundle["filings"][form] -- {"html", "metadata"}."""
    html = entry["html"]
    metadata = entry["metadata"]

    try:
        text_records = chunk_filing(doc_id, strip_tables(html), metadata=metadata)
    except Exception as exc:
        errors.append(f"index: text chunking failed for {form}: {exc}")
        text_records = []

    try:
        tables = extract_tables(doc_id, html, metadata=metadata)
    except Exception as exc:
        errors.append(f"index: table extraction failed for {form}: {exc}")
        tables = []
    table_records = [
        ChunkRecord(
            doc_id=doc_id,
            section=table.section,
            text=table.text,
            chunk_type="table",
            metadata={**table.metadata, "rows": len(table.rows), "cols": max((len(r) for r in table.rows), default=0)},
        )
        for table in tables
    ]

    return text_records + table_records


def _index_slide_deck(
    slide_deck: dict, bundle_metadata: dict, errors: list[str]
) -> tuple[list[ChunkRecord], list[PageImage]]:
    """The 8-K's slide-deck exhibit, if present. An HTML exhibit is text --
    chunked like any other document, no visual embedding. A PDF exhibit has
    real pages -- rasterized here, embedded/inserted by a later step."""
    doc_id = slide_deck.get("exhibit_name") or slide_deck.get("url") or "slide-deck"
    metadata = {
        "ticker": bundle_metadata.get("ticker"),
        "exhibit_name": slide_deck.get("exhibit_name"),
        "exhibit_type": slide_deck.get("exhibit_type"),
        "url": slide_deck.get("url"),
    }

    if slide_deck["content_type"] == "html":
        try:
            records = chunk_filing(doc_id, slide_deck["html"], metadata=metadata)
        except Exception as exc:
            errors.append(f"index: slide-deck HTML chunking failed: {exc}")
            return [], []
        return records, []

    try:
        images = rasterize_pdf(slide_deck["bytes"])
    except Exception as exc:
        errors.append(f"index: slide-deck PDF rasterization failed: {exc}")
        return [], []
    pages = [
        PageImage(doc_id=doc_id, page_number=page_number, image=image, metadata={**metadata, "page_number": page_number})
        for page_number, image in enumerate(images, start=1)
    ]
    return [], pages


def index_node(state: AgentState) -> dict:
    """Chunk every filing form present in ingest's document_bundle (text +
    tables), plus the slide deck (page images for a PDF exhibit, text
    chunks for an HTML one). Guards on a missing document_bundle instead of
    raising a KeyError."""
    bundle = state.get("document_bundle")
    if not bundle:
        return {"errors": ["index: no document_bundle to chunk (ingest likely failed)"]}

    filings_meta = state.get("filings") or {}
    bundle_filings = bundle.get("filings") or {}
    errors: list[str] = []

    chunks: list[ChunkRecord] = []
    for form in FORMS:
        filing = filings_meta.get(form)
        entry = bundle_filings.get(form)
        if not filing or not entry:
            continue
        chunks.extend(_chunk_form(filing["accessionNumber"], form, entry, errors))

    visual_pages: list[PageImage] = []
    slide_deck = bundle.get("slide_deck")
    if slide_deck is not None:
        slide_records, visual_pages = _index_slide_deck(slide_deck, bundle.get("metadata") or {}, errors)
        chunks.extend(slide_records)

    if not chunks and not visual_pages:
        errors.append("index: no chunkable content found in document_bundle")
        result = {"chunks": [], "chunk_count": 0, "section_counts": {}, "visual_pages": [], "visual_chunk_count": 0}
        if errors:
            result["errors"] = errors
        return result

    section_counts: dict[str, int] = {}
    for record in chunks:
        section_counts[record.section] = section_counts.get(record.section, 0) + 1

    result = {
        "chunks": chunks,
        "chunk_count": len(chunks),
        "section_counts": section_counts,
        "visual_pages": visual_pages,
        "visual_chunk_count": len(visual_pages),
    }
    if errors:
        result["errors"] = errors
    return result
