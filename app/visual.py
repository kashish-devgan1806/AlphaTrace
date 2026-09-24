"""Slide-deck page rasterization and visual embeddings, written to the
page_embeddings table (db/init/04_add_chunk_type_and_page_embeddings.sql).

Only the PDF slide-deck case goes through here: an HTML exhibit has no
natural page boundary to rasterize (app/agents/indexing.py chunks it as
text instead, through the ordinary app.chunker path). PyMuPDF (fitz) over
pdf2image/Poppler: it rasterizes directly from PDF bytes with no external
binary dependency, which matters on a Windows dev box where Poppler isn't
something `pip install` alone gets you.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import psycopg
import pymupdf
from PIL import Image
from psycopg.types.json import Jsonb

from app.embeddings import embed_images

DEFAULT_DPI = 150


@dataclass
class PageImage:
    """One rasterized slide-deck page, ready to be embedded and written to
    the page_embeddings table."""

    doc_id: str
    page_number: int
    image: Image.Image
    metadata: dict = field(default_factory=dict)


def rasterize_pdf(pdf_bytes: bytes, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    """PDF bytes -> one RGB PIL Image per page, in page order. 150 DPI is
    high enough for a CLIP-style visual embedder (which downsamples to a
    fixed small resolution internally anyway) without producing an
    unnecessarily large intermediate bitmap per page."""
    zoom = dpi / 72  # PDF points are 1/72 inch; pymupdf's default render is 72 DPI
    matrix = pymupdf.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix)
            images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    return images


def batch_insert_page_embeddings(conn: psycopg.Connection, pages: list[PageImage]) -> int:
    """Embed and insert a batch of slide-deck page images in one round
    trip. Returns the number of rows actually inserted -- fewer than
    len(pages) when a page is already in the table for this doc_id/page_number
    (page_embeddings_doc_page_idx), the same re-run-is-a-safe-no-op pattern
    app.chunks.batch_insert_chunks() uses for text/table chunks."""
    if not pages:
        return 0

    vectors = embed_images([p.image for p in pages])
    rows = [
        (page.doc_id, page.page_number, vector, Jsonb(page.metadata))
        for page, vector in zip(pages, vectors)
    ]

    inserted = 0
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO page_embeddings (doc_id, page_number, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id, page_number) DO NOTHING
                RETURNING id
                """,
                rows,
                returning=True,
            )
            while True:
                inserted += len(cur.fetchall())
                if not cur.nextset():
                    break
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return inserted
