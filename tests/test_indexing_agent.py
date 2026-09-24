"""Offline tests for app/agents/indexing.py. No live network/model: the
only model-touching call (rasterize_pdf's PyMuPDF pass, not an ML model)
runs for real against small in-process PDFs, same pattern
tests/test_visual.py uses; chunk_filing/extract_tables are the real
functions too -- both are already offline-safe (tests/test_chunker.py,
tests/test_tables.py)."""
from __future__ import annotations

import pymupdf
import pytest

from app.agents.indexing import index_node

FILING_HTML = (
    "<html><body>"
    "<div>Item 1A. Risk Factors</div>"
    "<p>Some risk factor text describing risk in enough detail to be a real paragraph.</p>"
    "<div>Item 8. Financial Statements</div>"
    "<table><tr><td>Metric</td><td>FY2025</td></tr><tr><td>Revenue</td><td>100</td></tr></table>"
    "</body></html>"
)

TENQ_HTML = (
    "<html><body>"
    "<div>Item 1. Financial Statements</div>"
    "<p>Interim financial statement text goes here for the quarter.</p>"
    "</body></html>"
)

HTML_DECK = "<html><body><p>Q3 earnings deck, plain HTML exhibit, no natural page boundary.</p></body></html>"


def _make_pdf_bytes(num_pages: int) -> bytes:
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page(width=200, height=100)
    data = doc.tobytes()
    doc.close()
    return data


def _bundle(filings, slide_deck=None, metadata=None):
    return {
        "filings": filings,
        "xbrl_facts": None,
        "slide_deck": slide_deck,
        "metadata": metadata or {"ticker": "AAPL", "primary_form": "10-K"},
    }


def test_index_node_missing_bundle_records_error():
    result = index_node({})

    assert "chunks" not in result
    assert "no document_bundle to chunk" in result["errors"][0]


def test_index_node_chunks_every_form_present():
    state = {
        "filings": {
            "10-K": {"accessionNumber": "acc-10k"},
            "10-Q": {"accessionNumber": "acc-10q"},
            "8-K": None,
        },
        "document_bundle": _bundle(
            {
                "10-K": {"html": FILING_HTML, "metadata": {"ticker": "AAPL", "form": "10-K"}},
                "10-Q": {"html": TENQ_HTML, "metadata": {"ticker": "AAPL", "form": "10-Q"}},
                "8-K": None,
            }
        ),
    }

    result = index_node(state)

    assert "errors" not in result
    doc_ids = {c.doc_id for c in result["chunks"]}
    assert doc_ids == {"acc-10k", "acc-10q"}
    assert result["chunk_count"] == len(result["chunks"])
    assert result["visual_chunk_count"] == 0
    assert result["visual_pages"] == []


def test_index_node_produces_both_text_and_table_chunk_types():
    state = {
        "filings": {"10-K": {"accessionNumber": "acc-10k"}, "10-Q": None, "8-K": None},
        "document_bundle": _bundle(
            {"10-K": {"html": FILING_HTML, "metadata": {"ticker": "AAPL", "form": "10-K"}}, "10-Q": None, "8-K": None}
        ),
    }

    result = index_node(state)

    types = {c.chunk_type for c in result["chunks"]}
    assert types == {"text", "table"}
    table_chunks = [c for c in result["chunks"] if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "Revenue" in table_chunks[0].text
    assert table_chunks[0].metadata["rows"] == 2
    assert table_chunks[0].metadata["cols"] == 2
    # the table's cell text doesn't also show up garbled in a text chunk
    text_chunks = [c for c in result["chunks"] if c.chunk_type == "text"]
    assert not any("100" in c.text for c in text_chunks)


def test_index_node_missing_form_is_skipped_not_an_error():
    state = {
        "filings": {"10-K": {"accessionNumber": "acc-10k"}, "10-Q": None, "8-K": None},
        "document_bundle": _bundle(
            {"10-K": {"html": FILING_HTML, "metadata": {"ticker": "AAPL", "form": "10-K"}}, "10-Q": None, "8-K": None}
        ),
    }

    result = index_node(state)

    assert "errors" not in result
    assert all(c.doc_id == "acc-10k" for c in result["chunks"])


def test_index_node_no_content_anywhere_records_error():
    state = {
        "filings": {"10-K": None, "10-Q": None, "8-K": None},
        "document_bundle": _bundle({"10-K": None, "10-Q": None, "8-K": None}),
    }

    result = index_node(state)

    assert result["chunk_count"] == 0
    assert result["chunks"] == []
    assert "no chunkable content found" in result["errors"][0]


def test_index_node_html_slide_deck_is_chunked_as_text_not_rasterized():
    state = {
        "filings": {"10-K": None, "10-Q": None, "8-K": {"accessionNumber": "acc-8k"}},
        "document_bundle": _bundle(
            {"10-K": None, "10-Q": None, "8-K": None},
            slide_deck={
                "exhibit_name": "a8-kex991.htm",
                "exhibit_type": "EX-99.1",
                "content_type": "html",
                "url": "https://example.com/a8-kex991.htm",
                "html": HTML_DECK,
                "bytes": None,
            },
        ),
    }

    result = index_node(state)

    assert "errors" not in result
    assert result["visual_pages"] == []
    assert result["visual_chunk_count"] == 0
    assert result["chunk_count"] > 0
    assert all(c.doc_id == "a8-kex991.htm" for c in result["chunks"])
    assert all(c.chunk_type == "text" for c in result["chunks"])


def test_index_node_pdf_slide_deck_is_rasterized_not_chunked():
    state = {
        "filings": {"10-K": None, "10-Q": None, "8-K": {"accessionNumber": "acc-8k"}},
        "document_bundle": _bundle(
            {"10-K": None, "10-Q": None, "8-K": None},
            slide_deck={
                "exhibit_name": "a8-kex991.pdf",
                "exhibit_type": "EX-99.1",
                "content_type": "pdf",
                "url": "https://example.com/a8-kex991.pdf",
                "html": None,
                "bytes": _make_pdf_bytes(3),
            },
        ),
    }

    result = index_node(state)

    assert "errors" not in result
    assert result["chunk_count"] == 0
    assert result["chunks"] == []
    assert result["visual_chunk_count"] == 3
    assert [p.page_number for p in result["visual_pages"]] == [1, 2, 3]
    assert all(p.doc_id == "a8-kex991.pdf" for p in result["visual_pages"])
    assert result["visual_pages"][0].metadata["ticker"] == "AAPL"


def test_index_node_pdf_rasterization_failure_is_non_fatal(monkeypatch):
    import app.agents.indexing as indexing_module

    def _raise(pdf_bytes):
        raise RuntimeError("corrupt pdf")

    monkeypatch.setattr(indexing_module, "rasterize_pdf", _raise)

    state = {
        "filings": {"10-K": {"accessionNumber": "acc-10k"}, "10-Q": None, "8-K": {"accessionNumber": "acc-8k"}},
        "document_bundle": _bundle(
            {"10-K": {"html": FILING_HTML, "metadata": {"ticker": "AAPL", "form": "10-K"}}, "10-Q": None, "8-K": None},
            slide_deck={
                "exhibit_name": "a8-kex991.pdf",
                "exhibit_type": "EX-99.1",
                "content_type": "pdf",
                "url": "https://example.com/a8-kex991.pdf",
                "html": None,
                "bytes": b"not a real pdf",
            },
        ),
    }

    result = index_node(state)

    assert any("slide-deck PDF rasterization failed" in e for e in result["errors"])
    # the filing's own chunks still populate despite the slide-deck failure
    assert result["chunk_count"] > 0
    assert result["visual_pages"] == []


def test_index_node_no_slide_deck_is_not_an_error():
    state = {
        "filings": {"10-K": {"accessionNumber": "acc-10k"}, "10-Q": None, "8-K": None},
        "document_bundle": _bundle(
            {"10-K": {"html": FILING_HTML, "metadata": {"ticker": "AAPL", "form": "10-K"}}, "10-Q": None, "8-K": None},
            slide_deck=None,
        ),
    }

    result = index_node(state)

    assert "errors" not in result
    assert result["visual_pages"] == []
    assert result["visual_chunk_count"] == 0
