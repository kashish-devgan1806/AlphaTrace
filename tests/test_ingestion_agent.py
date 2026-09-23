"""Offline tests for app/agents/ingestion.py. No live network: every EDGAR
function ingest_node calls is monkeypatched on app.agents.ingestion (where
the node looks the name up), same pattern tests/test_graph.py uses."""
from __future__ import annotations

import httpx
import pytest

from app.agents.ingestion import ingest_node

TICKER_MAP = {"AAPL": 320193}

SUBMISSIONS_ALL_THREE = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2025-11-01", "2025-08-01", "2025-07-31"],
            "reportDate": ["2025-09-27", "2025-06-28", ""],
            "accessionNumber": ["0000320193-25-000100", "0000320193-25-000090", "0000320193-25-000085"],
            "primaryDocument": ["aapl-20250927.htm", "aapl-20250628.htm", "aapl-8k.htm"],
        }
    }
}

SUBMISSIONS_NO_8K = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q"],
            "filingDate": ["2025-11-01", "2025-08-01"],
            "reportDate": ["2025-09-27", "2025-06-28"],
            "accessionNumber": ["0000320193-25-000100", "0000320193-25-000090"],
            "primaryDocument": ["aapl-20250927.htm", "aapl-20250628.htm"],
        }
    }
}

SUBMISSIONS_NONE = {"filings": {"recent": {"form": []}}}

FILING_HTML = "<html><body><div>Item 1A. Risk Factors</div><p>Risk text.</p></body></html>"
EXHIBIT_HTML = "<html><body>Q3 earnings deck.</body></html>"

EXHIBIT_ROW_HTML = {
    "seq": "2",
    "description": "EX-99.1",
    "name": "a8-kex991.htm",
    "href": "/Archives/edgar/data/320193/000032019325000085/a8-kex991.htm",
    "type": "EX-99.1",
    "size": "173484",
}
EXHIBIT_ROW_PDF = {**EXHIBIT_ROW_HTML, "name": "a8-kex991.pdf"}

XBRL_FACTS_FIXTURE = {
    "Revenues": {"tag": "Revenues", "val": 1, "fy": 2025, "end": "2025-09-27", "accn": "acc"},
    "GrossProfit": None,
    "NetIncomeLoss": None,
}


def _patch_common(monkeypatch, ticker_map=TICKER_MAP, submissions=SUBMISSIONS_ALL_THREE):
    monkeypatch.setattr("app.agents.ingestion.load_ticker_map", lambda client, force_refresh=False: ticker_map)
    monkeypatch.setattr("app.agents.ingestion.fetch_submissions", lambda client, cik: submissions)
    monkeypatch.setattr("app.agents.ingestion.fetch_companyfacts", lambda client, cik: {})
    monkeypatch.setattr("app.agents.ingestion.extract_gaap_facts", lambda companyfacts: XBRL_FACTS_FIXTURE)
    monkeypatch.setattr(
        "app.agents.ingestion.fetch_primary_document", lambda client, cik, accession, doc: FILING_HTML
    )
    monkeypatch.setattr("app.agents.ingestion.fetch_filing_index", lambda client, cik, accession: [])
    monkeypatch.setattr("app.agents.ingestion.find_exhibit_99", lambda rows: None)


def test_ingest_node_all_three_forms_present(monkeypatch):
    _patch_common(monkeypatch)

    result = ingest_node({"ticker": "AAPL"})

    assert "errors" not in result
    assert result["form"] == "10-K"
    for form in ("10-K", "10-Q", "8-K"):
        assert result["filings"][form] is not None
        assert result["document_bundle"]["filings"][form]["html"] == FILING_HTML
    assert result["document_bundle"]["xbrl_facts"] == XBRL_FACTS_FIXTURE
    assert result["document_bundle"]["slide_deck"] is None
    assert result["document_bundle"]["metadata"] == {"ticker": "AAPL", "primary_form": "10-K"}


def test_ingest_node_missing_8k_is_not_an_error(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.agents.ingestion.fetch_filing_index",
        lambda client, cik, accession: calls.append(accession) or [],
    )
    _patch_common(monkeypatch, submissions=SUBMISSIONS_NO_8K)

    result = ingest_node({"ticker": "AAPL"})

    assert "errors" not in result
    assert result["filings"]["8-K"] is None
    assert result["document_bundle"]["filings"]["8-K"] is None
    assert result["document_bundle"]["slide_deck"] is None
    assert calls == []  # no 8-K found -> the exhibit-index path is never attempted


def test_ingest_node_missing_all_filings_is_fatal(monkeypatch):
    _patch_common(monkeypatch, submissions=SUBMISSIONS_NONE)

    result = ingest_node({"ticker": "AAPL"})

    assert "document_bundle" not in result
    assert len(result["errors"]) == 1
    assert "no 10-K, 10-Q, or 8-K content" in result["errors"][0]
    assert all(v is None for v in result["filings"].values())


def test_ingest_node_xbrl_fetch_failure_is_non_fatal(monkeypatch):
    _patch_common(monkeypatch)

    def _raise(client, cik):
        raise httpx.HTTPStatusError("boom", request=None, response=httpx.Response(500))

    monkeypatch.setattr("app.agents.ingestion.fetch_companyfacts", _raise)

    result = ingest_node({"ticker": "AAPL"})

    assert result["document_bundle"]["xbrl_facts"] is None
    assert any("XBRL companyfacts fetch failed" in e for e in result["errors"])
    # the rest of the bundle still populates despite the XBRL failure
    assert result["document_bundle"]["filings"]["10-K"] is not None


def test_ingest_node_8k_present_no_exhibit_99(monkeypatch):
    _patch_common(monkeypatch)  # find_exhibit_99 -> None by default

    result = ingest_node({"ticker": "AAPL"})

    assert "errors" not in result
    assert result["document_bundle"]["slide_deck"] is None


def test_ingest_node_8k_with_html_exhibit_99(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr("app.agents.ingestion.fetch_filing_index", lambda client, cik, accession: [EXHIBIT_ROW_HTML])
    monkeypatch.setattr("app.agents.ingestion.find_exhibit_99", lambda rows: rows[0] if rows else None)

    result = ingest_node({"ticker": "AAPL"})

    deck = result["document_bundle"]["slide_deck"]
    assert deck["content_type"] == "html"
    assert deck["html"] == FILING_HTML
    assert deck["bytes"] is None
    assert deck["exhibit_name"] == "a8-kex991.htm"
    assert "a8-kex991.htm" in deck["url"]


def test_ingest_node_8k_with_pdf_exhibit_99(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr("app.agents.ingestion.fetch_filing_index", lambda client, cik, accession: [EXHIBIT_ROW_PDF])
    monkeypatch.setattr("app.agents.ingestion.find_exhibit_99", lambda rows: rows[0] if rows else None)
    monkeypatch.setattr(
        "app.agents.ingestion.fetch_document_bytes", lambda client, cik, accession, name: b"%PDF-1.4 raw bytes"
    )

    result = ingest_node({"ticker": "AAPL"})

    deck = result["document_bundle"]["slide_deck"]
    assert deck["content_type"] == "pdf"
    assert deck["bytes"] == b"%PDF-1.4 raw bytes"
    assert deck["html"] is None


def test_ingest_node_one_form_fetch_fails_others_still_populate(monkeypatch):
    _patch_common(monkeypatch)

    def _fetch_primary(client, cik, accession, doc):
        if doc == "aapl-20250628.htm":  # the 10-Q's primary document
            raise httpx.HTTPStatusError("boom", request=None, response=httpx.Response(500))
        return FILING_HTML

    monkeypatch.setattr("app.agents.ingestion.fetch_primary_document", _fetch_primary)

    result = ingest_node({"ticker": "AAPL"})

    assert result["document_bundle"]["filings"]["10-Q"] is None
    assert result["document_bundle"]["filings"]["10-K"] is not None
    assert result["document_bundle"]["filings"]["8-K"] is not None
    assert any("AAPL 10-Q" in e for e in result["errors"])


def test_ingest_node_unknown_ticker_is_fatal(monkeypatch):
    _patch_common(monkeypatch, ticker_map={})

    result = ingest_node({"ticker": "AAPL"})

    assert "document_bundle" not in result
    assert "filings" not in result
    assert len(result["errors"]) == 1
    assert "not in SEC's ticker list" in result["errors"][0]


def test_ingest_node_submissions_fetch_failure_is_fatal(monkeypatch):
    _patch_common(monkeypatch)

    def _raise(client, cik):
        raise httpx.HTTPStatusError("boom", request=None, response=httpx.Response(500))

    monkeypatch.setattr("app.agents.ingestion.fetch_submissions", _raise)

    result = ingest_node({"ticker": "AAPL"})

    assert "document_bundle" not in result
    assert len(result["errors"]) == 1
    assert "EDGAR submissions fetch failed" in result["errors"][0]
