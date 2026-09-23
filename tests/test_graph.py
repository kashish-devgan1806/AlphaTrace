"""Offline tests for app/graph.py's trivial ingest -> index scaffold. No
live network: the EDGAR functions app.graph imports are monkeypatched
directly, same pattern tests/test_chunk_filing.py uses for
scripts/chunk_filing.py's process_ticker()."""
from __future__ import annotations

from app.graph import build_graph, index_node, ingest_node

TICKER_MAP = {"AAPL": 320193}

SUBMISSIONS_WITH_10K = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K"],
            "filingDate": ["2025-11-01", "2025-07-31"],
            "reportDate": ["2025-09-27", ""],
            "accessionNumber": ["0000320193-25-000100", "0000320193-25-000085"],
            "primaryDocument": ["aapl-20250927.htm", "aapl-8k.htm"],
        }
    }
}

SUBMISSIONS_ONLY_8K = {
    "filings": {
        "recent": {
            "form": ["8-K"],
            "filingDate": ["2025-07-31"],
            "reportDate": [""],
            "accessionNumber": ["0000320193-25-000085"],
            "primaryDocument": ["aapl-8k.htm"],
        }
    }
}

FILING_HTML = (
    "<html><body>"
    "<div>Cover page text.</div>"
    "<div>Item 1A. Risk Factors</div>"
    "<p>Some risk factor text describing risk.</p>"
    "</body></html>"
)


def _patch_edgar(monkeypatch, ticker_map=TICKER_MAP, submissions=SUBMISSIONS_WITH_10K, html=FILING_HTML):
    monkeypatch.setattr("app.graph.load_ticker_map", lambda client, force_refresh=False: ticker_map)
    monkeypatch.setattr("app.graph.fetch_submissions", lambda client, cik: submissions)
    monkeypatch.setattr(
        "app.graph.fetch_primary_document", lambda client, cik, accession, doc: html
    )


def test_ingest_node_happy_path(monkeypatch):
    _patch_edgar(monkeypatch)

    result = ingest_node({"ticker": "AAPL"})

    assert "errors" not in result
    assert result["form"] == "10-K"
    assert result["filing"]["accessionNumber"] == "0000320193-25-000100"
    assert result["document_bundle"]["html"] == FILING_HTML
    assert result["document_bundle"]["metadata"] == {
        "ticker": "AAPL",
        "form": "10-K",
        "filing_date": "2025-11-01",
        "report_date": "2025-09-27",
    }


def test_ingest_node_unknown_ticker_records_error(monkeypatch):
    _patch_edgar(monkeypatch, ticker_map={})

    result = ingest_node({"ticker": "AAPL"})

    assert "document_bundle" not in result
    assert len(result["errors"]) == 1
    assert "not in SEC's ticker list" in result["errors"][0]


def test_ingest_node_no_matching_form_records_error(monkeypatch):
    _patch_edgar(monkeypatch, submissions=SUBMISSIONS_ONLY_8K)

    result = ingest_node({"ticker": "AAPL"})

    assert "document_bundle" not in result
    assert "no recent 10-K found" in result["errors"][0]


def test_index_node_happy_path():
    state = {
        "filing": {"accessionNumber": "0000320193-25-000100"},
        "document_bundle": {
            "html": FILING_HTML,
            "metadata": {"ticker": "AAPL", "form": "10-K"},
        },
    }

    result = index_node(state)

    assert "errors" not in result
    assert result["chunk_count"] == len(result["chunks"])
    assert result["chunk_count"] > 0
    assert sum(result["section_counts"].values()) == result["chunk_count"]


def test_index_node_missing_bundle_records_error():
    result = index_node({})

    assert "chunks" not in result
    assert "no document_bundle to chunk" in result["errors"][0]


def test_build_graph_happy_path_end_to_end(monkeypatch):
    _patch_edgar(monkeypatch)

    graph = build_graph()
    result = graph.invoke({"ticker": "AAPL"})

    # `ticker` came in on the initial state and no node overwrites it — the
    # merge preserves it, which is exactly the write-order guarantee
    # app/state.py's docstring documents.
    assert result["ticker"] == "AAPL"
    assert result["filing"]["accessionNumber"] == "0000320193-25-000100"
    assert result["chunk_count"] > 0
    # LangGraph seeds an operator.add-reduced field to [] by default rather
    # than leaving it absent — unlike a bare node return dict, which omits
    # a key it never set (see test_ingest_node_happy_path above).
    assert result["errors"] == []


def test_build_graph_unknown_ticker_flows_error_through_both_nodes(monkeypatch):
    _patch_edgar(monkeypatch, ticker_map={})

    graph = build_graph()
    result = graph.invoke({"ticker": "AAPL"})

    # ingest's failure and index's downstream failure both land in
    # `errors` — the operator.add reducer accumulates rather than the
    # second node's error silently overwriting the first.
    assert len(result["errors"]) == 2
    assert "not in SEC's ticker list" in result["errors"][0]
    assert "no document_bundle to chunk" in result["errors"][1]
    assert "chunks" not in result
