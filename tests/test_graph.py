"""Offline tests for app/graph.py's ingest -> index scaffold. No live
network/model: the EDGAR functions app.agents.ingestion imports are
monkeypatched there (where ingest_node actually looks the names up now
that it lives in its own module); index_node lives in
app.agents.indexing — see tests/test_indexing_agent.py for its fuller
coverage (multi-form chunking, tables, slide-deck rasterization). This
file covers index_node's basic wiring and the graph end to end."""
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

FILING_HTML = (
    "<html><body>"
    "<div>Cover page text.</div>"
    "<div>Item 1A. Risk Factors</div>"
    "<p>Some risk factor text describing risk.</p>"
    "</body></html>"
)


def _patch_edgar(monkeypatch, ticker_map=TICKER_MAP, submissions=SUBMISSIONS_WITH_10K, html=FILING_HTML):
    monkeypatch.setattr("app.agents.ingestion.load_ticker_map", lambda client, force_refresh=False: ticker_map)
    monkeypatch.setattr("app.agents.ingestion.fetch_submissions", lambda client, cik: submissions)
    monkeypatch.setattr("app.agents.ingestion.fetch_primary_document", lambda client, cik, accession, doc: html)
    monkeypatch.setattr("app.agents.ingestion.fetch_companyfacts", lambda client, cik: {})
    monkeypatch.setattr("app.agents.ingestion.extract_gaap_facts", lambda companyfacts: {})
    monkeypatch.setattr("app.agents.ingestion.fetch_filing_index", lambda client, cik, accession: [])
    monkeypatch.setattr("app.agents.ingestion.find_exhibit_99", lambda rows: None)


def test_index_node_happy_path():
    state = {
        "form": "10-K",
        "filings": {"10-K": {"accessionNumber": "0000320193-25-000100"}},
        "document_bundle": {
            "filings": {
                "10-K": {"html": FILING_HTML, "metadata": {"ticker": "AAPL", "form": "10-K"}},
            },
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


def test_index_node_chunks_whichever_forms_are_present_even_if_primary_is_missing():
    # document_bundle exists (e.g. only an 8-K got bundled) but the primary
    # form ("10-K") has no entry — index_node chunks every form actually
    # present rather than requiring the primary one specifically.
    state = {
        "form": "10-K",
        "filings": {"10-K": None, "8-K": {"accessionNumber": "0000320193-25-000085"}},
        "document_bundle": {
            "filings": {"10-K": None, "8-K": {"html": FILING_HTML, "metadata": {}}},
        },
    }

    result = index_node(state)

    assert "errors" not in result
    assert result["chunk_count"] > 0
    assert all(c.doc_id == "0000320193-25-000085" for c in result["chunks"])


def test_index_node_no_content_anywhere_records_error():
    state = {
        "filings": {"10-K": None, "10-Q": None, "8-K": None},
        "document_bundle": {"filings": {"10-K": None, "10-Q": None, "8-K": None}, "slide_deck": None},
    }

    result = index_node(state)

    assert result["chunk_count"] == 0
    assert "no chunkable content found in document_bundle" in result["errors"][0]


def test_build_graph_happy_path_end_to_end(monkeypatch):
    _patch_edgar(monkeypatch)

    graph = build_graph()
    result = graph.invoke({"ticker": "AAPL"})

    # `ticker` came in on the initial state and no node overwrites it — the
    # merge preserves it, which is exactly the write-order guarantee
    # app/state.py's docstring documents.
    assert result["ticker"] == "AAPL"
    assert result["filings"]["10-K"]["accessionNumber"] == "0000320193-25-000100"
    assert result["chunk_count"] > 0
    # LangGraph seeds an operator.add-reduced field to [] by default rather
    # than leaving it absent — unlike a bare node return dict, which omits
    # a key it never set (see tests/test_ingestion_agent.py).
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


def test_ingest_node_is_the_ingestion_agents_node():
    from app.agents.ingestion import ingest_node as agent_ingest_node

    assert ingest_node is agent_ingest_node


def test_index_node_is_the_indexing_agents_node():
    from app.agents.indexing import index_node as agent_index_node

    assert index_node is agent_index_node
