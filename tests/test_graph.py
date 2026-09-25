"""Offline tests for app/graph.py's ingest -> index -> analyst -> sentiment
scaffold. No live network/model/DB: the EDGAR functions app.agents.ingestion
imports are monkeypatched there (where ingest_node actually looks the names
up now that it lives in its own module); index_node lives in
app.agents.indexing — see tests/test_indexing_agent.py for its fuller
coverage (multi-form chunking, tables, slide-deck rasterization);
analyst_node lives in app.agents.analyst — see tests/test_analyst_agent.py
for its fuller coverage; sentiment_node lives in app.agents.sentiment —
see tests/test_sentiment_agent.py for its fuller coverage. This file
covers each node's basic wiring and the graph end to end."""
from __future__ import annotations

from app.graph import analyst_node, build_graph, index_node, ingest_node, sentiment_node
from app.rerank import RerankResult
from app.search import SearchResult
from app.sentiment import SentimentLabel
from app.transcript import QASegment

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


class _FakeConn:
    def close(self):
        pass


def _patch_analyst(monkeypatch):
    """No live Postgres/Groq: search()/generate() are monkeypatched where
    analyst_node looks them up, same as _patch_edgar does for ingest_node."""
    candidate = SearchResult(
        id=1,
        doc_id="doc-1",
        section="Item 1A",
        text="Risk factor text about supply chain exposure.",
        chunk_type="text",
        metadata={"ticker": "AAPL"},
        score=0.9,
    )
    monkeypatch.setattr("app.agents.analyst.get_connection", lambda: _FakeConn())
    monkeypatch.setattr("app.agents.analyst.search", lambda conn, query, k, ticker: [candidate])
    monkeypatch.setattr(
        "app.agents.analyst.rerank",
        lambda query, results, top_k: [RerankResult(result=r, rerank_score=1.0) for r in results],
    )
    monkeypatch.setattr(
        "app.agents.analyst.generate",
        lambda prompt, json_mode=False: (
            '{"answer": "The main risk is supply chain exposure [1].", '
            '"citations": [{"marker": "[1]", "chunk_id": 1, "quote": "supply chain exposure"}]}'
        ),
    )


def _patch_sentiment(monkeypatch):
    """No real classifier/network: parse_qa_segments()/classify_segments()
    are monkeypatched where sentiment_node looks them up, same as
    _patch_analyst does for analyst_node."""
    segment = QASegment(segment_id=1, question="Did tone shift?", answer="We remain confident in our guidance.")
    monkeypatch.setattr("app.agents.sentiment.parse_qa_segments", lambda text: [segment])
    monkeypatch.setattr(
        "app.agents.sentiment.classify_segments",
        lambda texts: [SentimentLabel(label="confident", confidence=0.9, scores={"confident": 0.9, "hedging": 0.1})],
    )


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
    _patch_analyst(monkeypatch)
    _patch_sentiment(monkeypatch)

    graph = build_graph()
    result = graph.invoke(
        {
            "ticker": "AAPL",
            "question": "What are the main risks?",
            "transcript": "Q: Did tone shift?\nA: We remain confident in our guidance.",
        }
    )

    # `ticker` came in on the initial state and no node overwrites it — the
    # merge preserves it, which is exactly the write-order guarantee
    # app/state.py's docstring documents.
    assert result["ticker"] == "AAPL"
    assert result["filings"]["10-K"]["accessionNumber"] == "0000320193-25-000100"
    assert result["chunk_count"] > 0
    assert result["draft_answer"] == "The main risk is supply chain exposure [1]."
    assert result["citations"] == [
        {"marker": "[1]", "chunk_id": 1, "doc_id": "doc-1", "section": "Item 1A", "chunk_type": "text", "quote": "supply chain exposure"}
    ]
    assert result["sentiment_result"]["segments"] == [
        {"segment_id": 1, "question": "Did tone shift?", "answer": "We remain confident in our guidance.", "label": "confident", "confidence": 0.9}
    ]
    assert result["sentiment_result"]["current_summary"]["hedging_ratio"] == 0.0
    # LangGraph seeds an operator.add-reduced field to [] by default rather
    # than leaving it absent — unlike a bare node return dict, which omits
    # a key it never set (see tests/test_ingestion_agent.py).
    assert result["errors"] == []


def test_build_graph_unknown_ticker_flows_error_through_every_node(monkeypatch):
    _patch_edgar(monkeypatch, ticker_map={})

    graph = build_graph()
    result = graph.invoke({"ticker": "AAPL"})

    # ingest's failure, index's downstream failure, analyst's own
    # missing-question failure, and sentiment's own missing-transcript
    # failure (neither `question` nor `transcript` was passed in) all land
    # in `errors` — the operator.add reducer accumulates rather than each
    # node's error silently overwriting the last.
    assert len(result["errors"]) == 4
    assert "not in SEC's ticker list" in result["errors"][0]
    assert "no document_bundle to chunk" in result["errors"][1]
    assert "no question to answer" in result["errors"][2]
    assert "no transcript to score" in result["errors"][3]
    assert "chunks" not in result


def test_ingest_node_is_the_ingestion_agents_node():
    from app.agents.ingestion import ingest_node as agent_ingest_node

    assert ingest_node is agent_ingest_node


def test_index_node_is_the_indexing_agents_node():
    from app.agents.indexing import index_node as agent_index_node

    assert index_node is agent_index_node


def test_analyst_node_is_the_analyst_agents_node():
    from app.agents.analyst import analyst_node as agent_analyst_node

    assert analyst_node is agent_analyst_node


def test_sentiment_node_is_the_sentiment_agents_node():
    from app.agents.sentiment import sentiment_node as agent_sentiment_node

    assert sentiment_node is agent_sentiment_node
