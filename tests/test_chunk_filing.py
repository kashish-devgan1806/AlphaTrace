"""Offline tests for scripts/chunk_filing.py's process_ticker() — the
per-ticker pipeline stage scripts/build_corpus.py depends on to isolate one
ticker's failure from the others. No live network or DB: httpx.MockTransport
stands in for SEC, and the insert stage is exercised via a monkeypatched
batch_insert_chunks (its own embed/insert internals are already covered end
to end by tests/test_chunks.py, so this file only checks process_ticker
calls it correctly)."""
from __future__ import annotations

import httpx

from app.chunker import chunk_filing
from scripts.chunk_filing import process_ticker

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


def _router(submissions_response, document_response):
    """httpx.MockTransport handler routing by host: data.sec.gov gets
    submissions_response, www.sec.gov (the filing archive) gets
    document_response."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "data.sec.gov":
            return submissions_response(request)
        return document_response(request)

    return _handler


def test_process_ticker_happy_path_insert_false():
    handler = _router(
        lambda req: httpx.Response(200, json=SUBMISSIONS_WITH_10K),
        lambda req: httpx.Response(200, text=FILING_HTML),
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = process_ticker(client, TICKER_MAP, "aapl", "10-K", insert=False)

    expected_records = chunk_filing(
        "0000320193-25-000100",
        FILING_HTML,
        metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-11-01", "report_date": "2025-09-27"},
    )
    expected_sections = {r.section for r in expected_records}

    assert result.status == "ok"
    assert result.ticker == "AAPL"
    assert result.stage is None
    assert result.chunk_count == len(expected_records)
    assert result.section_count == len(expected_sections)
    assert result.filing["accessionNumber"] == "0000320193-25-000100"
    assert result.inserted is None


def test_process_ticker_ticker_not_in_map_never_hits_network():
    def _unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not make any HTTP call for an unknown ticker")

    client = httpx.Client(transport=httpx.MockTransport(_unexpected))

    result = process_ticker(client, {}, "AAPL", "10-K", insert=False)

    assert result.status == "error"
    assert result.stage == "ticker_lookup"


def test_process_ticker_fetch_submissions_http_error():
    handler = _router(
        lambda req: httpx.Response(404),
        lambda req: httpx.Response(200, text=FILING_HTML),
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = process_ticker(client, TICKER_MAP, "AAPL", "10-K", insert=False)

    assert result.status == "error"
    assert result.stage == "fetch_submissions"


def test_process_ticker_no_filing_of_requested_form_never_fetches_document():
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "data.sec.gov":
            return httpx.Response(200, json=SUBMISSIONS_ONLY_8K)
        raise AssertionError("should not fetch a document when no matching filing was found")

    client = httpx.Client(transport=httpx.MockTransport(_handler))

    result = process_ticker(client, TICKER_MAP, "AAPL", "10-K", insert=False)

    assert result.status == "error"
    assert result.stage == "find_filing"


def test_process_ticker_fetch_document_http_error_keeps_filing():
    handler = _router(
        lambda req: httpx.Response(200, json=SUBMISSIONS_WITH_10K),
        lambda req: httpx.Response(404),
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = process_ticker(client, TICKER_MAP, "AAPL", "10-K", insert=False)

    assert result.status == "error"
    assert result.stage == "fetch_document"
    assert result.filing is not None
    assert result.filing["accessionNumber"] == "0000320193-25-000100"


def test_process_ticker_insert_true_calls_batch_insert_chunks(monkeypatch):
    handler = _router(
        lambda req: httpx.Response(200, json=SUBMISSIONS_WITH_10K),
        lambda req: httpx.Response(200, text=FILING_HTML),
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    calls = []

    def fake_batch_insert_chunks(conn, records):
        calls.append((conn, records))
        return 7

    monkeypatch.setattr("scripts.chunk_filing.batch_insert_chunks", fake_batch_insert_chunks)

    sentinel_conn = object()
    result = process_ticker(client, TICKER_MAP, "AAPL", "10-K", insert=True, conn=sentinel_conn)

    assert result.status == "ok"
    assert result.inserted == 7
    assert len(calls) == 1
    assert calls[0][0] is sentinel_conn
    assert len(calls[0][1]) == result.chunk_count


def test_process_ticker_insert_failure_reports_insert_stage(monkeypatch):
    handler = _router(
        lambda req: httpx.Response(200, json=SUBMISSIONS_WITH_10K),
        lambda req: httpx.Response(200, text=FILING_HTML),
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    def failing_batch_insert_chunks(conn, records):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("scripts.chunk_filing.batch_insert_chunks", failing_batch_insert_chunks)

    result = process_ticker(client, TICKER_MAP, "AAPL", "10-K", insert=True, conn=object())

    assert result.status == "error"
    assert result.stage == "insert"
    assert result.chunk_count > 0
    assert result.inserted is None
