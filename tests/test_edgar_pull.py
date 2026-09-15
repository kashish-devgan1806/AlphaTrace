"""Offline tests for scripts/edgar_pull.py.

No live network involved: fixtures below match the real shape of SEC's
submissions JSON (parallel arrays under filings.recent), so these exercise
the actual parsing/printing logic, not a toy schema.
"""
import json

import httpx
import pytest

from scripts.edgar_pull import fetch_submissions, load_ticker_map, print_filing_metadata

AAPL_SUBMISSIONS_FIXTURE = {
    "cik": 320193,
    "name": "Apple Inc.",
    "sic": "3571",
    "sicDescription": "Electronic Computers",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2025-11-01", "2025-08-01", "2025-07-31"],
            "reportDate": ["2025-09-27", "2025-06-28", ""],
            "accessionNumber": [
                "0000320193-25-000100",
                "0000320193-25-000090",
                "0000320193-25-000085",
            ],
            "primaryDocument": ["aapl-20250927.htm", "aapl-20250628.htm", "aapl-8k.htm"],
        }
    },
}

NO_FILINGS_FIXTURE = {
    "cik": 1999999,
    "name": "Shell Registrant Corp",
    "sic": "",
    "sicDescription": "",
    "filings": {"recent": {"form": []}},
}


def test_print_filing_metadata_normal_case(capsys):
    print_filing_metadata("AAPL", AAPL_SUBMISSIONS_FIXTURE, limit=10)
    out = capsys.readouterr().out
    assert "Apple Inc." in out
    assert "10-K" in out and "10-Q" in out and "8-K" in out
    assert "0000320193-25-000100" in out
    # the 8-K row has no reportDate — must render as "n/a", not crash or blank
    assert "report_date=n/a" in out


def test_print_filing_metadata_respects_limit(capsys):
    print_filing_metadata("AAPL", AAPL_SUBMISSIONS_FIXTURE, limit=1)
    out = capsys.readouterr().out
    assert "Most recent 1 filing(s)" in out
    assert "10-Q" not in out  # only the first (10-K) row should print


def test_print_filing_metadata_no_recent_filings(capsys):
    """Session 1 review question: what happens if the ticker has no recent
    filings? Answer: it's treated as a valid, empty result — not an error."""
    print_filing_metadata("SHELL", NO_FILINGS_FIXTURE, limit=10)
    out = capsys.readouterr().out
    assert "No recent filings found" in out


def test_load_ticker_map_uses_cache_and_skips_network(tmp_path, monkeypatch):
    cache_file = tmp_path / "company_tickers.json"
    cache_file.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."}})
    )
    monkeypatch.setattr("scripts.edgar_pull.CACHE_PATH", cache_file)

    def _unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not hit the network when a cache file exists")

    client = httpx.Client(transport=httpx.MockTransport(_unexpected_request))
    result = load_ticker_map(client)

    assert result == {"AAPL": 320193}  # ticker is upper-cased regardless of source casing


def test_load_ticker_map_downloads_and_caches_on_miss(tmp_path, monkeypatch):
    cache_file = tmp_path / "company_tickers.json"
    monkeypatch.setattr("scripts.edgar_pull.CACHE_PATH", cache_file)

    payload = {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"}}

    def _handler(request: httpx.Request) -> httpx.Response:
        assert "User-Agent" in request.headers
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    result = load_ticker_map(client)

    assert result == {"MSFT": 789019}
    assert cache_file.exists()  # second run would hit the cache branch above


def test_fetch_submissions_raises_on_http_error():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_submissions(client, cik=1)
