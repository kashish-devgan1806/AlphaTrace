"""Offline tests for scripts/edgar_pull.py.

No live network involved: fixtures below match the real shape of SEC's
submissions JSON (parallel arrays under filings.recent), so these exercise
the actual parsing/printing logic, not a toy schema.
"""
import json

import httpx
import pytest

from scripts.edgar_pull import (
    extract_gaap_facts,
    fetch_companyfacts,
    fetch_primary_document,
    fetch_submissions,
    load_ticker_map,
    print_filing_metadata,
)

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


def test_fetch_companyfacts_raises_on_http_error():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_companyfacts(client, cik=1)


def test_fetch_primary_document_builds_the_archive_url_without_dashes():
    """The archive URL's path segment drops the accession number's dashes
    (SEC's own convention) — this is the one thing most likely to be gotten
    wrong by hand, so it's asserted directly rather than just checking the
    response comes back."""
    seen_urls = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, text="<html>filing body</html>")

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    text = fetch_primary_document(client, cik=320193, accession_number="0000320193-25-000100", primary_document="aapl-20250927.htm")

    assert text == "<html>filing body</html>"
    assert seen_urls == [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000100/aapl-20250927.htm"
    ]


def test_fetch_primary_document_raises_on_http_error():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_primary_document(client, cik=1, accession_number="0000000001-25-000001", primary_document="x.htm")


def _annual_entry(fy: int, end: str, val: int, accn: str, fp: str = "FY", form: str = "10-K") -> dict:
    return {"fy": fy, "fp": fp, "form": form, "end": end, "val": val, "accn": accn}


def test_extract_gaap_facts_filters_to_annual_10k_entries():
    """A tag's units.USD array holds one entry per filing that reported it,
    including quarterly (10-Q, fp Q1/Q2/Q3) and prior-year comparative
    entries. Extraction must pick the annual (10-K, fp FY) entry, not just
    the last item in the array."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _annual_entry(2022, "2022-09-24", 99_803_000_000, "acc-2022-fy"),
                            _annual_entry(2023, "2022-12-31", 3_394_000_000, "acc-q1", fp="Q1", form="10-Q"),
                            _annual_entry(2023, "2023-09-30", 96_995_000_000, "acc-2023-fy"),
                        ]
                    }
                },
                "GrossProfit": {"units": {"USD": [_annual_entry(2023, "2023-09-30", 169_148_000_000, "acc-gp")]}},
            }
        }
    }

    facts = extract_gaap_facts(companyfacts)

    assert facts["NetIncomeLoss"]["val"] == 96_995_000_000
    assert facts["NetIncomeLoss"]["fy"] == 2023
    assert facts["NetIncomeLoss"]["tag"] == "NetIncomeLoss"
    assert facts["GrossProfit"]["val"] == 169_148_000_000
    # Revenues wasn't present under any candidate tag in this fixture.
    assert facts["Revenues"] is None


def test_extract_gaap_facts_falls_back_to_asc606_revenue_tag():
    """Most large filers (Apple included) tag revenue as
    RevenueFromContractWithCustomerExcludingAssessedTax post-ASC 606, not the
    older Revenues tag. Extraction must fall back rather than report a false
    'not found'."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_annual_entry(2023, "2023-09-30", 383_285_000_000, "acc-rev")]}
                }
            }
        }
    }

    facts = extract_gaap_facts(companyfacts)

    assert facts["Revenues"]["val"] == 383_285_000_000
    assert facts["Revenues"]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_extract_gaap_facts_ignores_non_usd_units():
    """A units block for a share-count-style unit (e.g. `shares`) must not be
    mistaken for a dollar figure just because the tag name matches."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {"shares": [_annual_entry(2023, "2023-09-30", 15_000_000_000, "acc-wrong-unit")]}
                }
            }
        }
    }

    facts = extract_gaap_facts(companyfacts)

    assert facts["NetIncomeLoss"] is None


def test_extract_gaap_facts_missing_tag_reports_none():
    facts = extract_gaap_facts({"facts": {"us-gaap": {}}})

    assert facts == {"Revenues": None, "GrossProfit": None, "NetIncomeLoss": None}
