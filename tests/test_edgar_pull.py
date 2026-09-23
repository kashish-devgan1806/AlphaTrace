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
    fetch_document_bytes,
    fetch_filing_index,
    fetch_primary_document,
    fetch_submissions,
    find_exhibit_99,
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
    """What happens if the ticker has no recent filings? It's treated as a
    valid, empty result — not an error."""
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


def _no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("scripts.edgar_pull.time.sleep", lambda s: slept.append(s))
    return slept


def test_get_retries_transient_5xx_then_succeeds(monkeypatch):
    slept = _no_sleep(monkeypatch)
    statuses = iter([503, 429, 200])

    def handler(request):
        code = next(statuses)
        return httpx.Response(code, json={"ok": True}) if code == 200 else httpx.Response(code)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_submissions(client, 320193) == {"ok": True}
    assert slept == [1.0, 2.0]  # exponential backoff between the two retries


def test_get_does_not_retry_a_404(monkeypatch):
    slept = _no_sleep(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_submissions(client, 320193)
    assert len(calls) == 1 and slept == []


def test_get_gives_up_after_max_attempts_and_raises(monkeypatch):
    slept = _no_sleep(monkeypatch)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_submissions(client, 320193)
    assert len(calls) == 3 and len(slept) == 2


def test_get_retries_timeouts_and_honours_retry_after(monkeypatch):
    slept = _no_sleep(monkeypatch)
    steps = iter(["timeout", "retry-after", "ok"])

    def handler(request):
        step = next(steps)
        if step == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if step == "retry-after":
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_submissions(client, 1) == {"ok": True}
    assert slept == [1.0, 7.0]


def test_load_ticker_map_recovers_from_a_corrupt_cache_file(tmp_path, monkeypatch):
    cache_file = tmp_path / "company_tickers.json"
    cache_file.write_text('{"0": {"cik_str": 32019', encoding="utf-8")  # truncated mid-write
    monkeypatch.setattr("scripts.edgar_pull.CACHE_PATH", cache_file)
    payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))

    assert load_ticker_map(client) == {"AAPL": 320193}
    assert json.loads(cache_file.read_text(encoding="utf-8")) == payload  # cache repaired
    assert not (tmp_path / "company_tickers.json.tmp").exists()  # atomic write left no temp file


def test_load_ticker_map_force_refresh_ignores_a_good_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "company_tickers.json"
    cache_file.write_text(json.dumps({"0": {"cik_str": 1, "ticker": "OLD", "title": "x"}}), encoding="utf-8")
    monkeypatch.setattr("scripts.edgar_pull.CACHE_PATH", cache_file)
    payload = {"0": {"cik_str": 2, "ticker": "NEW", "title": "y"}}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))

    assert load_ticker_map(client, force_refresh=True) == {"NEW": 2}


# Trimmed to the two tables that matter, shaped exactly like a live Apple 8-K's
# index page (fetched and verified 2026-09-23) — same table structure (Seq,
# Description, Document, Type, Size), same iXBRL-viewer link wrapping the
# primary document, same lack of a Seq number on the "Complete submission
# text file" row.
EARNINGS_8K_INDEX_HTML = """
<html><body>
<table class="tableFile" summary="Document Format Files">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr>
    <td>1</td><td>8-K</td>
    <td><a href="/ix?doc=/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm">aapl-20260730.htm</a>
        &nbsp;&nbsp;<span style="color: green">iXBRL</span></td>
    <td>8-K</td><td>38350</td>
  </tr>
  <tr>
    <td>2</td><td>EX-99.1</td>
    <td><a href="/Archives/edgar/data/320193/000032019326000018/a8-kex991q3202606272026.htm">a8-kex991q3202606272026.htm</a></td>
    <td>EX-99.1</td><td>173484</td>
  </tr>
  <tr>
    <td>&nbsp;</td><td>Complete submission text file</td>
    <td><a href="/Archives/edgar/data/320193/000032019326000018/0000320193-26-000018.txt">0000320193-26-000018.txt</a></td>
    <td>&nbsp;</td><td>417360</td>
  </tr>
</table>
<table class="tableFile" summary="Data Files">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr>
    <td>3</td><td>XBRL TAXONOMY EXTENSION SCHEMA DOCUMENT</td>
    <td><a href="/Archives/edgar/data/320193/000032019326000018/aapl-20260730.xsd">aapl-20260730.xsd</a></td>
    <td>EX-101.SCH</td><td>3650</td>
  </tr>
</table>
</body></html>
"""

NO_EXHIBIT_8K_INDEX_HTML = """
<html><body>
<table class="tableFile" summary="Document Format Files">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr>
    <td>1</td><td>8-K</td>
    <td><a href="/ix?doc=/Archives/edgar/data/320193/000032019326999999/aapl-20260420.htm">aapl-20260420.htm</a></td>
    <td>8-K</td><td>5120</td>
  </tr>
</table>
</body></html>
"""


def test_fetch_filing_index_builds_correct_url_and_parses_rows():
    seen_urls = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, text=EARNINGS_8K_INDEX_HTML)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    rows = fetch_filing_index(client, cik=320193, accession_number="0000320193-26-000018")

    assert seen_urls == [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/0000320193-26-000018-index.html"
    ]
    # 3 real numbered document rows across both tables; the "Complete
    # submission text file" row has no Seq number and must not be mistaken
    # for one (4 <td> rows exist in the fixture, only 3 are real documents).
    assert len(rows) == 3
    assert rows[0] == {
        "seq": "1",
        "description": "8-K",
        "name": "aapl-20260730.htm",
        "href": "/ix?doc=/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm",
        "type": "8-K",
        "size": "38350",
    }


def test_fetch_filing_index_raises_on_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_filing_index(client, cik=1, accession_number="0000000001-25-000001")


def test_find_exhibit_99_picks_first_match_and_strips_ix_viewer_prefix():
    parser_input = fetch_filing_index(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=EARNINGS_8K_INDEX_HTML))),
        cik=320193,
        accession_number="0000320193-26-000018",
    )
    exhibit = find_exhibit_99(parser_input)

    assert exhibit is not None
    assert exhibit["name"] == "a8-kex991q3202606272026.htm"
    assert exhibit["type"] == "EX-99.1"
    # The primary document's row (type "8-K") has the /ix?doc= wrapper; the
    # matched exhibit row is a plain link and must not have it stripped
    # incorrectly or left un-stripped on the wrong row.
    assert not exhibit["name"].startswith("/ix")


def test_find_exhibit_99_returns_none_when_absent():
    rows = fetch_filing_index(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=NO_EXHIBIT_8K_INDEX_HTML))),
        cik=320193,
        accession_number="0000320193-26-999999",
    )
    assert find_exhibit_99(rows) is None


def test_find_exhibit_99_is_case_insensitive_on_type():
    rows = [{"type": "ex-99.2", "name": "deck.htm"}]
    exhibit = find_exhibit_99(rows)
    assert exhibit is not None and exhibit["name"] == "deck.htm"


def test_find_exhibit_99_does_not_match_xbrl_ex101_rows():
    rows = fetch_filing_index(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=EARNINGS_8K_INDEX_HTML))),
        cik=320193,
        accession_number="0000320193-26-000018",
    )
    # Sanity check on the fixture itself: the Data Files table's EX-101.SCH
    # row must not accidentally satisfy an "EX-99" prefix match.
    assert any(r["type"] == "EX-101.SCH" for r in rows)
    assert find_exhibit_99(rows)["type"] != "EX-101.SCH"


def test_fetch_document_bytes_returns_raw_content_unmodified():
    body = b"%PDF-1.4 \x00\xff not valid utf-8"

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    result = fetch_document_bytes(client, cik=320193, accession_number="0000320193-26-000018", document_name="deck.pdf")

    assert result == body


def test_fetch_document_bytes_raises_on_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_document_bytes(client, cik=1, accession_number="0000000001-25-000001", document_name="x.pdf")
