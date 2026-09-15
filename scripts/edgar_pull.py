"""
Session 1 deliverable: hit SEC EDGAR's submissions API for one ticker and
print its filing metadata.

Usage:
    python scripts/edgar_pull.py AAPL
    python scripts/edgar_pull.py AAPL --limit 5
    python scripts/edgar_pull.py NOTATICKER      # exercises the error path

Two SEC endpoints are involved:
  1. https://www.sec.gov/files/company_tickers.json
     A static file mapping ticker -> CIK (SEC's internal entity ID). There is
     no "look up CIK by ticker" API endpoint, so every EDGAR tool downloads
     this file once and keeps a local map.
  2. https://data.sec.gov/submissions/CIK##########.json
     The per-company filing history: entity metadata plus every recent
     filing's form type, dates, and accession number. CIK must be
     zero-padded to 10 digits in the URL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import httpx

# Running this file directly (`python scripts/edgar_pull.py`) only puts
# scripts/ on sys.path, not the project root, so `app` wouldn't otherwise
# be importable. Adding the root explicitly keeps the documented usage
# working without requiring `python -m scripts.edgar_pull` instead.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "company_tickers.json"


def _headers() -> dict:
    # SEC's fair-access policy rejects any request with no User-Agent (or a
    # generic one like the Python default "python-requests/2.x") as an
    # "Undeclared Automated Tool" — a 403 with a plain-text body, not JSON.
    # The required shape is "<Company/App name> <contact email>".
    return {
        "User-Agent": settings.sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def load_ticker_map(client: httpx.Client, force_refresh: bool = False) -> dict[str, int]:
    """Return {TICKER: cik_int}.

    Cached to disk because (a) the file is ~800KB and never changes within a
    single dev session, and (b) SEC caps ALL traffic — sec.gov and
    data.sec.gov combined — at 10 requests/second, so a script that re-fetches
    this on every run burns rate-limit budget for no reason.
    """
    if CACHE_PATH.exists() and not force_refresh:
        raw = json.loads(CACHE_PATH.read_text())
    else:
        resp = client.get(TICKER_MAP_URL, headers=_headers(), timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(raw))

    # Shape on disk: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, "1": {...}, ...}
    # The outer keys are meaningless row indices — only the inner dicts matter.
    return {row["ticker"].upper(): row["cik_str"] for row in raw.values()}


def fetch_submissions(client: httpx.Client, cik: int) -> dict:
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = client.get(url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def print_filing_metadata(ticker: str, data: dict, limit: int) -> None:
    """`data` is the raw submissions JSON. Fields used here:
      - name / cik / sic / sicDescription: entity identity, at the top level.
      - filings.recent: a dict of *parallel arrays* (not a list of row
        objects) — index i across form[], filingDate[], accessionNumber[]
        etc. all describe the same filing. This is SEC's format, not ours;
        it's compact but means the arrays must be zipped by index.
    """
    name = data.get("name", "UNKNOWN")
    cik = data.get("cik", "UNKNOWN")
    sic = data.get("sic") or "n/a"
    sic_desc = data.get("sicDescription") or "n/a"

    print(f"\n{ticker} — {name}")
    print(f"  CIK: {cik}   SIC: {sic} ({sic_desc})")

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])

    if not forms:
        # Not an error: a brand-new registrant, a shell company, or a ticker
        # SEC has since delisted can legitimately have zero recent filings.
        # The script should say so plainly rather than crash on an index
        # into an empty list.
        print("  No recent filings found for this entity.")
        return

    dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(limit, len(forms))
    print(f"  Most recent {n} filing(s):")
    for i in range(n):
        report_date = report_dates[i] if i < len(report_dates) and report_dates[i] else "n/a"
        print(
            f"    [{dates[i]}] {forms[i]:<8} report_date={report_date:<10} "
            f"accession={accessions[i]} doc={primary_docs[i]}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pull filing metadata for one ticker from SEC EDGAR.")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--limit", type=int, default=10, help="How many recent filings to print (default 10)")
    parser.add_argument("--refresh-ticker-cache", action="store_true", help="Re-download company_tickers.json")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()

    with httpx.Client() as client:
        try:
            ticker_map = load_ticker_map(client, force_refresh=args.refresh_ticker_cache)
        except httpx.HTTPError as exc:
            print(f"ERROR: could not download SEC's ticker map: {exc}", file=sys.stderr)
            return 1

        cik = ticker_map.get(ticker)
        if cik is None:
            print(f"ERROR: '{ticker}' is not in SEC's ticker list — check the symbol.", file=sys.stderr)
            return 1

        try:
            data = fetch_submissions(client, cik)
        except httpx.HTTPStatusError as exc:
            print(f"ERROR: SEC returned HTTP {exc.response.status_code} for CIK {cik}: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"ERROR: network error contacting SEC: {exc}", file=sys.stderr)
            return 1

    print_filing_metadata(ticker, data, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
