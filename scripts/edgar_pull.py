"""
Session 1 deliverable: hit SEC EDGAR's submissions API for one ticker and
print its filing metadata.

Session 2 deliverable: also pull the companyfacts API and extract a handful
of GAAP tags (Revenues, GrossProfit, NetIncomeLoss) into a plain dict.

Usage:
    python scripts/edgar_pull.py AAPL
    python scripts/edgar_pull.py AAPL --limit 5
    python scripts/edgar_pull.py AAPL --facts
    python scripts/edgar_pull.py NOTATICKER      # exercises the error path

Session 5 deliverable: fetch_primary_document() — the actual filing HTML
(10-K/10-Q body text), not just metadata/facts about it. Feeds
app.chunker.chunk_filing() (see scripts/chunk_filing.py).

Four SEC endpoints are involved:
  1. https://www.sec.gov/files/company_tickers.json
     A static file mapping ticker -> CIK (SEC's internal entity ID). There is
     no "look up CIK by ticker" API endpoint, so every EDGAR tool downloads
     this file once and keeps a local map.
  2. https://data.sec.gov/submissions/CIK##########.json
     The per-company filing history: entity metadata plus every recent
     filing's form type, dates, and accession number. CIK must be
     zero-padded to 10 digits in the URL.
  3. https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
     Every XBRL fact the company has ever tagged, keyed by GAAP tag name.
     Same CIK format. Each tag's values live under
     facts.us-gaap.<Tag>.units.<UNIT>[], one array entry per filing that
     reported it — so picking "the" value for a fiscal year means filtering
     by fy/fp/form, not indexing the array directly.
  4. https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/<primary_document>
     The filing's primary document itself (the 10-K/10-Q HTML body).
     accessionNumber and primaryDocument both come from endpoint 2's
     parallel arrays — this endpoint doesn't do its own lookup, it just
     assembles the URL SEC's archive expects.
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
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "company_tickers.json"

# Canonical name -> candidate GAAP tags to try, in order. A single canonical
# concept can be filed under more than one tag: most filers adopted ASC 606
# years ago and tag revenue as RevenueFromContractWithCustomerExcludingAssessedTax
# instead of the older Revenues tag, so Revenues alone comes back empty for
# them. Trying candidates in order and recording which one hit means callers
# don't have to duplicate this knowledge.
GAAP_TAG_CANDIDATES: dict[str, list[str]] = {
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "GrossProfit": ["GrossProfit"],
    "NetIncomeLoss": ["NetIncomeLoss"],
}


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


def fetch_companyfacts(client: httpx.Client, cik: int) -> dict:
    url = COMPANYFACTS_URL.format(cik=cik)
    resp = client.get(url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_primary_document(client: httpx.Client, cik: int, accession_number: str, primary_document: str) -> str:
    """Fetch a filing's primary document — the actual 10-K/10-Q HTML, not
    just metadata about it. Needs accession_number and primary_document
    exactly as fetch_submissions() returns them (recent.accessionNumber[i],
    recent.primaryDocument[i] for the same filing index).

    The archive URL's path segment drops the dashes from the accession
    number (but the filing's own filename elsewhere keeps them) — that's
    SEC's own inconsistency, not a choice made here.

    Larger and slower than the JSON endpoints above (a filing body can run
    several MB), hence the longer timeout.
    """
    accession_nodash = accession_number.replace("-", "")
    url = DOCUMENT_URL.format(cik=cik, accession_nodash=accession_nodash, primary_document=primary_document)
    resp = client.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.text


def _latest_annual_entry(tag_data: dict) -> Optional[dict]:
    """Pick the single best entry out of one tag's units.<UNIT>[] array.

    Filters to full fiscal year 10-K entries (form == "10-K", fp == "FY") —
    SEC never tags a standalone Q4 duration fact, so fp == "FY" is how the
    annual figure is actually found, not an arbitrary choice. Ties are
    broken by taking the entry with the latest period end date, since the
    same fiscal year can appear more than once (e.g. also as a prior-year
    comparative in a later filing).

    Only looks at the USD unit: these three tags are dollar P&L line items,
    so a non-USD unit would mean something is wrong with the tag choice, not
    that a legitimate alternate value should be picked up.
    """
    entries = tag_data.get("units", {}).get("USD", [])
    annual = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    if not annual:
        return None
    return max(annual, key=lambda e: e.get("end", ""))


def extract_gaap_facts(companyfacts: dict) -> dict[str, Optional[dict]]:
    """Return {canonical_name: {tag, val, fy, end, accn} | None}.

    `None` means none of the candidate tags for that concept had a matching
    annual (10-K, FY) USD entry — worth surfacing explicitly rather than
    silently omitting the key, since a missing GAAP concept is itself useful
    information (e.g. a filer that reports under IFRS instead of US-GAAP).

    A canonical name can have multiple candidate tags with *some* annual
    data each — e.g. Apple's old `Revenues` tag still has entries up through
    FY2018, from before it switched to
    RevenueFromContractWithCustomerExcludingAssessedTax under ASC 606. Taking
    the first candidate tag that merely has a match would silently pin the
    result to that stale FY2018 value forever. So every candidate tag is
    checked and the entry with the latest period end wins, not the first
    tag in the list that happens to have anything at all.
    """
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    result: dict[str, Optional[dict]] = {}

    for canonical_name, candidate_tags in GAAP_TAG_CANDIDATES.items():
        candidates = []
        for tag in candidate_tags:
            if tag not in us_gaap:
                continue
            entry = _latest_annual_entry(us_gaap[tag])
            if entry is not None:
                candidates.append((tag, entry))

        if not candidates:
            result[canonical_name] = None
        else:
            used_tag, match = max(candidates, key=lambda pair: pair[1].get("end", ""))
            result[canonical_name] = {
                "tag": used_tag,
                "val": match["val"],
                "fy": match.get("fy"),
                "end": match.get("end"),
                "accn": match.get("accn"),
            }

    return result


def print_gaap_facts(ticker: str, facts: dict[str, Optional[dict]]) -> None:
    print(f"\n{ticker} — latest annual (10-K, FY) GAAP facts:")
    for canonical_name, entry in facts.items():
        if entry is None:
            print(f"  {canonical_name}: not found (checked {GAAP_TAG_CANDIDATES[canonical_name]})")
        else:
            print(
                f"  {canonical_name} [{entry['tag']}]: ${entry['val']:,} "
                f"(FY{entry['fy']}, period end {entry['end']}, accn={entry['accn']})"
            )


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
    parser.add_argument(
        "--facts", action="store_true", help="Also pull companyfacts and extract key GAAP tags"
    )
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

        if args.facts:
            try:
                companyfacts = fetch_companyfacts(client, cik)
            except httpx.HTTPStatusError as exc:
                print(
                    f"ERROR: SEC returned HTTP {exc.response.status_code} for companyfacts CIK {cik}: {exc}",
                    file=sys.stderr,
                )
                return 1
            except httpx.HTTPError as exc:
                print(f"ERROR: network error contacting SEC: {exc}", file=sys.stderr)
                return 1

            facts = extract_gaap_facts(companyfacts)
            print_gaap_facts(ticker, facts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
