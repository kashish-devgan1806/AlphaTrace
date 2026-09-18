"""
Session 5 deliverable: fetch a filer's most recent 10-K/10-Q, split it into
section-aware chunks (app.chunker.chunk_filing), and optionally embed +
write them into the chunks table (app.chunks.batch_insert_chunks).

Usage:
    python scripts/chunk_filing.py AAPL                # fetch + chunk + print a summary
    python scripts/chunk_filing.py AAPL --form 10-Q     # most recent 10-Q instead of 10-K
    python scripts/chunk_filing.py AAPL --insert        # also embed + insert into Postgres
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import chunk_filing  # noqa: E402
from app.chunks import batch_insert_chunks  # noqa: E402
from app.db import get_connection  # noqa: E402
from scripts.edgar_pull import (  # noqa: E402
    fetch_primary_document,
    fetch_submissions,
    load_ticker_map,
)


def find_latest_filing(submissions: dict, form: str) -> Optional[dict]:
    """Return {accessionNumber, primaryDocument, filingDate, reportDate} for
    the most recent filing of the given form type in submissions'
    filings.recent window, or None if there isn't one.

    Scans in order and returns on the first match: fetch_submissions()'s
    parallel arrays are already newest-first (SEC's own ordering), so the
    first match is the most recent one — no separate date sort needed.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    for i, f in enumerate(forms):
        if f == form:
            return {
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
                "filingDate": recent["filingDate"][i],
                "reportDate": report_dates[i] if i < len(report_dates) else "",
            }
    return None


def print_chunk_summary(ticker: str, form: str, filing: dict, records: list) -> None:
    section_counts: dict[str, int] = {}
    for r in records:
        section_counts[r.section] = section_counts.get(r.section, 0) + 1

    print(f"\n{ticker} {form} ({filing['accessionNumber']}, filed {filing['filingDate']})")
    print(f"  {len(records)} chunks across {len(section_counts)} section(s)")
    for section, count in section_counts.items():
        print(f"    [{count:>3}] {section}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk a filer's most recent 10-K/10-Q.")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"], help="Form type to chunk")
    parser.add_argument("--insert", action="store_true", help="Also embed and write chunks into Postgres")
    args = parser.parse_args(argv)

    ticker = args.ticker.upper()

    with httpx.Client() as client:
        try:
            ticker_map = load_ticker_map(client)
        except httpx.HTTPError as exc:
            print(f"ERROR: could not download SEC's ticker map: {exc}", file=sys.stderr)
            return 1

        cik = ticker_map.get(ticker)
        if cik is None:
            print(f"ERROR: '{ticker}' is not in SEC's ticker list — check the symbol.", file=sys.stderr)
            return 1

        try:
            submissions = fetch_submissions(client, cik)
        except httpx.HTTPError as exc:
            print(f"ERROR: could not fetch submissions for {ticker}: {exc}", file=sys.stderr)
            return 1

        filing = find_latest_filing(submissions, args.form)
        if filing is None:
            print(f"ERROR: no recent {args.form} found for {ticker}.", file=sys.stderr)
            return 1

        try:
            html = fetch_primary_document(client, cik, filing["accessionNumber"], filing["primaryDocument"])
        except httpx.HTTPError as exc:
            print(f"ERROR: could not fetch primary document: {exc}", file=sys.stderr)
            return 1

    metadata = {
        "ticker": ticker,
        "form": args.form,
        "filing_date": filing["filingDate"],
        "report_date": filing["reportDate"],
    }
    records = chunk_filing(filing["accessionNumber"], html, metadata=metadata)

    print_chunk_summary(ticker, args.form, filing, records)

    if args.insert:
        conn = get_connection()
        try:
            inserted = batch_insert_chunks(conn, records)
        finally:
            conn.close()
        print(f"\nInserted {inserted} chunks into Postgres.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
