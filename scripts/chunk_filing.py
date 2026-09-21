"""
Session 5 deliverable: fetch a filer's most recent 10-K/10-Q, split it into
section-aware chunks (app.chunker.chunk_filing), and optionally embed +
write them into the chunks table (app.chunks.batch_insert_chunks).

Session 6 deliverable: the single-ticker pipeline body that used to live
directly inside main() is now process_ticker() — a function that catches
each stage's failure itself and returns a ProcessResult instead of printing
+ returning an exit code. This is what lets scripts/build_corpus.py run the
same pipeline for several tickers in one process without one ticker's
failure aborting the others; main() below is now a thin wrapper around it,
preserving the single-ticker CLI's exact prior output and exit codes.

Usage:
    python scripts/chunk_filing.py AAPL                # fetch + chunk + print a summary
    python scripts/chunk_filing.py AAPL --form 10-Q     # most recent 10-Q instead of 10-K
    python scripts/chunk_filing.py AAPL --insert        # also embed + insert into Postgres
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import httpx
import psycopg

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


@dataclass
class ProcessResult:
    """Outcome of running one ticker through pull -> chunk -> (embed+insert).

    status="ok" means every requested stage completed; status="error" means
    exactly one stage failed, named by `stage`, with `error` holding the
    same message chunk_filing.py's old inline main() used to print
    directly. chunk_count/section_count/section_counts stay at their zero
    defaults when chunking was never reached. inserted is None unless
    insert=True and the insert actually succeeded (0 is a legitimate,
    distinct outcome from "never attempted").
    """

    ticker: str
    status: str  # "ok" | "error"
    stage: Optional[str] = None
    error: Optional[str] = None
    form: Optional[str] = None
    filing: Optional[dict] = None
    chunk_count: int = 0
    section_count: int = 0
    section_counts: dict = field(default_factory=dict)
    inserted: Optional[int] = None


def process_ticker(
    client: httpx.Client,
    ticker_map: dict[str, int],
    ticker: str,
    form: str,
    insert: bool,
    conn: Optional[psycopg.Connection] = None,
    replace: bool = False,
) -> ProcessResult:
    """Run one ticker through the full pull -> chunk -> (embed+insert)
    pipeline, catching every stage's failure itself rather than letting it
    raise. This is what lets a caller (main() below, or
    scripts/build_corpus.py) run several tickers back to back without one
    ticker's failure stopping the others — a failure here is just data on
    the returned ProcessResult, not an exception.
    """
    ticker = ticker.upper()

    if insert and conn is None:
        # Fail before any network call rather than after fetching and
        # chunking a whole filing, with a message that names the real cause.
        return ProcessResult(
            ticker, "error", stage="config", error="insert=True requires a database connection (conn)."
        )

    cik = ticker_map.get(ticker)
    if cik is None:
        return ProcessResult(
            ticker,
            "error",
            stage="ticker_lookup",
            error=f"'{ticker}' is not in SEC's ticker list — check the symbol.",
        )

    try:
        submissions = fetch_submissions(client, cik)
    except (httpx.HTTPError, ValueError) as exc:
        # ValueError covers a 200 response whose body isn't valid JSON
        # (resp.json() raises JSONDecodeError, a ValueError subclass).
        return ProcessResult(
            ticker, "error", stage="fetch_submissions", error=f"could not fetch submissions for {ticker}: {exc}"
        )

    filing = find_latest_filing(submissions, form)
    if filing is None:
        return ProcessResult(
            ticker, "error", stage="find_filing", error=f"no recent {form} found for {ticker}.", form=form
        )

    try:
        html = fetch_primary_document(client, cik, filing["accessionNumber"], filing["primaryDocument"])
    except httpx.HTTPError as exc:
        return ProcessResult(
            ticker,
            "error",
            stage="fetch_document",
            error=f"could not fetch primary document: {exc}",
            form=form,
            filing=filing,
        )

    metadata = {
        "ticker": ticker,
        "form": form,
        "filing_date": filing["filingDate"],
        "report_date": filing["reportDate"],
    }
    try:
        records = chunk_filing(filing["accessionNumber"], html, metadata=metadata)
    except Exception as exc:
        return ProcessResult(
            ticker,
            "error",
            stage="chunk",
            error=f"could not chunk filing {filing['accessionNumber']}: {exc}",
            form=form,
            filing=filing,
        )

    section_counts: dict[str, int] = {}
    for r in records:
        section_counts[r.section] = section_counts.get(r.section, 0) + 1

    inserted: Optional[int] = None
    if insert:
        try:
            # Only pass replace when set, so it stays a no-op for callers
            # (and test doubles) that predate the option.
            inserted = batch_insert_chunks(conn, records, **({"replace": True} if replace else {}))
        except Exception as exc:
            return ProcessResult(
                ticker,
                "error",
                stage="insert",
                error=f"could not insert chunks into Postgres: {exc}",
                form=form,
                filing=filing,
                chunk_count=len(records),
                section_count=len(section_counts),
                section_counts=section_counts,
            )

    return ProcessResult(
        ticker,
        "ok",
        form=form,
        filing=filing,
        chunk_count=len(records),
        section_count=len(section_counts),
        section_counts=section_counts,
        inserted=inserted,
    )


def print_chunk_summary(result: ProcessResult) -> None:
    filing = result.filing
    print(f"\n{result.ticker} {result.form} ({filing['accessionNumber']}, filed {filing['filingDate']})")
    print(f"  {result.chunk_count} chunks across {result.section_count} section(s)")
    for section, count in result.section_counts.items():
        print(f"    [{count:>3}] {section}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk a filer's most recent 10-K/10-Q.")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"], help="Form type to chunk")
    parser.add_argument("--insert", action="store_true", help="Also embed and write chunks into Postgres")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="With --insert: delete the filing's existing rows first (use after the chunker changes)",
    )
    parser.add_argument("--refresh-ticker-cache", action="store_true", help="Re-download company_tickers.json")
    args = parser.parse_args(argv)
    if args.replace and not args.insert:
        parser.error("--replace requires --insert")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    ticker = args.ticker.upper()

    # Opened before any EDGAR call (rather than after chunking, as the
    # pre-Session-6 version did) since process_ticker() now takes the
    # connection as a parameter — a dead DB now fails fast instead of
    # burning SEC rate-limit budget first.
    try:
        conn = get_connection() if args.insert else None
    except psycopg.Error as exc:
        print(f"ERROR: could not connect to Postgres: {exc}", file=sys.stderr)
        return 1
    try:
        with httpx.Client() as client:
            try:
                ticker_map = load_ticker_map(client, **({"force_refresh": True} if args.refresh_ticker_cache else {}))
            except (httpx.HTTPError, ValueError) as exc:
                print(f"ERROR: could not download SEC's ticker map: {exc}", file=sys.stderr)
                return 1

            result = process_ticker(
                client, ticker_map, ticker, args.form, args.insert, conn=conn, replace=args.replace
            )
    finally:
        if conn is not None:
            conn.close()

    if result.status == "error":
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    print_chunk_summary(result)
    if args.insert:
        print(f"\nInserted {result.inserted} chunks into Postgres.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
