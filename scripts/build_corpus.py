"""
Runs the pull -> chunk -> embed -> insert pipeline
(scripts.chunk_filing.process_ticker) for several tickers in one process,
sharing a single httpx.Client, a single ticker-map load, and (with
--insert) a single Postgres connection across all of them.

process_ticker() already catches every stage's failure and returns a
ProcessResult instead of raising, so this script's loop needs no
try/except of its own: one ticker's failure is just a ProcessResult with
status="error", and the loop moves on to the next ticker rather than
aborting the whole run.

Usage:
    python scripts/build_corpus.py AAPL MSFT NVDA            # fetch + chunk + print a summary per ticker
    python scripts/build_corpus.py AAPL MSFT NVDA --form 10-Q
    python scripts/build_corpus.py AAPL MSFT NVDA --insert   # also embed + insert into Postgres
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import httpx
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_connection  # noqa: E402
from scripts.chunk_filing import ProcessResult, process_ticker  # noqa: E402
from scripts.edgar_pull import load_ticker_map  # noqa: E402


# Distinct exit codes so a caller (shell, CI) can tell "some tickers failed"
# from "nothing worked" without parsing the output.
EXIT_OK = 0
EXIT_ALL_FAILED = 1  # every ticker failed, or the run couldn't start (ticker map, DB)
EXIT_PARTIAL = 2  # at least one ticker succeeded and at least one failed


def exit_code_for(results: list[ProcessResult]) -> int:
    failed = sum(r.status != "ok" for r in results)
    if failed == 0:
        return EXIT_OK
    return EXIT_ALL_FAILED if failed == len(results) else EXIT_PARTIAL


def format_result_detail(r: ProcessResult) -> str:
    """One-line outcome for a ticker, shared by the live line and the summary."""
    if r.status != "ok":
        return f"ERROR ({r.stage}): {r.error}"
    detail = f"{r.chunk_count} chunks, {r.section_count} section(s)"
    if r.inserted is not None:
        detail += f", inserted {r.inserted}"
    return detail


def print_summary_table(results: list[ProcessResult]) -> None:
    print("\nSummary:")
    for r in results:
        print(f"  {r.ticker:<6} {r.status:<5} {format_result_detail(r)}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk (and optionally insert) each ticker's most recent 10-K/10-Q.")
    parser.add_argument("tickers", nargs="+", help="Stock tickers, e.g. AAPL MSFT NVDA")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"], help="Form type to chunk")
    parser.add_argument("--insert", action="store_true", help="Also embed and write chunks into Postgres")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="With --insert: delete each filing's existing rows first (use after the chunker changes)",
    )
    parser.add_argument("--refresh-ticker-cache", action="store_true", help="Re-download company_tickers.json")
    args = parser.parse_args(argv)
    if args.replace and not args.insert:
        parser.error("--replace requires --insert")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    tickers = [t.upper() for t in args.tickers]

    # One connection shared across every ticker in the run (opened before
    # any EDGAR call, same reasoning as chunk_filing.py's main()) rather
    # than one per ticker.
    try:
        conn = get_connection() if args.insert else None
    except psycopg.Error as exc:
        print(f"ERROR: could not connect to Postgres: {exc}", file=sys.stderr)
        return EXIT_ALL_FAILED
    results: list[ProcessResult] = []
    try:
        with httpx.Client() as client:
            try:
                ticker_map = load_ticker_map(
                    client, **({"force_refresh": True} if args.refresh_ticker_cache else {})
                )
            except (httpx.HTTPError, ValueError) as exc:
                print(f"ERROR: could not download SEC's ticker map: {exc}", file=sys.stderr)
                return EXIT_ALL_FAILED

            for ticker in tickers:
                extra = {"replace": True} if args.replace else {}
                result = process_ticker(client, ticker_map, ticker, args.form, args.insert, conn=conn, **extra)
                results.append(result)
                if result.status == "ok":
                    print(f"[{ticker}] ok — {format_result_detail(result)}")
                else:
                    print(f"[{ticker}] {format_result_detail(result)}", file=sys.stderr)
    finally:
        if conn is not None:
            conn.close()

    print_summary_table(results)

    return exit_code_for(results)


if __name__ == "__main__":
    sys.exit(main())
