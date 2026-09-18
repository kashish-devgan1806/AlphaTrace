"""
Session 6 deliverable: run the pull -> chunk -> embed -> insert pipeline
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
import sys
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_connection  # noqa: E402
from scripts.chunk_filing import ProcessResult, process_ticker  # noqa: E402
from scripts.edgar_pull import load_ticker_map  # noqa: E402


def print_summary_table(results: list[ProcessResult]) -> None:
    print("\nSummary:")
    for r in results:
        if r.status == "ok":
            detail = f"{r.chunk_count} chunks, {r.section_count} section(s)"
            if r.inserted is not None:
                detail += f", inserted {r.inserted}"
        else:
            detail = f"ERROR ({r.stage}): {r.error}"
        print(f"  {r.ticker:<6} {r.status:<5} {detail}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk (and optionally insert) each ticker's most recent 10-K/10-Q.")
    parser.add_argument("tickers", nargs="+", help="Stock tickers, e.g. AAPL MSFT NVDA")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"], help="Form type to chunk")
    parser.add_argument("--insert", action="store_true", help="Also embed and write chunks into Postgres")
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.tickers]

    # One connection shared across every ticker in the run (opened before
    # any EDGAR call, same reasoning as chunk_filing.py's main()) rather
    # than one per ticker.
    conn = get_connection() if args.insert else None
    results: list[ProcessResult] = []
    try:
        with httpx.Client() as client:
            try:
                ticker_map = load_ticker_map(client)
            except httpx.HTTPError as exc:
                print(f"ERROR: could not download SEC's ticker map: {exc}", file=sys.stderr)
                return 1

            for ticker in tickers:
                result = process_ticker(client, ticker_map, ticker, args.form, args.insert, conn=conn)
                results.append(result)
                if result.status == "ok":
                    detail = f"{result.chunk_count} chunks, {result.section_count} section(s)"
                    if result.inserted is not None:
                        detail += f", inserted {result.inserted}"
                    print(f"[{ticker}] ok — {detail}")
                else:
                    print(f"[{ticker}] ERROR ({result.stage}): {result.error}", file=sys.stderr)
    finally:
        if conn is not None:
            conn.close()

    print_summary_table(results)

    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
