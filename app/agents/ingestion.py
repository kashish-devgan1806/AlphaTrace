"""Ingestion Agent: wraps Phase 0's EDGAR/XBRL clients into a LangGraph node.

Given a ticker, pulls the latest 10-K, 10-Q, and 8-K, the XBRL facts, and —
when the 8-K has one — its Exhibit 99.x slide deck, then hands off a
normalized document_bundle for the Indexing Agent to chunk/embed. Never
raises: every failure is recorded in `errors`. A missing form (e.g. no 8-K
this quarter) or missing slide-deck exhibit is data, not an error; a fetch
that actually fails for one piece is recorded but doesn't block the rest —
the node only comes back with no document_bundle at all when nothing could
be fetched.
"""
from __future__ import annotations

import httpx

from app.state import AgentState
from scripts.chunk_filing import find_latest_filing
from scripts.edgar_pull import (
    DOCUMENT_URL,
    extract_gaap_facts,
    fetch_companyfacts,
    fetch_document_bytes,
    fetch_filing_index,
    fetch_primary_document,
    fetch_submissions,
    find_exhibit_99,
    load_ticker_map,
)

FORMS = ("10-K", "10-Q", "8-K")


def _fetch_form_bundle(client: httpx.Client, cik: int, ticker: str, form: str, filing, errors: list[str]):
    """One form's fetch. `filing` is None when the form has no recent
    filing at all — that's not an error, so it's passed through as None
    without touching `errors`. A fetch failure for a filing that *was*
    found is recorded but doesn't stop the other forms from being tried."""
    if filing is None:
        return None
    try:
        html = fetch_primary_document(client, cik, filing["accessionNumber"], filing["primaryDocument"])
    except httpx.HTTPError as exc:
        errors.append(f"ingest: fetch failed for {ticker} {form}: {exc}")
        return None
    return {
        "html": html,
        "metadata": {
            "ticker": ticker,
            "form": form,
            "filing_date": filing["filingDate"],
            "report_date": filing["reportDate"],
        },
    }


def _fetch_slide_deck(client: httpx.Client, cik: int, ticker: str, eight_k, errors: list[str]):
    """The 8-K's Exhibit 99.x, if one exists — the earnings-deck convention
    under Item 2.02/7.01. No 8-K, or an 8-K with no Exhibit 99.x, is a
    legitimate absence and returns None without an error; only an actual
    fetch failure is recorded."""
    if eight_k is None:
        return None
    accession_number = eight_k["accessionNumber"]
    try:
        exhibit = find_exhibit_99(fetch_filing_index(client, cik, accession_number))
        if exhibit is None:
            return None
        name = exhibit["name"]
        url = DOCUMENT_URL.format(cik=cik, accession_nodash=accession_number.replace("-", ""), primary_document=name)
        if name.lower().endswith(".pdf"):
            content = fetch_document_bytes(client, cik, accession_number, name)
            return {
                "exhibit_name": name,
                "exhibit_type": exhibit.get("type"),
                "content_type": "pdf",
                "url": url,
                "html": None,
                "bytes": content,
            }
        html = fetch_primary_document(client, cik, accession_number, name)
        return {
            "exhibit_name": name,
            "exhibit_type": exhibit.get("type"),
            "content_type": "html",
            "url": url,
            "html": html,
            "bytes": None,
        }
    except httpx.HTTPError as exc:
        errors.append(f"ingest: slide deck fetch failed for {ticker}: {exc}")
        return None


def ingest_node(state: AgentState) -> dict:
    """Pull the ticker's latest 10-K, 10-Q, and 8-K, its XBRL facts, and its
    8-K's slide deck if one exists. `state["form"]` (default "10-K") no
    longer selects what's fetched — it only selects which form `index`
    treats as primary."""
    ticker = state["ticker"].upper()
    primary_form = state.get("form", "10-K")
    errors: list[str] = []

    with httpx.Client() as client:
        try:
            ticker_map = load_ticker_map(client)
        except (httpx.HTTPError, ValueError) as exc:
            return {"errors": [f"ingest: could not load SEC ticker map: {exc}"]}

        cik = ticker_map.get(ticker)
        if cik is None:
            return {"errors": [f"ingest: '{ticker}' is not in SEC's ticker list"]}

        try:
            submissions = fetch_submissions(client, cik)
        except (httpx.HTTPError, ValueError) as exc:
            return {"errors": [f"ingest: EDGAR submissions fetch failed for {ticker}: {exc}"]}

        filings = {form: find_latest_filing(submissions, form) for form in FORMS}
        bundle_filings = {
            form: _fetch_form_bundle(client, cik, ticker, form, filings[form], errors) for form in FORMS
        }

        if all(bundle_filings[form] is None for form in FORMS):
            errors.append(f"ingest: no 10-K, 10-Q, or 8-K content could be fetched for {ticker}")
            return {"filings": filings, "errors": errors}

        xbrl_facts = None
        try:
            xbrl_facts = extract_gaap_facts(fetch_companyfacts(client, cik))
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"ingest: XBRL companyfacts fetch failed for {ticker}: {exc}")

        slide_deck = _fetch_slide_deck(client, cik, ticker, filings["8-K"], errors)

    result = {
        "form": primary_form,
        "filings": filings,
        "document_bundle": {
            "filings": bundle_filings,
            "xbrl_facts": xbrl_facts,
            "slide_deck": slide_deck,
            "metadata": {"ticker": ticker, "primary_form": primary_form},
        },
    }
    if errors:
        result["errors"] = errors
    return result
