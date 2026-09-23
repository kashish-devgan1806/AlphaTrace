"""Trivial 2-node LangGraph scaffold.

Proves the AgentState wiring works end to end using pieces that already
exist — no LLM call, no new API key. `ingest` wraps the EDGAR pull
(scripts/edgar_pull.py) into a node; `index` wraps the section-aware
chunker (app/chunker.py) into a second node that consumes `ingest`'s
output. This is deliberately shaped like the eventual Ingestion Agent /
Indexing Agent split rather than one big node, so the boundary is already
right when those agents get built for real.

Why LangGraph's StateGraph over a plain function call chain: nothing here
needs a cycle yet, but the Critic -> Analyst revise loop does, and
StateGraph is what makes that cycle possible later without a rewrite —
building today's straight-line graph on the same primitive means the
later multi-agent wiring is an extension, not a migration.
"""
from __future__ import annotations

import httpx

from app.chunker import chunk_filing
from app.state import AgentState
from langgraph.graph import END, START, StateGraph
from scripts.chunk_filing import find_latest_filing
from scripts.edgar_pull import fetch_primary_document, fetch_submissions, load_ticker_map


def ingest_node(state: AgentState) -> dict:
    """Pull the ticker's latest filing of `form` and its primary document
    body. Never raises — a failure is recorded in `errors` and every
    downstream field is simply left unset, which `index_node` checks for."""
    ticker = state["ticker"].upper()
    form = state.get("form", "10-K")

    try:
        with httpx.Client() as client:
            ticker_map = load_ticker_map(client)
            cik = ticker_map.get(ticker)
            if cik is None:
                return {"errors": [f"ingest: '{ticker}' is not in SEC's ticker list"]}

            submissions = fetch_submissions(client, cik)
            filing = find_latest_filing(submissions, form)
            if filing is None:
                return {"errors": [f"ingest: no recent {form} found for {ticker}"]}

            html = fetch_primary_document(client, cik, filing["accessionNumber"], filing["primaryDocument"])
    except (httpx.HTTPError, ValueError) as exc:
        return {"errors": [f"ingest: EDGAR fetch failed for {ticker}: {exc}"]}

    return {
        "form": form,
        "filing": filing,
        "document_bundle": {
            "html": html,
            "metadata": {
                "ticker": ticker,
                "form": form,
                "filing_date": filing["filingDate"],
                "report_date": filing["reportDate"],
            },
        },
    }


def index_node(state: AgentState) -> dict:
    """Chunk `ingest`'s document_bundle. Requires `filing` and
    `document_bundle` to already be set — if ingest_node failed, both are
    absent and this records why rather than raising a KeyError."""
    bundle = state.get("document_bundle")
    filing = state.get("filing")
    if not bundle or not filing:
        return {"errors": ["index: no document_bundle to chunk (ingest likely failed)"]}

    records = chunk_filing(filing["accessionNumber"], bundle["html"], metadata=bundle["metadata"])
    section_counts: dict[str, int] = {}
    for record in records:
        section_counts[record.section] = section_counts.get(record.section, 0) + 1

    return {"chunks": records, "chunk_count": len(records), "section_counts": section_counts}


def build_graph():
    """Compile the 2-node ingest -> index graph. `state["ticker"]` (and
    optionally `state["form"]`, default "10-K") is the only required input."""
    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("index", index_node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "index")
    graph.add_edge("index", END)
    return graph.compile()
