"""2-node ingest -> index LangGraph scaffold.

`ingest` (app/agents/ingestion.py, the real Ingestion Agent) is only wired
in here, not implemented here; `index` wraps the section-aware chunker
(app/chunker.py) into a second node that consumes `ingest`'s output for
whichever form state["form"] names as primary. This is deliberately
shaped like the eventual Ingestion Agent / Indexing Agent split rather than
one big node, so the boundary is already right when the Indexing Agent gets
built for real.

Why LangGraph's StateGraph over a plain function call chain: nothing here
needs a cycle yet, but the Critic -> Analyst revise loop does, and
StateGraph is what makes that cycle possible later without a rewrite —
building today's straight-line graph on the same primitive means the
later multi-agent wiring is an extension, not a migration.
"""
from __future__ import annotations

from app.agents.ingestion import ingest_node
from app.chunker import chunk_filing
from app.state import AgentState
from langgraph.graph import END, START, StateGraph

__all__ = ["build_graph", "index_node", "ingest_node"]


def index_node(state: AgentState) -> dict:
    """Chunk the primary form's document out of `ingest`'s document_bundle
    (state["form"], default "10-K", selects which form is primary). Requires
    that form to actually be present in both `filings` and
    `document_bundle["filings"]` — if ingest_node failed, or that one form's
    fetch failed while others succeeded, this records why rather than
    raising a KeyError."""
    bundle = state.get("document_bundle")
    if not bundle:
        return {"errors": ["index: no document_bundle to chunk (ingest likely failed)"]}

    primary_form = state.get("form", "10-K")
    filing = (state.get("filings") or {}).get(primary_form)
    primary = bundle.get("filings", {}).get(primary_form)
    if not filing or not primary:
        return {"errors": [f"index: no {primary_form} filing available to chunk (ingest found none)"]}

    records = chunk_filing(filing["accessionNumber"], primary["html"], metadata=primary["metadata"])
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
