"""2-node ingest -> index LangGraph scaffold.

`ingest` (app/agents/ingestion.py, the Ingestion Agent) and `index`
(app/agents/indexing.py, the Indexing Agent) are only wired together here,
not implemented here.

Why LangGraph's StateGraph over a plain function call chain: nothing here
needs a cycle yet, but the Critic -> Analyst revise loop does, and
StateGraph is what makes that cycle possible later without a rewrite —
building today's straight-line graph on the same primitive means the
later multi-agent wiring is an extension, not a migration.
"""
from __future__ import annotations

from app.agents.indexing import index_node
from app.agents.ingestion import ingest_node
from app.state import AgentState
from langgraph.graph import END, START, StateGraph

__all__ = ["build_graph", "index_node", "ingest_node"]


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
