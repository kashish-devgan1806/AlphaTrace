"""ingest -> index -> analyst LangGraph scaffold.

`ingest` (app/agents/ingestion.py), `index` (app/agents/indexing.py), and
`analyst` (app/agents/analyst.py) are only wired together here, not
implemented here.

Why LangGraph's StateGraph over a plain function call chain: nothing here
needs a cycle yet, but the Critic -> Analyst revise loop does, and
StateGraph is what makes that cycle possible later without a rewrite —
building today's straight-line graph on the same primitive means the
later multi-agent wiring is an extension, not a migration.

`analyst` reads from pgvector (via app.search.search), not from this run's
own `chunks` state -- ingest/index only produce not-yet-persisted records
(see their own docstrings), and nothing in this graph inserts them. A
fresh ticker run's analyst step answers from whatever an earlier,
separate `batch_insert_chunks` call already put in Postgres, not from the
same invocation's ingest/index output. Worth keeping in mind until a
later phase closes that loop.
"""
from __future__ import annotations

from app.agents.analyst import analyst_node
from app.agents.indexing import index_node
from app.agents.ingestion import ingest_node
from app.state import AgentState
from langgraph.graph import END, START, StateGraph

__all__ = ["build_graph", "analyst_node", "index_node", "ingest_node"]


def build_graph():
    """Compile the ingest -> index -> analyst graph. `state["ticker"]`
    (and optionally `state["form"]`, default "10-K") is required for
    ingest/index; `state["question"]` is required for analyst to produce
    an answer instead of an error."""
    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("index", index_node)
    graph.add_node("analyst", analyst_node)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "index")
    graph.add_edge("index", "analyst")
    graph.add_edge("analyst", END)
    return graph.compile()
