"""Shared AgentState schema.

Every node in the LangGraph graph — today's trivial 2-node scaffold
(app/graph.py) and, once the rest of the agent roster exists, all seven
agents — reads and writes this one TypedDict. Defining it once here,
before any agent exists, is the point: it's the contract every later
agent gets built against instead of improvising its own shape.

Field ownership (who writes, who reads):

  - ticker: written by the caller (graph input); read by ingest, index.
  - form: written by the caller (default "10-K"); no longer selects what
    ingest fetches — ingest always pulls the latest 10-K, 10-Q, and 8-K. It
    only selects which of the three `index` treats as the primary filing to
    chunk (until the indexing agent chunks all three itself).
  - filings: written by ingest; read by index. {"10-K": {...}|None,
    "10-Q": {...}|None, "8-K": {...}|None} — the raw SEC filing-metadata
    dict (accessionNumber/primaryDocument/filingDate/reportDate) per form,
    or None when that form has no recent filing. A missing form is a
    legitimate, non-error state, distinct from an entry in `errors`.
  - document_bundle: written by ingest; read by index, and later the
    indexing agent. Holds document_bundle["filings"][form] (html +
    metadata, mirroring `filings` 1:1 but only present when that form's
    primary document was actually fetched), document_bundle["xbrl_facts"]
    (extract_gaap_facts() output, or None if the XBRL fetch failed), and
    document_bundle["slide_deck"] (the 8-K's first Exhibit 99.x — html
    text or raw pdf bytes, plus its archive url — or None if there's no
    8-K, no Exhibit 99.x, or the fetch failed).
  - chunks: written by index; read later by analyst/sentiment/quant.
  - chunk_count, section_counts: written by index; read by the caller and
    smoke tests.
  - retrieved_evidence, draft_answer, citations: written by the analyst
    agent; read by the critic and synthesis agents.
  - sentiment_result: written by the sentiment agent; read by critic and
    synthesis.
  - quant_result: written by the quant agent; read by critic and
    synthesis.
  - critic_feedback: written by the critic agent; read by the analyst on
    a revise edge.
  - revision_count: written by the critic agent; read by itself, as the
    retry cap.
  - final_brief: written by the synthesis agent; read by the caller.
  - errors: written by any node; read by the caller / observability.

Write-order matters in exactly one place today: `index` reads
`document_bundle` and `filing`, both written by `ingest` — so `ingest` must
run before `index` (enforced by the edge in app/graph.py, not by the state
schema itself). Every other field is written by at most one node in
today's scaffold, so there's no write-order ambiguity yet. That changes
once Analyst, Sentiment, and Quant fan out in parallel and Critic's revise
edge writes `critic_feedback` back for Analyst to re-read — flagged here
so it isn't a surprise when that wiring lands.

`errors` uses `operator.add` as its reducer so every node's failures
accumulate across a run instead of the last node's error silently
overwriting an earlier one — the same accumulation the Critic's
verify-and-revise loop will need once retries are in play.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from app.chunks import ChunkRecord


class AgentState(TypedDict, total=False):
    # --- input ---
    ticker: str
    form: str

    # --- written by `ingest` (the ingestion agent) ---
    filings: dict[str, Optional[dict]]
    document_bundle: Optional[dict]

    # --- written by `index` (the indexing agent) ---
    chunks: list[ChunkRecord]
    chunk_count: int
    section_counts: dict[str, int]

    # --- reserved for the rest of the agent roster; unpopulated by today's scaffold ---
    retrieved_evidence: list[Any]
    draft_answer: Optional[str]
    citations: list[Any]
    sentiment_result: Optional[dict]
    quant_result: Optional[dict]
    critic_feedback: Optional[str]
    revision_count: int
    final_brief: Optional[dict]

    # --- cross-cutting ---
    errors: Annotated[list[str], operator.add]
