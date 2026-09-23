"""Shared AgentState schema.

Every node in the LangGraph graph — today's trivial 2-node scaffold
(app/graph.py) and, once the rest of the agent roster exists, all seven
agents — reads and writes this one TypedDict. Defining it once here,
before any agent exists, is the point: it's the contract every later
agent gets built against instead of improvising its own shape.

Field ownership (who writes, who reads):

  - ticker: written by the caller (graph input); read by ingest, index.
  - form: written by the caller (default "10-K"); read by ingest, index.
  - filing: written by ingest; read by index.
  - document_bundle: written by ingest; read by index, and later the
    indexing agent.
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
    filing: Optional[dict]
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
