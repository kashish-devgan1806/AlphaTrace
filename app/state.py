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
  - chunks: written by index; read later by analyst/sentiment/quant. Holds
    text and table chunks together (ChunkRecord.chunk_type distinguishes
    them) across every filing form present in document_bundle, plus the
    slide deck's own text chunks when it's an HTML exhibit.
  - chunk_count, section_counts: written by index; read by the caller and
    smoke tests. chunk_count covers `chunks` only (text + table), not
    visual_pages.
  - visual_pages, visual_chunk_count: written by index. visual_pages holds
    one PageImage per rasterized slide-deck page (only when the deck is a
    PDF exhibit; empty otherwise) -- not yet embedded, ready for
    app.visual.batch_insert_page_embeddings(). visual_chunk_count is
    len(visual_pages), split out from chunk_count since they land in a
    different table (page_embeddings, not chunks).
  - question: written by the caller; read by the analyst agent. The
    question retrieval and the grounded-answer LLM call are both built
    against -- absent or blank, analyst_node records an error and returns
    without touching pgvector or the LLM.
  - transcript, prior_transcript: written by the caller; read by the
    sentiment agent. Raw Q&A transcript text (app.transcript's documented
    Q:/A: contract) for the current and prior quarter's earnings call --
    no live transcript-sourcing path exists yet, so these are supplied
    directly rather than populated by ingest_node/document_bundle.
    `transcript` absent or blank, sentiment_node records an error and
    returns without classifying anything; `prior_transcript` is optional
    -- its absence just skips the quarter-over-quarter comparison.
  - retrieved_evidence: written by the analyst agent; read by the critic
    and synthesis agents. The full reranked candidate pool actually handed
    to the LLM as context (app/agents/analyst.py's TOP_K, not just the
    chunks the LLM chose to cite) -- one dict per chunk: {chunk_id, doc_id,
    section, chunk_type, text, retrieval_score, rerank_score}. Keeping the
    whole pool (not just cited chunks) here means the Critic can check a
    numeric claim against evidence the Analyst saw but didn't cite.
  - draft_answer: written by the analyst agent; read by the critic and
    synthesis agents. Plain text with inline citation markers ("[1]",
    "[2]", ...) resolved against `citations` below -- the LLM's raw answer
    text, unreformatted.
  - citations: written by the analyst agent; read by the critic and
    synthesis agents. One dict per marker actually used in draft_answer,
    resolved against retrieved_evidence: {marker, chunk_id, doc_id,
    section, chunk_type, quote}. `quote` is the exact span the LLM claims
    supports that citation -- what a3d2's "verify 3 real answers'
    citations by hand" checks against the source chunk text. A marker the
    LLM cited that doesn't resolve to a retrieved chunk_id is dropped here
    and recorded in `errors` instead, not trusted silently.
  - sentiment_result: written by the sentiment agent; read by critic and
    synthesis. {segments: [{segment_id, question, answer, label,
    confidence}, ...], current_summary: {segment_count, hedging_count,
    confident_count, hedging_ratio}, prior_summary: <same shape as
    current_summary> | None, tone_shift: {hedging_ratio_delta, direction}
    | None}. prior_summary/tone_shift are None whenever `prior_transcript`
    wasn't supplied or its own comparison failed.
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
from typing import Annotated, Optional, TypedDict

from app.chunks import ChunkRecord
from app.visual import PageImage


class AgentState(TypedDict, total=False):
    # --- input ---
    ticker: str
    form: str
    question: str
    transcript: Optional[str]
    prior_transcript: Optional[str]

    # --- written by `ingest` (the ingestion agent) ---
    filings: dict[str, Optional[dict]]
    document_bundle: Optional[dict]

    # --- written by `index` (the indexing agent) ---
    chunks: list[ChunkRecord]
    chunk_count: int
    section_counts: dict[str, int]
    visual_pages: list[PageImage]
    visual_chunk_count: int

    # --- written by `analyst` (the research analyst agent) ---
    retrieved_evidence: list[dict]
    draft_answer: Optional[str]
    citations: list[dict]

    # --- reserved for the rest of the agent roster; unpopulated by today's scaffold ---
    sentiment_result: Optional[dict]
    quant_result: Optional[dict]
    critic_feedback: Optional[str]
    revision_count: int
    final_brief: Optional[dict]

    # --- cross-cutting ---
    errors: Annotated[list[str], operator.add]
