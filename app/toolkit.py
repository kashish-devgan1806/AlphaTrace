"""The Phase 0 toolkit surface (Session 8-9 / roadmap day f6 closeout).

Days 1-5 built five pieces — the EDGAR client, the XBRL extractor, the
Postgres/pgvector layer, the section-aware chunker, and (Session 8) the
LangGraph scaffold — as separate modules under app/ and scripts/. This
module is the one place that re-exports their stable entry points, so
Phase 1's seven agents import `from app.toolkit import ...` instead of
reaching into script internals that were never meant as a public surface
(scripts/edgar_pull.py and scripts/chunk_filing.py stay importable
directly too — nothing here moves their code, it only re-exports names).

Grouped by the piece each name belongs to, matching the roadmap's own
Day 1-5 breakdown:
"""
from __future__ import annotations

# --- EDGAR client (Day f1) ---
from scripts.edgar_pull import (
    fetch_primary_document,
    fetch_submissions,
    load_ticker_map,
)

# --- XBRL client (Day f2) ---
from scripts.edgar_pull import extract_gaap_facts, fetch_companyfacts

# --- DB layer (Day f3) ---
from app.db import get_connection
from app.embeddings import embed_query, embed_text, embed_texts
from app.chunks import ChunkRecord, batch_insert_chunks

# --- Chunker (Day f4) ---
from app.chunker import chunk_filing, split_into_sections, strip_html_to_text

# --- Graph scaffold (Day f5) ---
from app.graph import build_graph, ingest_node, index_node
from app.state import AgentState

__all__ = [
    # EDGAR
    "load_ticker_map",
    "fetch_submissions",
    "fetch_primary_document",
    # XBRL
    "fetch_companyfacts",
    "extract_gaap_facts",
    # DB
    "get_connection",
    "embed_text",
    "embed_texts",
    "embed_query",
    "ChunkRecord",
    "batch_insert_chunks",
    # Chunker
    "chunk_filing",
    "split_into_sections",
    "strip_html_to_text",
    # Graph
    "AgentState",
    "build_graph",
    "ingest_node",
    "index_node",
]
