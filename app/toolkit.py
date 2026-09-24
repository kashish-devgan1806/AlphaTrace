"""The shared toolkit surface.

Five pieces — the EDGAR client, the XBRL extractor, the Postgres/pgvector
layer, the section-aware chunker, and the LangGraph scaffold — live as
separate modules under app/ and scripts/. This module is the one place
that re-exports their stable entry points, so every agent imports
`from app.toolkit import ...` instead of reaching into script internals
that were never meant as a public surface (scripts/edgar_pull.py and
scripts/chunk_filing.py stay importable directly too — nothing here moves
their code, it only re-exports names).

Grouped by the piece each name belongs to:
"""
from __future__ import annotations

# --- EDGAR client ---
from scripts.chunk_filing import find_latest_filing
from scripts.edgar_pull import (
    fetch_document_bytes,
    fetch_filing_index,
    fetch_primary_document,
    fetch_submissions,
    find_exhibit_99,
    load_ticker_map,
)

# --- XBRL client ---
from scripts.edgar_pull import extract_gaap_facts, fetch_companyfacts

# --- DB layer ---
from app.db import get_connection
from app.embeddings import embed_image, embed_images, embed_query, embed_text, embed_texts
from app.chunks import ChunkRecord, batch_insert_chunks

# --- Retrieval ---
from app.search import SearchResult, search

# --- Reranker ---
from app.rerank import RerankResult, rerank

# --- LLM ---
from app.llm import generate

# --- Chunker ---
from app.chunker import chunk_filing, split_into_sections, strip_html_to_text

# --- Tables ---
from app.tables import TableRecord, extract_tables

# --- Visual (slide-deck pages) ---
from app.visual import PageImage, batch_insert_page_embeddings, rasterize_pdf

# --- Graph scaffold ---
from app.graph import analyst_node, build_graph, ingest_node, index_node
from app.state import AgentState

__all__ = [
    # EDGAR
    "load_ticker_map",
    "fetch_submissions",
    "fetch_primary_document",
    "find_latest_filing",
    "fetch_filing_index",
    "find_exhibit_99",
    "fetch_document_bytes",
    # XBRL
    "fetch_companyfacts",
    "extract_gaap_facts",
    # DB
    "get_connection",
    "embed_text",
    "embed_texts",
    "embed_query",
    "embed_image",
    "embed_images",
    "ChunkRecord",
    "batch_insert_chunks",
    # Retrieval
    "SearchResult",
    "search",
    # Reranker
    "RerankResult",
    "rerank",
    # LLM
    "generate",
    # Chunker
    "chunk_filing",
    "split_into_sections",
    "strip_html_to_text",
    # Tables
    "TableRecord",
    "extract_tables",
    # Visual
    "PageImage",
    "rasterize_pdf",
    "batch_insert_page_embeddings",
    # Graph
    "AgentState",
    "build_graph",
    "ingest_node",
    "index_node",
    "analyst_node",
]
