"""Top-k cosine-similarity search over the chunks table."""
from __future__ import annotations

from dataclasses import dataclass, field

import psycopg
from pgvector.psycopg.vector import Vector

from app.embeddings import embed_query

# Upper bound on k. Retrieval feeds a reranker and an LLM context window, so
# hundreds of chunks is never the intent; it also keeps hnsw.ef_search (max
# 1000) comfortably in range.
MAX_K = 100
# pgvector's default hnsw.ef_search; the per-query value is raised above it
# when k is larger.
DEFAULT_EF_SEARCH = 40


@dataclass
class SearchResult:
    """One retrieved chunk. `score` is cosine similarity (1 - cosine
    distance), so higher is more similar."""

    id: int
    doc_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


def search(
    conn: psycopg.Connection, query: str, k: int = 5, ticker: str | None = None
) -> list[SearchResult]:
    """Return the k chunks closest to `query` by cosine distance, best first.

    `ticker`, if given, restricts results to that company's chunks via
    metadata->>'ticker' (no documents table yet, so it lives in the JSONB).
    Caveat for the index: an HNSW scan fetches its nearest candidates first
    and applies the WHERE afterwards, so a selective filter can return fewer
    than k rows when the index path is chosen.

    The query is embedded with embed_query() (instruction-prefixed, per the
    model card) and compared against passage vectors with pgvector's `<=>`
    cosine-distance operator — the one matching the HNSW index's
    vector_cosine_ops class, so `ORDER BY embedding <=> ... LIMIT k` is the
    shape the planner can serve from the index.

    The vector is wrapped in Vector(): a bare list[float] bound as a query
    parameter (rather than into a column whose type psycopg already knows)
    fails with UndefinedFunction.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > MAX_K:
        raise ValueError(f"k must be <= {MAX_K}, got {k}")
    if ticker is not None:
        ticker = ticker.strip()
        if not ticker:
            raise ValueError("ticker must be a non-empty string when given")

    query_vector = Vector(embed_query(query))

    where = ""
    params: tuple = (query_vector, query_vector, k)
    if ticker is not None:
        where = "WHERE metadata->>'ticker' = %s"
        params = (query_vector, ticker.upper(), query_vector, k)

    # get_connection() is non-autocommit: a bare SELECT would leave the
    # connection "idle in transaction" until the caller closes it. The
    # transaction block ends it as soon as the query returns.
    with conn.transaction(), conn.cursor() as cur:
        # Per-query index settings, transaction-local (set_config(..., true)):
        # - hnsw.ef_search bounds how many candidates an index scan can return,
        #   so with the default 40 a k above 40 silently comes back short.
        # - hnsw.iterative_scan keeps scanning until LIMIT rows survive the
        #   WHERE, so a selective ticker filter can't return fewer than k rows
        #   (needs pgvector >= 0.8; strict_order keeps exact distance order).
        cur.execute(
            "SELECT set_config('hnsw.ef_search', %s, true), "
            "set_config('hnsw.iterative_scan', 'strict_order', true)",
            (str(max(k, DEFAULT_EF_SEARCH)),),
        )
        cur.execute(
            f"""
            SELECT id, doc_id, section, text, metadata, embedding <=> %s AS distance
            FROM chunks
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        SearchResult(
            id=row[0],
            doc_id=row[1],
            section=row[2],
            text=row[3],
            metadata=row[4],
            score=1.0 - row[5],
        )
        for row in rows
    ]
