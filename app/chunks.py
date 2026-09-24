"""Batch-insert chunks (+ their embeddings) into Postgres."""
from __future__ import annotations

from dataclasses import dataclass, field

import psycopg
from psycopg.types.json import Jsonb

from app.embeddings import embed_texts


@dataclass
class ChunkRecord:
    """One not-yet-embedded chunk, ready to be written to the chunks table."""

    doc_id: str
    section: str
    text: str
    chunk_type: str = "text"  # 'text' | 'table' -- see db/init/04_add_chunk_type_and_page_embeddings.sql
    metadata: dict = field(default_factory=dict)


def batch_insert_chunks(
    conn: psycopg.Connection, chunks: list[ChunkRecord], replace: bool = False
) -> int:
    """Embed and insert a batch of chunks in one round trip. Returns the
    number of rows actually inserted — which can be less than len(chunks)
    when one or more chunks are byte-for-byte identical (same doc_id,
    section, text) to a row already in the table, per the chunks table's
    content_hash unique index (db/init/03_add_chunks_content_hash.sql).
    That makes reprocessing an already-ingested filing a safe no-op per
    chunk instead of creating duplicate rows.

    replace=True first deletes every existing row for the doc_ids in this
    batch, in the same transaction as the insert. Use it when re-ingesting a
    filing after the chunker changed: the hash covers the section label and
    text, so re-chunked output never conflicts with the old rows and would
    otherwise sit next to them as duplicates. Because it is one transaction,
    a failed insert rolls the delete back too. An empty batch never deletes
    anything.

    Embedding the whole batch in a single embed_texts() call (rather than
    looping embed_text() per chunk) matters here: sentence-transformers
    batches its own forward passes internally, so one call over N texts is
    one batched inference pass instead of N separate ones — the difference
    that makes ingesting a filing's worth of chunks (hundreds) practical.
    Every chunk is still embedded even if it turns out to be a duplicate —
    there's no cheap way to know that before embedding without a separate
    round trip to check hashes first, and that's not worth the extra
    complexity at today's data volume.
    """
    if not chunks:
        return 0

    vectors = embed_texts([c.text for c in chunks])

    rows = [
        (chunk.doc_id, chunk.section, chunk.text, chunk.chunk_type, vector, Jsonb(chunk.metadata))
        for chunk, vector in zip(chunks, vectors)
    ]

    inserted = 0
    try:
        with conn.cursor() as cur:
            if replace:
                cur.execute(
                    "DELETE FROM chunks WHERE doc_id = ANY(%s)",
                    (sorted({c.doc_id for c in chunks}),),
                )
            cur.executemany(
                """
                INSERT INTO chunks (doc_id, section, text, chunk_type, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                rows,
                returning=True,
            )
            # executemany(..., returning=True) makes one result set per
            # statement rather than one combined result set — a conflicting
            # row's statement returns zero rows (not an error), so summing
            # each statement's fetchall() length gives the true inserted
            # count, not just len(rows).
            while True:
                inserted += len(cur.fetchall())
                if not cur.nextset():
                    break
        conn.commit()
    except Exception:
        # A failed statement leaves a non-autocommit connection in an aborted
        # transaction, where every later statement fails with "current
        # transaction is aborted". Callers share one connection across
        # tickers (scripts/build_corpus.py), so roll back before re-raising
        # or one bad insert takes down every ticker after it.
        conn.rollback()
        raise
    return inserted
