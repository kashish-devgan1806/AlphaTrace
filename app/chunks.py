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
    metadata: dict = field(default_factory=dict)


def batch_insert_chunks(conn: psycopg.Connection, chunks: list[ChunkRecord]) -> int:
    """Embed and insert a batch of chunks in one round trip. Returns the
    number of rows actually inserted — which can be less than len(chunks)
    when one or more chunks are byte-for-byte identical (same doc_id,
    section, text) to a row already in the table, per the chunks table's
    content_hash unique index (db/init/03_add_chunks_content_hash.sql).
    That makes reprocessing an already-ingested filing a safe no-op per
    chunk instead of creating duplicate rows.

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
        (chunk.doc_id, chunk.section, chunk.text, vector, Jsonb(chunk.metadata))
        for chunk, vector in zip(chunks, vectors)
    ]

    inserted = 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (doc_id, section, text, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
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
    return inserted
