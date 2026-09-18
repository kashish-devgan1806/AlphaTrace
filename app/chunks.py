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
    number of rows inserted.

    Embedding the whole batch in a single embed_texts() call (rather than
    looping embed_text() per chunk) matters here: sentence-transformers
    batches its own forward passes internally, so one call over N texts is
    one batched inference pass instead of N separate ones — the difference
    that makes ingesting a filing's worth of chunks (hundreds) practical.
    """
    if not chunks:
        return 0

    vectors = embed_texts([c.text for c in chunks])

    rows = [
        (chunk.doc_id, chunk.section, chunk.text, vector, Jsonb(chunk.metadata))
        for chunk, vector in zip(chunks, vectors)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (doc_id, section, text, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    return len(rows)
