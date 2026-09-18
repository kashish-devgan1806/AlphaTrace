"""Session 4 deliverable: embed_text() and batch_insert_chunks(), writing
into the `chunks` table Session 3's migration created.

Model choice fixes the vector column's dimension (see
db/init/02_create_chunks_table.sql), so it's decided here, once, as a
module-level constant instead of being left implicit in a function call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# pyrefly: ignore [missing-import]
import psycopg
# pyrefly: ignore [missing-import]
from pgvector.psycopg import register_vector
# pyrefly: ignore [missing-import]
from psycopg.types.json import Json

# BAAI/bge-small-en-v1.5: a retrieval-tuned sentence-transformer, small
# enough to run on CPU for this project's scope, that outputs 384-dim
# embeddings and is meant to be used with cosine similarity on normalized
# vectors (hence normalize_embeddings=True in embed_text below) — matching
# the vector_cosine_ops index Session 3 built.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model = None


def _get_model():
    """Lazily import and load the model.

    Importing sentence-transformers (and the torch it pulls in) at module
    load time would make every caller of batch_insert_chunks pay that cost
    even when they're just inserting pre-computed embeddings, and would make
    this module unimportable in any environment without those (heavy)
    dependencies installed — including this project's offline-safe test
    suite. Deferring both the import and the model download to first actual
    use avoids that.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # pyrefly: ignore [missing-import]

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_text(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of chunk texts, returning one 384-dim vector per input.

    bge-small-en-v1.5's max sequence length is 512 tokens. sentence-
    transformers truncates anything longer at the tokenizer level silently
    — no warning, no error, the excess text is simply never seen by the
    model. That's an acceptable gap for right now only because nothing
    upstream of this function yet guarantees a chunk stays under that
    budget; Session 5's section-aware chunker is what's supposed to make
    this a non-issue in practice, not this function.
    """
    model = _get_model()
    embeddings = model.encode(list(texts), normalize_embeddings=True)
    return embeddings.tolist()


@dataclass
class Chunk:
    """One row's worth of pre-embedding data for the `chunks` table."""

    doc_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def batch_insert_chunks(conn: "psycopg.Connection", chunks: Sequence[Chunk]) -> int:
    """Embed `chunks` and insert them into `chunks` in one transaction.

    Returns the number of rows inserted. An empty `chunks` is a no-op that
    never touches the connection — callers looping over documents shouldn't
    have to special-case "this one had nothing to insert".
    """
    if not chunks:
        return 0

    register_vector(conn)
    embeddings = embed_text([c.text for c in chunks])

    rows = [
        (c.doc_id, c.section, c.text, emb, Json(c.metadata))
        for c, emb in zip(chunks, embeddings)
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
