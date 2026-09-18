"""Thin Postgres connection helper shared by anything that touches the
chunks table. Nothing fancy — a single short-lived connection per caller,
not a pool, since nothing in the codebase yet holds one open across
requests (that's a FastAPI-integration concern for a later session)."""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def get_connection() -> psycopg.Connection:
    """Open a new connection with the pgvector type adapter registered.

    Without register_vector(), psycopg has no idea how to turn a Python
    list[float] into the `vector` column's wire format (or back) — it would
    round-trip through psycopg's default adapter and fail, not silently
    misbehave. Registering it here, once, means every caller downstream
    (batch_insert_chunks today, search() in a later session) gets a
    connection that already speaks `vector` correctly.
    """
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn
