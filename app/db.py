"""Thin Postgres connection helper shared by anything that touches the
chunks table. Nothing fancy — a single short-lived connection per caller,
not a pool, since nothing in the codebase yet holds one open across
requests (that's a FastAPI-integration concern for a later session)."""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings

# Without a timeout a dead or firewalled database makes connect() hang for the
# OS default (minutes) instead of failing fast.
CONNECT_TIMEOUT_SECONDS = 10


def get_connection() -> psycopg.Connection:
    """Open a new connection with the pgvector type adapter registered.

    Without register_vector(), psycopg has no idea how to turn a Python
    list[float] into the `vector` column's wire format (or back) — it would
    round-trip through psycopg's default adapter and fail, not silently
    misbehave. Registering it here, once, means every caller downstream
    (batch_insert_chunks and search()) gets a
    connection that already speaks `vector` correctly.
    """
    conn = psycopg.connect(settings.database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    register_vector(conn)
    return conn
