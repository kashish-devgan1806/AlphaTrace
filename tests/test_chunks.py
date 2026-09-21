"""Offline tests for app/chunks.py — a fake connection/cursor stands in for
Postgres, and embed_texts() is monkeypatched so no model or DB is touched."""
from __future__ import annotations

import pytest
from psycopg.types.json import Jsonb

from app.chunks import ChunkRecord, batch_insert_chunks


class FakeCursor:
    """`insert_flags`, if given, is a list of bools aligned with the rows
    passed to executemany() — False stands in for a row that Postgres'
    real ON CONFLICT (content_hash) DO NOTHING would have skipped as a
    duplicate. Defaults to "every row is a fresh insert" when omitted."""

    def __init__(self, insert_flags: list[bool] | None = None) -> None:
        self.executemany_calls: list[tuple[str, list[tuple], bool]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self._insert_flags = insert_flags
        self._pos = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def execute(self, query: str, params: tuple = ()) -> None:
        self.execute_calls.append((query, params))

    def executemany(self, query: str, rows: list[tuple], returning: bool = False) -> None:
        self.executemany_calls.append((query, rows, returning))
        if self._insert_flags is None:
            self._insert_flags = [True] * len(rows)
        self._pos = 0

    def fetchall(self) -> list[tuple]:
        return [(1,)] if self._insert_flags[self._pos] else []

    def nextset(self) -> bool:
        self._pos += 1
        return self._pos < len(self._insert_flags)


class FakeConnection:
    def __init__(self, insert_flags: list[bool] | None = None) -> None:
        self.cursor_obj = FakeCursor(insert_flags)
        self.commit_calls = 0
        self.rollback_calls = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


def test_batch_insert_chunks_empty_list_is_a_noop(monkeypatch):
    import app.chunks as chunks_module

    def fail_if_called(texts):
        raise AssertionError("embed_texts should not be called for an empty batch")

    monkeypatch.setattr(chunks_module, "embed_texts", fail_if_called)
    conn = FakeConnection()

    inserted = batch_insert_chunks(conn, [])

    assert inserted == 0
    assert conn.commit_calls == 0
    assert conn.cursor_obj.executemany_calls == []


def test_batch_insert_chunks_embeds_once_and_inserts_all_rows(monkeypatch):
    import app.chunks as chunks_module

    calls = []

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(i)] * 3 for i in range(len(texts))]

    monkeypatch.setattr(chunks_module, "embed_texts", fake_embed_texts)

    conn = FakeConnection()
    chunk_records = [
        ChunkRecord(doc_id="0000320193-25-000079", section="Item 1A Risk Factors", text="risk text one"),
        ChunkRecord(
            doc_id="0000320193-25-000079",
            section="MD&A",
            text="md&a text two",
            metadata={"fiscal_year": 2025},
        ),
    ]

    inserted = batch_insert_chunks(conn, chunk_records)

    assert inserted == 2
    # Embedded as a single batch call, not one call per chunk.
    assert calls == [["risk text one", "md&a text two"]]

    assert len(conn.cursor_obj.executemany_calls) == 1
    query, rows, returning = conn.cursor_obj.executemany_calls[0]
    assert "INSERT INTO chunks" in query
    assert "ON CONFLICT" in query
    assert returning is True
    assert len(rows) == 2

    row0 = rows[0]
    assert row0[0] == "0000320193-25-000079"
    assert row0[1] == "Item 1A Risk Factors"
    assert row0[2] == "risk text one"
    assert row0[3] == [0.0, 0.0, 0.0]
    assert isinstance(row0[4], Jsonb)
    assert row0[4].obj == {}

    row1 = rows[1]
    assert row1[3] == [1.0, 1.0, 1.0]
    assert row1[4].obj == {"fiscal_year": 2025}

    assert conn.commit_calls == 1


def test_batch_insert_chunks_skips_rows_that_conflict_on_content_hash(monkeypatch):
    """Reprocessing a filing that's already in the table (e.g. running
    scripts/build_corpus.py twice) must not create duplicate rows — the
    content_hash unique index (db/init/03_add_chunks_content_hash.sql) is
    what Postgres itself enforces; here that's simulated by telling the
    fake cursor which of the two rows would conflict."""
    import app.chunks as chunks_module

    monkeypatch.setattr(chunks_module, "embed_texts", lambda texts: [[0.0]] * len(texts))

    # First row is a fresh chunk; second is byte-for-byte identical to a
    # row already in the table, so its INSERT statement returns zero rows.
    conn = FakeConnection(insert_flags=[True, False])
    chunk_records = [
        ChunkRecord(doc_id="doc-1", section="Item 1A", text="new chunk"),
        ChunkRecord(doc_id="doc-1", section="Item 1A", text="already-inserted chunk"),
    ]

    inserted = batch_insert_chunks(conn, chunk_records)

    assert inserted == 1  # not 2 — the duplicate wasn't counted
    assert conn.commit_calls == 1


def test_batch_insert_chunks_rolls_back_and_reraises_when_the_insert_fails(monkeypatch):
    """A failed INSERT leaves a non-autocommit connection in an aborted
    transaction; without a rollback every later statement on the same
    (shared) connection would fail too."""
    import app.chunks as chunks_module

    monkeypatch.setattr(chunks_module, "embed_texts", lambda texts: [[0.0]] * len(texts))

    conn = FakeConnection()

    def boom(query, rows, returning=False):
        raise RuntimeError("insert failed")

    conn.cursor_obj.executemany = boom

    with pytest.raises(RuntimeError, match="insert failed"):
        batch_insert_chunks(conn, [ChunkRecord(doc_id="d", section="s", text="t")])

    assert conn.rollback_calls == 1
    assert conn.commit_calls == 0


def test_batch_insert_chunks_replace_deletes_the_filings_old_rows_first(monkeypatch):
    import app.chunks as chunks_module

    monkeypatch.setattr(chunks_module, "embed_texts", lambda texts: [[0.0]] * len(texts))
    conn = FakeConnection()
    chunk_records = [
        ChunkRecord(doc_id="doc-b", section="s", text="one"),
        ChunkRecord(doc_id="doc-a", section="s", text="two"),
        ChunkRecord(doc_id="doc-a", section="s", text="three"),
    ]

    batch_insert_chunks(conn, chunk_records, replace=True)

    (delete_sql, delete_params), = conn.cursor_obj.execute_calls
    assert delete_sql.startswith("DELETE FROM chunks WHERE doc_id")
    assert delete_params == (["doc-a", "doc-b"],)
    assert conn.commit_calls == 1  # delete + insert commit together


def test_batch_insert_chunks_default_never_deletes(monkeypatch):
    import app.chunks as chunks_module

    monkeypatch.setattr(chunks_module, "embed_texts", lambda texts: [[0.0]] * len(texts))
    conn = FakeConnection()

    batch_insert_chunks(conn, [ChunkRecord(doc_id="d", section="s", text="t")])

    assert conn.cursor_obj.execute_calls == []


def test_batch_insert_chunks_replace_with_empty_batch_deletes_nothing():
    conn = FakeConnection()

    assert batch_insert_chunks(conn, [], replace=True) == 0
    assert conn.cursor_obj.execute_calls == []
