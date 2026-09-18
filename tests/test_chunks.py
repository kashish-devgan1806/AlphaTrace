"""Offline tests for app/chunks.py — a fake connection/cursor stands in for
Postgres, and embed_texts() is monkeypatched so no model or DB is touched."""
from __future__ import annotations

from psycopg.types.json import Jsonb

from app.chunks import ChunkRecord, batch_insert_chunks


class FakeCursor:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple[str, list[tuple]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def executemany(self, query: str, rows: list[tuple]) -> None:
        self.executemany_calls.append((query, rows))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.commit_calls = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_calls += 1


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
    query, rows = conn.cursor_obj.executemany_calls[0]
    assert "INSERT INTO chunks" in query
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
