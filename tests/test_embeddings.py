"""Offline-safe tests for app/embeddings.py — no live DB, no model download.

The DB connection and the embedding model are both faked: batch_insert_chunks
is exercised against a stand-in psycopg connection/cursor, and embed_text is
exercised against a stand-in model swapped in for _get_model(). Both fakes
only need to speak the small surface this module actually calls.
"""
from __future__ import annotations

from app import embeddings


class _FakeArray:
    """Stands in for the numpy array model.encode() returns."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class _FakeModel:
    def __init__(self) -> None:
        self.encode_calls: list[tuple] = []

    def encode(self, texts, normalize_embeddings=False):
        self.encode_calls.append((list(texts), normalize_embeddings))
        # One short deterministic vector per input text.
        return _FakeArray([[float(len(t)), 0.0, 0.0] for t in texts])


class _FakeCursor:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, query, rows):
        self.executemany_calls.append((query, rows))


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_embed_text_normalizes_and_returns_one_vector_per_input(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: fake_model)

    result = embeddings.embed_text(["hello", "a longer chunk of text"])

    assert result == [[5.0, 0.0, 0.0], [22.0, 0.0, 0.0]]
    texts, normalize = fake_model.encode_calls[0]
    assert texts == ["hello", "a longer chunk of text"]
    assert normalize is True


def test_batch_insert_chunks_embeds_and_inserts_all_rows(monkeypatch):
    fake_model = _FakeModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: fake_model)
    monkeypatch.setattr(embeddings, "register_vector", lambda conn: None)

    conn = _FakeConnection()
    chunks = [
        embeddings.Chunk(doc_id="doc-1", section="Item 1A", text="risk factor text"),
        embeddings.Chunk(
            doc_id="doc-1",
            section="Item 7",
            text="md&a text",
            metadata={"page": 14},
        ),
    ]

    inserted = embeddings.batch_insert_chunks(conn, chunks)

    assert inserted == 2
    assert conn.committed is True
    [(query, rows)] = conn.cursor_obj.executemany_calls
    assert "INSERT INTO chunks" in query
    assert len(rows) == 2
    doc_id, section, text, embedding, metadata = rows[0]
    assert (doc_id, section, text) == ("doc-1", "Item 1A", "risk factor text")
    assert embedding == [16.0, 0.0, 0.0]  # len("risk factor text")


def test_batch_insert_chunks_empty_list_is_a_noop(monkeypatch):
    conn = _FakeConnection()

    inserted = embeddings.batch_insert_chunks(conn, [])

    assert inserted == 0
    assert conn.cursor_obj.executemany_calls == []
    assert conn.committed is False


def test_chunk_metadata_defaults_to_empty_dict():
    chunk = embeddings.Chunk(doc_id="doc-1", section="Item 1A", text="text")

    assert chunk.metadata == {}
