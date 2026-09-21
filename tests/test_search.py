"""Offline tests for app/search.py — a fake connection/cursor stands in for
Postgres, and embed_query() is monkeypatched so no model or DB is touched."""
from __future__ import annotations

import pytest
from pgvector.psycopg.vector import Vector

import app.search as search_module
from app.search import SearchResult, search


class FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.execute_calls: list[tuple[str, tuple]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def execute(self, query: str, params: tuple) -> None:
        self.execute_calls.append((query, params))

    def fetchall(self) -> list[tuple]:
        return self._rows


class FakeTransaction:
    def __enter__(self) -> "FakeTransaction":
        return self

    def __exit__(self, *exc_info) -> None:
        return None


class FakeConnection:
    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.cursor_obj = FakeCursor(rows or [])
        self.transaction_blocks = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def transaction(self) -> "FakeTransaction":
        self.transaction_blocks += 1
        return FakeTransaction()


def _patch_embed_query(monkeypatch, vector=None):
    calls: list[str] = []
    vec = vector if vector is not None else [0.1, 0.2, 0.3]

    def fake_embed_query(text: str) -> list[float]:
        calls.append(text)
        return vec

    monkeypatch.setattr(search_module, "embed_query", fake_embed_query)
    return calls


def test_search_issues_cosine_order_by_limit_query(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "supply chain risk", k=3)

    query, params = conn.cursor_obj.execute_calls[-1]
    assert "<=>" in query
    assert "ORDER BY embedding <=>" in query
    assert "LIMIT" in query
    assert "FROM chunks" in query
    assert params[2] == 3


def test_search_wraps_query_vector_and_embeds_once(monkeypatch):
    calls = _patch_embed_query(monkeypatch, vector=[0.5, 0.5])
    conn = FakeConnection()

    search(conn, "gross margin")

    assert calls == ["gross margin"]
    _, params = conn.cursor_obj.execute_calls[-1]
    # Bare list[float] would fail with UndefinedFunction at query time.
    assert isinstance(params[0], Vector)
    assert isinstance(params[1], Vector)


def test_search_default_k_is_five(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "anything")

    assert conn.cursor_obj.execute_calls[-1][1][2] == 5


def test_search_maps_rows_to_results_with_similarity_score(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection(
        rows=[
            (7, "0001045810-25-000023", "Item 1A", "supply text", {"ticker": "NVDA"}, 0.25),
            (9, "0000320193-25-000079", "Item 7", "md&a text", {"ticker": "AAPL"}, 0.5),
        ]
    )

    results = search(conn, "supply chain", k=2)

    assert results == [
        SearchResult(
            id=7,
            doc_id="0001045810-25-000023",
            section="Item 1A",
            text="supply text",
            metadata={"ticker": "NVDA"},
            score=0.75,
        ),
        SearchResult(
            id=9,
            doc_id="0000320193-25-000079",
            section="Item 7",
            text="md&a text",
            metadata={"ticker": "AAPL"},
            score=0.5,
        ),
    ]


def test_search_returns_empty_list_when_no_rows(monkeypatch):
    _patch_embed_query(monkeypatch)

    assert search(FakeConnection(rows=[]), "nothing matches") == []


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
def test_search_rejects_blank_query(monkeypatch, bad_query):
    def fail_if_called(text):
        raise AssertionError("embed_query should not be called for a blank query")

    monkeypatch.setattr(search_module, "embed_query", fail_if_called)
    conn = FakeConnection()

    with pytest.raises(ValueError, match="query"):
        search(conn, bad_query)
    assert conn.cursor_obj.execute_calls == []


@pytest.mark.parametrize("bad_k", [0, -1])
def test_search_rejects_non_positive_k(monkeypatch, bad_k):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    with pytest.raises(ValueError, match="k"):
        search(conn, "risk factors", k=bad_k)
    assert conn.cursor_obj.execute_calls == []


def test_search_ticker_filter_adds_where_clause_and_uppercases(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "supply chain", k=4, ticker="nvda")

    query, params = conn.cursor_obj.execute_calls[-1]
    assert "WHERE metadata->>'ticker' = %s" in query
    assert query.index("WHERE") < query.index("ORDER BY")
    assert isinstance(params[0], Vector)
    assert params[1] == "NVDA"
    assert isinstance(params[2], Vector)
    assert params[3] == 4


def test_search_without_ticker_has_no_where_clause(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "supply chain")

    query, _ = conn.cursor_obj.execute_calls[-1]
    assert "WHERE" not in query


def test_search_runs_inside_a_transaction_block(monkeypatch):
    """The connection is non-autocommit, so the SELECT must be wrapped in a
    transaction block or it leaves the connection idle-in-transaction."""
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "risk factors")

    assert conn.transaction_blocks == 1


def test_search_sets_ef_search_to_at_least_k_and_enables_iterative_scan(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "risk factors", k=75)

    settings_sql, settings_params = conn.cursor_obj.execute_calls[0]
    assert "set_config('hnsw.ef_search'" in settings_sql
    assert "set_config('hnsw.iterative_scan', 'strict_order', true)" in settings_sql
    assert settings_params == ("75",)


def test_search_keeps_default_ef_search_for_small_k(monkeypatch):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    search(conn, "risk factors", k=5)

    assert conn.cursor_obj.execute_calls[0][1] == ("40",)


def test_search_rejects_k_above_the_maximum(monkeypatch):
    from app.search import MAX_K

    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    with pytest.raises(ValueError, match="k must be <="):
        search(conn, "risk factors", k=MAX_K + 1)
    assert conn.cursor_obj.execute_calls == []


@pytest.mark.parametrize("bad_ticker", ["", "   "])
def test_search_rejects_blank_ticker(monkeypatch, bad_ticker):
    _patch_embed_query(monkeypatch)
    conn = FakeConnection()

    with pytest.raises(ValueError, match="ticker"):
        search(conn, "risk factors", ticker=bad_ticker)
    assert conn.cursor_obj.execute_calls == []

