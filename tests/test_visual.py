"""Offline tests for app/visual.py. rasterize_pdf uses real PyMuPDF against
a small PDF built in-process (no network, no external file); embed_images
is monkeypatched for the insert tests so no visual model is loaded."""
from __future__ import annotations

import pymupdf
import pytest
from psycopg.types.json import Jsonb

from app.visual import PageImage, batch_insert_page_embeddings, rasterize_pdf


def _make_pdf_bytes(num_pages: int, width: float = 200, height: float = 100) -> bytes:
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page(width=width, height=height)
    data = doc.tobytes()
    doc.close()
    return data


def test_rasterize_pdf_one_page_per_page():
    images = rasterize_pdf(_make_pdf_bytes(3))

    assert len(images) == 3
    assert all(image.mode == "RGB" for image in images)


def test_rasterize_pdf_scales_with_dpi():
    default_images = rasterize_pdf(_make_pdf_bytes(1))
    hi_res_images = rasterize_pdf(_make_pdf_bytes(1), dpi=300)

    assert hi_res_images[0].size[0] > default_images[0].size[0]


class FakeCursor:
    def __init__(self, insert_flags: list[bool] | None = None) -> None:
        self.executemany_calls: list[tuple[str, list[tuple], bool]] = []
        self._insert_flags = insert_flags
        self._pos = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

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


def test_batch_insert_page_embeddings_empty_list_is_a_noop(monkeypatch):
    import app.visual as visual_module

    monkeypatch.setattr(
        visual_module, "embed_images", lambda images: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    conn = FakeConnection()

    assert batch_insert_page_embeddings(conn, []) == 0
    assert conn.cursor_obj.executemany_calls == []


def test_batch_insert_page_embeddings_embeds_once_and_inserts_all_rows(monkeypatch):
    import app.visual as visual_module

    calls = []

    def fake_embed_images(images):
        calls.append(list(images))
        return [[float(i)] * 4 for i in range(len(images))]

    monkeypatch.setattr(visual_module, "embed_images", fake_embed_images)
    conn = FakeConnection()
    pages = [
        PageImage(doc_id="deck-1", page_number=1, image="page-1", metadata={"ticker": "AAPL"}),
        PageImage(doc_id="deck-1", page_number=2, image="page-2"),
    ]

    inserted = batch_insert_page_embeddings(conn, pages)

    assert inserted == 2
    assert calls == [["page-1", "page-2"]]

    query, rows, returning = conn.cursor_obj.executemany_calls[0]
    assert "INSERT INTO page_embeddings" in query
    assert "ON CONFLICT" in query
    assert returning is True
    assert rows[0][0] == "deck-1"
    assert rows[0][1] == 1
    assert rows[0][2] == [0.0, 0.0, 0.0, 0.0]
    assert isinstance(rows[0][3], Jsonb)
    assert rows[0][3].obj == {"ticker": "AAPL"}
    assert conn.commit_calls == 1


def test_batch_insert_page_embeddings_skips_rows_that_conflict(monkeypatch):
    import app.visual as visual_module

    monkeypatch.setattr(visual_module, "embed_images", lambda images: [[0.0]] * len(images))
    conn = FakeConnection(insert_flags=[True, False])
    pages = [
        PageImage(doc_id="deck-1", page_number=1, image="p1"),
        PageImage(doc_id="deck-1", page_number=1, image="p1-again"),
    ]

    inserted = batch_insert_page_embeddings(conn, pages)

    assert inserted == 1


def test_batch_insert_page_embeddings_rolls_back_and_reraises_on_failure(monkeypatch):
    import app.visual as visual_module

    monkeypatch.setattr(visual_module, "embed_images", lambda images: [[0.0]] * len(images))
    conn = FakeConnection()

    def boom(query, rows, returning=False):
        raise RuntimeError("insert failed")

    conn.cursor_obj.executemany = boom

    with pytest.raises(RuntimeError, match="insert failed"):
        batch_insert_page_embeddings(conn, [PageImage(doc_id="d", page_number=1, image="p")])

    assert conn.rollback_calls == 1
    assert conn.commit_calls == 0
