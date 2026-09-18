"""Offline tests for app/embeddings.py — no real model download, no
network. A fake SentenceTransformer stands in for BAAI/bge-small-en-v1.5 so
these run in CI without pulling model weights."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from app import embeddings


class FakeTokenizer:
    """Token count == word count, so tests can control it precisely by
    picking how many words a fake input has."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


class FakeModel:
    def __init__(self, max_seq_length: int = 8) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = FakeTokenizer()
        self.encode_calls: list[dict] = []

    def encode(self, texts: list[str], normalize_embeddings: bool, convert_to_numpy: bool) -> np.ndarray:
        self.encode_calls.append(
            {"texts": texts, "normalize_embeddings": normalize_embeddings, "convert_to_numpy": convert_to_numpy}
        )
        # One fixed-width fake vector per input text, deterministic on length.
        return np.array([[float(len(t))] * 3 for t in texts])


@pytest.fixture(autouse=True)
def reset_model_singleton(monkeypatch):
    """embeddings._model is a module-level lazy singleton — clear it before
    and after every test so tests don't leak a fake model into each other
    (or into a real one from a prior live run)."""
    monkeypatch.setattr(embeddings, "_model", None)
    yield
    monkeypatch.setattr(embeddings, "_model", None)


def test_embed_texts_empty_list_short_circuits(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("SentenceTransformer should not be constructed for an empty batch")

    monkeypatch.setattr(embeddings, "SentenceTransformer", fail_if_called)

    assert embeddings.embed_texts([]) == []


def test_embed_texts_normalizes_and_returns_lists(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: fake)

    result = embeddings.embed_texts(["hello world", "a"])

    assert result == [[11.0, 11.0, 11.0], [1.0, 1.0, 1.0]]
    assert fake.encode_calls[0]["normalize_embeddings"] is True
    assert fake.encode_calls[0]["convert_to_numpy"] is True


def test_embed_text_returns_single_vector(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: fake)

    result = embeddings.embed_text("hello world")

    assert result == [11.0, 11.0, 11.0]


def test_embed_query_prepends_instruction(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: fake)

    embeddings.embed_query("what was the gross margin?")

    sent_text = fake.encode_calls[0]["texts"][0]
    assert sent_text.startswith(embeddings.QUERY_INSTRUCTION)
    assert sent_text.endswith("what was the gross margin?")


def test_model_is_constructed_once_and_reused(monkeypatch):
    build_calls = []

    def build(name):
        build_calls.append(name)
        return FakeModel()

    monkeypatch.setattr(embeddings, "SentenceTransformer", build)

    embeddings.embed_text("first call")
    embeddings.embed_text("second call")

    assert build_calls == [embeddings.MODEL_NAME]


def test_warns_when_input_exceeds_max_seq_length(monkeypatch, caplog):
    fake = FakeModel(max_seq_length=3)
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: fake)

    with caplog.at_level(logging.WARNING, logger="app.embeddings"):
        embeddings.embed_text("this input has five words")

    assert any("truncated" in record.message for record in caplog.records)


def test_no_warning_when_input_is_within_max_seq_length(monkeypatch, caplog):
    fake = FakeModel(max_seq_length=10)
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: fake)

    with caplog.at_level(logging.WARNING, logger="app.embeddings"):
        embeddings.embed_text("short input")

    assert caplog.records == []
