"""Offline tests for app/rerank.py — no real model download, no network. A
fake CrossEncoder stands in for cross-encoder/ms-marco-MiniLM-L-6-v2 so
these run in CI without pulling model weights."""
from __future__ import annotations

import logging

import pytest

from app import rerank as rerank_module
from app.rerank import RerankResult, rerank
from app.search import SearchResult


class FakeTokenizer:
    """Token count == word count in the concatenated (query, passage) pair,
    so tests can control it precisely."""

    def encode(self, query: str, passage: str) -> list[int]:
        return list(range(len(query.split()) + len(passage.split())))


class FakeCrossEncoder:
    def __init__(self, max_seq_length: int = 20, scores: list[float] | None = None) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = FakeTokenizer()
        self._scores = scores
        self.predict_calls: list[list[tuple]] = []

    def predict(self, pairs: list[tuple]) -> list[float]:
        self.predict_calls.append(pairs)
        if self._scores is not None:
            return self._scores
        # Deterministic fallback: longer passage scores higher.
        return [float(len(p[1])) for p in pairs]


@pytest.fixture(autouse=True)
def reset_model_singleton(monkeypatch):
    monkeypatch.setattr(rerank_module, "_model", None)
    yield
    monkeypatch.setattr(rerank_module, "_model", None)


def _result(id_, text, chunk_type="text", score=0.5) -> SearchResult:
    return SearchResult(id=id_, doc_id=f"doc-{id_}", section="Item 1A", text=text, chunk_type=chunk_type, score=score)


def test_rerank_empty_results_short_circuits(monkeypatch):
    def fail_if_called(name):
        raise AssertionError("CrossEncoder should not be constructed for an empty candidate list")

    monkeypatch.setattr(rerank_module, "CrossEncoder", fail_if_called)

    assert rerank("query", []) == []


def test_rerank_orders_by_score_best_first(monkeypatch):
    fake = FakeCrossEncoder(scores=[0.1, 0.9, 0.5])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)
    results = [_result(1, "low"), _result(2, "high"), _result(3, "mid")]

    ranked = rerank("query", results, top_k=3)

    assert [rr.result.id for rr in ranked] == [2, 3, 1]
    assert ranked == [
        RerankResult(result=results[1], rerank_score=0.9),
        RerankResult(result=results[2], rerank_score=0.5),
        RerankResult(result=results[0], rerank_score=0.1),
    ]


def test_rerank_truncates_to_top_k(monkeypatch):
    fake = FakeCrossEncoder(scores=[0.1, 0.9, 0.5])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)
    results = [_result(1, "a"), _result(2, "b"), _result(3, "c")]

    ranked = rerank("query", results, top_k=1)

    assert len(ranked) == 1
    assert ranked[0].result.id == 2


def test_rerank_keeps_retrieval_score_separate_from_rerank_score(monkeypatch):
    fake = FakeCrossEncoder(scores=[0.7])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)
    result = _result(1, "text", score=0.33)

    ranked = rerank("query", [result], top_k=1)

    assert ranked[0].result.score == 0.33
    assert ranked[0].rerank_score == 0.7


def test_rerank_scores_query_passage_pairs(monkeypatch):
    fake = FakeCrossEncoder(scores=[0.5, 0.5])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)
    results = [_result(1, "first passage"), _result(2, "second passage")]

    rerank("what was the gross margin?", results, top_k=2)

    assert fake.predict_calls == [
        [("what was the gross margin?", "first passage"), ("what was the gross margin?", "second passage")]
    ]


def test_model_is_constructed_once_and_reused(monkeypatch):
    build_calls = []

    def build(name):
        build_calls.append(name)
        return FakeCrossEncoder(scores=[0.5])

    monkeypatch.setattr(rerank_module, "CrossEncoder", build)
    result = [_result(1, "text")]

    rerank("first", result, top_k=1)
    rerank("second", result, top_k=1)

    assert build_calls == [rerank_module.MODEL_NAME]


def test_rejects_non_positive_top_k(monkeypatch):
    fake = FakeCrossEncoder(scores=[0.5])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)

    with pytest.raises(ValueError, match="top_k"):
        rerank("query", [_result(1, "text")], top_k=0)


def test_warns_when_pair_exceeds_max_seq_length(monkeypatch, caplog):
    fake = FakeCrossEncoder(max_seq_length=3, scores=[0.5])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)

    with caplog.at_level(logging.WARNING, logger="app.rerank"):
        rerank("a long query here", [_result(1, "and a long passage too")], top_k=1)

    assert any("truncated" in record.message for record in caplog.records)


def test_no_warning_when_pair_is_within_max_seq_length(monkeypatch, caplog):
    fake = FakeCrossEncoder(max_seq_length=20, scores=[0.5])
    monkeypatch.setattr(rerank_module, "CrossEncoder", lambda name: fake)

    with caplog.at_level(logging.WARNING, logger="app.rerank"):
        rerank("short query", [_result(1, "short passage")], top_k=1)

    assert caplog.records == []
