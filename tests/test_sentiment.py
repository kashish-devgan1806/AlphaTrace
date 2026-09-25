"""Offline tests for app/sentiment.py -- no real model download, no
network. A fake zero-shot pipeline callable stands in for
facebook/bart-large-mnli so these run in CI without pulling model weights,
mirroring tests/test_rerank.py's FakeCrossEncoder pattern."""
from __future__ import annotations

import pytest

from app import sentiment as sentiment_module
from app.sentiment import CANDIDATE_LABELS, HYPOTHESIS_TEMPLATE, SentimentLabel, classify_segments


class FakePipeline:
    """Mimics transformers' zero-shot-classification pipeline: given a
    list of texts, returns one result dict per text, each with `labels`/
    `scores` already sorted best-first -- the real pipeline's own
    contract."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results
        self.calls: list[dict] = []

    def __call__(self, texts, candidate_labels, hypothesis_template):
        self.calls.append(
            {"texts": texts, "candidate_labels": candidate_labels, "hypothesis_template": hypothesis_template}
        )
        if self._results is not None:
            return self._results
        return [
            {"labels": ["hedging", "confident"], "scores": [0.7, 0.3]}
            if "hedg" in t
            else {"labels": ["confident", "hedging"], "scores": [0.8, 0.2]}
            for t in texts
        ]


@pytest.fixture(autouse=True)
def reset_classifier_singleton(monkeypatch):
    monkeypatch.setattr(sentiment_module, "_classifier", None)
    yield
    monkeypatch.setattr(sentiment_module, "_classifier", None)


def test_classify_segments_empty_list_short_circuits(monkeypatch):
    def fail_if_called(name, model):
        raise AssertionError("pipeline() should not be constructed for an empty text list")

    monkeypatch.setattr(sentiment_module, "pipeline", fail_if_called)

    assert classify_segments([]) == []


def test_classify_segments_builds_labels_from_top_score(monkeypatch):
    fake = FakePipeline(
        results=[
            {"labels": ["hedging", "confident"], "scores": [0.65, 0.35]},
            {"labels": ["confident", "hedging"], "scores": [0.9, 0.1]},
        ]
    )
    monkeypatch.setattr(sentiment_module, "pipeline", lambda task, model: fake)

    results = classify_segments(["text one", "text two"])

    assert results == [
        SentimentLabel(label="hedging", confidence=0.65, scores={"hedging": 0.65, "confident": 0.35}),
        SentimentLabel(label="confident", confidence=0.9, scores={"confident": 0.9, "hedging": 0.1}),
    ]


def test_classify_segments_passes_candidate_labels_and_template(monkeypatch):
    fake = FakePipeline(results=[{"labels": ["hedging", "confident"], "scores": [0.5, 0.5]}])
    monkeypatch.setattr(sentiment_module, "pipeline", lambda task, model: fake)

    classify_segments(["some text"])

    assert fake.calls == [
        {"texts": ["some text"], "candidate_labels": CANDIDATE_LABELS, "hypothesis_template": HYPOTHESIS_TEMPLATE}
    ]


def test_classifier_is_constructed_once_and_reused(monkeypatch):
    build_calls = []

    def build(task, model):
        build_calls.append((task, model))
        return FakePipeline()

    monkeypatch.setattr(sentiment_module, "pipeline", build)

    classify_segments(["first"])
    classify_segments(["second"])

    assert build_calls == [("zero-shot-classification", sentiment_module.MODEL_NAME)]


def test_classify_segments_handles_single_dict_result(monkeypatch):
    """Some pipeline versions return a bare dict (not a list) for a
    single-item batch -- classify_segments must not assume a list."""
    fake_result = {"labels": ["confident", "hedging"], "scores": [0.6, 0.4]}
    monkeypatch.setattr(sentiment_module, "pipeline", lambda task, model: (lambda *a, **k: fake_result))

    results = classify_segments(["only one text"])

    assert results == [SentimentLabel(label="confident", confidence=0.6, scores={"confident": 0.6, "hedging": 0.4})]
