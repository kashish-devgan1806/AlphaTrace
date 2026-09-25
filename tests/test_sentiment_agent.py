"""Offline tests for app/agents/sentiment.py. No real model/network:
parse_qa_segments and classify_segments are monkeypatched at the module
level where sentiment_node actually looks them up, the same pattern
tests/test_analyst_agent.py uses for app.agents.analyst."""
from __future__ import annotations

import pytest

import app.agents.sentiment as sentiment_module
from app.agents.sentiment import sentiment_node
from app.sentiment import SentimentLabel
from app.transcript import QASegment


def _segment(id_=1, question="Q?", answer="A.") -> QASegment:
    return QASegment(segment_id=id_, question=question, answer=answer)


def _label(label="confident", confidence=0.9) -> SentimentLabel:
    return SentimentLabel(label=label, confidence=confidence, scores={"confident": confidence, "hedging": 1 - confidence})


def _patch(
    monkeypatch,
    segments=None,
    labels=None,
    prior_segments=None,
    prior_labels=None,
    parse_error=None,
    classify_error=None,
    prior_parse_error=None,
    prior_classify_error=None,
):
    calls = {"parse": [], "classify": []}

    def fake_parse(text):
        calls["parse"].append(text)
        if text == "PRIOR" and prior_parse_error is not None:
            raise prior_parse_error
        if text == "PRIOR":
            return prior_segments if prior_segments is not None else [_segment()]
        if parse_error is not None:
            raise parse_error
        return segments if segments is not None else [_segment()]

    def fake_classify(texts):
        calls["classify"].append(texts)
        # Distinguish the prior-quarter call by its call order (parse is
        # always called before classify for the same transcript).
        is_prior = len(calls["classify"]) > 1
        if is_prior and prior_classify_error is not None:
            raise prior_classify_error
        if is_prior:
            return prior_labels if prior_labels is not None else [_label()] * len(texts)
        if classify_error is not None:
            raise classify_error
        return labels if labels is not None else [_label()] * len(texts)

    monkeypatch.setattr(sentiment_module, "parse_qa_segments", fake_parse)
    monkeypatch.setattr(sentiment_module, "classify_segments", fake_classify)
    return calls


def test_missing_transcript_returns_error_without_parsing(monkeypatch):
    def fail_if_called(text):
        raise AssertionError("parse_qa_segments should not be called with no transcript")

    monkeypatch.setattr(sentiment_module, "parse_qa_segments", fail_if_called)

    result = sentiment_node({"ticker": "AAPL"})

    assert result == {"errors": ["sentiment: no transcript to score"]}


@pytest.mark.parametrize("blank_transcript", ["", "   ", "\n"])
def test_blank_transcript_returns_error(monkeypatch, blank_transcript):
    def fail_if_called(text):
        raise AssertionError("parse_qa_segments should not be called with a blank transcript")

    monkeypatch.setattr(sentiment_module, "parse_qa_segments", fail_if_called)

    result = sentiment_node({"ticker": "AAPL", "transcript": blank_transcript})

    assert result == {"errors": ["sentiment: no transcript to score"]}


def test_no_segments_found_records_error(monkeypatch):
    _patch(monkeypatch, segments=[])

    result = sentiment_node({"transcript": "text with no Q/A markers"})

    assert result == {"errors": ["sentiment: no Q&A segments found in transcript"]}


def test_parse_failure_records_error(monkeypatch):
    _patch(monkeypatch, parse_error=ValueError("malformed transcript"))

    result = sentiment_node({"transcript": "anything"})

    assert result == {"errors": ["sentiment: failed to parse transcript: malformed transcript"]}


def test_classify_failure_records_error(monkeypatch):
    _patch(monkeypatch, classify_error=RuntimeError("model unavailable"))

    result = sentiment_node({"transcript": "anything"})

    assert result == {"errors": ["sentiment: classification failed: model unavailable"]}


def test_happy_path_without_prior_transcript(monkeypatch):
    _patch(monkeypatch, segments=[_segment(1, "Q1", "A1")], labels=[_label("hedging", 0.6)])

    result = sentiment_node({"transcript": "current quarter transcript"})

    assert "errors" not in result
    assert result["sentiment_result"] == {
        "segments": [{"segment_id": 1, "question": "Q1", "answer": "A1", "label": "hedging", "confidence": 0.6}],
        "current_summary": {"segment_count": 1, "hedging_count": 1, "confident_count": 0, "hedging_ratio": 1.0},
        "prior_summary": None,
        "tone_shift": None,
    }


def test_happy_path_with_prior_transcript_hedging_increased(monkeypatch):
    _patch(
        monkeypatch,
        segments=[_segment(1), _segment(2)],
        labels=[_label("hedging"), _label("hedging")],
        prior_segments=[_segment(1)],
        prior_labels=[_label("confident")],
    )

    result = sentiment_node({"transcript": "current", "prior_transcript": "PRIOR"})

    assert result["sentiment_result"]["current_summary"]["hedging_ratio"] == 1.0
    assert result["sentiment_result"]["prior_summary"]["hedging_ratio"] == 0.0
    assert result["sentiment_result"]["tone_shift"] == {"hedging_ratio_delta": 1.0, "direction": "increased"}


def test_happy_path_with_prior_transcript_hedging_decreased(monkeypatch):
    _patch(
        monkeypatch,
        segments=[_segment(1)],
        labels=[_label("confident")],
        prior_segments=[_segment(1)],
        prior_labels=[_label("hedging")],
    )

    result = sentiment_node({"transcript": "current", "prior_transcript": "PRIOR"})

    assert result["sentiment_result"]["tone_shift"] == {"hedging_ratio_delta": -1.0, "direction": "decreased"}


def test_happy_path_with_prior_transcript_hedging_unchanged(monkeypatch):
    _patch(
        monkeypatch,
        segments=[_segment(1)],
        labels=[_label("hedging")],
        prior_segments=[_segment(1)],
        prior_labels=[_label("hedging")],
    )

    result = sentiment_node({"transcript": "current", "prior_transcript": "PRIOR"})

    assert result["sentiment_result"]["tone_shift"] == {"hedging_ratio_delta": 0.0, "direction": "unchanged"}


def test_blank_prior_transcript_is_treated_as_absent(monkeypatch):
    calls = _patch(monkeypatch, segments=[_segment()], labels=[_label()])

    result = sentiment_node({"transcript": "current", "prior_transcript": "   "})

    assert result["sentiment_result"]["prior_summary"] is None
    assert result["sentiment_result"]["tone_shift"] is None
    assert calls["parse"] == ["current"]  # PRIOR never parsed


def test_prior_transcript_with_no_segments_keeps_current_result(monkeypatch):
    _patch(monkeypatch, segments=[_segment()], labels=[_label()], prior_segments=[])

    result = sentiment_node({"transcript": "current", "prior_transcript": "PRIOR"})

    assert result["sentiment_result"]["current_summary"]["segment_count"] == 1
    assert result["sentiment_result"]["prior_summary"] is None
    assert result["sentiment_result"]["tone_shift"] is None
    assert result["errors"] == ["sentiment: no Q&A segments found in prior_transcript -- comparison skipped"]


def test_prior_transcript_parse_failure_keeps_current_result(monkeypatch):
    _patch(
        monkeypatch,
        segments=[_segment()],
        labels=[_label()],
        prior_parse_error=ValueError("bad prior transcript"),
    )

    result = sentiment_node({"transcript": "current", "prior_transcript": "PRIOR"})

    assert result["sentiment_result"]["current_summary"]["segment_count"] == 1
    assert result["sentiment_result"]["prior_summary"] is None
    assert "sentiment: prior-quarter comparison failed: bad prior transcript" in result["errors"]


def test_prior_transcript_classify_failure_keeps_current_result(monkeypatch):
    _patch(
        monkeypatch,
        segments=[_segment()],
        labels=[_label()],
        prior_segments=[_segment()],
        prior_classify_error=RuntimeError("prior model failure"),
    )

    result = sentiment_node({"transcript": "current", "prior_transcript": "PRIOR"})

    assert result["sentiment_result"]["current_summary"]["segment_count"] == 1
    assert result["sentiment_result"]["prior_summary"] is None
    assert "sentiment: prior-quarter comparison failed: prior model failure" in result["errors"]
