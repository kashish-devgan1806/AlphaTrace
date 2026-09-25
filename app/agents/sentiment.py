"""Sentiment / Tone Agent: zero-shot classification over Q&A transcript
segments for hedging vs. confident language, compared against the prior
quarter's transcript.

`transcript`/`prior_transcript` are caller-supplied state inputs (see
app/state.py) -- no live transcript-sourcing path exists yet, so this node
takes the text directly rather than reading it from document_bundle. It
never touches Postgres or an LLM; its only external dependency is the
local zero-shot classifier in app.sentiment.

Never raises: parsing and classification are each individually wrapped, a
failure recorded into `errors` and returned as partial state. A failed
prior-quarter comparison doesn't discard an otherwise-successful current-
quarter result -- the two are isolated from each other, the same way
app/agents/analyst.py isolates search/rerank/generate failures from one
another.
"""
from __future__ import annotations

from app.sentiment import SentimentLabel, classify_segments
from app.state import AgentState
from app.transcript import QASegment, parse_qa_segments


def _summarize(labels: list[SentimentLabel]) -> dict:
    total = len(labels)
    hedging = sum(1 for label in labels if label.label == "hedging")
    return {
        "segment_count": total,
        "hedging_count": hedging,
        "confident_count": total - hedging,
        "hedging_ratio": hedging / total if total else 0.0,
    }


def _tone_shift(current_summary: dict, prior_summary: dict) -> dict:
    delta = current_summary["hedging_ratio"] - prior_summary["hedging_ratio"]
    if delta > 1e-9:
        direction = "increased"
    elif delta < -1e-9:
        direction = "decreased"
    else:
        direction = "unchanged"
    return {"hedging_ratio_delta": delta, "direction": direction}


def _segment_results(segments: list[QASegment], labels: list[SentimentLabel]) -> list[dict]:
    return [
        {
            "segment_id": segment.segment_id,
            "question": segment.question,
            "answer": segment.answer,
            "label": label.label,
            "confidence": label.confidence,
        }
        for segment, label in zip(segments, labels)
    ]


def sentiment_node(state: AgentState) -> dict:
    """Score `state["transcript"]`'s Q&A segments for hedging vs. confident
    language, and -- when `state["prior_transcript"]` is also present --
    diff the hedging ratio against the prior quarter."""
    transcript = state.get("transcript")
    if not transcript or not transcript.strip():
        return {"errors": ["sentiment: no transcript to score"]}

    try:
        segments = parse_qa_segments(transcript)
    except Exception as exc:
        return {"errors": [f"sentiment: failed to parse transcript: {exc}"]}

    if not segments:
        return {"errors": ["sentiment: no Q&A segments found in transcript"]}

    try:
        labels = classify_segments([segment.answer for segment in segments])
    except Exception as exc:
        return {"errors": [f"sentiment: classification failed: {exc}"]}

    current_summary = _summarize(labels)

    errors: list[str] = []
    prior_summary = None
    tone_shift = None
    prior_transcript = state.get("prior_transcript")
    if prior_transcript and prior_transcript.strip():
        try:
            prior_segments = parse_qa_segments(prior_transcript)
            if not prior_segments:
                errors.append("sentiment: no Q&A segments found in prior_transcript -- comparison skipped")
            else:
                prior_labels = classify_segments([segment.answer for segment in prior_segments])
                prior_summary = _summarize(prior_labels)
                tone_shift = _tone_shift(current_summary, prior_summary)
        except Exception as exc:
            errors.append(f"sentiment: prior-quarter comparison failed: {exc}")

    result = {
        "sentiment_result": {
            "segments": _segment_results(segments, labels),
            "current_summary": current_summary,
            "prior_summary": prior_summary,
            "tone_shift": tone_shift,
        }
    }
    if errors:
        result["errors"] = errors
    return result
