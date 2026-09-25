"""Zero-shot NLI classification over Q&A segment text -- hedging vs.
confident language, no task-specific fine-tuning.

Same singleton shape as app/rerank.py's CrossEncoder wrapper: a lazily
built, lock-guarded model singleton, loaded on first call so importing
this module never requires a model download.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

# Binary, matching the spec's own wording verbatim ("hedging vs. confident
# language") -- not a richer taxonomy design call.
CANDIDATE_LABELS = ["hedging", "confident"]

# The pipeline's default template ("This example is {}.") reads awkwardly
# for these two labels; a domain-fit hypothesis gives the NLI model a more
# meaningful entailment question to score against.
HYPOTHESIS_TEMPLATE = "The speaker's tone in this response is {}."

_classifier: Callable[..., Any] | None = None
_classifier_lock = threading.Lock()


def _get_classifier() -> Callable[..., Any]:
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
    return _classifier


@dataclass
class SentimentLabel:
    """One segment's classification result. `scores` keeps every candidate
    label's score (not just the winner) so a downstream consumer can judge
    how confident the win margin actually was."""

    label: str
    confidence: float
    scores: dict[str, float]


def classify_segments(texts: list[str]) -> list[SentimentLabel]:
    """Classify each text in `texts` as "hedging" or "confident". An empty
    list returns [] without loading the model -- cheap to call
    speculatively, mirrors rerank.rerank()'s empty-results short circuit."""
    if not texts:
        return []

    classifier = _get_classifier()
    raw_results = classifier(texts, candidate_labels=CANDIDATE_LABELS, hypothesis_template=HYPOTHESIS_TEMPLATE)
    if isinstance(raw_results, dict):
        raw_results = [raw_results]

    labels: list[SentimentLabel] = []
    for result in raw_results:
        scores = dict(zip(result["labels"], result["scores"]))
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        labels.append(SentimentLabel(label=top_label, confidence=top_score, scores=scores))
    return labels
