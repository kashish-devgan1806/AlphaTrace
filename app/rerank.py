"""Cross-encoder reranking over search()'s candidate pool.

Why a second stage: search() scores query and passage independently (a
bi-encoder) -- fast enough to rank the whole `chunks` table, but blind to
how the two actually interact. A cross-encoder scores each (query, passage)
pair jointly in one forward pass, which is far more accurate but too slow
to run over the full corpus -- so it only ever sees search()'s already-
narrowed candidate pool (k=20 by convention; app/agents/analyst.py's
CANDIDATE_POOL_SIZE), not the whole table.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.search import SearchResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(MODEL_NAME)
    return _model


@dataclass
class RerankResult:
    """One reranked candidate. `rerank_score` sits alongside, not over,
    `result.score` (the retrieval-stage cosine similarity) -- a
    cross-encoder's score isn't a cosine similarity and isn't comparable
    across queries, so overwriting the retrieval score would silently
    discard it."""

    result: SearchResult
    rerank_score: float


def _warn_on_truncation(model: CrossEncoder, query: str, results: list[SearchResult]) -> None:
    """The chunker targets ~480 text tokens; a query plus this model's own
    [CLS]/[SEP]/[SEP] special tokens can still push a (query, passage) pair
    past its 512-token max_seq_length. Truncation happens silently
    (truncation=True by default) -- warn instead of trusting every chunk to
    already fit."""
    max_len = model.max_seq_length
    tokenizer = model.tokenizer
    for r in results:
        token_count = len(tokenizer.encode(query, r.text))
        if token_count > max_len:
            logger.warning(
                "rerank: (query, chunk_id=%s) pair has %d tokens, exceeds "
                "max_seq_length=%d - it will be silently truncated.",
                r.id,
                token_count,
                max_len,
            )


def rerank(query: str, results: list[SearchResult], top_k: int = 5) -> list[RerankResult]:
    """Score every (query, result.text) pair with the cross-encoder and
    return the top_k, best first. An empty `results` returns [] without
    loading the model -- cheap to call speculatively."""
    if not results:
        return []
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    model = _get_model()
    _warn_on_truncation(model, query, results)
    scores = model.predict([(query, r.text) for r in results])

    ranked = sorted(
        (RerankResult(result=r, rerank_score=float(s)) for r, s in zip(results, scores)),
        key=lambda rr: rr.rerank_score,
        reverse=True,
    )
    return ranked[:top_k]
