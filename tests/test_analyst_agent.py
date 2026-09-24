"""Offline tests for app/agents/analyst.py. No live network/model/DB:
get_connection, search, rerank, and generate are monkeypatched at the
module level where analyst_node actually looks them up now that it lives
in its own module -- the same pattern tests/test_graph.py's _patch_edgar
uses for app.agents.ingestion, and the pattern this repo's convention
calls for the project's first LLM-call test (no existing test mocks an
LLM before this one)."""
from __future__ import annotations

import json

import pytest

import app.agents.analyst as analyst_module
from app.agents.analyst import analyst_node, build_prompt
from app.rerank import RerankResult
from app.search import SearchResult


class FakeConn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _candidate(id_=1, text="Supply chain risk text.", chunk_type="text", score=0.8) -> SearchResult:
    return SearchResult(id=id_, doc_id="doc-1", section="Item 1A", text=text, chunk_type=chunk_type, score=score)


def _patch(monkeypatch, conn=None, candidates=None, reranked=None, llm_response=None, llm_error=None, search_error=None):
    conn = conn if conn is not None else FakeConn()
    monkeypatch.setattr(analyst_module, "get_connection", lambda: conn)

    def fake_search(c, query, k, ticker):
        if search_error is not None:
            raise search_error
        return candidates if candidates is not None else [_candidate()]

    monkeypatch.setattr(analyst_module, "search", fake_search)

    def fake_rerank(query, results, top_k):
        if reranked is not None:
            return reranked
        return [RerankResult(result=r, rerank_score=1.0 - i * 0.1) for i, r in enumerate(results)][:top_k]

    monkeypatch.setattr(analyst_module, "rerank", fake_rerank)

    def fake_generate(prompt, json_mode=False):
        if llm_error is not None:
            raise llm_error
        if llm_response is not None:
            return llm_response
        return json.dumps(
            {
                "answer": "The company faces supply chain risk [1].",
                "citations": [{"marker": "[1]", "chunk_id": 1, "quote": "Supply chain risk text."}],
            }
        )

    monkeypatch.setattr(analyst_module, "generate", fake_generate)
    return conn


def test_missing_question_returns_error_without_touching_db(monkeypatch):
    def fail_if_called():
        raise AssertionError("get_connection should not be called with no question")

    monkeypatch.setattr(analyst_module, "get_connection", fail_if_called)

    result = analyst_node({"ticker": "AAPL"})

    assert result == {"errors": ["analyst: no question to answer"]}


@pytest.mark.parametrize("blank_question", ["", "   ", "\n"])
def test_blank_question_returns_error(monkeypatch, blank_question):
    def fail_if_called():
        raise AssertionError("get_connection should not be called with a blank question")

    monkeypatch.setattr(analyst_module, "get_connection", fail_if_called)

    result = analyst_node({"ticker": "AAPL", "question": blank_question})

    assert result == {"errors": ["analyst: no question to answer"]}


def test_happy_path_returns_evidence_answer_and_citations(monkeypatch):
    conn = _patch(monkeypatch)

    result = analyst_node({"ticker": "AAPL", "question": "What supply chain risks exist?"})

    assert "errors" not in result
    assert result["draft_answer"] == "The company faces supply chain risk [1]."
    assert result["citations"] == [
        {"marker": "[1]", "chunk_id": 1, "doc_id": "doc-1", "section": "Item 1A", "chunk_type": "text", "quote": "Supply chain risk text."}
    ]
    assert result["retrieved_evidence"] == [
        {
            "chunk_id": 1,
            "doc_id": "doc-1",
            "section": "Item 1A",
            "chunk_type": "text",
            "text": "Supply chain risk text.",
            "retrieval_score": 0.8,
            "rerank_score": 1.0,
        }
    ]
    assert conn.closed is True


def test_connection_closed_even_when_search_fails(monkeypatch):
    conn = _patch(monkeypatch, search_error=RuntimeError("connection refused"))

    result = analyst_node({"ticker": "AAPL", "question": "anything"})

    assert result == {"errors": ["analyst: retrieval failed for AAPL: connection refused"]}
    assert conn.closed is True


def test_no_candidates_retrieved_records_error(monkeypatch):
    _patch(monkeypatch, candidates=[])

    result = analyst_node({"ticker": "AAPL", "question": "anything"})

    assert result["retrieved_evidence"] == []
    assert result["draft_answer"] is None
    assert result["citations"] == []
    assert "no chunks retrieved for AAPL" in result["errors"][0]


def test_llm_call_failure_still_returns_retrieved_evidence(monkeypatch):
    _patch(monkeypatch, llm_error=RuntimeError("groq timeout"))

    result = analyst_node({"ticker": "AAPL", "question": "anything"})

    assert result["retrieved_evidence"] != []
    assert result["draft_answer"] is None
    assert "analyst: LLM call failed: groq timeout" in result["errors"][0]


def test_ticker_is_uppercased_and_passed_to_search(monkeypatch):
    seen = {}

    def fake_search(c, query, k, ticker):
        seen["ticker"] = ticker
        seen["k"] = k
        return [_candidate()]

    monkeypatch.setattr(analyst_module, "get_connection", lambda: FakeConn())
    monkeypatch.setattr(analyst_module, "search", fake_search)
    monkeypatch.setattr(analyst_module, "rerank", lambda query, results, top_k: [RerankResult(result=results[0], rerank_score=1.0)])
    monkeypatch.setattr(analyst_module, "generate", lambda prompt, json_mode=False: json.dumps({"answer": "ok", "citations": []}))

    analyst_node({"ticker": "nvda", "question": "risk?"})

    assert seen["ticker"] == "NVDA"
    assert seen["k"] == analyst_module.CANDIDATE_POOL_SIZE


# --- Adversarial cases (a3d2's own review bar: predict a way the model
# could ignore an instruction, then test that failure) ---


def test_hallucinated_citation_chunk_id_is_dropped_and_flagged(monkeypatch):
    """The prompt says "never invent a citation number that isn't listed"
    -- nothing stops the model from doing it anyway, so a chunk_id outside
    the retrieved pool must not be trusted as a real citation."""
    bad_response = json.dumps(
        {
            "answer": "A risk exists [1].",
            "citations": [{"marker": "[1]", "chunk_id": 999, "quote": "made up"}],
        }
    )
    _patch(monkeypatch, llm_response=bad_response)

    result = analyst_node({"ticker": "AAPL", "question": "anything"})

    assert result["citations"] == []
    assert any("cited chunk_id 999" in e for e in result["errors"])


def test_unresolved_marker_in_answer_is_flagged(monkeypatch):
    """The prompt says "every factual claim ... must carry an inline
    citation marker" -- nothing stops the model from writing a marker with
    no backing citation entry, so that mismatch has to be caught, not
    assumed away."""
    bad_response = json.dumps({"answer": "A risk exists [2].", "citations": []})
    _patch(monkeypatch, llm_response=bad_response)

    result = analyst_node({"ticker": "AAPL", "question": "anything"})

    assert result["draft_answer"] == "A risk exists [2]."
    assert any("no resolved citation" in e for e in result["errors"])


def test_non_json_response_falls_back_to_raw_text(monkeypatch):
    """The prompt says "respond with a single JSON object ... and nothing
    else" -- nothing stops the model from wrapping it in prose anyway, so
    a response that doesn't parse must not crash the node."""
    _patch(monkeypatch, llm_response="Sure! Here's my answer: supply chain risk is high.")

    result = analyst_node({"ticker": "AAPL", "question": "anything"})

    assert result["draft_answer"] == "Sure! Here's my answer: supply chain risk is high."
    assert result["citations"] == []
    assert any("not the expected JSON shape" in e for e in result["errors"])


def test_build_prompt_includes_question_and_numbered_evidence():
    evidence = [
        {"chunk_id": 1, "doc_id": "doc-1", "section": "Item 1A", "chunk_type": "text", "text": "risk text"},
    ]

    prompt = build_prompt("What are the risks?", evidence)

    assert "What are the risks?" in prompt
    assert "[1]" in prompt
    assert "chunk_id=1" in prompt
    assert "risk text" in prompt
