"""Research Analyst Agent: retrieve + rerank from pgvector, then call the
LLM with a grounded-answer prompt to produce a cited answer.

Unlike ingest_node/index_node, this node can't stay DB-free -- its entire
job is retrieval, so it opens its own short-lived connection via
get_connection() (same one-connection-per-caller model app/db.py already
uses) and closes it before returning, whether or not retrieval succeeded.

Scoped to text + table chunks only (search()'s existing chunk_type filter,
app/search.py) -- no page_embeddings/CLIP visual retrieval here. The
roadmap schedules "visual query routing" as i5 (Phase 2), and no ticker's
corpus has a real PDF slide-deck exhibit yet to test a visual path
against; text + table is the defensible scope today. Between text and
table it retrieves unfiltered rather than routing by keyword heuristic --
the reranker (below) sorts relevance more reliably than a guess at what
"the question demands" would.

Never raises: a missing question, empty retrieval, a failed rerank, or a
failed LLM call are all recorded in `errors` and returned as partial state
rather than propagated.
"""
from __future__ import annotations

import json
import re

from app.db import get_connection
from app.llm import generate
from app.rerank import RerankResult, rerank
from app.search import search
from app.state import AgentState

# k for the candidate pool handed to the reranker. MAX_K (app/search.py) is
# 100; 20 is deliberately much smaller -- large enough that a relevant
# chunk missing from today's top-5 cosine ranking (the NVDA foundry/TSMC
# case documented in docs/build-log/session-07-eval_retrieval.py) still has
# a chance to be in the pool the cross-encoder re-scores.
CANDIDATE_POOL_SIZE = 20
# Chunks actually handed to the LLM as context, post-rerank.
TOP_K = 5

MARKER_RE = re.compile(r"\[\d+\]")

SYSTEM_INSTRUCTIONS = """You are AlphaTrace's Research Analyst agent. Answer the analyst's question \
using ONLY the numbered evidence chunks below -- never information from outside them, even if you \
believe you already know the answer. If the evidence does not contain enough information to answer, \
say so explicitly instead of guessing, extrapolating, or filling the gap with general knowledge.

Every factual claim in your answer must carry an inline citation marker, e.g. [1], referencing one \
of the evidence numbers below. Never invent a citation number that isn't listed below. A claim with \
no matching evidence chunk must not be stated as fact -- omit it or flag it as unsupported instead.

Respond with a single JSON object and nothing else -- no prose before or after it -- of this exact \
shape:
{"answer": "<answer text with inline [n] markers>", \
"citations": [{"marker": "[n]", "chunk_id": <the evidence chunk_id integer>, \
"quote": "<the exact span from that chunk's text that supports the claim>"}]}"""


def _evidence_dict(rr: RerankResult) -> dict:
    r = rr.result
    return {
        "chunk_id": r.id,
        "doc_id": r.doc_id,
        "section": r.section,
        "chunk_type": r.chunk_type,
        "text": r.text,
        "retrieval_score": r.score,
        "rerank_score": rr.rerank_score,
    }


def build_prompt(question: str, evidence: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (chunk_id={e['chunk_id']}, section={e['section']}, type={e['chunk_type']})\n{e['text']}"
        for i, e in enumerate(evidence, start=1)
    )
    return f"{SYSTEM_INSTRUCTIONS}\n\nEVIDENCE:\n{context}\n\nQUESTION: {question}"


def _parse_response(raw: str, evidence: list[dict]) -> tuple[str | None, list[dict], list[str]]:
    """Turn the LLM's raw JSON text into (draft_answer, citations, errors).
    Two adversarial cases are checked explicitly, not just hoped away by
    the prompt wording: a citation naming a chunk_id outside the retrieved
    pool (dropped, not trusted), and an inline [n] marker in the answer
    text with no resolved citation behind it (flagged, answer kept as-is)."""
    errors: list[str] = []
    evidence_by_id = {e["chunk_id"]: e for e in evidence}

    try:
        parsed = json.loads(raw)
        answer = parsed["answer"]
        raw_citations = parsed.get("citations", [])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(f"analyst: LLM response was not the expected JSON shape: {exc}")
        return (raw.strip() if raw and raw.strip() else None), [], errors

    citations: list[dict] = []
    for entry in raw_citations:
        chunk_id = entry.get("chunk_id")
        source = evidence_by_id.get(chunk_id)
        if source is None:
            errors.append(
                f"analyst: LLM cited chunk_id {chunk_id!r}, not in the retrieved evidence pool -- dropped"
            )
            continue
        citations.append(
            {
                "marker": entry.get("marker", ""),
                "chunk_id": chunk_id,
                "doc_id": source["doc_id"],
                "section": source["section"],
                "chunk_type": source["chunk_type"],
                "quote": entry.get("quote", ""),
            }
        )

    resolved_markers = {c["marker"] for c in citations}
    unresolved = sorted({m for m in MARKER_RE.findall(answer) if m not in resolved_markers})
    if unresolved:
        errors.append(f"analyst: answer references marker(s) with no resolved citation: {unresolved}")

    return answer, citations, errors


def analyst_node(state: AgentState) -> dict:
    """Retrieve the ticker's top candidate chunks for `state['question']`,
    rerank them, and ask the LLM for a grounded, cited answer."""
    ticker = state["ticker"].upper()
    question = state.get("question")
    if not question or not question.strip():
        return {"errors": ["analyst: no question to answer"]}

    conn = get_connection()
    try:
        candidates = search(conn, question, k=CANDIDATE_POOL_SIZE, ticker=ticker)
    except Exception as exc:
        return {"errors": [f"analyst: retrieval failed for {ticker}: {exc}"]}
    finally:
        conn.close()

    if not candidates:
        return {
            "retrieved_evidence": [],
            "draft_answer": None,
            "citations": [],
            "errors": [f"analyst: no chunks retrieved for {ticker} -- has this ticker been indexed?"],
        }

    try:
        reranked = rerank(question, candidates, top_k=TOP_K)
    except Exception as exc:
        return {
            "retrieved_evidence": [],
            "draft_answer": None,
            "citations": [],
            "errors": [f"analyst: reranking failed for {ticker}: {exc}"],
        }
    evidence = [_evidence_dict(rr) for rr in reranked]
    prompt = build_prompt(question, evidence)

    try:
        raw_response = generate(prompt, json_mode=True)
    except Exception as exc:
        return {
            "retrieved_evidence": evidence,
            "draft_answer": None,
            "citations": [],
            "errors": [f"analyst: LLM call failed: {exc}"],
        }

    draft_answer, citations, parse_errors = _parse_response(raw_response, evidence)

    result = {"retrieved_evidence": evidence, "draft_answer": draft_answer, "citations": citations}
    if parse_errors:
        result["errors"] = parse_errors
    return result
