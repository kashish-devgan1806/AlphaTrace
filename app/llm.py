"""Groq chat-completion client -- the project's first LLM call.

A thin wrapper, same shape as app/db.py's get_connection(): one lazily-
built client singleton, one function callers actually use. Kept this small
on purpose so app/agents/analyst.py's tests can monkeypatch `generate`
directly instead of mocking the Groq SDK.
"""
from __future__ import annotations

import threading

from groq import Groq

from app.config import settings

# Every other network call in this codebase sets an explicit timeout
# (app/db.py's CONNECT_TIMEOUT_SECONDS, scripts/edgar_pull.py's 15/30s
# per-endpoint timeouts) so a dead remote fails fast instead of hanging the
# caller indefinitely -- the Groq call gets the same treatment.
REQUEST_TIMEOUT_SECONDS = 30.0

_client: Groq | None = None
_client_lock = threading.Lock()


def _get_client() -> Groq:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not settings.groq_api_key:
                    raise RuntimeError(
                        "GROQ_API_KEY is not set -- add a real key to .env before calling the LLM "
                        "(get one at console.groq.com)."
                    )
                _client = Groq(api_key=settings.groq_api_key)
    return _client


def generate(prompt: str, *, temperature: float = 0.0, json_mode: bool = False) -> str:
    """One single-turn chat completion. temperature=0.0 by default: the
    grounded-answer prompt wants a deterministic answer pinned to the
    retrieved evidence, not varied phrasing across a Critic-triggered
    retry. json_mode=True asks Groq to constrain output to valid JSON --
    used by the grounded-answer prompt's structured {answer, citations}
    response."""
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else None,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content
