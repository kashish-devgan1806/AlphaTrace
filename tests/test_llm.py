"""Offline tests for app/llm.py — no real Groq API call, no network. A fake
Groq client stands in so these run in CI without a GROQ_API_KEY."""
from __future__ import annotations

import pytest

from app import llm


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content: str = "a plain answer") -> None:
        self._content = content
        self.create_calls: list[dict] = []

    def create(self, **kwargs) -> FakeCompletion:
        self.create_calls.append(kwargs)
        return FakeCompletion(self._content)


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeGroq:
    def __init__(self, api_key: str, content: str = "a plain answer") -> None:
        self.api_key = api_key
        self.completions = FakeCompletions(content)
        self.chat = FakeChat(self.completions)


@pytest.fixture(autouse=True)
def reset_client_singleton(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    yield
    monkeypatch.setattr(llm, "_client", None)


def test_generate_raises_a_clear_error_when_api_key_unset(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.generate("hello")


def test_generate_returns_the_response_content(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-key")
    monkeypatch.setattr(llm, "Groq", lambda api_key: FakeGroq(api_key, content="the answer"))

    assert llm.generate("hello") == "the answer"


def test_generate_passes_model_and_temperature(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-key")
    monkeypatch.setattr(llm.settings, "groq_model", "llama-3.3-70b-versatile")
    fake = FakeGroq("fake-key")
    monkeypatch.setattr(llm, "Groq", lambda api_key: fake)

    llm.generate("hello", temperature=0.7)

    call = fake.completions.create_calls[0]
    assert call["model"] == "llama-3.3-70b-versatile"
    assert call["temperature"] == 0.7
    assert call["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_json_mode_sets_response_format(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-key")
    fake = FakeGroq("fake-key")
    monkeypatch.setattr(llm, "Groq", lambda api_key: fake)

    llm.generate("hello", json_mode=True)

    assert fake.completions.create_calls[0]["response_format"] == {"type": "json_object"}


def test_generate_without_json_mode_omits_response_format(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-key")
    fake = FakeGroq("fake-key")
    monkeypatch.setattr(llm, "Groq", lambda api_key: fake)

    llm.generate("hello")

    assert fake.completions.create_calls[0]["response_format"] is None


def test_client_is_constructed_once_and_reused(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-key")
    build_calls = []

    def build(api_key):
        build_calls.append(api_key)
        return FakeGroq(api_key)

    monkeypatch.setattr(llm, "Groq", build)

    llm.generate("first")
    llm.generate("second")

    assert build_calls == ["fake-key"]
