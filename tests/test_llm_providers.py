"""Tests for llm_providers — TDD D1."""
import pytest


# ── groq_client ──────────────────────────────────────────────────────────────

def test_groq_client_returns_instructor_client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-groq")
    from src.llm_providers import groq_client
    client = groq_client()
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")


def test_groq_client_explicit_key():
    from src.llm_providers import groq_client
    client = groq_client(api_key="explicit-key")
    assert hasattr(client, "chat")


def test_groq_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src.llm_providers import groq_client
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        groq_client()


# ── openrouter_client ────────────────────────────────────────────────────────

def test_openrouter_client_returns_instructor_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-or")
    from src.llm_providers import openrouter_client
    client = openrouter_client()
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")


def test_openrouter_client_explicit_key():
    from src.llm_providers import openrouter_client
    client = openrouter_client(api_key="explicit-key")
    assert hasattr(client, "chat")


def test_openrouter_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from src.llm_providers import openrouter_client
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter_client()
