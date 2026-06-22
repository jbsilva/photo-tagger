"""Tests for AI agent wiring that don't require a live model."""

import dataclasses
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from photo_tagger import ai as ai_module
from photo_tagger.errors import ProviderError
from photo_tagger.providers import get_backend


class _DummyResponse:
    def __init__(self, status_code: int, payload: Any) -> None:  # noqa: ANN401
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:  # noqa: ANN401
        return self._payload

    @property
    def text(self) -> str:
        return repr(self._payload)


def _patch_listing(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    """Make every model-listing request return *payload* with a 200."""

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)


# ---------------------------------------------------------------------------
# create_agent wiring
# ---------------------------------------------------------------------------


def test_create_agent_ollama_validates_and_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('ollama') validates against /api/tags and returns an Agent."""
    _patch_listing(monkeypatch, {"models": [{"name": "test-model"}]})
    agent = ai_module.create_agent(
        "ollama",
        "test-model",
        api_base_url="http://localhost:11434/v1",
        api_key=None,
        retries=2,
    )
    assert agent is not None


def test_create_agent_lmstudio_validates_and_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('lmstudio') validates against /v1/models and wires OpenAIProvider."""
    _patch_listing(monkeypatch, {"data": [{"id": "test-model"}]})
    agent = ai_module.create_agent(
        "lmstudio",
        "test-model",
        api_base_url="http://localhost:1234/v1",
        api_key=None,
        retries=1,
    )
    assert agent is not None


def test_create_agent_openai_validates_with_supplied_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('openai') accepts an explicit key and validates the model id."""
    _patch_listing(monkeypatch, {"data": [{"id": "gpt-4o-mini"}]})
    agent = ai_module.create_agent(
        "openai",
        "gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        api_key="sk-test",
        retries=1,
    )
    assert agent is not None


def test_create_agent_openai_without_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hosted OpenAI backend refuses to run without a key, before any network call."""

    def explode(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        msg = "httpx.get must not be called when the key is missing"
        raise AssertionError(msg)

    monkeypatch.setattr(httpx, "get", explode)
    # The default key is captured from the environment at import time, so build a
    # keyless clone (frozen dataclasses copy via dataclasses.replace) and route the
    # lookup to it regardless of what OPENAI_API_KEY happens to be on this machine.
    keyless = dataclasses.replace(get_backend("openai"), default_api_key=None)
    monkeypatch.setattr(ai_module, "get_backend", lambda _name: keyless)
    with pytest.raises(ProviderError):
        ai_module.create_agent(
            "openai",
            "gpt-4o-mini",
            api_base_url=None,
            api_key=None,
            retries=1,
        )


def test_create_agent_uses_default_url_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When api_base_url is None, the backend's default URL is used."""
    _patch_listing(monkeypatch, {"models": [{"name": "m"}]})
    agent = ai_module.create_agent(
        "ollama",
        "m",
        api_base_url=None,
        api_key=None,
        retries=0,
    )
    assert agent is not None


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


def test_extract_usage_returns_zeros_for_none() -> None:
    """None usage object returns (0, 0, 0)."""
    assert ai_module._extract_usage(None) == (0, 0, 0)  # noqa: SLF001


def test_extract_usage_reads_attributes() -> None:
    """Usage attributes are converted to ints."""
    usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
    assert ai_module._extract_usage(usage) == (10, 5, 15)  # noqa: SLF001


def test_extract_usage_handles_none_attributes() -> None:
    """None attribute values are coerced to 0."""
    usage = MagicMock(input_tokens=None, output_tokens=None, total_tokens=None)
    assert ai_module._extract_usage(usage) == (0, 0, 0)  # noqa: SLF001
