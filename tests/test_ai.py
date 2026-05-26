"""Tests for AI module helpers that don't require a live model."""

from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from photo_tagger import ai as ai_module
from photo_tagger.errors import ProviderError


class _DummyResponse:
    def __init__(self, status_code: int, payload: Any) -> None:  # noqa: ANN401
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:  # noqa: ANN401
        return self._payload

    @property
    def text(self) -> str:
        return repr(self._payload)


def test_validate_listing_url_rejects_missing_scheme() -> None:
    """A relative URL is caught up-front before httpx is invoked."""
    with pytest.raises(ProviderError):
        ai_module._validate_listing_url(  # noqa: SLF001
            "ftp:///models",
            event_prefix="lmstudio_model_listing",
        )


def test_validate_listing_url_rejects_missing_host() -> None:
    """A URL with no netloc never reaches the network."""
    with pytest.raises(ProviderError):
        ai_module._validate_listing_url(  # noqa: SLF001
            "http:///models",
            event_prefix="lmstudio_model_listing",
        )


def test_fetch_lmstudio_models_handles_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error is logged and exits the process."""

    def boom(url: str, *, headers: dict[str, str], timeout: float) -> Any:  # noqa: ANN401
        msg = "nope"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(ProviderError):
        ai_module._fetch_lmstudio_models("http://localhost:1234/v1/models", None)  # noqa: SLF001


def test_fetch_lmstudio_models_handles_non_ok_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-OK status code is fatal."""

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.SERVICE_UNAVAILABLE, {})

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ProviderError):
        ai_module._fetch_lmstudio_models("http://localhost:1234/v1/models", None)  # noqa: SLF001


def test_truncate_for_log_caps_long_bodies() -> None:
    """A response body well past the cap is shortened with an overflow marker."""
    body = "x" * 5_000
    out = ai_module._truncate_for_log(body)  # noqa: SLF001
    assert len(out) < len(body)
    assert out.startswith("x" * 100)
    assert "more chars" in out


def test_truncate_for_log_passes_short_bodies_through() -> None:
    """Short bodies are returned unchanged so the log stays useful for tiny errors."""
    body = "model not found"
    assert ai_module._truncate_for_log(body) == body  # noqa: SLF001


def test_fetch_lmstudio_models_handles_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 response with malformed JSON is also fatal."""

    class Broken(_DummyResponse):
        def json(self) -> Any:  # noqa: ANN401
            msg = "not json"
            raise ValueError(msg)

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> Broken:
        return Broken(HTTPStatus.OK, "not json")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ProviderError):
        ai_module._fetch_lmstudio_models("http://localhost:1234/v1/models", None)  # noqa: SLF001


def test_fetch_lmstudio_models_returns_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed listing returns just the id strings."""
    payload = {"data": [{"id": "alpha"}, {"id": "beta"}, {"name": "no-id"}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert ai_module._fetch_lmstudio_models(  # noqa: SLF001
        "http://localhost:1234/v1/models",
        api_key="secret",
    ) == ["alpha", "beta"]


def test_fetch_ollama_models_returns_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama's /api/tags returns a 'models' array of objects with a 'name' field."""
    payload = {"models": [{"name": "llava:34b"}, {"name": "qwen-vl"}, {"size": 1234}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert ai_module._fetch_ollama_models(  # noqa: SLF001
        "http://localhost:11434/api/tags",
        api_key=None,
    ) == ["llava:34b", "qwen-vl"]


def test_validate_ollama_model_strips_v1_suffix_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pydantic-ai-style /v1 base URL is rewritten to /api/tags before listing."""
    payload = {"models": [{"name": "vision-pro"}]}
    captured: list[str] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        captured.append(url)
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)
    ai_module.validate_ollama_model("http://localhost:11434/v1", "vision-pro", None)

    assert captured == ["http://localhost:11434/api/tags"]


def test_validate_ollama_model_exits_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ProviderError is raised when the requested model id is not in the local listing."""
    payload = {"models": [{"name": "other-model"}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ProviderError):
        ai_module.validate_ollama_model("http://localhost:11434", "missing", None)


# ---------------------------------------------------------------------------
# create_agent wiring
# ---------------------------------------------------------------------------


def test_create_agent_ollama_validates_and_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('ollama') calls validate_ollama_model, wires OllamaProvider, returns Agent."""
    payload = {"models": [{"name": "test-model"}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)

    agent = ai_module.create_agent(
        "ollama",
        "test-model",
        api_base_url="http://localhost:11434/v1",
        api_key=None,
        retries=2,
    )
    assert agent is not None


def test_create_agent_lmstudio_validates_and_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('lmstudio') calls validate_lmstudio_model, wires OpenAIProvider."""
    payload = {"data": [{"id": "test-model"}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)

    agent = ai_module.create_agent(
        "lmstudio",
        "test-model",
        api_base_url="http://localhost:1234/v1",
        api_key=None,
        retries=1,
    )
    assert agent is not None


def test_create_agent_uses_default_url_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When api_base_url is None, the provider's default URL is used."""
    payload = {"models": [{"name": "m"}]}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)

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
