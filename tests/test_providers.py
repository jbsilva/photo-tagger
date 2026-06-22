"""Tests for the provider registry: shared HTTP plumbing, parsers, and backends."""

import json
import typing
from http import HTTPStatus
from typing import Any

import httpx
import pytest

from photo_tagger import providers
from photo_tagger.errors import ProviderError
from photo_tagger.providers import PROVIDER_NAMES, ProviderName, get_backend


class _DummyResponse:
    """Minimal httpx-style response stub for listing tests."""

    def __init__(self, status_code: int, payload: Any) -> None:  # noqa: ANN401
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:  # noqa: ANN401
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


def _patch_httpx_get(
    monkeypatch: pytest.MonkeyPatch,
    response: _DummyResponse,
) -> list[dict[str, Any]]:
    """Record every httpx.get call and return *response* for each."""
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return response

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------


def test_provider_names_match_the_literal() -> None:
    """The runtime registry stays in lockstep with the ProviderName Literal."""
    assert set(PROVIDER_NAMES) == set(typing.get_args(ProviderName))


def test_get_backend_returns_each_registered_backend() -> None:
    """Every advertised name resolves to a backend that reports the same name."""
    for name in PROVIDER_NAMES:
        assert get_backend(name).name == name


def test_get_backend_rejects_unknown_name() -> None:
    """An unknown provider name raises ProviderError rather than KeyError."""
    with pytest.raises(ProviderError):
        get_backend("does-not-exist")


def test_openai_backend_requires_api_key_but_local_ones_do_not() -> None:
    """Only the hosted OpenAI backend flags itself as needing a key."""
    assert get_backend("openai").requires_api_key is True
    assert get_backend("ollama").requires_api_key is False
    assert get_backend("lmstudio").requires_api_key is False


# ---------------------------------------------------------------------------
# Shared HTTP plumbing
# ---------------------------------------------------------------------------


def test_validate_listing_url_rejects_missing_scheme() -> None:
    """A non-http(s) scheme is caught up-front before httpx is invoked."""
    with pytest.raises(ProviderError):
        providers._validate_listing_url("ftp:///models", event_prefix="x")  # noqa: SLF001


def test_validate_listing_url_rejects_missing_host() -> None:
    """A URL with no netloc never reaches the network."""
    with pytest.raises(ProviderError):
        providers._validate_listing_url("http:///models", event_prefix="x")  # noqa: SLF001


def test_truncate_for_log_caps_long_bodies() -> None:
    """A response body well past the cap is shortened with an overflow marker."""
    body = "x" * 5_000
    out = providers._truncate_for_log(body)  # noqa: SLF001
    assert len(out) < len(body)
    assert out.startswith("x" * 100)
    assert "more chars" in out


def test_truncate_for_log_passes_short_bodies_through() -> None:
    """Short bodies are returned unchanged so the log stays useful for tiny errors."""
    assert providers._truncate_for_log("model not found") == "model not found"  # noqa: SLF001


def test_fetch_listing_handles_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error is wrapped in ProviderError."""

    def boom(url: str, *, headers: dict[str, str], timeout: float) -> Any:  # noqa: ANN401
        msg = "nope"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(ProviderError):
        providers._fetch_listing("http://host/models", None, event_prefix="x")  # noqa: SLF001


def test_fetch_listing_handles_non_ok_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-OK status code is fatal."""
    _patch_httpx_get(monkeypatch, _DummyResponse(HTTPStatus.SERVICE_UNAVAILABLE, {}))
    with pytest.raises(ProviderError):
        providers._fetch_listing("http://host/models", None, event_prefix="x")  # noqa: SLF001


def test_fetch_listing_handles_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 response with malformed JSON is also fatal."""

    class Broken(_DummyResponse):
        def json(self) -> Any:  # noqa: ANN401
            msg = "not json"
            raise ValueError(msg)

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> Broken:
        return Broken(HTTPStatus.OK, "not json")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ProviderError):
        providers._fetch_listing("http://host/models", None, event_prefix="x")  # noqa: SLF001


def test_fetch_listing_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """When an api_key is given it travels as an Authorization header."""
    calls = _patch_httpx_get(monkeypatch, _DummyResponse(HTTPStatus.OK, {"data": []}))
    providers._fetch_listing("http://host/models", "secret", event_prefix="x")  # noqa: SLF001
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"


# ---------------------------------------------------------------------------
# Listing parsers
# ---------------------------------------------------------------------------


def test_openai_model_ids_returns_ids() -> None:
    """A well-formed OpenAI-style listing yields just the id strings."""
    listing = {"data": [{"id": "alpha"}, {"id": "beta"}, {"name": "no-id"}]}
    assert providers._openai_model_ids(listing) == ["alpha", "beta"]  # noqa: SLF001


def test_openai_model_ids_handles_non_list_data() -> None:
    """A listing whose 'data' field is not a list yields no models."""
    assert providers._openai_model_ids({"data": {"bad": "shape"}}) == []  # noqa: SLF001


def test_ollama_model_names_returns_names() -> None:
    """Ollama's /api/tags returns a 'models' array of objects with a 'name'."""
    listing = {"models": [{"name": "llava:34b"}, {"name": "qwen-vl"}, {"size": 1234}]}
    assert providers._ollama_model_names(listing) == ["llava:34b", "qwen-vl"]  # noqa: SLF001


def test_ollama_model_names_handles_non_list_models() -> None:
    """A listing whose 'models' field is not a list yields no models."""
    assert providers._ollama_model_names({"models": "not-a-list"}) == []  # noqa: SLF001


# ---------------------------------------------------------------------------
# Backend behavior (URL building + validation)
# ---------------------------------------------------------------------------


def test_lmstudio_backend_lists_models_from_v1_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LM Studio backend queries <base>/models and parses the OpenAI shape."""
    calls = _patch_httpx_get(monkeypatch, _DummyResponse(HTTPStatus.OK, {"data": [{"id": "m"}]}))
    models = get_backend("lmstudio").list_models("http://localhost:1234/v1", None)
    assert models == ["m"]
    assert calls[0]["url"].endswith("/models")
    assert calls[0]["timeout"] == pytest.approx(5.0)


def test_ollama_backend_strips_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama backend rewrites a /v1 base URL to /api/tags before listing."""
    calls = _patch_httpx_get(
        monkeypatch,
        _DummyResponse(HTTPStatus.OK, {"models": [{"name": "vision-pro"}]}),
    )
    get_backend("ollama").validate_model("http://localhost:11434/v1", "vision-pro", None)
    assert calls[0]["url"] == "http://localhost:11434/api/tags"


def test_validate_model_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ProviderError is raised when the requested model id is not in the listing."""
    _patch_httpx_get(monkeypatch, _DummyResponse(HTTPStatus.OK, {"data": [{"id": "other"}]}))
    with pytest.raises(ProviderError):
        get_backend("lmstudio").validate_model("http://localhost:1234/v1", "missing", None)


def test_build_provider_returns_a_provider() -> None:
    """Each backend can construct its concrete pydantic-ai provider object."""
    for name in PROVIDER_NAMES:
        provider = get_backend(name).build_provider("http://localhost/v1", "key")
        assert provider is not None
