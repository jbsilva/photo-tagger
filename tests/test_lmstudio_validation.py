"""Tests for LM Studio model validation helper."""

import json
from http import HTTPStatus
from typing import Any

import pytest

import photo_tagger.main as m


class _DummyResponse:
    """Minimal httpx-style response stub for validation tests."""

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
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return response

    monkeypatch.setattr(m.httpx, "get", fake_get)  # type: ignore[attr-defined]
    return calls


def test_validate_lmstudio_model_passes_when_list_contains_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper returns quietly when the requested model identifier is present."""
    payload = {"data": [{"id": "vision-pro"}]}
    response = _DummyResponse(HTTPStatus.OK, payload)
    calls = _patch_httpx_get(monkeypatch, response)

    m._validate_lmstudio_model("http://localhost:1234/v1", "vision-pro", None)  # noqa: SLF001

    assert calls, "Expected httpx.get to be invoked"
    recorded = calls[0]
    assert recorded["url"].endswith("/models")
    assert recorded["timeout"] == pytest.approx(5.0)
    assert recorded["headers"].get("Accept") == "application/json"


def test_validate_lmstudio_model_exits_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """SystemExit is raised when the requested model identifier is absent."""
    payload = {"data": [{"id": "other-model"}]}
    response = _DummyResponse(HTTPStatus.OK, payload)
    _patch_httpx_get(monkeypatch, response)

    with pytest.raises(SystemExit):
        m._validate_lmstudio_model("http://localhost:1234/v1", "vision-pro", None)  # noqa: SLF001
