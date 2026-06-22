"""Tests for the environment diagnostics behind `photo-tagger doctor`."""

import dataclasses
import io
import shutil
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from rich.console import Console

from photo_tagger import diagnostics
from photo_tagger.diagnostics import (
    CheckResult,
    check_exiftool,
    check_provider,
    render_report,
    run_checks,
)
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
    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx, "get", fake_get)


# ---------------------------------------------------------------------------
# check_exiftool
# ---------------------------------------------------------------------------


def test_check_exiftool_reports_path_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A binary on PATH yields ok=True and the resolved path as detail."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/exiftool")
    result = check_exiftool()
    assert result.ok is True
    assert result.detail == "/usr/bin/exiftool"


def test_check_exiftool_reports_failure_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing binary yields ok=False with an install hint."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    result = check_exiftool()
    assert result.ok is False
    assert "not found" in result.detail


# ---------------------------------------------------------------------------
# check_provider
# ---------------------------------------------------------------------------


def test_check_provider_ok_when_model_served(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reachable provider that lists the model passes."""
    _patch_listing(monkeypatch, {"data": [{"id": "vision-pro"}]})
    result = check_provider("lmstudio", "vision-pro", api_base_url=None, api_key=None)
    assert result.ok is True
    assert "available at" in result.detail


def test_check_provider_fails_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reachable provider that does not list the model fails."""
    _patch_listing(monkeypatch, {"data": [{"id": "other"}]})
    result = check_provider("lmstudio", "vision-pro", api_base_url=None, api_key=None)
    assert result.ok is False
    assert "not served" in result.detail


def test_check_provider_fails_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error is reported, not raised."""

    def boom(url: str, *, headers: dict[str, str], timeout: float) -> Any:  # noqa: ANN401
        msg = "refused"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(httpx, "get", boom)
    result = check_provider("ollama", "m", api_base_url="http://localhost:11434", api_key=None)
    assert result.ok is False
    assert "unreachable" in result.detail


def test_check_provider_flags_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hosted OpenAI backend fails fast (no network) when no key is configured."""

    def explode(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        msg = "network must not be touched"
        raise AssertionError(msg)

    monkeypatch.setattr(httpx, "get", explode)
    keyless = dataclasses.replace(get_backend("openai"), default_api_key=None)
    monkeypatch.setattr(diagnostics, "get_backend", lambda _name: keyless)
    result = check_provider("openai", "gpt-4o", api_base_url=None, api_key=None)
    assert result.ok is False
    assert "API key" in result.detail


# ---------------------------------------------------------------------------
# run_checks + render_report
# ---------------------------------------------------------------------------


def test_run_checks_returns_exiftool_then_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_checks runs both diagnostics in display order."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/exiftool")
    _patch_listing(monkeypatch, {"data": [{"id": "m"}]})
    results = run_checks("lmstudio", "m", api_base_url=None, api_key=None)
    assert results[0].name == "ExifTool"
    assert len(results) == 2  # noqa: PLR2004 - exiftool + provider


def test_render_report_returns_true_when_all_pass() -> None:
    """A clean checklist renders OK marks and returns True."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    results = [CheckResult("ExifTool", ok=True, detail="/usr/bin/exiftool")]
    assert render_report(results, console=console) is True
    output = buf.getvalue()
    assert "All checks passed" in output
    assert "ExifTool" in output


def test_render_report_returns_false_on_failure() -> None:
    """A failing check renders a FAIL mark and returns False."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    results = [CheckResult("ExifTool", ok=False, detail="missing")]
    assert render_report(results, console=console) is False
    assert "Some checks failed" in buf.getvalue()
