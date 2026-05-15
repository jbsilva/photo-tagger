"""Tests for environment-variable parsing helpers in photo_tagger.config."""

import pytest

from photo_tagger.config import _env_float, _env_int


def test_env_int_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent or blank env var falls back to the supplied default."""
    monkeypatch.delenv("PT_TEST_INT", raising=False)
    assert _env_int("PT_TEST_INT", 42) == 42  # noqa: PLR2004 - asserting default arg
    monkeypatch.setenv("PT_TEST_INT", "")
    assert _env_int("PT_TEST_INT", 42) == 42  # noqa: PLR2004 - asserting default arg


def test_env_int_parses_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numeric string is converted to int."""
    monkeypatch.setenv("PT_TEST_INT", "7")
    assert _env_int("PT_TEST_INT", 0) == 7  # noqa: PLR2004 - asserting parsed value


def test_env_int_warns_and_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric value emits a RuntimeWarning and uses the default; no crash."""
    monkeypatch.setenv("PT_TEST_INT", "high")
    with pytest.warns(RuntimeWarning, match="PT_TEST_INT"):
        result = _env_int("PT_TEST_INT", 80)
    assert result == 80  # noqa: PLR2004 - asserting default fallback


def test_env_float_parses_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A float-shaped string is converted to float."""
    monkeypatch.setenv("PT_TEST_FLOAT", "0.42")
    assert _env_float("PT_TEST_FLOAT", 0.0) == pytest.approx(0.42)


def test_env_float_warns_and_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric float value emits a RuntimeWarning and uses the default."""
    monkeypatch.setenv("PT_TEST_FLOAT", "warm")
    with pytest.warns(RuntimeWarning, match="PT_TEST_FLOAT"):
        result = _env_float("PT_TEST_FLOAT", 0.2)
    assert result == pytest.approx(0.2)
