"""Tests for environment-variable parsing helpers and prompts in photo_tagger.config."""

import pytest

from photo_tagger.config import (
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_SYSTEM_PROMPT,
    _env_float,
    _env_int,
    build_system_prompt,
    exiftool_executable,
)


def test_build_system_prompt_defaults_to_english() -> None:
    """With no argument the prompt asks for English and is what DEFAULT_SYSTEM_PROMPT holds."""
    prompt = build_system_prompt()
    assert DEFAULT_OUTPUT_LANGUAGE == "English"
    assert "Title Case, English." in prompt
    assert "English only, in every field" in prompt
    assert prompt == DEFAULT_SYSTEM_PROMPT


def test_build_system_prompt_swaps_in_the_requested_language() -> None:
    """A chosen language replaces English in every field rule, leaving no placeholder behind."""
    prompt = build_system_prompt("Brazilian Portuguese")
    assert "Title Case, Brazilian Portuguese." in prompt
    assert "Brazilian Portuguese only, in every field" in prompt
    assert "{language}" not in prompt
    assert "English" not in prompt


def test_system_prompt_rules_out_untranslated_loanwords() -> None:
    """The language rule covers a foreign word used AS the concept, not just mixed-up spelling."""
    prompt = build_system_prompt()
    assert "'Ocarina', never '陶笛'" in prompt
    assert "only when the photo shows that text" in prompt


def test_exiftool_executable_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No (or blank) PHOTO_TAGGER_EXIFTOOL means 'find exiftool on PATH' (None)."""
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    assert exiftool_executable() is None
    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "")
    assert exiftool_executable() is None


def test_exiftool_executable_returns_env_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit PHOTO_TAGGER_EXIFTOOL is returned verbatim."""
    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "/opt/et/exiftool")
    assert exiftool_executable() == "/opt/et/exiftool"


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
