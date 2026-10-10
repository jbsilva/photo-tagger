"""Tests for environment-variable parsing helpers and prompts in photo_tagger.config."""

import pytest

from photo_tagger.config import (
    DEFAULT_OUTPUT_LANGUAGE,
    _env_float,
    _env_int,
    build_repair_prompt,
    build_system_prompt,
    exiftool_executable,
)


def test_build_system_prompt_defaults_to_english() -> None:
    """With no argument the prompt asks for English."""
    prompt = build_system_prompt()
    assert DEFAULT_OUTPUT_LANGUAGE == "English"
    assert "Title Case, English." in prompt
    assert "English only, in every field" in prompt


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


def test_env_in_range_falls_back_on_a_value_outside_the_range() -> None:
    """DUTY_PERCENT=0 would be a run that may never work, so it warns and uses the default."""
    from photo_tagger.config import _env_in_range  # noqa: PLC0415

    with pytest.warns(RuntimeWarning, match="DUTY_PERCENT"):
        assert _env_in_range("DUTY_PERCENT", 0, 100, 1, 100) == 100  # noqa: PLR2004
    assert _env_in_range("DUTY_PERCENT", 70, 100, 1, 100) == 70  # noqa: PLR2004


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


def test_repair_prompt_quotes_the_answer_and_names_the_terms() -> None:
    """The follow-up gives the model everything it needs to edit rather than start over."""
    prompt = build_repair_prompt(
        title="Decorated陶笛 with Geometric Pattern",
        description="A陶笛 on a white surface.",
        keywords=["陶笛<Wind Instrument", "Geometric Pattern"],
        terms=["陶笛"],
    )
    assert "Decorated陶笛 with Geometric Pattern" in prompt
    assert "陶笛<Wind Instrument, Geometric Pattern" in prompt
    assert "Not in English: 陶笛" in prompt
    # The escape hatch survives: text the photo actually shows is kept, not stripped.
    assert "If the photo actually shows that text" in prompt


def test_repair_prompt_asks_for_the_chosen_language() -> None:
    """A German run is asked for German, in every sentence that names a language."""
    prompt = build_repair_prompt(
        title="Decorated陶笛",
        description="A陶笛.",
        keywords=[],
        terms=["陶笛"],
        output_language="German",
    )
    assert "must be written in German" in prompt
    assert "Not in German" in prompt
    assert "English" not in prompt
