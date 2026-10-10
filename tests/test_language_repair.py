"""Tests for the shared repair of answers that mix writing systems."""

from dataclasses import replace

import pytest

from photo_tagger.language_repair import RepairStatus, repair_mixed_script, repair_replay
from photo_tagger.models import InferenceResult


_MIXED = InferenceResult(
    title="Decorated陶笛",
    description="A陶笛 on a white surface.",
    keywords=["陶笛<Wind Instrument"],
    total_tokens=100,
)
_CLEAN = InferenceResult(
    title="Decorated Ocarina",
    description="An ocarina on a white surface.",
    keywords=["Ocarina<Wind Instrument"],
    total_tokens=40,
)


def _never(_prompt: str) -> InferenceResult:
    pytest.fail("no call expected")


def _fails(_prompt: str) -> InferenceResult:
    msg = "the model server died"
    raise RuntimeError(msg)


def _repair(inference: InferenceResult, ask: object) -> tuple[InferenceResult, RepairStatus]:
    return repair_mixed_script(inference, ask, output_language="English", file_name="a.jpg")  # type: ignore[arg-type]


def test_an_answer_in_one_script_asks_nothing() -> None:
    """The check is free; only a flagged answer pays for a second call."""
    assert _repair(_CLEAN, _never) == (_CLEAN, RepairStatus.NOT_NEEDED)


def test_a_fixed_answer_replaces_the_original_and_carries_both_costs() -> None:
    """The prompt names the words to replace, and the row counts both calls."""
    prompts: list[str] = []

    def ask(prompt: str) -> InferenceResult:
        prompts.append(prompt)
        return _CLEAN

    result, status = _repair(_MIXED, ask)

    assert status is RepairStatus.FIXED
    assert status.answered
    assert "陶笛" in prompts[0]
    assert result.title == _CLEAN.title
    assert result.total_tokens == _MIXED.total_tokens + _CLEAN.total_tokens


def test_an_answer_mixed_again_keeps_the_original_text() -> None:
    """The original is what the photo was flagged on, so it is what the person reviews."""
    again = replace(_CLEAN, title="Painted陶器")

    result, status = _repair(_MIXED, lambda _p: again)

    assert status is RepairStatus.MIXED_AGAIN
    assert status.answered
    assert result.title == _MIXED.title
    assert result.total_tokens == _MIXED.total_tokens + again.total_tokens


def test_a_failed_call_keeps_the_original_answer() -> None:
    """A dead server on the follow-up costs the fix, never the photo's tags."""
    result, status = _repair(_MIXED, _fails)

    assert (result, status) == (_MIXED, RepairStatus.FAILED)
    assert not status.answered


def test_a_repaired_replay_costs_only_the_repair() -> None:
    """An earlier run paid for the cached call."""
    result, status = repair_replay(
        _MIXED,
        lambda _p: _CLEAN,
        output_language="English",
        file_name="a.jpg",
    )

    assert status is RepairStatus.FIXED
    assert result.total_tokens == _CLEAN.total_tokens


@pytest.mark.parametrize("ask", [_fails, _never])
def test_a_replay_with_no_answer_stays_the_cached_one(ask: object) -> None:
    """A failed call, or none needed, leaves a plain replay at what it cost back then."""
    cached = _CLEAN if ask is _never else _MIXED

    result, _status = repair_replay(cached, ask, output_language="English", file_name="a.jpg")  # type: ignore[arg-type]

    assert result is cached
