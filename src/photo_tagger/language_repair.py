"""
Ask the model again when its answer mixed writing systems (``--repair-language``).

The CLI pipeline and the GUI worker call the model differently (the pipeline with its sampling
settings and its batch usage, the window with neither), so each passes the one call it makes, with
the photo already prepared, as *ask*. Everything else, what to ask, which answer to keep, and what
the photo then costs, lives here once.
"""

from collections.abc import Callable
from enum import StrEnum

from loguru import logger

from photo_tagger.config import build_repair_prompt
from photo_tagger.models import InferenceResult
from photo_tagger.script_guard import find_in_result


# One model call for the photo at hand, given the prompt to send with it.
type Ask = Callable[[str], InferenceResult]


class RepairStatus(StrEnum):
    """What a repair did, so a caller can count it and decide whether to cache the answer."""

    NOT_NEEDED = "not-needed"  # the answer was in one script, so nothing was asked
    FAILED = "failed"  # the call raised: the original answer stands
    MIXED_AGAIN = "mixed-again"  # the call answered, mixed again: the original answer stands
    FIXED = "fixed"  # the call answered in one script, and that answer replaces the original

    @property
    def answered(self) -> bool:
        """Whether a repair call came back with an answer, which is a call to count and pay."""
        return self in {RepairStatus.MIXED_AGAIN, RepairStatus.FIXED}


def repair_mixed_script(
    inference: InferenceResult,
    ask: Ask,
    *,
    output_language: str,
    file_name: str,
) -> tuple[InferenceResult, RepairStatus]:
    """
    Ask once more, in place, when *inference* mixes writing systems, and keep the better answer.

    Only the same agent and the same image: the question that decides the right fix is whether the
    photo shows that text, which no text-only pass can answer. A failed repair keeps the original,
    because a photo tagged imperfectly still beats one not tagged. So does a repair that comes back
    mixed again, with no third try: the original is the answer the photo was flagged on, so what the
    person reviews is what the check found. The answer kept carries what both calls cost.
    """
    finding = find_in_result(inference)
    if finding is None:
        return inference, RepairStatus.NOT_NEEDED
    logger.info("foreign_script_repair_started", file=file_name, terms=finding.terms)
    prompt = build_repair_prompt(
        title=inference.title,
        description=inference.description,
        keywords=inference.keywords,
        terms=finding.terms,
        output_language=output_language,
    )
    try:
        repaired = ask(prompt)
    except Exception as exc:  # noqa: BLE001 - a failed repair must not cost the photo its tags.
        logger.warning("foreign_script_repair_failed", file=file_name, error=str(exc))
        return inference, RepairStatus.FAILED
    still_mixed = find_in_result(repaired) is not None
    logger.info(
        "foreign_script_repair_finished",
        file=file_name,
        title=repaired.title,
        still_mixed=still_mixed,
    )
    if still_mixed:
        return inference.with_cost_of(repaired), RepairStatus.MIXED_AGAIN
    return repaired.with_cost_of(inference), RepairStatus.FIXED


def repair_replay(
    cached: InferenceResult,
    ask: Ask,
    *,
    output_language: str,
    file_name: str,
) -> tuple[InferenceResult, RepairStatus]:
    """
    Repair an answer replayed from the cache, which a run without the setting stored mixed.

    An earlier run paid for the cached call, so a repair that answered costs only itself. One that
    did not answer (or was not needed) leaves a plain replay, at what it cost back then. Only a
    FIXED answer is worth storing in place of the cached one: the caller does that.
    """
    result, status = repair_mixed_script(
        cached.without_cost(),
        ask,
        output_language=output_language,
        file_name=file_name,
    )
    return (result, status) if status.answered else (cached, status)
