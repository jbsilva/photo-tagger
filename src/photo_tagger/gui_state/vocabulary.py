"""The controlled vocabulary and shoot harmonization, as the window applies and reports them."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from photo_tagger.config import (
    DEFAULT_OUTPUT_LANGUAGE,
)
from photo_tagger.i18n import _, ngettext
from photo_tagger.pipeline import MAX_TRACKED_DROPPED_TERMS
from photo_tagger.sessions import build_session_vocabulary, plan_sessions
from photo_tagger.vocabulary import Vocabulary, VocabularyError, load_vocabulary


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from photo_tagger.vocabulary import Vocabulary


# How many rejected terms the vocabulary status line names before it stops. The point is to show
# what kind of keyword is being dropped, not to list every one; the log has them all.
_MAX_LISTED_DROPPED = 5


@dataclass(frozen=True, slots=True)
class VocabularyOutcome:
    """
    What a controlled vocabulary made of one photo's generated keywords.

    ``mapped`` counts the keywords rewritten to the catalog's own spelling and hierarchy, and
    ``dropped`` names the ones strict mode refused, so a run can report both instead of silently
    changing what the model said.
    """

    keywords: list[str] = field(default_factory=list)
    mapped: int = 0
    dropped: list[str] = field(default_factory=list)


def apply_vocabulary(
    keywords: Iterable[str],
    vocabulary: Vocabulary | None,
    *,
    strict: bool = False,
) -> VocabularyOutcome:
    """
    Snap generated keywords onto *vocabulary*, or pass them through when there is none.

    The GUI applies this to a proposal before it reaches the review pane, so what you edit is what a
    save writes, exactly as the CLI's ``--vocabulary`` rewrites keywords before merging them.
    """
    if not vocabulary:
        return VocabularyOutcome(keywords=list(keywords))
    result = vocabulary.snap(keywords, strict=strict)
    return VocabularyOutcome(
        keywords=result.keywords,
        mapped=len(result.mapped),
        dropped=list(result.dropped),
    )


def record_dropped_terms(tally: dict[str, int], dropped: Iterable[str]) -> None:
    """
    Fold one photo's rejected keywords into the run's tally, counting how often each came up.

    New terms stop being recorded past :data:`~photo_tagger.pipeline.MAX_TRACKED_DROPPED_TERMS`, the
    same bound the CLI keeps: a run against the wrong vocabulary rejects thousands of distinct
    keywords, and the summary only ever names the busiest few of them anyway.
    """
    for term in dropped:
        if term in tally:
            tally[term] += 1
        elif len(tally) < MAX_TRACKED_DROPPED_TERMS:
            tally[term] = 1


def load_vocabulary_file(
    path: Path,
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> tuple[Vocabulary | None, str]:
    """
    Load a vocabulary file, returning it or the message explaining why it could not be used.

    The window shows the message rather than raising: choosing the wrong file is a normal mistake,
    and the run it would have affected has not started yet.
    """
    try:
        return load_vocabulary(path, output_language=output_language), ""
    except VocabularyError as exc:
        return None, str(exc)


def vocabulary_status(path: Path | None, vocabulary: Vocabulary | None, error: str = "") -> str:
    """Describe the vocabulary in force, for the label under the file picker."""
    if error:
        return error
    if path is None or vocabulary is None:
        return _("No vocabulary: keywords are written as the model wrote them.")
    terms = ngettext("{n} keyword", "{n} keywords", len(vocabulary.terms)).format(
        n=len(vocabulary.terms),
    )
    return _("{terms} from {name}.").format(terms=terms, name=path.name)


def vocabulary_summary(mapped: int, dropped: Mapping[str, int]) -> str:
    """
    One line for the status bar after a run: what the vocabulary changed across the batch.

    Empty when it changed nothing, so a run with a vocabulary that already fits says nothing rather
    than reporting two zeros. The named terms are the most frequent rejections, which is what tells
    you whether the vocabulary needs a new keyword or the photos need a different one.
    """
    parts: list[str] = []
    if mapped:
        parts.append(
            ngettext("{n} keyword rewritten", "{n} keywords rewritten", mapped).format(n=mapped),
        )
    if dropped:
        ranked = sorted(dropped.items(), key=lambda item: (-item[1], item[0].casefold()))
        listed = ", ".join(term for term, _count in ranked[:_MAX_LISTED_DROPPED])
        if len(ranked) > _MAX_LISTED_DROPPED:
            listed += ", ..."
        parts.append(
            ngettext(
                "{n} keyword dropped ({terms})",
                "{n} keywords dropped ({terms})",
                len(ranked),
            ).format(n=len(ranked), terms=listed),
        )
    if not parts:
        return ""
    return _("Vocabulary: {summary}.").format(summary=", ".join(parts))


@dataclass(frozen=True, slots=True)
class HarmonizeResult:
    """
    What harmonizing the generated proposals changed.

    ``keywords`` maps a photo's key (its path as a string, as the window holds it) to its new
    keyword list, and only carries the photos that actually changed.
    """

    keywords: dict[str, list[str]] = field(default_factory=dict)
    sessions: int = 0


def harmonize_sessions(
    keywords_by_path: Mapping[Path, list[str]],
    *,
    gap_minutes: float,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> HarmonizeResult:
    """
    Group the generated photos into shoots and make each shoot's keywords agree with itself.

    The CLI holds a session's writes until every photo in it has been analyzed; the GUI writes
    nothing until you press Save, so the same harmonization runs over the finished proposals
    instead. Either way the vocabulary comes from the session's own output, which is what makes it
    deterministic.

    Returns an empty result when the feature is off (*gap_minutes* of zero) or there is nothing to
    group.
    """
    paths = list(keywords_by_path)
    plan = plan_sessions(paths, gap_minutes=gap_minutes)
    if plan is None:
        return HarmonizeResult()
    changed: dict[str, list[str]] = {}
    for session in plan.sessions:
        vocabulary = build_session_vocabulary(
            (keywords_by_path[path] for path in session),
            output_language=output_language,
        )
        if not vocabulary:
            continue
        for path in session:
            snapped = vocabulary.snap(keywords_by_path[path]).keywords
            if snapped != keywords_by_path[path]:
                changed[str(path)] = snapped
    return HarmonizeResult(keywords=changed, sessions=len(plan.sessions))


def harmonize_summary(result: HarmonizeResult) -> str:
    """Status line after harmonizing: how many shoots there were, and how many photos changed."""
    if not result.sessions:
        return _("Nothing to harmonize yet: generate some photos first.")
    shoots = ngettext("{n} shoot", "{n} shoots", result.sessions).format(n=result.sessions)
    changed = len(result.keywords)
    if not changed:
        return _("{shoots}: the keywords already agreed.").format(shoots=shoots)
    return ngettext(
        "{shoots}: harmonized the keywords on {n} photo.",
        "{shoots}: harmonized the keywords on {n} photos.",
        changed,
    ).format(shoots=shoots, n=changed)
