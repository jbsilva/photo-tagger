"""
Group a batch into shoots and make each shoot's keywords agree with itself.

Every photo is analyzed on its own, so forty frames of the same bird can come back as "Osprey" here
and "Ospreys" there, under "Bird<Animal" on one frame and "Raptor<Wildlife" on the next. Lightroom
then shows four keywords where there is one subject.

The fix has two halves. :func:`plan_sessions` splits the batch where the capture time jumps by more
than the configured gap, which is what a "shoot" actually is. :func:`build_session_vocabulary` then
turns everything the model said about one session into a :class:`~photo_tagger.vocabulary.
Vocabulary`, letting the majority spelling and the majority hierarchy win, and the pipeline snaps
every photo in the session onto it before writing.

Deriving the vocabulary from the batch's own output (rather than nudging the model with a prompt) is
what makes this deterministic: the result does not depend on which photo happened to finish first.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE
from photo_tagger.keywords import parse_hierarchical_keyword
from photo_tagger.metadata import read_capture_times
from photo_tagger.vocabulary import Vocabulary, folds_plurals, fuzzy_key_match, loose_key


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]


# Sorts before every real capture time, so photos with no readable timestamp land in one group of
# their own instead of silently joining whichever shoot they happen to sit next to. Naive, like
# every other timestamp here: these are camera clock readings, not instants on a timeline.
_UNKNOWN_TIME = datetime.min  # noqa: DTZ901


@dataclass(slots=True)
class SessionPlan:
    """
    The batch split into shoots, plus the vocabulary each shoot settled on.

    The pipeline fills in the vocabularies as it goes: a session's is only known once every photo in
    it has been analyzed. The retry pass reads them back, so a photo that needed a second attempt
    still lands on the same terms as the rest of its shoot.
    """

    sessions: list[list[Path]] = field(default_factory=list)
    index_of: dict[Path, int] = field(default_factory=dict)
    vocabularies: dict[int, Vocabulary] = field(default_factory=dict)

    def remember(self, index: int, vocabulary: Vocabulary) -> None:
        """Record the vocabulary session *index* settled on."""
        self.vocabularies[index] = vocabulary

    def vocabulary_for(self, path: Path) -> Vocabulary | None:
        """Return the vocabulary of *path*'s session, or None before it has been computed."""
        index = self.index_of.get(path)
        return None if index is None else self.vocabularies.get(index)


def _timestamps(paths: list[Path], *, et: ExifToolHelper | None) -> dict[Path, datetime]:
    """
    Return a capture time for every path, falling back to file mtime.

    EXIF DateTimeOriginal is the camera's own local time and mtime is converted to local time too,
    so the two are comparable: only the gaps between them matter here. A path we cannot stat at all
    gets :data:`_UNKNOWN_TIME`.
    """
    times = read_capture_times(paths, et=et)
    resolved: dict[Path, datetime] = {}
    for path in paths:
        if (captured := times.get(path)) is not None:
            resolved[path] = captured
            continue
        try:
            # Naive local time on purpose, to stay comparable with EXIF's camera-local stamps.
            resolved[path] = datetime.fromtimestamp(path.stat().st_mtime)  # noqa: DTZ006
        except OSError as exc:
            logger.warning("session_timestamp_failed", file=str(path), error=str(exc))
            resolved[path] = _UNKNOWN_TIME
    return resolved


def plan_sessions(
    image_files: list[Path],
    *,
    gap_minutes: float,
    et: ExifToolHelper | None = None,
) -> SessionPlan | None:
    """
    Split *image_files* into shoots separated by more than *gap_minutes* of idle time.

    Returns None when *gap_minutes* is zero or negative (the feature is off) or when there is
    nothing to group, so callers can treat "no plan" as "run the batch as usual". Photos are
    reordered into capture order, which is the order a session's vocabulary is built from.
    """
    if gap_minutes <= 0 or not image_files:
        return None

    times = _timestamps(image_files, et=et)
    ordered = sorted(image_files, key=lambda path: (times[path], str(path)))
    gap_seconds = gap_minutes * 60.0

    sessions: list[list[Path]] = []
    current: list[Path] = []
    previous: datetime | None = None
    for path in ordered:
        captured = times[path]
        if previous is not None and (captured - previous).total_seconds() > gap_seconds:
            sessions.append(current)
            current = []
        current.append(path)
        previous = captured
    # `ordered` is never empty here (an empty batch returned None above), so there is always a
    # final session to close.
    sessions.append(current)

    plan = SessionPlan(
        sessions=sessions,
        index_of={path: index for index, session in enumerate(sessions) for path in session},
    )
    logger.info(
        "sessions_planned",
        sessions=len(sessions),
        photos=len(ordered),
        gap_minutes=gap_minutes,
        largest=max((len(s) for s in sessions), default=0),
    )
    return plan


def _count_keywords(
    keyword_lists: Iterable[Iterable[str]],
    *,
    fold_plurals: bool,
) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, Counter[tuple[str, ...]]]]:
    """
    Tally how a session used each concept.

    Returns per-concept totals (leaves only, which is what orders the output), the spellings seen
    for each concept, and the hierarchies seen for each leaf concept. A concept is a loose key
    (case- and punctuation-insensitive, plus plurals for English), so "Reflections" and "reflection"
    are counted as one thing with two spellings.

    Hierarchies are counted as chains *of concepts*, not of spellings. Counting the spellings would
    split one shoot's vote between "Птица<Животное" and "птица<Животное" and let a third, genuinely
    rarer parent win on the tie-break.
    """
    totals: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    chains: dict[str, Counter[tuple[str, ...]]] = {}
    for keywords in keyword_lists:
        for keyword in keywords:
            _, parts = parse_hierarchical_keyword(keyword)
            concepts = [loose_key(part, fold_plurals=fold_plurals) for part in parts]
            if not parts or not all(concepts):
                continue
            for part, concept in zip(parts, concepts, strict=True):
                # Every segment, not just the leaf: a parent needs a canonical spelling too.
                spellings.setdefault(concept, Counter())[part] += 1
            totals[concepts[-1]] += 1
            if len(parts) > 1:
                chains.setdefault(concepts[-1], Counter())[tuple(concepts)] += 1
    return totals, spellings, chains


def _most_common_spelling(counter: Counter[str]) -> str:
    """
    Return the spelling used most often, breaking ties alphabetically.

    Ties are the norm in a short session (two photos, two spellings), so they cannot be left to
    insertion order if the run is to be reproducible. The order itself is by code point, which is
    arbitrary but stable in any script.
    """
    return min(counter.items(), key=lambda item: (-item[1], item[0]))[0]


def _most_common_chain(counter: Counter[tuple[str, ...]]) -> tuple[str, ...]:
    """Return the hierarchy used most often, breaking ties toward the deeper chain."""
    return max(counter.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]


def _spell(concepts: Iterable[str], spellings: dict[str, Counter[str]]) -> list[str]:
    """Render a chain of concepts with each segment's majority spelling."""
    return [_most_common_spelling(spellings[concept]) for concept in concepts]


def _variant_map(spellings: dict[str, Counter[str]]) -> dict[str, str]:
    """
    Map each concept onto the busier concept it is merely a variant of, if there is one.

    This is what unifies inflected forms in a language whose morphology nothing here knows: "Закаты"
    folds into "Закат" and "Landschaften" into "Landschaft" because they are close enough as
    strings, and because the more frequent one is settled first. Concepts too short for that
    judgement stand on their own (see :func:`~photo_tagger.vocabulary.fuzzy_key_match`), so "Alle"
    never swallows "Alles".

    A concept whose spellings differ only in case is already one concept by this point; this pass is
    strictly about wording that the loose key alone cannot equate. Every concept takes part, parents
    included, so a hierarchy converges the same way its leaves do.
    """
    mentions = Counter({concept: counter.total() for concept, counter in spellings.items()})
    canonical: dict[str, str] = {}
    accepted: list[str] = []
    for concept, _count in mentions.most_common():
        target = fuzzy_key_match(concept, accepted)
        if target is None:
            accepted.append(concept)
            canonical[concept] = concept
        else:
            canonical[concept] = target
            # The variant's spellings join the winner's tally, so a session that says "Закаты"
            # twice and "Закат" once still writes the form it used most.
            spellings[target].update(spellings[concept])
    return canonical


def build_session_vocabulary(
    keyword_lists: Iterable[Iterable[str]],
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> Vocabulary:
    """
    Derive one session's vocabulary from the keywords its photos generated.

    For every concept the session mentioned, the spelling used most often becomes the canonical one
    and the hierarchy used most often becomes its chain, with every segment of that chain rendered
    in its own majority spelling.

    What counts as "the same concept" depends on *output_language* exactly as it does for a
    vocabulary file: singular and plural are one concept in English, two in a language whose plurals
    this code cannot read (see :func:`~photo_tagger.vocabulary.folds_plurals`). Whatever the
    language, :func:`_variant_map` then folds the close-enough leftovers together, which is how a
    Russian or German shoot converges without anyone teaching this module those languages.
    """
    fold_plurals = folds_plurals(output_language)
    totals, spellings, chains = _count_keywords(keyword_lists, fold_plurals=fold_plurals)
    canonical = _variant_map(spellings)

    merged_chains: dict[str, Counter[tuple[str, ...]]] = {}
    for leaf, seen in chains.items():
        target = merged_chains.setdefault(canonical[leaf], Counter())
        for chain, count in seen.items():
            target[tuple(canonical[segment] for segment in chain)] += count

    entries: list[str] = []
    for concept, _count in totals.most_common():
        if canonical[concept] != concept:
            continue  # Folded into another concept, which carries its spellings already.
        if (seen := merged_chains.get(concept)) is not None:
            entries.append("|".join(_spell(_most_common_chain(seen), spellings)))
        else:
            entries.append(_most_common_spelling(spellings[concept]))
    return Vocabulary.from_entries(entries, fold_plurals=fold_plurals)
