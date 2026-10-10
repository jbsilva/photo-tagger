"""
Build a controlled vocabulary out of the keywords a library already uses.

``--vocabulary-strict`` is the flag that stops a catalog sprawling, and it is unusable without a
good keyword file to point it at. Hand-writing one is not realistic, and exporting the whole catalog
does not help either: a catalog that has been tagged by a tool rather than by hand runs to tens of
thousands of keywords, most of them used once, which locks the sprawl in place instead of ending it.

So the file is derived from the library itself, in two steps:

* :func:`census_from_photos` counts what the photos actually carry, read through exiftool. This is
  the source that works everywhere, because it asks the photos rather than the application: digiKam,
  darktable, Immich, PhotoPrism, Synology Photos and the rest all write XMP/IPTC keywords, and only
  Lightroom offers a keyword-list export at all. It is also the more truthful count, one per photo,
  and ``XMP-lr:HierarchicalSubject`` carries the hierarchy those photos really use.
* :func:`census_from_export` reads a Lightroom keyword export instead, for a catalog that is not on
  this machine. Its counts are occurrences in the keyword *tree*, not photos, so a term filed under
  forty parents scores forty. Treat it as the proxy it is.

:func:`trim` then applies deterministic rules only: how often a term is used, how long it is,
whether it looks like a measurement rather than a subject, and which of several spellings of one
concept wins. Nothing here asks a model. Judgment calls that rules genuinely cannot make (which
terms are synonyms of each other, what a sane hierarchy would be) are a separate, opt-in pass.

Nothing in this module writes to a photo or to a catalog. It reads, counts, and renders a file for
the user to review and edit.
"""

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.csv_report import neutralize_formula
from photo_tagger.errors import DiscoveryError
from photo_tagger.keywords import parse_hierarchical_keyword
from photo_tagger.metadata import read_keyword_sets
from photo_tagger.vocabulary import loose_key, parse_keyword_lines


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]

    from photo_tagger.models import KeywordSet

    # Annotation only: vocabulary_organize imports this module, so the runtime dependency has to
    # stay one-way. Python 3.14 evaluates annotations lazily, so the name is never needed at import.
    from photo_tagger.vocabulary_organize import OrganizeStats


# Photos read per exiftool call. Large enough that the IPC cost disappears into the read, small
# enough that a command line stays well inside every platform's argument limit.
_PHOTO_BATCH = 200

# Terms that are measurements, model numbers, or timestamps rather than subjects. A keyword with a
# digit in it is almost always one of those ("19.5V", "0 Percent Battery", "1J04 0615"), and the few
# real ones ("35Mm") are easier to add back by hand than the hundreds of others are to remove.
_DIGIT_RE = re.compile(r"\d")

# Punctuation no subject keyword needs. '<' and '|' are the hierarchy separators, so a term
# carrying one would be re-read as a path.
_ODD_PUNCTUATION_RE = re.compile(r"[%/<>@&*+=\\|]")

DROP_RARE = "rare"
DROP_DIGITS = "digits"
DROP_TOO_LONG = "too-long"
DROP_TOO_MANY_WORDS = "too-many-words"
DROP_PUNCTUATION = "punctuation"
DROP_VARIANT = "variant"
DROP_OVER_LIMIT = "over-limit"
# Not a rejection: the organize pass folded this keyword into another, which still writes it as a
# {synonym} of the keyword that kept it. Reported so the fold is visible rather than silent.
DROP_SYNONYM = "synonym"


@dataclass(slots=True)
class KeywordCensus:
    """
    How often each keyword is used, and the hierarchies it was seen in.

    ``uses`` counts photos (or tree occurrences, for an export). ``chains`` records every root-to-
    leaf chain a term appeared as the leaf of, with its own count, so the most-used one can win
    later; a term only ever seen flat has no entry.
    """

    uses: Counter[str] = field(default_factory=Counter)
    chains: dict[str, Counter[tuple[str, ...]]] = field(default_factory=dict)
    photos: int = 0

    def add(self, chain: Sequence[str], *, weight: int = 1, credit_ancestors: bool = True) -> None:
        """
        Record one use of *chain*.

        A photo tagged ``Animal|Bird|Osprey`` really does use all three keywords, so every level is
        credited. An export is different: each level already has its own line, and crediting
        ancestors again would count a root once per descendant. Hence *credit_ancestors*.
        """
        cleaned = [segment.strip() for segment in chain if segment.strip()]
        if not cleaned:
            return
        for depth, segment in enumerate(cleaned, start=1):
            if credit_ancestors or depth == len(cleaned):
                self.uses[segment] += weight
            # Every prefix is a placement of its own: seeing Animal|Bird|Osprey also says where
            # "Bird" belongs, and without this an intermediate term would render flat.
            if depth > 1:
                self.chains.setdefault(segment, Counter())[tuple(cleaned[:depth])] += weight

    def best_chain(self, term: str) -> tuple[str, ...]:
        """Return the most-used chain ending in *term*, or an empty tuple when it is always flat."""
        seen = self.chains.get(term)
        if not seen:
            return ()
        # Ties break on the longer chain, then alphabetically, so the same census always renders
        # the same file.
        return max(seen, key=lambda chain: (seen[chain], len(chain), chain))

    def merge(self, other: KeywordCensus) -> None:
        """
        Fold *other*'s counts into this census.

        Lets one build read both a keyword export and the photos themselves: two censuses of the
        same library, counted differently, added up before the rules run.
        """
        for term, count in other.uses.items():
            self.uses[term] += count
        for term, chains in other.chains.items():
            self.chains.setdefault(term, Counter()).update(chains)
        self.photos += other.photos


def _split_chain(entry: str) -> list[str]:
    """
    Split one stored keyword into its root-to-leaf segments.

    ``XMP-lr:HierarchicalSubject`` writes ``Animal|Bird|Osprey``; the model's own leaf-first
    ``Osprey<Bird<Animal`` turns up in hand-edited files, so both are accepted.
    """
    if "<" in entry or ">" in entry:
        return parse_hierarchical_keyword(entry)[1]
    return [segment.strip() for segment in entry.split("|") if segment.strip()]


def _chains_of(keywords: KeywordSet) -> list[list[str]]:
    """
    Return one chain per distinct keyword on a photo, hierarchies preferred over flat spellings.

    A photo usually carries the same keyword twice, once as a flat ``XMP-dc:Subject`` entry and once
    inside a ``Animal|Bird|Osprey`` path. Counting both would double every term and lose the tie
    between them, so a flat keyword is dropped when some path already ends in it.
    """
    chains = [_split_chain(entry) for entry in keywords.hierarchical]
    covered = {segment.casefold() for chain in chains for segment in chain}
    flat = dict.fromkeys(keywords.subject + keywords.weighted)
    chains.extend([term] for term in flat if term.strip() and term.casefold() not in covered)
    return [chain for chain in chains if chain]


def census_from_photos(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> KeywordCensus:
    """
    Count the keywords already written on *image_paths*, one vote per photo.

    Works with any application that writes XMP or IPTC, which is the point: only Lightroom exports a
    keyword list, while every DAM writes the keywords themselves. Photos are read in batches so the
    exiftool cost stays flat across a large library.

    Raises :class:`~photo_tagger.errors.DiscoveryError` when a batch cannot be read at all.
    ``read_keyword_sets`` maps every path it was given, so an empty result means exiftool never ran,
    not that the photos carry no keywords. Counting that as zero told a user whose library is full
    of keywords that they had none, with the real reason only in the log.
    """
    census = KeywordCensus()
    paths = list(image_paths)
    for start in range(0, len(paths), _PHOTO_BATCH):
        batch = paths[start : start + _PHOTO_BATCH]
        read = read_keyword_sets(batch, et=et)
        if batch and not read:
            msg = f"could not read keywords from {len(batch)} photo(s); is exiftool working?"
            raise DiscoveryError(msg)
        for keywords in read.values():
            census.photos += 1
            for chain in _chains_of(keywords):
                census.add(chain)
        logger.debug("keyword_census_progress", photos=census.photos, terms=len(census.uses))
    logger.info("keyword_census_done", photos=census.photos, terms=len(census.uses))
    return census


def census_from_export(text: str) -> KeywordCensus:
    """
    Count keywords from a Lightroom keyword export instead of from photos.

    The count is how often a term occurs in the keyword tree, not how many photos use it: an export
    carries no usage at all. It is a usable proxy (a term filed in many places is one the catalog
    leans on) but it is not the same measure, so a run should say which one it used.
    """
    census = KeywordCensus()
    for entry in parse_keyword_lines(text):
        census.add(entry.chain, credit_ancestors=False)
        for synonym in entry.synonyms:
            census.add([synonym])
    logger.info("keyword_census_done", source="export", terms=len(census.uses))
    return census


@dataclass(slots=True, frozen=True)
class TrimRules:
    """
    The deterministic filters that turn a census into a keyword list.

    Defaults aim at a file that stays under the 5,000 terms above which vocabulary matching drops
    its fuzzy pass, on a catalog of the size that makes this command worth running at all.
    """

    min_uses: int = 2
    max_terms: int | None = 4800
    max_words: int = 3
    max_chars: int = 30
    allow_digits: bool = False
    collapse_variants: bool = True
    fold_plurals: bool = True


@dataclass(slots=True, frozen=True)
class Dropped:
    """One term the rules rejected, and why."""

    term: str
    uses: int
    reason: str
    detail: str = ""


@dataclass(slots=True, frozen=True)
class TrimResult:
    """
    The kept terms (most used first) and every rejection, for the report.

    ``chains`` and ``synonyms`` are empty after a plain trim: hierarchies then come from the census,
    which is what the library itself says. The optional organize pass fills them in to override that
    with a hierarchy and a set of synonym groups it worked out instead.
    """

    kept: list[str] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    census: KeywordCensus = field(default_factory=KeywordCensus)
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    # The trim's TrimRules.fold_plurals, so the organize pass matches the model's reply the same
    # way: folded, "Alle" and "Alles" would be one keyword in German.
    fold_plurals: bool = True

    def chain_for(self, term: str) -> tuple[str, ...]:
        """Return *term*'s hierarchy: the organized one if there is one, else the census's."""
        if (chain := self.chains.get(term)) is not None:
            return chain
        return self.census.best_chain(term)


def _shape_reason(term: str, rules: TrimRules) -> str | None:
    """Return why *term* is not keyword-shaped, or None when it is."""
    if not rules.allow_digits and _DIGIT_RE.search(term):
        return DROP_DIGITS
    if _ODD_PUNCTUATION_RE.search(term):
        return DROP_PUNCTUATION
    if len(term) > rules.max_chars:
        return DROP_TOO_LONG
    if len(term.split()) > rules.max_words:
        return DROP_TOO_MANY_WORDS
    return None


def _rank(census: KeywordCensus) -> list[tuple[str, int]]:
    """Order the census most-used first, breaking ties alphabetically for a stable output."""
    return sorted(census.uses.items(), key=lambda item: (-item[1], item[0].casefold()))


def _collapse_variants(
    ranked: list[tuple[str, int]],
    rules: TrimRules,
) -> tuple[list[tuple[str, int]], list[Dropped]]:
    """
    Keep one spelling per concept: "Animals" and "Animal" are the same keyword twice.

    The most-used spelling wins, which is what the catalog itself votes for. The loser is not lost,
    only unlisted: vocabulary matching folds case, punctuation, and plurals, so a photo tagged
    "Animals" still snaps onto "Animal".
    """
    if not rules.collapse_variants:
        return ranked, []
    winners: dict[str, str] = {}
    kept: list[tuple[str, int]] = []
    dropped: list[Dropped] = []
    for term, uses in ranked:
        key = loose_key(term, fold_plurals=rules.fold_plurals)
        if (winner := winners.get(key)) is not None:
            dropped.append(Dropped(term, uses, DROP_VARIANT, f"kept {winner}"))
            continue
        winners[key] = term
        kept.append((term, uses))
    return kept, dropped


def trim(census: KeywordCensus, rules: TrimRules | None = None) -> TrimResult:
    """
    Apply the rules to *census* and report what was kept, what was dropped, and why.

    The order matters: shape first (a measurement is never a keyword however often it occurs), then
    rarity, then variants, then the cap. The cap is applied last so it removes the least-used terms
    that survived everything else, rather than terms a later rule would have removed anyway.
    """
    rules = rules or TrimRules()
    ranked = _rank(census)

    survivors: list[tuple[str, int]] = []
    dropped: list[Dropped] = []
    for term, uses in ranked:
        if (reason := _shape_reason(term, rules)) is not None:
            dropped.append(Dropped(term, uses, reason))
        elif uses < rules.min_uses:
            dropped.append(Dropped(term, uses, DROP_RARE, f"used {uses}x"))
        else:
            survivors.append((term, uses))

    survivors, variant_drops = _collapse_variants(survivors, rules)
    dropped.extend(variant_drops)

    if rules.max_terms is not None and len(survivors) > rules.max_terms:
        cut = survivors[rules.max_terms :]
        survivors = survivors[: rules.max_terms]
        dropped.extend(
            Dropped(term, uses, DROP_OVER_LIMIT, f"beyond {rules.max_terms}") for term, uses in cut
        )

    logger.info("vocabulary_trimmed", kept=len(survivors), dropped=len(dropped))
    return TrimResult(
        kept=[term for term, _ in survivors],
        dropped=dropped,
        census=census,
        fold_plurals=rules.fold_plurals,
    )


def render_vocabulary(result: TrimResult, *, header: str = "", flat: bool = False) -> str:
    """
    Render the kept terms as a vocabulary file, hierarchies included where they are known.

    A term the census saw inside a hierarchy is written as its ``Animal|Bird|Osprey`` path, which
    the vocabulary parser reads back as a path; a term only ever seen flat is written on its own.
    Chains are filtered down to segments that survived the trim, so a parent the rules rejected
    cannot re-enter through one of its children. Terms are sorted alphabetically, because the file
    is meant to be read and edited by hand.

    Synonyms worked out by the organize pass are written as ``{braces}`` on the keyword that kept
    them, so a photo tagged with a folded spelling still matches.

    Pass *flat* to drop the hierarchies and write bare terms. Worth doing when the source hierarchy
    is not trustworthy: an export from a catalog a tool has been writing to can file "Beach" under
    "Sand", and a vocabulary imposes its hierarchy on every photo it matches.
    """
    kept = set(result.kept)
    # A category invented by the organize pass is a legitimate parent even though it is not itself
    # a kept keyword; a parent from the census is only allowed if it survived the trim.
    organized = {segment for chain in result.chains.values() for segment in chain}
    lines = []
    for term in kept:
        chain = [] if flat else [s for s in result.chain_for(term) if s in kept or s in organized]
        line = "|".join(chain) if len(chain) > 1 else term
        if aliases := result.synonyms.get(term):
            line += "".join(f" {{{alias}}}" for alias in aliases)
        lines.append(line)
    # Sorted by the rendered line, not by the term, so a path files under its root and the tree
    # reads top-down.
    body = "\n".join(dict.fromkeys(sorted(lines, key=lambda line: (line.casefold(), line))))
    return f"{header}\n{body}\n" if header else f"{body}\n"


def vocabulary_header(
    source: str,
    kept: int,
    dropped: int,
    rules: TrimRules,
    stats: OrganizeStats | None = None,
) -> str:
    """
    Explain at the top of the generated file where it came from and how to change it.

    Rendered here rather than in the command, so a file built from the desktop GUI carries the same
    provenance as one built from the CLI.
    """
    cap = rules.max_terms if rules.max_terms is not None else "no cap"
    lines = [
        f"# photo-tagger vocabulary: {kept} keywords kept, {dropped} dropped.",
        f"# Source: {source}.",
        (
            f"# Rules: used at least {rules.min_uses}x, at most {cap} terms, "
            f"digits {'kept' if rules.allow_digits else 'dropped'}."
        ),
    ]
    if stats is not None:
        lines += [
            (
                f"# Organized by {stats.model_name}: {stats.grouped} keyword(s) folded into a "
                f"synonym, {stats.categorized} filed under a category."
            ),
            f"# Categories (written to your photos as parents): {', '.join(stats.categories)}.",
        ]
        if stats.failed_chunks:
            # Otherwise a run where the model went away mid-pass reads as a catalog that simply
            # had nothing to group, and the missing hierarchy looks like the tool's verdict.
            lines.append(
                f"# {stats.failed_chunks} chunk(s) failed: those keywords were left ungrouped. "
                f"Re-run --organize to fill them in.",
            )
    lines += [
        "#",
        "# Edit freely: one keyword per line, 'Parent|Child' for a hierarchy, indentation for a",
        "# tree, {braces} for a synonym. A line starting with '# ' is a comment.",
    ]
    return "\n".join(lines) + "\n"


def render_drop_report(result: TrimResult) -> str:
    """
    Render every dropped term as CSV: what it was, how often it was used, and which rule cut it.

    This is the half that makes the thresholds tunable. A run that drops a term you care about says
    so here, with the count that would have kept it.

    The keyword cell goes through :func:`~photo_tagger.csv_report.neutralize_formula` like every
    other CSV this project writes: a keyword arrives from whatever wrote it onto the photo, and a
    catalog can carry one starting with ``=`` or ``@``, which a spreadsheet would run as a formula
    instead of showing as text (CWE-1236). The other three cells are written here and always open
    with a digit or a fixed word.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["keyword", "uses", "reason", "detail"])
    for entry in sorted(result.dropped, key=lambda item: (-item.uses, item.term.casefold())):
        writer.writerow([neutralize_formula(entry.term), entry.uses, entry.reason, entry.detail])
    return buffer.getvalue()
