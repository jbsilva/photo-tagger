"""
Hierarchical-keyword parsing and merging logic.

Lightroom expresses hierarchical keywords as pipe-separated paths ("Animal|Bird|Duck"). The model
emits the inverse, leaf-first form ("Duck<Bird<Animal"). This module converts between the two and
merges fresh AI keywords with whatever already lives on the photo.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.config import MIN_HIERARCHICAL_DEPTH
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def parse_hierarchical_keyword(keyword: str) -> tuple[str, list[str]]:
    """
    Parse a hierarchical keyword in format 'Child<Parent<Grandparent' to Lightroom format.

    Args:
        keyword: Keyword string, either flat or hierarchical with '<' separators ('>' is read
                 as '<'). Example: "Duck<Bird<Animal" or just "Landscape"

    Returns:
        Tuple of (hierarchical_format, flat_list) where:
        - hierarchical_format: "Grandparent|Parent|Child" (Lightroom format)
        - flat_list: ["Grandparent", "Parent", "Child"] (all levels as separate keywords)

    Examples:
        >>> parse_hierarchical_keyword("Duck<Bird<Animal")
        ('Animal|Bird|Duck', ['Animal', 'Bird', 'Duck'])
        >>> parse_hierarchical_keyword("Landscape")
        ('Landscape', ['Landscape'])
    """
    keyword = keyword.strip()
    if not keyword:
        return ("", [])

    # The model occasionally flips the bracket and emits "Leaf>Parent" chains, or even mixes both
    # within one chain. Normalize every '>' to '<' unconditionally rather than only when '<' is
    # absent: treating a stray '>' as noise to strip (the previous behavior) silently fused the
    # segments on either side of it into one garbled keyword and dropped a hierarchy level.
    sanitized = keyword.replace(">", "<")
    if "<" not in sanitized:
        return (sanitized, [sanitized])

    parts = [p.strip() for p in sanitized.split("<") if p.strip()]
    if not parts:
        return ("", [])

    # Reverse so we emit Lightroom's root-to-leaf order.
    parts_reversed = list(reversed(parts))
    return ("|".join(parts_reversed), parts_reversed)


def dedupe_keywords(keywords: Iterable[str]) -> list[str]:
    """
    Drop blank entries and case-insensitive duplicates, keeping first-occurrence order.

    Models sometimes repeat a keyword (or emit it both flat and as a hierarchy leaf); showing or
    counting the repeats is noise, so callers collapse them as early as possible.

    Examples:
        >>> dedupe_keywords(["Perch", "Bird", " perch "])
        ['Perch', 'Bird']
    """
    seen: set[str] = set()
    out: list[str] = []
    for keyword in keywords:
        stripped = keyword.strip()
        key = stripped.casefold()
        if stripped and key not in seen:
            seen.add(key)
            out.append(stripped)
    return out


def _capitalize_segment(segment: str, verbatim: Mapping[str, str] | None = None) -> str:
    """
    Title-case *segment* only when it is fully lowercase and nobody has spelled it for us.

    Mixed or upper case is deliberate (``NYC``, ``iPhone``) and must survive untouched;
    ``str.title()`` would corrupt it (``Nyc``) and also capitalizes after apostrophes
    (``Bird'S Nest``), so lowercase segments capitalize per whitespace-separated word instead.

    *verbatim* maps casefolded terms to the exact spelling a controlled vocabulary declared for
    them. Those win outright: a catalog that writes its keywords in lower case ("gegenlicht",
    "птица") means it, and capitalizing them would seed the very near-duplicates the vocabulary
    exists to prevent.

    Examples:
        >>> _capitalize_segment("bird's nest")
        "Bird's Nest"
        >>> _capitalize_segment("NYC")
        'NYC'
        >>> _capitalize_segment("gegenlicht", {"gegenlicht": "gegenlicht"})
        'gegenlicht'
    """
    if verbatim is not None and (exact := verbatim.get(segment.casefold())) is not None:
        return exact
    if segment != segment.lower():
        return segment
    return " ".join(word.capitalize() for word in segment.split())


def _normalize_chain_parts(
    parts: Iterable[str],
    verbatim: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Return capitalized chain segments, skipping blanks.

    Examples:
        >>> _normalize_chain_parts([" duck ", "Bird", ""])
        ['Duck', 'Bird']
    """
    return [
        _capitalize_segment(segment.strip(), verbatim)
        for segment in parts
        if segment and segment.strip()
    ]


def _register_chain(registry: dict[str, list[str]], chain: list[str]) -> None:
    """
    Keep the longest chain for each leaf.

    Examples:
        >>> reg: dict[str, list[str]] = {}
        >>> _register_chain(reg, ["Animal", "Bird"])
        >>> reg["bird"]
        ['Animal', 'Bird']

    """
    if len(chain) < MIN_HIERARCHICAL_DEPTH:
        return
    leaf_key = chain[-1].casefold()
    current = registry.get(leaf_key)
    if current is None or len(chain) > len(current):
        registry[leaf_key] = chain


def _seed_longest_from_existing(hierarchical_keywords: Iterable[str]) -> dict[str, list[str]]:
    """
    Prime the longest-chain registry with existing hierarchical entries.

    Examples:
        >>> _seed_longest_from_existing(["Animal|Bird", "Plant"])
        {'bird': ['Animal', 'Bird']}
    """
    registry: dict[str, list[str]] = {}
    for entry in hierarchical_keywords:
        normalized = _normalize_chain_parts(entry.split("|"))
        _register_chain(registry, normalized)
    return registry


@dataclass(slots=True)
class _FlatKeywords:
    """
    The two flat keyword views a merge grows, each de-duplicated against its own contents.

    ``subject`` feeds the XMP-dc:Subject tag (mirrored to IPTC:Keywords) and ``weighted`` feeds the
    XMP-lr:WeightedFlatSubject tag. They usually hold the same terms, but they are read from
    different tags and a photo can carry a weighted entry that its Subject list does not, so each
    needs its own seen-set: de-duplicating both against ``subject`` alone let such a term be
    appended to ``weighted`` a second time and written back to the photo as a literal duplicate.
    """

    subject: list[str]
    weighted: list[str]
    _subject_seen: set[str] = field(init=False)
    _weighted_seen: set[str] = field(init=False)

    def __post_init__(self) -> None:
        """Index what the photo already carries, so an existing term is never re-appended."""
        self._subject_seen = {keyword.casefold() for keyword in self.subject}
        self._weighted_seen = {keyword.casefold() for keyword in self.weighted}

    def add(self, keyword: str) -> bool:
        """Append *keyword* to whichever view lacks it; report whether it is new to ``subject``."""
        key = keyword.casefold()
        if key not in self._weighted_seen:
            self._weighted_seen.add(key)
            self.weighted.append(keyword)
        if key in self._subject_seen:
            return False
        self._subject_seen.add(key)
        self.subject.append(keyword)
        return True


def _process_new_keywords(
    new_keywords: list[str],
    flat: _FlatKeywords,
    chain_registry: dict[str, list[str]],
    *,
    verbatim: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Append new flat keywords and update the longest-chain registry.

    Mutates: flat, chain_registry.

    Args:
        new_keywords: Flat subjects (e.g., "bird") or chains (e.g., "Duck<Bird<Animal").
        flat: The subject/weighted accumulators, each de-duplicated against its own contents.
        chain_registry: Maps casefolded leaf to longest observed chain (root-to-leaf list).
        verbatim: Exact spellings a controlled vocabulary declared, keyed by casefolded term.

    Returns:
        Subjects appended during this call, in append order.
    """
    added_subjects: list[str] = []
    for keyword in new_keywords:
        _, parts = parse_hierarchical_keyword(keyword)
        normalized = _normalize_chain_parts(parts, verbatim)
        if not normalized:
            continue
        added_subjects.extend(flat_kw for flat_kw in normalized if flat.add(flat_kw))
        _register_chain(chain_registry, normalized)
    return added_subjects


def _collect_cumulative_entries(
    chain_registry: dict[str, list[str]],
    hierarchical_seen: set[str],
) -> list[str]:
    """
    Generate Lightroom hierarchy paths from canonical chains.

    Lightroom writes hierarchical keywords as full pipe-separated paths in lr:HierarchicalSubject.
    You cannot add only "Animal|Bird|Duck"; you must also add "Animal|Bird".

    Mutates: hierarchical_seen.

    Args:
        chain_registry: Maps each leaf keyword to its full root-to-leaf path.
        hierarchical_seen: Casefolded set for de-duplicating cumulative paths.

    Returns:
        New cumulative paths like "A|B", "A|B|C", starting at MIN_HIERARCHICAL_DEPTH.

    Examples:
        >>> _collect_cumulative_entries({"duck": ["Animal", "Bird", "Duck"]}, set())
        ['Animal|Bird', 'Animal|Bird|Duck']
    """
    additions: list[str] = []
    for canonical_chain in chain_registry.values():
        for depth in range(MIN_HIERARCHICAL_DEPTH, len(canonical_chain) + 1):
            cumulative = "|".join(canonical_chain[:depth])
            key = cumulative.casefold()
            if key in hierarchical_seen:
                continue
            hierarchical_seen.add(key)
            additions.append(cumulative)
    return additions


def merge_keywords(
    existing_kw: KeywordSet,
    new_keywords: list[str],
    *,
    verbatim: Mapping[str, str] | None = None,
) -> KeywordSet:
    """
    Merge new AI-generated keywords with existing keywords, preserving hierarchy.

    Args:
        existing_kw: Existing keywords read off the photo (via read_image_context).
        new_keywords: List of new keywords from AI (may include hierarchical format).
        verbatim: Exact spellings to keep as-is, keyed by casefolded term. A controlled
            vocabulary passes its own index here so the catalog's capitalization survives.

    Returns:
        A new :class:`KeywordSet` with merged views:
        - ``subject``: all flat keywords (existing + new flattened)
        - ``hierarchical``: hierarchical keywords (existing + new hierarchical)
        - ``weighted``: weighted flat keywords, de-duplicated on its own (usually the subject terms)

    Note:
        Duplicate detection is case-insensitive (using casefold). Hierarchical keywords
        are flattened for Subject/WeightedFlatSubject; original hierarchy is preserved
        in HierarchicalSubject. The *existing_kw* argument is never mutated.

    Examples:
        >>> merge_keywords(
        ...     KeywordSet(
        ...         subject=["Beach"],
        ...         hierarchical=["Animal|Bird"],
        ...         weighted=["Beach"],
        ...     ),
        ...     ["Seagull<Bird<Animal", "bird"],
        ... ).hierarchical
        ['Animal|Bird', 'Animal|Bird|Seagull']
    """
    # Copy caller-owned lists so this stays a pure function from the caller's perspective.
    flat = _FlatKeywords(subject=list(existing_kw.subject), weighted=list(existing_kw.weighted))
    existing_hierarchical = [kw for kw in existing_kw.hierarchical if "|" in kw]
    hierarchical_seen = {kw.casefold() for kw in existing_hierarchical}

    chain_registry = _seed_longest_from_existing(existing_hierarchical)
    new_subjects = _process_new_keywords(
        new_keywords,
        flat,
        chain_registry,
        verbatim=verbatim,
    )
    new_hierarchical = _collect_cumulative_entries(chain_registry, hierarchical_seen)

    merged = KeywordSet(
        subject=flat.subject,
        hierarchical=existing_hierarchical + new_hierarchical,
        weighted=flat.weighted,
    )

    logger.debug(
        "keywords_merged",
        new_flat_count=len(new_subjects),
        new_hierarchical_count=len(new_hierarchical),
        total_flat=len(merged.subject),
        total_hierarchical=len(merged.hierarchical),
    )

    return merged
