"""
Hierarchical-keyword parsing and merging logic.

Lightroom expresses hierarchical keywords as pipe-separated paths ("Animal|Bird|Duck"). The model
emits the inverse, leaf-first form ("Duck<Bird<Animal"). This module converts between the two and
merges fresh AI keywords with whatever already lives on the photo.
"""

from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.config import MIN_HIERARCHICAL_DEPTH
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from collections.abc import Iterable


def parse_hierarchical_keyword(keyword: str) -> tuple[str, list[str]]:
    """
    Parse a hierarchical keyword in format 'Child<Parent<Grandparent' to Lightroom format.

    Args:
        keyword: Keyword string, either flat or hierarchical with '<' separators.
                 Example: "Duck<Bird<Animal" or just "Landscape"

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

    # The model occasionally emits stray '>' characters; drop them before parsing.
    sanitized = keyword.replace(">", "")
    if "<" not in sanitized:
        return (sanitized, [sanitized])

    parts = [p.strip() for p in sanitized.split("<") if p.strip()]
    if not parts:
        return ("", [])

    # Reverse so we emit Lightroom's root-to-leaf order.
    parts_reversed = list(reversed(parts))
    return ("|".join(parts_reversed), parts_reversed)


def _normalize_chain_parts(parts: Iterable[str]) -> list[str]:
    """
    Return Title Case chain segments, skipping blanks.

    Examples:
        >>> _normalize_chain_parts([" duck ", "Bird", ""])
        ['Duck', 'Bird']
    """
    return [segment.strip().title() for segment in parts if segment and segment.strip()]


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


def _process_new_keywords(
    new_keywords: list[str],
    subject_seen: set[str],
    subject_acc: list[str],
    weighted_acc: list[str],
    chain_registry: dict[str, list[str]],
) -> list[str]:
    """
    Append new flat keywords and update the longest-chain registry.

    Mutates: subject_seen, subject_acc, weighted_acc, chain_registry.

    Args:
        new_keywords: Flat subjects (e.g., "bird") or chains (e.g., "Duck<Bird<Animal").
        subject_seen: Casefolded set for de-duplication.
        subject_acc: Accumulates unique subjects (root-to-leaf order).
        weighted_acc: Parallel accumulator kept in sync with subject_acc.
        chain_registry: Maps casefolded leaf to longest observed chain (root-to-leaf list).

    Returns:
        Subjects appended during this call, in append order.
    """
    added_subjects: list[str] = []
    for keyword in new_keywords:
        _, parts = parse_hierarchical_keyword(keyword)
        normalized = _normalize_chain_parts(parts)
        if not normalized:
            continue
        for flat_kw in normalized:
            key = flat_kw.casefold()
            if key in subject_seen:
                continue
            subject_acc.append(flat_kw)
            weighted_acc.append(flat_kw)
            added_subjects.append(flat_kw)
            subject_seen.add(key)
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
        >>> collect_cumulative_entries({"duck": ["Animal", "Bird", "Duck"]}, set())
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
) -> KeywordSet:
    """
    Merge new AI-generated keywords with existing keywords, preserving hierarchy.

    Args:
        existing_kw: Existing keywords read off the photo (from read_existing_keywords).
        new_keywords: List of new keywords from AI (may include hierarchical format).

    Returns:
        A new :class:`KeywordSet` with merged views:
        - ``subject``: all flat keywords (existing + new flattened)
        - ``hierarchical``: hierarchical keywords (existing + new hierarchical)
        - ``weighted``: weighted flat keywords (mirrors subject)

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
    existing_subject = list(existing_kw.subject)
    existing_weighted = list(existing_kw.weighted)
    existing_hierarchical = [kw for kw in existing_kw.hierarchical if "|" in kw]

    subject_seen = {kw.casefold() for kw in existing_subject}
    hierarchical_seen = {kw.casefold() for kw in existing_hierarchical}

    chain_registry = _seed_longest_from_existing(existing_hierarchical)
    new_subjects = _process_new_keywords(
        new_keywords,
        subject_seen,
        existing_subject,
        existing_weighted,
        chain_registry,
    )
    new_hierarchical = _collect_cumulative_entries(chain_registry, hierarchical_seen)

    merged = KeywordSet(
        subject=existing_subject,
        hierarchical=existing_hierarchical + new_hierarchical,
        weighted=existing_weighted,
    )

    logger.debug(
        "keywords_merged",
        new_flat_count=len(new_subjects),
        new_hierarchical_count=len(new_hierarchical),
        total_flat=len(merged.subject),
        total_hierarchical=len(merged.hierarchical),
    )

    return merged
