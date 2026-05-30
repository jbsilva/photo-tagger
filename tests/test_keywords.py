"""Regression tests for keyword utilities and contextual prompt helpers."""

from typing import TYPE_CHECKING

from photo_tagger.discovery import parse_extensions, resolve_image_files
from photo_tagger.keywords import (
    _collect_cumulative_entries,
    _normalize_chain_parts,
    _process_new_keywords,
    _register_chain,
    merge_keywords,
    parse_hierarchical_keyword,
)
from photo_tagger.metadata import build_contextual_prompt


if TYPE_CHECKING:
    from pathlib import Path


def test_parse_hierarchical_keyword_handles_flat_and_hierarchical() -> None:
    """Flat keywords stay flat; hierarchical chains flip to Lightroom order."""
    flat_hierarchical, flat_parts = parse_hierarchical_keyword("Landscape")
    assert flat_hierarchical == "Landscape"
    assert flat_parts == ["Landscape"]

    nested_hierarchical, nested_parts = parse_hierarchical_keyword("Duck<Bird<Animal")
    assert nested_hierarchical == "Animal|Bird|Duck"
    assert nested_parts == ["Animal", "Bird", "Duck"]


def test_parse_hierarchical_keyword_returns_empty_list_for_blank_input() -> None:
    """Empty or whitespace-only input returns an empty format string and no parts."""
    fmt, parts = parse_hierarchical_keyword("")
    assert fmt == ""
    assert parts == []

    fmt_ws, parts_ws = parse_hierarchical_keyword("   ")
    assert fmt_ws == ""
    assert parts_ws == []


def test_parse_hierarchical_keyword_strips_trailing_gt() -> None:
    """Stray '>' characters in model output are removed before parsing."""
    hierarchical, parts = parse_hierarchical_keyword("Man<Human<Living Being>")
    assert hierarchical == "Living Being|Human|Man"
    assert parts == ["Living Being", "Human", "Man"]


def test_parse_hierarchical_keyword_returns_empty_for_separators_only() -> None:
    """Input that is only chain separators collapses to an empty result."""
    fmt, parts = parse_hierarchical_keyword("<")
    assert fmt == ""
    assert parts == []


def test_register_chain_keeps_existing_chain_of_equal_length() -> None:
    """A new chain no longer than the recorded one for the same leaf does not replace it."""
    registry = {"bird": ["Animal", "Bird"]}
    _register_chain(registry, ["Pet", "Bird"])  # same leaf, equal length
    assert registry["bird"] == ["Animal", "Bird"]


def test_process_new_keywords_skips_empty_and_duplicate_subjects() -> None:
    """Keywords that normalize to nothing are dropped; case-duplicate subjects are not re-added."""
    subjects = ["Bird"]
    weighted = ["Bird"]
    seen = {"bird"}
    added = _process_new_keywords(["<", "bird"], seen, subjects, weighted, {})
    assert added == []
    assert subjects == ["Bird"]


def test_normalize_chain_parts_title_cases_and_omits_blanks() -> None:
    """Whitespace and empty segments are ignored while remaining entries become Title Case."""
    result = _normalize_chain_parts([" duck ", "", "sea-lion", "bird"])
    assert result == ["Duck", "Sea-Lion", "Bird"]


def test_process_new_keywords_updates_subjects_and_registry() -> None:
    """New keywords extend subjects/weighted lists and record the longest hierarchy per leaf."""
    subjects = ["Beach"]
    weighted = ["Beach"]
    seen = {"beach"}
    registry: dict[str, list[str]] = {"eagle": ["Animal", "Bird", "Eagle"]}

    added = _process_new_keywords(
        ["Duck<Bird<Animal", "cloud"],
        seen,
        subjects,
        weighted,
        registry,
    )

    assert added == ["Animal", "Bird", "Duck", "Cloud"]
    assert subjects == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert weighted == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert registry["duck"] == ["Animal", "Bird", "Duck"]


def test_collect_cumulative_entries_skips_duplicates() -> None:
    """Repeated runs do not add duplicate Lightroom hierarchy paths."""
    chains = {"duck": ["Animal", "Bird", "Duck"]}
    seen: set[str] = set()

    first_pass = _collect_cumulative_entries(chains, seen)
    assert first_pass == ["Animal|Bird", "Animal|Bird|Duck"]

    second_pass = _collect_cumulative_entries(chains, seen)
    assert second_pass == []


def test_merge_keywords_preserves_existing_and_adds_hierarchies() -> None:
    """Existing metadata survives and new hierarchical links are appended once."""
    existing = {
        "subject": ["Beach"],
        "hierarchical": ["Duck", "Animal|Bird"],
        "weighted": ["Beach"],
    }

    merged = merge_keywords(existing, ["Duck<Bird<Animal", "cloud"])

    assert merged["subject"] == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert merged["weighted"] == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert merged["hierarchical"] == ["Animal|Bird", "Animal|Bird|Duck"]


def test_build_contextual_prompt_includes_metadata_and_truncates_keywords() -> None:
    """The contextual prompt surfaces existing metadata and shortens long keyword lists."""
    prompt = build_contextual_prompt(
        "Analyze the scene.",
        ["Beach", "Sunset", "Travel", "Landscape", "Vacation"],
        {
            "XMP-photoshop:Country": "Portugal",
            "XMP-photoshop:City": "Lisbon",
        },
        {"position": "38.7 N, 9.1 W"},
        max_prompt_flat_keywords=3,
    )

    assert prompt.startswith("Analyze the scene.")
    assert "- Existing Keywords: Beach, Sunset, Travel, ..." in prompt
    # Both city and country survive; the model gets the full place name.
    assert "- Location: Lisbon, Portugal" in prompt
    assert "- GPS: 38.7 N, 9.1 W" in prompt


def test_parse_extensions_normalizes_input() -> None:
    """Comma-separated extensions are normalized with leading dots preserved."""
    assert parse_extensions("cr3, jpg ,PNG") == {".cr3", ".jpg", ".PNG"}


def test_resolve_image_files_deduplicates_and_preserves_explicit(tmp_path: Path) -> None:
    """Explicit paths stay first and duplicates discovered via directories are filtered out."""
    folder = tmp_path / "images"
    folder.mkdir()

    explicit = folder / "explicit.cr3"
    explicit.write_text("data")
    duplicate = folder / "shared.jpg"
    duplicate.write_text("data")
    extra = folder / "other.jpg"
    extra.write_text("data")

    result = resolve_image_files(
        [explicit, folder],
        ext_set={".cr3", ".jpg"},
        recursive=False,
    )

    explicit_resolved = explicit.resolve()
    assert result[0] == explicit_resolved
    assert set(result) == {explicit_resolved, duplicate.resolve(), extra.resolve()}
    assert result.count(duplicate.resolve()) == 1
