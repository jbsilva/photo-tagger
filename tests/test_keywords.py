"""Regression tests for keyword utilities and contextual prompt helpers."""

from photo_tagger.keywords import (
    _capitalize_segment,
    _collect_cumulative_entries,
    _FlatKeywords,
    _normalize_chain_parts,
    _process_new_keywords,
    _register_chain,
    dedupe_keywords,
    merge_keywords,
    parse_hierarchical_keyword,
)
from photo_tagger.models import KeywordSet
from photo_tagger.prompt_context import build_contextual_prompt


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
    """A trailing stray '>' with nothing after it does not add a spurious empty level."""
    hierarchical, parts = parse_hierarchical_keyword("Man<Human<Living Being>")
    assert hierarchical == "Living Being|Human|Man"
    assert parts == ["Living Being", "Human", "Man"]


def test_parse_hierarchical_keyword_reads_gt_only_chains_as_hierarchy() -> None:
    """A chain the model wrote with '>' instead of '<' still parses leaf-first."""
    hierarchical, parts = parse_hierarchical_keyword("Green Foliage>Plant>Living Being")
    assert hierarchical == "Living Being|Plant|Green Foliage"
    assert parts == ["Living Being", "Plant", "Green Foliage"]


def test_parse_hierarchical_keyword_handles_mixed_separators() -> None:
    """A chain mixing '<' and '>' treats both as separators instead of fusing segments."""
    hierarchical, parts = parse_hierarchical_keyword("Duck<Bird>Animal")
    assert hierarchical == "Animal|Bird|Duck"
    assert parts == ["Animal", "Bird", "Duck"]


def test_dedupe_keywords_collapses_case_insensitive_repeats() -> None:
    """Repeats (any case) and blanks are dropped; first spelling and order win."""
    assert dedupe_keywords(["Perch", "Bird", " perch ", "", "PERCH", "Sky"]) == [
        "Perch",
        "Bird",
        "Sky",
    ]


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
    flat = _FlatKeywords(subject=["Bird"], weighted=["Bird"])
    added = _process_new_keywords(["<", "bird"], flat, {})
    assert added == []
    assert flat.subject == ["Bird"]
    assert flat.weighted == ["Bird"]


def test_normalize_chain_parts_title_cases_and_omits_blanks() -> None:
    """Whitespace and empty segments are ignored while lowercase entries are capitalized."""
    result = _normalize_chain_parts([" duck ", "", "sea-lion", "bird"])
    assert result == ["Duck", "Sea-lion", "Bird"]


def test_normalize_chain_parts_preserves_deliberate_casing() -> None:
    """
    Acronyms and mixed-case names survive; apostrophes do not trigger mid-word uppercase.

    Regression test: ``str.title()`` used to rewrite NYC as Nyc and bird's nest as Bird'S Nest in
    the keywords written to files.
    """
    result = _normalize_chain_parts(["NYC", "iPhone", "bird's nest", "McDonald's"])
    assert result == ["NYC", "iPhone", "Bird's Nest", "McDonald's"]


def test_process_new_keywords_updates_subjects_and_registry() -> None:
    """New keywords extend subjects/weighted lists and record the longest hierarchy per leaf."""
    flat = _FlatKeywords(subject=["Beach"], weighted=["Beach"])
    registry: dict[str, list[str]] = {"eagle": ["Animal", "Bird", "Eagle"]}

    added = _process_new_keywords(["Duck<Bird<Animal", "cloud"], flat, registry)

    assert added == ["Animal", "Bird", "Duck", "Cloud"]
    assert flat.subject == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert flat.weighted == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
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
    existing = KeywordSet(
        subject=["Beach"],
        hierarchical=["Duck", "Animal|Bird"],
        weighted=["Beach"],
    )

    merged = merge_keywords(existing, ["Duck<Bird<Animal", "cloud"])

    assert merged.subject == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert merged.weighted == ["Beach", "Animal", "Bird", "Duck", "Cloud"]
    assert merged.hierarchical == ["Animal|Bird", "Animal|Bird|Duck"]


def test_merge_keywords_dedupes_hierarchies_case_insensitively() -> None:
    """
    An existing lowercase hierarchy suppresses the same path in canonical casing.

    Guards the casefold in the hierarchical-seen set: without it, files tagged by other tools with
    "animal|bird" would gain a duplicate "Animal|Bird" entry on every run.
    """
    existing = KeywordSet(
        subject=["animal", "bird"],
        hierarchical=["animal|bird"],
        weighted=["animal", "bird"],
    )

    merged = merge_keywords(existing, ["Duck<Bird<Animal"])

    assert merged.hierarchical == ["animal|bird", "Animal|Bird|Duck"]  # no duplicate Animal|Bird


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


def test_merge_keywords_with_all_empty_inputs() -> None:
    """Merging an empty KeywordSet with no new keywords returns an empty KeywordSet."""
    merged = merge_keywords(KeywordSet(), [])
    assert merged == KeywordSet()
    assert merged.is_empty()


def test_merge_keywords_does_not_mutate_input() -> None:
    """The caller's existing keyword lists remain unmodified after merge."""
    existing = KeywordSet(subject=["A"], weighted=["A"])
    original_subject = list(existing.subject)
    merge_keywords(existing, ["B"])
    assert existing.subject == original_subject


def test_merge_keywords_does_not_duplicate_a_weighted_only_keyword() -> None:
    """
    A term on WeightedFlatSubject but not on Subject is not appended to it a second time.

    The two lists come from different tags (XMP-dc:Subject plus IPTC:Keywords for one, XMP-
    lr:WeightedFlatSubject for the other), so they can legitimately differ. Both were de-duplicated
    against the subject list alone, and the writer replayed the duplicate to disk.
    """
    merged = merge_keywords(
        KeywordSet(subject=["Beach"], weighted=["Beach", "Sunset"]),
        ["Sunset"],
    )
    assert merged.subject == ["Beach", "Sunset"]
    assert merged.weighted == ["Beach", "Sunset"]


def test_merge_keywords_keeps_a_vocabulary_spelling_verbatim() -> None:
    """A catalog that writes its keywords in lower case means it, in any script."""
    verbatim = {"gegenlicht": "gegenlicht", "птица": "птица", "животное": "животное"}

    merged = merge_keywords(
        KeywordSet(),
        ["gegenlicht", "птица<животное"],
        verbatim=verbatim,
    )

    assert merged.subject == ["gegenlicht", "животное", "птица"]
    assert merged.hierarchical == ["животное|птица"]


def test_merge_keywords_still_capitalizes_what_the_vocabulary_does_not_cover() -> None:
    """Only declared terms are left alone; the model's own wording is normalized as before."""
    merged = merge_keywords(
        KeywordSet(),
        ["gegenlicht", "golden hour"],
        verbatim={
            "gegenlicht": "gegenlicht",
        },
    )

    assert merged.subject == ["gegenlicht", "Golden Hour"]


def test_capitalize_segment_prefers_the_declared_spelling() -> None:
    """The verbatim map wins over the lowercase-only capitalization rule."""
    assert _capitalize_segment("b&w", {"b&w": "b&w"}) == "b&w"
    # Without it, the lowercase rule applies and str.capitalize() lowercases the rest.
    assert _capitalize_segment("b&w") == "B&w"
