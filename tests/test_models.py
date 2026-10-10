"""Schema-level guardrails for the metadata returned by the vision-language model."""

import pytest
from pydantic import ValidationError

from photo_tagger.models import GeneratedMetadata, KeywordSet


def test_generated_metadata_accepts_typical_payload() -> None:
    """A well-formed payload validates and roundtrips through the schema unchanged."""
    payload = GeneratedMetadata(
        title="A quiet harbour at dusk",
        description="Two boats sit moored as the sky turns pink behind a stone breakwater.",
        keywords=["Boat", "Sunset", "Harbour"],
    )
    assert payload.title.startswith("A quiet harbour")
    assert len(payload.keywords) == 3  # noqa: PLR2004 - asserting fixture shape


@pytest.mark.parametrize("field_name", ["title", "description"])
@pytest.mark.parametrize(
    "bad_value",
    ["", "x" * 2000],
    ids=["blank", "over-long"],
)
def test_generated_metadata_rejects_blank_and_runaway_text_fields(
    field_name: str,
    bad_value: str,
) -> None:
    """Blank or runaway titles/descriptions are model drift; reject so pydantic-ai retries."""
    values: dict[str, object] = {"title": "ok", "description": "ok", field_name: bad_value}
    with pytest.raises(ValidationError):
        GeneratedMetadata(**values)  # type: ignore[arg-type]


def test_generated_metadata_truncates_too_many_keywords() -> None:
    """Runaway keyword lists are silently capped rather than rejected."""
    meta = GeneratedMetadata(
        title="t",
        description="d",
        keywords=[f"kw-{i}" for i in range(50)],
    )
    assert len(meta.keywords) == 30  # noqa: PLR2004 - matches _MAX_KEYWORDS


def test_generated_metadata_drops_blank_keywords() -> None:
    """
    Blank keyword items are dropped, not rejected.

    Regression test: the per-item min_length used to fail the whole validation, making pydantic-ai
    re-run the full vision call up to `retries` times over a single stray empty string.
    """
    meta = GeneratedMetadata(title="t", description="d", keywords=["ok", "", "  "])
    assert meta.keywords == ["ok"]


def test_generated_metadata_clips_over_long_keyword_items() -> None:
    """An over-long keyword is clipped to the cap instead of failing validation."""
    meta = GeneratedMetadata(title="t", description="d", keywords=["x" * 200])
    assert meta.keywords == ["x" * 80]


def test_generated_metadata_bounds_work_before_cleaning_a_runaway_list() -> None:
    """
    The keyword list is sliced to the cap before per-item cleaning, not after.

    Regression test: cleaning used to run over the full raw list before truncating, so a degenerate
    response (a model stuck in a repetition loop, a known failure mode this project already guards
    against elsewhere via frequency_penalty) paid the per-item cost of every extra entry even though
    only _MAX_KEYWORDS of them could ever survive.
    """
    blanks = [""] * 40
    meta = GeneratedMetadata(title="t", description="d", keywords=[*blanks, "late"])
    # Every one of the first _MAX_KEYWORDS (30) raw items is blank, so slicing before cleaning
    # drops all of them and "late" (item 41) is never reached.
    assert meta.keywords == []


def test_generated_metadata_still_rejects_non_string_keywords() -> None:
    """Real schema violations (wrong item type) still fail so pydantic-ai retries."""
    with pytest.raises(ValidationError):
        GeneratedMetadata(title="t", description="d", keywords=["ok", 42])  # type: ignore[list-item]


def test_generated_metadata_accepts_hierarchies() -> None:
    """The dedicated hierarchies field holds specific-to-general '<' chains; default is empty."""
    meta = GeneratedMetadata(
        title="t",
        description="d",
        keywords=["Cat"],
        hierarchies=["Domestic Cat<Cat<Mammal<Animal"],
    )
    assert meta.hierarchies == ["Domestic Cat<Cat<Mammal<Animal"]
    assert GeneratedMetadata(title="t", description="d", keywords=[]).hierarchies == []


def test_generated_metadata_truncates_too_many_hierarchies() -> None:
    """A runaway hierarchy list is capped rather than rejected, like keywords."""
    meta = GeneratedMetadata(
        title="t",
        description="d",
        hierarchies=[f"Leaf{i}<Branch<Root" for i in range(40)],
    )
    assert len(meta.hierarchies) == 20  # noqa: PLR2004 - matches _MAX_HIERARCHIES


@pytest.mark.parametrize(
    ("keyword_set", "empty"),
    [
        (KeywordSet(), True),
        (KeywordSet(subject=["Bird"]), False),
        (KeywordSet(hierarchical=["Animal|Bird"]), False),
        (KeywordSet(weighted=["Bird"]), False),
    ],
    ids=["all-empty", "subject-only", "hierarchical-only", "weighted-only"],
)
def test_keyword_set_is_empty_checks_every_view(
    keyword_set: KeywordSet,
    empty: bool,  # noqa: FBT001 - parametrized expectation, not an API flag.
) -> None:
    """is_empty is False when ANY of the three views holds a keyword, not just subject."""
    assert keyword_set.is_empty() is empty
