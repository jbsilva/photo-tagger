"""Schema-level guardrails for the metadata returned by the vision-language model."""

import pytest
from pydantic import ValidationError

from photo_tagger.models import GeneratedMetadata


def test_generated_metadata_accepts_typical_payload() -> None:
    """A well-formed payload validates and roundtrips through the schema unchanged."""
    payload = GeneratedMetadata(
        title="A quiet harbour at dusk",
        description="Two boats sit moored as the sky turns pink behind a stone breakwater.",
        keywords=["Boat", "Sunset", "Harbour"],
    )
    assert payload.title.startswith("A quiet harbour")
    assert len(payload.keywords) == 3  # noqa: PLR2004 - asserting fixture shape


def test_generated_metadata_rejects_blank_title() -> None:
    """An empty title would silently strip the photo's display name; reject it."""
    with pytest.raises(ValidationError):
        GeneratedMetadata(title="", description="ok", keywords=[])


def test_generated_metadata_rejects_long_title() -> None:
    """A 500-char "title" is almost always model drift; cap it so pydantic-ai retries."""
    with pytest.raises(ValidationError):
        GeneratedMetadata(title="x" * 500, description="ok", keywords=[])


def test_generated_metadata_truncates_too_many_keywords() -> None:
    """Runaway keyword lists are silently capped rather than rejected."""
    meta = GeneratedMetadata(
        title="t",
        description="d",
        keywords=[f"kw-{i}" for i in range(50)],
    )
    assert len(meta.keywords) == 30  # noqa: PLR2004 - matches _MAX_KEYWORDS


def test_generated_metadata_rejects_blank_keyword() -> None:
    """Empty strings inside the keyword list are useless and break downstream merging."""
    with pytest.raises(ValidationError):
        GeneratedMetadata(title="t", description="d", keywords=["ok", ""])


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
