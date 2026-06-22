"""Pydantic schemas shared by the AI layer and the rest of the pipeline."""

from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# Hard caps on AI output. Pydantic enforces these and pydantic-ai will retry the model when
# they trip, so a runaway response (a 500-word "title", 80 keywords) cannot pollute the photo's
# metadata. Bounds are generous enough to absorb prompt drift without rejecting good outputs.
_MAX_TITLE_CHARS = 120
_MAX_DESCRIPTION_CHARS = 600
_MAX_KEYWORDS = 30
_MAX_KEYWORD_CHARS = 80
_MAX_HIERARCHIES = 20
# A hierarchy chain ("A<B<C<D<E") is several keywords joined, so it needs more room than a keyword.
_MAX_HIERARCHY_CHARS = 240

_Keyword = Annotated[str, Field(min_length=1, max_length=_MAX_KEYWORD_CHARS)]
_Hierarchy = Annotated[str, Field(min_length=1, max_length=_MAX_HIERARCHY_CHARS)]


class GeneratedMetadata(BaseModel):
    """Schema returned by the vision-language model."""

    title: str = Field(min_length=1, max_length=_MAX_TITLE_CHARS)
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION_CHARS)
    keywords: list[_Keyword] = Field(default_factory=list)
    # A dedicated field so structured output gives the model an explicit place for taxonomy.
    # Each entry is specific-to-general with '<' separators, e.g. "Golden Eagle<Bird of Prey".
    # Without this field the model, constrained to the schema, just emits flat keywords and the
    # hierarchy is lost.
    hierarchies: list[_Hierarchy] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def _cap_keywords(cls, v: list[str]) -> list[str]:
        """
        Silently truncate over-long keyword lists instead of failing validation.

        Thinking models (e.g. Qwen3) occasionally overshoot the requested cap by a few items. A hard
        ``max_length`` on the field makes pydantic-ai retry up to 5 times (and the model rarely
        self-corrects). Truncating here avoids wasting inference budget on retries that will all
        fail.
        """
        return v[:_MAX_KEYWORDS]

    @field_validator("hierarchies")
    @classmethod
    def _cap_hierarchies(cls, v: list[str]) -> list[str]:
        """Truncate an over-long hierarchy list, mirroring the keyword cap."""
        return v[:_MAX_HIERARCHIES]


@dataclass(slots=True)
class KeywordSet:
    """
    The three keyword views Lightroom tracks for a photo.

    Replaces the old ``dict[str, list[str]]`` that was threaded through the reader, the merge logic,
    and the writer keyed by the bare strings ``"subject"`` / ``"hierarchical"`` / ``"weighted"``.
    The typed fields remove the silent-typo failure mode (a misspelled key used to read back an
    empty list) and let the type checker see the shape end to end.

    Fields map to ExifTool tags as follows:
    - ``subject``: flat keywords (XMP-dc:Subject, mirrored to IPTC:Keywords)
    - ``hierarchical``: pipe-separated Lightroom paths (XMP-lr:HierarchicalSubject)
    - ``weighted``: flat keywords mirroring ``subject`` (XMP-lr:WeightedFlatSubject)
    """

    subject: list[str] = field(default_factory=list)
    hierarchical: list[str] = field(default_factory=list)
    weighted: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True when none of the three views holds a keyword."""
        return not (self.subject or self.hierarchical or self.weighted)


@dataclass(slots=True, frozen=True)
class InferenceResult:
    """One vision-language model call with token usage and wall time captured."""

    title: str
    description: str
    keywords: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
