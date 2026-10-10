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


def _clean_items(value: object, *, max_items: int, max_chars: int) -> object:
    """
    Truncate the list to *max_items* first, then drop blank strings and clip over-long ones.

    Runs in a ``mode="before"`` validator so the per-item ``min_length``/``max_length`` constraints
    (which the model sees in the JSON schema, and which validate *before* any after-mode validator)
    can no longer fail on a rambling item. A failure there makes pydantic-ai retry the full vision
    call up to ``retries`` times, and the model rarely self-corrects; cleaning is cheaper than
    burning that inference budget. Non-list values and non-string items pass through untouched so
    pydantic still reports real schema violations.

    The list is sliced to *max_items* before any per-item work, not after: a model stuck in a
    repetition loop (a known failure mode this project already guards against elsewhere via
    ``frequency_penalty``) can return far more than *max_items* entries, and processing every one of
    them first would let the advertised cap bound the *output* size without bounding the CPU and
    memory cost of getting there.
    """
    if not isinstance(value, list):
        return value
    cleaned: list[object] = []
    for item in value[:max_items]:
        if isinstance(item, str):
            stripped = item.strip()[:max_chars]
            if not stripped:
                continue
            cleaned.append(stripped)
        else:
            cleaned.append(item)
    return cleaned


class GeneratedMetadata(BaseModel):
    """Schema returned by the vision-language model."""

    title: str = Field(min_length=1, max_length=_MAX_TITLE_CHARS)
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION_CHARS)
    # The caps are declared on the fields, not only enforced by the validators below, so the JSON
    # schema the model is handed carries a maxItems. Nothing on the serving side is obliged to
    # honor it (LM Studio does not constrain tool-call arguments to it), but a schema that states
    # the limit it enforces is the one the model has a chance of following.
    keywords: list[_Keyword] = Field(default_factory=list, max_length=_MAX_KEYWORDS)
    # A dedicated field so structured output gives the model an explicit place for taxonomy.
    # Each entry is specific-to-general with '<' separators, e.g. "Golden Eagle<Bird of Prey".
    # Without this field the model, constrained to the schema, just emits flat keywords and the
    # hierarchy is lost.
    hierarchies: list[_Hierarchy] = Field(default_factory=list, max_length=_MAX_HIERARCHIES)

    @field_validator("keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, v: object) -> object:
        """Clean model keyword output instead of failing validation; see _clean_items."""
        return _clean_items(v, max_items=_MAX_KEYWORDS, max_chars=_MAX_KEYWORD_CHARS)

    @field_validator("hierarchies", mode="before")
    @classmethod
    def _clean_hierarchies(cls, v: object) -> object:
        """Clean model hierarchy output, mirroring the keyword cleaning."""
        return _clean_items(v, max_items=_MAX_HIERARCHIES, max_chars=_MAX_HIERARCHY_CHARS)


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
