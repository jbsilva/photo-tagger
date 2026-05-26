"""Pydantic schemas shared by the AI layer and the rest of the pipeline."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# Hard caps on AI output. Pydantic enforces these and pydantic-ai will retry the model when
# they trip, so a runaway response (a 500-word "title", 80 keywords) cannot pollute the photo's
# metadata. Bounds are generous enough to absorb prompt drift without rejecting good outputs.
_MAX_TITLE_CHARS = 120
_MAX_DESCRIPTION_CHARS = 600
_MAX_KEYWORDS = 30
_MAX_KEYWORD_CHARS = 80

_Keyword = Annotated[str, Field(min_length=1, max_length=_MAX_KEYWORD_CHARS)]


class GeneratedMetadata(BaseModel):
    """Schema returned by the vision-language model."""

    title: str = Field(min_length=1, max_length=_MAX_TITLE_CHARS)
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION_CHARS)
    keywords: list[_Keyword] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def _cap_keywords(cls, v: list[str]) -> list[str]:
        """
        Silently truncate over-long keyword lists instead of failing validation.

        Thinking models (e.g. Qwen3) occasionally overshoot the requested cap by a few items.
        A hard ``max_length`` on the field makes pydantic-ai retry up to 5 times (and the model
        rarely self-corrects). Truncating here avoids wasting inference budget on retries that will
        all fail.
        """
        return v[:_MAX_KEYWORDS]


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
