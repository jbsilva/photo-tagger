"""Pydantic schemas shared by the AI layer and the rest of the pipeline."""

from typing import Annotated

from pydantic import BaseModel, Field


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
    keywords: list[_Keyword] = Field(default_factory=list, max_length=_MAX_KEYWORDS)
