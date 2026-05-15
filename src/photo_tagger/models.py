"""Pydantic schemas shared by the AI layer and the rest of the pipeline."""

from pydantic import BaseModel, Field


class GeneratedMetadata(BaseModel):
    """Schema returned by the vision-language model."""

    title: str
    description: str
    keywords: list[str] = Field(default_factory=list)
