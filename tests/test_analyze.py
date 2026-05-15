"""Tests covering AI analysis helpers using a lightweight Agent stub."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, ModelSettings

from photo_tagger.main import DEFAULT_USER_PROMPT, GeneratedMetadata, analyze_image_with_ai


class StubAgent:
    """Minimal agent stub that returns a pre-built GeneratedMetadata payload."""

    def __init__(self, payload: str) -> None:
        """Store the canned JSON payload returned on every call."""
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def run_sync(
        self,
        items: list[object],
        model_settings: ModelSettings,
        output_type: type[GeneratedMetadata],
    ) -> SimpleNamespace:
        """Mimic Agent.run_sync by validating the canned payload through the output schema."""
        self.calls.append(
            {
                "items": items,
                "temperature": model_settings.get("temperature"),
                "max_tokens": model_settings.get("max_tokens"),
            },
        )
        metadata = output_type.model_validate_json(self._payload)
        return SimpleNamespace(output=metadata)


def test_analyze_image_with_ai_parses_payload() -> None:
    """Stubbed agent responses are parsed into GeneratedMetadata and returned."""
    payload = json.dumps(
        {
            "title": "Forest Companions",
            "description": "Two marmosets share a quiet branch in the canopy.",
            "keywords": ["Animal", "Forest", "Primate"],
        },
    )
    agent = StubAgent(payload)
    image_bytes = BinaryContent(data=b"\xff\xd8stubjpeg", media_type="image/jpeg")
    target_temperature = 0.42
    target_max_tokens = 88

    title, description, keywords = analyze_image_with_ai(
        image_bytes,
        agent,  # type: ignore[arg-type]
        user_prompt="Describe the photograph",
        temperature=target_temperature,
        max_tokens=target_max_tokens,
    )

    assert title == "Forest Companions"
    assert description == "Two marmosets share a quiet branch in the canopy."
    assert keywords == ["Animal", "Forest", "Primate"]

    assert len(agent.calls) == 1
    recorded = agent.calls[0]
    assert recorded["items"][0] == "Describe the photograph"
    assert isinstance(recorded["items"][1], BinaryContent)
    assert recorded["temperature"] == pytest.approx(target_temperature)
    assert recorded["max_tokens"] == target_max_tokens


def test_analyze_image_with_ai_uses_default_user_prompt() -> None:
    """Omitting user_prompt falls back to DEFAULT_USER_PROMPT."""
    payload = json.dumps({"title": "t", "description": "d", "keywords": []})
    agent = StubAgent(payload)
    image_bytes = BinaryContent(data=b"\xff\xd8stubjpeg", media_type="image/jpeg")

    analyze_image_with_ai(image_bytes, agent)  # type: ignore[arg-type]

    assert agent.calls[0]["items"][0] == DEFAULT_USER_PROMPT


def test_analyze_image_with_ai_invalid_payload_raises() -> None:
    """Invalid agent output bubbles up as a validation error."""
    agent = StubAgent("not-json")
    image_bytes = BinaryContent(data=b"\xff\xd8stubjpeg", media_type="image/jpeg")

    with pytest.raises(ValidationError):
        analyze_image_with_ai(image_bytes, agent)  # type: ignore[arg-type]
