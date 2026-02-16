"""Tests covering AI analysis helpers using LiteLLM mocks."""

import json
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, ModelSettings

from photo_tagger.main import GeneratedMetadata, analyze_image_with_ai


class LiteLLMAgentStub:
    """Minimal agent stub that delegates to LiteLLM's mock completion helper."""

    def __init__(self, payload: str, *, model: str = "gpt-4o-mini") -> None:
        """Store the canned payload and model name used for mock completions."""
        self._payload = payload
        self._model = model
        self.calls: list[dict[str, Any]] = []

    def run_sync(
        self,
        items: list[object],
        model_settings: ModelSettings,
        output_type: type[GeneratedMetadata],
    ) -> SimpleNamespace:
        """Mimic Agent.run_sync by validating LiteLLM mock output."""
        self.calls.append(
            {
                "items": items,
                "temperature": model_settings.get("temperature"),
                "max_tokens": model_settings.get("max_tokens"),
            },
        )

        response = litellm.mock_completion(
            model=self._model,
            messages=[{"role": "user", "content": "stub"}],
            mock_response=self._payload,
        )
        content = response.choices[0].message["content"]  # type: ignore[union-attr]
        metadata = output_type.model_validate_json(content)
        return SimpleNamespace(output=metadata)


def test_analyze_image_with_ai_parses_litellm_payload() -> None:
    """LiteLLM mock responses are parsed into GeneratedMetadata and returned."""
    payload = json.dumps(
        {
            "title": "Forest Companions",
            "description": "Two marmosets share a quiet branch in the canopy.",
            "keywords": ["Animal", "Forest", "Primate"],
        },
    )
    agent = LiteLLMAgentStub(payload)
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
    assert recorded["temperature"] == target_temperature
    assert recorded["max_tokens"] == target_max_tokens


def test_analyze_image_with_ai_invalid_litellm_payload_raises() -> None:
    """Invalid LiteLLM output bubbles up as a validation error."""
    agent = LiteLLMAgentStub("not-json")
    image_bytes = BinaryContent(data=b"\xff\xd8stubjpeg", media_type="image/jpeg")

    with pytest.raises(ValidationError):
        analyze_image_with_ai(image_bytes, agent)  # type: ignore[arg-type]
