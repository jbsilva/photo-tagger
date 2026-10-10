"""Tests for AI agent wiring that don't require a live model."""

import dataclasses
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx2
import pydantic_ai
import pytest

from photo_tagger import ai as ai_module
from photo_tagger.config import DEFAULT_USER_PROMPT
from photo_tagger.errors import ProviderError
from photo_tagger.models import GeneratedMetadata
from photo_tagger.providers import get_backend


class _DummyResponse:
    def __init__(self, status_code: int, payload: Any) -> None:  # noqa: ANN401
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:  # noqa: ANN401
        return self._payload

    @property
    def text(self) -> str:
        return repr(self._payload)


def _patch_listing(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> list[tuple[str, dict[str, str]]]:
    """Make every model-listing request return *payload* with a 200, recording (url, headers)."""
    requests: list[tuple[str, dict[str, str]]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> _DummyResponse:
        requests.append((url, headers))
        return _DummyResponse(HTTPStatus.OK, payload)

    monkeypatch.setattr(httpx2, "get", fake_get)
    return requests


def test_importing_ai_disables_the_pydantic_ai_banner() -> None:
    """Importing the agent module is enough to keep pydantic-ai's first-run banner off stderr."""
    assert pydantic_ai.BANNER_ENABLED is False


# ---------------------------------------------------------------------------
# create_agent wiring
# ---------------------------------------------------------------------------


def test_create_agent_ollama_validates_and_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('ollama') validates against Ollama's native /api/tags listing."""
    requests = _patch_listing(monkeypatch, {"models": [{"name": "test-model"}]})
    agent = ai_module.create_agent(
        "ollama",
        "test-model",
        api_base_url="http://localhost:11434/v1",
        api_key=None,
        retries=2,
    )
    assert agent is not None
    assert [url for url, _ in requests] == ["http://localhost:11434/api/tags"]


def test_create_agent_lmstudio_validates_and_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('lmstudio') validates against the OpenAI-style /v1/models listing."""
    requests = _patch_listing(monkeypatch, {"data": [{"id": "test-model"}]})
    agent = ai_module.create_agent(
        "lmstudio",
        "test-model",
        api_base_url="http://localhost:1234/v1",
        api_key=None,
        retries=1,
    )
    assert agent is not None
    assert [url for url, _ in requests] == ["http://localhost:1234/v1/models"]


def test_create_agent_system_prompt_follows_output_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output language lands in the system prompt handed to the Agent."""
    _patch_listing(monkeypatch, {"data": [{"id": "test-model"}]})
    captured: dict[str, Any] = {}
    # The real class comes from pydantic_ai; ai.py's import of it is not a re-export.
    from pydantic_ai import Agent  # noqa: PLC0415

    def spying_agent(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        captured.update(kwargs)
        return Agent(*args, **kwargs)

    monkeypatch.setattr(ai_module, "Agent", spying_agent)
    ai_module.create_agent(
        "lmstudio",
        "test-model",
        api_base_url="http://localhost:1234/v1",
        api_key=None,
        retries=1,
        output_language="German",
    )
    assert "German" in captured["system_prompt"]
    assert "English" not in captured["system_prompt"]


def test_create_agent_openai_validates_with_supplied_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_agent('openai') sends the supplied key as a bearer token to the listing."""
    requests = _patch_listing(monkeypatch, {"data": [{"id": "gpt-4o-mini"}]})
    agent = ai_module.create_agent(
        "openai",
        "gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        api_key="sk-test",
        retries=1,
    )
    assert agent is not None
    url, headers = requests[0]
    assert url == "https://api.openai.com/v1/models"
    assert headers.get("Authorization") == "Bearer sk-test"


def test_create_agent_openai_without_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hosted OpenAI backend refuses to run without a key, before any network call."""

    def explode(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        msg = "httpx2.get must not be called when the key is missing"
        raise AssertionError(msg)

    monkeypatch.setattr(httpx2, "get", explode)
    # The default key is captured from the environment at import time, so build a
    # keyless clone (frozen dataclasses copy via dataclasses.replace) and route the
    # lookup to it regardless of what OPENAI_API_KEY happens to be on this machine.
    keyless = dataclasses.replace(get_backend("openai"), default_api_key=None)
    monkeypatch.setattr(ai_module, "get_backend", lambda _name: keyless)
    with pytest.raises(ProviderError):
        ai_module.create_agent(
            "openai",
            "gpt-4o-mini",
            api_base_url=None,
            api_key=None,
            retries=1,
        )


def test_create_agent_uses_default_url_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When api_base_url is None, the backend's default URL is what gets validated against."""
    default_url = get_backend("ollama").default_base_url
    requests = _patch_listing(monkeypatch, {"models": [{"name": "m"}]})
    agent = ai_module.create_agent(
        "ollama",
        "m",
        api_base_url=None,
        api_key=None,
        retries=0,
    )
    assert agent is not None
    assert requests[0][0].startswith(default_url.removesuffix("/v1"))


# ---------------------------------------------------------------------------
# analyze_image_with_ai
# ---------------------------------------------------------------------------


# (input_tokens, output_tokens, total_tokens) the stub result reports.
_STUB_USAGE = (100, 20, 120)


def _stub_image() -> Any:  # noqa: ANN401
    """Return a tiny BinaryContent payload matching analyze_image_with_ai's signature."""
    from pydantic_ai import BinaryContent  # noqa: PLC0415

    return BinaryContent(data=b"\xff\xd8stub", media_type="image/jpeg")


class _StubAgent:
    """Capture run_sync's inputs and return a canned pydantic-ai result shape."""

    def __init__(self, output: Any) -> None:  # noqa: ANN401
        self._output = output
        self.calls: list[dict[str, Any]] = []

    def run_sync(
        self,
        prompt_parts: list[Any],
        *,
        model_settings: Any,  # noqa: ANN401
        output_type: Any,  # noqa: ANN401
        usage: Any = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        self.calls.append(
            {
                "parts": prompt_parts,
                "settings": model_settings,
                "output_type": output_type,
                "usage": usage,
            },
        )
        input_tokens, output_tokens, total_tokens = _STUB_USAGE
        return SimpleNamespace(
            output=self._output,
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
        )


def test_analyze_image_folds_hierarchies_and_dedupes_keywords() -> None:
    """
    The model's hierarchy chains join the keyword list, with case-insensitive repeats collapsed.

    The rest of the pipeline (merging, writing, the GUI) only ever sees InferenceResult.keywords, so
    dropping the fold here would silently lose every hierarchy the model produced.
    """
    output = GeneratedMetadata(
        title="A Title",
        description="A description.",
        keywords=["Duck", "duck", "Sky"],
        hierarchies=["Duck<Bird<Animal"],
    )
    agent = _StubAgent(output)

    result = ai_module.analyze_image_with_ai(
        image_bytes=_stub_image(),
        agent=agent,  # type: ignore[arg-type]
        user_prompt="Describe.",
    )

    usage = (result.input_tokens, result.output_tokens, result.total_tokens)

    assert result.keywords == ["Duck", "Sky", "Duck<Bird<Animal"]
    assert (result.title, result.description) == ("A Title", "A description.")
    assert usage == _STUB_USAGE
    assert result.seconds >= 0.0


def test_analyze_image_forwards_sampling_settings_and_prompt() -> None:
    """Every sampling knob reaches run_sync's ModelSettings; the image rides in the prompt."""
    output = GeneratedMetadata(title="T", description="D", keywords=["K"])
    agent = _StubAgent(output)

    image = _stub_image()
    ai_module.analyze_image_with_ai(
        image_bytes=image,
        agent=agent,  # type: ignore[arg-type]
        user_prompt="Describe this photo.",
        temperature=0.7,
        max_tokens=333,
        timeout_seconds=42.0,
        frequency_penalty=1.5,
    )

    call = agent.calls[0]
    assert call["parts"] == ["Describe this photo.", image]
    forwarded = {
        key: call["settings"][key]
        for key in ("temperature", "max_tokens", "timeout", "frequency_penalty")
    }
    assert forwarded == {
        "temperature": 0.7,
        "max_tokens": 333,
        "timeout": 42.0,
        "frequency_penalty": 1.5,
    }
    assert call["output_type"] is GeneratedMetadata


def test_analyze_image_falls_back_to_the_default_prompt() -> None:
    """An empty user prompt falls back to DEFAULT_USER_PROMPT instead of sending nothing."""
    agent = _StubAgent(GeneratedMetadata(title="T", description="D", keywords=[]))
    ai_module.analyze_image_with_ai(image_bytes=_stub_image(), agent=agent, user_prompt="")  # type: ignore[arg-type]
    assert agent.calls[0]["parts"][0] == DEFAULT_USER_PROMPT


def test_analyze_image_survives_a_result_without_usage() -> None:
    """A result object lacking .usage degrades to zero token counts, not a crash."""

    class _NoUsageAgent(_StubAgent):
        def run_sync(self, _prompt_parts: list[Any], **_kwargs: Any) -> Any:  # noqa: ANN401
            class _Result:
                output = self._output

                @property
                def usage(self) -> Any:  # noqa: ANN401
                    msg = "no usage on this result"
                    raise AttributeError(msg)

            return _Result()

    agent = _NoUsageAgent(GeneratedMetadata(title="T", description="D", keywords=[]))
    result = ai_module.analyze_image_with_ai(
        image_bytes=_stub_image(),
        agent=agent,  # type: ignore[arg-type]
        user_prompt="p",
    )
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (0, 0, 0)


def test_analyze_image_attaches_partial_usage_when_the_call_fails() -> None:
    """
    Tokens burned by attempts pydantic-ai retried internally before giving up are not lost.

    Regression test: analyze_image_with_ai only ever read usage off a successful result, so a
    run that raised after several billed-but-invalid attempts discarded their token cost entirely.
    """

    class _FailingAgent:
        def run_sync(self, *_a: Any, usage: Any, **_kw: Any) -> Any:  # noqa: ANN401
            # Simulates pydantic-ai folding a rejected attempt's usage into the passed-in
            # RunUsage before ultimately giving up and raising.
            usage.input_tokens = 50
            usage.output_tokens = 10
            msg = "model returned invalid structured output"
            raise ValueError(msg)

    image = _stub_image()
    agent = _FailingAgent()

    with pytest.raises(ValueError, match="invalid structured output") as exc_info:
        ai_module.analyze_image_with_ai(
            image_bytes=image,
            agent=agent,  # type: ignore[arg-type]
            user_prompt="Describe.",
        )

    assert ai_module.partial_usage_from(exc_info.value) == (50, 10, 60)


def test_partial_usage_from_returns_none_when_nothing_was_attached() -> None:
    """A plain exception unrelated to analyze_image_with_ai carries no partial usage."""
    assert ai_module.partial_usage_from(ValueError("boom")) is None


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


def test_extract_usage_returns_zeros_for_none() -> None:
    """None usage object returns (0, 0, 0)."""
    assert ai_module._extract_usage(None) == (0, 0, 0)  # noqa: SLF001


def test_extract_usage_reads_attributes() -> None:
    """Usage attributes are converted to ints."""
    usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
    assert ai_module._extract_usage(usage) == (10, 5, 15)  # noqa: SLF001


def test_extract_usage_handles_none_attributes() -> None:
    """None attribute values are coerced to 0."""
    usage = MagicMock(input_tokens=None, output_tokens=None, total_tokens=None)
    assert ai_module._extract_usage(usage) == (0, 0, 0)  # noqa: SLF001
