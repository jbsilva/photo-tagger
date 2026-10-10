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
        usage_limits: Any = None,  # noqa: ANN401, ARG002 - accepted as the real agent does.
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


# The two shapes a run that outgrew its output budget arrives in; see _is_token_limit_failure.
_LENGTH_MESSAGE = (
    "Model token limit (1200) exceeded before any response was generated. Increase the "
    "`max_tokens` model setting, or simplify the prompt to result in a shorter response."
)


class _LoopingAgent(_StubAgent):
    """Runs out of output budget *failures* times, then answers normally."""

    def __init__(self, output: Any, failures: int, error: BaseException) -> None:  # noqa: ANN401
        super().__init__(output)
        self.failures = failures
        self.error = error
        self.attempts = 0

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        self.attempts += 1
        if self.attempts <= self.failures:
            usage = kwargs.get("usage")
            if usage is not None:
                # Every discarded attempt still burned its whole budget.
                usage.output_tokens = getattr(usage, "output_tokens", 0) + 1200
            raise self.error
        return super().run_sync(*args, **kwargs)


def _metadata() -> GeneratedMetadata:
    """Build a plain, valid model answer."""
    return GeneratedMetadata(
        title="A Title",
        description="A description.",
        keywords=["Duck"],
        hierarchies=["Duck<Bird<Animal"],
    )


@pytest.mark.parametrize(
    "error",
    [
        pydantic_ai.exceptions.UnexpectedModelBehavior(_LENGTH_MESSAGE),
        pydantic_ai.exceptions.IncompleteToolCall("Model token limit (1200) exceeded while ..."),
    ],
)
def test_analyze_image_retries_a_run_that_hit_the_token_limit(error: BaseException) -> None:
    """
    A repetition loop is a sampling accident, so the photo is simply asked again.

    pydantic-ai refuses to retry this one itself, which left every loop as a failed photo for the
    user to spot and retry by hand.
    """
    agent = _LoopingAgent(_metadata(), failures=1, error=error)

    result = ai_module.analyze_image_with_ai(image_bytes=_stub_image(), agent=agent)  # type: ignore[arg-type]

    assert agent.attempts == 2  # noqa: PLR2004 - one loop, then the answer.
    assert result.title == "A Title"


def test_analyze_image_gives_up_after_the_retry_budget() -> None:
    """A photo that fails every time still reports the failure rather than looping forever."""
    error = pydantic_ai.exceptions.UnexpectedModelBehavior(_LENGTH_MESSAGE)
    agent = _LoopingAgent(_metadata(), failures=99, error=error)
    image = _stub_image()

    with pytest.raises(pydantic_ai.exceptions.UnexpectedModelBehavior):
        ai_module.analyze_image_with_ai(image_bytes=image, agent=agent)  # type: ignore[arg-type]

    assert agent.attempts == ai_module._LENGTH_RETRIES + 1  # noqa: SLF001 - the budget under test.


def test_analyze_image_counts_the_tokens_a_retried_loop_burned() -> None:
    """The discarded attempt's tokens reach the summary; they were spent either way."""
    error = pydantic_ai.exceptions.UnexpectedModelBehavior(_LENGTH_MESSAGE)
    agent = _LoopingAgent(_metadata(), failures=99, error=error)
    image = _stub_image()

    with pytest.raises(pydantic_ai.exceptions.UnexpectedModelBehavior) as exc_info:
        ai_module.analyze_image_with_ai(image_bytes=image, agent=agent)  # type: ignore[arg-type]

    partial = ai_module.partial_usage_from(exc_info.value)
    assert partial is not None
    # Three attempts, each burning the full budget.
    assert partial[1] == 3 * 1200


def test_analyze_image_does_not_retry_an_unrelated_failure() -> None:
    """Only the token-limit shape is retried; everything else fails on the first attempt."""
    agent = _LoopingAgent(_metadata(), failures=1, error=ValueError("connection reset"))
    image = _stub_image()

    with pytest.raises(ValueError, match="connection reset"):
        ai_module.analyze_image_with_ai(image_bytes=image, agent=agent)  # type: ignore[arg-type]

    assert agent.attempts == 1


def test_requests_per_photo_counts_retries_and_length_attempts() -> None:
    """Five retries per attempt, three attempts: eighteen requests at most."""
    assert ai_module.requests_per_photo(5) == 18  # noqa: PLR2004 - (5 + 1) x (2 + 1)


def test_a_run_is_not_capped_by_pydantic_ai_request_limit() -> None:
    """The attempts share one usage counter, so the default cap of 50 must not stop a photo."""
    assert ai_module._NO_REQUEST_CAP.request_limit is None  # noqa: SLF001 - the setting under test.
