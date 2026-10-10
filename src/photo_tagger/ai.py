"""Vision-language agent setup and inference helpers."""

import contextlib
import time
from typing import TYPE_CHECKING

import pydantic_ai
from loguru import logger
from pydantic_ai import Agent, AgentRunResult, ModelSettings
from pydantic_ai.exceptions import IncompleteToolCall, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.usage import RunUsage, UsageLimits

from photo_tagger.config import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_PROMPT,
    build_system_prompt,
)
from photo_tagger.errors import ProviderError
from photo_tagger.keywords import dedupe_keywords
from photo_tagger.models import GeneratedMetadata, InferenceResult
from photo_tagger.providers import ProviderName, get_backend


if TYPE_CHECKING:
    from pydantic_ai import BinaryContent


# pydantic-ai greets a first run with a multi-line banner advertising observability. photo-tagger
# owns its output: that banner lands on stderr in the middle of a batch, between the log lines, and
# in the terminal behind the GUI window, none of which asked for it. Set here rather
# than through the PYDANTIC_AI_NO_BANNER environment variable so it holds for every way the app is
# started, and in this module because every path that builds an Agent imports it.
pydantic_ai.BANNER_ENABLED = False


def build_chat_model(
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
) -> OpenAIChatModel:
    """
    Resolve a backend, check the model is served, and return the chat model to build an Agent on.

    Split out of :func:`create_agent` because the provider plumbing (URL defaults, key resolution,
    the served-model check) is the same whatever the agent is for. The vocabulary organizer builds a
    text-only agent with a different output schema on top of this.
    """
    backend = get_backend(provider_name)
    resolved_url = api_base_url or backend.default_base_url
    if api_base_url is None:
        logger.debug("using_default_provider_url", url=resolved_url)
    logger.info(
        "provider_config_resolved",
        provider=provider_name,
        url=resolved_url,
        model=model_name,
    )

    resolved_api_key = backend.resolve_api_key(api_key)
    if backend.requires_api_key and not resolved_api_key:
        msg = (
            f"Provider {provider_name!r} requires an API key. Set OPENAI_API_KEY, or give a key "
            "(--api-key, or the window's API key field)."
        )
        logger.error("provider_api_key_required", provider=provider_name)
        raise ProviderError(msg)

    backend.validate_model(resolved_url, model_name, resolved_api_key)
    provider = backend.build_provider(resolved_url, resolved_api_key)
    return OpenAIChatModel(model_name=model_name, provider=provider)


def create_agent(  # noqa: PLR0913 - each kwarg is an independent provider/agent knob
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
    retries: int,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> Agent[None, GeneratedMetadata]:
    """
    Build a configured pydantic-ai Agent backed by the requested provider.

    *output_language* is the language the system prompt asks for in every generated field (title,
    description, keywords, hierarchy segments).
    """
    chat_model = build_chat_model(
        provider_name,
        model_name,
        api_base_url=api_base_url,
        api_key=api_key,
    )
    logger.debug("agent_output_language", output_language=output_language)
    # pydantic-ai's Agent constructor does not propagate `output_type` into its
    # generic, so static analyzers see `Agent[None, str]` while the runtime
    # object actually decodes `GeneratedMetadata`. The two suppressions below
    # cover zuban's view of the constructor arg and pycroscope's view of the
    # return type respectively.
    return Agent(  # static analysis: ignore[incompatible_return_value]
        chat_model,
        output_type=GeneratedMetadata,  # type: ignore[arg-type]
        retries=retries,
        system_prompt=build_system_prompt(output_language),
    )


def _extract_usage(usage: object | None) -> tuple[int, int, int]:
    """Pull (input, output, total) token counts off a pydantic-ai RunUsage if present."""
    if usage is None:
        return (0, 0, 0)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        int(getattr(usage, "total_tokens", 0) or 0),
    )


# Attribute name used to smuggle partial usage onto an exception (see _attach_partial_usage).
# Namespaced to avoid ever colliding with an attribute the exception's own class defines.
_PARTIAL_USAGE_ATTR = "_photo_tagger_partial_usage"


def _attach_partial_usage(exc: BaseException, usage: RunUsage) -> None:
    """
    Best-effort: stash *usage*'s token counts on *exc* before it propagates.

    pydantic-ai retries invalid structured output internally (up to the agent's configured retry
    count) before giving up; each attempt is a real, billed request even though the run as a whole
    raises. ``result.usage`` is unavailable on a raised run (there is no ``result``), so this is the
    only way the caller can still count those tokens instead of silently losing them from the batch
    summary.
    """
    # A handful of exception types use __slots__ and reject new attributes. Losing the partial
    # count in that rare case is strictly better than crashing the failure path over it.
    with contextlib.suppress(AttributeError):
        exc._photo_tagger_partial_usage = _extract_usage(usage)  # type: ignore[attr-defined]  # noqa: SLF001


def partial_usage_from(exc: BaseException) -> tuple[int, int, int] | None:
    """Return the (input, output, total) tokens a failed analyze_image_with_ai call still burned."""
    return getattr(exc, _PARTIAL_USAGE_ATTR, None)


# How often a run that died on the token limit is simply asked again. The cap is only ever
# reached by a model that fell into a repetition loop (an endless "Animal<Animal" chain in the
# hierarchies field is the shape that shows up), which is a sampling accident: the very next
# attempt on the same photo almost always answers normally. pydantic-ai deliberately does not
# retry this one ("Don't retry if the token limit was exceeded, possibly during thinking"), so
# without this every loop costs the user a failed photo to notice and retry by hand.
_LENGTH_RETRIES = 2

# The attempts share one usage counter, so pydantic-ai's default cap of 50 requests per run could
# stop a photo whose own retries are well within bounds (3 attempts x 17 tries at --retries 16).
# The retries and the attempts already bound the calls, so the cap is off.
_NO_REQUEST_CAP = UsageLimits(request_limit=None)


def requests_per_photo(retries: int) -> int:
    """
    Return the most model requests one photo can take.

    Each attempt retries a bad answer *retries* times, and a photo that ran past the token limit
    gets :data:`_LENGTH_RETRIES` more attempts.
    """
    return (retries + 1) * (_LENGTH_RETRIES + 1)


def _is_token_limit_failure(exc: BaseException) -> bool:
    """
    Say whether *exc* is the model running out of output budget mid-answer.

    Two shapes reach here. A cut-off tool call raises pydantic-ai's own ``IncompleteToolCall``. A
    server that drops the unparsable call before answering (LM Studio does) leaves nothing to point
    at, and pydantic-ai raises a plain ``UnexpectedModelBehavior`` whose message is the only thing
    distinguishing it, hence the text match. A miss costs nothing: the failure is raised as it is
    today.
    """
    if isinstance(exc, IncompleteToolCall):
        return True
    return isinstance(exc, UnexpectedModelBehavior) and "token limit" in str(exc)


def _run_with_length_retries(
    agent: Agent[None, GeneratedMetadata],
    message: list[str | BinaryContent],
    *,
    settings: ModelSettings,
    usage: RunUsage,
) -> AgentRunResult[GeneratedMetadata]:
    """
    Run the agent, asking again when the answer ran past the token limit.

    Every attempt shares one :class:`RunUsage`, so the tokens a discarded answer burned still reach
    the batch summary instead of vanishing with it.
    """
    for attempt in range(_LENGTH_RETRIES + 1):
        try:
            return agent.run_sync(
                message,
                model_settings=settings,
                output_type=GeneratedMetadata,
                usage=usage,
                usage_limits=_NO_REQUEST_CAP,
            )
        except Exception as exc:
            if attempt >= _LENGTH_RETRIES or not _is_token_limit_failure(exc):
                raise
            logger.warning(
                "inference_token_limit_retry",
                attempt=attempt + 1,
                max_tokens=settings.get("max_tokens"),
                error=str(exc),
            )
    # Unreachable: the loop either returns or raises on its last attempt.
    raise AssertionError


def analyze_image_with_ai(  # noqa: PLR0913 - each kwarg is a distinct sampling knob; bundling adds indirection
    image_bytes: BinaryContent,
    agent: Agent[None, GeneratedMetadata],
    *,
    user_prompt: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
) -> InferenceResult:
    """
    Generate a short title, description, and keywords using a vision-language model.

    Returns:
        :class:`InferenceResult` carrying the model output plus per-call token usage and
        wall-clock seconds, so the pipeline can surface aggregate cost in the batch summary.
    """
    logger.info("analyzing_image_with_ai")
    started = time.perf_counter()
    prompt = user_prompt or DEFAULT_USER_PROMPT

    # Passed in (rather than left to default) so tokens from every internal attempt land here,
    # in place, even if the run as a whole raises: see _attach_partial_usage.
    run_usage = RunUsage()
    settings = ModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
        frequency_penalty=frequency_penalty,
    )
    try:
        result = _run_with_length_retries(
            agent,
            [prompt, image_bytes],
            settings=settings,
            usage=run_usage,
        )
    except Exception as exc:
        _attach_partial_usage(exc, run_usage)
        raise
    elapsed = round(time.perf_counter() - started, 3)
    usage_obj = None
    try:
        usage_obj = result.usage
    except AttributeError as exc:
        logger.debug("ai_usage_unavailable", error=str(exc))
    input_tokens, output_tokens, total_tokens = _extract_usage(usage_obj)
    logger.info(
        "ai_inference_completed",
        seconds=elapsed,
        temperature=temperature,
        max_tokens=max_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    logger.debug(
        "ai_generated_metadata",
        title=result.output.title,
        description=result.output.description,
        keywords=result.output.keywords,
        hierarchies=result.output.hierarchies,
    )
    # Fold the dedicated hierarchy chains into the keyword list. Downstream (merge_keywords,
    # the writer, the GUI) already parses the '<' form, so everything else stays unchanged; the
    # chains just need to reach it. Models repeat keywords now and then, so duplicates are
    # collapsed here, before anything shows or counts them.
    keywords = dedupe_keywords([*result.output.keywords, *result.output.hierarchies])
    return InferenceResult(
        title=result.output.title,
        description=result.output.description,
        keywords=keywords,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        seconds=elapsed,
    )
