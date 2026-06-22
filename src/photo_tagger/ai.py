"""Vision-language agent setup and inference helpers."""

import time
from typing import TYPE_CHECKING

from loguru import logger
from pydantic_ai import Agent, AgentRunResult, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel

from photo_tagger.config import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_PROMPT,
)
from photo_tagger.errors import ProviderError
from photo_tagger.models import GeneratedMetadata, InferenceResult
from photo_tagger.providers import ProviderName, get_backend


if TYPE_CHECKING:
    from pydantic_ai import BinaryContent


def create_agent(
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
    retries: int,
) -> Agent[None, GeneratedMetadata]:
    """Build a configured pydantic-ai Agent backed by the requested provider."""
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
            f"Provider {provider_name!r} requires an API key. Set OPENAI_API_KEY or pass --api-key."
        )
        logger.error("provider_api_key_required", provider=provider_name)
        raise ProviderError(msg)

    backend.validate_model(resolved_url, model_name, resolved_api_key)
    provider = backend.build_provider(resolved_url, resolved_api_key)
    chat_model = OpenAIChatModel(model_name=model_name, provider=provider)
    # pydantic-ai's Agent constructor does not propagate `output_type` into its
    # generic, so static analyzers see `Agent[None, str]` while the runtime
    # object actually decodes `GeneratedMetadata`. The two suppressions below
    # cover zuban's view of the constructor arg and pycroscope's view of the
    # return type respectively.
    return Agent(  # static analysis: ignore[incompatible_return_value]
        chat_model,
        output_type=GeneratedMetadata,  # type: ignore[arg-type]
        retries=retries,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
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

    result: AgentRunResult[GeneratedMetadata] = agent.run_sync(
        [prompt, image_bytes],
        model_settings=ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            frequency_penalty=frequency_penalty,
        ),
        output_type=GeneratedMetadata,
    )
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
    # chains just need to reach it. A chain is kept only when not already present verbatim.
    keywords = list(result.output.keywords)
    keywords += [chain for chain in result.output.hierarchies if chain not in keywords]
    return InferenceResult(
        title=result.output.title,
        description=result.output.description,
        keywords=keywords,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        seconds=elapsed,
    )
