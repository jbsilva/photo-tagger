"""Vision-language agent setup and inference helpers."""

import time
import urllib.parse
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal

import httpx
from loguru import logger
from pydantic_ai import Agent, AgentRunResult, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from photo_tagger.config import (
    DEFAULT_LMSTUDIO_API_KEY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_USER_PROMPT,
    PROVIDER_URLS,
)
from photo_tagger.models import GeneratedMetadata


if TYPE_CHECKING:
    from pydantic_ai import BinaryContent


ProviderName = Literal["ollama", "lmstudio"]


def _validate_lmstudio_url(url: str) -> None:
    """Reject malformed URLs before we hit the network."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        logger.error("lmstudio_model_listing_invalid_scheme", url=url, scheme=parsed.scheme)
        raise SystemExit(1)
    if not parsed.netloc:
        logger.error("lmstudio_model_listing_missing_host", url=url)
        raise SystemExit(1)


def _fetch_lmstudio_models(url: str, api_key: str | None) -> list[str]:
    """Return the list of model ids exposed by an LM Studio /models endpoint."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.error("lmstudio_model_listing_error", error=str(exc), url=url)
        raise SystemExit(1) from exc

    if response.status_code != HTTPStatus.OK:
        logger.error(
            "lmstudio_model_listing_failed",
            status=response.status_code,
            url=url,
            body=response.text,
        )
        raise SystemExit(1)

    try:
        listing = response.json()
    except ValueError as exc:
        logger.error("lmstudio_model_listing_invalid_json", error=str(exc), url=url)
        raise SystemExit(1) from exc

    return [
        str(entry["id"])
        for entry in listing.get("data", [])
        if isinstance(entry, dict) and "id" in entry
    ]


def validate_lmstudio_model(api_base_url: str, model_name: str, api_key: str | None) -> None:
    """Fail fast when LM Studio cannot resolve the requested model name."""
    url = urllib.parse.urljoin(api_base_url.rstrip("/") + "/", "models")
    _validate_lmstudio_url(url)
    models = _fetch_lmstudio_models(url, api_key)
    if model_name not in models:
        logger.error("lmstudio_model_not_available", requested=model_name, available=models)
        raise SystemExit(1)
    logger.debug("lmstudio_model_validated", model=model_name)


def create_agent(
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
    retries: int,
) -> Agent:
    """Build a configured pydantic-ai Agent backed by Ollama or LM Studio."""
    resolved_url = api_base_url or PROVIDER_URLS.get(provider_name, DEFAULT_OLLAMA_BASE_URL)
    if api_base_url is None:
        logger.debug("using_default_provider_url", url=resolved_url)
    logger.info(
        "provider_config_resolved",
        provider=provider_name,
        url=resolved_url,
        model=model_name,
    )

    if provider_name == "ollama":
        provider = OllamaProvider(
            base_url=resolved_url,
            api_key=api_key or DEFAULT_OLLAMA_API_KEY,
        )
    else:
        resolved_api_key = api_key or DEFAULT_LMSTUDIO_API_KEY
        validate_lmstudio_model(resolved_url, model_name, resolved_api_key)
        provider = OpenAIProvider(base_url=resolved_url, api_key=resolved_api_key)

    chat_model = OpenAIChatModel(model_name=model_name, provider=provider)
    return Agent(
        chat_model,
        output_type=GeneratedMetadata,  # type: ignore[arg-type]
        retries=retries,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


def analyze_image_with_ai(
    image_bytes: BinaryContent,
    agent: Agent,
    *,
    user_prompt: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, str, list[str]]:
    """
    Generate a short title, description, and keywords using a vision-language model.

    Returns:
        Tuple of (title, description, keywords).

    """
    logger.info("analyzing_image_with_ai")
    started = time.perf_counter()
    prompt = user_prompt or DEFAULT_USER_PROMPT

    result: AgentRunResult[GeneratedMetadata] = agent.run_sync(
        [prompt, image_bytes],
        model_settings=ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        output_type=GeneratedMetadata,
    )
    logger.info(
        "ai_inference_completed",
        seconds=round(time.perf_counter() - started, 3),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.debug(
        "ai_generated_metadata",
        title=result.output.title,
        description=result.output.description,
        keywords=result.output.keywords,
    )
    return result.output.title, result.output.description, result.output.keywords
