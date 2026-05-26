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
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_LMSTUDIO_API_KEY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_PROMPT,
    PROVIDER_URLS,
)
from photo_tagger.errors import ProviderError
from photo_tagger.models import GeneratedMetadata, InferenceResult


if TYPE_CHECKING:
    from pydantic_ai import BinaryContent


ProviderName = Literal["ollama", "lmstudio"]

# Cap on the response body included in failure logs. A misconfigured provider
# can return an entire HTML page (or paste back the request including an
# Authorization header), and we do not want a multi-MB string in the log file.
_MAX_LOGGED_BODY_CHARS = 500


def _truncate_for_log(text: str, *, limit: int = _MAX_LOGGED_BODY_CHARS) -> str:
    """Return *text* trimmed to *limit* chars with a tail marker on overflow."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"


def _validate_listing_url(url: str, *, event_prefix: str) -> None:
    """Reject malformed URLs before we hit the network."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"Invalid URL scheme {parsed.scheme!r} for {url}"
        logger.error(f"{event_prefix}_invalid_scheme", url=url, scheme=parsed.scheme)
        raise ProviderError(msg)
    if not parsed.netloc:
        msg = f"Missing host in URL {url}"
        logger.error(f"{event_prefix}_missing_host", url=url)
        raise ProviderError(msg)


def _fetch_listing(url: str, api_key: str | None, *, event_prefix: str) -> dict[str, object]:
    """GET *url* with an optional Bearer token; return the parsed JSON body."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.error(f"{event_prefix}_error", error=str(exc), url=url)
        raise ProviderError(str(exc)) from exc

    if response.status_code != HTTPStatus.OK:
        body = _truncate_for_log(response.text)
        logger.error(
            f"{event_prefix}_failed",
            status=response.status_code,
            url=url,
            body=body,
        )
        msg = f"HTTP {response.status_code} from {url}: {body}"
        raise ProviderError(msg)

    try:
        return response.json()  # type: ignore[no-any-return]
    except ValueError as exc:
        logger.error(f"{event_prefix}_invalid_json", error=str(exc), url=url)
        raise ProviderError(str(exc)) from exc


def _fetch_lmstudio_models(url: str, api_key: str | None) -> list[str]:
    """Return the list of model ids exposed by an LM Studio ``/v1/models`` endpoint."""
    listing = _fetch_listing(url, api_key, event_prefix="lmstudio_model_listing")
    raw_data = listing.get("data", [])
    if not isinstance(raw_data, list):
        return []
    return [str(entry["id"]) for entry in raw_data if isinstance(entry, dict) and "id" in entry]


def _fetch_ollama_models(url: str, api_key: str | None) -> list[str]:
    """Return the list of locally-installed Ollama models from ``/api/tags``."""
    listing = _fetch_listing(url, api_key, event_prefix="ollama_model_listing")
    raw_models = listing.get("models", [])
    if not isinstance(raw_models, list):
        return []
    return [
        str(entry["name"]) for entry in raw_models if isinstance(entry, dict) and "name" in entry
    ]


def validate_lmstudio_model(api_base_url: str, model_name: str, api_key: str | None) -> None:
    """Fail fast when LM Studio cannot resolve the requested model name."""
    url = urllib.parse.urljoin(api_base_url.rstrip("/") + "/", "models")
    _validate_listing_url(url, event_prefix="lmstudio_model_listing")
    models = _fetch_lmstudio_models(url, api_key)
    if model_name not in models:
        msg = f"Model {model_name!r} not available in LM Studio (available: {models})"
        logger.error("lmstudio_model_not_available", requested=model_name, available=models)
        raise ProviderError(msg)
    logger.debug("lmstudio_model_validated", model=model_name)


def validate_ollama_model(api_base_url: str, model_name: str, api_key: str | None) -> None:
    """
    Fail fast when Ollama cannot resolve the requested model name.

    Ollama exposes ``/api/tags`` (not ``/v1/models``) so we strip a trailing ``/v1`` from
    the configured OpenAI-compatible URL before querying. Matching is exact against the
    ``name`` field, which on Ollama already includes the ``:tag`` suffix where present.
    """
    base = api_base_url.rstrip("/").removesuffix("/v1").rstrip("/")
    url = base + "/api/tags"
    _validate_listing_url(url, event_prefix="ollama_model_listing")
    models = _fetch_ollama_models(url, api_key)
    if model_name not in models:
        msg = f"Model {model_name!r} not available in Ollama (available: {models})"
        logger.error("ollama_model_not_available", requested=model_name, available=models)
        raise ProviderError(msg)
    logger.debug("ollama_model_validated", model=model_name)


def create_agent(
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
    retries: int,
) -> Agent[None, GeneratedMetadata]:
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

    match provider_name:
        case "ollama":
            resolved_api_key = api_key or DEFAULT_OLLAMA_API_KEY
            validate_ollama_model(resolved_url, model_name, resolved_api_key)
            provider = OllamaProvider(base_url=resolved_url, api_key=resolved_api_key)
        case "lmstudio":
            resolved_api_key = api_key or DEFAULT_LMSTUDIO_API_KEY
            validate_lmstudio_model(resolved_url, model_name, resolved_api_key)
            provider = OpenAIProvider(base_url=resolved_url, api_key=resolved_api_key)

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
    )
    return InferenceResult(
        title=result.output.title,
        description=result.output.description,
        keywords=list(result.output.keywords),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        seconds=elapsed,
    )
