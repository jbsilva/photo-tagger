"""
Vision-language backend registry.

Each backend answers three questions: where do I list available models, how do I read model ids out
of that listing, and how do I build the pydantic-ai provider object? Bundling those as a
:class:`ProviderBackend` (the Strategy pattern) means adding a backend is a single entry in
``_BACKENDS`` plus its defaults in :mod:`photo_tagger.config`, with no edits scattered across the
agent setup, the CLI ``--provider`` choices, or the model-validation paths.

The shared HTTP plumbing (URL validation, listing fetch, body truncation for logs) lives here once;
every backend reuses it instead of re-implementing it.
"""

import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal

import httpx
from loguru import logger
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from photo_tagger.config import (
    DEFAULT_LMSTUDIO_API_KEY,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
)
from photo_tagger.errors import ProviderError


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


# Every supported backend name. Kept as a Literal so cyclopts can validate the
# ``--provider`` flag and static analysis can check exhaustiveness; a test asserts
# it stays in lockstep with the runtime registry below.
ProviderName = Literal["ollama", "lmstudio", "openai"]

# A pydantic-ai provider this package knows how to construct.
ChatProvider = OllamaProvider | OpenAIProvider

# Cap on the response body included in failure logs. A misconfigured provider can
# return an entire HTML page (or echo back the request including an Authorization
# header); we do not want a multi-MB string, or a secret, in the log file.
_MAX_LOGGED_BODY_CHARS = 500
_LISTING_TIMEOUT_SECONDS = 5.0


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


def _fetch_listing(
    url: str,
    api_key: str | None,
    *,
    event_prefix: str,
) -> Mapping[str, object]:
    """GET *url* with an optional Bearer token; return the parsed JSON body."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(url, headers=headers, timeout=_LISTING_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        logger.error(f"{event_prefix}_error", error=str(exc), url=url)
        raise ProviderError(str(exc)) from exc

    if response.status_code != HTTPStatus.OK:
        body = _truncate_for_log(response.text)
        logger.error(f"{event_prefix}_failed", status=response.status_code, url=url, body=body)
        msg = f"HTTP {response.status_code} from {url}: {body}"
        raise ProviderError(msg)

    try:
        return response.json()  # type: ignore[no-any-return]
    except ValueError as exc:
        logger.error(f"{event_prefix}_invalid_json", error=str(exc), url=url)
        raise ProviderError(str(exc)) from exc


# --- listing-URL builders -----------------------------------------------------------------------


def _openai_listing_url(base_url: str) -> str:
    """OpenAI-compatible servers list models at ``<base>/models``."""
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", "models")


def _ollama_listing_url(base_url: str) -> str:
    """
    Ollama lists models at ``/api/tags``, not the OpenAI-style ``/v1/models``.

    Strip a trailing ``/v1`` from the configured OpenAI-compatible URL first so the same base URL
    works for both inference and the model check.
    """
    base = base_url.rstrip("/").removesuffix("/v1").rstrip("/")
    return base + "/api/tags"


# --- listing parsers ----------------------------------------------------------------------------


def _openai_model_ids(listing: Mapping[str, object]) -> list[str]:
    """Read model ids from an OpenAI-style ``{"data": [{"id": ...}]}`` listing."""
    raw_data = listing.get("data", [])
    if not isinstance(raw_data, list):
        return []
    return [str(entry["id"]) for entry in raw_data if isinstance(entry, dict) and "id" in entry]


def _ollama_model_names(listing: Mapping[str, object]) -> list[str]:
    """Read model names from Ollama's ``{"models": [{"name": ...}]}`` listing."""
    raw_models = listing.get("models", [])
    if not isinstance(raw_models, list):
        return []
    return [
        str(entry["name"]) for entry in raw_models if isinstance(entry, dict) and "name" in entry
    ]


# --- provider factories -------------------------------------------------------------------------


def _make_ollama(base_url: str, api_key: str | None) -> ChatProvider:
    """Build the native Ollama provider."""
    return OllamaProvider(base_url=base_url, api_key=api_key)


def _make_openai(base_url: str, api_key: str | None) -> ChatProvider:
    """Build an OpenAI-compatible provider (LM Studio and hosted OpenAI both use this)."""
    return OpenAIProvider(base_url=base_url, api_key=api_key)


@dataclass(frozen=True, slots=True)
class ProviderBackend:
    """
    One vision-language backend's strategy: list models, parse them, build a provider.

    Instances are immutable and live in :data:`_BACKENDS`. The three callables are the only per-
    backend behavior; everything else (URL validation, fetch, membership check) is shared.
    """

    name: ProviderName
    default_base_url: str
    default_api_key: str | None
    build_listing_url: Callable[[str], str]
    parse_models: Callable[[Mapping[str, object]], list[str]]
    make_provider: Callable[[str, str | None], ChatProvider]
    # Hosted backends (real OpenAI) cannot work without a key; local ones default to None.
    requires_api_key: bool = False

    def resolve_api_key(self, override: str | None) -> str | None:
        """Return the explicit *override* if given, else this backend's default key."""
        return override or self.default_api_key

    def list_models(self, base_url: str, api_key: str | None) -> list[str]:
        """Query the backend and return the model ids it currently exposes."""
        url = self.build_listing_url(base_url)
        event_prefix = f"{self.name}_model_listing"
        _validate_listing_url(url, event_prefix=event_prefix)
        listing = _fetch_listing(url, api_key, event_prefix=event_prefix)
        return self.parse_models(listing)

    def validate_model(self, base_url: str, model_name: str, api_key: str | None) -> None:
        """Fail fast with :class:`ProviderError` when *model_name* is not available."""
        models = self.list_models(base_url, api_key)
        if model_name not in models:
            msg = f"Model {model_name!r} not available in {self.name} (available: {models})"
            logger.error(
                f"{self.name}_model_not_available",
                requested=model_name,
                available=models,
            )
            raise ProviderError(msg)
        logger.debug(f"{self.name}_model_validated", model=model_name)

    def build_provider(self, base_url: str, api_key: str | None) -> ChatProvider:
        """Construct the pydantic-ai provider for this backend."""
        return self.make_provider(base_url, api_key)


_BACKENDS: dict[str, ProviderBackend] = {
    backend.name: backend
    for backend in (
        ProviderBackend(
            name="ollama",
            default_base_url=DEFAULT_OLLAMA_BASE_URL,
            default_api_key=DEFAULT_OLLAMA_API_KEY,
            build_listing_url=_ollama_listing_url,
            parse_models=_ollama_model_names,
            make_provider=_make_ollama,
        ),
        ProviderBackend(
            name="lmstudio",
            default_base_url=DEFAULT_LMSTUDIO_BASE_URL,
            default_api_key=DEFAULT_LMSTUDIO_API_KEY,
            build_listing_url=_openai_listing_url,
            parse_models=_openai_model_ids,
            make_provider=_make_openai,
        ),
        ProviderBackend(
            name="openai",
            default_base_url=DEFAULT_OPENAI_BASE_URL,
            default_api_key=DEFAULT_OPENAI_API_KEY,
            build_listing_url=_openai_listing_url,
            parse_models=_openai_model_ids,
            make_provider=_make_openai,
            requires_api_key=True,
        ),
    )
}

# Names in CLI/registration order, handy for help text and the consistency test.
PROVIDER_NAMES: tuple[str, ...] = tuple(_BACKENDS)


def get_backend(name: str) -> ProviderBackend:
    """Look up a backend by name, raising :class:`ProviderError` for unknown names."""
    try:
        return _BACKENDS[name]
    except KeyError:
        msg = f"Unknown provider {name!r}; choose from {sorted(_BACKENDS)}"
        logger.error("unknown_provider", requested=name, available=sorted(_BACKENDS))
        raise ProviderError(msg) from None
