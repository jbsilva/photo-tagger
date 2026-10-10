"""
Opt-in OpenTelemetry tracing of model calls (``--otel``).

pydantic-ai can emit an OpenTelemetry span per model call: the prompt it sent, how long the model
took, how many tokens it burned, what came back. Pointed at a collector (Logfire, Jaeger, Grafana
Tempo, anything that speaks OTLP over HTTP) that turns a slow or misbehaving batch into something
you can read instead of guess at.

It is off by default and there is no hosted service behind it. Nothing is exported until someone
passes ``--otel`` *and* names an endpoint, and the endpoint is always theirs. That keeps this
feature inside the promise :mod:`photo_tagger.telemetry` makes, with two limits worth stating:

- **Photos are never exported.** ``include_binary_content`` is hard-wired off, so the JPEG bytes
  sent to the model never reach a span. There is no flag to turn that on.
- **Generated text stays home by default.** Prompts and the generated titles, descriptions, and
  keywords only leave with ``--otel-content``. It exists because reading the actual prompt is the
  point when you are debugging one, but it has to be asked for.

Instrumentation is process-global (``Agent.instrument_all``), so the single call at startup covers
every agent the process goes on to build. That is why nothing here is threaded through
:func:`photo_tagger.ai.create_agent`. The CLI makes the call for ``tag`` and ``watch`` only, and
the GUI makes it at launch, which covers its vocabulary organizer too.
"""

import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from photo_tagger import __version__
from photo_tagger.errors import ObservabilityError


if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider


# The OTLP environment variables, read in the precedence the OpenTelemetry spec gives them: the
# traces-specific one is a complete URL, the general one is a base that the signal path hangs off.
_ENV_TRACES_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
_ENV_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
_ENV_SERVICE_NAME = "OTEL_SERVICE_NAME"

_INSTALL_HINT = "Install it with: pip install 'photo-tagger[otel]'"

# True once a provider is in place. `watch` builds a run per batch and the GUI can start several,
# so without this a long session would stack a fresh exporter on every pass.
_configured = False


def _checked_url(url: str) -> str:
    """
    Return *url* as an OTLP/HTTP traces endpoint, or refuse one the exporter could never reach.

    The exporter sends in the background and only logs a failure, so a URL without a scheme would
    look traced while nothing arrives. A URL with no path gets the standard ``/v1/traces``: posting
    to a collector's root answers 404 on every batch.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        msg = f"--otel endpoint {url!r} is not an http:// or https:// URL with a host"
        raise ObservabilityError(msg)
    if parts.path in {"", "/"}:
        return urlunsplit(parts._replace(path="/v1/traces"))
    return parts.geturl()


def _resolve_endpoint(endpoint: str | None) -> str:
    """
    Work out where to export spans, or say clearly that there is nowhere.

    Refusing to start beats the OTLP default of ``localhost:4318``: a photo tagger has no reason to
    assume a collector is running there, and "I passed --otel and saw nothing" is a worse afternoon
    than an error naming the two ways to fix it.
    """
    if endpoint:
        return _checked_url(endpoint)
    if signal_endpoint := os.environ.get(_ENV_TRACES_ENDPOINT):
        return _checked_url(signal_endpoint)
    if base := os.environ.get(_ENV_ENDPOINT):
        return _checked_url(f"{base.rstrip('/')}/v1/traces")
    msg = (
        "--otel needs somewhere to send traces. Pass --otel-endpoint URL, or set "
        f"{_ENV_TRACES_ENDPOINT} (a full URL ending in /v1/traces) or {_ENV_ENDPOINT} (its base)."
    )
    raise ObservabilityError(msg)


def _build_provider(endpoint: str) -> TracerProvider:
    """
    Build a batching OTLP/HTTP tracer provider aimed at *endpoint*.

    The SDK is imported here rather than at module scope so that a CLI that never asks for tracing
    never pays to import it, and so a missing install is reported against the flag that needed it.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError as exc:
        msg = f"--otel needs the OpenTelemetry SDK, which is not installed. {_INSTALL_HINT}"
        raise ObservabilityError(msg) from exc

    # Resource.create already honors OTEL_SERVICE_NAME, so only name ourselves when the user has
    # not; passing it unconditionally would override their choice, and omitting it entirely would
    # label every span "unknown_service".
    attributes = {"service.version": __version__}
    if _ENV_SERVICE_NAME not in os.environ:
        attributes["service.name"] = "photo-tagger"

    provider = TracerProvider(resource=Resource.create(attributes))
    # Batched, not simple: a span per photo must not add a blocking HTTP round trip to a run whose
    # whole point is throughput. TracerProvider flushes what is queued at interpreter exit.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return provider


def _instrument_agents(provider: TracerProvider, *, include_content: bool) -> None:
    """Route every agent in this process to *provider*, with the photos themselves left out."""
    # Imported here to match _build_provider: nothing about tracing loads on the default path.
    from pydantic_ai import Agent  # noqa: PLC0415
    from pydantic_ai.models.instrumented import InstrumentationSettings  # noqa: PLC0415

    Agent.instrument_all(
        InstrumentationSettings(
            tracer_provider=provider,
            include_content=include_content,
            # Never negotiable: this is the photo, base64'd into a span attribute.
            include_binary_content=False,
        ),
    )


def configure_tracing(*, enabled: bool, endpoint: str | None, include_content: bool) -> bool:
    """
    Turn on OpenTelemetry tracing for every agent this process builds.

    Call once, early, before any agent exists. Returns whether tracing ended up on, which is
    always False when *enabled* is False: that is the default and the only no-op path.

    Raises:
        ObservabilityError: no endpoint was given, or the OpenTelemetry SDK is not installed.
            Both are the user asking for something that cannot be delivered, so the run stops
            rather than pretending to trace.
    """
    global _configured  # noqa: PLW0603 - the OTel provider it guards is process-global too.
    if not enabled:
        return False
    if _configured:
        return True
    resolved = _resolve_endpoint(endpoint)
    _instrument_agents(_build_provider(resolved), include_content=include_content)
    _configured = True
    logger.info("otel_tracing_enabled", endpoint=resolved, include_content=include_content)
    return True
