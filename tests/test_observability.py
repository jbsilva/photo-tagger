"""Tests for the opt-in OpenTelemetry tracing behind --otel."""

import base64
import json
import sys
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.test import TestModel

from photo_tagger import observability
from photo_tagger.errors import ObservabilityError


_ENV_VARS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")

# Distinctive bytes to hunt for in exported spans: if the photo ever leaks, this is what leaks.
_PHOTO_BYTES = b"\xff\xd8secret-photo-pixels"


@pytest.fixture(autouse=True)
def _isolate_global_instrumentation(monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ANN401
    """
    Keep each test's view of the two global switches this module flips to itself.

    Both ``Agent.instrument_all`` and the module's ``_configured`` flag are process-wide, so without
    this a test that turns tracing on would leave every later test in the suite instrumented.
    """
    monkeypatch.setattr(observability, "_configured", False)
    for name in (*_ENV_VARS, "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)
    yield
    Agent.instrument_all(False)  # noqa: FBT003 - pydantic-ai's own off switch is a positional bool.


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_explicit_endpoint_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """--otel-endpoint beats both OTLP environment variables."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://env/v1/traces")
    assert observability._resolve_endpoint("http://flag/v1/traces") == "http://flag/v1/traces"  # noqa: SLF001


def test_traces_endpoint_variable_is_used_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """The signal-specific variable is a complete URL, so no path is appended to it."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces")
    assert observability._resolve_endpoint(None) == "http://collector:4318/v1/traces"  # noqa: SLF001


def test_traces_endpoint_variable_beats_the_general_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """With both set, the OTLP spec gives the signal-specific variable precedence."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://signal/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://general")
    assert observability._resolve_endpoint(None) == "http://signal/v1/traces"  # noqa: SLF001


@pytest.mark.parametrize("base", ["http://collector:4318", "http://collector:4318/"])
def test_general_endpoint_variable_gains_the_signal_path(
    monkeypatch: pytest.MonkeyPatch,
    base: str,
) -> None:
    """The general variable is a base URL; /v1/traces hangs off it, with no doubled slash."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", base)
    assert observability._resolve_endpoint(None) == "http://collector:4318/v1/traces"  # noqa: SLF001


def test_no_endpoint_anywhere_is_an_error_naming_both_ways_to_fix_it() -> None:
    """Rather than quietly exporting to the OTLP default of localhost, say there is nowhere."""
    with pytest.raises(ObservabilityError) as exc_info:
        observability._resolve_endpoint(None)  # noqa: SLF001
    message = str(exc_info.value)
    assert "--otel-endpoint" in message
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in message


# ---------------------------------------------------------------------------
# configure_tracing
# ---------------------------------------------------------------------------


def test_disabled_never_touches_opentelemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default path does no work at all: no endpoint lookup, no provider, no instrumenting."""

    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("tracing must stay untouched when --otel was not passed")

    monkeypatch.setattr(observability, "_build_provider", fail)
    monkeypatch.setattr(observability, "_resolve_endpoint", fail)
    assert (
        observability.configure_tracing(
            enabled=False,
            endpoint="http://collector/v1/traces",
            include_content=True,
        )
        is False
    )


def test_second_call_does_not_stack_another_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A watch session configuring per batch must build the provider once, not once per batch."""
    built: list[str] = []
    monkeypatch.setattr(observability, "_build_provider", built.append)
    monkeypatch.setattr(observability, "_instrument_agents", lambda *_a, **_k: None)

    for _ in range(3):
        assert observability.configure_tracing(
            enabled=True,
            endpoint="http://collector/v1/traces",
            include_content=False,
        )
    assert built == ["http://collector/v1/traces"]


def test_missing_opentelemetry_sdk_points_at_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracing install hint beats an ImportError traceback from inside a photo tagger."""
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    with pytest.raises(ObservabilityError) as exc_info:
        observability.configure_tracing(
            enabled=True,
            endpoint="http://collector/v1/traces",
            include_content=False,
        )
    assert "photo-tagger[otel]" in str(exc_info.value)


def test_provider_names_the_service_and_its_version() -> None:
    """Spans arrive at the collector labelled as photo-tagger, not as unknown_service."""
    provider = observability._build_provider("http://collector/v1/traces")  # noqa: SLF001
    attributes = provider.resource.attributes
    assert attributes["service.name"] == "photo-tagger"
    assert attributes["service.version"]


def test_an_explicit_service_name_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """OTEL_SERVICE_NAME is the user's call; naming ourselves would override it."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-pipeline")
    provider = observability._build_provider("http://collector/v1/traces")  # noqa: SLF001
    assert provider.resource.attributes["service.name"] == "my-pipeline"


# ---------------------------------------------------------------------------
# What actually reaches a collector
# ---------------------------------------------------------------------------


def _run_traced_agent(*, include_content: bool) -> list[str]:
    """Trace one agent run over a photo, returning each exported span as a JSON string."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observability._instrument_agents(provider, include_content=include_content)  # noqa: SLF001

    agent = Agent(TestModel(custom_output_text="a hand-painted ocarina"))
    agent.run_sync(
        [
            "describe this photo",
            BinaryContent(data=_PHOTO_BYTES, media_type="image/jpeg"),
        ],
    )
    return [span.to_json() for span in exporter.get_finished_spans()]


def _image_parts(spans: list[str]) -> list[dict[str, Any]]:
    """Pull every image part back out of the exported prompt messages."""
    parts: list[dict[str, Any]] = []
    for span in spans:
        messages = json.loads(span)["attributes"].get("gen_ai.input.messages")
        if messages is None:
            continue
        for message in json.loads(messages):
            parts.extend(part for part in message["parts"] if part.get("type") == "blob")
    return parts


def test_the_photo_itself_is_never_exported() -> None:
    """
    The strongest promise this module makes, checked against real exported spans.

    Asserted with --otel-content *on*, which is the permissive setting: if the pixels stay out even
    then, there is no combination of flags that ships someone's photo to a collector.

    The image part has to be found first. Asserting only that the bytes are absent would pass just
    as well if the photo never reached the span at all, which would make this test meaningless the
    day the prompt shape changes.
    """
    spans = _run_traced_agent(include_content=True)
    images = _image_parts(spans)
    assert images, "no image part reached the spans, so this proves nothing"
    # The media type is recorded either way. That is useful and gives nothing away; the pixels do.
    assert all(part["mime_type"] == "image/jpeg" for part in images)
    assert all("content" not in part for part in images)
    exported = "\n".join(spans)
    assert base64.b64encode(_PHOTO_BYTES).decode() not in exported


def test_content_off_keeps_prompts_and_answers_out_of_the_spans() -> None:
    """The default: timings and token counts go out, the words describing the photo do not."""
    spans = _run_traced_agent(include_content=False)
    assert spans, "instrumentation produced no spans, so this proves nothing"
    exported = "\n".join(spans)
    assert "describe this photo" not in exported
    assert "hand-painted ocarina" not in exported


def test_content_on_exports_the_prompt_and_the_answer() -> None:
    """--otel-content is the whole point of the flag, so check it actually changes what ships."""
    exported = "\n".join(_run_traced_agent(include_content=True))
    assert "describe this photo" in exported
    assert "hand-painted ocarina" in exported


def test_token_usage_is_exported_whatever_content_setting_is_used() -> None:
    """Cost and latency are why you would turn tracing on, and they carry nothing private."""
    spans = _run_traced_agent(include_content=False)
    attributes = [json.loads(span)["attributes"] for span in spans]
    assert any(
        any(key.startswith("gen_ai.") and "usage" in key for key in attrs) for attrs in attributes
    )


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://localhost:4318/v1/traces", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318", "http://localhost:4318/v1/traces"),
        ("https://collector.example/", "https://collector.example/v1/traces"),
    ],
)
def test_the_endpoint_is_a_traces_url(given: str, expected: str) -> None:
    """A bare collector address gets the standard traces path, so batches do not hit a 404."""
    assert observability._resolve_endpoint(given) == expected  # noqa: SLF001


@pytest.mark.parametrize("given", ["localhost:4318", "collector/v1/traces", "ftp://x/v1/traces"])
def test_an_endpoint_that_cannot_work_is_refused(given: str) -> None:
    """A URL the exporter could never reach stops the run instead of failing in the background."""
    with pytest.raises(ObservabilityError, match="not an http"):
        observability._resolve_endpoint(given)  # noqa: SLF001
