"""Tests for the photo processing pipeline using lightweight stubs."""

import contextlib
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from pydantic_ai import BinaryContent

from photo_tagger.ai import InferenceResult
from photo_tagger.metadata import ImageContext
from photo_tagger.pipeline import (
    ProcessingOptions,
    _UsageAccumulator,
    execute_process,
    process_photo,
    run_batch,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pydantic_ai import Agent

    from photo_tagger.models import GeneratedMetadata

# Pipeline tests patch every IO call, so the agent value never gets touched.
_FAKE_AGENT = cast("Agent[None, GeneratedMetadata]", object())


@contextlib.contextmanager
def _stub_helper(_et: object | None = None) -> Iterator[object]:
    """Stand-in for photo_tagger.metadata._helper that never spawns an exiftool process."""
    yield object()


@pytest.fixture(autouse=True)
def _no_real_exiftool() -> Iterator[None]:
    """Make every test in this file safe to run without an exiftool binary on PATH."""
    with patch("photo_tagger.pipeline._helper", _stub_helper):
        yield


@pytest.fixture
def stub_image_bytes() -> BinaryContent:
    """Return a tiny placeholder JPEG payload used to short-circuit image preparation."""
    return BinaryContent(data=b"\xff\xd8stub", media_type="image/jpeg")


@pytest.fixture
def patched_pipeline(stub_image_bytes: BinaryContent) -> Any:  # noqa: ANN401
    """Patch every IO collaborator the pipeline uses with deterministic stubs."""
    with (
        patch("photo_tagger.pipeline.prepare_image_for_agent", return_value=stub_image_bytes),
        patch(
            "photo_tagger.pipeline.read_image_context",
            return_value=ImageContext(),
        ),
        patch(
            "photo_tagger.pipeline.analyze_image_with_ai",
            return_value=InferenceResult(
                title="Title",
                description="Description.",
                keywords=["Beach", "Sunset"],
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                seconds=0.1,
            ),
        ) as analyze,
        patch("photo_tagger.pipeline.write_metadata", return_value=True) as write,
    ):
        yield {"analyze": analyze, "write": write}


def test_process_photo_writes_metadata(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Happy path passes the merged keywords and AI fields to write_metadata."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions()

    assert process_photo(image, agent=_FAKE_AGENT, options=options) is True

    write_call = patched_pipeline["write"].call_args
    kwargs = write_call.kwargs
    assert kwargs["description"] == "Description."
    assert kwargs["title"] == "Title"
    assert kwargs["use_sidecar"] is True
    keywords = write_call.args[1]
    assert "Beach" in keywords["subject"]


def test_process_photo_skips_optional_fields_when_disabled(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Disabling write_title / write_description nulls the corresponding kwargs."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(write_description=False, write_title=False)

    process_photo(image, agent=_FAKE_AGENT, options=options)

    kwargs = patched_pipeline["write"].call_args.kwargs
    assert kwargs["description"] is None
    assert kwargs["title"] is None


def test_process_photo_returns_false_when_write_fails(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """write_metadata returning False bubbles up as a False process_photo result."""
    patched_pipeline["write"].return_value = False
    image = tmp_path / "img.cr3"
    image.write_text("x")
    assert process_photo(image, agent=_FAKE_AGENT, options=ProcessingOptions()) is False


@pytest.mark.usefixtures("patched_pipeline")
def test_process_photo_folds_token_usage_into_accumulator(tmp_path: Path) -> None:
    """Passing a shared accumulator records the InferenceResult's tokens and latency."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    usage = _UsageAccumulator()

    process_photo(image, agent=_FAKE_AGENT, options=ProcessingOptions(), usage=usage)

    expected_input = 10
    expected_output = 5
    expected_total = 15
    expected_calls = 1
    assert usage.input_tokens == expected_input
    assert usage.output_tokens == expected_output
    assert usage.total_tokens == expected_total
    assert usage.inference_calls == expected_calls


def test_process_photo_dry_run_skips_write_metadata(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """dry_run=True logs the preview and reports success without touching the writer."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(dry_run=True)

    assert process_photo(image, agent=_FAKE_AGENT, options=options) is True
    patched_pipeline["write"].assert_not_called()


def test_execute_process_returns_false_on_exception(tmp_path: Path) -> None:
    """Unexpected exceptions inside process_photo become a logged False result."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    with patch("photo_tagger.pipeline.process_photo", side_effect=RuntimeError("boom")):
        ok = execute_process(image, agent=_FAKE_AGENT, options=ProcessingOptions(), index="1/1")
    assert ok is False


def test_execute_process_logs_retry_path(tmp_path: Path) -> None:
    """The retry branch is exercised when retry=True is passed."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        ok = execute_process(
            image,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            index="1/1",
            retry=True,
        )
    assert ok is True


_BATCH_SIZE = 3


def test_run_batch_succeeds_when_all_pass(tmp_path: Path) -> None:
    """If every file processes successfully, run_batch returns totals without raising."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_BATCH_SIZE)]
    for f in files:
        f.write_text("x")

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        totals = run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions())
    assert totals.success == _BATCH_SIZE
    assert totals.initial_failures == 0


def test_run_batch_raises_system_exit_when_any_fails_after_retry(tmp_path: Path) -> None:
    """A file that keeps failing forces a SystemExit so CI marks the run as failed."""
    files = [tmp_path / "img.cr3"]
    files[0].write_text("x")

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(SystemExit),
    ):
        run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions())


def test_run_batch_retry_recovers_a_failure(tmp_path: Path) -> None:
    """A first-pass failure that succeeds on retry is counted in retry_successes."""
    files = [tmp_path / "img.cr3"]
    files[0].write_text("x")

    call_count = {"n": 0}
    succeed_after = 2

    def fake_process_photo(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401
        call_count["n"] += 1
        return call_count["n"] >= succeed_after  # fail first call, succeed on retry

    with patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo):
        totals = run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions())
    assert totals.success == 1
    assert totals.initial_failures == 1
    assert totals.retry_successes == 1


def test_run_batch_calls_on_success_for_each_completed_file(tmp_path: Path) -> None:
    """on_success fires for first-pass and retry-pass successes, never for failures."""
    success_first = tmp_path / "ok.cr3"
    success_retry = tmp_path / "retry.cr3"
    failure = tmp_path / "fail.cr3"
    for path in (success_first, success_retry, failure):
        path.write_text("x")

    # Map (filename, attempt_number) -> outcome. The retry file fails first then succeeds.
    attempts: dict[str, int] = {}
    retry_attempt_threshold = 2  # success_retry passes from its second attempt onwards.

    def fake_process_photo(image: Path, *_a: Any, **_kw: Any) -> bool:  # noqa: ANN401
        attempts[image.name] = attempts.get(image.name, 0) + 1
        if image.name == success_first.name:
            return True
        if image.name == failure.name:
            return False
        return attempts[image.name] >= retry_attempt_threshold

    notified: list[Path] = []

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo),
        pytest.raises(SystemExit),
    ):
        run_batch(
            [success_first, success_retry, failure],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_success=notified.append,
        )

    assert notified == [success_first, success_retry]


def test_run_batch_swallows_on_success_callback_errors(tmp_path: Path) -> None:
    """A callback that raises must not abort the batch; success count still reflects work."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    def boom(_path: Path) -> None:
        msg = "callback exploded"
        raise RuntimeError(msg)

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        totals = run_batch(
            [image],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_success=boom,
        )
    assert totals.success == 1


_CONCURRENT_BATCH_SIZE = 4


def test_run_batch_concurrent_processes_all_files(tmp_path: Path) -> None:
    """workers>1 dispatches to a thread pool and reports the same totals as serial."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_CONCURRENT_BATCH_SIZE)]
    for f in files:
        f.write_text("x")

    with patch("photo_tagger.pipeline.process_photo", return_value=True) as proc:
        totals = run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            workers=2,
        )

    assert totals.success == _CONCURRENT_BATCH_SIZE
    # Each task must have received et=None so it opens its own helper inside process_photo.
    for call in proc.call_args_list:
        assert call.kwargs["et"] is None


def test_run_batch_concurrent_calls_on_success_per_image(tmp_path: Path) -> None:
    """on_success fires once per image regardless of completion order."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_CONCURRENT_BATCH_SIZE)]
    for f in files:
        f.write_text("x")

    notified: list[Path] = []
    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_success=notified.append,
            workers=2,
        )

    assert sorted(p.name for p in notified) == sorted(p.name for p in files)


def test_run_batch_progress_callback_fires_per_image(tmp_path: Path) -> None:
    """progress=callable fires once per image with (path, ok) for both serial and concurrent."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    received: list[tuple[str, bool]] = []

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        run_batch(
            [image],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            progress=lambda path, ok: received.append((path.name, ok)),
        )

    assert received == [(image.name, True)]


@pytest.mark.usefixtures("patched_pipeline")
def test_run_batch_aggregates_token_usage_from_inference(tmp_path: Path) -> None:
    """Per-call token counts add up into the BatchTotals returned to the CLI."""
    files = [tmp_path / f"img{i}.cr3" for i in range(2)]
    for f in files:
        f.write_text("x")

    totals = run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions())

    expected_input = 10 * 2
    expected_output = 5 * 2
    expected_total = 15 * 2
    expected_calls = 2
    assert totals.input_tokens == expected_input
    assert totals.output_tokens == expected_output
    assert totals.total_tokens == expected_total
    assert totals.inference_calls == expected_calls
    assert sorted(totals.successful_files) == sorted(str(f) for f in files)
    assert totals.failed_files == []


def test_run_batch_calls_on_complete_with_totals_even_on_failure(tmp_path: Path) -> None:
    """on_complete fires before SystemExit so the CLI can persist a summary file."""
    files = [tmp_path / "img.cr3"]
    files[0].write_text("x")
    received: list[Any] = []

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(SystemExit),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_complete=received.append,
        )

    assert len(received) == 1
    totals = received[0]
    assert totals.success == 0
    assert totals.failed_files == [str(files[0])]


def test_run_batch_swallows_on_complete_errors(tmp_path: Path) -> None:
    """An on_complete that raises is logged but does not change the exit semantics."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    files = [tmp_path / "img.cr3"]
    files[0].write_text("x")

    def boom(_totals: BatchTotals) -> None:
        msg = "summary writer crashed"
        raise RuntimeError(msg)

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        totals = run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_complete=boom,
        )
    assert totals.success == 1


def test_process_photo_skips_ai_call_on_cache_hit(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A cache hit reuses the stored InferenceResult and never calls analyze_image_with_ai."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    cached = InferenceResult(
        title="Cached Title",
        description="Cached description.",
        keywords=["Cached"],
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        seconds=2.5,
    )

    class _StubCache:
        def __init__(self) -> None:
            self.put_calls = 0

        def get(self, _key: str) -> InferenceResult:
            return cached

        def put(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            self.put_calls += 1

    cache = _StubCache()
    options = ProcessingOptions()

    assert process_photo(image, agent=_FAKE_AGENT, options=options, cache=cache) is True  # type: ignore[arg-type]

    # The AI call must have been skipped entirely.
    patched_pipeline["analyze"].assert_not_called()
    # And the write call must have used the cached title.
    write_kwargs = patched_pipeline["write"].call_args.kwargs
    assert write_kwargs["title"] == "Cached Title"
    # Cache hits do not re-store the entry.
    assert cache.put_calls == 0


def test_process_photo_writes_to_cache_on_miss(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A cache miss runs the AI call and persists the result via cache.put."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    class _StubCache:
        def __init__(self) -> None:
            self.put_calls: list[Any] = []

        def get(self, _key: str) -> None:
            return None

        def put(self, key: str, result: InferenceResult) -> None:
            self.put_calls.append((key, result))

    cache = _StubCache()
    process_photo(image, agent=_FAKE_AGENT, options=ProcessingOptions(), cache=cache)  # type: ignore[arg-type]

    patched_pipeline["analyze"].assert_called_once()
    assert len(cache.put_calls) == 1
    _, stored = cache.put_calls[0]
    assert stored.title == "Title"


def test_run_batch_serial_handles_keyboard_interrupt(tmp_path: Path) -> None:
    """Ctrl-C in the serial path stops scheduling and lands remaining files in failed."""
    files = [tmp_path / f"img{i}.cr3" for i in range(3)]
    for f in files:
        f.write_text("x")

    call_count = {"n": 0}
    interrupt_after = 1

    def fake_process_photo(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401
        call_count["n"] += 1
        if call_count["n"] > interrupt_after:
            raise KeyboardInterrupt
        return True

    received_totals: list[Any] = []
    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo),
        pytest.raises(SystemExit),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_complete=received_totals.append,
        )

    totals = received_totals[0]
    assert totals.success == interrupt_after
    # The two photos after the first should appear as failed so a future
    # --skip-from rerun can pick them up. on_complete still fires so a
    # --summary-file is written even when the run was aborted.
    assert len(totals.failed_files) >= 1
