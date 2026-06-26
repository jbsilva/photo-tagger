"""Tests for the photo processing pipeline using lightweight stubs."""

import contextlib
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from pydantic_ai import BinaryContent

from photo_tagger.errors import BatchError
from photo_tagger.metadata import ImageContext
from photo_tagger.models import InferenceResult, KeywordSet
from photo_tagger.pipeline import (
    ImageOutcome,
    ProcessingOptions,
    _BatchContext,
    _emit_outcome,
    _InferenceScratch,
    _notify_success,
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


def _ctx(
    *,
    options: ProcessingOptions | None = None,
    cache: object | None = None,
    usage: _UsageAccumulator | None = None,
) -> _BatchContext:
    """Build a minimal _BatchContext for unit tests."""
    return _BatchContext(
        agent=_FAKE_AGENT,
        options=options or ProcessingOptions(),
        user_prompt="",
        usage=usage or _UsageAccumulator(),
        cache=cache,  # type: ignore[arg-type]
    )


@contextlib.contextmanager
def _stub_helper(_et: object | None = None) -> Iterator[object]:
    """Stand-in for photo_tagger.metadata.managed_helper that never spawns an exiftool process."""
    yield object()


@pytest.fixture(autouse=True)
def _no_real_exiftool() -> Iterator[None]:
    """Make every test in this file safe to run without an exiftool binary on PATH."""
    with patch("photo_tagger.pipeline.managed_helper", _stub_helper):
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

    assert process_photo(image, _ctx(options=options)) is True

    write_call = patched_pipeline["write"].call_args
    kwargs = write_call.kwargs
    assert kwargs["description"] == "Description."
    assert kwargs["title"] == "Title"
    assert kwargs["use_sidecar"] is True
    keywords = write_call.args[1]
    assert "Beach" in keywords.subject


def test_process_photo_skips_optional_fields_when_disabled(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Disabling write_title / write_description nulls the corresponding kwargs."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(write_description=False, write_title=False)

    process_photo(image, _ctx(options=options))

    kwargs = patched_pipeline["write"].call_args.kwargs
    assert kwargs["description"] is None
    assert kwargs["title"] is None


def test_process_photo_writes_no_keywords_when_disabled(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """write_keywords=False hands write_metadata an empty KeywordSet, keeping existing tags."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(write_keywords=False)

    process_photo(image, _ctx(options=options))

    write_call = patched_pipeline["write"].call_args
    written_keywords = write_call.args[1]
    assert written_keywords.subject == []
    assert written_keywords.hierarchical == []
    # Title and description still flow through, so only keywords are suppressed.
    assert write_call.kwargs["title"] == "Title"
    assert write_call.kwargs["description"] == "Description."


def test_process_photo_returns_false_when_write_fails(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """write_metadata returning False bubbles up as a False process_photo result."""
    patched_pipeline["write"].return_value = False
    image = tmp_path / "img.cr3"
    image.write_text("x")
    assert process_photo(image, _ctx()) is False


@pytest.mark.usefixtures("patched_pipeline")
def test_process_photo_folds_token_usage_into_accumulator(tmp_path: Path) -> None:
    """Passing a shared accumulator records the InferenceResult's tokens and latency."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    usage = _UsageAccumulator()

    process_photo(image, _ctx(usage=usage))

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

    assert process_photo(image, _ctx(options=options)) is True
    patched_pipeline["write"].assert_not_called()


def test_execute_process_returns_false_on_exception(tmp_path: Path) -> None:
    """Unexpected exceptions inside process_photo become a logged False result."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    with patch("photo_tagger.pipeline.process_photo", side_effect=RuntimeError("boom")):
        ok = execute_process(image, _ctx(), index="1/1")
    assert ok is False


def test_execute_process_logs_retry_path(tmp_path: Path) -> None:
    """The retry branch is exercised when retry=True is passed."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        ok = execute_process(
            image,
            _ctx(),
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
    """A file that keeps failing raises BatchError so CI marks the run as failed."""
    files = [tmp_path / "img.cr3"]
    files[0].write_text("x")

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(BatchError),
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
        pytest.raises(BatchError),
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
    """``workers>1`` dispatches to a thread pool and reports the same totals as serial."""
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
    """``progress=callable`` fires once per image with ``(path, ok)`` for serial and concurrent."""
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


def test_progress_callback_fires_once_per_image_with_flaky_files(tmp_path: Path) -> None:
    """A first-pass failure must not tick; a retry-pass success ticks exactly once."""
    images = [tmp_path / f"img{i}.cr3" for i in range(3)]
    for f in images:
        f.write_text("x")
    received: list[tuple[str, bool]] = []

    # First call fails for every image, retry pass succeeds.
    call_count: dict[str, int] = {}

    def _flaky_process(image_path: Path, *args: object, **kwargs: object) -> bool:
        name = image_path.name
        call_count[name] = call_count.get(name, 0) + 1
        return call_count[name] > 1  # fail first attempt, succeed on retry

    with patch("photo_tagger.pipeline.process_photo", side_effect=_flaky_process):
        run_batch(
            images,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            progress=lambda path, ok: received.append((path.name, ok)),
        )

    # Each image ticks exactly once, with the retry-pass success outcome (ok=True).
    assert len(received) == len(images)
    assert all(ok for _name, ok in received)
    assert sorted(name for name, _ok in received) == sorted(p.name for p in images)


def test_progress_callback_ticks_on_final_failure_after_retry(tmp_path: Path) -> None:
    """A file that fails in both passes must tick exactly once (on the retry failure)."""
    images = [tmp_path / f"img{i}.cr3" for i in range(2)]
    for f in images:
        f.write_text("x")
    received: list[tuple[str, bool]] = []

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(
            BatchError,
        ),
    ):
        run_batch(
            images,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            progress=lambda path, ok: received.append((path.name, ok)),
        )

    # Bar reaches 100% (one tick per file) even on permanent failures.
    assert len(received) == len(images)
    assert all(not ok for _name, ok in received)


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
    """on_complete fires before BatchError so the CLI can persist a summary file."""
    files = [tmp_path / "img.cr3"]
    files[0].write_text("x")
    received: list[Any] = []

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(BatchError),
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

    assert process_photo(image, _ctx(options=options, cache=cache)) is True

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
    process_photo(image, _ctx(cache=cache))

    patched_pipeline["analyze"].assert_called_once()
    assert len(cache.put_calls) == 1
    _, stored = cache.put_calls[0]
    assert stored.title == "Title"


def test_process_photo_keys_cache_on_content_hash(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """When exiftool supplies an image-data hash, that is the cache key, not the file hash."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    class _StubCache:
        def __init__(self) -> None:
            self.put_keys: list[str] = []

        def get(self, _key: str) -> None:
            return None

        def put(self, key: str, _result: InferenceResult) -> None:
            self.put_keys.append(key)

    cache = _StubCache()
    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(content_hash="img-data-hash"),
    ):
        assert process_photo(image, _ctx(cache=cache)) is True

    assert cache.put_keys == ["img-data-hash"]


def test_process_photo_cache_hit_survives_metadata_write(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A stable image-data hash lets a second embed run hit the cache and skip the model."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    store: dict[str, InferenceResult] = {}

    class _DictCache:
        def get(self, key: str) -> InferenceResult | None:
            return store.get(key)

        def put(self, key: str, result: InferenceResult) -> None:
            store[key] = result

    options = ProcessingOptions(use_sidecar=False)
    # Both runs see the same image-data hash even though embedding changed the file bytes.
    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(content_hash="stable-hash"),
    ):
        assert process_photo(image, _ctx(options=options, cache=_DictCache())) is True
        assert process_photo(image, _ctx(options=options, cache=_DictCache())) is True

    # The model ran only on the first pass; the second was a cache hit on the same content hash.
    patched_pipeline["analyze"].assert_called_once()


def test_process_photo_reads_content_hash_only_when_caching(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """ImageDataHash is requested only when a cache is configured, to avoid wasted hashing."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    class _StubCache:
        def get(self, _key: str) -> None:
            return None

        def put(self, *_a: Any, **_kw: Any) -> None:  # noqa: ANN401
            return None

    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(),
    ) as read_ctx:
        process_photo(image, _ctx(cache=_StubCache()))
        assert read_ctx.call_args.kwargs["include_content_hash"] is True

    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(),
    ) as read_ctx:
        process_photo(image, _ctx())
        assert read_ctx.call_args.kwargs["include_content_hash"] is False


def test_process_photo_survives_cache_get_raising(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A broken cache.get is treated as a miss; the photo still gets written."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    class _ExplodingCache:
        def __init__(self) -> None:
            self.put_calls = 0

        def get(self, _key: str) -> None:
            msg = "database is locked"
            raise RuntimeError(msg)

        def put(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            self.put_calls += 1

    cache = _ExplodingCache()
    ok = process_photo(image, _ctx(cache=cache))

    assert ok is True
    patched_pipeline["analyze"].assert_called_once()
    # The hash succeeded, get raised, put is still attempted with the fresh result.
    assert cache.put_calls == 1


def test_process_photo_survives_cache_put_raising(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A broken cache.put is logged and swallowed; the photo still succeeds."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    class _PutExploder:
        def get(self, _key: str) -> None:
            return None

        def put(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            msg = "disk full"
            raise OSError(msg)

    ok = process_photo(image, _ctx(cache=_PutExploder()))

    assert ok is True
    patched_pipeline["write"].assert_called_once()


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
        pytest.raises(BatchError),
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


def test_run_batch_concurrent_handles_keyboard_interrupt(tmp_path: Path) -> None:
    """Ctrl-C in the concurrent path cancels futures and lands pending files in failed."""
    files = [tmp_path / f"img{i}.cr3" for i in range(4)]
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
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            on_complete=received_totals.append,
            workers=2,
        )

    assert len(received_totals) == 1
    totals = received_totals[0]
    # Some files were processed, some should be in failed. The exact split
    # depends on scheduling, but we must see at least one failure and on_complete
    # must have fired so the summary file is written.
    assert len(totals.failed_files) >= 1


# ---------------------------------------------------------------------------
# process_photo edge cases
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patched_pipeline")
def test_process_photo_trims_keywords_when_max_new_keywords_set(tmp_path: Path) -> None:
    """max_new_keywords caps the AI-returned keyword list before merging."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(max_new_keywords=1)

    with patch(
        "photo_tagger.pipeline.analyze_image_with_ai",
        return_value=InferenceResult(
            title="T",
            description="D",
            keywords=["Alpha", "Beta", "Gamma"],
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            seconds=0.0,
        ),
    ):
        assert process_photo(image, _ctx(options=options)) is True


def test_process_photo_discards_existing_keywords_when_preserve_false(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """preserve_existing_kw=False replaces rather than merges existing keywords."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["New"],
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        seconds=0.0,
    )
    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(existing_keywords=KeywordSet(subject=["Old"])),
    ):
        process_photo(image, _ctx(options=ProcessingOptions(preserve_existing_kw=False)))

    written_kw = patched_pipeline["write"].call_args.args[1]
    assert "Old" not in written_kw.subject
    assert "New" in written_kw.subject


# ---------------------------------------------------------------------------
# _emit_outcome edge cases
# ---------------------------------------------------------------------------


def test_emit_outcome_emits_zeros_when_inference_is_absent(tmp_path: Path) -> None:
    """When process_photo crashes before setting scratch, outcome fields default to zero."""
    received: list[ImageOutcome] = []
    image = tmp_path / "img.cr3"
    image.write_text("x")

    _emit_outcome(received.append, image, {}, success=False, retry=False)

    assert len(received) == 1
    outcome = received[0]
    assert outcome.title is None
    assert outcome.keywords == []
    assert outcome.input_tokens == 0
    assert outcome.seconds == 0.0
    # The CSV-report fields fall back to empty too when the scratch is bare.
    assert outcome.written_keywords == []
    assert outcome.existing_keywords == []
    assert outcome.camera_info == {}
    assert outcome.location_tags == {}
    assert outcome.gps_position is None


def test_emit_outcome_includes_context_and_merged_keywords(tmp_path: Path) -> None:
    """A populated scratch flows EXIF, existing, and merged keywords into the outcome."""
    received: list[ImageOutcome] = []
    image = tmp_path / "img.cr3"
    image.write_text("x")
    scratch: _InferenceScratch = {
        "inference": InferenceResult(
            title="T",
            description="D",
            keywords=["Beach"],
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            seconds=0.5,
        ),
        "from_cache": True,
        "context": ImageContext(
            existing_keywords=KeywordSet(subject=["Old"]),
            location_tags={"XMP-photoshop:City": "Hamburg"},
            gps_position="53 N, 9 E",
            camera_info={"EXIF:Model": "Canon EOS R5"},
        ),
        "merged_keywords": KeywordSet(subject=["Old", "Beach"], hierarchical=["Nature|Beach"]),
    }

    _emit_outcome(received.append, image, scratch, success=True, retry=False)

    outcome = received[0]
    assert outcome.written_keywords == ["Old", "Beach"]
    assert outcome.hierarchical_keywords == ["Nature|Beach"]
    assert outcome.existing_keywords == ["Old"]
    assert outcome.camera_info == {"EXIF:Model": "Canon EOS R5"}
    assert outcome.location_tags == {"XMP-photoshop:City": "Hamburg"}
    assert outcome.gps_position == "53 N, 9 E"
    assert outcome.from_cache is True


@pytest.mark.usefixtures("patched_pipeline")
def test_process_photo_populates_outcome_sink(tmp_path: Path) -> None:
    """process_photo records context + merged keywords in the scratch for the CSV report."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    rich = ImageContext(
        existing_keywords=KeywordSet(subject=["Old"]),
        camera_info={"EXIF:Model": "Canon EOS R5"},
    )
    scratch: _InferenceScratch = {}
    with patch("photo_tagger.pipeline.read_image_context", return_value=rich):
        process_photo(image, _ctx(), outcome_sink=scratch)

    assert scratch["context"] is rich
    assert scratch["inference"].title == "Title"
    merged = scratch["merged_keywords"]
    assert "Old" in merged.subject
    assert "Beach" in merged.subject


def test_emit_outcome_swallows_callback_errors(tmp_path: Path) -> None:
    """A broken on_image_result callback must not propagate."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    def boom(_outcome: ImageOutcome) -> None:
        msg = "callback crashed"
        raise RuntimeError(msg)

    # Should not raise.
    _emit_outcome(boom, image, {}, success=True, retry=False)


def test_emit_outcome_noop_when_callback_is_none(tmp_path: Path) -> None:
    """Passing None as the callback is the normal no-op path."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    _emit_outcome(None, image, {}, success=True, retry=False)


# ---------------------------------------------------------------------------
# _cache_lookup hash failure
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patched_pipeline")
def test_process_photo_survives_hash_failure(tmp_path: Path) -> None:
    """A broken hash_image_file is treated as a cache miss with no put attempt."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    class _HashFailCache:
        def __init__(self) -> None:
            self.put_calls = 0

        def get(self, _key: str) -> None:
            return None

        def put(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            self.put_calls += 1

    cache = _HashFailCache()
    with patch("photo_tagger.pipeline.hash_image_file", side_effect=OSError("broken")):
        ok = process_photo(image, _ctx(cache=cache))

    assert ok is True
    # hash failed -> cache_key is None -> put must be skipped.
    assert cache.put_calls == 0


# ---------------------------------------------------------------------------
# Concurrent worker exception path
# ---------------------------------------------------------------------------


def test_run_batch_concurrent_records_worker_exception(tmp_path: Path) -> None:
    """An exception raised inside a worker thread is caught and counted as a failure."""
    files = [tmp_path / f"img{i}.cr3" for i in range(2)]
    for f in files:
        f.write_text("x")

    def _exploding_process(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401
        msg = "worker blew up"
        raise RuntimeError(msg)

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=_exploding_process),
        pytest.raises(BatchError),
    ):
        run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions(), workers=2)


def test_notify_success_is_a_noop_without_callback(tmp_path: Path) -> None:
    """_notify_success returns immediately when no on_success callback is registered."""
    # Must not raise and must not require a callable.
    _notify_success(None, tmp_path / "img.cr3")


def test_run_batch_concurrent_catches_future_result_exception(tmp_path: Path) -> None:
    """An error from future.result() (execute_process itself raising) counts as a failure."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_CONCURRENT_BATCH_SIZE)]
    for f in files:
        f.write_text("x")

    def _exploding_execute(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401
        msg = "execute_process blew up"
        raise RuntimeError(msg)

    with (
        patch("photo_tagger.pipeline.execute_process", side_effect=_exploding_execute),
        pytest.raises(BatchError),
    ):
        run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions(), workers=2)


def test_run_batch_concurrent_progress_callback_fires_per_image(tmp_path: Path) -> None:
    """Progress fires once per image on the concurrent path too."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_CONCURRENT_BATCH_SIZE)]
    for f in files:
        f.write_text("x")
    received: list[tuple[str, bool]] = []

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            progress=lambda path, ok: received.append((path.name, ok)),
            workers=2,
        )

    assert sorted(received) == sorted((f.name, True) for f in files)


def test_process_photo_trims_to_max_new_keywords(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """max_new_keywords caps the AI keyword list before merging with existing tags."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    options = ProcessingOptions(max_new_keywords=2)
    assert process_photo(image, _ctx(options=options)) is True

    write_call = patched_pipeline["write"].call_args
    keywords = write_call.args[1]
    # The stubbed AI returns ["Beach", "Sunset"] (2 items), which is at the cap.
    # Nothing is trimmed at 2, but the path is still exercised.
    assert len(keywords.subject) <= 2  # noqa: PLR2004 - matches the cap


def test_process_photo_trims_when_exceeding_max_new_keywords(
    tmp_path: Path,
) -> None:
    """When AI returns more keywords than the cap, the excess is dropped before merge."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    many_keywords = [f"kw-{i}" for i in range(10)]

    with (
        patch(
            "photo_tagger.pipeline.prepare_image_for_agent",
            return_value=BinaryContent(data=b"\xff\xd8stub", media_type="image/jpeg"),
        ),
        patch("photo_tagger.pipeline.read_image_context", return_value=ImageContext()),
        patch(
            "photo_tagger.pipeline.analyze_image_with_ai",
            return_value=InferenceResult(
                title="Title",
                description="Description.",
                keywords=many_keywords,
            ),
        ),
        patch("photo_tagger.pipeline.write_metadata", return_value=True) as write,
    ):
        options = ProcessingOptions(max_new_keywords=3)
        assert process_photo(image, _ctx(options=options)) is True

        write_call = write.call_args
        keywords = write_call.args[1]
        # Only the first 3 AI keywords survive the cap, then merge happens.
        assert "Kw-0" in keywords.subject
        assert "Kw-2" in keywords.subject
        # The 4th keyword (index 3) should NOT be present.
        assert "Kw-3" not in keywords.subject
