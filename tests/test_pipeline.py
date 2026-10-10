"""Tests for the photo processing pipeline using lightweight stubs."""

import contextlib
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelHTTPError

from photo_tagger.ai import _attach_partial_usage
from photo_tagger.errors import BatchError
from photo_tagger.failures import FAILURE_METADATA_WRITE, FAILURE_MODEL_API
from photo_tagger.fields import ExtraFields
from photo_tagger.metadata import FieldSources, ImageContext
from photo_tagger.models import InferenceResult, KeywordSet
from photo_tagger.pacing import Pacer, PacingPlan
from photo_tagger.pipeline import (
    ImageOutcome,
    ProcessingOptions,
    _BatchContext,
    _drain_after_interrupt,
    _emit_outcome,
    _notify_success,
    _PhotoRecord,
    _resolve_inference,
    _UsageAccumulator,
    execute_process,
    process_photo,
    run_batch,
)
from photo_tagger.sessions import SessionPlan
from photo_tagger.undo import UndoJournal
from photo_tagger.vocabulary import Vocabulary


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
        patch("photo_tagger.metadata.write_metadata", return_value=True) as write,
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
    # Every equality assert in this file is actual-first. The only literal-first comparisons are
    # membership tests (`"Beach" in ...`), which cannot be written the other way round.
    assert kwargs["description"] == "Description."  # NOSONAR S3415
    assert kwargs["title"] == "Title"
    assert kwargs["use_sidecar"] is True
    keywords = write_call.args[1]
    assert "Beach" in keywords.subject


@pytest.mark.parametrize(
    ("name", "sidecar"),
    [("img.dng", True), ("img.jpg", False)],
)
def test_process_photo_raw_mode_picks_the_target_per_photo(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
    name: str,
    *,
    sidecar: bool,
) -> None:
    """One run over a mixed folder: the DNG gets a sidecar, the JPEG is written in place."""
    image = tmp_path / name
    image.write_text("x")

    assert process_photo(image, _ctx(options=ProcessingOptions(sidecar_mode="raw"))) is True

    assert patched_pipeline["write"].call_args.kwargs["use_sidecar"] is sidecar


def test_process_photo_both_mode_writes_the_photo_and_the_sidecar(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Mode "both" writes each photo twice, with the payload a sidecar can hold in the .xmp."""
    image = tmp_path / "img.dng"
    image.write_text("x")

    assert process_photo(image, _ctx(options=ProcessingOptions(sidecar_mode="both"))) is True

    calls = patched_pipeline["write"].call_args_list
    assert [call.kwargs["use_sidecar"] for call in calls] == [False, True]


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


@pytest.mark.parametrize(
    ("existing", "written"),
    [
        (("Camera Title", "default"), (None, None)),
        ((None, None), ("Title", "Description.")),
    ],
    ids=["photo-has-both", "photo-has-neither"],
)
def test_process_photo_keeps_a_preserved_caption_the_photo_already_has(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
    existing: tuple[str | None, str | None],
    written: tuple[str | None, str | None],
) -> None:
    """Preserving a field keeps what the photo has, and still fills it in where there is none."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(
        preserve_existing_title=True,
        preserve_existing_description=True,
    )
    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(existing_title=existing[0], existing_description=existing[1]),
    ):
        process_photo(image, _ctx(options=options))

    kwargs = patched_pipeline["write"].call_args.kwargs
    assert (kwargs["title"], kwargs["description"]) == written


def test_process_photo_replaces_an_existing_caption_by_default(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Each field answers for itself: keywords merge while the placeholder caption is replaced."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(
            existing_keywords=KeywordSet(subject=["Old"]),
            existing_description="default",
        ),
    ):
        process_photo(image, _ctx(options=ProcessingOptions()))

    write_call = patched_pipeline["write"].call_args
    assert write_call.kwargs["description"] == "Description."
    assert "Old" in write_call.args[1].subject


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


def test_run_model_folds_partial_usage_when_the_call_fails(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """
    Tokens burned by attempts that ultimately failed still land in the batch's usage totals.

    Regression test: _run_model only called ctx.usage.add() on success, so a run that raised after
    billing several internally-retried attempts silently lost that cost from the summary.
    """
    image = tmp_path / "img.cr3"
    image.write_text("x")
    usage = _UsageAccumulator()

    def boom(**_kwargs: Any) -> InferenceResult:  # noqa: ANN401
        exc = ValueError("model returned invalid structured output")
        _attach_partial_usage(
            exc,
            SimpleNamespace(input_tokens=50, output_tokens=10, total_tokens=60),  # type: ignore[arg-type]
        )
        raise exc

    patched_pipeline["analyze"].side_effect = boom
    ok = execute_process(image, _ctx(usage=usage), index="1/1")

    expected_input = 50
    expected_output = 10
    expected_total = 60
    assert ok is False
    assert usage.input_tokens == expected_input
    assert usage.output_tokens == expected_output
    assert usage.total_tokens == expected_total
    assert usage.inference_calls == 0  # not a completed, usable call


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

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(BatchError),
    ):
        run_batch(files, agent=_FAKE_AGENT, options=options)


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


def test_run_batch_pauses_before_the_retry_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The retry pass waits instead of immediately re-hitting a struggling server.

    No pause happens when the first pass was clean (nothing pending to retry).
    """
    monkeypatch.setattr("photo_tagger.pipeline._RETRY_PASS_DELAY_SECONDS", 5.0)
    sleeps: list[float] = []
    monkeypatch.setattr("photo_tagger.pipeline.time.sleep", sleeps.append)
    image = tmp_path / "img.cr3"
    image.write_text("x")

    calls = {"n": 0}

    def fail_once(*_a: Any, **_kw: Any) -> bool:  # noqa: ANN401
        calls["n"] += 1
        return calls["n"] > 1

    with patch("photo_tagger.pipeline.process_photo", side_effect=fail_once):
        run_batch([image], agent=_FAKE_AGENT, options=ProcessingOptions())
    assert sleeps == [5.0]

    sleeps.clear()
    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        run_batch([image], agent=_FAKE_AGENT, options=ProcessingOptions())
    assert sleeps == []


def test_run_batch_skips_retry_for_a_rejected_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A 401/403 from the provider is not retried: waiting and asking again cannot fix it.

    Regression test: the retry pass used to retry every first-pass failure uniformly, so a batch
    failing on a bad or revoked API key paid for a second full round of doomed requests before
    reporting.
    """
    monkeypatch.setattr("photo_tagger.pipeline._RETRY_PASS_DELAY_SECONDS", 5.0)
    sleeps: list[float] = []
    monkeypatch.setattr("photo_tagger.pipeline.time.sleep", sleeps.append)
    image = tmp_path / "img.cr3"
    image.write_text("x")
    calls = {"n": 0}

    def unauthorized(*_a: Any, **_kw: Any) -> bool:  # noqa: ANN401
        calls["n"] += 1
        raise ModelHTTPError(status_code=401, model_name="test-model")

    received: list[Any] = []
    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=unauthorized),
        pytest.raises(BatchError),
    ):
        run_batch(
            [image],
            agent=_FAKE_AGENT,
            options=options,
            on_complete=received.append,
        )

    assert calls["n"] == 1  # never retried
    assert sleeps == []  # no pause paid for a doomed retry
    totals = received[0]
    assert totals.failed_files == [str(image)]
    assert totals.retry_successes == 0
    assert totals.failure_kinds == {FAILURE_MODEL_API: 1}
    assert totals.failure_types == {"ModelHTTPError(401)": 1}


def test_run_batch_still_reports_on_ctrl_c_during_the_retry_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A Ctrl-C during the pre-retry pause still emits a BatchTotals, like one mid-pass does.

    Regression test: the sleep before the retry pass sat outside any KeyboardInterrupt handling,
    unlike the passes on either side of it, so this exact window could propagate the interrupt
    straight out of run_batch, skipping on_complete and the summary file entirely.
    """
    monkeypatch.setattr("photo_tagger.pipeline._RETRY_PASS_DELAY_SECONDS", 5.0)

    def interrupted_sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("photo_tagger.pipeline.time.sleep", interrupted_sleep)
    image = tmp_path / "img.cr3"
    image.write_text("x")

    received: list[Any] = []
    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(BatchError),
    ):
        run_batch(
            [image],
            agent=_FAKE_AGENT,
            options=options,
            on_complete=received.append,
        )

    assert len(received) == 1
    totals = received[0]
    assert totals.failed_files == [str(image)]
    assert totals.retry_successes == 0


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

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo),
        pytest.raises(BatchError),
    ):
        run_batch(
            [success_first, success_retry, failure],
            agent=_FAKE_AGENT,
            options=options,
            on_success=notified.append,
        )

    assert notified == [success_first, success_retry]


def test_run_batch_dry_run_never_fires_on_success(tmp_path: Path) -> None:
    """
    A dry run must not report photos as done to the caller's callback.

    Regression test: the CLI wires on_success to the --append-to-skip-file appender, and a dry run
    used to append every previewed photo, so a later real run silently skipped them.
    """
    image = tmp_path / "img.cr3"
    image.write_text("x")

    notified: list[Path] = []
    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        totals = run_batch(
            [image],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(dry_run=True),
            on_success=notified.append,
        )

    assert totals.success == 1
    assert notified == []


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


def test_run_batch_reports_failure_kinds_in_totals(tmp_path: Path) -> None:
    """Final failures land in BatchTotals.failure_kinds, bucketed by cause."""
    timeout_file = tmp_path / "slow.cr3"
    write_file = tmp_path / "readonly.cr3"
    for path in (timeout_file, write_file):
        path.write_text("x")

    def fake_process_photo(image: Path, *_a: Any, **_kw: Any) -> bool:  # noqa: ANN401
        if image.name == timeout_file.name:
            raise TimeoutError
        return False  # metadata write failed

    received: list[Any] = []
    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo),
        pytest.raises(BatchError),
    ):
        run_batch(
            [timeout_file, write_file],
            agent=_FAKE_AGENT,
            options=options,
            on_complete=received.append,
        )

    assert received[0].failure_kinds == {"timeout": 1, "metadata-write": 1}
    # A write that returned False raised nothing, so only the timeout has a type to report.
    assert received[0].failure_types == {"TimeoutError": 1}


class _DictCache:
    """A dict-backed stand-in for InferenceCache used by the stampede tests."""

    def __init__(self) -> None:
        self.store: dict[str, InferenceResult] = {}
        # Set on the second miss: the follower has checked the cache and is heading for the
        # in-flight coordination, which pins the interleaving the tests mean to exercise.
        self.follower_missed = threading.Event()
        self._misses = 0

    def get(self, key: str) -> InferenceResult | None:
        result = self.store.get(key)
        if result is None:
            self._misses += 1
            if self._misses >= 2:  # noqa: PLR2004 - the second miss is the follower's
                self.follower_missed.set()
        return result

    def put(self, key: str, result: InferenceResult) -> None:
        self.store[key] = result


def test_resolve_inference_coordinates_duplicate_content(tmp_path: Path) -> None:
    """
    Concurrent misses on the same content key share one model call.

    Regression test: two workers with identical pixels (burst duplicates, one image in two folders)
    used to both miss the cache and each pay a full inference; only the last put mattered. The first
    worker now leads, the second waits and replays the cache. The leader is held until the follower
    has demonstrably missed the cache, so this exercises the real wait-then-replay path rather than
    a lucky plain cache hit.
    """
    calls = {"n": 0}
    leader_started = threading.Event()
    release_leader = threading.Event()

    def slow_analyze(**_kwargs: Any) -> InferenceResult:  # noqa: ANN401
        calls["n"] += 1
        leader_started.set()
        release_leader.wait(timeout=5.0)
        return InferenceResult(title="T", description="D", keywords=["K"])

    cache = _DictCache()
    ctx = _ctx(cache=cache)
    with (
        patch("photo_tagger.pipeline.analyze_image_with_ai", side_effect=slow_analyze),
        patch("photo_tagger.pipeline.prepare_image_for_agent", return_value=b"jpeg"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        leader = pool.submit(
            _resolve_inference,
            tmp_path / "a.cr3",
            ctx,
            contextual_prompt="p",
            content_key="same-pixels",
        )
        assert leader_started.wait(timeout=5.0)
        follower = pool.submit(
            _resolve_inference,
            tmp_path / "b.cr3",
            ctx,
            contextual_prompt="p",
            content_key="same-pixels",
        )
        # Only release the leader once the follower is provably past its cache miss (and thus
        # waiting on the in-flight event, not racing toward a plain hit).
        assert cache.follower_missed.wait(timeout=5.0)
        release_leader.set()
        leader_result, leader_from_cache = leader.result(timeout=10.0)
        follower_result, follower_from_cache = follower.result(timeout=10.0)

    assert calls["n"] == 1  # one model call for two photos
    assert leader_from_cache is False
    assert follower_from_cache is True
    assert follower_result.title == leader_result.title == "T"
    assert ctx.usage.cache_hits == 1
    assert ctx.inflight == {}  # the coordination entry was cleaned up


def test_resolve_inference_follower_falls_back_when_leader_fails(tmp_path: Path) -> None:
    """A waiting follower pays its own model call when the leader's inference raises."""
    calls = {"n": 0}
    leader_started = threading.Event()
    release_leader = threading.Event()

    def flaky_analyze(**_kwargs: Any) -> InferenceResult:  # noqa: ANN401
        calls["n"] += 1
        if calls["n"] == 1:
            leader_started.set()
            release_leader.wait(timeout=5.0)
            msg = "model exploded"
            raise RuntimeError(msg)
        return InferenceResult(title="Recovered", description="D", keywords=[])

    ctx = _ctx(cache=_DictCache())
    with (
        patch("photo_tagger.pipeline.analyze_image_with_ai", side_effect=flaky_analyze),
        patch("photo_tagger.pipeline.prepare_image_for_agent", return_value=b"jpeg"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        leader = pool.submit(
            _resolve_inference,
            tmp_path / "a.cr3",
            ctx,
            contextual_prompt="p",
            content_key="same-pixels",
        )
        assert leader_started.wait(timeout=5.0)
        follower = pool.submit(
            _resolve_inference,
            tmp_path / "b.cr3",
            ctx,
            contextual_prompt="p",
            content_key="same-pixels",
        )
        release_leader.set()
        assert isinstance(leader.exception(timeout=10.0), RuntimeError)
        follower_result, follower_from_cache = follower.result(timeout=10.0)

    assert calls["n"] == 2  # noqa: PLR2004 - leader failed, follower paid its own call
    assert follower_from_cache is False
    assert follower_result.title == "Recovered"
    assert ctx.inflight == {}


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

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(
            BatchError,
        ),
    ):
        run_batch(
            images,
            agent=_FAKE_AGENT,
            options=options,
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

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
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

    options = ProcessingOptions(sidecar_mode="none")
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
    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            on_complete=received_totals.append,
        )

    totals = received_totals[0]
    assert totals.success == interrupt_after
    # The two photos after the first should appear as failed so a future
    # --skip-from rerun can pick them up. on_complete still fires so a
    # --summary-file is written even when the run was aborted.
    assert len(totals.failed_files) >= 1
    assert totals.interrupted is True


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
    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=fake_process_photo),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            on_complete=received_totals.append,
            workers=2,
        )

    assert len(received_totals) == 1
    totals = received_totals[0]
    # Some files were processed, some should be in failed. The exact split
    # depends on scheduling, but we must see at least one failure and on_complete
    # must have fired so the summary file is written.
    assert len(totals.failed_files) >= 1


def test_run_batch_concurrent_interrupt_during_submission(tmp_path: Path) -> None:
    """
    Ctrl-C while futures are still being queued takes the cancel-and-drain path.

    Regression test: submission used to sit outside the KeyboardInterrupt handler, so an early
    Ctrl-C fell through to the blocking shutdown(wait=True) and the batch ground on to the end with
    no summary accounting.
    """
    files = [tmp_path / f"img{i}.cr3" for i in range(4)]
    for f in files:
        f.write_text("x")
    interrupt_after = 2

    class InterruptingPool(ThreadPoolExecutor):
        submissions = 0

        def submit(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            type(self).submissions += 1
            if type(self).submissions > interrupt_after:
                raise KeyboardInterrupt
            return super().submit(*args, **kwargs)

    received_totals: list[Any] = []
    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.ThreadPoolExecutor", InterruptingPool),
        patch("photo_tagger.pipeline.process_photo", return_value=True),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            on_complete=received_totals.append,
            workers=2,
        )

    totals = received_totals[0]
    # The two submitted photos settle during the drain; the two never-submitted ones must be
    # reported as failed so a --skip-from rerun picks them up.
    assert totals.success == interrupt_after
    assert len(totals.failed_files) == len(files) - interrupt_after


# ---------------------------------------------------------------------------
# process_photo edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cap", "expected_subjects"),
    [
        (1, ["Alpha"]),  # below the count: excess dropped
        (3, ["Alpha", "Beta", "Gamma"]),  # exactly at the count: nothing trimmed
        (5, ["Alpha", "Beta", "Gamma"]),  # above the count: nothing trimmed
        (None, ["Alpha", "Beta", "Gamma"]),  # no cap configured
    ],
)
def test_process_photo_caps_ai_keywords_before_merging(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
    cap: int | None,
    expected_subjects: list[str],
) -> None:
    """max_new_keywords trims the AI keyword list (and only it) before merging, order kept."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["Alpha", "Beta", "Gamma"],
    )

    options = ProcessingOptions(max_new_keywords=cap)
    assert process_photo(image, _ctx(options=options)) is True

    written = patched_pipeline["write"].call_args.args[1]
    assert written.subject == expected_subjects


def test_process_photo_snaps_keywords_onto_the_vocabulary(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A near-miss keyword is written with the catalog's spelling and hierarchy."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["ospreys", "Golden Hour"],
    )

    options = ProcessingOptions(vocabulary=Vocabulary.from_entries(["Animal|Bird|Osprey"]))
    usage = _UsageAccumulator()
    assert process_photo(image, _ctx(options=options, usage=usage)) is True

    written = patched_pipeline["write"].call_args.args[1]
    assert written.subject == ["Animal", "Bird", "Osprey", "Golden Hour"]
    assert written.hierarchical == ["Animal|Bird", "Animal|Bird|Osprey"]
    assert usage.vocabulary_mapped == 1
    assert usage.vocabulary_dropped == {}


def test_process_photo_drops_unknown_keywords_in_strict_mode(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Strict mode writes only catalog terms and records what it rejected."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["Osprey", "Golden Hour"],
    )

    options = ProcessingOptions(
        vocabulary=Vocabulary.from_entries(["Osprey"]),
        vocabulary_strict=True,
    )
    usage = _UsageAccumulator()
    assert process_photo(image, _ctx(options=options, usage=usage)) is True

    written = patched_pipeline["write"].call_args.args[1]
    assert written.subject == ["Osprey"]
    assert usage.vocabulary_dropped == {"Golden Hour": 1}


def test_process_photo_applies_the_vocabulary_before_the_cap(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The cap counts keywords that survive the vocabulary, not ones about to be dropped."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["Unknown One", "Unknown Two", "Osprey", "Mallard"],
    )

    options = ProcessingOptions(
        vocabulary=Vocabulary.from_entries(["Osprey", "Mallard"]),
        vocabulary_strict=True,
        max_new_keywords=2,
    )
    assert process_photo(image, _ctx(options=options)) is True

    written = patched_pipeline["write"].call_args.args[1]
    assert written.subject == ["Osprey", "Mallard"]


def test_process_photo_leaves_keywords_alone_without_a_vocabulary(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The snap is skipped entirely when no vocabulary is configured."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    usage = _UsageAccumulator()

    assert process_photo(image, _ctx(usage=usage)) is True

    written = patched_pipeline["write"].call_args.args[1]
    assert written.subject == ["Beach", "Sunset"]
    assert usage.vocabulary_mapped == 0


def test_usage_accumulator_stops_tracking_dropped_terms_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run against the wrong vocabulary cannot grow the report without bound."""
    monkeypatch.setattr("photo_tagger.vocabulary.MAX_TRACKED_DROPPED_TERMS", 2)
    usage = _UsageAccumulator()

    usage.add_vocabulary(mapped=0, dropped=["A", "B", "C"])
    usage.add_vocabulary(mapped=1, dropped=["A", "D"])

    assert usage.vocabulary_dropped == {"A": 2, "B": 1}
    assert usage.vocabulary_mapped == 1


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

    _emit_outcome(received.append, image, _PhotoRecord(), success=False, retry=False)

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
    scratch = _PhotoRecord(
        inference=InferenceResult(
            title="T",
            description="D",
            keywords=["Beach"],
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            seconds=0.5,
        ),
        from_cache=True,
        context=ImageContext(
            existing_keywords=KeywordSet(subject=["Old"]),
            location_tags={"XMP-photoshop:City": "Hamburg"},
            gps_position="53 N, 9 E",
            camera_info={"EXIF:Model": "Canon EOS R5"},
        ),
        merged_keywords=KeywordSet(subject=["Old", "Beach"], hierarchical=["Nature|Beach"]),
    )

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
    scratch = _PhotoRecord()
    with patch("photo_tagger.pipeline.read_image_context", return_value=rich):
        process_photo(image, _ctx(), record=scratch)

    assert scratch.context is rich
    assert scratch.inference is not None
    assert scratch.inference.title == "Title"
    merged = scratch.merged_keywords
    assert merged is not None
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
    _emit_outcome(boom, image, _PhotoRecord(), success=True, retry=False)


def test_emit_outcome_noop_when_callback_is_none(tmp_path: Path) -> None:
    """Passing None as the callback is the normal no-op path."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    _emit_outcome(None, image, _PhotoRecord(), success=True, retry=False)


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
    with patch("photo_tagger.cache.hash_image_file", side_effect=OSError("broken")):
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

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=_exploding_process),
        pytest.raises(BatchError),
    ):
        run_batch(files, agent=_FAKE_AGENT, options=options, workers=2)


def test_notify_success_is_a_noop_without_callback(tmp_path: Path) -> None:
    """_notify_success returns immediately when no on_success callback is registered."""
    # Must not raise and must not require a callable.
    _notify_success(_ctx(), tmp_path / "img.cr3")


def test_notify_success_stays_silent_while_a_session_is_pending(tmp_path: Path) -> None:
    """Nothing is on disk yet during a session's analysis, so the resume list must not grow."""
    notified: list[Path] = []
    ctx = _ctx()
    ctx.on_success = notified.append
    ctx.pending = {}

    _notify_success(ctx, tmp_path / "img.cr3")
    assert notified == []

    ctx.pending = None
    _notify_success(ctx, tmp_path / "img.cr3")
    assert notified == [tmp_path / "img.cr3"]


def test_drain_after_interrupt_counts_completed_work(tmp_path: Path) -> None:
    """Photos finished during the interrupt drain are successes; cancelled ones stay pending."""
    done_ok: Future[bool] = Future()
    done_ok.set_result(True)
    done_bad: Future[bool] = Future()
    done_bad.set_result(False)
    exploded: Future[bool] = Future()
    exploded.set_exception(RuntimeError("worker blew up"))
    never_started: Future[bool] = Future()
    never_started.cancel()

    notified: list[Path] = []
    ctx = _ctx()
    ctx.on_success = notified.append
    successes, failures, cancelled = _drain_after_interrupt(
        {
            done_ok: tmp_path / "ok.cr3",
            done_bad: tmp_path / "bad.cr3",
            exploded: tmp_path / "boom.cr3",
            never_started: tmp_path / "pending.cr3",
        },
        ctx,
    )

    assert successes == 1
    assert notified == [tmp_path / "ok.cr3"]
    assert failures == [tmp_path / "bad.cr3", tmp_path / "boom.cr3"]
    assert cancelled == [tmp_path / "pending.cr3"]


def test_run_batch_concurrent_catches_future_result_exception(tmp_path: Path) -> None:
    """An error from future.result() (execute_process itself raising) counts as a failure."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_CONCURRENT_BATCH_SIZE)]
    for f in files:
        f.write_text("x")

    def _exploding_execute(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401
        msg = "execute_process blew up"
        raise RuntimeError(msg)

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.execute_process", side_effect=_exploding_execute),
        pytest.raises(BatchError),
    ):
        run_batch(files, agent=_FAKE_AGENT, options=options, workers=2)


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


# The keyword-cap behavior is covered by the parametrized
# test_process_photo_caps_ai_keywords_before_merging above (below/at/above the count, and None).


# ---------------------------------------------------------------------------
# Session mode: analyze a whole shoot, harmonize it, then write
# ---------------------------------------------------------------------------


# The pipeline's own process_photo, kept aside so tests can patch the name while still running
# the real thing underneath (session tests need the analyze/write split, not a stub).
_real_process_photo = process_photo


def _session_plan(files: list[Path]) -> SessionPlan:
    """Put every file in one session, as plan_sessions would for a single shoot."""
    return SessionPlan(sessions=[list(files)], index_of=dict.fromkeys(files, 0))


@pytest.fixture
def two_photo_session(tmp_path: Path, patched_pipeline: dict[str, Any]) -> dict[str, Any]:
    """Two photos of one subject that the model described inconsistently."""
    first = tmp_path / "a.cr3"
    second = tmp_path / "b.cr3"
    for path in (first, second):
        path.write_text("x")
    by_file = {
        first: InferenceResult(title="T1", description="D1", keywords=["Osprey<Bird<Animal"]),
        second: InferenceResult(title="T2", description="D2", keywords=["ospreys<Raptor"]),
    }
    written: list[tuple[Path, KeywordSet]] = []
    current: list[Path] = [first]

    def fake_write(image_path: Path, keywords: KeywordSet, **_kwargs: Any) -> bool:  # noqa: ANN401
        written.append((image_path, keywords))
        return True

    patched_pipeline["write"].side_effect = fake_write
    # process_photo runs per file; hand each one the inference recorded for it.
    patched_pipeline["analyze"].side_effect = lambda **_kw: by_file[current[0]]

    def process(path: Path, ctx: _BatchContext, **kwargs: Any) -> bool:  # noqa: ANN401
        current[0] = path
        return _real_process_photo(path, ctx, **kwargs)

    return {"files": [first, second], "written": written, "process": process}


def test_run_batch_harmonizes_a_session_before_writing(
    two_photo_session: dict[str, Any],
) -> None:
    """Both photos of one shoot are written with the session's majority term and hierarchy."""
    files = two_photo_session["files"]
    with patch("photo_tagger.pipeline.process_photo", side_effect=two_photo_session["process"]):
        totals = run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            session_plan=_session_plan(files),
        )

    assert totals.success == len(files)
    written = dict(two_photo_session["written"])
    for path in files:
        assert written[path].subject == ["Animal", "Bird", "Osprey"]
        assert written[path].hierarchical == ["Animal|Bird", "Animal|Bird|Osprey"]


def test_run_batch_session_defers_success_until_the_write(
    two_photo_session: dict[str, Any],
) -> None:
    """The resume skip list and the per-photo outcome only fire once the file is on disk."""
    files = two_photo_session["files"]
    notified: list[Path] = []
    outcomes: list[ImageOutcome] = []

    def failing_write(image_path: Path, _keywords: KeywordSet, **_kwargs: Any) -> bool:  # noqa: ANN401
        return image_path != files[1]

    options = ProcessingOptions()
    plan = _session_plan(files)

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=two_photo_session["process"]),
        patch("photo_tagger.metadata.write_metadata", side_effect=failing_write),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            on_success=notified.append,
            on_image_result=outcomes.append,
            session_plan=plan,
        )

    assert notified == [files[0]]
    assert [(o.file, o.success) for o in outcomes] == [(files[0], True), (files[1], False)]
    # The outcome carries the merged keywords, which only exist after the deferred write.
    assert outcomes[0].written_keywords == ["Animal", "Bird", "Osprey"]


def test_run_batch_session_write_failure_is_final(
    two_photo_session: dict[str, Any],
) -> None:
    """A failed write is not re-inferred: the model work is done and exiftool is the problem."""
    files = two_photo_session["files"]
    calls: list[Path] = []

    def counting_process(path: Path, ctx: _BatchContext, **kwargs: Any) -> bool:  # noqa: ANN401
        calls.append(path)
        return bool(two_photo_session["process"](path, ctx, **kwargs))

    options = ProcessingOptions()
    plan = _session_plan(files)

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=counting_process),
        patch("photo_tagger.metadata.write_metadata", return_value=False),
        pytest.raises(BatchError) as batch_error,
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            session_plan=plan,
        )

    totals = batch_error.value.args[0]
    assert calls == files  # analyzed once each, never retried
    assert totals.success == 0
    assert totals.failure_kinds == {FAILURE_METADATA_WRITE: len(files)}
    assert sorted(totals.failed_files) == sorted(str(f) for f in files)


def test_run_batch_session_progress_reports_the_write_not_the_analysis(
    two_photo_session: dict[str, Any],
) -> None:
    """
    In session mode a photo is finished when it is written, so that is when the bar moves.

    Ticking at analysis time reported a photo as done and successful while its write was still to
    come, which for a write that then failed is a green bar over a failed run.
    """
    files = two_photo_session["files"]
    ticks: list[tuple[Path, bool]] = []

    options = ProcessingOptions()
    plan = _session_plan(files)

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=two_photo_session["process"]),
        patch("photo_tagger.metadata.write_metadata", return_value=False),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            session_plan=plan,
            progress=lambda path, ok: ticks.append((path, ok)),
        )

    # Once per photo, and each one says the write failed.
    assert ticks == [(files[0], False), (files[1], False)]


def test_run_batch_session_progress_ticks_once_per_written_photo(
    two_photo_session: dict[str, Any],
) -> None:
    """The bar still reaches 100% on a healthy session, without overshooting."""
    files = two_photo_session["files"]
    ticks: list[tuple[Path, bool]] = []

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=two_photo_session["process"]),
        patch("photo_tagger.metadata.write_metadata", return_value=True),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            session_plan=_session_plan(files),
            progress=lambda path, ok: ticks.append((path, ok)),
        )

    assert ticks == [(files[0], True), (files[1], True)]


def test_run_batch_session_retry_reuses_the_session_vocabulary(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A photo that only succeeds on retry still lands on its shoot's agreed terms."""
    steady = tmp_path / "steady.cr3"
    flaky = tmp_path / "flaky.cr3"
    for path in (steady, flaky):
        path.write_text("x")
    written: dict[Path, KeywordSet] = {}
    attempts = {"flaky": 0}

    def fake_write(image_path: Path, keywords: KeywordSet, **_kwargs: Any) -> bool:  # noqa: ANN401
        written[image_path] = keywords
        return True

    patched_pipeline["write"].side_effect = fake_write

    def process(path: Path, ctx: _BatchContext, **kwargs: Any) -> bool:  # noqa: ANN401
        if path == flaky:
            attempts["flaky"] += 1
            if attempts["flaky"] == 1:
                msg = "model hiccup"
                raise RuntimeError(msg)
            patched_pipeline["analyze"].return_value = InferenceResult(
                title="T",
                description="D",
                keywords=["ospreys<Raptor"],
            )
        else:
            patched_pipeline["analyze"].return_value = InferenceResult(
                title="T",
                description="D",
                keywords=["Osprey<Bird<Animal"],
            )
        return _real_process_photo(path, ctx, **kwargs)

    with patch("photo_tagger.pipeline.process_photo", side_effect=process):
        totals = run_batch(
            [steady, flaky],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(),
            session_plan=_session_plan([steady, flaky]),
        )

    assert totals.retry_successes == 1
    assert written[flaky].subject == ["Animal", "Bird", "Osprey"]


def test_run_batch_session_skips_empty_and_filtered_sessions(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A plan that mentions photos outside this batch processes only the batch's own."""
    kept = tmp_path / "kept.cr3"
    absent = tmp_path / "absent.cr3"
    kept.write_text("x")
    patched_pipeline["write"].return_value = True
    plan = SessionPlan(sessions=[[absent], [kept]], index_of={absent: 0, kept: 1})

    totals = run_batch(
        [kept],
        agent=_FAKE_AGENT,
        options=ProcessingOptions(),
        session_plan=plan,
    )

    assert totals.success == 1
    assert totals.total_files == 1


def test_run_batch_reports_what_the_vocabulary_changed(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The totals name the rewrites and the rejections, which is what the summary file shows."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["ospreys", "Golden Hour"],
    )

    totals = run_batch(
        [image],
        agent=_FAKE_AGENT,
        options=ProcessingOptions(
            vocabulary=Vocabulary.from_entries(["Osprey"]),
            vocabulary_strict=True,
        ),
    )

    assert totals.vocabulary_mapped == 1
    assert totals.vocabulary_dropped == {"Golden Hour": 1}


def test_run_batch_leaves_the_vocabulary_totals_at_zero_when_nothing_changed(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Keywords that already match the catalog are not reported as rewrites."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["Osprey"],
    )

    totals = run_batch(
        [image],
        agent=_FAKE_AGENT,
        options=ProcessingOptions(vocabulary=Vocabulary.from_entries(["Osprey"])),
    )

    assert totals.vocabulary_mapped == 0
    assert totals.vocabulary_dropped == {}


def test_run_batch_session_with_no_usable_analysis_writes_nothing(tmp_path: Path) -> None:
    """When every photo in a session fails, there is nothing to harmonize and nothing to write."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    options = ProcessingOptions()
    plan = _session_plan([image])

    with (
        patch("photo_tagger.pipeline.process_photo", return_value=False),
        patch("photo_tagger.metadata.write_metadata") as write,
        pytest.raises(BatchError),
    ):
        run_batch(
            [image],
            agent=_FAKE_AGENT,
            options=options,
            session_plan=plan,
        )

    write.assert_not_called()


def test_run_batch_session_interrupt_stops_after_the_session_in_flight(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Ctrl-C during a session still writes what that session analyzed, and skips the rest."""
    first = tmp_path / "a.cr3"
    second = tmp_path / "b.cr3"
    for path in (first, second):
        path.write_text("x")
    patched_pipeline["write"].return_value = True
    plan = SessionPlan(sessions=[[first], [second]], index_of={first: 0, second: 1})

    def process(path: Path, ctx: _BatchContext, **kwargs: Any) -> bool:  # noqa: ANN401
        if path == second:  # pragma: no cover - the interrupt lands before this session runs
            msg = "should not be reached"
            raise AssertionError(msg)
        _real_process_photo(path, ctx, **kwargs)
        raise KeyboardInterrupt

    options = ProcessingOptions()

    with (
        patch("photo_tagger.pipeline.process_photo", side_effect=process),
        pytest.raises(BatchError) as batch_error,
    ):
        run_batch(
            [first, second],
            agent=_FAKE_AGENT,
            options=options,
            session_plan=plan,
        )

    totals = batch_error.value.args[0]
    # The interrupted photo had already been analyzed, so the flush wrote it and it counts as a
    # success; only the session that never started is reported as pending.
    assert totals.success == 1
    assert totals.failed_files == [str(second)]


def test_process_photo_writes_the_vocabulary_spelling_verbatim(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A lower-case catalog term reaches the file as the catalog spelled it."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = InferenceResult(
        title="T",
        description="D",
        keywords=["ПТИЦА", "gegenlicht"],
    )
    vocabulary = Vocabulary.from_entries(["птица", "gegenlicht"], fold_plurals=False)

    assert process_photo(image, _ctx(options=ProcessingOptions(vocabulary=vocabulary))) is True

    written = patched_pipeline["write"].call_args.args[1]
    assert written.subject == ["птица", "gegenlicht"]


def test_run_batch_harmonizes_a_session_in_the_configured_language(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Session harmonization uses the run's output language, not English by default."""
    first = tmp_path / "a.cr3"
    second = tmp_path / "b.cr3"
    for path in (first, second):
        path.write_text("x")
    by_file = {
        first: InferenceResult(title="T", description="D", keywords=["Alle"]),
        second: InferenceResult(title="T", description="D", keywords=["Alles"]),
    }
    written: dict[Path, KeywordSet] = {}
    current = [first]

    patched_pipeline["analyze"].side_effect = lambda **_kw: by_file[current[0]]
    patched_pipeline["write"].side_effect = lambda path, keywords, **_kw: bool(
        written.setdefault(path, keywords) or True,
    )

    def process(path: Path, ctx: _BatchContext, **kwargs: Any) -> bool:  # noqa: ANN401
        current[0] = path
        return _real_process_photo(path, ctx, **kwargs)

    with patch("photo_tagger.pipeline.process_photo", side_effect=process):
        run_batch(
            [first, second],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(output_language="German"),
            session_plan=_session_plan([first, second]),
        )

    assert written[first].subject == ["Alle"]
    assert written[second].subject == ["Alles"]


def test_process_photo_records_the_write_in_the_undo_journal(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Each successful write is recorded, with whether the target already existed."""
    fresh = tmp_path / "fresh.cr3"
    existing = tmp_path / "existing.cr3"
    for path in (fresh, existing):
        path.write_text("x")
    existing.with_suffix(".xmp").write_text("older sidecar")
    journal = UndoJournal(tmp_path / "run.jsonl")
    ctx = _ctx()
    ctx.journal = journal

    # write_metadata is stubbed, so create the sidecar the way exiftool would.
    def fake_write(image_path: Path, _keywords: KeywordSet, **_kwargs: Any) -> bool:  # noqa: ANN401
        image_path.with_suffix(".xmp").write_text("written")
        return True

    patched_pipeline["write"].side_effect = fake_write
    assert process_photo(fresh, ctx) is True
    assert process_photo(existing, ctx) is True

    lines = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    assert [entry["created"] for entry in lines] == [True, False]
    assert lines[0]["target"] == str(fresh.with_suffix(".xmp"))


def test_process_photo_journals_both_targets(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Undo has to know about each file a "both" save wrote, the photo included."""
    image = tmp_path / "img.dng"
    image.write_text("x")
    journal = UndoJournal(tmp_path / "run.jsonl")
    ctx = _ctx(options=ProcessingOptions(sidecar_mode="both"))
    ctx.journal = journal

    def fake_write(image_path: Path, _keywords: KeywordSet, **_kwargs: Any) -> bool:  # noqa: ANN401
        image_path.with_suffix(".xmp").write_text("written")
        return True

    patched_pipeline["write"].side_effect = fake_write
    assert process_photo(image, ctx) is True

    lines = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    assert [entry["target"] for entry in lines] == [str(image), str(image.with_suffix(".xmp"))]
    # The photo was there before the run; the sidecar is this run's, so undo deletes it.
    assert [entry["created"] for entry in lines] == [False, True]


def test_process_photo_journals_nothing_when_the_write_fails(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Undo must never be told about a write that did not happen."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["write"].return_value = False
    journal = UndoJournal(tmp_path / "run.jsonl")
    ctx = _ctx()
    ctx.journal = journal

    assert process_photo(image, ctx) is False
    assert journal.entries == 0


_MIXED_SCRIPT_RESULT = InferenceResult(
    title="Decorated陶笛 with Geometric Pattern",
    description="A陶笛 with a geometric pattern rests on a plain white surface.",
    keywords=["陶笛<Wind Instrument", "Geometric Pattern"],
)


def test_process_photo_flags_mixed_script_output(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A model that answered in two writing systems marks the photo for review."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = _MIXED_SCRIPT_RESULT
    ctx = _ctx()
    scratch = _PhotoRecord()

    assert process_photo(image, ctx, record=scratch) is True

    assert scratch.foreign_terms == ["陶笛"]
    assert ctx.usage.flagged_for_review == {image}


def test_process_photo_leaves_clean_output_unflagged(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The usual case costs nothing and says nothing."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    ctx = _ctx()
    scratch = _PhotoRecord()

    assert process_photo(image, ctx, record=scratch) is True

    assert scratch.foreign_terms == []
    assert ctx.usage.flagged_for_review == set()


def test_a_flagged_photo_is_still_written(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Mixed script is sometimes right, so the flag reports rather than withholds the metadata."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = _MIXED_SCRIPT_RESULT

    assert process_photo(image, _ctx()) is True

    assert patched_pipeline["write"].call_args.kwargs["title"] == _MIXED_SCRIPT_RESULT.title


def test_run_batch_lists_the_photos_worth_a_second_look(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The flagged files reach BatchTotals, so the summary file and the console can name them."""
    files = [tmp_path / f"img{i}.cr3" for i in range(2)]
    for path in files:
        path.write_text("x")
    patched_pipeline["analyze"].return_value = _MIXED_SCRIPT_RESULT
    outcomes: list[ImageOutcome] = []

    totals = run_batch(
        files,
        agent=_FAKE_AGENT,
        options=ProcessingOptions(),
        on_image_result=outcomes.append,
    )

    assert totals.flagged_files == sorted(str(path) for path in files)
    assert [outcome.foreign_terms for outcome in outcomes] == [["陶笛"], ["陶笛"]]


def test_a_photo_analyzed_twice_is_one_photo_to_review(tmp_path: Path) -> None:
    """A write failure sends the photo back through the retry pass; the flag must not double."""
    image = tmp_path / "img.cr3"
    usage = _UsageAccumulator()
    usage.add_review(image)
    usage.add_review(image)
    assert usage.flagged_for_review == {image}


# One first answer plus one repair: the whole point of the option is that it is exactly two.
_WITH_REPAIR = 2

_REPAIRED_RESULT = InferenceResult(
    title="Decorated Ocarina with Geometric Pattern",
    description="An ocarina with a geometric pattern rests on a plain white surface.",
    keywords=["Ocarina<Wind Instrument", "Geometric Pattern"],
    input_tokens=7,
    output_tokens=3,
    total_tokens=10,
    seconds=0.2,
)


def test_repair_is_off_by_default(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A mixed-script answer costs one call and is written as it came unless repair is asked for."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].return_value = _MIXED_SCRIPT_RESULT

    assert process_photo(image, _ctx()) is True

    assert patched_pipeline["analyze"].call_count == 1


def test_repair_costs_nothing_on_a_clean_answer(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """Turning the option on must not put a second call on every photo in the batch."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    assert process_photo(image, _ctx(options=ProcessingOptions(repair_language=True))) is True

    assert patched_pipeline["analyze"].call_count == 1


def test_repair_asks_the_model_again_with_the_same_image(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
    stub_image_bytes: BinaryContent,
) -> None:
    """The follow-up quotes the previous answer, names the terms, and carries the photo."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].side_effect = [_MIXED_SCRIPT_RESULT, _REPAIRED_RESULT]
    ctx = _ctx(options=ProcessingOptions(repair_language=True))
    scratch = _PhotoRecord()

    assert process_photo(image, ctx, record=scratch) is True

    repair_call = patched_pipeline["analyze"].call_args_list[1].kwargs
    assert repair_call["image_bytes"] is stub_image_bytes
    assert "陶笛" in repair_call["user_prompt"]
    assert _MIXED_SCRIPT_RESULT.title in repair_call["user_prompt"]
    assert "English" in repair_call["user_prompt"]
    # The repaired text is what lands on the file, and the flag is gone with it.
    assert patched_pipeline["write"].call_args.kwargs["title"] == _REPAIRED_RESULT.title
    assert scratch.foreign_terms == []
    assert ctx.usage.flagged_for_review == set()


def test_repair_rewrites_the_prompt_in_the_chosen_language(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The follow-up asks for --output-language, not for English."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].side_effect = [_MIXED_SCRIPT_RESULT, _REPAIRED_RESULT]
    options = ProcessingOptions(repair_language=True, output_language="Brazilian Portuguese")

    process_photo(image, _ctx(options=options))

    assert "Brazilian Portuguese" in patched_pipeline["analyze"].call_args.kwargs["user_prompt"]


def test_repair_reports_what_both_calls_cost(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A repaired photo's row must not understate its spend by hiding the first call."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    first = InferenceResult(
        title=_MIXED_SCRIPT_RESULT.title,
        description=_MIXED_SCRIPT_RESULT.description,
        keywords=list(_MIXED_SCRIPT_RESULT.keywords),
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        seconds=0.1,
    )
    patched_pipeline["analyze"].side_effect = [first, _REPAIRED_RESULT]
    ctx = _ctx(options=ProcessingOptions(repair_language=True))
    scratch = _PhotoRecord()

    process_photo(image, ctx, record=scratch)

    merged = scratch.inference
    assert merged is not None
    assert merged.total_tokens == first.total_tokens + _REPAIRED_RESULT.total_tokens
    assert merged.seconds == pytest.approx(0.3)
    # Both calls were counted once each on the batch, not folded into one.
    assert ctx.usage.inference_calls == _WITH_REPAIR
    assert ctx.usage.language_repairs == 1


def test_a_repair_that_fails_keeps_the_original_answer(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A photo tagged imperfectly beats a photo not tagged, so the follow-up may fail quietly."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].side_effect = [_MIXED_SCRIPT_RESULT, RuntimeError("boom")]
    ctx = _ctx(options=ProcessingOptions(repair_language=True))
    scratch = _PhotoRecord()

    assert process_photo(image, ctx, record=scratch) is True

    assert patched_pipeline["write"].call_args.kwargs["title"] == _MIXED_SCRIPT_RESULT.title
    # Still flagged: nothing was fixed, so the photo is still one to go back to.
    assert scratch.foreign_terms == ["陶笛"]
    assert ctx.usage.language_repairs == 0


def test_a_repair_that_does_not_help_leaves_the_photo_flagged(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A model that answers in two scripts twice gets no third try; the flag does the rest."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].side_effect = [_MIXED_SCRIPT_RESULT, _MIXED_SCRIPT_RESULT]
    ctx = _ctx(options=ProcessingOptions(repair_language=True))
    scratch = _PhotoRecord()

    process_photo(image, ctx, record=scratch)

    assert patched_pipeline["analyze"].call_count == _WITH_REPAIR
    assert scratch.foreign_terms == ["陶笛"]


def test_a_repair_that_mixes_scripts_again_keeps_the_original_text(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """
    The photo keeps the answer it was flagged on, and its row counts both calls.

    A second answer that mixes scripts too is no better, and replacing the flagged text with a
    different, also flagged one would leave the person reviewing something the check never saw.
    """
    image = tmp_path / "img.cr3"
    image.write_text("x")
    mixed_again = replace(
        _REPAIRED_RESULT,
        title="Painted陶器 Ocarina",
        description="A painted ocarina.",
    )
    patched_pipeline["analyze"].side_effect = [_MIXED_SCRIPT_RESULT, mixed_again]
    ctx = _ctx(options=ProcessingOptions(repair_language=True))
    scratch = _PhotoRecord()

    process_photo(image, ctx, record=scratch)

    inference = scratch.inference
    assert inference is not None
    assert inference.title == _MIXED_SCRIPT_RESULT.title
    assert inference.total_tokens == _MIXED_SCRIPT_RESULT.total_tokens + mixed_again.total_tokens


def _mixed_cache() -> tuple[Any, dict[str, InferenceResult]]:
    """Return a cache already holding a mixed-script answer under "key", and its store."""
    stored: dict[str, InferenceResult] = {"key": _MIXED_SCRIPT_RESULT}
    cache = cast("Any", SimpleNamespace(get=stored.get, put=stored.__setitem__))
    return cache, stored


def test_a_cached_mixed_answer_is_repaired_when_the_setting_is_on(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """An answer cached before --repair-language was turned on is fixed, and the fix is stored."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].side_effect = [_REPAIRED_RESULT]
    cache, stored = _mixed_cache()
    ctx = _ctx(options=ProcessingOptions(repair_language=True), cache=cache)
    scratch = _PhotoRecord()

    with patch("photo_tagger.pipeline.content_cache_key", return_value="key"):
        process_photo(image, ctx, record=scratch)

    assert patched_pipeline["analyze"].call_count == 1
    assert stored["key"].title == _REPAIRED_RESULT.title
    assert scratch.inference is not None
    assert scratch.inference.total_tokens == _REPAIRED_RESULT.total_tokens
    assert scratch.from_cache is False


def test_a_cached_mixed_answer_is_replayed_as_is_without_the_setting(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """With repair off, a cache hit costs nothing, mixed or not."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    cache, stored = _mixed_cache()
    ctx = _ctx(options=ProcessingOptions(), cache=cache)

    with patch("photo_tagger.pipeline.content_cache_key", return_value="key"):
        process_photo(image, ctx)

    patched_pipeline["analyze"].assert_not_called()
    assert stored["key"] is _MIXED_SCRIPT_RESULT


def test_the_cache_stores_the_repaired_answer(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A rerun replays the fixed text and never pays for the second call again."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    patched_pipeline["analyze"].side_effect = [_MIXED_SCRIPT_RESULT, _REPAIRED_RESULT]
    stored: dict[str, InferenceResult] = {}
    cache = cast(
        "Any",
        SimpleNamespace(
            get=stored.get,
            put=stored.__setitem__,
        ),
    )
    ctx = _ctx(options=ProcessingOptions(repair_language=True), cache=cache)

    with patch("photo_tagger.pipeline.content_cache_key", return_value="key"):
        process_photo(image, ctx)

    assert stored["key"].title == _REPAIRED_RESULT.title


def test_run_batch_counts_the_repair_calls(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The extra calls show in the totals, so the cost of the option is visible in the summary."""
    files = [tmp_path / f"img{i}.cr3" for i in range(2)]
    for path in files:
        path.write_text("x")
    patched_pipeline["analyze"].side_effect = [
        _MIXED_SCRIPT_RESULT,
        _REPAIRED_RESULT,
        _MIXED_SCRIPT_RESULT,
        _REPAIRED_RESULT,
    ]

    totals = run_batch(
        files,
        agent=_FAKE_AGENT,
        options=ProcessingOptions(repair_language=True),
    )

    assert totals.language_repairs == _WITH_REPAIR
    assert totals.flagged_files == []


def test_process_photo_writes_the_copyright_fields_with_the_capture_year(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """The notice is filled from the photo's own tags, read with the rest of its context."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(
        extra_fields=ExtraFields(
            artist="Ada Lovelace",
            notice="Copyright (c) {year} {artist}",
            marked=True,
        ),
    )

    context = ImageContext(
        field_sources=FieldSources(image={"EXIF:DateTimeOriginal": "2019:07:14 10:32:11"}),
    )
    with patch("photo_tagger.pipeline.read_image_context", return_value=context) as read:
        assert process_photo(image, _ctx(options=options)) is True

    assert "EXIF:DateTimeOriginal" in read.call_args.kwargs["field_tags"]
    extra = patched_pipeline["write"].call_args.kwargs["extra"]
    assert extra["EXIF:Copyright"] == "Copyright (c) 2019 Ada Lovelace"
    assert extra["XMP-dc:Creator"] == "Ada Lovelace"
    assert extra["XMP-xmpRights:Marked"] == "True"


def test_process_photo_writes_no_extra_tags_by_default(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A run that asked for no copyright touches no copyright tag."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    assert process_photo(image, _ctx(options=ProcessingOptions())) is True

    assert patched_pipeline["write"].call_args.kwargs["extra"] == {}


def test_process_photo_keeps_a_notice_the_photo_already_has(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """With preserve on (the default), a tagged photo keeps its own notice and credit."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    options = ProcessingOptions(extra_fields=ExtraFields(artist="Ada", notice="(c) {year}"))
    kept = {"XMP:Rights": "(c) 2011", "XMP:Creator": "Grace"}
    context = ImageContext(field_sources=FieldSources(sidecar=kept))

    with patch("photo_tagger.pipeline.read_image_context", return_value=context):
        assert process_photo(image, _ctx(options=options)) is True

    assert patched_pipeline["write"].call_args.kwargs["extra"] == {}


def test_process_photo_ignores_a_notice_in_a_file_it_does_not_write(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """An embedded save still stamps a photo whose only notice sits in a sidecar."""
    image = tmp_path / "img.jpg"
    image.write_text("x")
    options = ProcessingOptions(
        extra_fields=ExtraFields(notice="(c) {year}"),
        sidecar_mode="none",
    )
    sources = FieldSources(
        image={"EXIF:DateTimeOriginal": "2019:01:02 03:04:05"},
        sidecar={"XMP:Rights": "(c) 2011"},
    )

    with patch(
        "photo_tagger.pipeline.read_image_context",
        return_value=ImageContext(field_sources=sources),
    ):
        assert process_photo(image, _ctx(options=options)) is True

    assert patched_pipeline["write"].call_args.kwargs["extra"]["EXIF:Copyright"] == "(c) 2019"


def test_session_mode_keeps_each_photo_its_own_copyright_year(
    tmp_path: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    """A deferred write still carries the year of the photo it belongs to, not of its session."""
    old = tmp_path / "old.cr3"
    new = tmp_path / "new.cr3"
    for path in (old, new):
        path.write_text("x")
    years = {
        old: ImageContext(
            field_sources=FieldSources(image={"EXIF:DateTimeOriginal": "2011:01:02 03:04:05"}),
        ),
        new: ImageContext(
            field_sources=FieldSources(image={"EXIF:DateTimeOriginal": "2025:01:02 03:04:05"}),
        ),
    }

    with patch(
        "photo_tagger.pipeline.read_image_context",
        side_effect=lambda path, **_kwargs: years[path],
    ):
        run_batch(
            [old, new],
            agent=_FAKE_AGENT,
            options=ProcessingOptions(
                extra_fields=ExtraFields(notice="(c) {year}"),
                # One session for the batch, so both writes are deferred to the same flush.
                output_language="English",
            ),
            session_plan=SessionPlan(sessions=[[old, new]], index_of={old: 0, new: 0}),
        )

    written = {
        call.args[0]: call.kwargs["extra"]["EXIF:Copyright"]
        for call in patched_pipeline["write"].call_args_list
    }
    assert written == {old: "(c) 2011", new: "(c) 2025"}


class _TickingClock:
    """A clock the test moves by hand, so a paced run takes no real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        """Return the current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake time instead of waiting."""
        self.now += seconds


_PHOTO_SECONDS = 60.0


def test_run_batch_rests_between_work_windows(tmp_path: Path) -> None:
    """A paced run rests at photo boundaries, and the totals say what that cost."""
    files = [tmp_path / f"img{i}.cr3" for i in range(_BATCH_SIZE)]
    for f in files:
        f.write_text("x")
    clock = _TickingClock()

    def work(*_args: Any, **_kwargs: Any) -> bool:  # noqa: ANN401 - stands in for process_photo.
        clock.sleep(_PHOTO_SECONDS)
        return True

    pacer = Pacer(
        PacingPlan(duty_percent=50, work_seconds=_PHOTO_SECONDS),
        battery_reader=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with patch("photo_tagger.pipeline.process_photo", side_effect=work):
        totals = run_batch(files, agent=_FAKE_AGENT, options=ProcessingOptions(), pacer=pacer)

    assert totals.success == _BATCH_SIZE
    # The first photo only starts the clock; the two after it each earn a full window of rest.
    assert totals.rests_taken == _BATCH_SIZE - 1
    assert totals.rested_seconds == pytest.approx(2 * _PHOTO_SECONDS)


def test_run_batch_without_a_pacer_reports_no_rest(tmp_path: Path) -> None:
    """A run with no pacer works flat out and says so in its totals."""
    image = tmp_path / "img.cr3"
    image.write_text("x")

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        totals = run_batch([image], agent=_FAKE_AGENT, options=ProcessingOptions())

    assert totals.rested_seconds == 0
    assert totals.rests_taken == 0


def test_run_batch_interrupted_after_every_photo_still_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted batch is an error even with every photo done, so a watch stops on it."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    monkeypatch.setattr("photo_tagger.pipeline._run_pass", lambda *_a, **_k: (1, [], True))

    options = ProcessingOptions()

    with pytest.raises(BatchError) as exc_info:
        run_batch([image], agent=_FAKE_AGENT, options=options)

    assert exc_info.value.args[0].interrupted


def test_execute_process_skips_a_photo_when_the_rest_is_cancelled(
    tmp_path: Path,
) -> None:
    """A cancelled rest means the run is ending, so the photo is never touched."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    pacer = Pacer(PacingPlan(duty_percent=50))
    pacer.cancel()
    ctx = _ctx()
    ctx.pacer = pacer

    with patch("photo_tagger.pipeline.process_photo", return_value=True) as proc:
        ok = execute_process(image, ctx, index="1/1")

    assert ok is False
    assert proc.call_count == 0


def test_run_batch_concurrent_interrupt_cancels_the_pacer(tmp_path: Path) -> None:
    """
    Ctrl-C wakes every worker asleep in a rest.

    Without this the drain that follows the interrupt would wait out the whole rest, which can be
    minutes, before the run could report and exit.
    """
    files = [tmp_path / f"img{i}.cr3" for i in range(2)]
    for f in files:
        f.write_text("x")
    pacer = Pacer(PacingPlan(duty_percent=50))
    options = ProcessingOptions()

    class InterruptingPool(ThreadPoolExecutor):
        def submit(self, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401 - pool signature.
            raise KeyboardInterrupt

    with (
        patch("photo_tagger.pipeline.ThreadPoolExecutor", InterruptingPool),
        patch("photo_tagger.pipeline.process_photo", return_value=True),
        pytest.raises(BatchError),
    ):
        run_batch(
            files,
            agent=_FAKE_AGENT,
            options=options,
            workers=2,
            pacer=pacer,
        )

    assert pacer.wait_before_next() is False


def test_run_batch_reports_only_its_own_rests(tmp_path: Path) -> None:
    """
    A watch shares one pacer across batches, so each batch reports the rests it took itself.

    Reading the pacer's running totals would count every earlier rest again in each summary.
    """
    image = tmp_path / "img.cr3"
    image.write_text("x")
    pacer = Pacer(PacingPlan(duty_percent=50))
    pacer._rested_seconds = 120.0  # noqa: SLF001 - an earlier batch of the same watch.
    pacer._rests_taken = 2  # noqa: SLF001

    with patch("photo_tagger.pipeline.process_photo", return_value=True):
        totals = run_batch([image], agent=_FAKE_AGENT, options=ProcessingOptions(), pacer=pacer)

    assert totals.rested_seconds == 0
    assert totals.rests_taken == 0


def test_a_ctrl_c_in_the_retry_pass_marks_the_batch_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watch stops on an interrupted batch, so the retry pass must report its Ctrl-C too."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    passes = iter([(0, [image], False), (0, [image], True)])
    monkeypatch.setattr("photo_tagger.pipeline._run_pass", lambda *_a, **_k: next(passes))
    monkeypatch.setattr("photo_tagger.pipeline._RETRY_PASS_DELAY_SECONDS", 0)

    options = ProcessingOptions()

    with pytest.raises(BatchError) as exc_info:
        run_batch([image], agent=_FAKE_AGENT, options=options)

    assert exc_info.value.args[0].interrupted
