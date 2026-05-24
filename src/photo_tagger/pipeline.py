"""High-level photo processing pipeline shared by the CLI and the retry loop."""

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from photo_tagger.ai import InferenceResult, analyze_image_with_ai
from photo_tagger.cache import InferenceCache, hash_image_file
from photo_tagger.config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_USER_PROMPT,
)
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.keywords import merge_keywords
from photo_tagger.metadata import (
    _helper,
    build_contextual_prompt,
    read_image_context,
    write_metadata,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]
    from pydantic_ai import Agent

    from photo_tagger.models import GeneratedMetadata

    OnSuccess = Callable[[Path], None]
    ProgressCallback = Callable[[Path, bool], None]
    OnComplete = Callable[["BatchTotals"], None]
    OnImageResult = Callable[["ImageOutcome"], None]


@dataclass(slots=True, frozen=True)
class ImageOutcome:
    """
    Per-image result the pipeline streams to consumers via ``on_image_result``.

    Carries the AI fields (or what the cache replayed) plus the success bit and
    a ``from_cache`` flag. The CLI uses this to emit one NDJSON line per photo
    when ``--json`` is set, so downstream tools can act on each result as soon
    as it lands instead of waiting for the BatchTotals summary at the end.
    """

    file: Path
    success: bool
    from_cache: bool
    retry: bool
    title: str | None
    description: str | None
    keywords: list[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    seconds: float


@dataclass(slots=True)
class ProcessingOptions:
    """Bundle of per-photo settings that the CLI hands to the pipeline."""

    preserve_existing_kw: bool = True
    write_description: bool = True
    write_title: bool = True
    backup_xmp: bool = True
    use_sidecar: bool = True
    dry_run: bool = False
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    jpeg_dimensions: int = DEFAULT_DIMENSIONS
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    max_new_keywords: int | None = None


_EMPTY_KEYWORDS: dict[str, list[str]] = {
    "subject": [],
    "hierarchical": [],
    "weighted": [],
}


@contextlib.contextmanager
def _no_helper() -> Iterator[None]:
    """Yield None as the shared ExifToolHelper for the concurrent path."""
    yield None


@dataclass(slots=True)
class _UsageAccumulator:
    """Thread-safe running totals for token usage across a batch."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_seconds: float = 0.0
    inference_calls: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, result: InferenceResult) -> None:
        """Fold *result*'s usage into the running totals under the lock."""
        with self._lock:
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.total_tokens += result.total_tokens
            self.inference_seconds += result.seconds
            self.inference_calls += 1

    def add_cache_hit(self) -> None:
        """Count one photo that skipped the model call thanks to a cache hit."""
        with self._lock:
            self.cache_hits += 1


def _cache_lookup(
    cache: InferenceCache,
    image_path: Path,
) -> tuple[str | None, InferenceResult | None]:
    """
    Look up *image_path* in *cache*; return ``(cache_key, hit_or_none)``.

    A failure to hash the source file or read from the cache is logged and
    treated as a miss, never raised. ``cache_key`` is ``None`` when hashing
    failed, signaling the caller to skip ``put`` as well.
    """
    try:
        cache_key = hash_image_file(image_path)
    except OSError as exc:
        logger.warning("inference_cache_hash_failed", file=image_path.name, error=str(exc))
        return None, None
    try:
        cached = cache.get(cache_key)
    except Exception as exc:  # noqa: BLE001 - sqlite errors must not abort the photo.
        logger.warning("inference_cache_get_failed", file=image_path.name, error=str(exc))
        return cache_key, None
    return cache_key, cached


def _cache_store(
    cache: InferenceCache,
    cache_key: str,
    inference: InferenceResult,
    *,
    file_name: str,
) -> None:
    """Store *inference* under *cache_key*, logging and swallowing any storage error."""
    try:
        cache.put(cache_key, inference)
    except Exception as exc:  # noqa: BLE001 - sqlite errors must not abort the photo.
        logger.warning("inference_cache_put_failed", file=file_name, error=str(exc))


def _resolve_inference(  # noqa: PLR0913 - cache lookup needs every model knob to call.
    image_path: Path,
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    contextual_prompt: str,
    cache: InferenceCache | None,
    usage: _UsageAccumulator | None,
) -> tuple[InferenceResult, bool]:
    """
    Return ``(inference, from_cache)`` for *image_path*.

    Hits the on-disk cache when one is provided and the photo's content hash
    matches a prior entry recorded under the same namespace. On miss, prepares
    the JPEG bytes, calls the model, and writes the result back to the cache.

    Cache I/O failures are logged at warning level but never raised: a broken
    SQLite file or full disk degrades the run to "no cache" without aborting
    photos that the model would otherwise process successfully.
    """
    cache_key: str | None = None
    if cache is not None:
        cache_key, cached = _cache_lookup(cache, image_path)
        if cached is not None:
            logger.info("cache_hit", file=image_path.name)
            if usage is not None:
                usage.add_cache_hit()
            return cached, True

    jpeg_bytes = prepare_image_for_agent(
        image_path,
        jpg_quality=options.jpeg_quality,
        max_size=options.jpeg_dimensions,
    )
    inference = analyze_image_with_ai(
        image_bytes=jpeg_bytes,
        agent=agent,
        user_prompt=contextual_prompt,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )
    if usage is not None:
        usage.add(inference)
    if cache is not None and cache_key is not None:
        _cache_store(cache, cache_key, inference, file_name=image_path.name)
    return inference, False


def process_photo(  # noqa: PLR0913 - knobs collapse to one ProcessingOptions plus context.
    image_path: Path,
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    et: ExifToolHelper | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
    usage: _UsageAccumulator | None = None,
    cache: InferenceCache | None = None,
    outcome_sink: dict[str, Any] | None = None,
) -> bool:
    """
    Convert an image to JPEG bytes in memory, query the model, and persist metadata.

    Args:
        image_path: Image to process.
        agent: Configured pydantic-ai agent that returns the metadata schema.
        options: Per-photo settings (sampling, sidecar mode, dry-run, etc.).
        et: Optional pre-opened ExifToolHelper. Reused for every read/write in this call
            when supplied, otherwise one helper is opened for the duration of this photo.
            Concurrent callers must NOT share a helper across threads (the underlying
            -stay_open subprocess uses a single stdin/stdout pipe); pass et=None instead
            and let each task get its own.
        user_prompt: Base prompt; existing photo metadata is appended automatically.
        usage: Optional thread-safe accumulator. When given, the AI call's token counts
            and latency are folded into it. ``run_batch`` passes one shared accumulator
            so the BatchTotals at the end of the run reflect every photo.
        cache: Optional InferenceCache. When supplied, the photo's content hash is
            looked up first; a hit reuses the cached title/description/keywords and
            skips both image preparation and the model call.
        outcome_sink: Optional per-call scratch dict. When provided, this function
            populates ``inference`` (InferenceResult) and ``from_cache`` (bool) so the
            caller can wrap them into an ImageOutcome without retracing the work.

    Returns:
        True if every step succeeded, False if metadata writing failed.

    """
    logger.info("processing_photo")

    with _helper(et) as helper:
        context = read_image_context(image_path, et=helper)
        existing_keywords_full = context.existing_keywords
        if any(existing_keywords_full.values()):
            logger.info(
                "existing_keywords_found",
                count=len(existing_keywords_full["subject"]),
            )

        gps_info = {"position": context.gps_position} if context.gps_position else {}
        contextual_prompt = build_contextual_prompt(
            user_prompt,
            list(dict.fromkeys(existing_keywords_full.get("subject", []))),
            context.location_tags,
            gps_info,
            camera_info=context.camera_info,
        )

        inference, from_cache = _resolve_inference(
            image_path,
            agent,
            options,
            contextual_prompt=contextual_prompt,
            cache=cache,
            usage=usage,
        )
        if outcome_sink is not None:
            outcome_sink["inference"] = inference
            outcome_sink["from_cache"] = from_cache

        title = inference.title
        description = inference.description
        keywords = inference.keywords

        if options.max_new_keywords is not None and len(keywords) > options.max_new_keywords:
            logger.info(
                "trimming_ai_keywords",
                returned=len(keywords),
                cap=options.max_new_keywords,
            )
            keywords = keywords[: options.max_new_keywords]

        base = existing_keywords_full if options.preserve_existing_kw else dict(_EMPTY_KEYWORDS)
        merged_keywords = merge_keywords(base, keywords)

        if options.dry_run:
            logger.info(
                "dry_run_preview",
                file=image_path.name,
                title=title if options.write_title else None,
                description=description if options.write_description else None,
                subject_keywords=merged_keywords.get("subject", []),
                hierarchical_keywords=merged_keywords.get("hierarchical", []),
            )
            return True

        return write_metadata(
            image_path,
            merged_keywords,
            description=description if options.write_description else None,
            title=title if options.write_title else None,
            backup=options.backup_xmp,
            use_sidecar=options.use_sidecar,
            et=helper,
        )


def _emit_outcome(
    on_image_result: OnImageResult | None,
    image_file: Path,
    scratch: dict[str, Any],
    *,
    success: bool,
    retry: bool,
) -> None:
    """Build an ImageOutcome from *scratch* and *success* and call *on_image_result*."""
    if on_image_result is None:
        return
    inference: InferenceResult | None = scratch.get("inference")
    outcome = ImageOutcome(
        file=image_file,
        success=success,
        from_cache=bool(scratch.get("from_cache", False)),
        retry=retry,
        title=inference.title if inference is not None else None,
        description=inference.description if inference is not None else None,
        keywords=list(inference.keywords) if inference is not None else [],
        input_tokens=inference.input_tokens if inference is not None else 0,
        output_tokens=inference.output_tokens if inference is not None else 0,
        total_tokens=inference.total_tokens if inference is not None else 0,
        seconds=inference.seconds if inference is not None else 0.0,
    )
    try:
        on_image_result(outcome)
    except Exception as exc:  # noqa: BLE001 - callback errors must not break the batch.
        logger.exception("on_image_result_callback_failed", file=image_file.name, error=str(exc))


def execute_process(  # noqa: PLR0913 - extra kwargs are passthrough to process_photo.
    image_file: Path,
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    index: str,
    retry: bool = False,
    et: ExifToolHelper | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
    usage: _UsageAccumulator | None = None,
    cache: InferenceCache | None = None,
    on_image_result: OnImageResult | None = None,
) -> bool:
    """Run process_photo once with consistent logging and error handling."""
    context_kwargs: dict[str, Any] = {"file": image_file.name}
    if retry:
        context_kwargs["retry"] = True

    scratch: dict[str, Any] = {}
    with logger.contextualize(**context_kwargs):
        try:
            ok = process_photo(
                image_file,
                agent,
                options,
                et=et,
                user_prompt=user_prompt,
                usage=usage,
                cache=cache,
                outcome_sink=scratch,
            )
        except Exception as exc:  # noqa: BLE001 - process_photo wraps several SDKs
            event = "processing_retry_exception" if retry else "processing_exception"
            logger.exception(event, error=str(exc))
            _emit_outcome(on_image_result, image_file, scratch, success=False, retry=retry)
            return False

        if ok:
            event = "retry_success" if retry else "processing_success"
            logger.info(event, index=index)
            _emit_outcome(on_image_result, image_file, scratch, success=True, retry=retry)
            return True

        if retry:
            logger.error("retry_failed", index=index)
        else:
            logger.error("processing_failed", index=index, queued_for_retry=True)
        _emit_outcome(on_image_result, image_file, scratch, success=False, retry=retry)
        return False


@dataclass(slots=True)
class BatchTotals:
    """
    Public summary of one ``run_batch`` invocation.

    The CLI surfaces this in logs and uses it to write the optional JSON summary file.
    """

    total_files: int = 0
    success: int = 0
    initial_failures: int = 0
    retry_successes: int = 0
    failed_files: list[str] = field(default_factory=list)
    successful_files: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_seconds: float = 0.0
    inference_calls: int = 0
    cache_hits: int = 0
    workers: int = 1
    dry_run: bool = False


# Backwards-compatible alias for tests that still import the private name.
_BatchTotals = BatchTotals


def _notify_success(on_success: OnSuccess | None, image_file: Path) -> None:
    """Invoke *on_success* defensively; a callback failure must not abort the batch."""
    if on_success is None:
        return
    try:
        on_success(image_file)
    except Exception as exc:  # noqa: BLE001 - any callback error must not break the batch.
        logger.exception("on_success_callback_failed", file=image_file.name, error=str(exc))


def _run_pass_serial(  # noqa: PLR0913 - pass-through arguments to execute_process.
    image_files: list[Path],
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    retry: bool,
    et: ExifToolHelper | None,
    user_prompt: str,
    on_success: OnSuccess | None,
    progress: ProgressCallback | None,
    usage: _UsageAccumulator | None,
    cache: InferenceCache | None,
    on_image_result: OnImageResult | None,
) -> tuple[int, list[Path], bool]:
    """
    Run *image_files* one after another in the calling thread.

    A ``KeyboardInterrupt`` raised between photos stops the pass and returns
    ``(successes, still-pending, interrupted=True)`` so the caller can still
    emit a BatchTotals + summary file. Photos that were not yet attempted land
    in the failed list so they show up in the summary too.
    """
    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    for idx, image_file in enumerate(image_files, start=1):
        try:
            ok = execute_process(
                image_file,
                agent,
                options,
                index=f"{idx}/{total}",
                retry=retry,
                et=et,
                user_prompt=user_prompt,
                usage=usage,
                cache=cache,
                on_image_result=on_image_result,
            )
        except KeyboardInterrupt:
            logger.warning("batch_interrupted_by_user", remaining=total - idx + 1)
            failed.extend(image_files[idx - 1 :])
            return successes, failed, True
        if ok:
            successes += 1
            _notify_success(on_success, image_file)
        elif retry:
            logger.error("file_failed_after_retry", file=image_file.name)
            failed.append(image_file)
        else:
            logger.warning("file_queued_for_retry", file=image_file.name)
            failed.append(image_file)
        if progress is not None:
            progress(image_file, ok)
    return successes, failed, False


def _run_pass_concurrent(  # noqa: PLR0913 - pass-through arguments to execute_process.
    image_files: list[Path],
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    retry: bool,
    user_prompt: str,
    on_success: OnSuccess | None,
    workers: int,
    progress: ProgressCallback | None,
    usage: _UsageAccumulator | None,
    cache: InferenceCache | None,
    on_image_result: OnImageResult | None,
) -> tuple[int, list[Path], bool]:
    """
    Run *image_files* across a thread pool. Each task gets its own ExifToolHelper.

    pyexiftool's -stay_open subprocess uses one stdin/stdout pipe per helper, so a
    helper cannot be shared across threads safely. Each task receives et=None and
    process_photo opens a short-lived helper for the duration of one photo.

    A ``KeyboardInterrupt`` during the as_completed loop cancels pending futures
    and returns ``(successes, still-pending, interrupted=True)``. Already in-flight
    workers cannot be interrupted from the outside, so this is a best-effort
    quick stop rather than an instant abort.
    """
    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    interrupted = False
    indexed = list(enumerate(image_files, start=1))

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        future_to_image = {
            pool.submit(
                execute_process,
                image_file,
                agent,
                options,
                index=f"{idx}/{total}",
                retry=retry,
                et=None,
                user_prompt=user_prompt,
                usage=usage,
                cache=cache,
                on_image_result=on_image_result,
            ): image_file
            for idx, image_file in indexed
        }
        try:
            for future in as_completed(future_to_image):
                image_file = future_to_image[future]
                try:
                    ok = future.result()
                except Exception as exc:  # noqa: BLE001 - per-file failure stays per-file.
                    logger.exception(
                        "concurrent_worker_exception",
                        file=image_file.name,
                        error=str(exc),
                    )
                    ok = False

                if ok:
                    successes += 1
                    _notify_success(on_success, image_file)
                elif retry:
                    logger.error("file_failed_after_retry", file=image_file.name)
                    failed.append(image_file)
                else:
                    logger.warning("file_queued_for_retry", file=image_file.name)
                    failed.append(image_file)
                if progress is not None:
                    progress(image_file, ok)
        except KeyboardInterrupt:
            interrupted = True
            pending = [img for fut, img in future_to_image.items() if not fut.done()]
            logger.warning("batch_interrupted_by_user", pending=len(pending))
            failed.extend(pending)
            pool.shutdown(wait=False, cancel_futures=True)
    finally:
        pool.shutdown(wait=True)
    return successes, failed, interrupted


def _run_pass(  # noqa: PLR0913 - dispatch to serial or concurrent worker.
    image_files: list[Path],
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    retry: bool,
    et: ExifToolHelper | None,
    user_prompt: str,
    on_success: OnSuccess | None,
    workers: int,
    progress: ProgressCallback | None,
    usage: _UsageAccumulator | None,
    cache: InferenceCache | None,
    on_image_result: OnImageResult | None,
) -> tuple[int, list[Path], bool]:
    """
    Run a single pass (initial or retry) over the batch.

    Returns ``(success_count, still_failing, interrupted)``. The first pass logs
    failures with ``file_queued_for_retry``; the retry pass logs them with
    ``file_failed_after_retry``. The interrupted flag is True if a Ctrl-C caused
    the pass to abort early; the caller uses it to skip the retry pass and to
    mark the run as a partial completion.
    """
    if not image_files:
        return 0, [], False

    if retry:
        logger.info("retrying_failed_files", count=len(image_files))

    if workers <= 1:
        return _run_pass_serial(
            image_files,
            agent,
            options,
            retry=retry,
            et=et,
            user_prompt=user_prompt,
            on_success=on_success,
            progress=progress,
            usage=usage,
            cache=cache,
            on_image_result=on_image_result,
        )
    return _run_pass_concurrent(
        image_files,
        agent,
        options,
        retry=retry,
        user_prompt=user_prompt,
        on_success=on_success,
        workers=workers,
        progress=progress,
        usage=usage,
        cache=cache,
        on_image_result=on_image_result,
    )


def run_batch(  # noqa: PLR0913 - distinct optional knobs are clearer as kwargs.
    image_files: list[Path],
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    on_success: OnSuccess | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
    workers: int = 1,
    progress: ProgressCallback | None = None,
    on_complete: OnComplete | None = None,
    cache: InferenceCache | None = None,
    on_image_result: OnImageResult | None = None,
) -> BatchTotals:
    """
    Run the initial pass plus a single retry pass and return summary totals.

    When workers == 1 a single ExifToolHelper is opened for the whole batch so every
    metadata read and write reuses one long-running exiftool subprocess. When workers
    > 1 the tasks run on a ThreadPoolExecutor and each task opens its own short-lived
    helper (pyexiftool's -stay_open pipe is not safe to share across threads).

    The optional *on_success* callback fires once per image that completes successfully
    (whether on the first pass or after a retry). It receives the image path. The CLI
    uses this to append filenames to a skip list as work progresses, so a killed run can
    be resumed without redoing finished photos.

    The optional *progress* callback fires once per image (success or failure). It is
    used by the CLI to drive a rich progress bar.
    """
    usage = _UsageAccumulator()
    successful_files: list[Path] = []

    def _record_success(path: Path) -> None:
        successful_files.append(path)
        if on_success is not None:
            on_success(path)

    with _helper(None) if workers <= 1 else _no_helper() as et:
        success, pending, interrupted = _run_pass(
            image_files,
            agent,
            options,
            retry=False,
            et=et,
            user_prompt=user_prompt,
            on_success=_record_success,
            workers=workers,
            progress=progress,
            usage=usage,
            cache=cache,
            on_image_result=on_image_result,
        )
        if interrupted:
            # Don't retry after the user asked us to stop. Mark everything that
            # was still pending as failed so the summary file reflects reality.
            retry_successes = 0
            still_failing = pending
        else:
            retry_successes, still_failing, _ = _run_pass(
                pending,
                agent,
                options,
                retry=True,
                et=et,
                user_prompt=user_prompt,
                on_success=_record_success,
                workers=workers,
                progress=progress,
                usage=usage,
                cache=cache,
                on_image_result=on_image_result,
            )

    totals = BatchTotals(
        total_files=len(image_files),
        success=success + retry_successes,
        initial_failures=len(pending),
        retry_successes=retry_successes,
        failed_files=[str(path) for path in still_failing],
        successful_files=[str(path) for path in successful_files],
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        inference_seconds=round(usage.inference_seconds, 3),
        inference_calls=usage.inference_calls,
        cache_hits=usage.cache_hits,
        workers=workers,
        dry_run=options.dry_run,
    )

    logger.info(
        "processing_summary",
        total_files=totals.total_files,
        successful=totals.success,
        failed=len(still_failing),
        initial_failures=totals.initial_failures,
        retry_successes=totals.retry_successes,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        inference_calls=totals.inference_calls,
        inference_seconds=totals.inference_seconds,
        cache_hits=totals.cache_hits,
        dry_run=options.dry_run,
        workers=workers,
    )
    if still_failing:
        logger.error("files_failed_after_retry", files=totals.failed_files)

    if on_complete is not None:
        try:
            on_complete(totals)
        except Exception as exc:  # noqa: BLE001 - summary writers must not abort the run.
            logger.exception("on_complete_callback_failed", error=str(exc))

    if totals.success < len(image_files):
        raise SystemExit(1)
    return totals
