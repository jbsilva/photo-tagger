"""High-level photo processing pipeline shared by the CLI and the retry loop."""

import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from loguru import logger

from photo_tagger.ai import analyze_image_with_ai
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
    read_existing_keywords,
    read_gps_coordinates,
    read_location_tags,
    write_metadata,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pydantic_ai import Agent

    OnSuccess = Callable[[Path], None]
    ProgressCallback = Callable[[Path, bool], None]


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
def _no_helper() -> "Iterator[None]":
    """Yield None as the shared ExifToolHelper for the concurrent path."""
    yield None


def process_photo(
    image_path: Path,
    agent: Agent,
    options: ProcessingOptions,
    *,
    et: ExifToolHelper | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
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

    Returns:
        True if every step succeeded, False if metadata writing failed.

    """
    logger.info("processing_photo")

    jpeg_bytes = prepare_image_for_agent(
        image_path,
        jpg_quality=options.jpeg_quality,
        max_size=options.jpeg_dimensions,
    )

    with _helper(et) as helper:
        existing_keywords_full = read_existing_keywords(image_path, et=helper)
        if any(existing_keywords_full.values()):
            logger.info(
                "existing_keywords_found",
                count=len(existing_keywords_full["subject"]),
            )

        contextual_prompt = build_contextual_prompt(
            user_prompt,
            list(dict.fromkeys(existing_keywords_full.get("subject", []))),
            read_location_tags(image_path, et=helper),
            read_gps_coordinates(image_path, et=helper),
        )

        title, description, keywords = analyze_image_with_ai(
            image_bytes=jpeg_bytes,
            agent=agent,
            user_prompt=contextual_prompt,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
        )

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


def execute_process(  # noqa: PLR0913 - extra kwargs are passthrough to process_photo.
    image_file: Path,
    agent: Agent,
    options: ProcessingOptions,
    *,
    index: str,
    retry: bool = False,
    et: ExifToolHelper | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> bool:
    """Run process_photo once with consistent logging and error handling."""
    context_kwargs: dict[str, Any] = {"file": image_file.name}
    if retry:
        context_kwargs["retry"] = True

    with logger.contextualize(**context_kwargs):
        try:
            ok = process_photo(image_file, agent, options, et=et, user_prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001 - process_photo wraps several SDKs
            event = "processing_retry_exception" if retry else "processing_exception"
            logger.exception(event, error=str(exc))
            return False

        if ok:
            event = "retry_success" if retry else "processing_success"
            logger.info(event, index=index)
            return True

        if retry:
            logger.error("retry_failed", index=index)
        else:
            logger.error("processing_failed", index=index, queued_for_retry=True)
        return False


@dataclass(slots=True)
class _BatchTotals:
    success: int = 0
    initial_failures: int = 0
    retry_successes: int = 0


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
    agent: Agent,
    options: ProcessingOptions,
    *,
    retry: bool,
    et: ExifToolHelper | None,
    user_prompt: str,
    on_success: OnSuccess | None,
    progress: ProgressCallback | None,
) -> tuple[int, list[Path]]:
    """Run *image_files* one after another in the calling thread."""
    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    for idx, image_file in enumerate(image_files, start=1):
        ok = execute_process(
            image_file,
            agent,
            options,
            index=f"{idx}/{total}",
            retry=retry,
            et=et,
            user_prompt=user_prompt,
        )
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
    return successes, failed


def _run_pass_concurrent(  # noqa: PLR0913 - pass-through arguments to execute_process.
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
    *,
    retry: bool,
    user_prompt: str,
    on_success: OnSuccess | None,
    workers: int,
    progress: ProgressCallback | None,
) -> tuple[int, list[Path]]:
    """
    Run *image_files* across a thread pool. Each task gets its own ExifToolHelper.

    pyexiftool's -stay_open subprocess uses one stdin/stdout pipe per helper, so a
    helper cannot be shared across threads safely. Each task receives et=None and
    process_photo opens a short-lived helper for the duration of one photo.
    """
    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    completed = 0
    indexed = list(enumerate(image_files, start=1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
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
            ): image_file
            for idx, image_file in indexed
        }
        for future in as_completed(future_to_image):
            image_file = future_to_image[future]
            completed += 1
            try:
                ok = future.result()
            except Exception as exc:  # noqa: BLE001 - any worker error is a per-file failure.
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
    return successes, failed


def _run_pass(  # noqa: PLR0913 - dispatch to serial or concurrent worker.
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
    *,
    retry: bool,
    et: ExifToolHelper | None,
    user_prompt: str,
    on_success: OnSuccess | None,
    workers: int,
    progress: ProgressCallback | None,
) -> tuple[int, list[Path]]:
    """
    Run a single pass (initial or retry) over the batch.

    Returns (success_count, still_failing). The first pass logs failures with
    ``file_queued_for_retry``; the retry pass logs them with ``file_failed_after_retry``.
    Dispatches to the serial or concurrent worker depending on *workers*.
    """
    if not image_files:
        return 0, []

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
    )


def run_batch(  # noqa: PLR0913 - distinct optional knobs are clearer as kwargs.
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
    *,
    on_success: OnSuccess | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> _BatchTotals:
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
    with _helper(None) if workers <= 1 else _no_helper() as et:
        success, pending = _run_pass(
            image_files,
            agent,
            options,
            retry=False,
            et=et,
            user_prompt=user_prompt,
            on_success=on_success,
            workers=workers,
            progress=progress,
        )
        retry_successes, still_failing = _run_pass(
            pending,
            agent,
            options,
            retry=True,
            et=et,
            user_prompt=user_prompt,
            on_success=on_success,
            workers=workers,
            progress=progress,
        )

    totals = _BatchTotals(
        success=success + retry_successes,
        initial_failures=len(pending),
        retry_successes=retry_successes,
    )

    logger.info(
        "processing_summary",
        total_files=len(image_files),
        successful=totals.success,
        failed=len(still_failing),
        initial_failures=totals.initial_failures,
        retry_successes=totals.retry_successes,
        dry_run=options.dry_run,
        workers=workers,
    )
    if still_failing:
        logger.error(
            "files_failed_after_retry",
            files=[str(path) for path in still_failing],
        )

    if totals.success < len(image_files):
        raise SystemExit(1)
    return totals
