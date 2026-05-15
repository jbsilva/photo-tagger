"""High-level photo processing pipeline shared by the CLI and the retry loop."""

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
    build_contextual_prompt,
    read_existing_keywords,
    read_gps_coordinates,
    read_location_tags,
    write_metadata,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic_ai import Agent

    OnSuccess = Callable[[Path], None]


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


_EMPTY_KEYWORDS: dict[str, list[str]] = {
    "subject": [],
    "hierarchical": [],
    "weighted": [],
}


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
        et: Optional pre-opened ExifToolHelper to share across the batch and avoid one
            subprocess startup per call.
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

    existing_keywords_full = read_existing_keywords(image_path, et=et)
    if any(existing_keywords_full.values()):
        logger.info(
            "existing_keywords_found",
            count=len(existing_keywords_full["subject"]),
        )

    contextual_prompt = build_contextual_prompt(
        user_prompt,
        list(dict.fromkeys(existing_keywords_full.get("subject", []))),
        read_location_tags(image_path, et=et),
        read_gps_coordinates(image_path, et=et),
    )

    title, description, keywords = analyze_image_with_ai(
        image_bytes=jpeg_bytes,
        agent=agent,
        user_prompt=contextual_prompt,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
    )

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
        et=et,
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


def _run_pass(  # noqa: PLR0913 - pass-through arguments to execute_process.
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
    *,
    retry: bool,
    et: ExifToolHelper | None,
    user_prompt: str,
    on_success: OnSuccess | None,
) -> tuple[int, list[Path]]:
    """
    Run a single pass (initial or retry) over the batch.

    Returns (success_count, still_failing). The first pass logs failures with
    ``file_queued_for_retry``; the retry pass logs them with ``file_failed_after_retry``.
    """
    if not image_files:
        return 0, []

    if retry:
        logger.info("retrying_failed_files", count=len(image_files))

    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    for idx, image_file in enumerate(image_files, start=1):
        if execute_process(
            image_file,
            agent,
            options,
            index=f"{idx}/{total}",
            retry=retry,
            et=et,
            user_prompt=user_prompt,
        ):
            successes += 1
            _notify_success(on_success, image_file)
        elif retry:
            logger.error("file_failed_after_retry", file=image_file.name)
            failed.append(image_file)
        else:
            logger.warning("file_queued_for_retry", file=image_file.name)
            failed.append(image_file)
    return successes, failed


def run_batch(
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
    *,
    on_success: OnSuccess | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> _BatchTotals:
    """
    Run the initial pass plus a single retry pass and return summary totals.

    A single ExifToolHelper is opened for the whole batch so every metadata read and
    write reuses one long-running exiftool subprocess. This is the dominant non-AI cost
    on large batches.

    The optional *on_success* callback fires once per image that completes successfully
    (whether on the first pass or after a retry). It receives the image path. The CLI
    uses this to append filenames to a skip list as work progresses, so a killed run can
    be resumed without redoing finished photos.
    """
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        success, pending = _run_pass(
            image_files,
            agent,
            options,
            retry=False,
            et=et,
            user_prompt=user_prompt,
            on_success=on_success,
        )
        retry_successes, still_failing = _run_pass(
            pending,
            agent,
            options,
            retry=True,
            et=et,
            user_prompt=user_prompt,
            on_success=on_success,
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
    )
    if still_failing:
        logger.error(
            "files_failed_after_retry",
            files=[str(path) for path in still_failing],
        )

    if totals.success < len(image_files):
        raise SystemExit(1)
    return totals
