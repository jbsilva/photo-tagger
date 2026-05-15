"""High-level photo processing pipeline shared by the CLI and the retry loop."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    from pathlib import Path

    from pydantic_ai import Agent


@dataclass(slots=True)
class ProcessingOptions:
    """Bundle of per-photo settings that the CLI hands to the pipeline."""

    preserve_existing_kw: bool = True
    write_description: bool = True
    write_title: bool = True
    backup_xmp: bool = True
    use_sidecar: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    jpeg_dimensions: int = DEFAULT_DIMENSIONS
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


_EMPTY_KEYWORDS: dict[str, list[str]] = {
    "subject": [],
    "hierarchical": [],
    "weighted": [],
}


def process_photo(image_path: Path, agent: Agent, options: ProcessingOptions) -> bool:
    """
    Convert an image to JPEG bytes in memory, query the model, and persist metadata.

    Returns True if every step succeeded, False if metadata writing failed.
    """
    logger.info("processing_photo")

    jpeg_bytes = prepare_image_for_agent(
        image_path,
        jpg_quality=options.jpeg_quality,
        max_size=options.jpeg_dimensions,
    )

    existing_keywords_full = read_existing_keywords(image_path)
    if any(existing_keywords_full.values()):
        logger.info(
            "existing_keywords_found",
            count=len(existing_keywords_full["subject"]),
        )

    contextual_prompt = build_contextual_prompt(
        DEFAULT_USER_PROMPT,
        list(dict.fromkeys(existing_keywords_full.get("subject", []))),
        read_location_tags(image_path),
        read_gps_coordinates(image_path),
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

    return write_metadata(
        image_path,
        merged_keywords,
        description=description if options.write_description else None,
        title=title if options.write_title else None,
        backup=options.backup_xmp,
        use_sidecar=options.use_sidecar,
    )


def execute_process(
    image_file: Path,
    agent: Agent,
    options: ProcessingOptions,
    *,
    index: str,
    retry: bool = False,
) -> bool:
    """Run process_photo once with consistent logging and error handling."""
    context_kwargs: dict[str, Any] = {"file": image_file.name}
    if retry:
        context_kwargs["retry"] = True

    with logger.contextualize(**context_kwargs):
        try:
            ok = process_photo(image_file, agent, options)
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


def _run_initial_pass(
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
) -> tuple[int, list[Path]]:
    """First pass over the batch; returns (success_count, files_to_retry)."""
    total = len(image_files)
    successes = 0
    pending: list[Path] = []
    for idx, image_file in enumerate(image_files, start=1):
        if execute_process(image_file, agent, options, index=f"{idx}/{total}"):
            successes += 1
        else:
            logger.warning("file_queued_for_retry", file=image_file.name)
            pending.append(image_file)
    return successes, pending


def _run_retry_pass(
    pending: list[Path],
    agent: Agent,
    options: ProcessingOptions,
) -> tuple[int, list[Path]]:
    """Retry every previously failed file once; returns (retry_successes, still_failing)."""
    if not pending:
        return 0, []

    logger.info("retrying_failed_files", count=len(pending))
    successes = 0
    still_failing: list[Path] = []
    for idx, image_file in enumerate(pending, start=1):
        if execute_process(
            image_file,
            agent,
            options,
            index=f"{idx}/{len(pending)}",
            retry=True,
        ):
            successes += 1
        else:
            logger.error("file_failed_after_retry", file=image_file.name)
            still_failing.append(image_file)
    return successes, still_failing


def run_batch(
    image_files: list[Path],
    agent: Agent,
    options: ProcessingOptions,
) -> _BatchTotals:
    """Run the initial pass plus a single retry pass and return summary totals."""
    success, pending = _run_initial_pass(image_files, agent, options)
    retry_successes, still_failing = _run_retry_pass(pending, agent, options)

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
    )
    if still_failing:
        logger.error(
            "files_failed_after_retry",
            files=[str(path) for path in still_failing],
        )

    if totals.success < len(image_files):
        raise SystemExit(1)
    return totals
