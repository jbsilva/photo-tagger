"""The default ``tag`` command: tag a batch of photos once and exit."""

from dataclasses import replace
from pathlib import Path  # noqa: TC003 - cyclopts resolves the flag annotations at runtime.
from typing import Annotated

from cyclopts import Parameter, validators
from loguru import logger

from photo_tagger import cli_run, telemetry
from photo_tagger.cli_options import (
    CONFIG_SOURCE,
    DEFAULT_ARTIFACTS_OPTIONS,
    DEFAULT_DISPLAY_OPTIONS,
    DEFAULT_EXTENSIONS,
    DEFAULT_FILTER_OPTIONS,
    DEFAULT_INFERENCE_OPTIONS,
    DEFAULT_LOG_OPTIONS,
    DEFAULT_OUTPUT_OPTIONS,
    DEFAULT_PACING_OPTIONS,
    DEFAULT_PROVIDER_OPTIONS,
    DEFAULT_RECURSIVE,
    DEFAULT_TELEMETRY_OPTIONS,
    DEFAULT_WORKERS,
    ArtifactConfig,
    DisplayConfig,
    FilterConfig,
    InferenceConfig,
    LogConfig,
    OutputConfig,
    PacingConfig,
    ProviderConfig,
    RunOptions,
    TelemetryConfig,
    validate_pacing,
)
from photo_tagger.cli_telemetry import (
    Beacons,
    show_notice_if_due,
)
from photo_tagger.discovery import resolve_image_batch
from photo_tagger.errors import PhotoTaggerError


def tag(  # noqa: PLR0913 - cyclopts entry point; each arg is a CLI flag group.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            validator=validators.Path(exists=True),
            help="One or more paths: files and/or directories (repeat this option)",
        ),
    ] = None,
    *,
    image_extensions: Annotated[
        str,
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(
            name=("--recursive", "-r"),
            help="Process files in subdirectories recursively",
        ),
    ] = DEFAULT_RECURSIVE,
    workers: Annotated[
        int,
        Parameter(
            name=("--workers", "-w"),
            help=(
                "Number of photos to process concurrently. Defaults to 1 (serial). The model "
                "server is the bottleneck; raising this past what your provider can serve in "
                "parallel will not help"
            ),
        ),
    ] = DEFAULT_WORKERS,
    filter_: Annotated[FilterConfig, Parameter(name="*")] = DEFAULT_FILTER_OPTIONS,
    display: Annotated[DisplayConfig, Parameter(name="*")] = DEFAULT_DISPLAY_OPTIONS,
    artifacts: Annotated[ArtifactConfig, Parameter(name="*")] = DEFAULT_ARTIFACTS_OPTIONS,
    provider: Annotated[ProviderConfig, Parameter(name="*")] = DEFAULT_PROVIDER_OPTIONS,
    output: Annotated[OutputConfig, Parameter(name="*")] = DEFAULT_OUTPUT_OPTIONS,
    inference: Annotated[InferenceConfig, Parameter(name="*")] = DEFAULT_INFERENCE_OPTIONS,
    log: Annotated[LogConfig, Parameter(name="*")] = DEFAULT_LOG_OPTIONS,
    telemetry_config: Annotated[TelemetryConfig, Parameter(name="*")] = DEFAULT_TELEMETRY_OPTIONS,
    pacing: Annotated[
        PacingConfig,
        Parameter(name="*", validator=validate_pacing),
    ] = DEFAULT_PACING_OPTIONS,
) -> None:
    """
    Tag images with AI and write Lightroom-compatible metadata (sidecar or embedded).

    Requirements:
    - ExifTool installed and on PATH.
    - Vision-language model API reachable (e.g., Ollama server).

    Inputs:
    - One or more --input/-i paths (files and/or directories; repeatable).
    - Files are processed as is. Directories use --ext (add --recursive for subfolders).
    - You can mix files and directories; order is preserved, duplicates skipped.

    Behavior:
    - Loads image (RAW supported), converts to in-memory JPEG, queries the model.
    - Generates title, description, and keywords; merges with existing XMP by default
        (use --overwrite-keywords to replace).
    - Writes metadata to an XMP sidecar (default), embeds it directly with --embed-in-photo,
        splits the two per file type with --sidecar-mode raw (sidecars for RAW, embedded for JPEG
        and friends), or does both with --sidecar-mode both. --sidecar-override dng=photo sends
        one file type elsewhere (into each DNG here, as Lightroom Classic does). Use
        --no-write-title/--no-write-description to skip fields; --no-backup-xmp to avoid backups.

    Skipping:
    - --skip-from FILE: skip files listed in FILE (one name or path per line).
    - --append-to-skip-file FILE: append each successfully processed file's path to FILE so a
        later run with --skip-from FILE resumes where this one stopped.
    - --skip-tagged: skip files that already have keywords, a description, or a title in
        the image or its XMP sidecar.

    Dry runs:
    - --dry-run: query the model and log the generated title, description, and keywords
        for each image but do not write any metadata. Useful for prompt iteration.

    Performance:
    - --workers N: process N photos concurrently using a thread pool. Each worker opens
        its own ExifToolHelper. The model server is the dominant bottleneck.
    - --max-keywords N: cap the number of AI-generated keywords kept before merging.

    Pacing (for long runs on a laptop):
    - --duty N: work only N percent of the time, resting in between. Keeps the fans down,
        leaves the machine usable, and holds the average power draw below what the charger
        supplies. A run rests at a photo boundary, never mid-photo.
    - --battery-floor N: rest until the battery charges back up whenever it falls to N
        percent, so a long run on a laptop does not flatten it. Only while plugged in.
    - --work-minutes / --max-pause-minutes: how long a work window runs, and the longest a
        single rest may last.

    Customization:
    - --prompt-file PATH: replace the default user prompt with the file's contents.
        Existing photo metadata (keywords, GPS, location) is still appended automatically.

    Exit status: returns 1 if no inputs are given, discovery finds no matching images, or any
        file fails. Returns 0 if discovery finds images but every one of them is subsequently
        excluded by --skip-from, --skip-tagged, or the date filters: that is "nothing left to
        do", not an error, and scripted incremental reruns rely on it not failing the batch.

    Examples:
        photo-tagger -i ./photos/IMG_0001.CR3

        photo-tagger \
            --extensions cr3,jpg \
            --provider lmstudio \
            --url http://localhost:1234/v1 \
            --recursive \
            --skip-from processed.txt \
            --append-to-skip-file processed.txt \
            Pictures/Camera

        photo-tagger -i Pictures/Mixed --skip-tagged
    """
    run = RunOptions(
        provider=provider,
        inference=inference,
        output=output,
        artifacts=artifacts,
        display=display,
        filter_=filter_,
        log=log,
        pacing=pacing,
        telemetry_enabled=telemetry_config.enabled,
        workers=workers,
        extensions=image_extensions,
        recursive=recursive,
        inputs=inputs,
    )
    beacons = Beacons.for_invocation("tag", run, CONFIG_SOURCE.invocation)
    try:
        cli_run.start_run(log)
        with cli_run.optional_lock(artifacts.lock_file):
            try:
                _tag_inside_lock(run, beacons)
            except PhotoTaggerError as exc:
                raise SystemExit(1) from exc
    except (SystemExit, KeyboardInterrupt) as exc:
        # A run that stopped before its first photo (no model, no photos, a held lock) sends no
        # batch beacon of its own, so it is reported here.
        beacons.stopped(exc)
        raise


def _tag_inside_lock(run: RunOptions, beacons: Beacons) -> None:
    """Body of ``tag`` that runs once the optional file lock has been acquired."""
    show_notice_if_due(enabled=beacons.enabled)
    newer_than = cli_run.parse_filter_date(run.filter_.newer_than, flag="--newer-than")
    older_than = cli_run.parse_filter_date(run.filter_.older_than, flag="--older-than")
    # Discovery first: building the run truncates the CSV report, opens the cache and validates the
    # model against the server. A batch that turns out to be empty should cost none of that, and
    # least of all last run's report.
    image_files = cli_run.filter_batch(
        resolve_image_batch(run.inputs, run.extensions, recursive=run.recursive),
        run,
        newer_than=newer_than,
        older_than=older_than,
    )
    if not image_files:
        logger.info("no_files_to_process_after_skipping")
        beacons.emit(replace(beacons.base_run(), outcome=telemetry.OUTCOME_EMPTY))
        return
    setup = cli_run.build_run_setup(run, beacons)
    cli_run.log_startup(setup)

    try:
        cli_run.process_batch(image_files, setup)
    finally:
        setup.close()
