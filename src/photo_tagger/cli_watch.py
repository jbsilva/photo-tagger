"""The ``watch`` command: tag photos as they land in a folder, until Ctrl-C."""

from dataclasses import replace
from pathlib import Path  # noqa: TC003 - cyclopts resolves the flag annotations at runtime.
from typing import Annotated

from cyclopts import Parameter, validators
from loguru import logger

from photo_tagger import cli_run
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
    DEFAULT_WATCH_OPTIONS,
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
    WatchConfig,
    validate_pacing,
)
from photo_tagger.cli_telemetry import (
    Beacons,
    show_notice_if_due,
)
from photo_tagger.errors import (
    BatchError,
    PhotoTaggerError,
)
from photo_tagger.watch import watch_batches


def watch(  # noqa: PLR0913 - cyclopts entry point; each arg is a CLI flag group.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            validator=validators.Path(exists=True),
            help="One or more folders (or files) to watch (repeat this option)",
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
        Parameter(name=("--recursive", "-r"), help="Watch subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    watch_config: Annotated[WatchConfig, Parameter(name="*")] = DEFAULT_WATCH_OPTIONS,
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
    Watch folders and tag photos as they arrive, until you stop it with Ctrl-C.

    This is the import-time workflow: point it at the folder your card reader, tethered capture, or
    sync client fills, and leave it running. Photos already in the folder are tagged first, then
    each new one as it lands.

    A file is only tagged once it has stopped changing (see ``--settle``), so a photo still being
    copied is left alone until the copy finishes. Scanning is a plain directory listing every
    ``--interval`` seconds, which behaves the same on every platform and over network shares.

    Every flag ``photo-tagger`` itself takes works here too and applies to each batch, with one
    agent, cache, and report file shared by the whole session. Each batch records its own undo
    journal, so ``photo-tagger undo`` puts back the last import rather than every import since the
    watch started. A batch where some photo fails is logged and the watch continues; the failure
    does not stop the loop.

    Examples:
        photo-tagger watch -i ~/Pictures/Inbox

        photo-tagger watch -i ~/Pictures/Inbox -r --interval 10 --skip-tagged
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
        workers=watch_config.workers,
        extensions=image_extensions,
        recursive=recursive,
        inputs=inputs,
        watch=watch_config,
    )
    beacons = Beacons.for_invocation("watch", run, CONFIG_SOURCE.invocation)
    try:
        cli_run.start_run(log)
        folders = cli_run.require_inputs(inputs)
        with cli_run.optional_lock(artifacts.lock_file):
            try:
                _watch_forever(replace(run, inputs=folders), watch_config, beacons)
            except PhotoTaggerError as exc:
                raise SystemExit(1) from exc
    except (SystemExit, KeyboardInterrupt) as exc:
        beacons.stopped(exc)
        raise


def _watch_forever(run: RunOptions, watch_config: WatchConfig, beacons: Beacons) -> None:
    """Poll *run*'s inputs for new photos and tag each settled batch until interrupted."""
    show_notice_if_due(enabled=beacons.enabled)
    newer_than = cli_run.parse_filter_date(run.filter_.newer_than, flag="--newer-than")
    older_than = cli_run.parse_filter_date(run.filter_.older_than, flag="--older-than")
    setup = cli_run.build_run_setup(run, beacons)
    inputs = run.inputs or []
    logger.info(
        "watching_for_new_photos",
        inputs=[str(path) for path in inputs],
        extensions=run.extensions,
        recursive=run.recursive,
        interval=watch_config.interval,
        settle=watch_config.settle,
    )
    batches = 0
    tagged = 0
    try:
        for batch in watch_batches(
            inputs,
            run.extensions,
            recursive=run.recursive,
            interval_seconds=watch_config.interval,
            settle_seconds=watch_config.settle,
        ):
            image_files = cli_run.filter_batch(
                batch,
                run,
                newer_than=newer_than,
                older_than=older_than,
            )
            if not image_files:
                continue
            batches += 1
            tagged += len(image_files)
            # Each batch is its own run to undo: a watch left running all week must not fold a
            # month of imports into one journal.
            setup.start_journal()
            try:
                cli_run.process_batch(image_files, setup)
            except BatchError as exc:
                # run_batch turns a Ctrl-C into an interrupted batch, and that must end the watch.
                if cli_run.was_interrupted(exc):
                    raise KeyboardInterrupt from exc
                # One bad batch must not end the watch: the next photo to land deserves its turn.
                logger.error("watch_batch_had_failures", error=str(exc))
    except KeyboardInterrupt:
        logger.info("watch_stopped_by_user", batches=batches, photos=tagged)
    finally:
        setup.close()
        beacons.emit(beacons.session())
