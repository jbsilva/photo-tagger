"""
The run behind ``tag`` and ``watch``: the setup a run builds once, and one batch.

``tag`` processes one batch and ``watch`` a new one whenever photos land, so both build
a :class:`RunSetup` here and hand each batch to :func:`process_batch`.
"""

import contextlib
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.ai import create_agent
from photo_tagger.cache import build_cache_namespace, open_cache
from photo_tagger.cli_options import (
    LogConfig,
    RunOptions,
    resolve_notify,
    to_pacing_plan,
    to_processing_options,
)
from photo_tagger.cli_reports import (
    CsvImageResultSink,
    NDJSONEmitter,
    alert_message,
    combine_image_result_callbacks,
    open_csv_report,
    write_summary_file,
)
from photo_tagger.cli_telemetry import (
    Beacons,
    PhotoStats,
    run_from_totals,
)
from photo_tagger.config import DEFAULT_USER_PROMPT
from photo_tagger.config_file import export_exiftool_path
from photo_tagger.discovery import (
    apply_date_filter,
    apply_skip_file,
    apply_skip_tagged,
    make_skip_list_appender,
)
from photo_tagger.errors import (
    BatchError,
    ObservabilityError,
)
from photo_tagger.locking import FileLock, LockHeldError
from photo_tagger.logging_setup import setup_logging
from photo_tagger.notify import APP_NAME, notify
from photo_tagger.observability import configure_tracing
from photo_tagger.pacing import Pacer, build_pacer
from photo_tagger.pipeline import BatchTotals, ImageOutcome, ProcessingOptions, run_batch
from photo_tagger.progress import batch_progress
from photo_tagger.prompt_context import prompt_with_hint
from photo_tagger.sessions import plan_sessions
from photo_tagger.undo import open_journal
from photo_tagger.vocabulary import load_vocabulary, prompt_with_vocabulary


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from pydantic_ai import Agent

    from photo_tagger.cache import InferenceCache
    from photo_tagger.csv_report import CsvReportWriter
    from photo_tagger.models import GeneratedMetadata
    from photo_tagger.undo import UndoJournal


def enable_tracing(log: LogConfig) -> None:
    """
    Switch on OpenTelemetry tracing when ``--otel`` asked for it, or stop with a clear reason.

    Called before any agent exists, because instrumentation is applied to the agent class rather
    than to instances. A failure here ends the run: somebody who passed ``--otel`` wants the traces,
    and silently tagging photos without them wastes the batch they were trying to observe.
    """
    try:
        configure_tracing(
            enabled=log.otel,
            endpoint=log.otel_endpoint,
            include_content=log.otel_content,
        )
    except ObservabilityError as exc:
        logger.error("otel_setup_failed", error=str(exc))
        raise SystemExit(1) from exc


def read_prompt_file(prompt_file: Path | None) -> str:
    """Return the contents of *prompt_file* (stripped) or DEFAULT_USER_PROMPT when None."""
    if prompt_file is None:
        return DEFAULT_USER_PROMPT
    try:
        text = prompt_file.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        logger.error("prompt_file_read_failed", file=str(prompt_file), error=str(exc))
        raise SystemExit(1) from exc
    if not text:
        logger.error("prompt_file_empty", file=str(prompt_file))
        raise SystemExit(1)
    logger.info("prompt_file_loaded", file=str(prompt_file), chars=len(text))
    return text


def parse_filter_date(value: str | None, *, flag: str) -> datetime | None:
    """
    Parse an ISO 8601 string from *flag* into a timezone-aware datetime, or None.

    A naive timestamp like ``2024-01-01`` is read as **local** time (matching ``git log --since``
    and ``find -newer`` conventions). The system local zone is attached, and the value is returned
    aware so downstream comparisons with UTC mtimes do the right thing.
    """
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        logger.error("date_filter_parse_failed", flag=flag, value=value, error=str(exc))
        raise SystemExit(1) from exc
    if parsed.tzinfo is not None:
        return parsed
    # datetime.astimezone() on a naive value attaches the system local zone
    # (per the stdlib docs), turning it into a tz-aware value without shifting
    # the wall-clock fields. That is exactly the "user meant their local time"
    # semantics we want here.
    return parsed.astimezone()


def _maybe_str(value: Path | None) -> str | None:
    """Return ``str(value)`` or ``None`` so loguru extras keep a clean type."""
    return str(value) if value is not None else None


def start_run(log: LogConfig) -> None:
    """Set up the log sinks, tracing, and the ExifTool path a tag or watch run needs."""
    try:
        setup_logging(
            file_log_level=log.file_log_level,
            console_log_level=log.console_log_level,
            log_folder=log.log_folder,
        )
    except OSError as exc:
        # No sink is guaranteed to be active yet (setup_logging removes the default one before it
        # can fail), so this cannot rely on the logger; write directly to stderr like the gui
        # command's own pre-logging error path does.
        sys.stderr.write(f"Could not set up logging at {log.log_folder}: {exc}\n")
        raise SystemExit(1) from exc
    enable_tracing(log)
    export_exiftool_path()


def require_inputs(inputs: list[Path] | None) -> list[Path]:
    """Return *inputs*, or stop with the hint to pass some when there are none."""
    if not inputs:
        logger.error("no_inputs_provided", hint="Pass one or more --input/-i paths to watch")
        raise SystemExit(1)
    return inputs


@contextlib.contextmanager
def optional_lock(lock_file: Path | None) -> Generator[None]:
    """
    Hold *lock_file* for the run, when one was given, or stop with exit code 1.

    A lock another run holds, or one that cannot be opened (a folder that is not writable, a full
    disk), ends the run with a log line instead of a traceback in the user's terminal.
    """
    with contextlib.ExitStack() as stack:
        if lock_file is not None:
            try:
                stack.enter_context(FileLock(lock_file))
            except LockHeldError as exc:
                logger.error("lock_held_by_other_process", file=str(lock_file), error=str(exc))
                raise SystemExit(1) from exc
            except OSError as exc:
                logger.error("lock_file_open_failed", file=str(lock_file), error=str(exc))
                raise SystemExit(1) from exc
        yield


@dataclass(slots=True)
class RunSetup:
    """
    The collaborators a tagging run builds once and reuses for every batch it processes.

    ``tag`` processes exactly one batch, ``watch`` processes a new one whenever photos land, and
    both want the same agent, cache, and report writers for the whole invocation: reopening them per
    batch would repeat the CSV header and re-prune the cache.

    The undo journal is the exception. A watch can run for days, and one journal for all of it would
    make "undo the import" mean every photo since Tuesday; each batch is its own run, so
    :meth:`start_journal` opens one per batch. ``tag`` has a single batch and never calls it.
    """

    run: RunOptions
    options: ProcessingOptions
    agent: Agent[None, GeneratedMetadata]
    user_prompt: str
    workers: int
    beacons: Beacons
    # Resolved from the two --notify flags, not the group itself: --notify-sound implies --notify,
    # and the batch runner should not have to know that.
    notify_enabled: bool
    notify_sound: bool
    cache: InferenceCache | None = None
    csv_writer: CsvReportWriter | None = None
    # None unless the run was asked to slow down. Shared by every batch a watch processes, so the
    # work window carries across batches instead of restarting with each one.
    pacer: Pacer | None = None
    on_image_result: Callable[[ImageOutcome], None] | None = None
    on_success: Callable[[Path], None] | None = None
    journal: UndoJournal | None = None
    # Whether writes are recorded at all, so a rotated journal knows to stay switched off.
    journal_enabled: bool = True

    def start_journal(self) -> None:
        """Open a journal for the batch about to run, reporting what the previous one recorded."""
        self._report_journal()
        self.journal = open_journal(datetime.now(tz=UTC), enabled=self.journal_enabled)

    def _report_journal(self) -> None:
        """Log the journal's path once it holds something, so the user can name it to undo."""
        if self.journal is not None and self.journal.entries:
            logger.info(
                "undo_journal_written",
                file=str(self.journal.path),
                entries=self.journal.entries,
            )

    def close(self) -> None:
        """Release everything the run opened, in the order it was opened."""
        if self.cache is not None:
            self.cache.close()
        if self.csv_writer is not None:
            self.csv_writer.close()
        self._report_journal()


def build_run_setup(run: RunOptions, beacons: Beacons) -> RunSetup:
    """Build the agent, cache, report sinks, pacer, and undo journal shared by every batch."""
    provider, inference, output, artifacts = run.provider, run.inference, run.output, run.artifacts
    # Raises VocabularyError (a PhotoTaggerError) on an unusable file, which the commands turn into
    # a clean exit 1. Loading it up front means a typo in the path fails before any model call.
    vocabulary = (
        load_vocabulary(output.vocabulary, output_language=inference.output_language)
        if output.vocabulary is not None
        else None
    )
    if vocabulary is not None:
        beacons.settings = replace(beacons.settings, vocabulary_terms=len(vocabulary.terms))
    # A --hint and the --vocabulary listing ride inside the user prompt, so the cache namespace
    # below picks them up too: a hinted run never replays results generated without the hint (and
    # vice versa), and the same holds for a vocabulary.
    user_prompt = prompt_with_vocabulary(
        prompt_with_hint(read_prompt_file(artifacts.prompt_file), inference.hint),
        vocabulary,
    )
    # Fold the prompt + language + sampling/JPEG settings into the cache namespace so a
    # different configuration writes to a fresh slice instead of replaying
    # stale entries generated under earlier settings.
    cache_namespace = build_cache_namespace(
        provider.model_name,
        user_prompt=user_prompt,
        temperature=inference.temperature,
        frequency_penalty=inference.frequency_penalty,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
        output_language=inference.output_language,
    )
    csv_writer = open_csv_report(artifacts.csv_file)
    ndjson_emitter = NDJSONEmitter(sys.stdout) if run.display.json_output else None
    csv_sink = CsvImageResultSink(csv_writer) if csv_writer is not None else None
    # A dry run writes nothing, so there is nothing for undo to put back.
    journal_enabled = artifacts.undo_log and not output.dry_run
    # Anything below can raise before a RunSetup exists, and then the caller has nothing to
    # close: the report file and the cache's SQLite handle would stay open until the interpreter
    # got round to them. create_agent validates the model over HTTP, which is the failure that
    # actually happens here; it runs first, so a wrong model name never opens the cache at all.
    cache: InferenceCache | None = None
    notify_enabled, notify_sound = resolve_notify(run.display)
    try:
        agent = create_agent(
            provider.provider_name,
            provider.model_name,
            api_base_url=provider.api_base_url,
            api_key=provider.api_key,
            retries=provider.retries,
            output_language=inference.output_language,
        )
        cache = open_cache(artifacts.cache_file, namespace=cache_namespace)
        return RunSetup(
            run=run,
            options=to_processing_options(
                output,
                inference,
                vocabulary=vocabulary,
                retries=provider.retries,
            ),
            agent=agent,
            user_prompt=user_prompt,
            workers=max(1, run.workers),
            beacons=beacons,
            notify_enabled=notify_enabled,
            notify_sound=notify_sound,
            cache=cache,
            csv_writer=csv_writer,
            pacer=build_pacer(to_pacing_plan(run.pacing)),
            on_image_result=combine_image_result_callbacks(ndjson_emitter, csv_sink),
            on_success=make_skip_list_appender(artifacts.append_to_skip_file),
            journal=open_journal(datetime.now(tz=UTC), enabled=journal_enabled),
            journal_enabled=journal_enabled,
        )
    except Exception:
        if cache is not None:
            cache.close()
        if csv_writer is not None:
            csv_writer.close()
        raise


def log_startup(setup: RunSetup) -> None:
    """Single-shot info log so the run's full configuration is captured up-front."""
    run = setup.run
    provider, inference, artifacts = run.provider, run.inference, run.artifacts
    options = setup.options
    logger.info(
        "starting_photo_tagger",
        inputs=[str(p) for p in (run.inputs or [])],
        extensions=run.extensions,
        model=provider.model_name,
        provider=provider.provider_name,
        api_base_url=provider.api_base_url,
        api_key_present=bool(provider.api_key),
        recursive=run.recursive,
        workers=setup.workers,
        prompt_file=_maybe_str(artifacts.prompt_file),
        summary_file=_maybe_str(artifacts.summary_file),
        csv_file=_maybe_str(artifacts.csv_file),
        cache_file=_maybe_str(artifacts.cache_file),
        lock_file=_maybe_str(artifacts.lock_file),
        json_output=run.display.json_output,
        progress_bar=run.display.progress_bar,
        notify=setup.notify_enabled,
        notify_sound=setup.notify_sound,
        telemetry=setup.beacons.enabled,
        skip_from=_maybe_str(artifacts.skip_from),
        append_to_skip_file=_maybe_str(artifacts.append_to_skip_file),
        skip_tagged=run.filter_.skip_tagged,
        newer_than=run.filter_.newer_than,
        older_than=run.filter_.older_than,
        preserve_keywords=options.preserve_existing_kw,
        preserve_title=options.preserve_existing_title,
        preserve_description=options.preserve_existing_description,
        write_description=options.write_description,
        write_title=options.write_title,
        write_keywords=options.write_keywords,
        backup_xmp=options.backup_xmp,
        sidecar_mode=options.sidecar_mode,
        dry_run=options.dry_run,
        max_keywords=options.max_new_keywords,
        vocabulary_terms=len(options.vocabulary.terms) if options.vocabulary else 0,
        vocabulary_strict=options.vocabulary_strict,
        extra_fields=not (options.extra_fields is None or options.extra_fields.is_empty),
        session_gap_minutes=run.output.session_gap_minutes,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        timeout_seconds=options.timeout_seconds,
        frequency_penalty=options.frequency_penalty,
        jpeg_dimensions=options.jpeg_dimensions,
        jpeg_quality=options.jpeg_quality,
        output_language=inference.output_language,
        hint=inference.hint,
        retries=provider.retries,
        log_folder=str(run.log.log_folder),
    )


def process_batch(image_files: list[Path], setup: RunSetup) -> None:
    """
    Tag one resolved batch, writing the summary file and firing the telemetry beacon.

    Raises :class:`~photo_tagger.errors.BatchError` when any photo fails, which ``tag`` turns into
    exit 1 and ``watch`` logs before waiting for the next batch.
    """
    started_at = datetime.now(tz=UTC)
    stats = PhotoStats()
    plan = plan_sessions(image_files, gap_minutes=setup.run.output.session_gap_minutes)

    def _on_complete(totals: BatchTotals) -> None:
        # Runs once before run_batch raises, so the summary file is written, the desktop alert
        # raised, and the telemetry beacon fired whether the batch succeeded fully or only
        # partially. A run that half failed is exactly the one worth being called back for.
        notify(
            APP_NAME,
            alert_message(totals),
            enabled=setup.notify_enabled,
            sound=setup.notify_sound,
            # `tag` exits right after this, and `watch` goes back to sleep; either way nothing
            # else is waiting on the few seconds the notifier takes.
            block=True,
        )
        write_summary_file(
            setup.run.artifacts.summary_file,
            totals,
            started_at=started_at,
            model_name=setup.run.provider.model_name,
            provider_name=setup.run.provider.provider_name,
            user_prompt_chars=len(setup.user_prompt),
        )
        setup.beacons.record(
            run_from_totals(
                totals,
                image_files,
                beacons=setup.beacons,
                started_at=started_at,
                stats=stats,
                plan=plan,
            ),
        )

    with batch_progress(len(image_files), enabled=setup.run.display.progress_bar) as progress:
        run_batch(
            image_files,
            setup.agent,
            setup.options,
            on_success=setup.on_success,
            on_complete=_on_complete,
            user_prompt=setup.user_prompt,
            workers=setup.workers,
            progress=progress,
            cache=setup.cache,
            on_image_result=combine_image_result_callbacks(setup.on_image_result, stats),
            session_plan=plan,
            journal=setup.journal,
            pacer=setup.pacer,
        )


def filter_batch(
    image_files: list[Path],
    run: RunOptions,
    *,
    newer_than: datetime | None,
    older_than: datetime | None,
) -> list[Path]:
    """Apply the skip list, the date window, and --skip-tagged to a resolved batch."""
    image_files = apply_skip_file(image_files, run.artifacts.skip_from)
    image_files = apply_date_filter(image_files, newer_than=newer_than, older_than=older_than)
    return apply_skip_tagged(image_files, skip_tagged=run.filter_.skip_tagged)


def was_interrupted(error: BatchError) -> bool:
    """Say whether *error* reports a batch that a Ctrl-C cut short."""
    totals = error.args[0] if error.args else None
    return isinstance(totals, BatchTotals) and totals.interrupted
