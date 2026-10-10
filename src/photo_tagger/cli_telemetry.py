"""
The CLI's telemetry: how a tag or watch run was set up, and what each batch reports.

The counterpart of :mod:`photo_tagger.gui_usage` for the command line. It turns the parsed option
groups into the settings and setup a beacon carries, and the totals of a finished batch into its run
beacon, so the commands only say when to send.
"""

import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger import i18n, telemetry
from photo_tagger.cli_options import (
    known_config_keys,
    to_telemetry_settings,
)
from photo_tagger.config_file import config_file_kind, load_config
from photo_tagger.telemetry_facts import (
    changed_settings,
    config_key_names,
    env_var_names,
    joined,
)


if TYPE_CHECKING:
    from pathlib import Path

    from photo_tagger.cli_options import ConfigFileSource, Invocation, RunOptions
    from photo_tagger.pipeline import BatchTotals, ImageOutcome
    from photo_tagger.sessions import SessionPlan


def show_notice_if_due(*, enabled: bool) -> None:
    """
    Print the one-time telemetry disclosure to stderr on the first run telemetry is active.

    Only shown when telemetry will actually send (config/flag on and no environment opt-out), so a
    user who has disabled it never sees the notice. Written straight to stderr, not the loguru sink,
    so it reaches the terminal regardless of the configured log level.
    """
    if not telemetry.should_send(config_enabled=enabled):
        return
    notice = telemetry.first_run_notice()
    if notice is not None:
        sys.stderr.write(notice + "\n")


def invocation_setup(invocation: Invocation) -> telemetry.Setup:
    """Describe how this invocation was configured: the config file, the flags, the environment."""
    keys, unknown = config_key_names(load_config(), known_config_keys())
    return telemetry.Setup(
        config_file=config_file_kind(),
        config_keys=keys,
        unknown_config_keys=unknown,
        cli_flags=joined(invocation.cli_flags),
        env_vars=env_var_names(),
    )


@dataclass(slots=True)
class Beacons:
    """
    The telemetry of one tag or watch invocation: how it was set up, and what it has reported.

    Built before anything can fail, so a run that stops before its first photo is reported too.
    """

    enabled: bool
    command: str
    provider: str
    model: str
    output_language: str
    settings: telemetry.Settings
    setup: telemetry.Setup
    started: float = field(default_factory=time.monotonic)
    sent: bool = False
    runs: int = 0
    photos: int = 0
    busy_seconds: float = 0.0
    first_run_seconds: float = -1.0

    @classmethod
    def for_invocation(cls, command: str, options: RunOptions, invocation: Invocation) -> Beacons:
        """
        Snapshot what *options* set up, before the run can fail.

        Telemetry must never be why a run fails, so a snapshot that raises (a bug, or an input no
        test foresaw) leaves the invocation with telemetry off instead of a traceback.
        """
        try:
            return cls._snapshot(command, options, invocation)
        except Exception as exc:  # noqa: BLE001 - any snapshot bug must not stop the run.
            logger.warning("telemetry_snapshot_failed", error=repr(exc))
            # Off means off: the command beacon on the way out must not stand in for this run.
            telemetry.mark_reported()
            return cls(
                enabled=False,
                command=command,
                provider="",
                model="",
                output_language="",
                settings=telemetry.Settings(),
                setup=telemetry.Setup(),
            )

    @classmethod
    def _snapshot(cls, command: str, options: RunOptions, invocation: Invocation) -> Beacons:
        """Build the beacons of an invocation from its option groups."""
        settings = to_telemetry_settings(options)
        return cls(
            enabled=options.telemetry_enabled,
            command=command,
            provider=options.provider.provider_name,
            model=options.provider.model_name,
            output_language=options.inference.output_language,
            settings=settings,
            setup=invocation_setup(invocation),
        )

    def emit(self, event: telemetry.Event) -> None:
        """Send *event*, showing the one-time notice first if it is still due."""
        show_notice_if_due(enabled=self.enabled)
        telemetry.emit(event, enabled=self.enabled, block=True)
        self.sent = True

    def described(self) -> telemetry.Setup:
        """Return the setup, naming the settings that differ from the built-in defaults."""
        return replace(self.setup, changed=changed_settings(self.settings, telemetry.Settings()))

    def base_run(self) -> telemetry.RunInfo:
        """Return a run of this invocation with nothing done yet, for a call site to fill in."""
        return telemetry.RunInfo(
            interface="cli",
            command=self.command,
            trigger="watch" if self.command == "watch" else "",
            provider=self.provider,
            model=self.model,
            batch_size=0,
            duration_seconds=0.0,
            output_language=self.output_language,
            ui_language=i18n.current_language(),
            file_types="",
            settings=self.settings,
            setup=self.described(),
        )

    def record(self, run: telemetry.RunInfo) -> None:
        """Send one finished batch and add it to the session a watch reports at the end."""
        if not self.runs:
            self.first_run_seconds = time.monotonic() - self.started - run.duration_seconds
        self.runs += 1
        # Photos that came out done, as the GUI counts them: a failed photo was not generated.
        self.photos += run.success_count
        self.busy_seconds += run.duration_seconds
        self.emit(run)

    def stopped(self, error: SystemExit | KeyboardInterrupt) -> None:
        """Report a run that ended before its first photo, unless a beacon already covers it."""
        if self.sent or (isinstance(error, SystemExit) and not exit_status(error)):
            return
        interrupted = isinstance(error, KeyboardInterrupt)
        cause = error.__cause__ or error
        self.emit(
            replace(
                self.base_run(),
                duration_seconds=time.monotonic() - self.started,
                outcome=(
                    telemetry.OUTCOME_INTERRUPTED if interrupted else telemetry.OUTCOME_SETUP_FAILED
                ),
                error_kind="" if interrupted else type(cause).__name__,
            ),
        )

    def session(self) -> telemetry.SessionInfo:
        """Summarize a watch from start to stop: how long, how busy, how many batches."""
        elapsed = time.monotonic() - self.started
        return telemetry.SessionInfo(
            interface="cli",
            command=self.command,
            session_seconds=elapsed,
            provider=self.provider,
            model=self.model,
            output_language=self.output_language,
            ui_language=i18n.current_language(),
            busy_seconds=self.busy_seconds,
            idle_seconds=max(0.0, elapsed - self.busy_seconds),
            runs=self.runs,
            first_run_seconds=self.first_run_seconds,
            photos_generated=self.photos,
            settings=self.settings,
            setup=self.described(),
        )


class PhotoStats:
    """``on_image_result`` sink that keeps what telemetry reports per photo: timing and sizes."""

    def __init__(self) -> None:
        """Start empty; workers may call concurrently, hence the lock."""
        self._lock = threading.Lock()
        self._figures = telemetry.PhotoFigures()

    def __call__(self, outcome: ImageOutcome) -> None:
        """Keep one finished photo's model time and text sizes, never the text."""
        if not outcome.success:
            return
        with self._lock:
            self._figures.add(
                seconds=0.0 if outcome.from_cache else outcome.seconds,
                keywords=len(outcome.keywords),
                title=outcome.title or "",
                description=outcome.description or "",
            )

    def apply(self, run: telemetry.RunInfo) -> telemetry.RunInfo:
        """Return *run* with the per-photo figures filled in."""
        return self._figures.apply(run)


def run_from_totals(  # noqa: PLR0913 - the batch and its beacons and what the run tracked beside it.
    totals: BatchTotals,
    image_files: list[Path],
    *,
    beacons: Beacons,
    started_at: datetime,
    stats: PhotoStats,
    plan: SessionPlan | None,
) -> telemetry.RunInfo:
    """Describe one finished batch for its telemetry beacon."""
    run = replace(
        beacons.base_run(),
        batch_size=totals.total_files,
        duration_seconds=(datetime.now(tz=UTC) - started_at).total_seconds(),
        file_types=telemetry.file_types_summary(image_files),
        outcome=telemetry.outcome_for(
            totals.total_files,
            len(totals.failed_files),
            interrupted=totals.interrupted,
        ),
        success_count=totals.success,
        failure_count=len(totals.failed_files),
        initial_failures=totals.initial_failures,
        cache_hits=totals.cache_hits,
        retry_successes=totals.retry_successes,
        workers=totals.workers,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        inference_calls=totals.inference_calls,
        inference_seconds=totals.inference_seconds,
        dry_run=totals.dry_run,
        failure_kinds=telemetry.failure_kinds_summary(totals.failure_kinds),
        failure_types=telemetry.failure_kinds_summary(totals.failure_types),
        language_repairs=totals.language_repairs,
        flagged=len(totals.flagged_files),
        vocabulary_mapped=totals.vocabulary_mapped,
        vocabulary_dropped=len(totals.vocabulary_dropped),
        rested_seconds=totals.rested_seconds,
        rests_taken=totals.rests_taken,
        shoots=len(plan.sessions) if plan is not None else 0,
    )
    return stats.apply(run)


def exit_status(exc: SystemExit) -> int:
    """Return the status a SystemExit leaves the process with."""
    if exc.code is None:
        return 0
    return exc.code if isinstance(exc.code, int) else 1


# Exit statuses the command beacon names: cyclopts' own for a bad flag, and the shell's for Ctrl-C.
USAGE_EXIT = 2
INTERRUPTED_EXIT = 130


def telemetry_enabled_for(tokens: list[str]) -> bool:
    """
    Best-effort telemetry choice for an invocation without a parsed ``--no-telemetry``.

    A crash may happen before (or during) CLI parsing, and the other commands have no such flag, so
    scan the raw *tokens* for it and read the config file's [telemetry] table. The environment
    switches are enforced inside emit and emit_crash themselves.
    """
    if "--no-telemetry" in tokens:
        return False
    table = load_config().get("telemetry", {})
    return bool(table.get("enabled", True)) if isinstance(table, dict) else True


def _command_outcome(exit_code: int, error: BaseException | None) -> str:
    """Name how a command ended, from its exit status."""
    if isinstance(error, KeyboardInterrupt):
        return telemetry.OUTCOME_INTERRUPTED
    if exit_code == 0:
        return telemetry.OUTCOME_OK
    return telemetry.OUTCOME_USAGE_ERROR if exit_code == USAGE_EXIT else telemetry.OUTCOME_FAILED


@dataclass(slots=True)
class CommandReporter:
    """
    The beacons of one invocation as a whole: its crash, and the command beacon on the way out.

    A tag or watch run reports itself through :class:`Beacons`. This covers every other command, and
    a run that stopped before it got that far. *commands* names the CLI's subcommands, and *source*
    is read only once the command line has been parsed, for the flags it recorded.
    """

    tokens: list[str]
    commands: frozenset[str]
    source: ConfigFileSource
    started: float = field(default_factory=time.monotonic)

    @property
    def command(self) -> str:
        """Name what the tokens run: a subcommand, help, the version, or the default ``tag``."""
        if {"--help", "-h"}.intersection(self.tokens):
            return "help"
        if "--version" in self.tokens:
            return "version"
        first = self.tokens[0] if self.tokens else ""
        return first if first in self.commands else "tag"

    def crashed(self, exc: Exception) -> None:
        """Send the crash beacon of *exc*; the window reports its own, under its own consent."""
        if self.command == "gui":
            return
        enabled = telemetry_enabled_for(self.tokens)
        # A crash on the very first run is still the first beacon, so the disclosure comes first.
        show_notice_if_due(enabled=enabled)
        telemetry.emit_crash(
            exc,
            interface="cli",
            command=self.command,
            enabled=enabled,
            block=True,
        )

    def finished(self, exit_code: int, error: BaseException | None) -> None:
        """
        Send the beacon for an invocation that sent none of its own.

        That is every command but ``tag``, ``watch``, and ``gui``, plus a ``tag`` or ``watch`` that
        stopped at the command line. The GUI is left out: its window reports itself, under its own
        toggle. Never raises, because it runs while the real exit is on its way out.
        """
        command = self.command
        if telemetry.sent_any() or command == "gui":
            return
        try:
            enabled = telemetry_enabled_for(self.tokens)
            if not telemetry.should_send(config_enabled=enabled):
                return
            show_notice_if_due(enabled=enabled)
            failed = error is not None and not isinstance(error, KeyboardInterrupt)
            telemetry.emit(
                telemetry.CommandInfo(
                    command=command,
                    outcome=_command_outcome(exit_code, error),
                    exit_code=exit_code,
                    duration_seconds=time.monotonic() - self.started,
                    error_kind=type(error).__name__ if failed else "",
                    setup=invocation_setup(self.source.invocation),
                    details=telemetry.command_notes(),
                ),
                enabled=enabled,
                block=True,
            )
        except Exception as exc:  # noqa: BLE001 - a beacon must never change how the CLI exits.
            logger.debug("telemetry_command_report_failed", error=str(exc))
