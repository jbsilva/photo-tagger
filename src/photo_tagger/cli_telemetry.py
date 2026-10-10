"""
The CLI's telemetry: how a tag or watch run was set up, and what each batch reports.

The counterpart of :mod:`photo_tagger.gui_usage` for the command line. It turns the parsed option
groups into the settings and setup a beacon carries, and the totals of a finished batch into its run
beacon, so :mod:`photo_tagger.main` only says when to send.
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
    ArtifactConfig,
    DisplayConfig,
    FilterConfig,
    InferenceConfig,
    LogConfig,
    OutputConfig,
    PacingConfig,
    ProviderConfig,
    WatchConfig,
    known_config_keys,
    to_telemetry_settings,
)
from photo_tagger.config_file import config_file_kind, load_config
from photo_tagger.telemetry_facts import (
    changed_settings,
    config_key_names,
    env_var_names,
    joined,
    mean,
    quantile,
)


if TYPE_CHECKING:
    from pathlib import Path

    from photo_tagger.cli_options import Invocation
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


@dataclass(slots=True, frozen=True)
class RunGroups:
    """The option groups one tag or watch invocation parsed, for the telemetry describing it."""

    provider: ProviderConfig
    inference: InferenceConfig
    output: OutputConfig
    artifacts: ArtifactConfig
    display: DisplayConfig
    filter_: FilterConfig
    log: LogConfig
    pacing: PacingConfig
    telemetry_enabled: bool
    workers: int
    extensions: str
    recursive: bool
    watch: WatchConfig | None = None


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
    def for_invocation(cls, command: str, groups: RunGroups, invocation: Invocation) -> Beacons:
        """
        Snapshot what *groups* set up, before the run can fail.

        Telemetry must never be why a run fails, so a snapshot that raises (a bug, or an input no
        test foresaw) leaves the invocation with telemetry off instead of a traceback.
        """
        try:
            return cls._snapshot(command, groups, invocation)
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
    def _snapshot(cls, command: str, groups: RunGroups, invocation: Invocation) -> Beacons:
        """Build the beacons of an invocation from its option groups."""
        settings = to_telemetry_settings(
            provider=groups.provider,
            inference=groups.inference,
            output=groups.output,
            artifacts=groups.artifacts,
            display=groups.display,
            filter_=groups.filter_,
            log=groups.log,
            pacing=groups.pacing,
            workers=groups.workers,
            extensions=groups.extensions,
            recursive=groups.recursive,
            watch=groups.watch,
        )
        return cls(
            enabled=groups.telemetry_enabled,
            command=command,
            provider=groups.provider.provider_name,
            model=groups.provider.model_name,
            output_language=groups.inference.output_language,
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
        self._seconds: list[float] = []
        self._keywords: list[float] = []
        self._titles: list[float] = []
        self._descriptions: list[float] = []

    def __call__(self, outcome: ImageOutcome) -> None:
        """Keep one finished photo's model time and text sizes, never the text."""
        if not outcome.success:
            return
        with self._lock:
            if not outcome.from_cache and outcome.seconds > 0:
                self._seconds.append(outcome.seconds)
            self._keywords.append(len(outcome.keywords))
            self._titles.append(len(outcome.title or ""))
            self._descriptions.append(len(outcome.description or ""))

    def apply(self, run: telemetry.RunInfo) -> telemetry.RunInfo:
        """Return *run* with the per-photo figures filled in."""
        return replace(
            run,
            photo_seconds_median=quantile(self._seconds, 0.5),
            photo_seconds_p90=quantile(self._seconds, 0.9),
            keywords_mean=mean(self._keywords),
            title_chars_mean=mean(self._titles),
            description_chars_mean=mean(self._descriptions),
        )


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
