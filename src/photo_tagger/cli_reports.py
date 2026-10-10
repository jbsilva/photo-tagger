"""
The report files and streams of a CLI run: NDJSON on stdout, the CSV report, and the summary.

Each sink is an ``on_image_result`` callback, so the pipeline knows nothing of formats. The GUI
writes its own CSV through csv_report, which is why this module lives on the CLI side.
"""

import contextlib
import json
import os
import tempfile
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from photo_tagger.csv_report import CsvReportWriter, ReportRow
from photo_tagger.prompt_context import select_camera_fields, select_location


if TYPE_CHECKING:
    from collections.abc import Callable

    from photo_tagger.pipeline import BatchTotals, ImageOutcome


class TextSink(Protocol):
    """
    Minimal text-output interface the NDJSON emitter relies on.

    ``sys.stdout`` and ``io.StringIO`` both satisfy this without any extra glue, so tests can pass a
    buffer and the production path uses the real stream.
    """

    def write(self, s: str, /) -> int:
        """Write *s* and return how many characters went out."""
        raise NotImplementedError  # pragma: no cover - structural Protocol, never called

    def flush(self) -> None:
        """Push what was written to the reader, so each NDJSON line arrives whole."""
        raise NotImplementedError  # pragma: no cover - structural Protocol, never called


class NDJSONEmitter:
    """
    Thread-safe ``on_image_result`` callback that writes NDJSON to a stream.

    Workers call this concurrently. The lock prevents two lines being interleaved in stdout under
    ``--workers > 1``. Each line is a complete JSON object that downstream consumers can parse with
    one ``json.loads`` per readline.
    """

    __slots__ = ("_lock", "_stream")

    _lock: threading.Lock
    _stream: TextSink

    def __init__(self, stream: TextSink) -> None:
        """Wrap *stream* (any object exposing ``write`` and ``flush``)."""
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, outcome: ImageOutcome) -> None:
        """Serialize *outcome* to a single NDJSON line and flush."""
        payload = {
            "file": str(outcome.file),
            "status": "ok" if outcome.success else "failed",
            "from_cache": outcome.from_cache,
            "retry": outcome.retry,
            "title": outcome.title,
            "description": outcome.description,
            "keywords": outcome.keywords,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "total_tokens": outcome.total_tokens,
            "seconds": outcome.seconds,
            "foreign_terms": outcome.foreign_terms,
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


def open_csv_report(path: Path | None) -> CsvReportWriter | None:
    """
    Open the CSV report at *path*, or log and skip the report on failure.

    Returns ``None`` when *path* is ``None`` or the file cannot be opened (parent dir unwritable,
    filesystem full). A report we cannot write should never block tagging, so the run proceeds
    without one, mirroring how the cache degrades.
    """
    if path is None:
        return None
    try:
        writer = CsvReportWriter(path)
    except OSError as exc:
        logger.error("csv_report_open_failed", file=str(path), error=str(exc))
        return None
    logger.info("csv_report_opened", file=str(path))
    return writer


def outcome_to_report_row(outcome: ImageOutcome) -> ReportRow:
    """Flatten a pipeline :class:`ImageOutcome` into a CSV :class:`ReportRow`."""
    model, lens, captured = select_camera_fields(outcome.camera_info)
    city, country = select_location(outcome.location_tags)
    return ReportRow(
        file=str(outcome.file),
        filename=outcome.file.name,
        status="ok" if outcome.success else "failed",
        title=outcome.title or "",
        description=outcome.description or "",
        keywords=list(outcome.written_keywords),
        hierarchical_keywords=list(outcome.hierarchical_keywords),
        existing_keywords=list(outcome.existing_keywords),
        camera_model=model or "",
        lens_model=lens or "",
        capture_date=captured or "",
        gps_position=outcome.gps_position or "",
        city=city or "",
        country=country or "",
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        total_tokens=outcome.total_tokens,
        seconds=outcome.seconds,
        from_cache=outcome.from_cache,
        retry=outcome.retry,
        foreign_terms=list(outcome.foreign_terms),
    )


class CsvImageResultSink:
    """``on_image_result`` callback that appends each outcome to the CSV report."""

    __slots__ = ("_writer",)

    def __init__(self, writer: CsvReportWriter) -> None:
        """Wrap an open :class:`CsvReportWriter`."""
        self._writer = writer

    def __call__(self, outcome: ImageOutcome) -> None:
        """Convert *outcome* to a report row and stream it to the file."""
        self._writer.write(outcome_to_report_row(outcome))


def combine_image_result_callbacks(
    *callbacks: Callable[[ImageOutcome], None] | None,
) -> Callable[[ImageOutcome], None] | None:
    """
    Fan one ImageOutcome out to every non-None *callback*, in order.

    Lets ``--json`` and ``--csv-file`` both observe each photo from the single ``on_image_result``
    hook. Returns the lone callback when only one is active, or ``None`` when none are, so the
    common single-sink path adds no wrapper.
    """
    active = [cb for cb in callbacks if cb is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _fan_out(outcome: ImageOutcome) -> None:
        for callback in active:
            callback(outcome)

    return _fan_out


def atomic_write_text(target: Path, text: str) -> None:
    """
    Write *text* to *target* via a temp file + rename.

    Avoids leaving a half-written file on disk if the process is killed or the filesystem fills mid-
    write. The temp file lives in the same directory so the rename is a same-filesystem op, which
    POSIX guarantees is atomic.

    Uses ``tempfile.mkstemp`` instead of a PID-based name so the temp path is unpredictable and
    opened with ``O_EXCL``, preventing symlink-based attacks in shared directories.

    Creates the parent directory as needed so callers can point at a fresh location like
    ``reports/run.json`` without pre-mkdir.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1  # fdopen owns the descriptor now and closes it with the with-block.
            fh.write(text)
        tmp_path.replace(target)
    except BaseException:
        if fd != -1:
            # fdopen itself failed, so the raw descriptor is still open. Close it before the
            # unlink: Windows refuses to delete a file that still has an open handle.
            with contextlib.suppress(OSError):
                os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise


def write_summary_file(  # noqa: PLR0913 - distinct optional fields are clearer as kwargs.
    summary_file: Path | None,
    totals: BatchTotals | None,
    *,
    started_at: datetime,
    model_name: str,
    provider_name: str,
    user_prompt_chars: int,
) -> None:
    """
    Serialize *totals* to *summary_file* as JSON.

    Errors are logged, never raised.
    """
    if summary_file is None or totals is None:
        return
    payload: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "provider": provider_name,
        "model": model_name,
        "user_prompt_chars": user_prompt_chars,
        **asdict(totals),
    }
    try:
        atomic_write_text(summary_file, json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        logger.error("summary_file_write_failed", file=str(summary_file), error=str(exc))
        return
    logger.info("summary_file_written", file=str(summary_file))


def alert_message(totals: BatchTotals) -> str:
    """
    Word the one line the desktop alert carries: what the batch got through, and what it did not.

    Deliberately short, because a notification is read at a glance from across the room. The log,
    the summary file, and the CSV report are all still there for anyone who wants the detail.
    """
    photos = "photo" if totals.total_files == 1 else "photos"
    verb = "checked" if totals.dry_run else "tagged"
    line = f"{totals.success} of {totals.total_files} {photos} {verb}"
    if totals.interrupted:
        # After a Ctrl-C the photos never started are in failed_files too, but none of them failed.
        line = f"Stopped: {line}"
    elif failed := len(totals.failed_files):
        line = f"{line}, {failed} failed"
    return f"Dry run: {line}." if totals.dry_run else f"{line}."
