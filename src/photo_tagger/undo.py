"""
Record what a run wrote, so it can be put back.

A single bad prompt across five hundred photos used to be unwound by hand: ExifTool leaves
``*_original`` backups and photo-tagger creates sidecars, but nothing tied them to a run or knew
which sidecars existed beforehand. This module keeps that record.

Every write appends one line to a JSON-lines journal under the state directory (see
:func:`photo_tagger.config.state_dir`). ``photo-tagger undo`` reads the newest journal back and, for
each entry, either restores the ExifTool backup over the file or deletes the sidecar the run
created. Recording is on by default precisely because the runs people want to undo are the ones they
did not plan to.

Safety comes from the recorded size and mtime: an entry whose file has changed since the run is
refused rather than reverted, because that change is the user's later edit, not ours.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

from photo_tagger.config import state_dir
from photo_tagger.errors import PhotoTaggerError


if TYPE_CHECKING:
    from collections.abc import Iterable

    from photo_tagger.write_plan import WriteTarget


# ExifTool's backup suffix. It appends this to the full name, so "IMG.xmp" backs up to
# "IMG.xmp_original" rather than "IMG_original.xmp".
BACKUP_SUFFIX = "_original"

_JOURNAL_SUFFIX = ".jsonl"
# Slack when comparing a recorded mtime with the one on disk. Float seconds do not survive a
# JSON round trip exactly, and this is far below any real edit interval.
_MTIME_TOLERANCE_SECONDS = 1e-6
# Journals are a few hundred bytes per photo, but they must not accumulate forever on a machine
# that tags every day. Both limits apply: the newest N, and nothing older than the cutoff.
_KEEP_RUNS = 50
_KEEP_DAYS = 90


class UndoError(PhotoTaggerError):
    """There is nothing to undo, or the journal cannot be read."""


@dataclass(slots=True, frozen=True)
class WriteRecord:
    """
    One file a run wrote, and what putting it back would take.

    ``target`` is the file that changed: the sidecar, or the photo itself when the run embeds.
    ``backup`` is ExifTool's ``*_original`` copy of its previous contents, absent when the target
    did not exist before (``created``) or when the run disabled backups. ``size`` and ``mtime``
    describe the target as we left it, so a later edit can be detected before anything is reverted.
    """

    image: str
    target: str
    created: bool
    backup: str | None = None
    size: int = 0
    mtime: float = 0.0

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self | None:
        """Build a record from one journal line, or None when the line is unusable."""
        image, target = payload.get("image"), payload.get("target")
        if not isinstance(image, str) or not isinstance(target, str):
            return None
        backup = payload.get("backup")
        return cls(
            image=image,
            target=target,
            created=bool(payload.get("created", False)),
            backup=backup if isinstance(backup, str) else None,
            size=int(payload.get("size", 0)),
            mtime=float(payload.get("mtime", 0.0)),
        )


def runs_dir() -> Path:
    """Return the directory holding the per-run undo journals."""
    return state_dir() / "runs"


def _journal_name(started_at: datetime) -> str:
    """
    Name a journal after its start time and pid, so two runs cannot collide.

    The timestamp carries microseconds because one process can start two runs inside a second: the
    desktop GUI opens a journal per save, and two quick batches would otherwise share a name and be
    undone as one. Fixed width either way, so sorting by name still sorts by time.
    """
    return f"{started_at.strftime('%Y%m%d%H%M%S%f')}-{os.getpid()}{_JOURNAL_SUFFIX}"


def list_journals(directory: Path | None = None) -> list[Path]:
    """
    Return the undo journals, newest first.

    The name starts with a fixed-width timestamp (callers pass UTC), so sorting by name sorts by
    time without a stat of every file.
    """
    folder = runs_dir() if directory is None else directory
    try:
        return sorted(folder.glob(f"*{_JOURNAL_SUFFIX}"), key=lambda p: p.name, reverse=True)
    except OSError as exc:
        logger.warning("undo_journal_list_failed", folder=str(folder), error=str(exc))
        return []


def latest_journal(directory: Path | None = None) -> Path | None:
    """Return the most recent undo journal, or None when there is none."""
    journals = list_journals(directory)
    return journals[0] if journals else None


def prune_journals(directory: Path | None = None) -> int:
    """
    Delete journals past the retention limits; return how many went.

    Best-effort: a failure here must never disturb a run, so every error is logged and swallowed.
    """
    journals = list_journals(directory)
    cutoff = datetime.now(tz=UTC).timestamp() - _KEEP_DAYS * 86400
    removed = 0
    for index, path in enumerate(journals):
        try:
            too_old = path.stat().st_mtime < cutoff
        except OSError:
            continue
        if index < _KEEP_RUNS and not too_old:
            continue
        try:
            path.unlink()
        except OSError as exc:  # pragma: no cover - unlikely, and never worth failing a run over
            logger.warning("undo_journal_prune_failed", file=str(path), error=str(exc))
            continue
        removed += 1
    if removed:
        logger.debug("undo_journals_pruned", removed=removed)
    return removed


class UndoJournal:
    """
    Append-only record of what one run wrote.

    The file is created on the first record, so a dry run (or a run where every photo failed) leaves
    nothing behind. Writes are serialized: worker threads record concurrently, and a half-written
    line would cost the entry it belongs to.

    Every failure here is logged and swallowed. Losing the ability to undo is bad; failing a run
    that is otherwise writing metadata correctly is worse.
    """

    __slots__ = ("_broken", "_entries", "_lock", "_own_backups", "_path")

    def __init__(self, path: Path) -> None:
        """Prepare (but do not create) the journal at *path*."""
        self._path = path
        self._lock = threading.Lock()
        self._entries = 0
        self._broken = False
        # Targets this journal recorded a backup for. ExifTool keeps the first backup of a file, so
        # a later write of the same file in this run still owns it (see record).
        self._own_backups: set[str] = set()

    @property
    def path(self) -> Path:
        """Return the journal's path, whether or not it exists yet."""
        return self._path

    @property
    def entries(self) -> int:
        """Return how many writes have been recorded."""
        return self._entries

    def record(
        self,
        image_path: Path,
        target: Path,
        *,
        created: bool,
        backed_up: bool = True,
    ) -> None:
        """
        Record that *target* was written for *image_path*.

        *created* says the target did not exist before the write, which is what tells undo to delete
        the file rather than look for a backup.

        *backed_up* says this write left a backup of its own: backups were on, and no ``*_original``
        was there before it (see :meth:`WriteTarget.made_backup`). ExifTool keeps an existing backup
        and writes the file anyway, so a backup that predates the write belongs to an earlier run or
        to the user's own ExifTool use. Adopting it would point undo at content this run never
        wrote, and restoring it would destroy every edit made in between.
        """
        if self._broken:
            return
        if backed_up:
            self._own_backups.add(str(target))
        elif str(target) in self._own_backups:
            # A second write of a file this run already backed up: the backup is still ours.
            backed_up = True
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        try:
            stat = target.stat()
        except OSError as exc:
            logger.warning("undo_record_stat_failed", file=str(target), error=str(exc))
            return
        record = WriteRecord(
            image=str(image_path),
            target=str(target),
            created=created,
            backup=str(backup) if backed_up and backup.exists() else None,
            size=stat.st_size,
            mtime=stat.st_mtime,
        )
        self._append(record)

    def record_writes(
        self,
        image_path: Path,
        targets: Iterable[WriteTarget],
        *,
        backup: bool,
    ) -> None:
        """Record every file one save wrote for *image_path*, with backups set to *backup*."""
        for target in targets:
            self.record(
                image_path,
                target.path,
                created=not target.existed,
                backed_up=target.made_backup(backup=backup),
            )

    def _append(self, record: WriteRecord) -> None:
        """Serialize one record to the journal, disabling the journal on a write failure."""
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                self._entries += 1
        except OSError as exc:
            logger.warning("undo_journal_write_failed", file=str(self._path), error=str(exc))
            # One failure means the next will fail the same way (unwritable dir, full disk).
            # Stop trying rather than log once per photo for the rest of the run.
            self._broken = True


def open_journal(started_at: datetime, *, enabled: bool = True) -> UndoJournal | None:
    """Return a journal for a run started at *started_at*, or None when recording is off."""
    if not enabled:
        return None
    prune_journals()
    journal = UndoJournal(runs_dir() / _journal_name(started_at))
    logger.debug("undo_journal_ready", file=str(journal.path))
    return journal


def read_journal(path: Path) -> list[WriteRecord]:
    """
    Read a journal into records, skipping lines that are not usable.

    A run killed mid-write can leave a truncated final line; dropping it costs one entry, while
    refusing the whole file would cost the entire undo.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Could not read undo journal {path}: {exc}"
        raise UndoError(msg) from exc

    records: list[WriteRecord] = []
    skipped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        record = WriteRecord.from_json(payload) if isinstance(payload, dict) else None
        if record is None:
            skipped += 1
            continue
        records.append(record)
    if skipped:
        logger.warning("undo_journal_lines_skipped", file=str(path), skipped=skipped)
    return records


# What became of one entry. Reported per file and tallied in the summary.
RESTORED = "restored"
DELETED = "deleted"
MISSING = "missing"
CHANGED = "changed"
NO_BACKUP = "no-backup"
FAILED = "failed"


@dataclass(slots=True, frozen=True)
class UndoResult:
    """One entry's outcome, ready to print and to count."""

    target: Path
    action: str
    detail: str = ""


def _target_state(record: WriteRecord, target: Path) -> str | None:
    """Return the blocking state of *target* (missing or changed), or None when it is safe."""
    try:
        stat = target.stat()
    except OSError:
        return MISSING
    # Compare both: a rewrite of the same length still moves the mtime, and a filesystem with a
    # coarse mtime can leave it equal while the size changes.
    if stat.st_size != record.size or abs(stat.st_mtime - record.mtime) > _MTIME_TOLERANCE_SECONDS:
        return CHANGED
    return None


def _existing_backup(record: WriteRecord) -> Path | None:
    """Return the recorded backup path, or None when it was never made or has since gone."""
    if not record.backup:
        return None
    backup = Path(record.backup)
    return backup if backup.exists() else None


def _blocked(
    record: WriteRecord,
    target: Path,
    backup: Path | None,
    *,
    force: bool,
) -> UndoResult | None:
    """Return the outcome that stops this entry from being reverted, or None to go ahead."""
    state = _target_state(record, target)
    if state == MISSING:
        return UndoResult(target, MISSING, "no longer there")
    if state == CHANGED and not force:
        return UndoResult(target, CHANGED, "changed since the run; pass --force to revert anyway")
    if backup is None and not record.created:
        return UndoResult(target, NO_BACKUP, "written without an ExifTool backup")
    return None


def _apply_undo(target: Path, backup: Path | None) -> UndoResult:
    """Delete a file the run created, or move its backup back over it."""
    try:
        if backup is None:
            target.unlink()
            return UndoResult(target, DELETED)
        backup.replace(target)
    except OSError as exc:
        return UndoResult(target, FAILED, str(exc))
    return UndoResult(target, RESTORED)


def _undo_one(record: WriteRecord, *, force: bool, dry_run: bool) -> UndoResult:
    """
    Put one recorded write back, or explain why it was left alone.

    A file the run *created* is deleted rather than restored, even if some ``*_original`` sits next
    to it: that copy predates this run, and "no sidecar" is what was there before.
    """
    target = Path(record.target)
    backup = None if record.created else _existing_backup(record)
    if (blocked := _blocked(record, target, backup, force=force)) is not None:
        return blocked
    if dry_run:
        return UndoResult(target, DELETED if backup is None else RESTORED, "dry run")
    return _apply_undo(target, backup)


def undo_run(
    records: list[WriteRecord],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> list[UndoResult]:
    """
    Undo every recorded write, newest first.

    Reverse order matters when one file was written more than once in a run: ExifTool keeps the
    backup its first write made, so the newest record restores the state from before the run.
    """
    results = [_undo_one(record, force=force, dry_run=dry_run) for record in reversed(records)]
    counts: dict[str, int] = {}
    for result in results:
        counts[result.action] = counts.get(result.action, 0) + 1
    logger.info("undo_finished", entries=len(records), dry_run=dry_run, counts=counts)
    return results
