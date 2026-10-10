"""The Undo Writes dialog's texts: journal names, actions, and the summary of a run."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from photo_tagger.i18n import _, gettext_noop, ngettext
from photo_tagger.undo import (
    CHANGED,
    DELETED,
    FAILED as UNDO_FAILED,
    MISSING,
    NO_BACKUP,
    RESTORED,
)


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from photo_tagger.undo import UndoResult, WriteRecord


# How to read the timestamp a journal's filename starts with, by its width. The shorter one is a
# journal written before the stamp carried microseconds. Width decides, because strptime is happy
# to read the second-only stamp under the longer format and land on the wrong time.
_JOURNAL_STAMP_FORMATS = {20: "%Y%m%d%H%M%S%f", 14: "%Y%m%d%H%M%S"}


# What each undo outcome is called in the window. The module's own constants are log-facing
# identifiers; these are the phrases a user reads next to a file name.
_UNDO_ACTION_LABELS = {
    RESTORED: gettext_noop("restored"),
    DELETED: gettext_noop("deleted"),
    MISSING: gettext_noop("no longer there"),
    CHANGED: gettext_noop("changed since the run"),
    NO_BACKUP: gettext_noop("no backup to restore from"),
    UNDO_FAILED: gettext_noop("failed"),
}


# Outcomes that mean the file is back as it was; everything else was left alone.
UNDO_OK_ACTIONS = frozenset({RESTORED, DELETED})


def journal_time(path: Path) -> datetime | None:
    """Parse a journal's start time out of its filename, or None when the name is not ours."""
    stamp = path.name.split("-", 1)[0]
    time_format = _JOURNAL_STAMP_FORMATS.get(len(stamp))
    if time_format is None:
        return None
    try:
        return datetime.strptime(stamp, time_format).replace(tzinfo=UTC)
    except ValueError:
        return None


def journal_label(path: Path, entries: int) -> str:
    """Describe one recorded run for the undo list: when it ran, and how much it wrote."""
    started = journal_time(path)
    when = started.astimezone().strftime("%Y-%m-%d %H:%M") if started else path.stem
    files = ngettext("{n} file", "{n} files", entries).format(n=entries)
    return f"{when} · {files}"


def undo_action_label(action: str) -> str:
    """Translate one undo outcome for display, falling back to the raw name."""
    label = _UNDO_ACTION_LABELS.get(action)
    return _(label) if label else action


def undo_summary(results: Iterable[UndoResult]) -> str:
    """One line for the status bar: how many writes were put back, and how many were left alone."""
    outcomes = list(results)
    if not outcomes:
        return _("That run recorded no writes.")
    ok = sum(1 for result in outcomes if result.action in UNDO_OK_ACTIONS)
    left = len(outcomes) - ok
    restored = ngettext("Put back {n} file", "Put back {n} files", ok).format(n=ok)
    if not left:
        return f"{restored}."
    return _("{restored}; left {left} alone.").format(restored=restored, left=left)


def reverted_images(records: Iterable[WriteRecord], results: Iterable[UndoResult]) -> list[str]:
    """
    Name the photos an undo put back, each once, in the order their writes were reverted.

    A photo is back once any file written for it was restored or deleted. Everything read from such
    a photo is stale afterwards, so the window reads it again.
    """
    images = {record.target: record.image for record in records}
    reverted = (
        images[str(result.target)]
        for result in results
        if result.action in UNDO_OK_ACTIONS and str(result.target) in images
    )
    return list(dict.fromkeys(reverted))


def undo_details_text(results: Iterable[UndoResult], *, dry_run: bool) -> str:
    """Render one line per recorded write: what happened to it, and why when it was left alone."""
    lines = [_("What undoing would do:") if dry_run else _("What was undone:")]
    for result in results:
        detail = f" ({result.detail})" if result.detail and not dry_run else ""
        lines.append(f"  {undo_action_label(result.action)}: {result.target.name}{detail}")
    return "\n".join(lines)
