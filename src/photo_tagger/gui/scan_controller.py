# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The background metadata scan behind the Tagged column, and the row under the list for it."""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QToolButton, QWidget

from photo_tagger.gui.threads import release_thread, start_in_thread
from photo_tagger.gui.workers import MetadataScanWorker
from photo_tagger.gui_state import tooltip
from photo_tagger.i18n import _, gettext_noop


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable, Iterable

    from PySide6.QtCore import QThread

    from photo_tagger.gui_state import PhotoItem, TaggedCriterion


# Grace period stop() gives an in-flight batched exiftool call to finish on its own before
# detaching rather than blocking the caller. Comfortably above a normal scan's duration.
_SCAN_STOP_TIMEOUT_MS = 3000
# How long a metadata scan must run before the window reports it. A local folder is read well
# inside this, so the usual case never shows a progress bar at all; a network share blows past it.
_SCAN_PROGRESS_DELAY_MS = 700

_STOPPED_NOTHING_UNCHECKED = gettext_noop("Stopped reading metadata. No photo was unchecked.")


class ScanController(QObject):
    """
    Read which fields each photo already carries, off the UI thread, one batch at a time.

    The window owns the photos, so the scan reads them through the callables its constructor takes,
    and hands back what it read with :attr:`fields_read`. An "Uncheck Already Tagged" pick that
    needs the scan to cover every photo first waits here (:meth:`wait_for`), and comes back through
    :attr:`uncheck_requested` once it has. It reports a status line with :attr:`status`.
    """

    status = Signal(str)
    fields_read = Signal(object)  # dict[str, set[str]]: the fields found on each photo, by key
    uncheck_requested = Signal(object)  # the TaggedCriterion that waited for the scan

    def __init__(
        self,
        parent: QWidget,
        *,
        photos: Callable[[], Iterable[tuple[str, PhotoItem]]],
        photo: Callable[[str], PhotoItem | None],
        closing: Callable[[], bool],
    ) -> None:
        """
        Build the scan and its row; nothing reads until :meth:`start`.

        *photos* lists the photos of the list by key, *photo* looks one up by key, and *closing*
        says whether the window is going away.
        """
        super().__init__(parent)
        self._photos = photos
        self._photo = photo
        self._closing = closing
        self._thread: QThread | None = None
        self._worker: MetadataScanWorker | None = None
        # The "Uncheck Already Tagged" pick waiting for the scan to cover every photo, and whether
        # the user stopped that scan. Both None/False while no one is waiting on a scan.
        self._pending: TaggedCriterion | None = None
        self._cancelled = False
        # A scan was asked for while Stop wound the last one down: a new request, not the old one.
        self._again = False
        # The waiting uncheck was picked after Stop, so it is a new one, not the one Stop dropped.
        self._picked_after_stop = False
        # The photos the running scan was handed, and the ones a finished scan could not read (see
        # _on_finished): those are not handed out again until they leave the list.
        self._batch: set[str] = set()
        self._tried: set[str] = set()
        # Photos of the running scan's batch that an undo rewrote: what it reads of them is stale.
        self._stale: set[str] = set()
        self.row = self._build_row()

    def _build_row(self) -> QWidget:
        """
        Build the row under the list that reports the metadata scan behind the Tagged column.

        Hidden while nothing is scanning. It carries its own Stop button rather than borrowing the
        bottom bar's Cancel, because a scan and a generate/save run can be in flight at once (a
        watch adds photos mid-run) and the two must stay separately stoppable.
        """
        self._progress = QProgressBar()
        self._stop_button = QToolButton()
        self._stop_button.setText(_("Stop"))
        self._stop_button.setToolTip(
            tooltip(
                "Stop reading metadata. The photos already read keep their Tagged column; the "
                "rest stay unknown until the next scan. Nothing is written either way.",
            ),
        )
        self._stop_button.clicked.connect(self.cancel)
        # Only show the row for a scan that actually runs long enough to be worth reporting, so a
        # handful of local photos does not make a progress bar flash on and off.
        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.setInterval(_SCAN_PROGRESS_DELAY_MS)
        self._show_timer.timeout.connect(self.show_row)

        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._progress, stretch=1)
        row.addWidget(self._stop_button)
        row_widget.setVisible(False)
        return row_widget

    def start(self) -> None:
        """Scan photos with an unknown Tagged state in the background, one batch at a time."""
        if self._thread is not None:
            # A running scan picks up stragglers when it finishes. One that is stopping would not,
            # so a photo added or an uncheck picked now asks it to start again.
            self._again = self._again or self._cancelled
            return
        # A photo a scan already tried and could not read stays unknown; asking again straight away
        # would only fail again, and restart the scan forever.
        pending = [
            item.path
            for key, item in self._photos()
            if item.known_fields is None and key not in self._tried
        ]
        if not pending:
            return
        self._batch = {str(path) for path in pending}
        self._stale = set()
        self._worker = MetadataScanWorker(pending)
        self._worker.done.connect(self._on_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._progress.setFormat(_("Reading metadata... %v / %m"))
        self._stop_button.setEnabled(True)
        self._thread = start_in_thread(self, self._worker)
        self._show_timer.start()

    def wait_for(self, criterion: TaggedCriterion) -> None:
        """
        Park *criterion* until the scan has covered every photo, then ask for it to be applied.

        Reading a few thousand photos off a network share takes minutes, and doing it in the click
        handler froze the whole window, so the uncheck waits for the scan instead.
        """
        self._pending = criterion  # a second pick replaces one that is still waiting
        self._picked_after_stop = self._cancelled
        self.status.emit(
            _("Reading metadata before unchecking photos with {phrase}...").format(
                phrase=_(criterion.phrase),
            ),
        )
        unknown = [key for key, item in self._photos() if item.known_fields is None]
        if all(key in self._tried for key in unknown):
            # Nothing left that a scan can read: settle on what is known, unknown photos kept.
            self.end(stopped=False)
            return
        self.start()
        self.show_row()  # the user is waiting on this scan, so do not hold the bar back

    def _on_progress(self, scanned: int, total: int) -> None:
        """Move the scan's progress bar as each batch of photos is read."""
        self._progress.setRange(0, total)
        self._progress.setValue(scanned)

    def show_row(self) -> None:
        """Reveal the scan row, unless the scan it would report has already finished."""
        self.row.setVisible(self._thread is not None)

    def hide_row(self) -> None:
        """Put the scan row away, and cancel a pending reveal so it cannot pop up after."""
        self._show_timer.stop()
        self.row.setVisible(False)

    def cancel(self) -> None:
        """Stop the metadata scan after the batch it is on (the Stop button under the list)."""
        if self._worker is None:
            return
        self._cancelled = True
        self._worker.stop()
        self._stop_button.setEnabled(False)
        self._progress.setFormat(_("Stopping..."))

    def _on_done(self, presence: dict[str, set[str]]) -> None:
        """Hand on the scanned field sets, leaving out the photos an undo made stale."""
        self.fields_read.emit(
            {key: fields for key, fields in presence.items() if key not in self._stale},
        )

    def _on_finished(self) -> None:
        """Tear down the scan thread and pick up photos added while it ran."""
        if self._closing():
            # A scan detached by stop()'s timeout can finish after closeEvent already moved on;
            # the window is going away, so there is nothing left to restart a scan for.
            return
        stopped = self._cancelled
        again, self._again = self._again, False
        picked_after_stop, self._picked_after_stop = self._picked_after_stop, False
        self.stop()
        if stopped and again and not picked_after_stop and self._pending is not None:
            # Stop drops the uncheck it was pressed on, even when a photo added since then starts
            # the scan again. Only an uncheck picked after Stop waits for that new scan.
            self._pending = None
            self.status.emit(_(_STOPPED_NOTHING_UNCHECKED))
        if not stopped:
            # A scan that ran to its end read every photo it could; those still unknown are
            # unreadable, so the restart below leaves them out. A stopped scan tried nothing more.
            self._tried |= {
                key
                for key in self._batch
                if (item := self._photo(key)) is not None and item.known_fields is None
            }
        if not stopped or again:
            self._cancelled = False
            self.start()
        if self._thread is None:
            # Nothing left unknown (or the user stopped it): the scan is over for now.
            self.end(stopped=stopped and not again)

    def end(self, *, stopped: bool) -> None:
        """Put the scan row away and settle whatever was waiting on the scan."""
        self._cancelled = False
        self._again = False
        self.hide_row()
        criterion, self._pending = self._pending, None
        if criterion is None:
            return
        if stopped:
            self.status.emit(_(_STOPPED_NOTHING_UNCHECKED))
        else:
            self.uncheck_requested.emit(criterion)

    def stop(self) -> None:
        """Let go of the scan in flight, waiting for it only a short while."""
        if self._worker is not None:
            # Between batches the worker checks this, so a detached scan below stops reading the
            # rest of the library instead of running it out in the background.
            self._worker.stop()
            # Cut the worker loose before letting go of it. A scan detached by the timeout below
            # keeps running, and its late finished signal would otherwise reach _on_finished and
            # tear down whatever scan is current by then, not the one that just ended.
            self._worker.disconnect(self)
            self._worker.deleteLater()
        if self._thread is not None:
            # A big folder, or a hung exiftool, can outlast the timeout. pyexiftool exposes no way
            # to cancel a batched call mid-flight, and the scan only reads metadata, so losing its
            # result is harmless: detach rather than block the caller (closeEvent, a cleared list).
            release_thread(
                self._thread,
                timeout_ms=_SCAN_STOP_TIMEOUT_MS,
                detached_event="gui_metadata_scan_stop_timed_out",
            )
            self._thread = None
        self._worker = None

    def reset(self) -> None:
        """Stop the scan and forget every photo it tried: the list is being emptied."""
        self.stop()
        self.end(stopped=True)  # the list is gone, so nothing is left to uncheck in it
        self._tried.clear()

    def drop(self, keys: list[str]) -> None:
        """Tell the scan in flight to skip photos that are leaving the list."""
        if self._worker is not None:
            self._worker.drop(Path(key) for key in keys)

    def forget(self, key: str) -> None:
        """Forget a photo that left the list, so the scan treats it as new if it comes back."""
        self._tried.discard(key)  # added again later, it deserves a fresh read
        self._batch.discard(key)  # the scan in flight skips it, so it has not tried it

    def settle_after_removal(self) -> None:
        """
        Settle a waiting uncheck that no scan will finish for, against the photos that are left.

        The photos it was still waiting to have scanned may be the ones just removed, and with no
        scan running nothing else would ever settle it.
        """
        if self._pending is not None and self._thread is None:
            self.end(stopped=False)

    def reread(self, key: str) -> None:
        """Read *key* again on the next scan: the file changed under the scan (an undo)."""
        self._tried.discard(key)
        if key in self._batch:
            # What the scan in flight read of it, or is about to deliver, is stale too.
            self._batch.discard(key)
            self._stale.add(key)
