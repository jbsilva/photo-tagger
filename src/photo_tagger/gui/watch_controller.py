# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The folder watch behind Tools > Watch Folder: its worker, its queue, and its pacer."""

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction

from photo_tagger.gui.threads import release_thread, start_in_thread
from photo_tagger.gui.watch_dialog import WatchDialog
from photo_tagger.gui.workers import WatchWorker
from photo_tagger.gui_state import PENDING, tooltip, watch_status_text
from photo_tagger.gui_usage import TRIGGER_WATCH
from photo_tagger.i18n import _
from photo_tagger.pacing import build_pacer


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable
    from pathlib import Path

    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QWidget

    from photo_tagger.gui_state import PhotoItem, WatchSettings
    from photo_tagger.pacing import Pacer, PacingPlan


# The grace period for the folder watcher, which reacts to a stop within a fraction of a second
# unless it is inside a slow poll (a huge folder, or a network share).
_WATCH_STOP_TIMEOUT_MS = 3000


class WatchController(QObject):
    """
    Poll a folder on a background thread and hand each new photo to the window.

    The window owns the photos and the runs, so the watch asks it for what it needs through the
    callables its constructor takes, and asks it to generate with :attr:`generate_requested`. It
    reports a status line with :attr:`status` and the feature used with :attr:`used`. Photos that
    land while a run is in flight wait in a queue, which the window drains with :meth:`take_queued`
    once that run is over.
    """

    status = Signal(str)
    used = Signal(str)
    generate_requested = Signal(object)  # list[PhotoItem]: the queued photos still pending

    def __init__(  # noqa: PLR0913 - each callable is one thing the watch asks the window for.
        self,
        parent: QWidget,
        *,
        recursive: bool,
        extensions: Callable[[], str],
        add_photos: Callable[[list[Path]], list[Path]],
        photo: Callable[[str], PhotoItem | None],
        busy: Callable[[], bool],
        closing: Callable[[], bool],
        pacing_plan: Callable[[], PacingPlan],
    ) -> None:
        """
        Build the watch and its dialog; nothing polls until :meth:`start`.

        *recursive* seeds the dialog's subfolder toggle. *extensions* reads the folder-scan file
        types, *add_photos* adds paths to the list and returns the new ones, *photo* looks one up by
        key, *busy* says whether a run holds the window, *closing* whether the window is going away,
        and *pacing_plan* reads the plan a watch batch rests by.
        """
        super().__init__(parent)
        self._extensions = extensions
        self._add_photos = add_photos
        self._photo = photo
        self._busy = busy
        self._closing = closing
        self._pacing_plan = pacing_plan
        self._dialog = WatchDialog(parent, recursive=recursive)
        self._thread: QThread | None = None
        self._worker: WatchWorker | None = None
        # What the running watch was started with, and what it has picked up. None while idle.
        self._settings: WatchSettings | None = None
        # One pacer for every batch of the watch (see pacer_for), built at its first batch.
        self._pacer: Pacer | None = None
        self._added = 0
        # Photos the watch added while a run was already in flight, generated when it frees up.
        self._pending: list[str] = []
        self.action = QAction(_("Watch Folder..."), self)
        self.action.setToolTip(
            tooltip(
                "Tag photos as they arrive: point it at the folder your card reader or sync client "
                "fills, and each new photo joins the list and is generated for review.",
            ),
        )
        self.action.triggered.connect(self.toggle)

    @property
    def settings(self) -> WatchSettings | None:
        """What the running watch was started with, or None while no watch runs."""
        return self._settings

    @property
    def saves(self) -> bool:
        """Whether a watch runs that writes what it generates without a review."""
        return self._settings is not None and self._settings.save

    def toggle(self) -> None:
        """Stop the running watch, or open the dialog to start one."""
        if self._settings is not None:
            self.stop()
            return
        settings = self._dialog.ask(extensions=self._extensions())
        if settings is not None:
            self.start(settings)

    def start(self, settings: WatchSettings) -> None:
        """Poll *settings*' folders on a background thread until the watch is stopped."""
        if self._thread is not None:
            return
        self.used.emit("watch_start")
        self._settings = settings
        self._added = 0
        self._pending = []
        self._worker = WatchWorker(settings)
        self._worker.batch.connect(self._on_batch)
        self._worker.finished.connect(self._on_finished)
        self._thread = start_in_thread(self, self._worker)
        self.action.setText(_("Stop Watching"))
        self.status.emit(watch_status_text(settings, added=0))
        logger.info(
            "gui_watch_started",
            folders=[str(folder) for folder in settings.folders],
            interval=settings.interval,
            generate=settings.generate,
            save=settings.save,
        )

    def pacer_for(self, trigger: str) -> Pacer | None:
        """
        Return the watch's pacer for a batch it started, None for any other run.

        A watch shares one across its batches, as the CLI does, so a work window spans them and a
        stream of short batches still rests. A new plan, or a cancelled run (Stop cancels the pacer
        too), starts it afresh.
        """
        if trigger != TRIGGER_WATCH or self._settings is None:
            return None
        plan = self._pacing_plan()
        pacer = self._pacer
        if pacer is None or pacer.cancelled or pacer.plan != plan:
            self._pacer = build_pacer(plan)
        return self._pacer

    def _on_batch(self, paths: list[Path]) -> None:
        """Add the photos a poll found and queue them for generation when asked to."""
        settings = self._settings
        if settings is None or self._closing():
            # A batch delivered while the window is going away has nothing left to be added to.
            return
        fresh = self._add_photos(list(paths))
        if not fresh:
            return
        self._added += len(fresh)
        self.status.emit(watch_status_text(settings, added=self._added))
        if not settings.generate:
            return
        self._pending += [str(path) for path in fresh]
        if not self._busy():
            self._generate_queued()

    def _generate_queued(self) -> None:
        """Ask for the photos the watch queued to be generated, skipping any already dealt with."""
        if self._busy():
            # Draining the queue now would lose it: the window refuses to start a run mid-run.
            return
        queued, self._pending = self._pending, []
        targets = [
            item
            for key in queued
            if (item := self._photo(key)) is not None and item.status == PENDING
        ]
        if targets:
            self.generate_requested.emit(targets)

    def take_queued(self) -> None:
        """Take whatever the watch picked up while the last run was busy."""
        if self._settings is None or self._busy() or not self._pending:
            return
        self._generate_queued()

    def _on_finished(self) -> None:
        """Clean up after a watch that ended on its own (an error, or a stop already asked for)."""
        if self._closing():
            return
        self.stop()

    def stop(self) -> None:
        """End the watch and put the menu action back to offering a new one."""
        if self._settings is None and self._thread is None:
            # Already stopped: the worker's own finished signal arrives just after a Stop click,
            # and re-running this would overwrite whatever the status bar says by then.
            return
        self.used.emit("watch_stop")
        if self._worker is not None:
            self._worker.stop()
        self.teardown()
        self._settings = None
        self._pacer = None
        self._pending = []
        if not self._closing():
            self.action.setText(_("Watch Folder..."))
            self.status.emit(_("Stopped watching."))

    def stop_worker(self) -> None:
        """Ask the worker to end at its next poll without waiting for it (the window closes)."""
        if self._worker is not None:
            self._worker.stop()

    def teardown(self) -> None:
        """
        Stop waiting on the watch thread (detaching it after a timeout) and release its worker.

        The worker is cut loose first, as the scan's and harmonization's are: a detached one can
        still deliver a batch, and its finished signal would then stop whatever watch runs by then.
        """
        if self._worker is not None:
            self._worker.disconnect(self)
            self._worker.deleteLater()
        if self._thread is not None:
            # A poll of a huge folder, or of a network share that is not answering, can outlast
            # the timeout. The worker only lists directories, so leaving it to finish costs
            # nothing, and detaching spares the caller (a Stop click, or closeEvent) the wait.
            release_thread(
                self._thread,
                timeout_ms=_WATCH_STOP_TIMEOUT_MS,
                detached_event="gui_watch_stop_timed_out",
            )
            self._thread = None
        self._worker = None
