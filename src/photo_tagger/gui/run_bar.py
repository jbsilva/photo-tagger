# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The bar under the panes: the status line, and the batch run with its progress and clock."""

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QToolButton,
    QWidget,
)

from photo_tagger.gui_state import progress_timing_text, tooltip
from photo_tagger.i18n import _


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from PySide6.QtGui import QAction


# How often the elapsed/remaining readout is repainted during a run. Half a second reads as a live
# clock without being busywork.
_TIMING_TICK_MS = 500


class RunBar(QWidget):
    """
    Carry the window's status line and the batch workflow: generate the checked photos, then save.

    While a run is in flight the bar shows a progress bar and a clock beside it: the elapsed time,
    and once a photo has finished, an estimate of what is left. A long save used to show nothing at
    all, which looks like the app hung. The buttons ask the window for the work through signals;
    the window turns them on and off as runs start and end.
    """

    generate_requested = Signal(bool)  # use the cache
    save_requested = Signal()
    retry_requested = Signal()
    cancel_requested = Signal()

    def __init__(
        self,
        parent: QWidget,
        *,
        save_options_menu: QMenu,
        autosave_action: QAction,
    ) -> None:
        """Build the bar; the menu and the toggle are the window's own, shared with the pane's."""
        super().__init__(parent)
        # The window writes this line for everything it has to say, not only for runs.
        self.status = QLabel(_("Drag photos or folders here to begin."))
        self.status.setObjectName("status")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setFormat("%v / %m")
        self._progress.setVisible(False)
        self._timing = QLabel()
        self._timing.setObjectName("timing")
        self._timing.setVisible(False)
        # When the run in flight started, for the elapsed/remaining readout. None while idle.
        self._started: float | None = None
        # Repaints the readout between photos so the clock keeps moving during a slow one.
        self._timer = QTimer(self)
        self._timer.setInterval(_TIMING_TICK_MS)
        self._timer.timeout.connect(self._update_timing)
        self._build_buttons(save_options_menu, autosave_action)

        row = QHBoxLayout(self)
        # Flush with the window's own margins, as the row was before it had a widget of its own.
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.status, stretch=1)
        row.addWidget(self._timing)
        row.addWidget(self._progress)
        row.addWidget(self.retry_button)
        row.addWidget(self.cancel_button)
        row.addWidget(self.generate_button)
        row.addWidget(self.save_button)

    def _build_buttons(self, save_options_menu: QMenu, autosave_action: QAction) -> None:
        """Build Retry, Cancel, and the two split buttons that generate and save the batch."""
        self.retry_button = QPushButton(_("Retry Failed"))
        self.retry_button.setToolTip(
            tooltip(
                "Re-run the model on every photo that failed to generate. Enabled once a photo "
                "has actually failed.",
            ),
        )
        self.retry_button.setEnabled(False)
        self.retry_button.clicked.connect(self.retry_requested)
        self.cancel_button = QPushButton(_("Cancel"))
        self.cancel_button.setToolTip(
            tooltip(
                "Stop the run in progress, generating or saving. The photo currently in flight "
                "finishes; the rest are left untouched so you can resume them later.",
            ),
        )
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)

        # A split button: a click generates normally (cache included); the arrow offers the
        # one-time skip-cache run without changing the Settings toggle.
        self.generate_button = QToolButton()
        self.generate_button.setObjectName("primarysplit")
        self.generate_button.setText(_("Generate Selected"))
        self.generate_button.setToolTip(tooltip("Run the model on the checked photos."))
        self.generate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.generate_button.clicked.connect(
            lambda: self.generate_requested.emit(True),  # noqa: FBT003 - use the cache
        )
        # Kept on self: Qt's menu() accessor returns a transient wrapper shiboken may delete.
        self.generate_menu = QMenu(self)
        self.generate_menu.setToolTipsVisible(True)
        skip_all = self.generate_menu.addAction(
            _("Generate Selected (Skip Cache)"),
            lambda: self.generate_requested.emit(False),  # noqa: FBT003 - skip the cache
        )
        skip_all.setToolTip(
            tooltip("One-time: call the model even for photos with cached results."),
        )
        self.generate_menu.addSeparator()
        self.generate_menu.addAction(autosave_action)
        self.generate_menu.addMenu(save_options_menu)
        self.generate_button.setMenu(self.generate_menu)

        self.save_button = QToolButton()
        self.save_button.setObjectName("primarysplit")
        self.save_button.setText(_("Save Selected"))
        self.save_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.save_button.clicked.connect(self.save_requested)
        self.save_button.setMenu(save_options_menu)

    def start_clock(self, total: int) -> None:
        """Show the bar and start the clock for a run over *total* photos."""
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._started = time.monotonic()
        self._timing.setVisible(True)
        self._update_timing()
        self._timer.start()

    def stop_clock(self) -> None:
        """Hide the bar and the clock; the run is over."""
        self._timer.stop()
        self._started = None
        self._progress.setVisible(False)
        self._timing.setVisible(False)
        self._timing.clear()

    def advance(self) -> None:
        """Tick the progress bar for one finished (or failed) photo."""
        if self._started is None:
            return
        self._progress.setValue(self._progress.value() + 1)
        self._update_timing()

    def _update_timing(self) -> None:
        """
        Repaint the elapsed/remaining readout from the progress bar's own counters.

        Driven by a timer as well as by each finished photo, so the elapsed time keeps moving while
        a single slow photo is in flight. That movement is the point: it is what tells the user the
        program is working rather than wedged.
        """
        if self._started is None:
            return
        self._timing.setText(
            progress_timing_text(
                self._progress.value(),
                self._progress.maximum(),
                time.monotonic() - self._started,
            ),
        )
