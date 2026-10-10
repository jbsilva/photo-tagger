# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The full-size photo viewer that the detail pane's preview opens."""

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QPoint, Qt

# QKeyEvent only annotates the keyPressEvent override below, but it is imported at runtime on
# purpose, for the reason given above MainWindow.closeEvent: Qt itself calls that override.
from PySide6.QtGui import QKeyEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import CLOSE_LABEL
from photo_tagger.gui_state import (
    anchored_scroll,
    clamp_zoom,
    fit_zoom,
    step_zoom,
    tooltip,
    zoom_label,
)
from photo_tagger.i18n import _


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from pathlib import Path


class ImageViewerDialog(QDialog):
    """
    One photo at full window size, with zoom and pan.

    The detail pane's preview shares its panel with the metadata fields, so it is necessarily small:
    too small to check whether a generated keyword is really in the frame. Clicking it opens this
    window, which starts at fit-to-window and zooms up to 8:1.

    Zoom is kept as a factor of the decoded image's own pixel size, and every zoom rescales the
    original pixmap rather than the last scaled one, so repeated zooming in and out does not
    accumulate resampling blur.
    """

    def __init__(self, parent: QWidget, path: Path, pixmap: QPixmap) -> None:
        """Build the viewer around an already-decoded *pixmap* of *path*."""
        super().__init__(parent)
        self.setWindowTitle(path.name)
        self._source = pixmap
        self._zoom = 1.0
        # While true, a window resize refits the photo. Any explicit zoom turns it off, so the
        # user's chosen magnification survives resizing (and entering full screen).
        self._fit_to_window = True
        self._pan_from: QPoint | None = None

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("viewer")
        # Not widgetResizable: the canvas is resized to the scaled pixmap so the scrollbars
        # describe the image. The alignment centers it while it is smaller than the viewport.
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._canvas)
        self._scroll.viewport().installEventFilter(self)
        self._scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        box.addWidget(self._scroll, stretch=1)
        box.addWidget(self._build_toolbar())
        self._install_shortcuts()

    def _build_toolbar(self) -> QWidget:
        """Build the bottom strip: zoom controls, the current percentage, and Close."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 10, 6)
        out = QPushButton(_("Zoom out"))
        out.clicked.connect(lambda: self._zoom_by(-1))
        row.addWidget(out)
        self._zoom_label = QLabel()
        self._zoom_label.setMinimumWidth(56)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._zoom_label)
        in_ = QPushButton(_("Zoom in"))
        in_.clicked.connect(lambda: self._zoom_by(1))
        row.addWidget(in_)
        fit = QPushButton(_("Fit"))
        fit.setToolTip(tooltip("Scale the photo to the window (Ctrl+0)."))
        fit.clicked.connect(self._fit)
        row.addWidget(fit)
        actual = QPushButton(_("100%"))
        actual.setToolTip(tooltip("Show the photo at one screen pixel per image pixel (Ctrl+1)."))
        actual.clicked.connect(lambda: self._set_zoom(1.0))
        row.addWidget(actual)
        row.addStretch(1)
        hint = QLabel(_("Scroll to pan · Ctrl+scroll or +/- to zoom · double-click to toggle fit"))
        hint.setObjectName("hint")
        row.addWidget(hint)
        row.addStretch(1)
        full = QPushButton(_("Full screen"))
        full.setToolTip(tooltip("Fill the whole screen, hiding the window frame (F)."))
        full.clicked.connect(self._toggle_full_screen)
        row.addWidget(full)
        close = QPushButton(_(CLOSE_LABEL))
        close.clicked.connect(self.accept)
        row.addWidget(close)
        return bar

    def _install_shortcuts(self) -> None:
        """
        Wire the keyboard: the standard zoom keys, plus fit, 1:1, and full screen.

        Ctrl+= and a bare +/- join Qt's standard zoom keys because the standard ones are Ctrl++ and
        Ctrl+-, which need a shifted key on many layouts.
        """
        # Every QShortcut is parented to the dialog, which owns it for as long as it is open.
        for key in (QKeySequence.StandardKey.ZoomIn, QKeySequence("Ctrl+="), QKeySequence("+")):
            QShortcut(key, self, lambda: self._zoom_by(1))
        for key in (QKeySequence.StandardKey.ZoomOut, QKeySequence("-")):
            QShortcut(key, self, lambda: self._zoom_by(-1))
        QShortcut(QKeySequence("Ctrl+0"), self, self._fit)
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self._set_zoom(1.0))
        for key in (QKeySequence("F"), QKeySequence("F11")):
            QShortcut(key, self, self._toggle_full_screen)

    # --- zoom ----------------------------------------------------------------------------------

    def showEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override.
        """Fit the photo the first time the window has a real size to fit it to."""
        super().showEvent(event)
        if self._fit_to_window:
            self._fit()

    def resizeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override.
        """Refit on resize, but only while the user has not picked a zoom of their own."""
        super().resizeEvent(event)
        if self._fit_to_window:
            self._fit()

    def _fit(self) -> None:
        """
        Scale the photo down to the viewport (never up: a small photo stays its own size).

        Measured against maximumViewportSize, not the viewport itself: while zoomed in there are
        scrollbars, and fitting to the space they leave would scale the photo to slightly less than
        the window it is about to have all of.
        """
        viewport = self._scroll.maximumViewportSize()
        zoom = fit_zoom(
            (self._source.width(), self._source.height()),
            (viewport.width(), viewport.height()),
        )
        self._apply_zoom(zoom)
        self._fit_to_window = True

    def _set_zoom(self, zoom: float) -> None:
        """Jump to an exact zoom factor, leaving fit-to-window behind."""
        self._fit_to_window = False
        self._apply_zoom(clamp_zoom(zoom))

    def _zoom_by(self, notches: int) -> None:
        """Zoom one or more notches in (positive) or out (negative) about the viewport center."""
        self._set_zoom(step_zoom(self._zoom, notches))

    def _apply_zoom(self, zoom: float) -> None:
        """Rescale the original pixmap to *zoom* and keep the viewport centered where it was."""
        if self._source.isNull():
            return
        factor = zoom / self._zoom if self._zoom else 1.0
        self._zoom = zoom
        scaled = self._source.scaled(
            max(1, round(self._source.width() * zoom)),
            max(1, round(self._source.height() * zoom)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._canvas.setPixmap(scaled)
        self._canvas.resize(scaled.size())
        self._zoom_label.setText(zoom_label(zoom))
        for scrollbar in (self._scroll.horizontalScrollBar(), self._scroll.verticalScrollBar()):
            scrollbar.setValue(
                anchored_scroll(scrollbar.value(), scrollbar.pageStep(), factor),
            )

    # --- mouse and keys ------------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override.
        """Let the viewport's own handler run only for what the viewer does not claim."""
        if watched is not self._scroll.viewport() or not self._handle_viewport(event):
            return super().eventFilter(watched, event)
        return True

    def _handle_viewport(self, event: QEvent) -> bool:
        """
        Turn drags on the image into panning and Ctrl+wheel (or a trackpad pinch) into zooming.

        Returns whether the event was consumed. A plain wheel is left alone so it still scrolls a
        zoomed-in photo, which is what a scroll area is for.
        """
        match event.type():
            case QEvent.Type.Wheel if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._zoom_by(1 if event.angleDelta().y() > 0 else -1)
            case QEvent.Type.MouseButtonPress if event.button() == Qt.MouseButton.LeftButton:
                self._pan_from = event.position().toPoint()
                self._scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            case QEvent.Type.MouseMove if self._pan_from is not None:
                self._pan(event.position().toPoint())
            case QEvent.Type.MouseButtonRelease if self._pan_from is not None:
                self._pan_from = None
                self._scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            case QEvent.Type.MouseButtonDblClick:
                self._toggle_fit()
            case _:
                return False
        return True

    def _pan(self, position: QPoint) -> None:
        """Drag the image under the cursor: scroll by however far the pointer moved."""
        if self._pan_from is None:
            return
        delta = position - self._pan_from
        self._pan_from = position
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        horizontal.setValue(horizontal.value() - delta.x())
        vertical.setValue(vertical.value() - delta.y())

    def _toggle_fit(self) -> None:
        """Double-click shortcut between fit-to-window and 1:1, whichever the view is not on."""
        if self._fit_to_window:
            self._set_zoom(1.0)
        else:
            self._fit()

    def _toggle_full_screen(self) -> None:
        """Swap between a normal window and a borderless one filling the screen."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override.
        """Let Escape leave full screen first, so it takes two presses to lose the viewer."""
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
        super().keyPressEvent(event)
