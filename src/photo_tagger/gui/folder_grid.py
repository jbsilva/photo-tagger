# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The folder grid: one folder's photos as thumbnails, filtered and sorted."""

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import PATH_ROLE
from photo_tagger.gui.threads import join_thread, release_thread, start_in_thread
from photo_tagger.gui.widgets import check_state, fit_combo, selection_modifiers_active
from photo_tagger.gui.workers import ThumbnailWorker
from photo_tagger.gui_state import (
    BADGE_COPYRIGHT,
    BADGE_FAILED,
    BADGE_METADATA,
    BADGE_MIXED_SCRIPT,
    BADGE_SAVED,
    BADGE_SIDECAR,
    BADGE_UNSAVED,
    FILTER_ALL,
    FILTER_CHOICES,
    SORT_CHOICES,
    SORT_NAME,
    filter_photos,
    paths_under,
    sort_photos,
    thumb_badges,
    thumb_tooltip,
    tooltip,
)
from photo_tagger.i18n import _, ngettext


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from pathlib import Path

    from PySide6.QtCore import QThread

    from photo_tagger.gui_state import PhotoItem


_THUMB_SIZE = 160  # icon box in the grid
# How long stop_thumbs waits for the thumbnail job. It runs on every navigation, so this is the
# longest the window may sit still when the user clicks away from a folder that is still loading.
# One decode is uninterruptible, and a big RAW takes seconds, so the wait usually times out.
_THUMB_STOP_TIMEOUT_MS = 250

# Thumbnail badge rendering: fill color and glyph per badge name. The lifecycle badge (first
# three) draws top-right; the informational ones stack top-left.
_BADGE_STYLE = {
    BADGE_FAILED: ("#f85149", "✗"),  # red cross
    BADGE_SAVED: ("#3fb950", "✓"),  # green check
    BADGE_UNSAVED: ("#6366f1", "•"),  # indigo dot: generated, not saved yet
    BADGE_METADATA: ("#8a8a8a", "M"),  # file already carries metadata
    BADGE_COPYRIGHT: ("#8957e5", "C"),  # purple: the file carries a copyright notice
    BADGE_SIDECAR: ("#0e7490", "S"),  # an XMP sidecar exists
    BADGE_MIXED_SCRIPT: ("#d29922", "!"),  # amber: the text mixes writing systems
}
_LIFECYCLE_BADGES = frozenset({BADGE_FAILED, BADGE_SAVED, BADGE_UNSAVED})


class FolderGrid(QWidget):
    """
    Show one folder's photos as a thumbnail grid, with its own filter and sort order.

    The grid shares the window's photo dict, reads each photo from it, and writes only a photo's
    checkbox. It has no column headers to click, so a Show filter narrows which photos appear and a
    Sort by picker orders what remains. The choices are session state, never written to disk: they
    carry across folders and reset on relaunch.

    Thumbnails are decoded on a thread of their own and kept for the session. The grid tells the
    window through signals when a checkbox was toggled, a thumbnail was clicked, a right-click wants
    a menu, or the filter or the order changed.
    """

    check_toggled = Signal(str)  # key of the photo whose checkbox the user changed
    photo_activated = Signal(str)  # key of the photo whose thumbnail the user clicked
    # Where to open it (global), the photo clicked, and the photos selected with it.
    context_menu_requested = Signal(QPoint, str, list)
    view_changed = Signal()  # the filter or the order changed
    status = Signal(str)
    used = Signal(str)

    def __init__(self, parent: QWidget, items: dict[str, PhotoItem]) -> None:
        """Build the toolbar and the grid over *items*, the window's own photo dict."""
        super().__init__(parent)
        self._items = items
        self._cells: dict[str, QListWidgetItem] = {}
        self._folder: Path | None = None
        self._sort = SORT_NAME
        self._sort_desc = False
        self._filter = FILTER_ALL
        # Held while the grid changes its own cells: setIcon and setToolTip emit itemChanged just
        # like setCheckState does, and none of those is the user toggling a checkbox.
        self._syncing = False
        # Set by a checkbox click, so the itemClicked of the same release does not open the photo.
        self._check_toggled = False
        # Keyed by path and kept for the session: small pixmaps, and a folder of them is cheap.
        self._thumbs: dict[str, QPixmap] = {}
        self._placeholder = _make_placeholder()
        self._thumb_thread: QThread | None = None
        self._thumb_worker: ThumbnailWorker | None = None

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._build_toolbar())
        grid = QListWidget()
        grid.setViewMode(QListView.ViewMode.IconMode)
        grid.setResizeMode(QListView.ResizeMode.Adjust)
        grid.setMovement(QListView.Movement.Static)
        grid.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        grid.setGridSize(QSize(_THUMB_SIZE + 24, _THUMB_SIZE + 40))
        grid.setSpacing(8)
        grid.setUniformItemSizes(True)
        grid.setWordWrap(True)
        grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        grid.itemClicked.connect(self._on_thumb_activated)
        grid.itemChanged.connect(self._on_item_changed)
        # Thumbnails answer to the same right-click menu as their row in the tree.
        grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        grid.customContextMenuRequested.connect(self._on_context_menu)
        self._list = grid
        box.addWidget(grid, stretch=1)

    def _build_toolbar(self) -> QHBoxLayout:
        """
        Build the filter and sort controls, standing in for the tree's column headers.

        Each combo is sized to its widest entry so no label is clipped. Signals are wired only after
        the widgets are populated, so building them fires no spurious rebuild.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Show")))
        self._filter_combo = QComboBox()
        for label, criterion in FILTER_CHOICES:
            self._filter_combo.addItem(_(label), criterion)
        self._filter_combo.setToolTip(tooltip("Show only the photos in the chosen state."))
        fit_combo(self._filter_combo)
        row.addWidget(self._filter_combo)
        row.addSpacing(16)

        row.addWidget(QLabel(_("Sort by")))
        self._sort_combo = QComboBox()
        for label, criterion in SORT_CHOICES:
            self._sort_combo.addItem(_(label), criterion)
        self._sort_combo.setToolTip(
            tooltip("Order the thumbnails by name, file type, status, or existing metadata."),
        )
        fit_combo(self._sort_combo)
        row.addWidget(self._sort_combo)

        self._sort_dir = QToolButton()
        self._sort_dir.setObjectName("sortdir")
        self._sort_dir.setCheckable(True)
        self._sort_dir.setText("↑")
        self._sort_dir.setToolTip(tooltip("Ascending. Click to sort descending."))
        row.addWidget(self._sort_dir)
        row.addStretch(1)

        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self._sort_dir.toggled.connect(self._on_sort_dir_toggled)
        return row

    @property
    def folder(self) -> Path | None:
        """Return the folder the grid shows, or None when it shows none."""
        return self._folder

    @property
    def filter(self) -> str:
        """Return the Show filter in force, one of the FILTER_* criteria."""
        return self._filter

    @property
    def view(self) -> QListWidget:
        """Return the thumbnail list itself, for the window's input tracking."""
        return self._list

    def _on_filter_changed(self) -> None:
        """Re-filter the grid when the Show choice changes."""
        self.used.emit("grid_filter")
        self._filter = self._filter_combo.currentData()
        self.view_changed.emit()

    def _on_sort_changed(self) -> None:
        """Re-sort the grid when the sort field changes."""
        self.used.emit("grid_sort")
        self._sort = self._sort_combo.currentData()
        self.view_changed.emit()

    def _on_sort_dir_toggled(self, descending: bool) -> None:  # noqa: FBT001 - Qt toggled slot
        """Flip the grid between ascending and descending, updating the toggle's arrow."""
        self.used.emit("grid_sort")
        self._sort_desc = descending
        self._sort_dir.setText("↓" if descending else "↑")
        self._sort_dir.setToolTip(
            tooltip("Descending. Click to sort ascending.")
            if descending
            else _("Ascending. Click to sort descending."),
        )
        self.view_changed.emit()

    def show_folder(self, folder: Path) -> None:
        """Fill the grid with *folder*'s photos, filtered and sorted, reusing cached thumbnails."""
        if folder != self._folder:  # a refresh of the grid on show is not a new visit
            self.used.emit("grid")
        self.stop_thumbs()
        self._folder = folder
        self._list.clear()
        self._cells = {}
        paths = [item.path for item in self._items.values()]
        under = [self._items[str(path)] for path in paths_under(paths, folder)]
        shown = sort_photos(
            filter_photos(under, self._filter),
            self._sort,
            descending=self._sort_desc,
        )
        pending: list[Path] = []
        self._syncing = True
        for item in shown:
            key = str(item.path)
            cell = QListWidgetItem(item.path.name)
            cell.setData(PATH_ROLE, key)
            cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._list.addItem(cell)
            self._cells[key] = cell
            self._paint(item, cell)
            if key not in self._thumbs:
                pending.append(item.path)
        self._syncing = False
        self.status.emit(_status_text(folder, shown=len(shown), total=len(under)))
        if pending:
            self._start_thumbs(pending)

    def close_folder(self) -> None:
        """Forget the folder shown, once nothing of it is left in the list."""
        self._folder = None

    def clear(self) -> None:
        """Empty the grid and every thumbnail with it, for a list that was cleared."""
        self.stop_thumbs()
        self._thumbs.clear()
        self._list.clear()
        self._cells = {}
        self._folder = None

    def forget(self, key: str) -> None:
        """Drop the thumbnail of a photo that left the list."""
        self._thumbs.pop(key, None)

    def refresh(self, item: PhotoItem) -> None:
        """Repaint *item*'s thumbnail, when the grid shows it, after its state changed."""
        cell = self._cells.get(str(item.path))
        if cell is not None:
            self._paint(item, cell)

    def sync_checks(self) -> None:
        """Repaint every thumbnail checkbox from the photos, after they were checked elsewhere."""
        was_syncing = self._syncing
        self._syncing = True
        try:
            for key, cell in self._cells.items():
                item = self._items.get(key)
                if item is not None:
                    cell.setCheckState(check_state(item.selected))
        finally:
            self._syncing = was_syncing

    def _paint(self, item: PhotoItem, cell: QListWidgetItem) -> None:
        """Refresh a grid thumbnail: image (or placeholder), state badges, and checkbox."""
        # Hold _syncing across the whole refresh: an unguarded emission from any of these would
        # flip _check_toggled, which then swallows the user's next real click on a thumbnail.
        was_syncing = self._syncing
        self._syncing = True
        try:
            cell.setCheckState(check_state(item.selected))
            key = str(item.path)
            base = self._thumbs.get(key, self._placeholder)
            badges = thumb_badges(item, has_sidecar=item.path.with_suffix(".xmp").exists())
            cell.setIcon(QIcon(_badged_pixmap(base, badges)))
            cell.setToolTip(thumb_tooltip(item, badges))
        finally:
            self._syncing = was_syncing

    def _on_item_changed(self, cell: QListWidgetItem) -> None:
        """Mirror a thumbnail checkbox change onto its photo, and tell the window about it."""
        if self._syncing:
            return
        key = cell.data(PATH_ROLE)
        item = self._items.get(key)
        if item is None:
            return
        checked = cell.checkState() == Qt.CheckState.Checked
        if checked == item.selected:
            # Icon and tooltip updates also emit itemChanged; only a real check-state change
            # counts as a toggle.
            return
        # Remember that this was a checkbox click so the itemClicked that follows the same
        # mouse release does not also open the photo. The itemClicked (when there is one)
        # arrives within the same event dispatch, so clear the flag once the loop settles;
        # otherwise a keyboard (space key) toggle would leave it stale and swallow the user's
        # next real click on a thumbnail.
        self._check_toggled = True
        QTimer.singleShot(0, self._clear_check_toggled)
        item.selected = checked
        self.check_toggled.emit(key)

    def _clear_check_toggled(self) -> None:
        """Reset the checkbox-click marker once the current event dispatch has finished."""
        self._check_toggled = False

    def _on_thumb_activated(self, cell: QListWidgetItem) -> None:
        """Open the clicked photo, unless the click was on its checkbox or builds a selection."""
        if self._check_toggled:
            # This click landed on the checkbox: the toggle already happened, and jumping to
            # the detail page would yank the user out of the grid they are working in.
            self._check_toggled = False
            return
        if selection_modifiers_active():
            # Shift/Cmd clicks are building a multi-selection; navigating away would kill it.
            return
        self.photo_activated.emit(cell.data(PATH_ROLE))

    def _on_context_menu(self, pos: QPoint) -> None:
        """Ask the window for the right-click menu of the thumbnail under the cursor."""
        cell = self._list.itemAt(pos)
        if cell is None:
            return
        selected = [chosen.data(PATH_ROLE) for chosen in self._list.selectedItems()]
        self.context_menu_requested.emit(
            self._list.viewport().mapToGlobal(pos),
            cell.data(PATH_ROLE),
            selected,
        )

    def _on_thumb_ready(self, path: str, data: bytes) -> None:
        """Keep a decoded thumbnail, and paint it if the grid still shows its photo."""
        item = self._items.get(path)
        if item is None:
            # Removed from the list while it was decoding, which a detached job makes likelier.
            # Caching it would strand a pixmap that nothing prunes.
            return
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        self._thumbs[path] = pixmap
        cell = self._cells.get(path)
        if cell is not None:
            self._paint(item, cell)

    def _start_thumbs(self, paths: list[Path]) -> None:
        """Decode the thumbnails of *paths* on a background thread."""
        self._thumb_worker = ThumbnailWorker(paths)
        self._thumb_worker.ready.connect(self._on_thumb_ready)
        # Stop the thread's event loop once every thumbnail is decoded; without this the idle
        # thread would keep running until the next navigation or the window closed.
        self._thumb_thread = start_in_thread(self, self._thumb_worker, quit_when_done=True)

    def stop_thumbs(self, *, blocking: bool = False) -> None:
        """
        Ask the thumbnail job to stop, waiting only briefly for it.

        The worker checks the stop flag between photos, so a decode already under way has to finish
        first: seconds, for a big RAW. Waiting that out froze the window on every click that leaves
        a loading folder, and with the Untagged filter on, every metadata scan batch re-enters here.
        Detach instead, the way the metadata scan does. A detached job only feeds _on_thumb_ready,
        which keys everything by path, so a late thumbnail is still the right thumbnail.

        The window's closeEvent passes ``blocking``: the grid owns the QThread, so leaving one
        running past the teardown would destroy it mid-run.
        """
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
            self._thumb_worker.deleteLater()
        if self._thumb_thread is not None:
            if blocking:
                join_thread(self._thumb_thread)
            else:
                release_thread(
                    self._thumb_thread,
                    timeout_ms=_THUMB_STOP_TIMEOUT_MS,
                    detached_event="gui_thumbnail_stop_detached",
                    level="DEBUG",
                )
            self._thumb_thread = None
        self._thumb_worker = None


def _status_text(folder: Path, *, shown: int, total: int) -> str:
    """Describe the grid's contents, noting when a filter is hiding some of the folder."""
    name = folder.name or folder
    if shown == total:
        return ngettext("{n} photo in {folder}.", "{n} photos in {folder}.", total).format(
            n=total,
            folder=name,
        )
    return _("Showing {shown} of {total} in {folder}.").format(
        shown=shown,
        total=total,
        folder=name,
    )


def _make_placeholder() -> QPixmap:
    """Build a neutral grey tile shown in the grid until a thumbnail loads."""
    pixmap = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
    pixmap.fill(QColor(50, 50, 56))
    return pixmap


def _badged_pixmap(base: QPixmap, badges: list[str]) -> QPixmap:
    """Overlay the badge dots for *badges* onto a copy of *base*."""
    if not badges or base.isNull():
        return base
    pixmap = QPixmap(base)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    radius = max(12, pixmap.width() // 8)
    margin = max(3, radius // 4)
    left = margin
    font = painter.font()
    font.setPixelSize(int(radius * 0.7))
    font.setBold(True)
    painter.setFont(font)
    for name in badges:
        color, glyph = _BADGE_STYLE[name]
        x = pixmap.width() - radius - margin if name in _LIFECYCLE_BADGES else left
        if name not in _LIFECYCLE_BADGES:
            left += radius + margin
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(x, margin, radius, radius)
        painter.setPen(QColor("white"))
        painter.drawText(QRect(x, margin, radius, radius), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return pixmap
