# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""Small widgets and widget helpers the main window and its dialogs share."""

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, Qt, Signal

# QMouseEvent only annotates ClickableLabel's Qt override, but it is imported at runtime on
# purpose, for the reason given above MainWindow.closeEvent: Qt itself calls that override.
from PySide6.QtGui import QMouseEvent  # noqa: TC002 - runtime on purpose, see the note above.
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTreeWidgetItem,
)

from photo_tagger.gui.common import (
    CLOSE_LABEL,
    COL_NAME,
    COL_STATUS,
    COL_TAGGED,
    COL_TYPE,
    IS_DIR_ROLE,
    STATUS_RANK_ROLE,
)
from photo_tagger.i18n import _


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable


def readonly_box(min_height: int) -> QPlainTextEdit:
    """Return a read-only, scrollable text box for displaying existing metadata."""
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setMinimumHeight(min_height)
    return box


def hint_label(text: str) -> QLabel:
    """Return a dimmed note that wraps, the kind a dialog puts under its settings."""
    label = QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label


def close_row(dialog: QDialog) -> QHBoxLayout:
    """Return the row a settings dialog ends with: one default Close button, on the right."""
    close = QPushButton(_(CLOSE_LABEL))
    close.setDefault(True)
    close.clicked.connect(dialog.accept)
    buttons = QHBoxLayout()
    buttons.addStretch(1)
    buttons.addWidget(close)
    return buttons


def fit_text_height(box: QPlainTextEdit, *, min_h: int = 44, max_h: int = 140) -> None:
    """
    Size *box* to its content, within bounds.

    Short text keeps the box short so the pane's space goes to fields that need it; long text grows
    the box up to *max_h* and scrolls past that. QPlainTextEdit reports its document height in line
    counts, hence the line-spacing multiplication.
    """
    lines = max(1, int(box.document().size().height()))
    height = lines * box.fontMetrics().lineSpacing() + 14
    box.setFixedHeight(max(min_h, min(max_h, height)))


def fit_combo(combo: QComboBox) -> None:
    """
    Size a combo box to its widest entry so the drop-down arrow never clips the label.

    The grid toolbar's labels vary in length (and more so once translated), and the default policy
    sizes to the current item only, which clipped the wider entries behind the arrow. Sizing to the
    widest entry keeps every option readable in any language.
    """
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setMinimumContentsLength(max(len(combo.itemText(i)) for i in range(combo.count())))


def _status_sort_key(item: QTreeWidgetItem) -> tuple[int, str]:
    """Sort key for the Status column: lifecycle rank first, then name as a stable tiebreak."""
    rank = item.data(COL_STATUS, STATUS_RANK_ROLE)
    return (int(rank) if rank is not None else 0, item.text(COL_NAME).casefold())


class SortableTreeItem(QTreeWidgetItem):  # NOSONAR S8500 - Qt sorts items via __lt__ only
    """
    A tree row that sorts sensibly when the user clicks a column header.

    Folders stay grouped above files whichever way the sort runs; the Photos column sorts by name
    (case-insensitive) and the Status column by lifecycle rank rather than the raw label.

    Only ``__lt__`` is overridden: Qt drives item sorting entirely through it, and the comparison is
    context-dependent (it follows the active sort column and direction), so a real total ordering or
    ``functools.total_ordering`` would be wrong here. Hence the S8500 suppression on the class.
    """

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        """Order this row before *other* under the sort column and direction now in effect."""
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        self_dir = bool(self.data(0, IS_DIR_ROLE))
        if self_dir != bool(other.data(0, IS_DIR_ROLE)):
            # Keep folders above files in both directions: Qt reverses the result for a
            # descending sort, so invert there to cancel that out.
            ascending = (
                tree is None or tree.header().sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
            )
            return self_dir if ascending else not self_dir
        if column == COL_STATUS:
            return _status_sort_key(self) < _status_sort_key(other)
        if column in (COL_TYPE, COL_TAGGED):
            return self.text(column).casefold() < other.text(column).casefold()
        return self.text(COL_NAME).casefold() < other.text(COL_NAME).casefold()


class ClickableLabel(QLabel):
    """A label that reports left clicks, so the detail pane's preview can open the full viewer."""

    clicked = Signal()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override.
        """Treat a left click as a request to open whatever the label is showing."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class InputWatcher(QObject):
    """Event filter that reports key presses, clicks, and scrolling; it never consumes them."""

    _INPUT = frozenset({QEvent.Type.KeyPress, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel})

    def __init__(self, touch: Callable[[], None], parent: QObject) -> None:
        """Call *touch* on every input event the watched widgets receive."""
        super().__init__(parent)
        self._touch = touch

    def eventFilter(self, _watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override.
        """Note the input and let the event through untouched."""
        if event.type() in self._INPUT:
            self._touch()
        return False


def check_state(selected: bool) -> Qt.CheckState:  # noqa: FBT001 - tiny private bool mapper.
    """Map a selected flag to a Qt check state."""
    return Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked


def selection_modifiers_active() -> bool:
    """Report whether a multi-select modifier (Shift or Ctrl/Cmd) is held right now."""
    modifiers = QApplication.keyboardModifiers()
    return bool(
        modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier),
    )
