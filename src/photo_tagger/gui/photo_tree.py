# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The photo list: checkable rows, nested under folder rows or flat, found through an index."""

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTreeWidget, QTreeWidgetItem, QWidget

from photo_tagger.gui.common import (
    COL_NAME,
    COL_STATUS,
    COL_TAGGED,
    COL_TYPE,
    DIR_MARK,
    IS_DIR_ROLE,
    PATH_ROLE,
    STATUS_RANK_ROLE,
)
from photo_tagger.gui.widgets import SortableTreeItem, check_state
from photo_tagger.gui_state import (
    FAILED,
    PENDING,
    READY,
    SAVED,
    WORKING,
    build_tree,
    file_type_label,
    flat_labels,
    foreign_terms_note,
    mark_mixed_script,
    status_sort_rank,
    tagged_legend,
    tagged_summary,
    tagged_tooltip,
    tooltip,
    wrap_tooltip,
)
from photo_tagger.i18n import _, gettext_noop


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from pathlib import Path

    from photo_tagger.gui_state import FolderNode, Location, PhotoItem


# Status colors: failures pop red, saved photos settle green; other states use the palette.
_STATUS_COLOR = {FAILED: QColor("#f85149"), SAVED: QColor("#3fb950")}

# Short status word shown in the tree's second column.
_STATUS_LABEL = {
    PENDING: "",
    WORKING: gettext_noop("working..."),
    READY: gettext_noop("ready"),
    SAVED: gettext_noop("saved ✓"),
    FAILED: gettext_noop("failed ✗"),
}


class PhotoTree(QTreeWidget):
    """
    List the photos as checkable rows, grouped under folder rows or flat, one sortable table.

    Every row is found through an index, never by walking the tree. The only widget-side walker,
    QTreeWidgetItemIterator, is never destroyed by PySide, so each walk leaves an iterator
    registered with QTreeModel pointing at whatever row it stopped on. The next teardown frees that
    row and the removal after it dereferences the dangling pointer. An index also turns the per-
    photo status refresh from a full walk into a dict hit.

    The tree shares the window's photo dict: it reads each photo from it, and a checkbox writes the
    photo's ``selected`` flag. The window guards against its own programmatic changes, which fire
    the same signals a click does.
    """

    def __init__(self, parent: QWidget, items: dict[str, PhotoItem]) -> None:
        """Build the columns over *items*, the window's own photo dict."""
        super().__init__(parent)
        self._items = items
        # rebuild() owns both dicts and drops them before it empties the tree, so no row is
        # reachable from Python while Qt is tearing it down.
        self._folder_rows: dict[str, QTreeWidgetItem] = {}
        self._leaf_rows: dict[str, QTreeWidgetItem] = {}
        self.setHeaderLabels([_("Photos"), _("Type"), _("Status"), _("Tagged")])
        # Extended: shift+click selects a range, Cmd/Ctrl+click adds single rows, and
        # shift+arrows grow the selection from the keyboard.
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.header()
        header.setStretchLastSection(False)
        for column, width in (
            (COL_NAME, 250),
            (COL_TYPE, 64),
            (COL_STATUS, 80),
            (COL_TAGGED, 56),
        ):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.setColumnWidth(column, width)
        # Click a header to sort by any column; folders stay grouped above files.
        self.setSortingEnabled(True)
        self.sortByColumn(COL_NAME, Qt.SortOrder.AscendingOrder)
        header.setToolTip(
            tooltip(
                "Click a column header to sort. Type: file extension, +xmp when a sidecar "
                "exists.\nTagged: metadata already on the file ({legend}).",
                legend=tagged_legend(),
            ),
        )
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def leaf_for(self, path: Path) -> QTreeWidgetItem | None:
        """Return the row of the photo at *path*, or None when the list does not draw it."""
        return self._leaf_rows.get(str(path))

    def row_for(self, location: Location) -> QTreeWidgetItem | None:
        """Return the row *location* stands for, or None when it has none."""
        rows = self._folder_rows if location.is_dir else self._leaf_rows
        return rows.get(str(location.path))

    def has_folder_row(self, folder: Path) -> bool:
        """Report whether *folder* has a row of its own (the nested list collapses some away)."""
        return str(folder) in self._folder_rows

    def shown_count(self) -> int:
        """Return how many photos the last rebuild drew."""
        return len(self._leaf_rows)

    def empty(self) -> None:
        """
        Drop the row index, then take the rows out one at a time.

        Deliberately not ``QTreeWidget.clear()``. PySide's clear() hands each top-level row back to
        Python and deletes only the ones nothing references; C++ frees the rest after detaching them
        from the model, so it never emits the removals. Any QTreeWidgetItemIterator still registered
        with QTreeModel is left pointing at freed memory, and the next removal reads it. A context
        menu holding the row the user right-clicked is enough to arm that.

        takeTopLevelItem always goes through QTreeModel::beginRemoveItems, which tells every
        registered iterator what is going away. The index goes first so no row is reachable from
        Python while Qt is tearing it down.
        """
        self._folder_rows.clear()
        self._leaf_rows.clear()
        while self.topLevelItemCount():
            self.takeTopLevelItem(0)

    def rebuild(self, paths: list[Path], *, flat: bool) -> None:
        """
        Draw *paths* afresh, nested under their folders or, with *flat*, at one level.

        Built with sorting off so rows do not shuffle on every insert; turning it back on at the end
        re-applies whatever column and direction the header is set to. The caller holds its guard,
        since emptying the tree emits a change of current row like a click does.
        """
        self.setSortingEnabled(False)
        self.empty()
        if flat:
            self._add_flat_rows(paths)
        else:
            for node in build_tree(paths):
                self._add_folder_node(self, node)
        self.setSortingEnabled(True)

    def _add_flat_rows(self, paths: list[Path]) -> None:
        """Add every photo as a top-level row, so a column sort runs across the whole list."""
        labels = flat_labels(paths)
        for path in paths:
            leaf = self._add_leaf(self, path, labels[path])
            # The label is relative to the folder the photos share, so the full path is worth
            # a tooltip here in a way it is not under a folder row that already names it.
            leaf.setToolTip(COL_NAME, str(path))

    def _add_folder_node(self, parent: object, node: FolderNode) -> None:
        """Add one folder's row under *parent*, its subfolders and photos beneath it."""
        folder_item = SortableTreeItem(parent, [node.label, ""])
        folder_item.setData(0, PATH_ROLE, str(node.path))
        folder_item.setData(0, IS_DIR_ROLE, DIR_MARK)
        folder_item.setFlags(folder_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        folder_item.setExpanded(True)
        self._folder_rows[str(node.path)] = folder_item
        for sub in node.folders:
            self._add_folder_node(folder_item, sub)
        for path in node.files:
            self._add_leaf(folder_item, path, path.name)
        self._sync_folder_check(folder_item)

    def _add_leaf(self, parent: object, path: Path, label: str) -> SortableTreeItem:
        """Add one photo's row under *parent*, showing it as *label*."""
        item = self._items[str(path)]
        leaf = SortableTreeItem(parent, [label, file_type_label(path), "", ""])
        leaf.setData(0, PATH_ROLE, str(path))
        # Files leave IS_DIR_ROLE unset (None), which reads as "not a folder".
        leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        leaf.setCheckState(0, check_state(item.selected))
        self._leaf_rows[str(path)] = leaf
        _render_status_cells(leaf, item)
        return leaf

    def select_row(self, path: str, *, is_dir: bool) -> None:
        """Highlight the row for *path*, when there is one (the caller holds its guard)."""
        rows = self._folder_rows if is_dir else self._leaf_rows
        entry = rows.get(path)
        if entry is not None:
            self.setCurrentItem(entry)

    def apply_check(self, row: QTreeWidgetItem) -> None:
        """
        Follow a checkbox the user changed on *row* into the photos and the folder checkboxes.

        A folder's checkbox checks or unchecks everything under it; a photo's sets that photo.
        Either way, every folder above the row may have gone mixed.
        """
        if bool(row.data(0, IS_DIR_ROLE)):
            self._set_descendants_checked(row, row.checkState(0))
        else:
            path = row.data(0, PATH_ROLE)
            if path is not None:
                self._items[path].selected = row.checkState(0) == Qt.CheckState.Checked
        self.sync_ancestors(row)

    def set_checked(self, path: Path, *, checked: bool) -> QTreeWidgetItem | None:
        """Set the checkbox of the photo at *path*; return its row, or None when not drawn."""
        leaf = self.leaf_for(path)
        if leaf is not None:
            leaf.setCheckState(0, check_state(checked))
        return leaf

    def _set_descendants_checked(self, folder_item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        """Give every row under *folder_item* its *state*, and every photo there the same."""
        for index in range(folder_item.childCount()):
            child = folder_item.child(index)
            child.setCheckState(0, state)
            if bool(child.data(0, IS_DIR_ROLE)):
                self._set_descendants_checked(child, state)
            else:
                path = child.data(0, PATH_ROLE)
                if path is not None:
                    self._items[path].selected = state == Qt.CheckState.Checked

    def sync_ancestors(self, row: QTreeWidgetItem) -> None:
        """Re-derive the tristate of every folder above *row* (the caller holds its guard)."""
        parent = row.parent()
        while parent is not None:
            self._sync_folder_check(parent)
            parent = parent.parent()

    def _sync_folder_check(self, folder_item: QTreeWidgetItem) -> None:
        """Check a folder when all of it is checked, clear it when none is, mix it otherwise."""
        states = {folder_item.child(i).checkState(0) for i in range(folder_item.childCount())}
        if states == {Qt.CheckState.Checked}:
            folder_item.setCheckState(0, Qt.CheckState.Checked)
        elif states == {Qt.CheckState.Unchecked}:
            folder_item.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            folder_item.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def refresh(self, item: PhotoItem) -> None:
        """Repaint *item*'s Status and Tagged cells, when the list draws it."""
        leaf = self.leaf_for(item.path)
        if leaf is not None:
            _render_status_cells(leaf, item)

    def resort(self) -> None:
        """Re-apply the active sort so changed statuses settle when sorting by the Status column."""
        header = self.header()
        self.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())


def _render_status_cells(leaf: QTreeWidgetItem, item: PhotoItem) -> None:
    """Paint the Status and Tagged columns for *item*'s row."""
    label = _(_STATUS_LABEL[item.status]) if _STATUS_LABEL[item.status] else ""
    if item.status == READY and item.from_cache:
        label = _("ready (cached)")
    leaf.setText(COL_STATUS, mark_mixed_script(label, item.foreign_terms))
    leaf.setData(COL_STATUS, STATUS_RANK_ROLE, status_sort_rank(item.status))
    color = _STATUS_COLOR.get(item.status)
    leaf.setData(
        COL_STATUS,
        Qt.ItemDataRole.ForegroundRole,
        QBrush(color) if color is not None else None,
    )
    # Surface on hover what the cell only had room to mark: the failure reason, or the words
    # that made the text mixed. Model errors can be a paragraph long, hence the wrap.
    hover = item.error if item.status == FAILED else foreign_terms_note(item.foreign_terms)
    leaf.setToolTip(COL_STATUS, wrap_tooltip(hover) if hover else "")
    if item.known_fields is not None:
        leaf.setText(COL_TAGGED, tagged_summary(item.known_fields))
        leaf.setToolTip(COL_TAGGED, wrap_tooltip(tagged_tooltip(item.known_fields)))
