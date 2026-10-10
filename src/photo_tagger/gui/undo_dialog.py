# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Tools > Undo Writes dialog: put back what a recorded run wrote."""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import CLOSE_LABEL, PATH_ROLE
from photo_tagger.gui.widgets import busy_cursor, readonly_box
from photo_tagger.gui_state import (
    journal_label,
    reverted_images,
    tooltip,
    undo_details_text,
    undo_summary,
)
from photo_tagger.i18n import _, ngettext
from photo_tagger.undo import UndoError, list_journals, read_journal, undo_run


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable


class UndoDialog(QDialog):
    """
    List every recorded run and put back what the chosen one wrote.

    Runs from this window and from the command line share one journal folder, so both show up.
    Undoing deletes the sidecars a run created and restores the files it overwrote from their
    ExifTool backup. The dialog reports a status line with :attr:`status`, the feature used with
    :attr:`used`, and the photos it put back with :attr:`undone`.
    """

    status = Signal(str)
    used = Signal(str)
    undone = Signal(object)  # list[str]: the photos whose writes were put back

    def __init__(self, parent: QWidget, *, busy: Callable[[], bool]) -> None:
        """Build the dialog; *busy* says whether a run is writing, which an undo must wait out."""
        super().__init__(parent)
        self._busy = busy
        self.setWindowTitle(_("Undo writes"))
        self.setMinimumWidth(660)
        box = QVBoxLayout(self)
        intro = QLabel(
            _(
                "Every recorded run, newest first, from this window and from the command line. "
                "Undoing deletes the sidecars a run created and restores the files it overwrote "
                "from their ExifTool backup.",
            ),
        )
        intro.setWordWrap(True)
        box.addWidget(intro)

        self._journals = QListWidget()
        self._journals.setMinimumHeight(150)
        box.addWidget(self._journals, stretch=1)

        self._force = QCheckBox(_("Also revert files changed since the run"))
        self._force.setToolTip(
            tooltip(
                "Off by default: a file that changed since the run was edited afterwards, and "
                "that edit is not this run's to undo.",
            ),
        )
        box.addWidget(self._force)

        self._details = readonly_box(120)
        self._details.setPlaceholderText(_("What happened to each file appears here."))
        box.addWidget(self._details)

        preview = QPushButton(_("Preview"))
        preview.setToolTip(
            tooltip("Report what undoing would put back, without touching anything."),
        )
        preview.clicked.connect(lambda: self._run(dry_run=True))
        self._undo_button = QPushButton(_("Undo"))
        self._undo_button.setObjectName("primary")
        self._undo_button.clicked.connect(lambda: self._run(dry_run=False))
        close = QPushButton(_(CLOSE_LABEL))
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(preview)
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(self._undo_button)
        box.addLayout(buttons)

    def show_runs(self) -> None:
        """Open the dialog on a freshly-read list of recorded runs."""
        self.refresh()
        self.exec()

    def refresh(self) -> None:
        """List the recorded runs, newest first, with how many files each one wrote."""
        self._journals.clear()
        for path in list_journals():
            try:
                entries = len(read_journal(path))
            except UndoError as exc:  # pragma: no cover - one unreadable journal, not the list
                logger.warning("gui_journal_unreadable", file=str(path), error=str(exc))
                continue
            entry = QListWidgetItem(journal_label(path, entries))
            entry.setData(PATH_ROLE, str(path))
            self._journals.addItem(entry)
        listed = self._journals.count()
        if listed:
            self._journals.setCurrentRow(0)
        else:
            self._details.setPlainText(_("No recorded runs to undo."))
        self._undo_button.setEnabled(bool(listed))

    def _run(self, *, dry_run: bool) -> None:
        """Put back (or preview putting back) everything the selected run wrote."""
        self.used.emit("undo_preview" if dry_run else "undo")
        if not dry_run and self._busy():
            # A save in flight is writing the very files an undo would be reverting.
            self._details.setPlainText(_("Wait for the run in progress to finish first."))
            return
        entry = self._journals.currentItem()
        if entry is None:
            self._details.setPlainText(_("Pick a run first."))
            return
        try:
            records = read_journal(Path(entry.data(PATH_ROLE)))
        except UndoError as exc:
            QMessageBox.warning(self, _("Could not read that run"), str(exc))
            return
        if not records:
            self._details.setPlainText(_("That run recorded no writes."))
            return
        if not dry_run and not self._confirm(len(records)):
            return
        with busy_cursor():
            results = undo_run(records, force=self._force.isChecked(), dry_run=dry_run)
        self._details.setPlainText(undo_details_text(results, dry_run=dry_run))
        summary = undo_summary(results)
        self.status.emit(_("Preview: {summary}").format(summary=summary) if dry_run else summary)
        if not dry_run:
            self.undone.emit(reverted_images(records, results))
            self.refresh()

    def _confirm(self, count: int) -> bool:
        """Ask before reverting: this rewrites files, and the run may not be the one in mind."""
        reply = QMessageBox.question(
            self,
            _("Undo this run?"),
            ngettext(
                "Put back {n} file? Sidecars the run created are deleted, and files it "
                "overwrote are restored from their backup.",
                "Put back {n} files? Sidecars the run created are deleted, and files it "
                "overwrote are restored from their backup.",
                count,
            ).format(n=count),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes
