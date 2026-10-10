# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Tools > Watch Folder dialog: which folder to poll, how, and what to do with new photos."""

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.config import DEFAULT_INTERVAL_SECONDS, DEFAULT_SETTLE_SECONDS
from photo_tagger.gui.widgets import blurb, path_row
from photo_tagger.gui_state import WatchSettings, tooltip
from photo_tagger.i18n import _


class WatchDialog(QDialog):
    """
    Ask for the folder to watch and how to treat what lands in it.

    Only the question lives here: the watch itself runs in the WatchController, which hands each
    new photo to the window's list and its generate-then-save chain. The dialog is kept for the
    window's lifetime, so the folder typed last time is still there the next time it opens.
    """

    def __init__(self, parent: QWidget, *, recursive: bool) -> None:
        """Build the dialog; *recursive* seeds its subfolder toggle from the folder-scan option."""
        super().__init__(parent)
        self._extensions = ""
        self.setWindowTitle(_("Watch a folder"))
        self.setMinimumWidth(600)
        box = QVBoxLayout(self)
        intro = blurb(
            _(
                "Point this at the folder your card reader, tethered capture, or sync client "
                "fills. Photos already there are picked up first, then each new one as it lands.",
            ),
        )
        box.addWidget(intro)
        box.addLayout(self._build_form(recursive=recursive))

        self._generate = QCheckBox(_("Generate each new photo"))
        self._generate.setChecked(True)
        box.addWidget(self._generate)
        self._save = QCheckBox(_("Save it too, without reviewing"))
        self._save.setToolTip(
            tooltip(
                "Off by default: the window is built around reviewing before writing. Turn it on "
                "for an unattended import, where it behaves like the CLI's watch command.",
            ),
        )
        box.addWidget(self._save)

        start = QPushButton(_("Start Watching"))
        start.setObjectName("primary")
        start.clicked.connect(self._on_start)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(start)
        box.addLayout(buttons)

    def _build_form(self, *, recursive: bool) -> QFormLayout:
        """Build the form: which folder to poll, how often, and how long to wait."""
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._folder = QLineEdit()
        self._folder.setPlaceholderText(_("(pick a folder)"))
        form.addRow(_("Folder"), path_row(self._folder, self._choose_folder))

        self._recursive = QCheckBox(_("Include subfolders"))
        self._recursive.setChecked(recursive)
        form.addRow("", self._recursive)

        self._interval = QDoubleSpinBox()
        self._interval.setRange(1.0, 3600.0)
        self._interval.setDecimals(0)
        self._interval.setValue(DEFAULT_INTERVAL_SECONDS)
        self._interval.setSuffix(_(" seconds"))
        self._interval.setToolTip(
            tooltip(
                "How often the folder is listed. Polling behaves the same on every platform and "
                "over network shares.",
            ),
        )
        form.addRow(_("Check every"), self._interval)

        self._settle = QDoubleSpinBox()
        self._settle.setRange(0.0, 600.0)
        self._settle.setDecimals(0)
        self._settle.setValue(DEFAULT_SETTLE_SECONDS)
        self._settle.setSuffix(_(" seconds"))
        self._settle.setToolTip(
            tooltip(
                "How long a file must sit unchanged before it is picked up, so a photo still "
                "being copied is left alone until the copy finishes.",
            ),
        )
        form.addRow(_("Settle for"), self._settle)
        return form

    def _choose_folder(self) -> None:
        """Pick the folder to watch."""
        folder = QFileDialog.getExistingDirectory(
            self,
            _("Choose a folder to watch"),
            self._folder.text(),
        )
        if folder:
            self._folder.setText(folder)

    def ask(self, *, extensions: str) -> WatchSettings | None:
        """
        Open the dialog and return the watch to start, or None when it was cancelled.

        *extensions* are the folder-scan file types in force now; the watch picks up the same ones.
        """
        self._extensions = extensions
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        return self.settings()

    def settings(self) -> WatchSettings:
        """Return the watch the dialog describes."""
        return WatchSettings(
            folders=(Path(self._folder.text().strip()),),
            extensions=self._extensions,
            recursive=self._recursive.isChecked(),
            interval=self._interval.value(),
            settle=self._settle.value(),
            generate=self._generate.isChecked(),
            save=self._save.isChecked(),
        )

    def _on_start(self) -> None:
        """Close on Start, unless there is no folder yet to watch."""
        if not self._folder.text().strip():
            QMessageBox.warning(self, _("Pick a folder"), _("Choose the folder to watch first."))
            return
        self.accept()
