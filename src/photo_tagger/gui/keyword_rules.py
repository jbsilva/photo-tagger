# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Settings > Keyword Rules dialog: the controlled vocabulary and the session gap."""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import CHOOSE_LABEL, SESSION_NOTE
from photo_tagger.gui.widgets import close_row, hint_label
from photo_tagger.gui_state import load_vocabulary_file, tooltip, vocabulary_status
from photo_tagger.i18n import _


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable

    from photo_tagger.cli_options import OutputConfig
    from photo_tagger.vocabulary import Vocabulary


class KeywordRulesDialog(QDialog):
    """
    Hold the two settings that decide which keywords a run ends up writing.

    The controlled vocabulary snaps generated keywords onto a keyword file's own terms, and the
    session gap makes the keywords of one shoot agree with each other. The dialog owns both: the
    window reads them from here for every run and every settings save. It reports a status line with
    :attr:`status` and each feature used with :attr:`used`, and asks the window to open the
    vocabulary builder with :attr:`build_requested`.
    """

    status = Signal(str)
    used = Signal(str)
    build_requested = Signal()

    def __init__(
        self,
        parent: QWidget,
        output: OutputConfig,
        *,
        output_language: Callable[[], str],
    ) -> None:
        """
        Build the dialog on what the config file set.

        The vocabulary file itself is not read here: it is disk IO, and a failure needs the status
        bar to report it, so the window calls :meth:`load_vocabulary` once it can. The metadata
        language decides whether plurals fold onto their singular, so *output_language* is asked
        each time a file is read.
        """
        super().__init__(parent)
        self._output_language = output_language
        self._path: Path | None = output.vocabulary
        self._vocabulary: Vocabulary | None = None
        self._error = ""
        self._strict = output.vocabulary_strict
        self._session_gap = output.session_gap_minutes
        self.setWindowTitle(_("Keyword rules"))
        self.setMinimumWidth(620)
        box = QVBoxLayout(self)
        box.addWidget(self._build_vocabulary_group())
        box.addWidget(self._build_session_group())
        box.addWidget(hint_label(_(SESSION_NOTE)))
        box.addLayout(close_row(self))

    @property
    def vocabulary(self) -> Vocabulary | None:
        """Return the vocabulary in force, or None when there is none (or the file was refused)."""
        return self._vocabulary

    @property
    def vocabulary_path(self) -> Path | None:
        """Return the chosen keyword file, kept even when it could not be used."""
        return self._path

    @property
    def strict(self) -> bool:
        """Report whether generated keywords the vocabulary does not cover are dropped."""
        return self._strict

    @property
    def session_gap(self) -> float:
        """Return the idle minutes that separate one shoot from the next; zero means off."""
        return self._session_gap

    def _build_vocabulary_group(self) -> QGroupBox:
        """Build the controlled-vocabulary half of the dialog."""
        group = QGroupBox(_("Controlled vocabulary"))
        layout = QVBoxLayout(group)
        blurb = QLabel(
            _(
                "Restrict generated keywords to the terms in a keyword file: every match is "
                "rewritten to the file's own spelling and hierarchy, so a run cannot fill your "
                "catalog with near-duplicates of keywords you already have.",
            ),
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._vocabulary_field = QLineEdit(str(self._path or ""))
        self._vocabulary_field.setReadOnly(True)
        self._vocabulary_field.setPlaceholderText(_("(no vocabulary file)"))
        choose = QPushButton(_(CHOOSE_LABEL))
        choose.clicked.connect(self._choose_vocabulary)
        clear = QPushButton(_("Clear"))
        clear.clicked.connect(lambda: self.load_vocabulary(None))
        build = QPushButton(_("Build..."))
        build.setToolTip(tooltip("Write one from the keywords your own photos already carry."))
        build.clicked.connect(self.build_requested)
        row = QHBoxLayout()
        row.addWidget(self._vocabulary_field, stretch=1)
        row.addWidget(choose)
        row.addWidget(clear)
        row.addWidget(build)
        layout.addLayout(row)

        self._vocabulary_label = hint_label(vocabulary_status(None, None))
        layout.addWidget(self._vocabulary_label)

        self._strict_box = QCheckBox(_("Write only keywords the vocabulary covers"))
        self._strict_box.setChecked(self._strict)
        self._strict_box.setToolTip(
            tooltip(
                "Drop generated keywords the vocabulary does not have instead of writing them as "
                "they came. The status bar names what was dropped, so the vocabulary can grow on "
                "purpose rather than by accident.",
            ),
        )
        self._strict_box.toggled.connect(self._on_strict_toggled)
        layout.addWidget(self._strict_box)
        return group

    def _build_session_group(self) -> QGroupBox:
        """Build the shoot-harmonization half of the dialog."""
        group = QGroupBox(_("Shoot harmonization"))
        layout = QVBoxLayout(group)
        blurb = QLabel(
            _(
                "Group photos into shoots separated by this many idle minutes (by capture time, "
                "falling back to the file date) and make each shoot's keywords agree with itself: "
                "the spelling and the hierarchy most of the shoot used win for all of it. It runs "
                "over the proposals after each generation, so you still review before saving.",
            ),
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._session_gap_box = QDoubleSpinBox()
        self._session_gap_box.setRange(0.0, 1440.0)
        self._session_gap_box.setDecimals(0)
        self._session_gap_box.setSingleStep(15.0)
        self._session_gap_box.setValue(self._session_gap)
        self._session_gap_box.setSuffix(_(" minutes"))
        self._session_gap_box.setSpecialValueText(_("off"))
        self._session_gap_box.setToolTip(
            tooltip("Zero treats every photo on its own, which is the default."),
        )
        self._session_gap_box.valueChanged.connect(self._on_session_gap_changed)
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Split shoots after")))
        row.addWidget(self._session_gap_box)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    def refresh(self) -> None:
        """Put the values currently in force back into the widgets, before the dialog opens."""
        self._vocabulary_field.setText(str(self._path or ""))
        self._strict_box.setChecked(self._strict)
        self._session_gap_box.setValue(self._session_gap)
        self._refresh_vocabulary_label()

    def _choose_vocabulary(self) -> None:
        """Pick a keyword file and put it in force for the next generation."""
        self.used.emit("choose_vocabulary")
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            _("Choose a vocabulary file"),
            str(self._path or ""),
            _("Keyword files (*.txt *.csv);;All files (*)"),
        )
        if chosen:
            self.load_vocabulary(Path(chosen))

    def load_vocabulary(self, path: Path | None, *, announce: bool = True) -> None:
        """
        Put *path* in force as the controlled vocabulary, or clear it when None.

        A file that cannot be used is kept as the chosen path but not as a vocabulary, so the dialog
        can show which file was refused and why instead of quietly writing without one. The status
        line says so either way; with *announce* off it speaks only for a refused file.
        """
        self._path = path
        self._vocabulary = None
        self._error = ""
        if path is not None:
            self._vocabulary, self._error = load_vocabulary_file(
                path,
                output_language=self._output_language(),
            )
            if self._vocabulary is None:
                logger.warning("gui_vocabulary_unusable", file=str(path), error=self._error)
        self._vocabulary_field.setText(str(path or ""))
        self._refresh_vocabulary_label()
        if announce or self._error:
            self.status.emit(vocabulary_status(self._path, self._vocabulary, self._error))

    def _refresh_vocabulary_label(self) -> None:
        """Keep the dialog's status line describing the vocabulary actually in force."""
        self._vocabulary_label.setText(vocabulary_status(self._path, self._vocabulary, self._error))

    def _on_strict_toggled(self, strict: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Remember whether keywords outside the vocabulary are dropped."""
        self.used.emit("vocabulary_strict")
        self._strict = strict

    def _on_session_gap_changed(self, minutes: float) -> None:
        """Remember the idle gap that separates one shoot from the next."""
        self.used.emit("session_gap")
        self._session_gap = minutes
