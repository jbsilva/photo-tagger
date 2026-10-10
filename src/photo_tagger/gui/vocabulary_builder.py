# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Tools > Build Vocabulary dialog and the background build it runs."""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import CHOOSE_LABEL, CLOSE_LABEL, DEFAULT_VOCABULARY_FILE
from photo_tagger.gui.threads import release_thread, start_in_thread
from photo_tagger.gui.widgets import blurb, hint_label
from photo_tagger.gui.workers import VocabularyBuildWorker
from photo_tagger.gui_state import tooltip
from photo_tagger.i18n import _, ngettext
from photo_tagger.vocabulary import folds_plurals
from photo_tagger.vocabulary_build import TrimRules


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable

    from photo_tagger.gui_state import Connection


# How long closing the window waits for a build before detaching it. A build has no stop hook at
# all: its model calls run to completion.
_STOP_TIMEOUT_MS = 3000


class VocabularyBuilderDialog(QDialog):
    """
    Build a controlled vocabulary out of a library's own keywords.

    The same three steps as ``photo-tagger vocabulary``: count what the photos (or a Lightroom
    export) carry, trim the count with deterministic rules, and optionally ask the model to fold
    synonyms and give the list a hierarchy. Only the result file is written; no photo is touched.
    The build runs on its own thread. The dialog reports a status line with :attr:`status`, the
    feature used with :attr:`used`, and a file to put in force with :attr:`use_vocabulary`.
    """

    status = Signal(str)
    used = Signal(str)
    use_vocabulary = Signal(object)  # Path

    def __init__(
        self,
        parent: QWidget,
        *,
        photo_paths: Callable[[], list[Path]],
        connection: Callable[[], Connection],
        output_language: Callable[[], str],
    ) -> None:
        """
        Build the dialog; the three callables are asked when a build starts, never before.

        *photo_paths* lists the photos in the window, *connection* names the model that organizes
        the result, and *output_language* decides whether plurals fold onto their singular.
        """
        super().__init__(parent)
        self._photo_paths = photo_paths
        self._connection = connection
        self._output_language = output_language
        self._thread: QThread | None = None
        self._worker: VocabularyBuildWorker | None = None
        # Where the build last wrote, so a finished one can offer to use it.
        self._built: Path | None = None
        self._closing = False
        self.setWindowTitle(_("Build a vocabulary"))
        self.setMinimumWidth(660)
        box = QVBoxLayout(self)
        intro = blurb(
            _(
                "Reads the keywords your photos already carry, keeps the ones that earn their "
                "place, and writes them as a keyword file to review and edit. Nothing is written "
                "to your photos.",
            ),
        )
        box.addWidget(intro)
        box.addWidget(self._build_source_group())
        box.addWidget(self._build_rules_group())
        box.addWidget(self._build_organize_group())
        box.addWidget(self._build_output_group())

        self._status = hint_label("")
        box.addWidget(self._status)

        self._button = QPushButton(_("Build"))
        self._button.setObjectName("primary")
        self._button.clicked.connect(self._start)
        close = QPushButton(_(CLOSE_LABEL))
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(self._button)
        box.addLayout(buttons)

    def _build_source_group(self) -> QGroupBox:
        """Build the source picker: the photos in the list, a keyword export, or both."""
        group = QGroupBox(_("Read from"))
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._from_photos = QCheckBox(_("The photos in the list"))
        self._from_photos.setChecked(True)
        self._from_photos.setToolTip(
            tooltip(
                "Counts the keywords on the photos themselves, through ExifTool, so any "
                "application that writes XMP or IPTC counts, not only Lightroom.",
            ),
        )
        form.addRow("", self._from_photos)
        self._export = QLineEdit()
        self._export.setPlaceholderText(_("(optional)"))
        self._export.setToolTip(
            tooltip(
                "A Lightroom keyword export (.txt or .csv), for a catalog that is not on this "
                "machine. Its counts are occurrences in the keyword tree, not photos.",
            ),
        )
        choose = QPushButton(_(CHOOSE_LABEL))
        choose.clicked.connect(self._choose_keyword_export)
        row = QHBoxLayout()
        row.addWidget(self._export, stretch=1)
        row.addWidget(choose)
        form.addRow(_("Keyword export"), row)
        return group

    def _build_rules_group(self) -> QGroupBox:
        """Build the deterministic filters that turn the keyword count into a file."""
        defaults = TrimRules()
        group = QGroupBox(_("Keep a keyword when"))
        form = QFormLayout(group)
        self._min_uses = QSpinBox()
        self._min_uses.setRange(1, 1000)
        self._min_uses.setValue(defaults.min_uses)
        self._min_uses.setToolTip(
            tooltip(
                "The most useful knob: in a catalog an AI has been writing to, most keywords are "
                "used once and are one-offs rather than vocabulary.",
            ),
        )
        form.addRow(_("Used at least this often"), self._min_uses)
        self._max_terms = QSpinBox()
        self._max_terms.setRange(0, 100_000)
        self._max_terms.setValue(defaults.max_terms or 0)
        self._max_terms.setSpecialValueText(_("no cap"))
        self._max_terms.setToolTip(
            tooltip(
                "Caps the file, dropping the least used first. The default keeps it under the "
                "5000 terms above which matching gives up its fuzzy pass.",
            ),
        )
        form.addRow(_("And the file holds at most"), self._max_terms)
        self._digits = QCheckBox(_("Keep keywords containing digits"))
        self._digits.setToolTip(
            tooltip(
                "Off by default: a keyword with a digit is nearly always a measurement or a model "
                "number ('19.5V', '0 Percent Battery') rather than a subject.",
            ),
        )
        form.addRow("", self._digits)
        self._flat = QCheckBox(_("Write bare keywords, without their hierarchies"))
        self._flat.setToolTip(
            tooltip(
                "Worth using when the source hierarchy is not trustworthy, since a vocabulary "
                "imposes its own on every photo it matches.",
            ),
        )
        form.addRow("", self._flat)
        return group

    def _build_organize_group(self) -> QGroupBox:
        """Build the opt-in model pass over the keywords that survived the count."""
        group = self._organize = QGroupBox(_("Organize with the model"))
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(
            tooltip(
                "Asks the model for the two things counting cannot settle: which keywords are "
                "synonyms of each other (folded into one, the rest kept so they still match), and "
                "what hierarchy the list should have. It never decides what to keep, and never "
                "invents a keyword. Needs a reachable provider.",
            ),
        )
        form = QFormLayout(group)
        self._organize_workers = QSpinBox()
        self._organize_workers.setRange(1, 16)
        self._organize_workers.setValue(1)
        self._organize_workers.setToolTip(
            tooltip("Model requests to run at once; the list is sent in chunks."),
        )
        form.addRow(_("Requests at once"), self._organize_workers)
        return group

    def _build_output_group(self) -> QGroupBox:
        """Build the output pickers: the generated file, and the optional drop report."""
        group = QGroupBox(_("Write to"))
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._output = QLineEdit(str(DEFAULT_VOCABULARY_FILE))
        output_choose = QPushButton(_(CHOOSE_LABEL))
        output_choose.clicked.connect(self._choose_vocabulary_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output, stretch=1)
        output_row.addWidget(output_choose)
        form.addRow(_("Vocabulary file"), output_row)

        self._report = QLineEdit()
        self._report.setPlaceholderText(_("(optional)"))
        self._report.setToolTip(
            tooltip(
                "A CSV naming every dropped keyword, its count, and the rule that cut it. This is "
                "what makes the thresholds tunable rather than a guess.",
            ),
        )
        report_choose = QPushButton(_(CHOOSE_LABEL))
        report_choose.clicked.connect(self._choose_drop_report)
        report_row = QHBoxLayout()
        report_row.addWidget(self._report, stretch=1)
        report_row.addWidget(report_choose)
        form.addRow(_("Drop report"), report_row)
        return group

    def show_for(self, photo_count: int) -> None:
        """Open the dialog, naming how many photos a build would read from."""
        self._from_photos.setText(
            ngettext(
                "The {n} photo in the list",
                "The {n} photos in the list",
                photo_count,
            ).format(n=photo_count),
        )
        self._status.setText("")
        self.exec()

    def _choose_keyword_export(self) -> None:
        """Pick the Lightroom keyword export to count instead of (or as well as) the photos."""
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            _("Choose a keyword export"),
            "",
            _("Keyword exports (*.txt *.csv);;All files (*)"),
        )
        if chosen:
            self._export.setText(chosen)

    def _choose_vocabulary_output(self) -> None:
        """Pick where the generated vocabulary file is written."""
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            _("Write the vocabulary to"),
            self._output.text() or str(DEFAULT_VOCABULARY_FILE),
            _("Keyword files (*.txt);;All files (*)"),
        )
        if chosen:
            self._output.setText(chosen)

    def _choose_drop_report(self) -> None:
        """Pick where the CSV of dropped keywords is written."""
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            _("Write the drop report to"),
            self._report.text() or "dropped-keywords.csv",
            _("CSV files (*.csv);;All files (*)"),
        )
        if chosen:
            self._report.setText(chosen)

    def _start(self) -> None:
        """Read the dialog's choices and run the build on a background thread."""
        if self._thread is not None:
            return
        self.used.emit("build_vocabulary")
        paths = self._photo_paths() if self._from_photos.isChecked() else []
        # Read both optional paths first. An assignment inside the argument list below reads as a
        # puzzle, and the worker takes twelve arguments already.
        export_text = self._export.text().strip()
        report_text = self._report.text().strip()
        export = Path(export_text) if export_text else None
        if not paths and export is None:
            self._status.setText(
                _("Pick a source: the photos in the list, a keyword export, or both."),
            )
            return
        output = Path(self._output.text().strip() or str(DEFAULT_VOCABULARY_FILE))
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._status.setText(str(exc))
            return
        self._built = output
        connection = self._connection()
        self._worker = VocabularyBuildWorker(
            paths,
            export,
            output,
            rules=TrimRules(
                min_uses=self._min_uses.value(),
                max_terms=self._max_terms.value() or None,
                allow_digits=self._digits.isChecked(),
                # The metadata language from Settings: folding plurals is English morphology, and
                # applying it to another language drops real keywords as variants of each other.
                fold_plurals=folds_plurals(self._output_language()),
            ),
            flat=self._flat.isChecked(),
            report_file=Path(report_text) if report_text else None,
            organize_workers=self._organize_workers.value(),
            provider=connection.provider if self._organize.isChecked() else None,
            model=connection.model,
            api_base_url=connection.api_base_url,
            api_key=connection.api_key,
        )
        self._worker.progress.connect(self._status.setText)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._button.setEnabled(False)
        self._status.setText(_("Working..."))
        self._thread = start_in_thread(self, self._worker)

    def _on_done(self, message: str, kept: int) -> None:
        """Report the finished file and offer to put it straight to work."""
        if self._closing:
            # The file is written; the window it would report to is on its way out.
            return
        self._status.setText(message)
        self.status.emit(message)
        if not kept:
            return
        reply = QMessageBox.question(
            self,
            _("Use this vocabulary?"),
            _("Snap the keywords of the next generation onto the file you just built?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.use_vocabulary.emit(self._built)

    def _on_failed(self, message: str) -> None:
        """Report a build that produced nothing, in the dialog and the status bar."""
        if self._closing:
            return
        self._status.setText(message)
        self.status.emit(message)

    def _on_finished(self) -> None:
        """Release the build thread and let the dialog be used again."""
        if self._closing:
            return
        self._teardown()
        self._button.setEnabled(True)

    def shutdown(self) -> None:
        """Stop reporting and let go of the build, for a window that is closing."""
        self._closing = True
        self._teardown()

    def _teardown(self) -> None:
        """
        Join the build thread and release both it and its worker.

        A build that is still organizing is a run of model calls that cannot be interrupted, so a
        thread that does not stop promptly is detached rather than allowed to block the close. It
        writes one file and exits; nothing it does afterwards touches the window.
        """
        if self._thread is not None:
            release_thread(
                self._thread,
                timeout_ms=_STOP_TIMEOUT_MS,
                detached_event="gui_vocabulary_build_still_running",
            )
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
