# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Settings > Copyright dialog: the credit and rights every save writes."""

from functools import partial
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import SESSION_NOTE
from photo_tagger.gui.widgets import close_row, fit_text_height, hint_label, readonly_box
from photo_tagger.gui_state import (
    ascii_warning,
    build_copyright_fields,
    copyright_summary,
    tooltip,
)
from photo_tagger.i18n import _, gettext_noop


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable

    from photo_tagger.cli_options import OutputConfig
    from photo_tagger.fields import ExtraFields


# How much room the preview gets. The floor holds the "nothing is written" line plus one more; the
# ceiling is nine tag lines, past which it scrolls rather than pushing the note and the Close button
# off a small screen.
_PREVIEW_MIN_HEIGHT = 60
_PREVIEW_MAX_HEIGHT = 200

# What the Insert buttons type, and why. The text is the button's own label: these are syntax, not
# prose, so they read the same in every language.
_INSERTS: tuple[tuple[str, str], ...] = (
    (
        "©",
        gettext_noop(
            "The copyright sign. XMP is UTF-8 and carries it. EXIF types its copyright field as "
            "ASCII before Exif 3.0, so '(c)' is the safe spelling for the notice.",
        ),
    ),
    (
        "{year}",
        gettext_noop("The year the photo was taken, from its capture date."),
    ),
    (
        "{artist}",
        gettext_noop("The name in the Creator field above."),
    ),
)

# The three answers to XMP-xmpRights:Marked, in the order the dialog lists them. None is first
# because leaving the tag alone is what a photo gets until somebody decides otherwise.
_MARKED_CHOICES: tuple[tuple[str, bool | None], ...] = (
    (gettext_noop("Leave as is"), None),
    (gettext_noop("Under copyright"), True),
    (gettext_noop("Public domain"), False),
)


class CopyrightDialog(QDialog):
    """
    Hold the credit and copyright notice every save writes, previewed on a real year.

    The model cannot know who took a photo or who owns it, so this is one setting for every photo
    rather than a per-photo edit. Every edit is read straight into :attr:`fields` and announced with
    :attr:`changed`, so the Save buttons and the open photo's Copyright row stay in step.
    """

    changed = Signal()

    def __init__(
        self,
        parent: QWidget,
        output: OutputConfig,
        *,
        preview_year: Callable[[], int],
    ) -> None:
        """
        Build the dialog on the config file's [copyright] table.

        *preview_year* names the year the preview dates its templates by, asked afresh on every
        refresh. ``set_tags`` is config-only: the dialog lists what it writes but offers no way to
        edit raw tag assignments.
        """
        super().__init__(parent)
        self._preview_year = preview_year
        self._tags: list[str] = list(output.set_tags or [])
        self._fields, error = build_copyright_fields(
            artist=output.artist or "",
            notice=output.copyright_notice or "",
            rights=output.rights or "",
            marked=output.marked,
            assignments=self._tags,
            preserve=output.preserve_copyright,
        )
        if error:
            # A config the CLI would refuse must not take the window down with it. The dialog
            # shows the same message the moment it opens.
            logger.warning("gui_copyright_config_unusable", error=error)
        self.setWindowTitle(_("Copyright"))
        self.setMinimumWidth(620)
        box = QVBoxLayout(self)
        # Two sentences, broken by hand: a QLabel's size hint is the width of its longest line, so
        # one long sentence opens the dialog as wide as itself however much it would wrap.
        blurb = QLabel(
            _("The model cannot know who took a photo or who owns it.")
            + "\n"
            + _("Use {year} for the year that photo was taken, and {artist} for the name below."),
        )
        blurb.setWordWrap(True)
        box.addWidget(blurb)
        box.addLayout(self._build_form(output))
        box.addLayout(self._build_inserts())

        # A box rather than a label: the preview is one line per tag and grows as fields are filled
        # in, and a label that grows inside a dialog gets squeezed instead, cutting off its last
        # line. This scrolls past _PREVIEW_MAX_HEIGHT, and the lines can be copied out.
        self._preview = readonly_box(_PREVIEW_MIN_HEIGHT)
        box.addWidget(self._preview)

        self._warning = QLabel("")
        self._warning.setObjectName("warning")
        self._warning.setWordWrap(True)
        box.addWidget(self._warning)
        box.addWidget(hint_label(_(SESSION_NOTE)))
        box.addLayout(close_row(self))
        self.refresh()

    @property
    def fields(self) -> ExtraFields | None:
        """Return what a save writes, or None while the dialog holds something it cannot use."""
        return self._fields

    @property
    def usable(self) -> bool:
        """Report whether there is a copyright to write at all."""
        return self._fields is not None and not self._fields.is_empty

    def config_values(self) -> dict[str, str | bool | None]:
        """Return the dialog's fields as the config file stores them, keyed like GuiConfigValues."""
        return {
            "artist": self._artist_field.text().strip(),
            "copyright_notice": self._notice_field.text().strip(),
            "rights": self._rights_field.text().strip(),
            "marked": self._marked_box.currentData(),
            "preserve_copyright": self._preserve_copyright_box.isChecked(),
        }

    def _build_form(self, output: OutputConfig) -> QFormLayout:
        """
        Build the five rows the dialog edits.

        The growth policy is set rather than left to the style: QMacStyle keeps a form's fields at
        their size hint, which left the notice field a couple of hundred pixels wide in a dialog
        several times that, with the example template it carries cut off mid-word.
        """
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._artist_field = QLineEdit(output.artist or "")
        self._artist_field.setPlaceholderText(_("Your name"))
        self._artist_field.setToolTip(
            tooltip("Written to EXIF:Artist, XMP-dc:Creator, and IPTC:By-line."),
        )
        self._notice_field = QLineEdit(output.copyright_notice or "")
        self._notice_field.setPlaceholderText(
            _("Copyright (c) {year} {artist}, All rights reserved"),
        )
        self._notice_field.setToolTip(
            tooltip("Written to EXIF:Copyright, IPTC:CopyrightNotice, and XMP-dc:Rights."),
        )
        self._rights_field = QLineEdit(output.rights or "")
        self._rights_field.setPlaceholderText(_("(same as the notice)"))
        self._rights_field.setToolTip(
            tooltip(
                "XMP-dc:Rights on its own, for when it should read differently: XMP is UTF-8, so "
                "a real copyright sign is safe there.",
            ),
        )
        form.addRow(_("Creator"), self._artist_field)
        form.addRow(_("Copyright notice"), self._notice_field)
        form.addRow(_("XMP rights"), self._rights_field)

        self._marked_box = QComboBox()
        for label, value in _MARKED_CHOICES:
            self._marked_box.addItem(_(label), value)
        self._marked_box.setCurrentIndex(self._marked_box.findData(output.marked))
        self._marked_box.setToolTip(
            tooltip("XMP-xmpRights:Marked, the flag that says the photo is under copyright."),
        )
        form.addRow(_("Rights flag"), self._marked_box)

        self._preserve_copyright_box = QCheckBox(_("Keep the name and notice a photo already has"))
        self._preserve_copyright_box.setChecked(output.preserve_copyright)
        self._preserve_copyright_box.setToolTip(
            tooltip(
                "A photo with a creator keeps it, and one with a notice keeps the notice, the XMP "
                "rights, and the flag. Uncheck to replace them.",
            ),
        )
        form.addRow("", self._preserve_copyright_box)

        for widget in (self._artist_field, self._notice_field, self._rights_field):
            widget.textChanged.connect(self.refresh)
        self._marked_box.currentIndexChanged.connect(self.refresh)
        self._preserve_copyright_box.toggled.connect(self.refresh)
        return form

    def _build_inserts(self) -> QHBoxLayout:
        """
        Build the row of buttons that type the awkward parts of a notice for you.

        ``©`` is the one nobody has on a key, and the two placeholders are worth showing rather than
        only describing. Each button takes no focus, so the text field the user was typing in is
        still the one that gets the insert (see :meth:`_insert`).
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Insert:")))
        for text, hint in _INSERTS:
            button = QPushButton(text)
            button.setToolTip(tooltip(hint))
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(partial(self._insert, text))
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _insert(self, text: str) -> None:
        """Type *text* into whichever template field has the cursor, the notice by default."""
        target = QApplication.focusWidget()
        if target not in (self._notice_field, self._rights_field):
            target = self._notice_field
        target.insert(text)
        target.setFocus()

    def refresh(self) -> None:
        """Re-read the dialog into the fields a save writes, and preview them on a real year."""
        self._fields, error = build_copyright_fields(
            artist=self._artist_field.text(),
            notice=self._notice_field.text(),
            rights=self._rights_field.text(),
            marked=self._marked_box.currentData(),
            assignments=self._tags,
            preserve=self._preserve_copyright_box.isChecked(),
        )
        year = self._preview_year()
        self._preview.setPlainText(error or copyright_summary(self._fields, year))
        fit_text_height(self._preview, min_h=_PREVIEW_MIN_HEIGHT, max_h=_PREVIEW_MAX_HEIGHT)
        self._warning.setText("" if error else ascii_warning(self._fields, year))
        self.changed.emit()
