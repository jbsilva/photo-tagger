# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Save Options menu: what a save writes, replaces, and where."""

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu, QWidget

from photo_tagger.cli_options import resolve_sidecar_mode
from photo_tagger.config import DEFAULT_SIDECAR_MODE, SidecarMode
from photo_tagger.gui_state import SaveOptions, sidecar_overrides_summary, tooltip
from photo_tagger.i18n import _, gettext_noop
from photo_tagger.metadata import FIELD_DESCRIPTION, FIELD_KEYWORDS, FIELD_TITLE


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable

    from photo_tagger.cli_options import OutputConfig
    from photo_tagger.config import SidecarOverrides
    from photo_tagger.fields import ExtraFields


# The Write To submenu, mirroring the CLI's --sidecar-mode. Each entry is (mode, menu label,
# tooltip). Exclusive: a save writes one way, and the mixed mode decides per photo.
_WRITE_TO_CHOICES: tuple[tuple[SidecarMode, str, str], ...] = (
    (
        "all",
        gettext_noop("XMP Sidecar"),
        gettext_noop("Write a .xmp file next to every photo, leaving the photos untouched."),
    ),
    (
        "none",
        gettext_noop("The Photo Itself"),
        gettext_noop("Write into every image file, creating no sidecars."),
    ),
    (
        "raw",
        gettext_noop("Sidecar for RAW, Photo Otherwise"),
        gettext_noop(
            "Write a .xmp file beside each RAW photo and write into everything else. For a "
            "folder of RAW+JPEG pairs.",
        ),
    ),
    (
        "both",
        gettext_noop("Both, Photo and Sidecar"),
        gettext_noop(
            "Write into every photo and put the same metadata in a .xmp beside it. Twice the "
            "writes, and the photo travels tagged even where the sidecar is left behind.",
        ),
    ),
)

# The Overwrite Existing submenu, as (field, label, tooltip). Keywords merge, so that entry starts
# unchecked; a title and a description hold one value each, and replacing is what clears a
# camera-written placeholder.
_OVERWRITE_CHOICES: tuple[tuple[str, str, str], ...] = (
    (
        FIELD_TITLE,
        gettext_noop("Title"),
        gettext_noop("Replace an existing title. Uncheck to fill in only photos without one."),
    ),
    (
        FIELD_DESCRIPTION,
        gettext_noop("Description"),
        gettext_noop(
            "Replace an existing description. Uncheck to fill in only photos without one.",
        ),
    ),
    (
        FIELD_KEYWORDS,
        gettext_noop("Keywords"),
        gettext_noop("Replace existing keywords instead of merging the new ones in."),
    ),
)

# How each Write To mode reads inside the Save buttons' "Currently writes ..." sentence. Lambdas,
# not strings: the sentence is built on demand, so each phrase has to be translated on demand too.
_WRITE_TO_SUMMARIES: dict[SidecarMode, Callable[[], str]] = {
    "all": lambda: _("to an XMP sidecar"),
    "none": lambda: _("into the image file"),
    "raw": lambda: _("to a sidecar for RAW, into the image file otherwise"),
    "both": lambda: _("into the image file and an XMP sidecar"),
}


class SaveOptionsMenu(QMenu):
    """
    Decide what a save writes: the fields, replace or keep for each, where to, and the backup.

    One menu serves every save: the popup of both Save buttons, and a submenu of both Generate
    menus (for the save a run chains for itself). That is why it carries a title, and why a button
    about to write can say what it will write. Any toggle is announced with :attr:`changed`.
    """

    changed = Signal()

    def __init__(
        self,
        parent: QWidget,
        output: OutputConfig,
        *,
        sidecar_overrides: SidecarOverrides,
    ) -> None:
        """
        Build the menu, every entry pre-picked from the config file.

        Choices saved via Settings > Save Settings as Defaults come back on the next launch that
        way. *sidecar_overrides* is config-only, so the Write To submenu only names it.
        """
        super().__init__(_("Save Options"), parent)
        self._sidecar_overrides = sidecar_overrides
        self.setToolTipsVisible(True)
        self.write_title = QAction(_("Write Title"), self)
        self.write_title.setToolTip(
            tooltip("Write the title. Uncheck to leave the existing title as is."),
        )
        self.write_description = QAction(_("Write Description"), self)
        self.write_description.setToolTip(
            tooltip("Write the description. Uncheck to leave the existing description as is."),
        )
        self.write_keywords = QAction(_("Write Keywords"), self)
        self.write_keywords.setToolTip(
            tooltip(
                "Write keywords. Uncheck to leave existing keywords untouched, e.g. to refresh "
                "only the title and description.",
            ),
        )
        # Its tooltip and enabled state follow whether a copyright is set at all, so both are left
        # to set_copyright_usable, which the window calls once the Copyright dialog exists.
        self.write_copyright = QAction(_("Write Copyright"), self)
        for action, checked in (
            (self.write_title, output.write_title),
            (self.write_description, output.write_description),
            (self.write_keywords, output.write_keywords),
            (self.write_copyright, output.write_copyright),
        ):
            action.setCheckable(True)
            action.setChecked(checked)
            self.addAction(action)
        self.addSeparator()
        self.addMenu(self._build_overwrite_menu(output))
        # Connect only after the Overwrite submenu exists, since the handler enables and disables
        # its entries.
        for action in (self.write_title, self.write_description, self.write_keywords):
            action.toggled.connect(self._sync_overwrite_enabled)

        self.backup = QAction(_("Keep ExifTool Backup"), self)
        self.backup.setToolTip(
            tooltip(
                "Let ExifTool save the untouched file as *_original before writing. Uncheck to "
                "write in place, which leaves no extra copies filling up the disk on a large "
                "batch (make sure you have a backup elsewhere).",
            ),
        )
        self.backup.setCheckable(True)
        self.backup.setChecked(output.backup_xmp)
        self.addAction(self.backup)
        self.addMenu(self._build_write_to_menu(resolve_sidecar_mode(output)))
        # A config that starts with a field off must also start with its Overwrite entry grayed out.
        self._sync_overwrite_enabled()
        # Connected after the setChecked calls above so construction announces nothing.
        for action in (
            self.write_title,
            self.write_description,
            self.write_keywords,
            self.write_copyright,
            *self.overwrite_actions.values(),
            self.backup,
            *self.sidecar_actions.values(),
        ):
            action.toggled.connect(lambda _on: self.changed.emit())

    def _build_overwrite_menu(self, output: OutputConfig) -> QMenu:
        """Build the Overwrite Existing submenu, each entry pre-picked from the config file."""
        checked = {
            FIELD_TITLE: not output.preserve_title,
            FIELD_DESCRIPTION: not output.preserve_description,
            FIELD_KEYWORDS: not output.preserve_keywords,
        }
        menu = QMenu(_("Overwrite Existing"), self)
        menu.setToolTipsVisible(True)
        menu.setToolTip(tooltip("Which fields a save replaces when the photo already has one."))
        self.overwrite_actions: dict[str, QAction] = {}
        for field_name, label, hint in _OVERWRITE_CHOICES:
            action = QAction(_(label), self)
            action.setCheckable(True)
            action.setChecked(checked[field_name])
            action.setToolTip(tooltip(hint))
            menu.addAction(action)
            self.overwrite_actions[field_name] = action
        return menu

    def _build_write_to_menu(self, mode: SidecarMode) -> QMenu:
        """Build the exclusive Write To submenu, with *mode* pre-picked."""
        menu = QMenu(_("Write To"), self)
        menu.setToolTipsVisible(True)
        menu.setToolTip(tooltip("Where a save puts the metadata."))
        group = QActionGroup(self)
        self.sidecar_actions: dict[SidecarMode, QAction] = {}
        for value, label, hint in _WRITE_TO_CHOICES:
            action = QAction(_(label), self)
            action.setCheckable(True)
            action.setChecked(value == mode)
            action.setToolTip(tooltip(hint))
            group.addAction(action)
            menu.addAction(action)
            self.sidecar_actions[value] = action
        if overrides := sidecar_overrides_summary(self._sidecar_overrides):
            # Set in the config file, with no control here, so the menu at least says so.
            menu.addSeparator()
            note = QAction(_("From the config file: {overrides}").format(overrides=overrides), self)
            note.setEnabled(False)
            menu.addAction(note)
        return menu

    def _sync_overwrite_enabled(self) -> None:
        """Replace-or-keep only matters for a field being written; gray the rest out."""
        for field_name, action in (
            (FIELD_TITLE, self.write_title),
            (FIELD_DESCRIPTION, self.write_description),
            (FIELD_KEYWORDS, self.write_keywords),
        ):
            self.overwrite_actions[field_name].setEnabled(action.isChecked())

    def set_copyright_usable(self, *, usable: bool) -> None:
        """
        Gray out Write Copyright while there is no copyright to write.

        The same rule the Overwrite entries follow: a toggle for something that cannot happen yet
        should say so rather than look like a choice. It stays checked while disabled, so filling
        in the Copyright dialog switches it straight on.
        """
        self.write_copyright.setEnabled(usable)
        self.write_copyright.setToolTip(
            tooltip(
                "Write the credit and copyright notice from Settings > Copyright. Uncheck to skip "
                "them for now without clearing what you typed there.",
            )
            if usable
            else tooltip("Nothing to write yet. Fill in Settings > Copyright first."),
        )

    def overwrites(self, field_name: str) -> bool:
        """Report whether a save replaces *field_name* on a photo that already carries one."""
        return self.overwrite_actions[field_name].isChecked()

    def sidecar_mode(self) -> SidecarMode:
        """Return the picked Write To mode; the action group keeps exactly one checked."""
        for value, action in self.sidecar_actions.items():
            if action.isChecked():
                return value
        return DEFAULT_SIDECAR_MODE

    def options(
        self,
        *,
        copyright_fields: ExtraFields | None,
        verbatim: dict[str, str] | None,
    ) -> SaveOptions:
        """
        Snapshot the toggles, so a background write never has to read a widget.

        *copyright_fields* is what the Copyright dialog holds, and *verbatim* the vocabulary's own
        spelling of each term. The fields are left out while Write Copyright is off, so an unchecked
        toggle skips the whole block without anyone having to empty the dialog that holds it.
        """
        return SaveOptions(
            write_title=self.write_title.isChecked(),
            write_description=self.write_description.isChecked(),
            write_keywords=self.write_keywords.isChecked(),
            overwrite_title=self.overwrites(FIELD_TITLE),
            overwrite_description=self.overwrites(FIELD_DESCRIPTION),
            overwrite_keywords=self.overwrites(FIELD_KEYWORDS),
            backup=self.backup.isChecked(),
            sidecar_mode=self.sidecar_mode(),
            sidecar_overrides=self._sidecar_overrides,
            verbatim=verbatim,
            extra_fields=copyright_fields if self.write_copyright.isChecked() else None,
        )

    def config_values(self) -> dict[str, bool | SidecarMode]:
        """Return the toggles as the config file stores them, keyed like GuiConfigValues."""
        return {
            "write_title": self.write_title.isChecked(),
            "write_description": self.write_description.isChecked(),
            "write_keywords": self.write_keywords.isChecked(),
            "write_copyright": self.write_copyright.isChecked(),
            "preserve_keywords": not self.overwrites(FIELD_KEYWORDS),
            "preserve_title": not self.overwrites(FIELD_TITLE),
            "preserve_description": not self.overwrites(FIELD_DESCRIPTION),
            "sidecar_mode": self.sidecar_mode(),
            "backup_xmp": self.backup.isChecked(),
        }

    def summary(self, *, writes_copyright: bool) -> str:
        """Describe what a save currently writes, for the Save buttons' tooltips."""
        fields = [
            name
            for action, name in (
                (self.write_title, _("title")),
                (self.write_description, _("description")),
                (self.write_keywords, _("keywords")),
            )
            if action.isChecked()
        ]
        if writes_copyright:
            fields.append(_("your copyright"))
        parts = [", ".join(fields) if fields else _("nothing (pick a field in the arrow menu)")]
        if self.write_keywords.isChecked():
            parts.append(
                _("overwriting existing keywords")
                if self.overwrites(FIELD_KEYWORDS)
                else _("merging with existing keywords"),
            )
        if kept := self._kept_captions():
            parts.append(_("keeping any existing {fields}").format(fields=", ".join(kept)))
        parts.append(_WRITE_TO_SUMMARIES[self.sidecar_mode()]())
        if overrides := sidecar_overrides_summary(self._sidecar_overrides):
            parts.append(_("but {overrides}").format(overrides=overrides))
        parts.append(
            _("keeping a *_original backup")
            if self.backup.isChecked()
            else _("with no *_original backup"),
        )
        return ", ".join(parts)

    def _kept_captions(self) -> list[str]:
        """Name the text fields a save writes only where the photo has none, for the summary."""
        return [
            name
            for action, field_name, name in (
                (self.write_title, FIELD_TITLE, _("title")),
                (self.write_description, FIELD_DESCRIPTION, _("description")),
            )
            if action.isChecked() and not self.overwrites(field_name)
        ]
