# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Select menu above the photo list: check and uncheck photos in bulk."""

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMenu, QWidget

from photo_tagger.gui_state import (
    FILTER_FAILED,
    FILTER_GENERATED,
    FILTER_MIXED_SCRIPT,
    FILTER_NO_COPYRIGHT,
    FILTER_PENDING,
    FILTER_SAVED,
    FILTER_UNTAGGED,
    SELECT_CHECK,
    SELECT_ONLY,
    SELECT_UNCHECK,
    TaggedCriterion,
    extension_counts,
    matches_extension,
    photo_matches_filter,
    tooltip,
)
from photo_tagger.i18n import _, gettext_noop
from photo_tagger.metadata import FIELD_DESCRIPTION, FIELD_KEYWORDS, FIELD_TITLE


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable, Iterable

    from photo_tagger.gui_state import PhotoItem


# The field-aware "deselect already-tagged" menu, mirroring the CLI's --skip-tagged but letting
# the user pick which fields count as "done". Each entry pairs a menu label with the criterion it
# applies. "Any metadata" is the broad OR criterion (the original skip-tagged); the rest require all
# of their fields, so a keyword-only photo survives "a title and a description" and stays selected
# for title/description generation.
_TAGGED_PRESETS: tuple[tuple[str, TaggedCriterion], ...] = (
    (
        gettext_noop("Has any metadata"),
        TaggedCriterion(
            frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}),
            match_all=False,
            phrase=gettext_noop("any metadata"),
        ),
    ),
    (
        gettext_noop("Has a title"),
        TaggedCriterion(
            frozenset({FIELD_TITLE}),
            match_all=True,
            phrase=gettext_noop("a title"),
        ),
    ),
    (
        gettext_noop("Has a description"),
        TaggedCriterion(
            frozenset({FIELD_DESCRIPTION}),
            match_all=True,
            phrase=gettext_noop("a description"),
        ),
    ),
    (
        gettext_noop("Has a title and a description"),
        TaggedCriterion(
            frozenset({FIELD_TITLE, FIELD_DESCRIPTION}),
            match_all=True,
            phrase=gettext_noop("a title and a description"),
        ),
    ),
    (
        gettext_noop("Has keywords"),
        TaggedCriterion(
            frozenset({FIELD_KEYWORDS}),
            match_all=True,
            phrase=gettext_noop("keywords"),
        ),
    ),
)

# The menu's three bulk verbs, each offering the same criteria below. Each entry is (verb, menu
# label, tooltip).
_SELECT_VERBS: tuple[tuple[str, str, str], ...] = (
    (
        SELECT_CHECK,
        gettext_noop("Check"),
        gettext_noop("Check the matching photos, leaving every other photo as it is."),
    ),
    (
        SELECT_UNCHECK,
        gettext_noop("Uncheck"),
        gettext_noop("Uncheck the matching photos, leaving every other photo as it is."),
    ),
    (
        SELECT_ONLY,
        gettext_noop("Check Only"),
        gettext_noop("Check the matching photos and uncheck every other photo in the list."),
    ),
)

# The lifecycle criteria a bulk verb can act on, reusing the folder grid's own filters so "saved"
# and "generated" mean the same thing in both places. Each entry is (filter, menu label); the
# label doubles as the phrase the status line reports back.
_SELECT_STATUSES: tuple[tuple[str, str], ...] = (
    (FILTER_PENDING, gettext_noop("Not generated")),
    (FILTER_GENERATED, gettext_noop("Generated")),
    (FILTER_SAVED, gettext_noop("Saved")),
    (FILTER_FAILED, gettext_noop("Failed")),
    (FILTER_UNTAGGED, gettext_noop("Untagged")),
    (FILTER_NO_COPYRIGHT, gettext_noop("No copyright")),
    (FILTER_MIXED_SCRIPT, gettext_noop("Mixed writing systems")),
)


class SelectMenu(QMenu):
    """
    Offer the bulk check and uncheck actions, including the CLI's --skip-tagged and --skip-from.

    Each entry only says what it wants: the window owns the photos and carries every action out,
    so the menu asks for them through signals. Its File Type submenus list the types the list holds
    right now, which *photos* is asked for each time one opens.
    """

    check_all_requested = Signal()
    uncheck_all_requested = Signal()
    invert_requested = Signal()
    # Which photos match (a PhotoItem -> bool), the verb to apply to them, and the phrase that
    # names the criterion in the status line.
    matching_requested = Signal(object, str, str)
    pattern_requested = Signal(str)  # the verb; the window asks for the pattern itself
    tagged_requested = Signal(object)  # the TaggedCriterion to uncheck by
    skip_list_requested = Signal()

    def __init__(self, parent: QWidget, *, photos: Callable[[], Iterable[PhotoItem]]) -> None:
        """Build the menu; *photos* lists the photos in the list, for the File Type submenus."""
        super().__init__(parent)
        self._photos = photos
        self.setToolTipsVisible(True)
        self.addAction(_("Check All")).triggered.connect(self.check_all_requested)
        self.addAction(_("Uncheck All")).triggered.connect(self.uncheck_all_requested)
        self.addAction(_("Invert Checked")).triggered.connect(self.invert_requested)
        self.addSeparator()

        for verb, label, hint in _SELECT_VERBS:
            self._build_criteria_menu(self.addMenu(_(label)), verb).setToolTip(tooltip(hint))
        self.addSeparator()

        self._tagged_menu = self.addMenu(_("Uncheck Already Tagged"))
        self._tagged_menu.setToolTip(
            tooltip(
                "Uncheck photos that already have the chosen metadata (in the image or its XMP "
                "sidecar), e.g. 'a title and a description' to skip those while keeping "
                "keyword-only photos. Mirrors the CLI's --skip-tagged. It reads what the Tagged "
                "column shows, so it waits for the scan under the list to reach the end.",
            ),
        )
        for text, criterion in _TAGGED_PRESETS:
            action = self._tagged_menu.addAction(_(text))
            action.triggered.connect(
                lambda _checked=False, chosen=criterion: self.tagged_requested.emit(chosen),
            )

        from_file = self.addAction(_("Uncheck From Skip List..."))
        from_file.triggered.connect(self.skip_list_requested)
        from_file.setToolTip(
            tooltip(
                "Uncheck photos whose filename or full path is listed in a text file (one per "
                "line), like the CLI's --skip-from.",
            ),
        )

    def _build_criteria_menu(self, menu: QMenu, verb: str) -> QMenu:
        """Fill one verb's submenu with the criteria it can act on, and return the menu."""
        menu.setToolTipsVisible(True)
        types = menu.addMenu(_("File Type"))
        types.setToolTip(tooltip("The file types actually in the list, with how many of each."))
        # Filled when the menu opens: which types are in the list changes with every photo added.
        types.aboutToShow.connect(lambda: self._fill_file_type_menu(types, verb))
        status = menu.addMenu(_("Status"))
        status.setToolTip(tooltip("Where each photo is in the generate-and-save cycle."))
        for criterion, label in _SELECT_STATUSES:
            status.addAction(
                _(label),
                lambda crit=criterion, text=label: self.matching_requested.emit(
                    lambda item: photo_matches_filter(item, crit),
                    verb,
                    _(text),
                ),
            )
        pattern = menu.addAction(
            _("Name Pattern..."),
            lambda: self.pattern_requested.emit(verb),
        )
        pattern.setToolTip(
            tooltip("Match filenames against text or a glob, e.g. IMG_ or *_edit.jpg."),
        )
        return menu

    def _fill_file_type_menu(self, menu: QMenu, verb: str) -> None:
        """Rebuild *menu* from the file types currently in the list, most common first."""
        menu.clear()
        counts = extension_counts(self._photos())
        if not counts:
            menu.addAction(_("No photos in the list")).setEnabled(False)
            return
        for extension, count in counts:
            label = extension or _("no extension")
            menu.addAction(
                f"{label} ({count})",
                lambda ext=extension, name=label: self.matching_requested.emit(
                    lambda item: matches_extension(item, ext),
                    verb,
                    name,
                ),
            )
