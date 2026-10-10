# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The detail pane: one photo's preview, its metadata beside the proposal, and its buttons."""

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.common import NONE_LABEL
from photo_tagger.gui.viewer import ImageViewerDialog
from photo_tagger.gui.widgets import (
    ClickableLabel,
    blurb,
    busy_cursor,
    fit_text_height,
    hint_label,
    readonly_box,
)
from photo_tagger.gui_state import (
    ADDED,
    FAILED,
    REMOVED,
    caption_source_note,
    foreign_terms_note,
    format_existing_keywords,
    hierarchy_preview,
    keeps_copyright,
    keyword_diff,
    keyword_diff_html,
    keywords_to_text,
    parse_keyword_lines,
    planned_copyright,
    save_targets,
    source_label,
    tooltip,
)
from photo_tagger.gui_usage import TRIGGER_HINT, TRIGGER_SINGLE
from photo_tagger.i18n import _
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.metadata import FIELD_DESCRIPTION, FIELD_TITLE


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable

    from PySide6.QtGui import QAction

    from photo_tagger.gui_state import PhotoItem, SaveOptions


# The on-screen preview in the detail pane. Not what the model sees: that is
# [inference] jpeg_dimensions, read into MainWindow._model_image_size.
_PREVIEW_MAX = 640
# What the full-size viewer decodes to. Large enough that zooming in shows real detail (more than
# a 5K display can put up at once), small enough that a RAW file still opens in about a second.
_VIEWER_MAX = 3200


class DetailPane(QScrollArea):
    """
    Review one photo: its preview, what it carries now beside what a save would write, and why.

    The Existing column is read-only and the New column is editable, row by row, so the two compare
    at a glance. Under them, the keyword changes and the hierarchy a save would write, and the
    photo's own Generate and Save buttons. The pane shows the photo the window hands it with
    :meth:`show_photo` until :meth:`close_photo`, and copies edits back with :meth:`commit`.

    The buttons ask the window for the work with :attr:`generate_requested` and
    :attr:`save_requested`. The pane also reports each feature used with :attr:`used` and a status
    line with :attr:`status`.
    """

    generate_requested = Signal(bool, str)  # use the cache, what triggered it
    save_requested = Signal()
    used = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        *,
        save_options_menu: QMenu,
        autosave_action: QAction,
        save_options: Callable[[], SaveOptions],
    ) -> None:
        """
        Build the pane.

        *save_options_menu* and *autosave_action* are the window's own, shared with its other Save
        and Generate buttons. *save_options* snapshots what a save would write, which the keyword
        preview and the Copyright row both describe.
        """
        super().__init__(parent)
        self._save_options = save_options
        self._item: PhotoItem | None = None
        # Keyed by path and kept for the session: small pixmaps, and a folder of them is cheap.
        self._preview_cache: dict[str, QPixmap] = {}
        # One entry only: at _VIEWER_MAX a pixmap is tens of megabytes, and the last one is enough
        # to reopen the photo just looked at instantly without growing over a long session.
        self._viewer_cache: tuple[str, QPixmap] | None = None

        content = QWidget()
        box = QVBoxLayout(content)
        # Shown only when the selected photo failed to generate: carries the reason and a hint
        # that "Open logs" has the full traceback. Hidden for healthy photos.
        self._error_banner = blurb(object_name="error")
        self._error_banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._error_banner.hide()
        box.addWidget(self._error_banner)
        self._preview = ClickableLabel(_("Select a photo to preview it."))
        self._preview.setObjectName("preview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(240)
        self._preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self._preview.setToolTip(
            tooltip("Click the photo to open it full size, where it can be zoomed and panned."),
        )
        self._preview.clicked.connect(self._open_viewer)
        box.addWidget(self._preview)
        box.addLayout(self._build_compare_grid())
        box.addLayout(self._build_details_section())
        box.addLayout(self._build_save_row(save_options_menu, autosave_action))

        self.setWidgetResizable(True)
        self.setWidget(content)

    def _build_compare_grid(self) -> QGridLayout:
        """Existing (read-only) and New (editable) columns side by side for easy comparison."""
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        # The metadata source rides along in the Existing header (it has no "New" counterpart,
        # so a full grid row of its own broke the columns' symmetry).
        existing_header = QHBoxLayout()
        existing_header.addWidget(_section_label(_("Existing")))
        self._existing_source = QLabel("")
        self._existing_source.setObjectName("hint")
        self._existing_source.setToolTip(
            tooltip(
                "Where the existing metadata was read from: the image file, an XMP sidecar, "
                "or both.",
            ),
        )
        existing_header.addWidget(self._existing_source)
        existing_header.addStretch(1)
        grid.addLayout(existing_header, 0, 1)
        # The New header carries the review note the same way, so a warning about the generated
        # text sits over the fields it is about instead of somewhere else in the pane.
        new_header = QHBoxLayout()
        new_header.addWidget(_section_label(_("New (editable)")))
        self._review_note = QLabel("")
        self._review_note.setObjectName("warning")
        self._review_note.setToolTip(
            tooltip(
                "The generated text mixes writing systems. That is right when the photo shows "
                "that text, and a word the model failed to translate when it does not.",
            ),
        )
        new_header.addWidget(self._review_note)
        new_header.addStretch(1)
        grid.addLayout(new_header, 0, 2)

        self._existing_title = QLineEdit()
        self._existing_title.setReadOnly(True)
        self._title = QLineEdit()
        self._title.setToolTip(tooltip("The title to write. Edit freely before saving."))
        grid.addWidget(QLabel(_("Title")), 1, 0)
        grid.addWidget(self._existing_title, 1, 1)
        grid.addWidget(self._title, 1, 2)
        # A row of its own under the field, not a layout nested in its cell: nesting makes the grid
        # size the row by the box's sizeHint instead of its minimum, which stretches the row.
        self._title_source = _source_note()
        grid.addWidget(self._title_source, 2, 1)

        top = Qt.AlignmentFlag.AlignTop
        self._existing_description = readonly_box(44)
        self._description = QPlainTextEdit()
        self._description.setToolTip(tooltip("The description to write."))
        # Descriptions are usually a sentence or two; grow the boxes with the text instead of
        # reserving a fixed block of the pane (textChanged also fires on programmatic fills).
        self._description.textChanged.connect(lambda: fit_text_height(self._description))
        fit_text_height(self._description)
        grid.addWidget(QLabel(_("Description")), 3, 0, top)
        grid.addWidget(self._existing_description, 3, 1)
        grid.addWidget(self._description, 3, 2)
        self._description_source = _source_note()
        grid.addWidget(self._description_source, 4, 1)

        self._existing_keywords = readonly_box(150)
        self._keywords = QPlainTextEdit()
        self._keywords.setMinimumHeight(150)
        self._keywords.setPlaceholderText(_("One per line. Use < for hierarchy (Duck<Bird<Animal)"))
        self._keywords.setToolTip(
            tooltip(
                "Keywords to write, one per line. Use '<' for a hierarchy "
                "(e.g. 'Duck<Bird<Animal'); the changes and resulting paths show below.",
            ),
        )
        self._keywords.textChanged.connect(self.refresh_derived)
        grid.addWidget(QLabel(_("Keywords")), 5, 0, top)
        grid.addWidget(self._existing_keywords, 5, 1)
        grid.addWidget(self._keywords, 5, 2)
        self._add_copyright_row(grid, 6)
        return grid

    def _add_copyright_row(self, grid: QGridLayout, row: int) -> None:
        """
        Add the Copyright row: the notice on the file, and the one a save writes.

        Read-only on both sides: the notice is one setting for every photo, not a per-photo edit,
        so the New side only says what that setting puts on this one.
        """
        self._existing_copyright = QLineEdit()
        self._existing_copyright.setReadOnly(True)
        self._planned_copyright = QLineEdit()
        self._planned_copyright.setReadOnly(True)
        self._planned_copyright.setToolTip(
            tooltip(
                "The notice a save writes on this photo, dated by the year it was taken. Change "
                "it in Settings > Copyright.",
            ),
        )
        grid.addWidget(QLabel(_("Copyright")), row, 0)
        grid.addWidget(self._existing_copyright, row, 1)
        grid.addWidget(self._planned_copyright, row, 2)

    def _build_details_section(self) -> QVBoxLayout:
        """Collapsible keyword-change details: the diff and the resulting hierarchy paths."""
        box = QVBoxLayout()
        self._details_toggle = QToolButton()
        self._details_toggle.setText(_("Keyword changes"))
        self._details_toggle.setCheckable(True)
        self._details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._details_toggle.setToolTip(
            tooltip(
                "Show exactly what saving will change: added and removed keywords, plus the "
                "resulting keyword tree.",
            ),
        )
        self._details_toggle.toggled.connect(self._on_details_toggled)
        box.addWidget(self._details_toggle)

        self._details_panel = QWidget()
        form = QFormLayout(self._details_panel)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._diff = QTextEdit()
        self._diff.setReadOnly(True)
        self._diff.setMinimumHeight(90)
        self._diff.setToolTip(
            tooltip("Keyword changes a save will make: green added, red removed."),
        )
        self._hierarchy = readonly_box(60)
        self._hierarchy.setObjectName("tree")  # monospace, so the branch guides line up
        self._hierarchy.setToolTip(
            tooltip(
                "The keyword tree that saving will write (stored as Lightroom hierarchy paths).",
            ),
        )
        form.addRow(_("Changes"), self._diff)
        form.addRow(_("Tree"), self._hierarchy)
        self._details_panel.hide()
        box.addWidget(self._details_panel)
        return box

    def _on_details_toggled(self, expanded: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Expand or collapse the keyword-change details under the disclosure arrow."""
        if expanded:
            self.used.emit("keyword_changes")
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self._details_toggle.setArrowType(arrow)
        self._details_panel.setVisible(expanded)

    def _build_save_row(self, save_options_menu: QMenu, autosave_action: QAction) -> QHBoxLayout:
        """Per-photo actions at the bottom of the pane; the batch actions live in the window."""
        row = QHBoxLayout()
        hint_caption = QLabel(_("Hint for the AI"))
        self._hint = QLineEdit()
        self._hint.setPlaceholderText(_("e.g. 'The animal is a deer, not a boar'"))
        self._hint.setClearButtonEnabled(True)
        self._hint.setToolTip(
            tooltip(
                "A note about this photo that the model trusts over its own reading of the "
                "image; useful when it misidentifies the subject. It is sent along on every "
                "generation of this photo (a hinted photo skips the cached result) and is "
                "never written to the file. Press Enter to regenerate right away.",
            ),
        )
        hint_caption.setToolTip(self._hint.toolTip())
        self._hint.returnPressed.connect(
            lambda: self.generate_requested.emit(True, TRIGGER_HINT),  # noqa: FBT003 - use cache
        )
        row.addWidget(hint_caption)
        row.addWidget(self._hint, stretch=1)
        # The window keeps the label and the tooltip of both Generate buttons in step, so these
        # two, and the menu, are public.
        self.generate_button = QToolButton()
        self.generate_button.setObjectName("split")
        self.generate_button.setText(_("Generate This Photo"))
        self.generate_button.setToolTip(
            tooltip("Run the model on just this photo, regardless of which photos are checked."),
        )
        self.generate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.generate_button.clicked.connect(
            lambda: self.generate_requested.emit(True, TRIGGER_SINGLE),  # noqa: FBT003 - use cache
        )
        self.generate_menu = QMenu(self)
        self.generate_menu.setToolTipsVisible(True)
        skip_one = self.generate_menu.addAction(
            _("Generate This Photo (Skip Cache)"),
            lambda: self.generate_requested.emit(False, TRIGGER_SINGLE),  # noqa: FBT003 - no cache
        )
        skip_one.setToolTip(tooltip("One-time: call the model even when a cached result exists."))
        # The same QAction hangs in both Generate menus, so the two can never disagree about it,
        # and the save options follow it: what the toggle writes with is one click away from the
        # toggle rather than something to know about the other button.
        self.generate_menu.addSeparator()
        self.generate_menu.addAction(autosave_action)
        self.generate_menu.addMenu(save_options_menu)
        self.generate_button.setMenu(self.generate_menu)
        # The save options live on the button's own arrow, so what a save writes is
        # discoverable right where the save happens (both Save buttons share one menu).
        self.save_button = QToolButton()
        self.save_button.setObjectName("split")
        self.save_button.setText(_("Save This Photo"))
        self.save_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.save_button.clicked.connect(self.save_requested)
        self.save_button.setMenu(save_options_menu)
        row.addWidget(self.generate_button)
        row.addWidget(self.save_button)
        return row

    def input_widgets(self) -> tuple[QWidget, ...]:
        """Return the fields a review is typed into, for the window's input tracking."""
        return (self._title, self._description, self._keywords, self._hint)

    def show_photo(self, item: PhotoItem) -> None:
        """
        Fill the pane from *item*, whose metadata the window has already read, and enable it.

        The preview is drawn separately, by :meth:`render_preview`, so the window can keep the busy
        cursor up around the reads and the decode together.
        """
        self._item = item
        sources = ", ".join(source_label(source) for source in item.existing_sources)
        self._existing_source.setText(
            _("(from {sources})").format(sources=sources) if sources else _("(no metadata found)"),
        )
        self._existing_title.setText(item.existing_title or _(NONE_LABEL))
        self._existing_description.setPlainText(item.existing_description or _(NONE_LABEL))
        fit_text_height(self._existing_description)
        self._fill_source_notes(item)
        existing_kw = format_existing_keywords(item.existing_keywords)
        self._existing_keywords.setPlainText(existing_kw or _(NONE_LABEL))
        self._existing_copyright.setText(item.existing_copyright or _(NONE_LABEL))
        self.refresh_planned_copyright()
        self._title.setText(item.title)
        self._description.setPlainText(item.description)
        self._keywords.setPlainText(keywords_to_text(item.keywords))
        self._hint.setText(item.hint)
        self._review_note.setText(foreign_terms_note(item.foreign_terms))
        self._set_enabled(enabled=True)
        self.update_error(item)
        self.refresh_derived()

    def close_photo(self) -> None:
        """Stop showing a photo: gray the fields out and put the error banner away."""
        self._item = None
        self._set_enabled(enabled=False)

    def _set_enabled(self, *, enabled: bool) -> None:
        """Turn the editable fields and the photo's own buttons on or off."""
        for widget in (
            self._title,
            self._description,
            self._keywords,
            self._hint,
            self.save_button,
            self.generate_button,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self._error_banner.hide()

    def commit(self) -> None:
        """Copy the visible editable fields back onto the photo on show."""
        item = self._item
        if item is None:
            return
        item.title = self._title.text().strip()
        item.description = self._description.toPlainText().strip()
        item.keywords = parse_keyword_lines(self._keywords.toPlainText())
        item.hint = self._hint.text().strip()

    def refresh_review_note(self) -> None:
        """Say again which words mix writing systems, once the check ran on the edited text."""
        if self._item is not None:
            self._review_note.setText(foreign_terms_note(self._item.foreign_terms))

    def set_keywords(self, keywords: list[str]) -> None:
        """Put *keywords* in the editable field, for a change made elsewhere (harmonization)."""
        self._keywords.setPlainText(keywords_to_text(keywords))

    def _fill_source_notes(self, item: PhotoItem) -> None:
        """Say under each Existing caption where it was read from, and what it shadows."""
        for field_name, label in (
            (FIELD_TITLE, self._title_source),
            (FIELD_DESCRIPTION, self._description_source),
        ):
            note = caption_source_note(item.existing_caption_sources.get(field_name, []))
            label.setText(note)
            # A field no file carries has nothing to say, and an empty label would still take a
            # line of the pane.
            label.setVisible(bool(note))

    def update_error(self, item: PhotoItem) -> None:
        """Show the failure reason for a failed photo; hide the banner otherwise."""
        if item.status == FAILED and item.error:
            self._error_banner.setText(
                _(
                    "Generation failed: {error}\nUse 'Retry Failed' to try again, or "
                    "Help > Open Logs for the full traceback.",
                ).format(error=item.error),
            )
            self._error_banner.show()
        else:
            self._error_banner.hide()

    def refresh_derived(self) -> None:
        """Recompute the keyword-change diff and hierarchy preview from the edited fields."""
        if self._item is None:
            return
        options = self._save_options()
        if not options.write_keywords:
            # Keywords are not being written, so the diff and hierarchy do not apply.
            self._diff.setHtml(_("(keywords will not be written)"))
            self._hierarchy.setPlainText(_(NONE_LABEL))
            self._details_toggle.setText(_("Keyword changes (not written)"))
            return
        edited = parse_keyword_lines(self._keywords.toPlainText())
        overwrite = options.overwrite_keywords
        existing = self._item.existing_keywords
        verbatim = options.verbatim
        paths = hierarchy_preview(existing, edited, overwrite=overwrite, verbatim=verbatim)
        self._hierarchy.setPlainText(paths or _(NONE_LABEL))
        diff = keyword_diff(existing, edited, overwrite=overwrite, verbatim=verbatim)
        self._diff.setHtml(keyword_diff_html(diff))
        # A collapsed section still tells the user whether saving changes anything.
        added = sum(1 for _kw, state in diff if state == ADDED)
        removed = sum(1 for _kw, state in diff if state == REMOVED)
        summary = f"+{added} / -{removed}" if added or removed else _("no change")
        self._details_toggle.setText(_("Keyword changes ({summary})").format(summary=summary))

    def refresh_planned_copyright(self) -> None:
        """Say in the Copyright row what a save writes on the photo on show, if anything."""
        if self._item is None:
            return
        options = self._save_options()
        fields = options.extra_fields
        sources = self._item.field_sources
        targets = save_targets(options, self._item.path)
        if keeps_copyright(fields, sources, targets):
            self._planned_copyright.setText(_("(keeps the notice it has)"))
            return
        planned = planned_copyright(fields, sources, targets)
        self._planned_copyright.setText(planned or _("(not written)"))

    def render_preview(self, item: PhotoItem) -> None:
        """Draw *item*'s preview, scaled to the space the pane gives it."""
        pixmap = self._preview_pixmap(item)
        if pixmap is None or pixmap.isNull():
            self._preview.setText(_("(no preview available)"))
            return
        scaled = pixmap.scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def _preview_pixmap(self, item: PhotoItem) -> QPixmap | None:
        """Decode *item* at preview size, once per photo per session."""
        key = str(item.path)
        if key in self._preview_cache:
            return self._preview_cache[key]
        try:
            content = prepare_image_for_agent(item.path, max_size=_PREVIEW_MAX)
        except Exception as exc:  # noqa: BLE001
            # A preview must never crash the window; degrade to a placeholder.
            logger.warning("gui_preview_failed", file=item.path.name, error=str(exc))
            return None
        pixmap = QPixmap()
        pixmap.loadFromData(content.data)
        self._preview_cache[key] = pixmap
        return pixmap

    def forget(self, key: str) -> None:
        """Drop every decoded image of a photo that left the list."""
        self._preview_cache.pop(key, None)
        if self._viewer_cache is not None and self._viewer_cache[0] == key:
            self._viewer_cache = None

    def forget_all(self) -> None:
        """Drop every decoded image, for a list that was cleared."""
        self._preview_cache.clear()
        self._viewer_cache = None

    def _open_viewer(self) -> None:
        """Open the photo on show in a window of its own, big enough to zoom into."""
        self.used.emit("viewer")
        item = self._item
        if item is None:
            return
        # Decoding a RAW file at _VIEWER_MAX takes about a second, which is short enough to just
        # wait through and long enough that the window must say it is working.
        with busy_cursor():
            pixmap = self._viewer_pixmap(item)
        if pixmap is None or pixmap.isNull():
            self.status.emit(_("That photo cannot be opened for viewing."))
            return
        window = self.window()
        viewer = ImageViewerDialog(window, item.path, pixmap)
        viewer.resize(window.size())
        viewer.exec()

    def _viewer_pixmap(self, item: PhotoItem) -> QPixmap | None:
        """
        Decode *item* at viewer resolution, reusing the last one and falling back to the preview.

        A failure here is not worth an error dialog: the small preview is already on screen and good
        enough to look at, so the viewer opens on that rather than on nothing.
        """
        key = str(item.path)
        if self._viewer_cache is not None and self._viewer_cache[0] == key:
            return self._viewer_cache[1]
        try:
            content = prepare_image_for_agent(item.path, max_size=_VIEWER_MAX)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gui_viewer_decode_failed", file=item.path.name, error=str(exc))
            return self._preview_pixmap(item)
        pixmap = QPixmap()
        pixmap.loadFromData(content.data)
        self._viewer_cache = (key, pixmap)
        return pixmap


def _section_label(text: str) -> QLabel:
    """Build the bold heading over one column of the pane."""
    label = QLabel(text)
    label.setObjectName("section")
    return label


def _source_note() -> QLabel:
    """Build the dimmed line under an Existing field that says where its value came from."""
    label = hint_label("")
    label.setToolTip(
        tooltip(
            "Which file this value came from. A sidecar wins, so a value left in the photo is "
            "named rather than hidden: only a save to the photo itself replaces it.",
        ),
    )
    return label
