# mypy: ignore-errors
"""
PySide6 desktop frontend for photo-tagger.

A review-before-write workflow over the same pipeline the CLI uses:

1. Drag photos or folders onto the window (or use *Add files/folder*). They appear in a
   checkable, nested tree on the left; uncheck or remove anything you do not want.
2. *Generate* runs the vision model on the checked photos on a background thread,
   streaming each proposal back through Qt signals (no pipeline code is Qt-aware).
3. Click a photo to see its preview, its existing title/description/keywords, and the
   proposed values in editable fields, then *Save* writes them with ExifTool.

Requires the optional ``[gui]`` extra (``pip install 'photo-tagger[gui]'``). The
``photo-tagger gui`` command imports this module lazily, so the base CLI never depends
on Qt. The Qt-free logic lives in :mod:`photo_tagger.gui_state`; this file is the widget
and event-loop shell and is excluded from coverage and the static analyzers.
"""

import html
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger
from PySide6.QtCore import QObject, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from photo_tagger import __version__
from photo_tagger.ai import analyze_image_with_ai, create_agent
from photo_tagger.cli_options import load_defaults
from photo_tagger.config import DEFAULT_USER_PROMPT
from photo_tagger.diagnostics import run_checks
from photo_tagger.discovery import load_skip_list, skip_list_matches
from photo_tagger.errors import DiscoveryError, PhotoTaggerError, ProviderError
from photo_tagger.gui_state import (
    ADDED,
    DEFAULT_GUI_EXTENSIONS,
    FAILED,
    PENDING,
    PROVIDER_LABELS,
    READY,
    REMOVED,
    SAVED,
    WORKING,
    FolderNode,
    PhotoItem,
    Proposal,
    apply_proposal,
    build_tree,
    deselect_paths,
    expand_inputs,
    format_existing_keywords,
    hierarchy_preview,
    keyword_diff,
    keywords_to_save,
    keywords_to_text,
    new_paths,
    parse_keyword_lines,
    paths_matching_fields,
    paths_under,
    rank_vision_models,
    status_sort_rank,
    status_summary,
)
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.logging_setup import setup_logging
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    build_contextual_prompt,
    find_field_presence,
    read_caption,
    read_image_context,
    read_metadata_sources,
    write_metadata,
)
from photo_tagger.models import KeywordSet
from photo_tagger.providers import PROVIDER_NAMES, ProviderName, get_backend


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent


_RESOURCES = Path(__file__).parent / "resources"
# A stable, cwd-independent place for the GUI's logs. The CLI defaults to ./logs, but a windowed
# app has no meaningful working directory (it may be launched from Finder with cwd "/"), so the
# logs live under the user's home where the "Open logs" button can always find them.
_LOG_FOLDER = Path.home() / ".photo-tagger" / "logs"
_PREVIEW_MAX = 640
_THUMB_MAX = 200  # pixels for the grid thumbnails the model never sees
_THUMB_SIZE = 160  # icon box in the grid
_GENERATE_RETRIES = 2
_PATH_ROLE = Qt.ItemDataRole.UserRole
_IS_DIR_ROLE = Qt.ItemDataRole.UserRole + 1
_STATUS_RANK_ROLE = Qt.ItemDataRole.UserRole + 2  # lifecycle rank for sorting the Status column
_PAGE_DETAIL = 0  # right-pane stack index for one photo's detail
_PAGE_GRID = 1  # right-pane stack index for a folder's thumbnail grid
_DIR_MARK = "dir"  # truthy sentinel stored on folder tree items; files leave the role unset
_NONE = "(none)"  # placeholder shown when a photo has no existing title/description/keywords

# Short status word shown in the tree's second column.
_STATUS_LABEL = {
    PENDING: "",
    WORKING: "working...",
    READY: "ready",
    SAVED: "saved ✓",
    FAILED: "failed ✗",
}

# The field-aware "deselect already-tagged" menu, mirroring the CLI's --skip-tagged but letting
# the user pick which fields count as "done". Each entry is (menu label, required fields, whether
# ALL must be present, status-bar phrase). "Any metadata" is the broad OR criterion (the original
# skip-tagged); the rest require all of their fields, so a keyword-only photo survives "a title and
# a description" and stays selected for title/description generation.
_TAGGED_PRESETS: tuple[tuple[str, frozenset[str], bool, str], ...] = (
    (
        "Has any metadata",
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}),
        False,
        "any metadata",
    ),
    ("Has a title", frozenset({FIELD_TITLE}), True, "a title"),
    ("Has a description", frozenset({FIELD_DESCRIPTION}), True, "a description"),
    (
        "Has a title and a description",
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION}),
        True,
        "a title and a description",
    ),
    ("Has keywords", frozenset({FIELD_KEYWORDS}), True, "keywords"),
)

# Theme-agnostic polish: only spacing/rounding plus the brand accent on primary actions and
# the preview area. Colors for text and input backgrounds are left to the OS palette, so the
# window stays readable in both light and dark mode (hardcoding a light background would put
# the palette's light text on white in dark mode).
_STYLESHEET = """
QWidget { font-size: 13px; }
QPushButton {
    padding: 6px 12px; border-radius: 6px;
    border: 1px solid rgba(130, 130, 140, 60%);
    background: rgba(130, 130, 140, 14%);
}
QPushButton:hover { background: rgba(130, 130, 140, 26%); }
QPushButton:pressed { background: rgba(130, 130, 140, 36%); }
QPushButton:disabled { color: rgba(130, 130, 140, 70%); border-color: rgba(130, 130, 140, 25%); }
QPushButton#primary {
    background: #6366f1; color: white; border: 1px solid #6366f1; font-weight: 600;
}
QPushButton#primary:hover { background: #4f46e5; border-color: #4f46e5; }
QPushButton#primary:disabled {
    background: #9aa0e8; border-color: #9aa0e8; color: #eaeaff;
}
QLineEdit, QPlainTextEdit, QComboBox { padding: 4px 6px; border-radius: 5px; }
QTreeWidget::item { padding: 2px; }
QLabel#preview { background: #1f1f24; border-radius: 8px; color: #9a9aa5; }
QLabel#hint, QLabel#status { color: #8a8a8a; }
QLabel#section { font-weight: 600; }
QLabel#error {
    background: rgba(248, 81, 73, 18%); color: #f85149;
    border: 1px solid rgba(248, 81, 73, 45%); border-radius: 6px; padding: 8px;
}
"""


def _app_icon() -> QIcon:
    """Load the bundled app icon, or an empty icon if it is not present."""
    icon_path = _RESOURCES / "icon.svg"
    return QIcon(str(icon_path)) if icon_path.exists() else QIcon()


def _readonly_box(min_height: int) -> QPlainTextEdit:
    """Return a read-only, scrollable text box for displaying existing metadata."""
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setMinimumHeight(min_height)
    return box


class GenerateWorker(QObject):
    """
    Generates AI proposals for a list of photos off the UI thread.

    For each photo it reads the existing metadata, builds the same contextual prompt as the CLI,
    runs the model, and emits a :class:`~photo_tagger.gui_state.Proposal`. All communication with
    the window is via Qt signals; widgets are never touched here.
    """

    started = Signal(int)
    file_done = Signal(object)  # Proposal
    file_failed = Signal(str, str)  # path, error message
    finished = Signal()

    def __init__(
        self,
        provider: ProviderName,
        model: str,
        api_base_url: str | None,
        paths: list[Path],
        api_key: str | None = None,
    ) -> None:
        """Store the run parameters; nothing happens until :meth:`run`."""
        super().__init__()
        self._provider = provider
        self._model = model
        self._api_base_url = api_base_url
        self._paths = paths
        self._api_key = api_key

    def run(self) -> None:
        """Build the agent once, then generate a proposal per photo."""
        try:
            agent = create_agent(
                self._provider,
                self._model,
                api_base_url=self._api_base_url,
                api_key=self._api_key,
                retries=_GENERATE_RETRIES,
            )
        except PhotoTaggerError as exc:
            for path in self._paths:
                self.file_failed.emit(str(path), str(exc))
            self.finished.emit()
            return

        self.started.emit(len(self._paths))
        for path in self._paths:
            try:
                proposal = self._generate_one(agent, path)
            except Exception as exc:  # noqa: BLE001
                # One photo's failure must not stop the rest of the batch.
                logger.exception("gui_generate_failed", file=path.name, error=str(exc))
                self.file_failed.emit(str(path), str(exc))
            else:
                self.file_done.emit(proposal)
        self.finished.emit()

    def _generate_one(self, agent: object, path: Path) -> Proposal:
        """Read existing metadata, run the model, and assemble a proposal for *path*."""
        context = read_image_context(path)
        existing_title, existing_description = read_caption(path)
        gps_info = {"position": context.gps_position} if context.gps_position else {}
        prompt = build_contextual_prompt(
            DEFAULT_USER_PROMPT,
            context.existing_keywords.subject,
            context.location_tags,
            gps_info,
            camera_info=context.camera_info,
        )
        jpeg = prepare_image_for_agent(path, max_size=_PREVIEW_MAX)
        inference = analyze_image_with_ai(image_bytes=jpeg, agent=agent, user_prompt=prompt)
        return Proposal(
            path=path,
            existing_title=existing_title,
            existing_description=existing_description,
            existing_keywords=context.existing_keywords,
            title=inference.title,
            description=inference.description,
            keywords=list(inference.keywords),
        )


class ThumbnailWorker(QObject):
    """
    Decodes grid thumbnails off the UI thread.

    Each thumbnail is a small JPEG (the same RAW-aware loader the model uses, at a tiny size). The
    bytes are emitted back to the main thread, which builds the QPixmap there (QPixmap must not be
    created off the GUI thread). :meth:`stop` lets the window abandon a folder's load when the user
    navigates away.
    """

    ready = Signal(str, bytes)  # path, JPEG bytes
    finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        """Store the paths to decode; nothing runs until :meth:`run`."""
        super().__init__()
        self._paths = paths
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to stop before the next thumbnail."""
        self._stop = True

    def run(self) -> None:
        """Decode each thumbnail and emit its bytes, until done or stopped."""
        for path in self._paths:
            if self._stop:
                break
            try:
                content = prepare_image_for_agent(path, max_size=_THUMB_MAX)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gui_thumbnail_failed", file=path.name, error=str(exc))
                continue
            self.ready.emit(str(path), bytes(content.data))
        self.finished.emit()


def _status_sort_key(item: QTreeWidgetItem) -> tuple[int, str]:
    """Sort key for the Status column: lifecycle rank first, then name as a stable tiebreak."""
    rank = item.data(1, _STATUS_RANK_ROLE)
    return (int(rank) if rank is not None else 0, item.text(0).casefold())


class _SortableTreeItem(QTreeWidgetItem):  # NOSONAR S8500 - Qt sorts items via __lt__ only
    """
    A tree row that sorts sensibly when the user clicks a column header.

    Folders stay grouped above files whichever way the sort runs; the Photos column sorts by name
    (case-insensitive) and the Status column by lifecycle rank rather than the raw label.

    Only ``__lt__`` is overridden: Qt drives item sorting entirely through it, and the comparison is
    context-dependent (it follows the active sort column and direction), so a real total ordering or
    ``functools.total_ordering`` would be wrong here. Hence the S8500 suppression on the class.
    """

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        self_dir = bool(self.data(0, _IS_DIR_ROLE))
        if self_dir != bool(other.data(0, _IS_DIR_ROLE)):
            # Keep folders above files in both directions: Qt reverses the result for a
            # descending sort, so invert there to cancel that out.
            ascending = (
                tree is None or tree.header().sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
            )
            return self_dir if ascending else not self_dir
        if column == 1:
            return _status_sort_key(self) < _status_sort_key(other)
        return self.text(0).casefold() < other.text(0).casefold()


class MainWindow(QMainWindow):
    """The main window: a toolbar, a checkable file tree, and an editable detail pane."""

    def __init__(self) -> None:
        """Build the widgets, enable drag-and-drop, and wire the actions."""
        super().__init__()
        self._defaults = load_defaults()
        self._items: dict[str, PhotoItem] = {}
        self._preview_cache: dict[str, QPixmap] = {}
        self._thumb_cache: dict[str, QPixmap] = {}
        self._grid_items: dict[str, QListWidgetItem] = {}
        self._current: PhotoItem | None = None
        self._thread: QThread | None = None
        self._worker: GenerateWorker | None = None
        self._thumb_thread: QThread | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._syncing = False

        self.setWindowTitle(f"Photo Tagger {__version__}")
        self.setWindowIcon(_app_icon())
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self._placeholder_icon = _make_placeholder()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addLayout(self._build_toolbar())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_tree_panel())
        splitter.addWidget(self._build_right_pane())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, stretch=1)

        status_row = QHBoxLayout()
        self._status = QLabel("Drag photos or folders here to begin.")
        self._status.setObjectName("status")
        logs_button = QPushButton("Open logs")
        logs_button.setToolTip(f"Open the log folder ({_LOG_FOLDER}) in your file browser.")
        logs_button.clicked.connect(self._open_logs)
        status_row.addWidget(self._status, stretch=1)
        status_row.addWidget(logs_button)
        layout.addLayout(status_row)
        self._show_detail(enabled=False)

    # --- construction ----------------------------------------------------------------------

    def _build_toolbar(self) -> QVBoxLayout:
        # Two rows: connection settings on top, actions below, so neither gets cramped.
        bar = QVBoxLayout()
        bar.addLayout(self._build_connection_row())
        bar.addLayout(self._build_action_row())
        return bar

    def _build_connection_row(self) -> QHBoxLayout:
        provider = self._defaults.provider

        self._provider = QComboBox()
        for name in PROVIDER_NAMES:
            self._provider.addItem(PROVIDER_LABELS.get(name, name), name)
        self._provider.setCurrentIndex(max(0, list(PROVIDER_NAMES).index(provider.provider_name)))
        # Size to the widest label so "LM Studio" is not clipped.
        self._provider.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._provider.setMinimumContentsLength(10)
        self._provider.setToolTip("Backend that serves the vision-language model.")

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setMinimumWidth(220)
        self._model.setCurrentText(provider.model_name)
        self._model.setToolTip(
            "Model identifier. Type it, or press Refresh to list what the provider serves.",
        )
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Query the provider for the models it currently serves.")
        refresh.clicked.connect(self._refresh_models)

        self._url = QLineEdit(provider.api_base_url or "")
        self._url.setPlaceholderText("(provider default URL)")
        self._url.setToolTip("Provider API base URL. Leave blank to use the provider's default.")

        # Pre-filled from a config-file key if one is set, never from an environment variable: an
        # env key stays in the environment and is resolved at call time, so it never lands in the
        # widget. A typed key is masked, used only for this session, and never written to disk.
        self._api_key = QLineEdit(provider.api_key or "")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setClearButtonEnabled(True)
        self._api_key.setMinimumWidth(170)
        self._api_key.setPlaceholderText("(uses provider env var)")
        self._api_key.setToolTip(
            "API key for the provider. Leave blank to use the provider's environment variable "
            "(OPENAI_API_KEY, LM_STUDIO_API_KEY, or OLLAMA_API_KEY). Required for OpenAI. A typed "
            "key is used for this session only and is never written to disk.",
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("Provider"))
        row.addWidget(self._provider)
        row.addWidget(QLabel("Model"))
        row.addWidget(self._model)
        row.addWidget(refresh)
        row.addWidget(QLabel("URL"))
        row.addWidget(self._url, stretch=1)
        row.addWidget(QLabel("API key"))
        row.addWidget(self._api_key)
        return row

    def _build_action_row(self) -> QHBoxLayout:
        self._test_button = QPushButton("Test connection")
        self._test_button.setToolTip("Check ExifTool and that the provider serves the model.")
        self._test_button.clicked.connect(self._test_connection)
        self._retry_button = QPushButton("Retry failed")
        self._retry_button.setToolTip("Re-run the model on every photo that failed to generate.")
        self._retry_button.clicked.connect(self._retry_failed)
        self._generate_button = QPushButton("Generate selected")
        self._generate_button.setObjectName("primary")
        self._generate_button.setToolTip("Run the model on the checked photos.")
        self._generate_button.clicked.connect(self._generate)

        row = QHBoxLayout()
        row.addWidget(self._test_button)
        row.addWidget(self._retry_button)
        row.addStretch(1)
        row.addWidget(self._generate_button)
        return row

    def _build_tree_panel(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.addLayout(self._build_tree_controls())

        options = QHBoxLayout()
        self._extensions = QLineEdit(DEFAULT_GUI_EXTENSIONS)
        self._extensions.setToolTip(
            "Extensions to scan for in folders (comma-separated).\n"
            "Case-insensitive: jpg matches .JPG. Note jpeg is separate from jpg.",
        )
        self._recursive = QCheckBox("Recurse")
        self._recursive.setChecked(True)
        self._recursive.setToolTip("Descend into subfolders when adding a folder.")
        options.addWidget(QLabel("File types"))
        options.addWidget(self._extensions, stretch=1)
        options.addWidget(self._recursive)
        box.addLayout(options)
        box.addLayout(self._build_skip_controls())

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Photos", "Status"])
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self._tree.setColumnWidth(0, 320)
        self._tree.setColumnWidth(1, 90)
        # Click a header to sort by name (Photos) or status; folders stay grouped above files.
        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        header.setToolTip("Click a column header to sort by name or status.")
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        for key in (QKeySequence.StandardKey.Delete, QKeySequence(Qt.Key.Key_Backspace)):
            shortcut = QShortcut(key, self._tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._remove_selected)
        box.addWidget(self._tree, stretch=1)

        hint = QLabel("Drag photos or folders here. Select one and press Delete to remove it.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        box.addWidget(hint)
        return panel

    def _build_tree_controls(self) -> QHBoxLayout:
        controls = QHBoxLayout()
        for label, tip, slot in (
            ("Add files...", "Add individual photos.", self._choose_files),
            ("Add folder...", "Add a folder of photos.", self._choose_folder),
            ("Remove", "Remove the selected folder or photo from the list.", self._remove_selected),
            ("Clear", "Remove every photo from the list.", self._clear),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            controls.addWidget(button)
        controls.addStretch(1)
        return controls

    def _build_skip_controls(self) -> QHBoxLayout:
        """One-click filters that uncheck photos in bulk, mirroring the CLI's skip flags."""
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Deselect"))

        tagged = QPushButton("Already tagged")
        tagged.setToolTip(
            "Uncheck photos that already have the chosen metadata (in the image or its XMP "
            "sidecar). Pick which fields count from the menu, e.g. 'a title and a description' "
            "to skip those while keeping keyword-only photos. Mirrors the CLI's --skip-tagged.",
        )
        menu = QMenu(tagged)
        for text, required, match_all, phrase in _TAGGED_PRESETS:
            action = menu.addAction(text)
            action.triggered.connect(
                lambda _checked=False, req=required, all_=match_all, ph=phrase: (
                    self._deselect_tagged(
                        req,
                        match_all=all_,
                        phrase=ph,
                    )
                ),
            )
        tagged.setMenu(menu)
        controls.addWidget(tagged)

        from_file = QPushButton("From file...")
        from_file.setToolTip(
            "Uncheck photos whose filename or full path is listed in a text file (one per "
            "line), like the CLI's --skip-from.",
        )
        from_file.clicked.connect(self._deselect_from_file)
        controls.addWidget(from_file)

        controls.addStretch(1)
        return controls

    def _build_right_pane(self) -> QWidget:
        """Build a stack showing either one photo's detail or a folder's thumbnail grid."""
        self._right = QStackedWidget()
        self._right.addWidget(self._build_detail_panel())  # _PAGE_DETAIL
        self._right.addWidget(self._build_grid())  # _PAGE_GRID
        return self._right

    def _build_grid(self) -> QListWidget:
        grid = QListWidget()
        grid.setViewMode(QListView.ViewMode.IconMode)
        grid.setResizeMode(QListView.ResizeMode.Adjust)
        grid.setMovement(QListView.Movement.Static)
        grid.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        grid.setGridSize(QSize(_THUMB_SIZE + 24, _THUMB_SIZE + 40))
        grid.setSpacing(8)
        grid.setUniformItemSizes(True)
        grid.setWordWrap(True)
        grid.itemClicked.connect(self._on_thumb_activated)
        self._grid = grid
        return grid

    def _build_detail_panel(self) -> QWidget:
        content = QWidget()
        box = QVBoxLayout(content)
        # Shown only when the selected photo failed to generate: carries the reason and a hint
        # that "Open logs" has the full traceback. Hidden for healthy photos.
        self._error_banner = QLabel()
        self._error_banner.setObjectName("error")
        self._error_banner.setWordWrap(True)
        self._error_banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._error_banner.hide()
        box.addWidget(self._error_banner)
        self._preview = QLabel("Select a photo to preview it.")
        self._preview.setObjectName("preview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(240)
        box.addWidget(self._preview)
        box.addLayout(self._build_compare_grid())
        box.addLayout(self._build_diff_form())
        box.addLayout(self._build_save_row())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def _build_compare_grid(self) -> QGridLayout:
        """Existing (read-only) and New (editable) columns side by side for easy comparison."""
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.addWidget(self._section_label("Existing"), 0, 1)
        grid.addWidget(self._section_label("New (editable)"), 0, 2)

        self._existing_source = QLineEdit()
        self._existing_source.setReadOnly(True)
        self._existing_source.setToolTip(
            "Where the existing metadata was read from: the image file, an XMP sidecar, or both.",
        )
        grid.addWidget(QLabel("Source"), 1, 0)
        grid.addWidget(self._existing_source, 1, 1)

        self._existing_title = QLineEdit()
        self._existing_title.setReadOnly(True)
        self._title = QLineEdit()
        self._title.setToolTip("The title to write. Edit freely before saving.")
        grid.addWidget(QLabel("Title"), 2, 0)
        grid.addWidget(self._existing_title, 2, 1)
        grid.addWidget(self._title, 2, 2)

        top = Qt.AlignmentFlag.AlignTop
        self._existing_description = _readonly_box(70)
        self._description = QPlainTextEdit()
        self._description.setMinimumHeight(70)
        self._description.setToolTip("The description to write.")
        grid.addWidget(QLabel("Description"), 3, 0, top)
        grid.addWidget(self._existing_description, 3, 1)
        grid.addWidget(self._description, 3, 2)

        self._existing_keywords = _readonly_box(150)
        self._keywords = QPlainTextEdit()
        self._keywords.setMinimumHeight(150)
        self._keywords.setPlaceholderText("One per line. Use < for hierarchy (Duck<Bird<Animal)")
        self._keywords.setToolTip(
            "Keywords to write, one per line. Use '<' for a hierarchy "
            "(e.g. 'Duck<Bird<Animal'); the changes and resulting paths show below.",
        )
        self._keywords.textChanged.connect(self._refresh_derived)
        grid.addWidget(QLabel("Keywords"), 4, 0, top)
        grid.addWidget(self._existing_keywords, 4, 1)
        grid.addWidget(self._keywords, 4, 2)
        return grid

    def _build_diff_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._diff = QTextEdit()
        self._diff.setReadOnly(True)
        self._diff.setMinimumHeight(90)
        self._diff.setToolTip("Keyword changes a save will make: green added, red removed.")
        self._hierarchy = _readonly_box(60)
        self._hierarchy.setToolTip("Lightroom hierarchy paths that saving will write.")
        form.addRow("Keyword changes", self._diff)
        form.addRow("Hierarchy", self._hierarchy)
        return form

    def _build_write_fields_row(self) -> QHBoxLayout:
        """Checkboxes choosing which fields a save writes, like the CLI's --no-write-* flags."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Write"))
        self._write_title = QCheckBox("Title")
        self._write_title.setToolTip("Write the title. Uncheck to leave the existing title as is.")
        self._write_description = QCheckBox("Description")
        self._write_description.setToolTip(
            "Write the description. Uncheck to leave the existing description as is.",
        )
        self._write_keywords = QCheckBox("Keywords")
        self._write_keywords.setToolTip(
            "Write keywords. Uncheck to leave existing keywords untouched, e.g. to refresh only "
            "the title and description.",
        )
        for checkbox in (self._write_title, self._write_description, self._write_keywords):
            checkbox.setChecked(True)
            row.addWidget(checkbox)
        # Connect only after setChecked above, so building the row does not fire the handler
        # before _overwrite (which it toggles) has been created further down.
        self._write_keywords.toggled.connect(self._on_write_keywords_toggled)
        row.addStretch(1)
        return row

    def _build_save_row(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.addLayout(self._build_write_fields_row())

        toggles = QHBoxLayout()
        self._overwrite = QCheckBox("Overwrite existing keywords")
        self._overwrite.setToolTip("Replace existing keywords instead of merging the new ones in.")
        self._overwrite.toggled.connect(self._refresh_derived)
        self._embed = QCheckBox("Embed in photo")
        self._embed.setToolTip("Write into the image file instead of an XMP sidecar.")
        toggles.addWidget(self._overwrite)
        toggles.addWidget(self._embed)
        toggles.addStretch(1)
        box.addLayout(toggles)

        buttons = QHBoxLayout()
        self._generate_one_button = QPushButton("Generate this photo")
        self._generate_one_button.setToolTip(
            "Run the model on just this photo, regardless of which photos are checked.",
        )
        self._generate_one_button.clicked.connect(self._generate_current)
        self._save_button = QPushButton("Save this photo")
        self._save_button.setToolTip("Write the checked fields (see the Write row) to this photo.")
        self._save_button.clicked.connect(self._save_current)
        self._save_selected_button = QPushButton("Save selected")
        self._save_selected_button.setObjectName("primary")
        self._save_selected_button.setToolTip(
            "Write the checked photos that have a generated proposal, using the toggles above.",
        )
        self._save_selected_button.clicked.connect(self._save_selected)
        buttons.addWidget(self._generate_one_button)
        buttons.addWidget(self._save_button)
        buttons.addStretch(1)
        buttons.addWidget(self._save_selected_button)
        box.addLayout(buttons)
        return box

    # --- drag and drop ---------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override.
        """Accept a drag that carries file/folder URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override.
        """Expand dropped files/folders and add the resulting photos to the tree."""
        dropped = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        if dropped:
            self._add_inputs(dropped)

    # --- adding, removing, listing photos --------------------------------------------------

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos")
        if files:
            self._add_inputs([Path(f) for f in files])

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of photos")
        if folder:
            self._add_inputs([Path(folder)])

    def _add_inputs(self, paths: list[Path]) -> None:
        found = expand_inputs(
            paths,
            self._extensions.text().strip(),
            recursive=self._recursive.isChecked(),
        )
        fresh = new_paths([Path(p) for p in self._items], found)
        if not fresh:
            self._update_status()
            return
        for path in fresh:
            self._items[str(path)] = PhotoItem(path=path)
        self._rebuild_tree()
        self._update_status()

    def _remove_selected(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        path = item.data(0, _PATH_ROLE)
        is_dir = bool(item.data(0, _IS_DIR_ROLE))
        if path is None:
            return
        if is_dir:
            prefix = Path(path)
            removed = [k for k in self._items if Path(k).is_relative_to(prefix)]
        else:
            removed = [path]
        for key in removed:
            self._items.pop(key, None)
            self._preview_cache.pop(key, None)
            if self._current is not None and str(self._current.path) == key:
                self._current = None
                self._show_detail(enabled=False)
        self._rebuild_tree()
        self._update_status()

    def _clear(self) -> None:
        if self._thread is not None:
            return
        self._stop_thumbs()
        self._items.clear()
        self._preview_cache.clear()
        self._thumb_cache.clear()
        self._grid.clear()
        self._grid_items = {}
        self._current = None
        self._rebuild_tree()
        self._show_detail(enabled=False)
        self._right.setCurrentIndex(_PAGE_DETAIL)
        self._status.setText("Drag photos or folders here to begin.")

    def _deselect(self, paths: set[Path]) -> int:
        """Uncheck the matched photos and refresh the tree; return how many changed."""
        changed = deselect_paths(self._items, paths)
        if changed:
            self._rebuild_tree()
        return changed

    def _deselect_tagged(self, required: frozenset[str], *, match_all: bool, phrase: str) -> None:
        """Uncheck photos that already carry the chosen field(s); *phrase* names the criterion."""
        if not self._items:
            self._status.setText("Add photos before deselecting.")
            return
        # One batched exiftool read, like the CLI's --skip-tagged. A wait cursor covers the
        # brief pause, the same way opening a photo's metadata does.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            presence = find_field_presence([item.path for item in self._items.values()])
        finally:
            QApplication.restoreOverrideCursor()
        matched = paths_matching_fields(presence, set(required), match_all=match_all)
        changed = self._deselect(matched)
        if changed:
            self._status.setText(
                f"Deselected {changed} photo(s) with {phrase}; "
                f"{self._selected_count()} still selected.",
            )
        else:
            self._status.setText(f"No checked photos have {phrase}.")

    def _deselect_from_file(self) -> None:
        """Pick a skip-list file and uncheck the photos it names."""
        if not self._items:
            self._status.setText("Add photos before deselecting.")
            return
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a skip-list file")
        if chosen:
            self._apply_skip_file(Path(chosen))

    def _apply_skip_file(self, skip_file: Path) -> None:
        """Uncheck every photo whose name or path is listed in *skip_file*."""
        try:
            entries = load_skip_list(skip_file)
        except DiscoveryError as exc:
            QMessageBox.warning(self, "Could not read the skip list", str(exc))
            return
        if not entries:
            # The file read fine but had nothing usable (empty, blank lines, or only comments).
            # Say so, rather than the ambiguous "Deselected 0" a real no-match would also show.
            self._status.setText("That skip list had no usable entries (empty or only comments).")
            return
        matched = skip_list_matches([item.path for item in self._items.values()], entries)
        changed = self._deselect(matched)
        if changed:
            self._status.setText(
                f"Deselected {changed} photo(s) from the skip list; "
                f"{self._selected_count()} still selected.",
            )
        else:
            self._status.setText("No photos in the list matched the skip list.")

    def _rebuild_tree(self) -> None:
        self._syncing = True
        # Build with sorting off so items do not shuffle on every insert; re-enabling at the
        # end re-applies whatever column/direction the header is currently set to.
        self._tree.setSortingEnabled(False)
        self._tree.clear()
        for node in build_tree([item.path for item in self._items.values()]):
            self._add_folder_node(self._tree, node)
        self._tree.setSortingEnabled(True)
        self._syncing = False

    def _add_folder_node(self, parent: object, node: FolderNode) -> None:
        folder_item = _SortableTreeItem(parent, [node.label, ""])
        folder_item.setData(0, _PATH_ROLE, str(node.path))
        folder_item.setData(0, _IS_DIR_ROLE, _DIR_MARK)
        folder_item.setFlags(folder_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        folder_item.setExpanded(True)
        for sub in node.folders:
            self._add_folder_node(folder_item, sub)
        for path in node.files:
            item = self._items[str(path)]
            leaf = _SortableTreeItem(folder_item, [path.name, _STATUS_LABEL[item.status]])
            leaf.setData(0, _PATH_ROLE, str(path))
            leaf.setData(1, _STATUS_RANK_ROLE, status_sort_rank(item.status))
            # Files leave _IS_DIR_ROLE unset (None), which reads as "not a folder".
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, _checked(item.selected))
        self._sync_folder_check(folder_item)

    # --- tree interaction ------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        self._syncing = True
        if bool(item.data(0, _IS_DIR_ROLE)):
            self._set_descendants_checked(item, item.checkState(0))
        else:
            path = item.data(0, _PATH_ROLE)
            if path is not None:
                self._items[path].selected = item.checkState(0) == Qt.CheckState.Checked
            parent = item.parent()
            while parent is not None:
                self._sync_folder_check(parent)
                parent = parent.parent()
        self._syncing = False
        self._update_status()

    def _set_descendants_checked(self, folder_item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for index in range(folder_item.childCount()):
            child = folder_item.child(index)
            child.setCheckState(0, state)
            if bool(child.data(0, _IS_DIR_ROLE)):
                self._set_descendants_checked(child, state)
            else:
                path = child.data(0, _PATH_ROLE)
                if path is not None:
                    self._items[path].selected = state == Qt.CheckState.Checked

    def _sync_folder_check(self, folder_item: QTreeWidgetItem) -> None:
        states = {folder_item.child(i).checkState(0) for i in range(folder_item.childCount())}
        if states == {Qt.CheckState.Checked}:
            folder_item.setCheckState(0, Qt.CheckState.Checked)
        elif states == {Qt.CheckState.Unchecked}:
            folder_item.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            folder_item.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def _on_current_changed(self, current: QTreeWidgetItem | None, _previous: object) -> None:
        path = current.data(0, _PATH_ROLE) if current is not None else None
        if current is not None and bool(current.data(0, _IS_DIR_ROLE)) and path is not None:
            # A folder: show its thumbnail grid instead of a single photo's detail.
            self._current = None
            self._show_detail(enabled=False)
            self._show_grid(Path(path))
            return
        self._stop_thumbs()
        self._right.setCurrentIndex(_PAGE_DETAIL)
        if path is None:
            self._current = None
            self._show_detail(enabled=False)
            return
        self._current = self._items[path]
        self._show_item(self._items[path])

    # --- folder thumbnail grid -------------------------------------------------------------

    def _show_grid(self, folder: Path) -> None:
        self._stop_thumbs()
        self._grid.clear()
        self._grid_items = {}
        under = paths_under([item.path for item in self._items.values()], folder)
        pending: list[Path] = []
        for path in under:
            key = str(path)
            cached = self._thumb_cache.get(key)
            icon = QIcon(cached) if cached is not None else self._placeholder_icon
            grid_item = QListWidgetItem(icon, path.name)
            grid_item.setData(_PATH_ROLE, key)
            self._grid.addItem(grid_item)
            self._grid_items[key] = grid_item
            if cached is None:
                pending.append(path)
        self._right.setCurrentIndex(_PAGE_GRID)
        self._status.setText(f"{len(under)} photo(s) in {folder.name or folder}.")
        if pending:
            self._start_thumbs(pending)

    def _on_thumb_activated(self, item: QListWidgetItem) -> None:
        key = item.data(_PATH_ROLE)
        leaf = self._leaf_for(Path(key))
        if leaf is not None:
            self._tree.setCurrentItem(leaf)  # routes to the detail page

    def _on_thumb_ready(self, path: str, data: bytes) -> None:
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        self._thumb_cache[path] = pixmap
        grid_item = self._grid_items.get(path)
        if grid_item is not None:
            grid_item.setIcon(QIcon(pixmap))

    def _start_thumbs(self, paths: list[Path]) -> None:
        self._thumb_thread = QThread(self)
        self._thumb_worker = ThumbnailWorker(paths)
        self._thumb_worker.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.ready.connect(self._on_thumb_ready)
        self._thumb_thread.start()

    def _stop_thumbs(self) -> None:
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
        if self._thumb_thread is not None:
            self._thumb_thread.quit()
            self._thumb_thread.wait()
            self._thumb_thread = None
        self._thumb_worker = None

    # --- detail pane -----------------------------------------------------------------------

    def _show_item(self, item: PhotoItem) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._ensure_loaded(item)
            self._ensure_sources(item)
            self._render_preview(item)
        finally:
            QApplication.restoreOverrideCursor()
        self._existing_source.setText(", ".join(item.existing_sources) or _NONE)
        self._existing_title.setText(item.existing_title or _NONE)
        self._existing_description.setPlainText(item.existing_description or _NONE)
        existing_kw = format_existing_keywords(item.existing_keywords)
        self._existing_keywords.setPlainText(existing_kw or _NONE)
        self._title.setText(item.title)
        self._description.setPlainText(item.description)
        self._keywords.setPlainText(keywords_to_text(item.keywords))
        self._show_detail(enabled=True)
        self._update_error_banner(item)
        self._refresh_derived()

    def _update_error_banner(self, item: PhotoItem) -> None:
        """Show the failure reason for a failed photo; hide the banner otherwise."""
        if item.status == FAILED and item.error:
            self._error_banner.setText(
                f"Generation failed: {item.error}\n"
                "Use 'Retry failed' to try again, or 'Open logs' for the full traceback.",
            )
            self._error_banner.show()
        else:
            self._error_banner.hide()

    def _ensure_loaded(self, item: PhotoItem) -> None:
        if item.loaded:
            return
        title, description = read_caption(item.path)
        context = read_image_context(item.path)
        item.existing_title = title
        item.existing_description = description
        item.existing_keywords = context.existing_keywords
        item.loaded = True
        if not item.has_proposal:
            # Seed the editable copy from the existing values so a file can be edited
            # and saved even without generating a proposal first.
            item.title = title or ""
            item.description = description or ""
            item.keywords = list(context.existing_keywords.subject)

    def _ensure_sources(self, item: PhotoItem) -> None:
        if item.sources_read:
            return
        item.existing_sources = read_metadata_sources(item.path)
        item.sources_read = True

    def _render_preview(self, item: PhotoItem) -> None:
        pixmap = self._preview_pixmap(item)
        if pixmap is None or pixmap.isNull():
            self._preview.setText("(no preview available)")
            return
        scaled = pixmap.scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def _preview_pixmap(self, item: PhotoItem) -> QPixmap | None:
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

    def _on_write_keywords_toggled(self) -> None:
        """Overwrite-vs-merge only matters when keywords are written; gray it out otherwise."""
        self._overwrite.setEnabled(self._write_keywords.isChecked())
        self._refresh_derived()

    def _refresh_derived(self) -> None:
        """Recompute the keyword-change diff and hierarchy preview from the edited fields."""
        if self._current is None:
            return
        if not self._write_keywords.isChecked():
            # Keywords are not being written, so the diff and hierarchy do not apply.
            self._diff.setHtml("(keywords will not be written)")
            self._hierarchy.setPlainText(_NONE)
            return
        edited = parse_keyword_lines(self._keywords.toPlainText())
        overwrite = self._overwrite.isChecked()
        existing = self._current.existing_keywords
        paths = hierarchy_preview(existing, edited, overwrite=overwrite)
        self._hierarchy.setPlainText(paths or _NONE)
        self._diff.setHtml(_diff_html(existing, edited, overwrite=overwrite))

    def _show_detail(self, *, enabled: bool) -> None:
        for widget in (
            self._title,
            self._description,
            self._keywords,
            self._overwrite,
            self._embed,
            self._save_button,
            self._generate_one_button,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self._error_banner.hide()

    def _commit_current(self) -> None:
        """Copy the visible editable fields back onto the selected item."""
        item = self._current
        if item is None:
            return
        item.title = self._title.text().strip()
        item.description = self._description.toPlainText().strip()
        item.keywords = parse_keyword_lines(self._keywords.toPlainText())

    def _write_fields_chosen(self) -> bool:
        """Report whether at least one write toggle (Title/Description/Keywords) is on."""
        return (
            self._write_title.isChecked()
            or self._write_description.isChecked()
            or self._write_keywords.isChecked()
        )

    def _write_item(self, item: PhotoItem) -> bool:
        """
        Write the item's checked fields to disk; return success.

        Unchecked fields stay as is.
        """
        keywords = (
            keywords_to_save(
                item.existing_keywords,
                item.keywords,
                overwrite=self._overwrite.isChecked(),
            )
            if self._write_keywords.isChecked()
            else KeywordSet()
        )
        ok = write_metadata(
            item.path,
            keywords,
            description=(item.description or None) if self._write_description.isChecked() else None,
            title=(item.title or None) if self._write_title.isChecked() else None,
            use_sidecar=not self._embed.isChecked(),
        )
        item.status = SAVED if ok else FAILED
        self._refresh_status_cell(item)
        return ok

    def _save_current(self) -> None:
        if self._current is None:
            return
        if not self._write_fields_chosen():
            self._status.setText(
                "Pick at least one field to write (Title, Description, or Keywords).",
            )
            return
        self._commit_current()
        ok = self._write_item(self._current)
        name = self._current.path.name
        self._status.setText(f"Saved {name}." if ok else f"Failed to save {name}.")
        self._resort()
        self._update_status()

    def _save_selected(self) -> None:
        if not self._write_fields_chosen():
            self._status.setText(
                "Pick at least one field to write (Title, Description, or Keywords).",
            )
            return
        self._commit_current()
        targets = [item for item in self._items.values() if item.selected and item.has_proposal]
        if not targets:
            self._status.setText("No checked photos have a proposal to save.")
            return
        saved = sum(int(self._write_item(item)) for item in targets)
        self._status.setText(f"Saved {saved} of {len(targets)} checked photo(s).")
        self._resort()
        self._update_status()

    # --- generation ------------------------------------------------------------------------

    def _generate(self) -> None:
        selected = [item for item in self._items.values() if item.selected]
        if not selected:
            self._status.setText("Check at least one photo first.")
            return
        self._run_generation(selected)

    def _generate_current(self) -> None:
        if self._current is None:
            self._status.setText("Open a photo to generate it.")
            return
        self._run_generation([self._current])

    def _retry_failed(self) -> None:
        failed = [item for item in self._items.values() if item.status == FAILED]
        if not failed:
            self._status.setText("No failed photos to retry.")
            return
        self._run_generation(failed)

    def _run_generation(self, items: list[PhotoItem]) -> None:
        if self._thread is not None or not items:
            return
        for item in items:
            item.status = WORKING
            self._refresh_status_cell(item)
        current = self._current
        if current is not None and current in items:
            # Clear a stale failure banner the moment its photo is re-queued.
            self._update_error_banner(current)
        self._set_running(running=True)
        self._status.setText(f"Generating {len(items)} photo(s)...")

        self._thread = QThread(self)
        self._worker = GenerateWorker(
            self._provider_name(),
            self._model.currentText().strip(),
            self._url.text().strip() or None,
            [item.path for item in items],
            api_key=self._api_key_value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.finished.connect(self._on_generate_finished)
        self._thread.start()

    def _on_file_done(self, proposal: Proposal) -> None:
        item = self._items.get(str(proposal.path))
        if item is None:
            return
        apply_proposal(item, proposal)
        self._refresh_status_cell(item)
        if self._current is item:
            self._show_item(item)
        self._update_status()

    def _on_file_failed(self, path: str, message: str) -> None:
        item = self._items.get(path)
        if item is None:
            return
        item.status = FAILED
        item.error = message
        self._refresh_status_cell(item)
        if self._current is item:
            self._update_error_banner(item)
        self._update_status()

    def _on_generate_finished(self) -> None:
        self._status.setText("Generation finished.")
        self._resort()
        self._teardown_thread()

    def _set_running(self, *, running: bool) -> None:
        self._generate_button.setEnabled(not running)
        self._generate_one_button.setEnabled(not running)
        self._retry_button.setEnabled(not running)
        self._test_button.setEnabled(not running)

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._set_running(running=False)

    # --- providers and diagnostics ---------------------------------------------------------

    def _refresh_models(self) -> None:
        backend = get_backend(self._provider_name())
        base_url = self._url.text().strip() or backend.default_base_url
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            models = backend.list_models(base_url, backend.resolve_api_key(self._api_key_value()))
        except ProviderError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not list models", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        current = self._model.currentText()
        self._model.clear()
        self._model.addItems(rank_vision_models(models))
        self._model.setCurrentText(current)
        self._status.setText(f"Found {len(models)} model(s) on {self._provider_name()}.")

    def _test_connection(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = run_checks(
                self._provider_name(),
                self._model.currentText().strip(),
                api_base_url=self._url.text().strip() or None,
                api_key=self._api_key_value(),
            )
        finally:
            QApplication.restoreOverrideCursor()
        lines = [f"{'OK  ' if r.ok else 'FAIL'}  {r.name}: {r.detail}" for r in results]
        box = QMessageBox(self)
        box.setWindowTitle("Connection check")
        box.setText("\n".join(lines))
        all_ok = all(r.ok for r in results)
        box.setIcon(QMessageBox.Icon.Information if all_ok else QMessageBox.Icon.Warning)
        box.exec()

    def _open_logs(self) -> None:
        """Reveal the log folder in the OS file browser so the user can read the run logs."""
        _LOG_FOLDER.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(_LOG_FOLDER)))

    # --- helpers ---------------------------------------------------------------------------

    def _provider_name(self) -> ProviderName:
        """Return the selected provider; the combo carries the internal name as item data."""
        return cast("ProviderName", self._provider.currentData())

    def _api_key_value(self) -> str | None:
        """Return the typed API key, or None to fall back to the provider's env var/default."""
        return self._api_key.text().strip() or None

    def _refresh_status_cell(self, item: PhotoItem) -> None:
        leaf = self._leaf_for(item.path)
        if leaf is not None:
            leaf.setText(1, _STATUS_LABEL[item.status])
            leaf.setData(1, _STATUS_RANK_ROLE, status_sort_rank(item.status))
            # Surface the failure reason on hover so it is discoverable straight from the tree.
            tip = item.error if item.status == FAILED else ""
            leaf.setToolTip(1, tip)

    def _resort(self) -> None:
        """Re-apply the active sort so changed statuses settle when sorting by the Status column."""
        header = self._tree.header()
        self._tree.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())

    def _leaf_for(self, path: Path) -> QTreeWidgetItem | None:
        target = str(path)
        iterator = QTreeWidgetItemIterator(self._tree)
        while iterator.value():
            item = iterator.value()
            if not bool(item.data(0, _IS_DIR_ROLE)) and item.data(0, _PATH_ROLE) == target:
                return item
            iterator += 1
        return None

    def _selected_count(self) -> int:
        """How many photos are currently checked (used in deselect feedback)."""
        return sum(1 for item in self._items.values() if item.selected)

    def _update_status(self) -> None:
        if self._items:
            self._status.setText(status_summary(self._items.values()))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override.
        """
        Wait for any in-flight generation to finish before closing.

        Generation has no cancellation hook, so closing mid-run blocks until the current batch
        completes. Waiting keeps a running QThread from being destroyed.
        """
        self._stop_thumbs()
        self._teardown_thread()
        super().closeEvent(event)


def _make_placeholder() -> QIcon:
    """Build a neutral grey tile shown in the grid until a thumbnail loads."""
    pixmap = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
    pixmap.fill(QColor(50, 50, 56))
    return QIcon(pixmap)


def _checked(selected: bool) -> Qt.CheckState:  # noqa: FBT001 - tiny private bool mapper.
    """Map a selected flag to a Qt check state."""
    return Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked


_DIFF_STYLE = {
    ADDED: ("color:#3fb950", "+&nbsp;"),
    REMOVED: ("color:#f85149;text-decoration:line-through", "&minus;&nbsp;"),
}


def _diff_html(existing: KeywordSet, edited_keywords: list[str], *, overwrite: bool) -> str:
    """Render the keyword diff as HTML: green added, red struck-through removed, grey kept."""
    rows: list[str] = []
    for keyword, state in keyword_diff(existing, edited_keywords, overwrite=overwrite):
        safe = html.escape(keyword)
        style, marker = _DIFF_STYLE.get(state, ("color:#8a8a8a", "&nbsp;&nbsp;&nbsp;"))
        rows.append(f'<span style="{style}">{marker}{safe}</span>')
    return "<br>".join(rows) or "(no change)"


def launch(argv: list[str] | None = None) -> int:
    """Create the application, show the main window, and run the event loop."""
    # File-only logging: the window carries the live status, so the terminal stays quiet, but a
    # durable log (with full tracebacks for failed photos) is written for the "Open logs" button.
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=_LOG_FOLDER)
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Photo Tagger")
    app.setApplicationDisplayName("Photo Tagger")
    app.setWindowIcon(_app_icon())
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()
