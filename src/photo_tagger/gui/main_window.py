# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""
The main window, laid out along the add -> generate -> review -> save workflow.

It owns the photo list and wires the widgets, the dialogs, and the background workers together.
"""

import subprocess  # nosec B404 - only used to reveal a photo in the OS file browser
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QRect, QSignalBlocker, QSize, Qt, QThread, QTimer, QUrl, qVersion

# QCloseEvent, QDragEnterEvent and QDropEvent are used only in annotations, but they are imported
# here at runtime on purpose: see the note above closeEvent. Moving them into the TYPE_CHECKING
# block below reopens a PySide crash.
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from photo_tagger import __version__, i18n, telemetry
from photo_tagger.cli_options import (
    known_config_keys,
    load_defaults,
    resolve_notify,
    to_pacing_plan,
)
from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE
from photo_tagger.config_file import (
    config_file_kind,
    find_config_file,
    load_config,
    user_config_path,
)
from photo_tagger.csv_report import write_report
from photo_tagger.diagnostics import run_checks
from photo_tagger.discovery import load_skip_list, skip_list_matches
from photo_tagger.errors import ConfigFileError, DiscoveryError, ProviderError
from photo_tagger.gui.common import (
    CHOOSE_LABEL,
    CLOSE_LABEL,
    COL_NAME,
    COL_STATUS,
    COL_TAGGED,
    COL_TYPE,
    CONFIG_SAVE_ERROR_TITLE,
    DEFAULT_CACHE_FILE,
    DIR_MARK,
    DOCS_URL,
    IS_DIR_ROLE,
    LOG_FOLDER,
    NONE_LABEL,
    PATH_ROLE,
    PICK_A_FIELD,
    STATUS_RANK_ROLE,
)
from photo_tagger.gui.copyright import CopyrightDialog
from photo_tagger.gui.keyword_rules import KeywordRulesDialog
from photo_tagger.gui.pacing import PacingDialog
from photo_tagger.gui.save_options import SaveOptionsMenu
from photo_tagger.gui.style import app_icon, svg_icon
from photo_tagger.gui.viewer import ImageViewerDialog
from photo_tagger.gui.vocabulary_builder import VocabularyBuilderDialog
from photo_tagger.gui.widgets import (
    ClickableLabel,
    InputWatcher,
    SortableTreeItem,
    check_state,
    fit_combo,
    fit_text_height,
    readonly_box,
    selection_modifiers_active,
)
from photo_tagger.gui.workers import (
    GenerateWorker,
    HarmonizeWorker,
    MetadataScanWorker,
    SaveWorker,
    ThumbnailWorker,
    WatchWorker,
)
from photo_tagger.gui_state import (
    ADDED,
    BADGE_COPYRIGHT,
    BADGE_FAILED,
    BADGE_METADATA,
    BADGE_MIXED_SCRIPT,
    BADGE_SAVED,
    BADGE_SIDECAR,
    BADGE_UNSAVED,
    DEFAULT_GUI_EXTENSIONS,
    EXTRA_PASSES,
    FAILED,
    FILTER_ALL,
    FILTER_FAILED,
    FILTER_GENERATED,
    FILTER_MIXED_SCRIPT,
    FILTER_NO_COPYRIGHT,
    FILTER_PENDING,
    FILTER_SAVED,
    FILTER_SELECTED,
    FILTER_UNTAGGED,
    OUTPUT_LANGUAGE_SUGGESTIONS,
    PENDING,
    READY,
    REMOVED,
    RERUN_FAILED,
    RERUN_MIXED_SCRIPT,
    SAVED,
    SCAN_FILTERS,
    SELECT_CHECK,
    SELECT_ONLY,
    SELECT_UNCHECK,
    SORT_NAME,
    SORT_STATUS,
    SORT_TAGGED,
    SORT_TYPE,
    UNDO_OK_ACTIONS,
    WORKING,
    Connection,
    FolderNode,
    GenerateOptions,
    GuiConfigValues,
    HarmonizeResult,
    Location,
    NavigationHistory,
    PhotoItem,
    Proposal,
    SaveJob,
    SaveOptions,
    SelectionChange,
    TaggedCriterion,
    WatchSettings,
    apply_proposal,
    apply_selection,
    apply_write_result,
    build_save_job,
    build_tree,
    caption_source_note,
    capture_year,
    check_results_html,
    config_text_with_language,
    config_text_with_output_language,
    config_toml_text,
    deselect_paths,
    expand_inputs,
    extension_counts,
    extra_pass_status,
    failure_summary,
    field_tags_to_read,
    file_dialog_name_filters,
    file_type_label,
    filter_photos,
    flat_labels,
    foreign_terms_note,
    format_existing_keywords,
    harmonize_summary,
    hierarchy_preview,
    journal_label,
    keeps_copyright,
    keys_by_check,
    keyword_diff,
    keyword_diff_html,
    keyword_lines,
    keywords_to_text,
    load_sidecar_overrides,
    location_crumb,
    mark_mixed_script,
    matches_extension,
    matches_name_pattern,
    merged_config_text,
    navigation_shortcuts,
    new_paths,
    parse_keyword_lines,
    paths_under,
    photo_item_to_report_row,
    photo_matches_filter,
    photos_for_pass,
    photos_to_save,
    photos_to_write,
    planned_copyright,
    progress_timing_text,
    rank_vision_models,
    record_dropped_terms,
    reports_status,
    resolve_extra_tags,
    reveal_command,
    reveal_label,
    review_summary,
    save_options_for,
    save_targets,
    search_summary,
    sort_photos,
    source_label,
    status_sort_rank,
    status_summary,
    tagged_legend,
    tagged_summary,
    tagged_tooltip,
    telemetry_toggle_note,
    thumb_badges,
    tooltip,
    undo_details_text,
    undo_summary,
    vocabulary_summary,
    watch_status_text,
    wrap_tooltip,
    writes_nothing,
)
from photo_tagger.gui_usage import (
    GUI_BASELINE,
    JOB_GENERATE,
    JOB_SAVE,
    TRIGGER_BATCH,
    TRIGGER_HINT,
    TRIGGER_RETRY,
    TRIGGER_SINGLE,
    TRIGGER_WATCH,
    GuiUsage,
    RunContext,
    RunEnd,
    RunTally,
    WindowFacts,
    batch_outcome,
    window_settings,
)
from photo_tagger.i18n import _, gettext_noop, ngettext
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    first_caption,
    read_caption_values,
    read_field_sources,
    read_image_context,
    read_metadata_sources,
    write_metadata_everywhere,
)
from photo_tagger.notify import APP_NAME, notify
from photo_tagger.pacing import SECONDS_PER_MINUTE
from photo_tagger.pipeline import FAILURE_OTHER
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES, ProviderName, get_backend
from photo_tagger.telemetry_facts import changed_settings, config_key_names, env_var_names
from photo_tagger.undo import UndoError, list_journals, open_journal, read_journal, undo_run
from photo_tagger.watch import DEFAULT_INTERVAL_SECONDS, DEFAULT_SETTLE_SECONDS


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable

    from photo_tagger.undo import UndoJournal, UndoResult, WriteRecord


# The on-screen preview in the detail pane. Not what the model sees: that is
# [inference] jpeg_dimensions, read into MainWindow._model_image_size.
_PREVIEW_MAX = 640
# What the full-size viewer decodes to. Large enough that zooming in shows real detail (more than
# a 5K display can put up at once), small enough that a RAW file still opens in about a second.
_VIEWER_MAX = 3200
_THUMB_SIZE = 160  # icon box in the grid

# How long a run that writes by itself waits before re-running the photos it failed on. The CLI
# pauses before its own retry pass for the same reason (`_RETRY_PASS_DELAY_SECONDS` in pipeline):
# a first-pass failure usually means the model server is overloaded or mid-restart, and asking
# again straight away just retries into the same outage.
_RETRY_PASS_DELAY_MS = 5000

# How often the elapsed/remaining readout is repainted during a run. Half a second reads as a live
# clock without being busywork.
_TIMING_TICK_MS = 500

# Status colors: failures pop red, saved photos settle green; other states use the palette.
_STATUS_COLOR = {FAILED: QColor("#f85149"), SAVED: QColor("#3fb950")}

_PAGE_EMPTY = 0  # right-pane stack index for the idle "add or pick a photo" placeholder
_PAGE_DETAIL = 1  # right-pane stack index for one photo's detail
_PAGE_GRID = 2  # right-pane stack index for a folder's thumbnail grid

# Right-pane placeholder copy. It adapts to the list: a getting-started nudge while empty, and a
# "pick a photo" nudge once photos are loaded but none is open. This is what fills the right pane
# when there is nothing to inspect, instead of an empty (and confusing) detail form.
_EMPTY_START = gettext_noop(
    "Add photos to get started.\n\n"
    "Drag photos or folders onto the window, or use the Add Photos button.",
)
_EMPTY_PICK = gettext_noop(
    "Select a photo to review it.\n\n"
    "Generate proposes a title, description, and keywords you can edit before saving.",
)

# Short status word shown in the tree's second column.
_STATUS_LABEL = {
    PENDING: "",
    WORKING: gettext_noop("working..."),
    READY: gettext_noop("ready"),
    SAVED: gettext_noop("saved ✓"),
    FAILED: gettext_noop("failed ✗"),
}

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

# Edge of the small drawn icons in the row above the photo list, in pixels.
_TOOLBAR_ICON = 14

# How long the search box waits after a keystroke before it re-filters the list. Long enough that
# typing a word rebuilds the tree once rather than once per letter, short enough to feel live.
_SEARCH_DEBOUNCE_MS = 150

# How the photo list is laid out, picked from the small view button above it. Each entry is
# (mode, menu label, tooltip). The wording follows what file browsers call these, rather than
# inventing our own.
VIEW_TREE = "tree"
VIEW_LIST = "list"

_VIEW_MODES: tuple[tuple[str, str, str], ...] = (
    (
        VIEW_TREE,
        gettext_noop("View as Tree"),
        gettext_noop("Group each photo under the folder it came from."),
    ),
    (
        VIEW_LIST,
        gettext_noop("View as List"),
        gettext_noop(
            "List every photo at one level, so clicking Status or Type sorts the whole list "
            "rather than each folder on its own. Folder rows (and their checkboxes) are gone "
            "while it is on; a photo's folder still opens from Go > Enclosing Folder.",
        ),
    ),
)

# The Select menu's three bulk verbs, each offering the same criteria below. Each entry is
# (verb, menu label, tooltip).
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

# Grace period _stop_scan gives an in-flight batched exiftool call to finish on its own before
# detaching rather than blocking the caller. Comfortably above a normal scan's duration.
_SCAN_STOP_TIMEOUT_MS = 3000
# How long a metadata scan must run before the window reports it. A local folder is read well
# inside this, so the usual case never shows a progress bar at all; a network share blows past it.
_SCAN_PROGRESS_DELAY_MS = 700
# The same grace period for the folder watcher, which reacts to a stop within a fraction of a
# second unless it is inside a slow poll (a huge folder, or a network share).
_WATCH_STOP_TIMEOUT_MS = 3000
# And for shoot harmonization, whose batched capture-time read has no stop hook either.
_HARMONIZE_STOP_TIMEOUT_MS = 3000
# Shorter for the grid thumbnails: _stop_thumbs runs on every navigation, so this is the longest
# the window may sit still when the user clicks away from a folder that is still loading. One
# decode is uninterruptible, and a big RAW takes seconds, so the wait usually times out.
_THUMB_STOP_TIMEOUT_MS = 250


class MainWindow(QMainWindow):
    """
    The main window, laid out along the add -> generate -> review -> save workflow.

    A one-row header picks the provider and model (URL/key live in a Connection dialog), the left
    panel holds the checkable file tree with its add/select controls, the right pane reviews one
    photo (or a folder grid), and the bottom bar carries the batch actions with a progress bar.
    """

    def __init__(self) -> None:
        """Build the widgets, enable drag-and-drop, and wire the actions."""
        super().__init__()
        # One disk read serves both views of the config: the Defaults dataclasses and the raw
        # dict (for keys whose CLI-oriented defaults the GUI overrides, like extensions).
        self._raw_config = load_config()
        self._defaults = load_defaults(self._raw_config)
        self._items: dict[str, PhotoItem] = {}
        self._grid_items: dict[str, QListWidgetItem] = {}
        self._init_pixmap_caches()
        self._init_tree_row_index()
        self._init_grid_view_state()
        self._init_navigation()
        self._init_worker_state()
        self._syncing = False
        self._grid_check_toggled = False
        self._closing = False
        # The persisted UI language choice; "auto" means follow the OS locale. Changing it in the
        # Settings menu rewrites this key and takes effect on the next launch.
        self._language = str(self._raw_config.get("language", i18n.AUTO))
        # The language the model writes titles, descriptions, and keywords in. Distinct from the
        # UI language above; persisted under [inference] output_language, which the CLI shares.
        self._output_language = self._defaults.inference.output_language
        self._cache_file = self._defaults.artifacts.cache_file or DEFAULT_CACHE_FILE
        # What this window does, for its telemetry: runs, saves, features, busy and idle time.
        self._usage = GuiUsage(time.monotonic())
        # The photos of the run in flight, kept apart from _generated_keys, which a chained save
        # empties before the run is over.
        self._run_keys: list[str] = []
        # Telemetry on/off: a persisted Settings-menu choice wins over the config-file default, and
        # a toggle turned on by hand is an opt-in that overrules DO_NOT_TRACK.
        self._telemetry_enabled, self._telemetry_opted_in = telemetry.gui_choice(
            config_enabled=self._defaults.telemetry.enabled,
        )
        # End-of-run desktop alert, resolved the same way the CLI resolves its two flags: a config
        # that asks only for the sound is asking for the notification that carries it.
        self._notify_enabled, self._notify_sound = resolve_notify(self._defaults.display)
        # The size the model sees, from [inference] jpeg_dimensions. The window used to hardcode
        # its preview size here, which silently ignored the configured value and gave CLI and GUI
        # runs different answers (and separate cache namespaces) for the same photo.
        self._model_image_size = self._defaults.inference.jpeg_dimensions
        self._init_keyword_rules()

        self.setWindowTitle(f"Photo Tagger {__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self._placeholder_pixmap = _make_placeholder()
        # Built before the panes: both Save buttons (detail pane and bottom bar) attach it.
        self._save_options_menu = SaveOptionsMenu(
            self,
            self._defaults.output,
            sidecar_overrides=self._sidecar_overrides,
        )
        self._save_options_menu.changed.connect(self._on_save_options_changed)
        # Same here, for the one action both Generate buttons offer in their arrow menus.
        self._autosave_action = self._build_autosave_action()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addLayout(self._build_header())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_tree_panel())
        splitter.addWidget(self._build_right_pane())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        # Start with the left pane wide enough for all four tree columns.
        splitter.setSizes([500, 680])
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(self._build_bottom_bar())
        self._build_dialogs()
        self._build_menus()
        self._refresh_save_tooltips()
        self._show_empty()
        self._watch_activity()

    def _build_dialogs(self) -> None:
        """
        Build the dialogs the menus open, once each.

        Called after the panes, because these read the header and folder-scan widgets when they
        open. They are kept alive for the window's lifetime rather than rebuilt per use, so what was
        typed in one is still there the next time it opens.
        """
        self._keyword_rules_dialog = self._build_keyword_rules_dialog()
        self._copyright_dialog = CopyrightDialog(
            self,
            self._defaults.output,
            preview_year=self._preview_year,
        )
        self._copyright_dialog.changed.connect(self._on_copyright_changed)
        self._on_copyright_changed()
        # How hard a run may push the machine, from the config file's [pacing] table.
        self._pacing_dialog = PacingDialog(self, to_pacing_plan(self._defaults.pacing))
        self._builder_dialog = VocabularyBuilderDialog(
            self,
            photo_paths=self._item_paths,
            connection=self._connection,
            output_language=lambda: self._output_language,
        )
        self._builder_dialog.status.connect(self._status.setText)
        self._builder_dialog.used.connect(self._count)
        self._builder_dialog.use_vocabulary.connect(self._keyword_rules_dialog.load_vocabulary)
        self._undo_dialog = self._build_undo_dialog()
        self._watch_dialog = self._build_watch_dialog()

    def _init_worker_state(self) -> None:
        """
        Seed the handles for every background job the window runs.

        Each job has its own thread so one never waits on another: generation, saving, grid
        thumbnails, the metadata scan, shoot harmonization, and the watcher. The vocabulary build
        belongs to its dialog.
        """
        self._thread: QThread | None = None
        self._worker: GenerateWorker | None = None
        self._cancelling = False
        self._save_thread: QThread | None = None
        self._save_worker: SaveWorker | None = None
        # The batch being written, keyed by path: the done handler needs each job's fields to
        # update the Tagged column without re-reading the file.
        self._save_jobs: dict[str, SaveJob] = {}
        self._saved_ok = 0
        # Keys of the photos the last generation run covered, so a save it chains writes those and
        # leaves every other proposal for review. Emptied once that save has them.
        self._generated_keys: list[str] = []
        # Which of EXTRA_PASSES this batch has already had (each is worth one go, no more), and
        # the one currently waiting out the pause before it starts.
        self._passes_done: set[str] = set()
        self._pending_pass: str | None = None
        self._batch_uses_cache = True
        # Set by Cancel between the chain's steps, so the next step ends the chain instead.
        self._chain_cancelled = False
        # What a chained save adds to its closing line: the vocabulary and review counts the
        # generation reported, which nobody read if the run was left alone.
        self._save_note = ""
        # The undo journal the batch in flight is recording into. None while idle, or when
        # recording is switched off in Settings.
        self._save_journal: UndoJournal | None = None
        # The one journal every photo-by-photo save of this session shares, opened on the first.
        self._session_journal: UndoJournal | None = None
        # When the run in flight started, for the elapsed/remaining readout. None while idle.
        self._run_started: float | None = None
        self._thumb_thread: QThread | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._scan_thread: QThread | None = None
        self._scan_worker: MetadataScanWorker | None = None
        # The "Uncheck Already Tagged" pick waiting for the scan to cover every photo, and whether
        # the user stopped that scan. Both None/False while no one is waiting on a scan.
        self._scan_pending: TaggedCriterion | None = None
        self._scan_cancelled = False
        # The photos the running scan was handed, and the ones a finished scan could not read (see
        # _on_scan_finished): those are not handed out again until they leave the list.
        self._scan_batch: set[str] = set()
        self._scan_tried: set[str] = set()
        self._harmonize_thread: QThread | None = None
        self._harmonize_worker: HarmonizeWorker | None = None
        self._watch_thread: QThread | None = None
        self._watch_worker: WatchWorker | None = None
        # What the running watch was started with, and what it has picked up. None while idle.
        self._watch_settings: WatchSettings | None = None
        self._watch_added = 0
        # Photos the watch added while a run was already in flight, generated when it frees up.
        self._watch_pending: list[str] = []
        # The status line a run owns while it works, restored after every pacing rest.
        self._run_status_text = ""

    def _init_keyword_rules(self) -> None:
        """
        Seed the settings that decide which keywords a run ends up writing.

        The vocabulary and the session gap live in the Keyword Rules dialog; this records the rest
        of what was configured.
        """
        output = self._defaults.output
        self._repair_language = self._defaults.inference.repair_language
        self._sidecar_overrides, overrides_error = load_sidecar_overrides(output.sidecar_overrides)
        if overrides_error:
            # A config the CLI would refuse must not take the window down with it.
            logger.warning("gui_sidecar_overrides_unusable", error=overrides_error)
        # What the vocabulary did over the run in flight, for its closing summary line.
        self._vocabulary_mapped = 0
        self._vocabulary_dropped: dict[str, int] = {}

    def _init_pixmap_caches(self) -> None:
        """
        Seed the decoded-image caches, one per size the window draws photos at.

        The preview and thumbnail caches are keyed by path and live for the session: both hold small
        pixmaps, and a folder of them is cheap. The viewer's is deliberately a single entry, because
        at ``_VIEWER_MAX`` one pixmap is tens of megabytes; keeping the last one is enough to reopen
        the photo just looked at instantly, without growing over a long session.
        """
        self._preview_cache: dict[str, QPixmap] = {}
        self._thumb_cache: dict[str, QPixmap] = {}
        self._viewer_cache: tuple[str, QPixmap] | None = None

    def _init_tree_row_index(self) -> None:
        """
        Seed the path -> tree row lookup the whole window reads.

        Walking the tree for a row is not an option: the only widget-side walker,
        QTreeWidgetItemIterator, is never destroyed by PySide, so each walk leaves an iterator
        registered with QTreeModel pointing at whatever row it stopped on. The next teardown frees
        that row and the removal after it dereferences the dangling pointer. An index also turns the
        per-photo status refresh from a full walk into a dict hit.

        _rebuild_tree owns both dicts and drops them before it empties the tree, so no row is
        reachable from Python while Qt is tearing it down.
        """
        self._folder_rows: dict[str, QTreeWidgetItem] = {}
        self._leaf_rows: dict[str, QTreeWidgetItem] = {}

    def _init_navigation(self) -> None:
        """
        Seed what the right-hand pane is showing and the trail of where it has been.

        The Back/Forward actions are created here, before the panes: the nav row above the pane
        attaches them, and the Go menu (built last, with the rest of the menu bar) adds them again
        so they get their keyboard shortcuts and a discoverable home.
        """
        self._current: PhotoItem | None = None
        self._history = NavigationHistory()
        back_key, forward_key, up_key = navigation_shortcuts(sys.platform)
        self._back_action = self._nav_action(_("Back"), back_key, self._go_back)
        self._forward_action = self._nav_action(_("Forward"), forward_key, self._go_forward)
        # The file manager's move up a level, and the fallback when nothing was visited yet: a
        # photo opened straight from the tree has no grid to go Back to.
        self._up_action = self._nav_action(_("Enclosing Folder"), up_key, self._go_up)

    def _nav_action(self, label: str, shortcut: str, handler: Callable[[], None]) -> QAction:
        """Build one navigation action, shown in the Go menu and driven by its shortcut."""
        action = QAction(label, self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(handler)
        return action

    def _init_grid_view_state(self) -> None:
        """
        Seed the folder grid's view state: which folder is shown and how it is filtered and sorted.

        Kept as session state (never written to disk), so the choices carry across folders but reset
        on relaunch. ``_grid_folder`` is what lets the filter/sort controls rebuild the grid.
        """
        self._grid_folder: Path | None = None
        self._grid_sort = SORT_NAME
        self._grid_sort_desc = False
        self._grid_filter = FILTER_ALL
        # The Select menu's last name pattern, so refining one is a small edit, not a retype.
        self._last_name_pattern = ""

    # --- construction ----------------------------------------------------------------------

    def _build_menus(self) -> None:
        """Build the menu bar: File actions, the Tools jobs, the Settings toggles, and Help."""
        menubar = self.menuBar()

        # Kept on self: QAction.menu() hands out a transient wrapper that shiboken may delete,
        # so tests (and future code) need a stable reference to the menu itself.
        file_menu = self._file_menu = menubar.addMenu(_("File"))
        file_menu.addAction(_("Add Photos..."), self._choose_files)
        file_menu.addAction(_("Add Folder..."), self._choose_folder)
        file_menu.addSeparator()
        export_action = file_menu.addAction(_("Export CSV Report..."), self._export_csv)
        export_action.setToolTip(
            tooltip(
                "Save a CSV report of every photo: generated and existing metadata, EXIF, and "
                "token usage.",
            ),
        )
        file_menu.addSeparator()
        file_menu.addAction(_("Clear List"), self._clear)
        file_menu.addSeparator()
        quit_action = file_menu.addAction(_("Quit"), self.close)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)

        self._build_go_menu(menubar)
        self._build_tools_menu(menubar)
        self._build_settings_menu(menubar)

        help_menu = self._help_menu = menubar.addMenu(_("Help"))
        help_menu.addAction(
            _("Documentation"),
            lambda: (self._count("docs"), QDesktopServices.openUrl(QUrl(DOCS_URL))),
        )
        help_menu.addSeparator()
        help_menu.addAction(_("Test Connection"), self._test_connection)
        help_menu.addAction(_("Open Logs"), self._open_logs)
        help_menu.addSeparator()
        about_action = help_menu.addAction(_("About Photo Tagger"), self._show_about)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)

    def _build_go_menu(self, menubar: QMenu) -> None:
        """
        Build the Go menu: the Back/Forward trail plus the jump up to the enclosing folder.

        The actions themselves were built in ``_init_navigation`` and are already on the nav row
        above the pane; adding them to a menu is what arms their keyboard shortcuts.
        """
        menu = self._go_menu = menubar.addMenu(_("Go"))
        menu.setToolTipsVisible(True)
        menu.addAction(self._back_action)
        menu.addAction(self._forward_action)
        menu.addSeparator()
        self._up_action.setToolTip(
            tooltip("Show the thumbnail grid of the folder holding the open photo."),
        )
        menu.addAction(self._up_action)

    def _build_settings_menu(self, menubar: QMenu) -> None:
        """Build the Settings menu: the toggles and the two languages, then the config actions."""
        settings_menu = self._settings_menu = menubar.addMenu(_("Settings"))
        settings_menu.setToolTipsVisible(True)
        self._cache_action = QAction(_("Cache AI Results"), self)
        self._cache_action.setCheckable(True)
        self._cache_action.setChecked(True)
        self._cache_action.setToolTip(
            tooltip(
                "Reuse earlier results for unchanged photos ({cache_file}). Uncheck to call "
                "the model again for everything; a single photo can skip the cache from its "
                "right-click menu.",
                cache_file=self._cache_file,
            ),
        )
        self._cache_action.toggled.connect(lambda _on: self._count("cache_toggle"))
        settings_menu.addAction(self._cache_action)
        self._telemetry_action = QAction(_("Send Anonymous Telemetry"), self)
        self._telemetry_action.setCheckable(True)
        self._refresh_telemetry_action()
        self._telemetry_action.toggled.connect(self._on_telemetry_toggled)
        settings_menu.addAction(self._telemetry_action)
        self._undo_log_action = QAction(_("Record Saves for Undo"), self)
        self._undo_log_action.setCheckable(True)
        self._undo_log_action.setChecked(self._defaults.artifacts.undo_log)
        self._undo_log_action.setToolTip(
            tooltip(
                "Record every file a save writes, so Tools > Undo Writes can put it back. On by "
                "default: the saves worth undoing are the ones nobody planned to.",
            ),
        )
        self._undo_log_action.toggled.connect(lambda _on: self._count("undo_log_toggle"))
        settings_menu.addAction(self._undo_log_action)
        self._build_repair_action(settings_menu)
        self._build_notify_actions(settings_menu)
        settings_menu.addSeparator()
        keyword_rules = settings_menu.addAction(
            _("Keyword Rules..."),
            self._show_keyword_rules,
        )
        keyword_rules.setToolTip(
            tooltip(
                "The controlled vocabulary generated keywords are snapped onto, and whether the "
                "keywords of one shoot are made to agree with each other.",
            ),
        )
        copyright_action = settings_menu.addAction(_("Copyright..."), self._show_copyright)
        copyright_action.setToolTip(
            tooltip(
                "Your credit and copyright notice, written by every save. The year comes from "
                "each photo, so one setting covers every shoot.",
            ),
        )
        pacing_action = settings_menu.addAction(_("Pacing..."), self._show_pacing)
        pacing_action.setToolTip(
            tooltip(
                "Rest a long run now and then, so it does not flatten the battery, cook the "
                "machine, or take it over while you work.",
            ),
        )
        settings_menu.addSeparator()
        self._build_language_menu(settings_menu)
        self._build_output_language_menu(settings_menu)
        settings_menu.addSeparator()
        save_defaults = settings_menu.addAction(
            _("Save Settings as Defaults..."),
            self._save_config,
        )
        save_defaults.setToolTip(
            tooltip(
                "Update the config file with the current provider, model, URL, file types, and "
                "save options. Other settings and comments in the file are preserved; the API "
                "key is never written.",
            ),
        )
        edit_config = settings_menu.addAction(_("Edit Config File..."), self._edit_config)
        edit_config.setToolTip(
            tooltip(
                "Open the config file in your default editor for the settings the GUI does not "
                "surface (prompt file, sampling, workers, filters, ...). Created if missing.",
            ),
        )

    def _build_repair_action(self, settings_menu: QMenu) -> None:
        """Add the toggle that lets a run pay a second call to undo a mixed-script answer."""
        self._repair_action = QAction(_("Rewrite Mixed-Language Output"), self)
        self._repair_action.setCheckable(True)
        self._repair_action.setChecked(self._repair_language)
        self._repair_action.setToolTip(
            tooltip(
                "When the model answers in two writing systems, ask it again for the output "
                "language ('Ocarina' for a stray '陶笛'). Costs one extra call per affected "
                "photo. Off by default; those photos are marked either way.",
            ),
        )
        self._repair_action.toggled.connect(self._on_repair_toggled)
        settings_menu.addAction(self._repair_action)

    def _on_repair_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Record the repair choice so the next generate run and a settings save both see it."""
        self._count("repair_toggle")
        self._repair_language = enabled

    def _build_notify_actions(self, settings_menu: QMenu) -> None:
        """Add the two end-of-run alert toggles, the sound one tied to the notification one."""
        self._notify_action = QAction(_("Notify When a Run Ends"), self)
        self._notify_action.setCheckable(True)
        self._notify_action.setChecked(self._notify_enabled)
        self._notify_action.setToolTip(
            tooltip(
                "Raise a desktop notification when a generation or save run finishes, so a long "
                "batch can be started and left alone. Uses the notifier your system already has; "
                "nothing is shown if it has none.",
            ),
        )
        self._notify_action.toggled.connect(self._on_notify_toggled)
        settings_menu.addAction(self._notify_action)
        self._notify_sound_action = QAction(_("Play a Sound With Notifications"), self)
        self._notify_sound_action.setCheckable(True)
        self._notify_sound_action.setChecked(self._notify_sound)
        self._notify_sound_action.setEnabled(self._notify_enabled)
        self._notify_sound_action.setToolTip(
            tooltip("Play the system alert sound with the notification."),
        )
        settings_menu.addAction(self._notify_sound_action)

    def _on_notify_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Follow the notification toggle with its sound sub-toggle: no notification, no sound."""
        self._count("notify_toggle")
        self._notify_enabled = enabled
        self._notify_sound_action.setEnabled(enabled)

    def _alert_run_finished(self, message: str) -> None:
        """
        Raise the desktop alert for a run that just ended, when the user asked for one.

        *message* is the status line the run wrote, so the notification and the window agree on what
        happened. Never blocking: the alert spawns a helper process, and the window has to stay
        responsive while it does.
        """
        notify(
            APP_NAME,
            message,
            enabled=self._notify_action.isChecked(),
            sound=self._notify_sound_action.isChecked(),
        )

    def _build_tools_menu(self, menubar: QMenu) -> None:
        """Build the Tools menu: the jobs that act on a whole library rather than one photo."""
        menu = self._tools_menu = menubar.addMenu(_("Tools"))
        menu.setToolTipsVisible(True)

        build = menu.addAction(_("Build Vocabulary..."), self._show_vocabulary_builder)
        build.setToolTip(
            tooltip(
                "Write a keyword file out of the keywords your photos already carry, which is what "
                "the strict vocabulary needs to enforce.",
            ),
        )
        self._harmonize_action = menu.addAction(_("Harmonize Shoots Now"), self._harmonize_now)
        self._harmonize_action.setToolTip(
            tooltip(
                "Make the keywords of each shoot agree with each other: the spelling and the "
                "hierarchy most of the shoot used win for all of it. Runs automatically after "
                "every generation once a session gap is set (Settings > Keyword Rules).",
            ),
        )
        menu.addSeparator()
        self._watch_action = menu.addAction(_("Watch Folder..."), self._toggle_watch)
        self._watch_action.setToolTip(
            tooltip(
                "Tag photos as they arrive: point it at the folder your card reader or sync client "
                "fills, and each new photo joins the list and is generated for review.",
            ),
        )
        menu.addSeparator()
        undo = menu.addAction(_("Undo Writes..."), self._show_undo_dialog)
        undo.setToolTip(
            tooltip(
                "Put back what a recorded run wrote: sidecars it created are deleted, files it "
                "overwrote are restored from the ExifTool backup. Covers CLI runs too.",
            ),
        )

    def _build_language_menu(self, settings_menu: QMenu) -> None:
        """Add the Language submenu: System Default plus every shipped catalog."""
        self._language_menu = settings_menu.addMenu(_("Language"))
        self._language_menu.menuAction().setToolTip(
            tooltip(
                "Language of the app itself (menus, buttons, messages). The language of the "
                "generated metadata is set under Metadata Language.",
            ),
        )
        self._language_group = QActionGroup(self)
        entries = [(i18n.AUTO, _("System Default")), *i18n.SUPPORTED_LANGUAGES.items()]
        for code, label in entries:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == self._language)
            action.triggered.connect(
                lambda _checked=False, chosen=code: self._set_language(chosen),
            )
            self._language_group.addAction(action)
            self._language_menu.addAction(action)

    def _set_language(self, code: str) -> None:
        """Persist the UI language choice into the config file; applied on the next launch."""
        if code == self._language:
            return
        self._count("ui_language")
        self._language = code
        target = self._config_target()
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(config_text_with_language(existing, code), encoding="utf-8")
        except (OSError, ConfigFileError) as exc:
            # ConfigFileError: the file on disk is not TOML we can rewrite in place. Reporting it
            # is the whole point of the dialog; letting it out of a Qt slot crashed the window.
            QMessageBox.warning(self, _(CONFIG_SAVE_ERROR_TITLE), str(exc))
            return
        self._status.setText(_("Language saved. Restart Photo Tagger to apply it."))

    def _build_output_language_menu(self, settings_menu: QMenu) -> None:
        """Add the Metadata Language submenu: one click sets the generated-metadata language."""
        menu = self._output_language_menu = settings_menu.addMenu(_("Metadata Language"))
        menu.setToolTipsVisible(True)
        menu.menuAction().setToolTip(
            tooltip(
                "Language of the generated titles, descriptions, and keywords. The language of "
                "the app itself is set under Language.",
            ),
        )
        self._output_language_group = QActionGroup(self)
        self._output_language_actions: dict[str, QAction] = {}
        # The trailing rows go in first so the language entries can insert above the separator
        # (a language chosen later via Other... joins the same list).
        self._output_language_separator = menu.addSeparator()
        other = menu.addAction(_("Other..."), self._choose_other_output_language)
        other.setToolTip(
            tooltip(
                "Any language name the model understands works; it is sent to the model as is. "
                "Saved to the config file, which the CLI's --output-language default also reads.",
            ),
        )
        # English (the default) first, then the suggestions sorted by their translated label.
        rest = sorted(
            (name for name in OUTPUT_LANGUAGE_SUGGESTIONS if name != DEFAULT_OUTPUT_LANGUAGE),
            key=lambda name: _(name).casefold(),
        )
        for name in (DEFAULT_OUTPUT_LANGUAGE, *rest):
            self._add_output_language_action(name)
        self._check_output_language_action(self._output_language)

    def _add_output_language_action(self, name: str) -> QAction:
        """
        Insert a checkable entry for *name* above the Other...

        row and return it.
        """
        # The label is translated for display; *name* (the English form) is what the prompt,
        # config file, and cache namespace carry. A custom name passes through _() unchanged.
        action = QAction(_(name), self)
        action.setCheckable(True)
        action.triggered.connect(
            lambda _checked=False, chosen=name: self._set_output_language(chosen),
        )
        self._output_language_group.addAction(action)
        self._output_language_menu.insertAction(self._output_language_separator, action)
        self._output_language_actions[name] = action
        return action

    def _check_output_language_action(self, name: str) -> None:
        """Check *name*'s menu entry, first creating one for a language not in the list."""
        action = self._output_language_actions.get(name)
        if action is None:
            action = self._add_output_language_action(name)
        action.setChecked(True)

    def _choose_other_output_language(self) -> None:
        """Ask for a free-form language name (Other...) and apply it."""
        language, ok = QInputDialog.getText(
            self,
            _("Metadata Language"),
            _("Language for the generated titles, descriptions, and keywords:"),
            text=self._output_language,
        )
        if ok:
            self._set_output_language(language)

    def _set_output_language(self, language: str) -> None:
        """Persist the metadata language into the config file; used from the next generation on."""
        normalized = language.strip() or DEFAULT_OUTPUT_LANGUAGE
        if normalized == self._output_language:
            # Still sync the menu: Other... may have re-entered the current language, and the
            # clicked action must not end up unchecked.
            self._check_output_language_action(normalized)
            return
        self._count("output_language")
        self._output_language = normalized
        self._check_output_language_action(normalized)
        target = self._config_target()
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                config_text_with_output_language(existing, normalized),
                encoding="utf-8",
            )
        except (OSError, ConfigFileError) as exc:
            # ConfigFileError: the file on disk is not TOML we can rewrite in place. Reporting it
            # is the whole point of the dialog; letting it out of a Qt slot crashed the window.
            QMessageBox.warning(self, _(CONFIG_SAVE_ERROR_TITLE), str(exc))
            return
        rules = self._keyword_rules_dialog
        if rules.vocabulary_path is not None:
            # The language decides whether plurals fold onto their singular, and that shapes the
            # vocabulary's lookup index, so it has to be built again for the new one.
            rules.load_vocabulary(rules.vocabulary_path, announce=False)
        self._status.setText(
            _("Metadata language set to {language} for the next generation.").format(
                language=normalized,
            ),
        )

    def _telemetry_consent(self) -> telemetry.Consent:
        """Resolve whether this window may send, from its toggle and the environment."""
        return telemetry.resolve_consent(
            enabled=self._telemetry_enabled,
            opted_in=self._telemetry_opted_in,
        )

    def _refresh_telemetry_action(self) -> None:
        """Check the toggle only when beacons go out; gray it out under an environment opt-out."""
        consent = self._telemetry_consent()
        with QSignalBlocker(self._telemetry_action):
            self._telemetry_action.setChecked(consent.send)
        self._telemetry_action.setEnabled(consent.reason != telemetry.CONSENT_ENV_OPT_OUT)
        text = tooltip(
            "Anonymous usage stats (the settings and features you use, model, batch size, timing, "
            "OS, CPU/GPU, RAM, battery, screen) and crash reports (error type and stack trace, "
            "code locations only). No photos, text you type, error messages, or personal data.",
        )
        if note := telemetry_toggle_note(consent.reason, do_not_track=telemetry.do_not_track()):
            text = f"{text}\n\n{wrap_tooltip(note)}"
        self._telemetry_action.setToolTip(text)

    def _on_telemetry_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Persist the choice; checking the box is an opt-in that overrules DO_NOT_TRACK."""
        self._telemetry_enabled = enabled
        self._telemetry_opted_in = enabled
        telemetry.write_gui_pref(enabled=enabled)
        self._refresh_telemetry_action()
        self._status.setText(
            _("Anonymous telemetry on.") if enabled else _("Anonymous telemetry off."),
        )

    def _active_cache_file(self) -> Path | None:
        """Return the cache file generation should use, or None when caching is toggled off."""
        return self._cache_file if self._cache_action.isChecked() else None

    def _current_config_values(self) -> GuiConfigValues:
        """Collect the GUI's current choices that persist to the config file (never the API key)."""
        pacing = self._pacing_dialog.plan
        return GuiConfigValues(
            provider_name=self._provider_name(),
            model_name=self._model.currentText().strip(),
            api_base_url=self._url.text().strip() or None,
            extensions=self._extensions.text().strip(),
            recursive=self._recursive.isChecked(),
            **self._save_options_menu.config_values(),
            telemetry_enabled=self._telemetry_enabled,
            vocabulary=self._keyword_rules_dialog.vocabulary_path,
            vocabulary_strict=self._keyword_rules_dialog.strict,
            session_gap_minutes=self._keyword_rules_dialog.session_gap,
            undo_log=self._undo_log_action.isChecked(),
            notify=self._notify_action.isChecked(),
            notify_sound=self._notify_sound_action.isChecked(),
            repair_language=self._repair_language,
            **self._copyright_dialog.config_values(),
            save_after_generating=self._autosave_action.isChecked(),
            duty=pacing.duty_percent,
            battery_floor=pacing.battery_floor,
            battery_resume=pacing.battery_resume,
            work_minutes=pacing.work_seconds / SECONDS_PER_MINUTE,
            max_pause_minutes=pacing.max_pause_seconds / SECONDS_PER_MINUTE,
        )

    def _config_target(self) -> Path:
        """Return the config file to save into: the one in effect, or the user default path."""
        return find_config_file() or user_config_path()

    def _save_config(self) -> None:
        """
        Persist the GUI's current choices into the config file.

        An existing file is merged, not replaced: only the GUI-managed keys change, and comments,
        ordering, and every other setting survive. A missing file is created from a template.
        """
        self._count("save_defaults")
        target = self._config_target()
        values = self._current_config_values()
        try:
            if target.exists():
                text = merged_config_text(target.read_text(encoding="utf-8"), values)
                note = _("Updated {target} (other settings and comments preserved).")
            else:
                text = config_toml_text(values)
                note = _("Saved defaults to {target}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except (OSError, ConfigFileError) as exc:
            # ConfigFileError: the file on disk is not TOML we can rewrite in place. Reporting it
            # is the whole point of the dialog; letting it out of a Qt slot crashed the window.
            QMessageBox.warning(self, _(CONFIG_SAVE_ERROR_TITLE), str(exc))
            return
        self._status.setText(note.format(target=target))

    def _edit_config(self) -> None:
        """Open the config file in the user's editor, creating it first if it does not exist."""
        self._count("edit_config")
        target = self._config_target()
        if not target.exists():
            self._save_config()
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _show_about(self) -> None:
        """Show a small About dialog with the version and project link."""
        self._count("about")
        QMessageBox.about(
            self,
            _("About Photo Tagger"),
            f"<b>Photo Tagger {__version__}</b><br><br>"
            + _("Describe photos and add keywords with a vision-language model.")
            + "<br><br>"
            '<a href="https://github.com/jbsilva/photo-tagger">github.com/jbsilva/photo-tagger</a>',
        )

    def _build_header(self) -> QHBoxLayout:
        """One compact row: the model choice the user changes often, the rest behind Connection."""
        provider = self._defaults.provider

        self._provider = QComboBox()
        for name in PROVIDER_NAMES:
            self._provider.addItem(PROVIDER_LABELS.get(name, name), name)
        self._provider.setCurrentIndex(max(0, list(PROVIDER_NAMES).index(provider.provider_name)))
        # Size to the widest label so "LM Studio" is not clipped.
        self._provider.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._provider.setMinimumContentsLength(10)
        self._provider.setToolTip(tooltip("Backend that serves the vision-language model."))

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setMinimumWidth(260)
        self._model.setCurrentText(provider.model_name)
        self._model.setToolTip(
            tooltip(
                "Model identifier. Type it, or press Refresh to list what the provider serves.",
            ),
        )
        refresh = QPushButton(_("Refresh"))
        refresh.setToolTip(tooltip("Query the provider for the models it currently serves."))
        refresh.clicked.connect(self._refresh_models)

        self._connection_dialog = self._build_connection_dialog()
        connection = QPushButton(_("Connection..."))
        connection.setToolTip(tooltip("Server URL, API key, and a connection test."))
        connection.clicked.connect(
            lambda: (self._count("connection"), self._connection_dialog.exec()),
        )

        row = QHBoxLayout()
        row.addWidget(QLabel(_("Provider")))
        row.addWidget(self._provider)
        row.addWidget(QLabel(_("Model")))
        row.addWidget(self._model, stretch=1)
        row.addWidget(refresh)
        row.addWidget(connection)
        return row

    def _build_connection_dialog(self) -> QDialog:
        """URL, API key, and the connection test: set-once settings, out of the main window."""
        provider = self._defaults.provider
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Connection settings"))
        dialog.setMinimumWidth(520)
        form = QFormLayout(dialog)
        # macOS style defaults to fixed-size fields; let them fill the dialog width instead.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._url = QLineEdit(provider.api_base_url or "")
        self._url.setMinimumWidth(380)
        self._url.setPlaceholderText(_("(provider default URL)"))
        self._url.setToolTip(
            tooltip("Provider API base URL. Leave blank to use the provider's default."),
        )
        form.addRow(_("Base URL"), self._url)

        # Pre-filled from a config-file key if one is set, never from an environment variable: an
        # env key stays in the environment and is resolved at call time, so it never lands in the
        # widget. A typed key is masked, used only for this session, and never written to disk.
        self._api_key = QLineEdit(provider.api_key or "")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setClearButtonEnabled(True)
        self._api_key.setMinimumWidth(380)
        self._api_key.setPlaceholderText(_("(uses provider env var)"))
        self._api_key.setToolTip(
            tooltip(
                "API key for the provider. Leave blank to use the provider's environment variable "
                "(OPENAI_API_KEY, LM_STUDIO_API_KEY, LLAMA_CPP_API_KEY, or OLLAMA_API_KEY). "
                "Required for OpenAI. A typed key is used for this session only and is never "
                "written to disk.",
            ),
        )
        form.addRow(_("API key"), self._api_key)

        self._test_button = QPushButton(_("Test Connection"))
        self._test_button.setToolTip(
            tooltip("Check ExifTool and that the provider serves the model."),
        )
        self._test_button.clicked.connect(self._test_connection)
        close = QPushButton(_(CLOSE_LABEL))
        close.setDefault(True)
        close.clicked.connect(dialog.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self._test_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        form.addRow(buttons)
        return dialog

    def _build_tree_panel(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.addLayout(self._build_tree_controls())
        box.addLayout(self._build_search_row())

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels([_("Photos"), _("Type"), _("Status"), _("Tagged")])
        # Extended: shift+click selects a range, Cmd/Ctrl+click adds single rows, and
        # shift+arrows grow the selection from the keyboard.
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self._tree.header()
        header.setStretchLastSection(False)
        for column, width in (
            (COL_NAME, 250),
            (COL_TYPE, 64),
            (COL_STATUS, 80),
            (COL_TAGGED, 56),
        ):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self._tree.setColumnWidth(column, width)
        # Click a header to sort by any column; folders stay grouped above files.
        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(COL_NAME, Qt.SortOrder.AscendingOrder)
        header.setToolTip(
            tooltip(
                "Click a column header to sort. Type: file extension, +xmp when a sidecar "
                "exists.\nTagged: metadata already on the file ({legend}).",
                legend=tagged_legend(),
            ),
        )
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        for key in (QKeySequence.StandardKey.Delete, QKeySequence(Qt.Key.Key_Backspace)):
            shortcut = QShortcut(key, self._tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._remove_selected)
        box.addWidget(self._tree, stretch=1)
        box.addWidget(self._build_scan_row())

        hint = QLabel(_("Drag photos or folders here. Select one and press Delete to remove it."))
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        box.addWidget(hint)
        return panel

    def _build_scan_row(self) -> QWidget:
        """
        Build the row under the list that reports the metadata scan behind the Tagged column.

        Hidden while nothing is scanning. It carries its own Stop button rather than borrowing the
        bottom bar's Cancel, because a scan and a generate/save run can be in flight at once (a
        watch adds photos mid-run) and the two must stay separately stoppable.
        """
        self._scan_progress = QProgressBar()
        self._scan_stop_button = QToolButton()
        self._scan_stop_button.setText(_("Stop"))
        self._scan_stop_button.setToolTip(
            tooltip(
                "Stop reading metadata. The photos already read keep their Tagged column; the "
                "rest stay unknown until the next scan. Nothing is written either way.",
            ),
        )
        self._scan_stop_button.clicked.connect(self._cancel_scan)
        # Only show the row for a scan that actually runs long enough to be worth reporting, so a
        # handful of local photos does not make a progress bar flash on and off.
        self._scan_show_timer = QTimer(self)
        self._scan_show_timer.setSingleShot(True)
        self._scan_show_timer.setInterval(_SCAN_PROGRESS_DELAY_MS)
        self._scan_show_timer.timeout.connect(self._show_scan_row)

        self._scan_row = QWidget()
        row = QHBoxLayout(self._scan_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._scan_progress, stretch=1)
        row.addWidget(self._scan_stop_button)
        self._scan_row.setVisible(False)
        return self._scan_row

    def _build_tree_controls(self) -> QHBoxLayout:
        controls = QHBoxLayout()

        # One split button: a click adds files; the arrow offers the folder dialog and the scan
        # options. Native file dialogs cannot select files and folders at once, so the split is
        # the closest single-control equivalent (drag-and-drop takes both anyway).
        add = QToolButton()
        add.setObjectName("add")
        add.setText(_("Add Photos..."))
        add.setToolTip(
            tooltip(
                "Add photos (click), or open the arrow for adding a whole folder and for the "
                "folder-scan options. Dragging files or folders onto the window also works.",
            ),
        )
        add.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        add.clicked.connect(self._choose_files)
        add.setMenu(self._build_add_menu())
        controls.addWidget(add)
        controls.addStretch(1)

        select = QPushButton(_("Select"))
        select.setObjectName("menubutton")
        select.setToolTip(tooltip("Check or uncheck photos in bulk."))
        select.setMenu(self._build_select_menu())
        controls.addWidget(select)

        # A split button like Add: a click removes the highlighted rows, and the arrow removes by
        # checkbox, since the rows a user highlights and the photos they checked are not the same.
        remove = QToolButton()
        remove.setObjectName("split")
        remove.setText(_("Remove"))
        remove.setToolTip(
            tooltip(
                "Remove the selected folder or photo from the list (or press Delete). Open the "
                "arrow to remove every checked or every unchecked photo instead.",
            ),
        )
        remove.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        remove.clicked.connect(self._remove_selected)
        remove.setMenu(self._build_remove_menu())
        controls.addWidget(remove)
        return controls

    def _build_remove_menu(self) -> QMenu:
        """Build the Remove button's arrow menu: drop photos by their checkbox, not by selection."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        checked = menu.addAction(_("Remove Checked"), lambda: self._remove_by_check(checked=True))
        checked.setToolTip(tooltip("Take every checked photo off the list. Files are untouched."))
        unchecked = menu.addAction(
            _("Remove Unchecked"),
            lambda: self._remove_by_check(checked=False),
        )
        unchecked.setToolTip(
            tooltip(
                "Keep only the checked photos: take the rest off the list. Files are untouched.",
            ),
        )
        return menu

    def _build_search_row(self) -> QHBoxLayout:
        """Build the slim row right above the list: the search box, and the view-mode button."""
        row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setObjectName("search")
        self._search.setPlaceholderText(_("Search photos"))
        self._search.setClearButtonEnabled(True)
        self._search.addAction(
            svg_icon("search.svg"),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        self._search.setToolTip(
            tooltip(
                "Show only the photos whose name matches. Plain text matches anywhere in the "
                "name, and *, ? and [] are wildcards, so *_edit.jpg works. This changes what "
                "the list shows, not what a run processes: a hidden photo that is checked is "
                "still generated and saved.",
            ),
        )
        # Re-filtering rebuilds every row, so a long list is rebuilt once per word typed rather
        # than once per letter.
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(_SEARCH_DEBOUNCE_MS)
        self._search_timer.timeout.connect(self._apply_search)
        self._search.textChanged.connect(lambda _text: self._search_timer.start())
        row.addWidget(self._search, stretch=1)
        row.addWidget(self._build_view_button())
        return row

    def _apply_search(self) -> None:
        """Redraw the list through the current search text."""
        if self._search.text().strip():
            self._count("search")
        self._rebuild_tree()
        self._update_status()

    def _visible_paths(self) -> list[Path]:
        """
        Return the photos the list draws: all of them, or those the search box matches.

        Only the view narrows. What a run processes is still every checked photo, the same way the
        thumbnail grid's own filter never changed what Generate Selected picks up.
        """
        query = self._search.text().strip()
        if not query:
            return self._item_paths()
        return [item.path for item in self._items.values() if matches_name_pattern(item, query)]

    def _build_view_button(self) -> QToolButton:
        """
        Build the small view-mode button that sits above the photo list.

        Deliberately quiet: an icon that opens a menu, the way a file browser offers its list and
        tree views, rather than a labelled button competing with Select and Remove for a choice most
        people make once.
        """
        button = self._view_button = QToolButton()
        button.setObjectName("viewmode")
        button.setIcon(svg_icon("list-view.svg"))
        button.setIconSize(QSize(_TOOLBAR_ICON, _TOOLBAR_ICON))
        # Flat: without this some styles paint a raised button frame around the icon, which is
        # exactly the weight this control is meant not to have.
        button.setAutoRaise(True)
        # The menu is popped by hand rather than attached with setMenu: an attached one makes Qt
        # paint its arrow over the icon, and the arrow is not reliably styled away.
        button.clicked.connect(self._show_view_menu)
        menu = self._view_menu = QMenu(self)
        menu.setToolTipsVisible(True)
        group = self._view_group = QActionGroup(self)
        self._view_actions = {}
        for mode, label, hint in _VIEW_MODES:
            action = QAction(_(label), self)
            action.setCheckable(True)
            action.setChecked(mode == VIEW_TREE)
            action.setToolTip(tooltip(hint))
            group.addAction(action)
            menu.addAction(action)
            self._view_actions[mode] = action
        self._refresh_view_button()
        # One connection covers both directions: the group unchecks the other mode for us.
        self._view_actions[VIEW_LIST].toggled.connect(self._on_view_mode_changed)
        return button

    def _show_view_menu(self) -> None:
        """Drop the view menu under the button, the way an attached menu would."""
        self._view_menu.popup(self._view_button.mapToGlobal(self._view_button.rect().bottomLeft()))

    def _flat_list(self) -> bool:
        """Whether the photo list is flat, rather than grouped under folder rows."""
        return self._view_actions[VIEW_LIST].isChecked()

    def _on_view_mode_changed(self, _flat: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Rebuild the list the other way round; the rebuild restores what was open."""
        self._count("view_mode")
        self._refresh_view_button()
        self._rebuild_tree()

    def _refresh_view_button(self) -> None:
        """Name the active mode in the button's tooltip, since the button itself never says."""
        active = next(
            label for mode, label, _hint in _VIEW_MODES if self._view_actions[mode].isChecked()
        )
        self._view_button.setToolTip(
            tooltip("How the photo list is laid out. Now: {mode}.", mode=_(active)),
        )

    def _build_add_menu(self) -> QMenu:
        """Build the Add button's arrow menu: the folder dialog plus the folder-scan settings."""
        menu = QMenu(self)
        menu.addAction(_("Add Folder..."), self._choose_folder)
        menu.addSeparator()
        panel = QWidget()
        form = QFormLayout(panel)
        self._extensions = QLineEdit(self._raw_config.get("extensions", DEFAULT_GUI_EXTENSIONS))
        self._extensions.setMinimumWidth(280)
        self._extensions.setToolTip(
            tooltip(
                "Extensions to scan for in folders (comma-separated).\n"
                "Case-insensitive: jpg matches .JPG. Note jpeg is separate from jpg.",
            ),
        )
        form.addRow(_("File types"), self._extensions)
        self._recursive = QCheckBox(_("Include subfolders"))
        self._recursive.setChecked(bool(self._raw_config.get("recursive", True)))
        self._recursive.setToolTip(tooltip("Descend into subfolders when adding a folder."))
        form.addRow("", self._recursive)
        host = QWidgetAction(menu)
        host.setDefaultWidget(panel)
        menu.addAction(host)
        return menu

    def _build_select_menu(self) -> QMenu:
        """Bulk check/uncheck actions, including the CLI's --skip-tagged/--skip-from mirrors."""
        menu = self._select_menu = QMenu(self)
        menu.setToolTipsVisible(True)
        menu.addAction(_("Check All"), lambda: self._set_all_checked(checked=True))
        menu.addAction(_("Uncheck All"), lambda: self._set_all_checked(checked=False))
        menu.addAction(_("Invert Checked"), self._invert_checked)
        menu.addSeparator()

        for verb, label, hint in _SELECT_VERBS:
            self._build_criteria_menu(menu.addMenu(_(label)), verb).setToolTip(tooltip(hint))
        menu.addSeparator()

        self._tagged_menu = menu.addMenu(_("Uncheck Already Tagged"))
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
                lambda _checked=False, chosen=criterion: self._deselect_tagged(chosen),
            )

        from_file = menu.addAction(_("Uncheck From Skip List..."), self._deselect_from_file)
        from_file.setToolTip(
            tooltip(
                "Uncheck photos whose filename or full path is listed in a text file (one per "
                "line), like the CLI's --skip-from.",
            ),
        )
        return menu

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
                lambda crit=criterion, text=label: self._select_matching(
                    lambda item: photo_matches_filter(item, crit),
                    verb=verb,
                    phrase=_(text),
                ),
            )
        pattern = menu.addAction(_("Name Pattern..."), lambda: self._select_by_pattern(verb))
        pattern.setToolTip(
            tooltip("Match filenames against text or a glob, e.g. IMG_ or *_edit.jpg."),
        )
        return menu

    def _fill_file_type_menu(self, menu: QMenu, verb: str) -> None:
        """Rebuild *menu* from the file types currently in the list, most common first."""
        menu.clear()
        counts = extension_counts(self._items.values())
        if not counts:
            menu.addAction(_("No photos in the list")).setEnabled(False)
            return
        for extension, count in counts:
            label = extension or _("no extension")
            menu.addAction(
                f"{label} ({count})",
                lambda ext=extension, name=label: self._select_matching(
                    lambda item: matches_extension(item, ext),
                    verb=verb,
                    phrase=name,
                ),
            )

    def _warn_if_no_photos(self) -> bool:
        """
        Say the list is empty, and report whether the caller should give up.

        Every bulk selection action starts with this. With nothing loaded there is nothing to
        select, and returning silently would look like a dead menu item.
        """
        if self._items:
            return False
        self._status.setText(_("Add photos before selecting."))
        return True

    def _select_by_pattern(self, verb: str) -> None:
        """Ask for a name pattern and apply *verb* to the photos it matches."""
        self._count("select")
        if self._warn_if_no_photos():
            return
        pattern, accepted = QInputDialog.getText(
            self,
            _("Name Pattern"),
            _("Match filenames containing this text, or a glob such as *_edit.jpg:"),
            text=self._last_name_pattern,
        )
        pattern = pattern.strip()
        if not accepted or not pattern:
            return
        self._last_name_pattern = pattern
        self._select_matching(
            lambda item: matches_name_pattern(item, pattern),
            verb=verb,
            phrase=pattern,
        )

    def _select_matching(
        self,
        matches: Callable[[PhotoItem], bool],
        *,
        verb: str,
        phrase: str,
    ) -> None:
        """Apply one bulk selection action and report what it did; *phrase* names the criterion."""
        self._count("select")
        if self._warn_if_no_photos():
            return
        result = apply_selection(self._items.values(), matches, verb)
        if result.changed:
            self._rebuild_tree()
        self._update_status()
        self._status.setText(self._selection_message(result, phrase))

    def _selection_message(self, result: SelectionChange, phrase: str) -> str:
        """Word the status line for a bulk selection: nothing matched, nothing moved, or a tally."""
        if not result.matched:
            return _("No photos match {phrase}.").format(phrase=phrase)
        if not result.changed:
            return _("Nothing to change for {phrase}; {selected} still checked.").format(
                phrase=phrase,
                selected=self._selected_count(),
            )
        return ngettext(
            "Changed {n} photo ({phrase}); {selected} now checked.",
            "Changed {n} photos ({phrase}); {selected} now checked.",
            result.changed,
        ).format(n=result.changed, phrase=phrase, selected=self._selected_count())

    def _set_all_checked(self, *, checked: bool) -> None:
        """Check or uncheck every photo at once."""
        self._count("select")
        if self._warn_if_no_photos():
            return
        for item in self._items.values():
            item.selected = checked
        self._rebuild_tree()
        self._update_status()

    def _invert_checked(self) -> None:
        """Flip every photo's checkbox: checked becomes unchecked and vice versa."""
        self._count("select")
        if self._warn_if_no_photos():
            return
        for item in self._items.values():
            item.selected = not item.selected
        self._rebuild_tree()
        self._update_status()

    # --- tree context menu -------------------------------------------------------------------

    def _on_tree_context_menu(self, pos: object) -> None:
        """Show the per-row context menu for the tree item under the cursor."""
        tree_item = self._tree.itemAt(pos)
        menu = self._build_tree_context_menu(tree_item)
        if menu is not None:
            menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _build_tree_context_menu(self, tree_item: QTreeWidgetItem | None) -> QMenu | None:
        """Build the context menu for *tree_item* (a file leaf or a folder), or None."""
        if tree_item is None:
            return None
        path = tree_item.data(0, PATH_ROLE)
        if path is None:
            return None
        # A right-click on one of several selected photos acts on the whole selection.
        selected = self._selected_photo_items()
        clicked_in_selection = any(str(item.path) == path for item in selected)
        if not bool(tree_item.data(0, IS_DIR_ROLE)) and len(selected) > 1 and clicked_in_selection:
            return self._build_bulk_context_menu(selected)
        menu = QMenu(self._tree)
        item = self._items.get(path)
        if not bool(tree_item.data(0, IS_DIR_ROLE)) and item is not None:
            label = _("Retry Generation") if item.status == FAILED else _("Generate")
            trigger = TRIGGER_RETRY if item.status == FAILED else TRIGGER_SINGLE
            generate = menu.addAction(
                label,
                lambda: self._run_generation([item], trigger=trigger),
            )
            generate.setEnabled(self._thread is None)
            fresh = menu.addAction(
                _("Generate (Skip Cache)"),
                lambda: self._run_generation([item], use_cache=False, trigger=TRIGGER_SINGLE),
            )
            fresh.setToolTip(
                tooltip("Call the model even when a cached result exists for this photo."),
            )
            fresh.setEnabled(self._thread is None)
            menu.addSeparator()
        menu.addAction(reveal_label(sys.platform), lambda: self._reveal(Path(path)))
        menu.addSeparator()
        remove = menu.addAction(_("Remove From List"))
        remove.triggered.connect(
            lambda: (self._tree.setCurrentItem(tree_item), self._remove_selected()),
        )
        return menu

    def _selected_photo_items(self) -> list[PhotoItem]:
        """Return the photos behind the tree's currently selected file rows."""
        out: list[PhotoItem] = []
        for tree_item in self._tree.selectedItems():
            if bool(tree_item.data(0, IS_DIR_ROLE)):
                continue
            item = self._items.get(tree_item.data(0, PATH_ROLE))
            if item is not None:
                out.append(item)
        return out

    def _build_bulk_context_menu(self, items: list[PhotoItem]) -> QMenu:
        """Build the context menu shown when several photos are selected at once."""
        menu = QMenu(self)
        n = len(items)
        menu.addAction(
            ngettext("Check {n} Photo", "Check {n} Photos", n).format(n=n),
            lambda: self._set_items_checked(items, checked=True),
        )
        menu.addAction(
            ngettext("Uncheck {n} Photo", "Uncheck {n} Photos", n).format(n=n),
            lambda: self._set_items_checked(items, checked=False),
        )
        only = menu.addAction(
            ngettext("Check Only {n} Photo", "Check Only {n} Photos", n).format(n=n),
            lambda: self._check_only_items(items),
        )
        only.setToolTip(
            tooltip("Check the selected photos and uncheck every other photo in the list."),
        )
        menu.addSeparator()
        generate = menu.addAction(
            ngettext("Generate {n} Photo", "Generate {n} Photos", n).format(n=n),
            lambda: self._run_generation(items),
        )
        generate.setEnabled(self._thread is None)
        fresh = menu.addAction(
            ngettext(
                "Generate {n} Photo (Skip Cache)",
                "Generate {n} Photos (Skip Cache)",
                n,
            ).format(n=n),
            lambda: self._run_generation(items, use_cache=False),
        )
        fresh.setToolTip(tooltip("One-time: call the model even for photos with cached results."))
        fresh.setEnabled(self._thread is None)
        menu.addSeparator()
        menu.addAction(
            _("Remove From List"),
            lambda: self._remove_items([str(item.path) for item in items]),
        )
        return menu

    def _check_only_items(self, items: list[PhotoItem]) -> None:
        """Check exactly *items*: everything else in the list is unchecked."""
        keys = {str(item.path) for item in items}
        others = [item for item in self._items.values() if str(item.path) not in keys]
        self._set_items_checked(others, checked=False)
        self._set_items_checked(items, checked=True)

    def _set_items_checked(self, items: list[PhotoItem], *, checked: bool) -> None:
        """Check or uncheck *items* in place, keeping tree, folders, and grid in step."""
        self._syncing = True
        leaves = []
        for item in items:
            item.selected = checked
            leaf = self._leaf_for(item.path)
            if leaf is not None:
                leaf.setCheckState(0, check_state(checked))
                leaves.append(leaf)
        # Folder tristates re-derive after every leaf is set, walking each chain upward.
        for leaf in leaves:
            self._sync_ancestors(leaf)
        self._sync_grid_checks()
        self._syncing = False
        self._update_status()

    def _on_grid_item_changed(self, grid_item: QListWidgetItem) -> None:
        """Mirror a thumbnail checkbox change onto the photo and its tree row."""
        if self._syncing:
            return
        key = grid_item.data(PATH_ROLE)
        item = self._items.get(key)
        if item is None:
            return
        checked = grid_item.checkState() == Qt.CheckState.Checked
        if checked == item.selected:
            # Icon and tooltip updates also emit itemChanged; only a real check-state change
            # counts as a toggle.
            return
        # Remember that this was a checkbox click so the itemClicked that follows the same
        # mouse release does not also open the photo. The itemClicked (when there is one)
        # arrives within the same event dispatch, so clear the flag once the loop settles;
        # otherwise a keyboard (space key) toggle would leave it stale and swallow the user's
        # next real click on a thumbnail.
        self._grid_check_toggled = True
        QTimer.singleShot(0, self._clear_grid_check_toggled)
        item.selected = checked
        self._syncing = True
        leaf = self._leaf_for(Path(key))
        if leaf is not None:
            leaf.setCheckState(0, check_state(item.selected))
            self._sync_ancestors(leaf)
        self._syncing = False
        self._update_status()

    def _clear_grid_check_toggled(self) -> None:
        """Reset the checkbox-click marker once the current event dispatch has finished."""
        self._grid_check_toggled = False

    def _sync_grid_checks(self) -> None:
        """Repaint every visible thumbnail checkbox from the model (callers hold _syncing)."""
        for key, grid_item in self._grid_items.items():
            item = self._items.get(key)
            if item is not None:
                grid_item.setCheckState(check_state(item.selected))

    def _on_grid_context_menu(self, pos: object) -> None:
        """Offer a thumbnail the same right-click actions as its row in the tree."""
        grid_item = self._grid.itemAt(pos)
        if grid_item is None:
            return
        selected_keys = [gi.data(PATH_ROLE) for gi in self._grid.selectedItems()]
        if len(selected_keys) > 1 and grid_item.data(PATH_ROLE) in selected_keys:
            items = [self._items[key] for key in selected_keys if key in self._items]
            menu: QMenu | None = self._build_bulk_context_menu(items)
        else:
            menu = self._build_tree_context_menu(self._leaf_for(Path(grid_item.data(PATH_ROLE))))
        if menu is not None:
            menu.exec(self._grid.viewport().mapToGlobal(pos))

    def _reveal(self, path: Path) -> None:
        """Show *path* selected in the OS file browser, or open its folder where unsupported."""
        self._count("reveal")
        argv = reveal_command(path, sys.platform)
        if argv is not None:
            try:
                subprocess.Popen(argv)  # noqa: S603  # nosec B603 - resolved argv, path from our list
            except OSError as exc:
                # The browser is there (reveal_command resolved it) but would not start. Fall
                # through to the folder rather than letting it out of a slot and taking the
                # window down over a context-menu action.
                logger.warning("reveal_failed", file=path.name, error=str(exc))
            else:
                return
        folder = path if path.is_dir() else path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # --- background metadata scan (the Tagged column) ----------------------------------------

    def _start_metadata_scan(self) -> None:
        """Scan photos with an unknown Tagged state in the background, one batch at a time."""
        if self._scan_thread is not None:
            return  # a scan is running; _on_scan_finished re-checks for stragglers
        # A photo a scan already tried and could not read stays unknown; asking again straight away
        # would only fail again, and restart the scan forever.
        pending = [
            item.path
            for key, item in self._items.items()
            if item.known_fields is None and key not in self._scan_tried
        ]
        if not pending:
            return
        self._scan_batch = {str(path) for path in pending}
        self._scan_thread = QThread(self)
        self._scan_worker = MetadataScanWorker(pending)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_progress.setFormat(_("Reading metadata... %v / %m"))
        self._scan_stop_button.setEnabled(True)
        self._scan_thread.start()
        self._scan_show_timer.start()

    def _on_scan_progress(self, scanned: int, total: int) -> None:
        """Move the scan's progress bar as each batch of photos is read."""
        self._scan_progress.setRange(0, total)
        self._scan_progress.setValue(scanned)

    def _show_scan_row(self) -> None:
        """Reveal the scan row, unless the scan it would report has already finished."""
        self._scan_row.setVisible(self._scan_thread is not None)

    def _hide_scan_row(self) -> None:
        """Put the scan row away, and cancel a pending reveal so it cannot pop up after."""
        self._scan_show_timer.stop()
        self._scan_row.setVisible(False)

    def _cancel_scan(self) -> None:
        """Stop the metadata scan after the batch it is on (the Stop button under the list)."""
        if self._scan_worker is None:
            return
        self._scan_cancelled = True
        self._scan_worker.stop()
        self._scan_stop_button.setEnabled(False)
        self._scan_progress.setFormat(_("Stopping..."))

    def _on_scan_done(self, presence: dict[str, set[str]]) -> None:
        """Record the scanned field sets and repaint the Tagged column."""
        for key, fields in presence.items():
            item = self._items.get(key)
            if item is not None:
                # Merge rather than replace: a save may have added fields while the scan ran.
                item.known_fields = set(fields) | (item.known_fields or set())
                self._refresh_status_cell(item)
        # The Untagged and No copyright filters depend on this scan, so a grid showing one must
        # re-evaluate once the results land; the rest read state the scan does not change.
        if self._grid_filter in SCAN_FILTERS:
            self._resort_grid()

    def _on_scan_finished(self) -> None:
        """Tear down the scan thread and pick up photos added while it ran."""
        if self._closing:
            # A scan detached by _stop_scan's timeout can finish after closeEvent already moved
            # on; the window is going away, so there is nothing left to restart a scan for.
            return
        stopped = self._scan_cancelled
        self._stop_scan()
        if not stopped:
            # A scan that ran to its end read every photo it could; those still unknown are
            # unreadable, so the restart below leaves them out. A stopped scan tried nothing more.
            self._scan_tried |= {
                key
                for key in self._scan_batch
                if (item := self._items.get(key)) is not None and item.known_fields is None
            }
            self._start_metadata_scan()
        if self._scan_thread is None:
            # Nothing left unknown (or the user stopped it): the scan is over for now.
            self._end_scan(stopped=stopped)

    def _end_scan(self, *, stopped: bool) -> None:
        """Put the scan row away and settle whatever was waiting on the scan."""
        self._scan_cancelled = False
        self._hide_scan_row()
        criterion, self._scan_pending = self._scan_pending, None
        if criterion is None:
            return
        if stopped:
            self._status.setText(_("Stopped reading metadata. No photo was unchecked."))
        else:
            self._apply_tagged_criterion(criterion)

    def _stop_scan(self) -> None:
        if self._scan_worker is not None:
            # Between batches the worker checks this, so a detached scan below stops reading the
            # rest of the library instead of running it out in the background.
            self._scan_worker.stop()
            # Cut the worker loose before letting go of it. A scan detached by the timeout below
            # keeps running, and its late finished signal would otherwise reach _on_scan_finished
            # and tear down whatever scan is current by then, not the one that just ended.
            self._scan_worker.disconnect(self)
            self._scan_worker.deleteLater()
        if self._scan_thread is not None:
            thread = self._scan_thread
            thread.quit()
            if thread.wait(_SCAN_STOP_TIMEOUT_MS):
                thread.deleteLater()
            else:
                # Still running: a big folder, or a hung exiftool. pyexiftool exposes no way to
                # cancel a batched call mid-flight, so detach instead of blocking the caller
                # (closeEvent, _clear) indefinitely. The scan only reads metadata; losing its
                # result is harmless. Clean up once it actually finishes, whenever that is.
                logger.warning("gui_metadata_scan_stop_timed_out")
                thread.finished.connect(thread.deleteLater)
            self._scan_thread = None
        self._scan_worker = None

    def _build_right_pane(self) -> QWidget:
        """Build the nav row over a stack: the idle placeholder, a photo's detail, a folder grid."""
        self._right = QStackedWidget()
        self._right.addWidget(self._build_empty_state())  # _PAGE_EMPTY
        self._right.addWidget(self._build_detail_panel())  # _PAGE_DETAIL
        self._right.addWidget(self._build_grid())  # _PAGE_GRID
        pane = QWidget()
        box = QVBoxLayout(pane)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._build_nav_row())
        box.addWidget(self._right, stretch=1)
        return pane

    def _build_nav_row(self) -> QHBoxLayout:
        """
        Build the row above the pane: Back, Forward, and the name of what is open.

        The arrows are the answer to "how do I get back to the grid I came from": every page of the
        stack shares them, so they are in the same spot whether a photo or a folder is open. The
        label to their right says where that is, since the tree row may be scrolled out of sight.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._back_button = self._nav_button("←", self._go_back)
        self._forward_button = self._nav_button("→", self._go_forward)
        row.addWidget(self._back_button)
        row.addWidget(self._forward_button)
        row.addSpacing(8)
        self._crumb = QLabel("")
        self._crumb.setObjectName("crumb")
        row.addWidget(self._crumb, stretch=1)
        return row

    def _nav_button(self, glyph: str, handler: Callable[[], None]) -> QToolButton:
        """Build one arrow button; its tooltip and enabled state come from ``_refresh_nav``."""
        button = QToolButton()
        button.setObjectName("navbtn")
        button.setText(glyph)
        button.clicked.connect(handler)
        return button

    def _build_empty_state(self) -> QWidget:
        """Build the idle placeholder shown when no photo is open, so the pane is never empty."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.addStretch(1)
        icon = QLabel()
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = app_icon().pixmap(96, 96)
        if not pixmap.isNull():
            icon.setPixmap(pixmap)
        box.addWidget(icon)
        self._empty_message = QLabel(_(_EMPTY_START))
        self._empty_message.setObjectName("empty")
        self._empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_message.setWordWrap(True)
        box.addWidget(self._empty_message)
        box.addStretch(1)
        return page

    def _build_grid(self) -> QWidget:
        """Build the folder grid page: a filter/sort toolbar above the thumbnail list."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._build_grid_toolbar())

        grid = QListWidget()
        grid.setViewMode(QListView.ViewMode.IconMode)
        grid.setResizeMode(QListView.ResizeMode.Adjust)
        grid.setMovement(QListView.Movement.Static)
        grid.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        grid.setGridSize(QSize(_THUMB_SIZE + 24, _THUMB_SIZE + 40))
        grid.setSpacing(8)
        grid.setUniformItemSizes(True)
        grid.setWordWrap(True)
        grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        grid.itemClicked.connect(self._on_thumb_activated)
        grid.itemChanged.connect(self._on_grid_item_changed)
        # Thumbnails answer to the same right-click menu as their row in the tree.
        grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        grid.customContextMenuRequested.connect(self._on_grid_context_menu)
        self._grid = grid
        box.addWidget(grid, stretch=1)
        return page

    def _build_grid_toolbar(self) -> QHBoxLayout:
        """
        Filter and sort controls for the folder grid, standing in for the tree's column headers.

        The grid has no headers to click, so a Show filter narrows which photos appear and a Sort by
        field picker plus an ascending/descending toggle order what remains. Each combo is sized to
        its widest entry so no label is clipped. Signals are wired only after the widgets are
        populated, so building them fires no spurious rebuild.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Show")))
        self._grid_filter_combo = QComboBox()
        for label, criterion in (
            (_("All"), FILTER_ALL),
            (_("Selected"), FILTER_SELECTED),
            (_("Not generated"), FILTER_PENDING),
            (_("Generated"), FILTER_GENERATED),
            (_("Saved"), FILTER_SAVED),
            (_("Failed"), FILTER_FAILED),
            (_("Untagged"), FILTER_UNTAGGED),
            (_("No copyright"), FILTER_NO_COPYRIGHT),
            (_("Mixed writing systems"), FILTER_MIXED_SCRIPT),
        ):
            self._grid_filter_combo.addItem(label, criterion)
        self._grid_filter_combo.setToolTip(tooltip("Show only the photos in the chosen state."))
        fit_combo(self._grid_filter_combo)
        row.addWidget(self._grid_filter_combo)
        row.addSpacing(16)

        row.addWidget(QLabel(_("Sort by")))
        self._grid_sort_combo = QComboBox()
        for label, criterion in (
            (_("Name"), SORT_NAME),
            (_("Type"), SORT_TYPE),
            (_("Status"), SORT_STATUS),
            (_("Tagged"), SORT_TAGGED),
        ):
            self._grid_sort_combo.addItem(label, criterion)
        self._grid_sort_combo.setToolTip(
            tooltip("Order the thumbnails by name, file type, status, or existing metadata."),
        )
        fit_combo(self._grid_sort_combo)
        row.addWidget(self._grid_sort_combo)

        self._grid_sort_dir = QToolButton()
        self._grid_sort_dir.setObjectName("sortdir")
        self._grid_sort_dir.setCheckable(True)
        self._grid_sort_dir.setText("↑")
        self._grid_sort_dir.setToolTip(tooltip("Ascending. Click to sort descending."))
        row.addWidget(self._grid_sort_dir)
        row.addStretch(1)

        self._grid_filter_combo.currentIndexChanged.connect(self._on_grid_filter_changed)
        self._grid_sort_combo.currentIndexChanged.connect(self._on_grid_sort_changed)
        self._grid_sort_dir.toggled.connect(self._on_grid_sort_dir_toggled)
        return row

    def _on_grid_filter_changed(self) -> None:
        """Re-filter the visible grid when the Show choice changes."""
        self._count("grid_filter")
        self._grid_filter = self._grid_filter_combo.currentData()
        self._resort_grid()

    def _on_grid_sort_changed(self) -> None:
        """Re-sort the visible grid when the sort field changes."""
        self._count("grid_sort")
        self._grid_sort = self._grid_sort_combo.currentData()
        self._resort_grid()

    def _on_grid_sort_dir_toggled(self, descending: bool) -> None:  # noqa: FBT001 - Qt toggled slot
        """Flip the grid between ascending and descending, updating the toggle's arrow."""
        self._count("grid_sort")
        self._grid_sort_desc = descending
        self._grid_sort_dir.setText("↓" if descending else "↑")
        self._grid_sort_dir.setToolTip(
            tooltip("Descending. Click to sort ascending.")
            if descending
            else _("Ascending. Click to sort descending."),
        )
        self._resort_grid()

    def _resort_grid(self) -> None:
        """Rebuild the grid with the current filter and order; cached thumbnails are reused."""
        if self._grid_folder is not None and self._right.currentIndex() == _PAGE_GRID:
            self._show_grid(self._grid_folder)

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
        # The metadata source rides along in the Existing header (it has no "New" counterpart,
        # so a full grid row of its own broke the columns' symmetry).
        existing_header = QHBoxLayout()
        existing_header.addWidget(self._section_label(_("Existing")))
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
        new_header.addWidget(self._section_label(_("New (editable)")))
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
        self._title_source = self._source_note()
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
        self._description_source = self._source_note()
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
        self._keywords.textChanged.connect(self._refresh_derived)
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

    def _source_note(self) -> QLabel:
        """Build the dimmed line under an Existing field that says where its value came from."""
        label = QLabel("")
        label.setObjectName("hint")
        label.setWordWrap(True)
        label.setToolTip(
            tooltip(
                "Which file this value came from. A sidecar wins, so a value left in the photo is "
                "named rather than hidden: only a save to the photo itself replaces it.",
            ),
        )
        return label

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
            self._count("keyword_changes")
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self._details_toggle.setArrowType(arrow)
        self._details_panel.setVisible(expanded)

    def _on_save_options_changed(self) -> None:
        """Follow a Save Options toggle into everything that describes or previews a save."""
        self._count("save_options")
        self._refresh_save_tooltips()
        self._refresh_derived()
        self._refresh_planned_copyright()

    def _save_options_summary(self) -> str:
        """Describe what a save currently writes, for the Save and Generate buttons' tooltips."""
        return self._save_options_menu.summary(writes_copyright=self._writes_copyright())

    def _writes_copyright(self) -> bool:
        """Report whether a save stamps a copyright: one is set, and Write Copyright is on."""
        return self._save_options_menu.write_copyright.isChecked() and self._copyright_dialog.usable

    def _refresh_save_tooltips(self) -> None:
        """Keep both Save buttons' tooltips describing the currently chosen options."""
        summary = _("Currently writes {options}.").format(options=self._save_options_summary())
        self._save_button.setToolTip(
            tooltip(
                "Write this photo. {summary} Change what is written with the arrow.",
                summary=summary,
            ),
        )
        self._save_selected_button.setToolTip(
            tooltip(
                "Write the checked photos that have a generated proposal. With a copyright set, "
                "also stamp it on the other checked photos, leaving their text alone. {summary} "
                "Change what is written with the arrow.",
                summary=summary,
            ),
        )
        self._refresh_generate_labels()

    def _build_autosave_action(self) -> QAction:
        """
        Build the toggle that lets a finished generation write its own photos.

        Off by default: the window is built around reviewing before writing. Turning it on is what
        an overnight batch needs, where waiting for a run to end just to press Save defeats the
        point of leaving the machine alone.
        """
        action = QAction(_("Save After Generating"), self)
        action.setCheckable(True)
        action.setChecked(bool(self._raw_config.get("gui", {}).get("save_after_generating")))
        action.setToolTip(
            tooltip(
                "Write every photo a run generates as soon as the run finishes, without "
                "reviewing them. It is the same write the Save buttons make: Save Options, just "
                "below, decides what goes onto the file, and Cancel stops it. First the run has "
                "one more go at the photos that failed and at the ones whose text came back in "
                "two writing systems, since a second answer usually settles both.",
            ),
        )
        action.toggled.connect(self._on_autosave_toggled)
        return action

    def _on_autosave_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Follow the toggle: the buttons now promise a write, and must be able to make one."""
        self._count("autosave")
        self._refresh_generate_labels()
        if enabled and not self._write_fields_chosen():
            self._status.setText(_(PICK_A_FIELD))

    def _refresh_generate_labels(self) -> None:
        """Keep both Generate buttons saying whether the run writes when it is done, and what."""
        if not self._autosave_action.isChecked():
            self._generate_button.setText(_("Generate Selected"))
            self._generate_button.setToolTip(tooltip("Run the model on the checked photos."))
            self._generate_one_button.setText(_("Generate This Photo"))
            self._generate_one_button.setToolTip(
                tooltip(
                    "Run the model on just this photo, regardless of which photos are checked.",
                ),
            )
            return
        summary = _("Currently writes {options}.").format(options=self._save_options_summary())
        self._generate_button.setText(_("Generate and Save Selected"))
        self._generate_button.setToolTip(
            tooltip(
                "Run the model on the checked photos, then write them with no review, exactly as "
                "Save Selected would. {summary} Change that under Save Options, in this button's "
                "menu.",
                summary=summary,
            ),
        )
        self._generate_one_button.setText(_("Generate and Save This Photo"))
        self._generate_one_button.setToolTip(
            tooltip(
                "Run the model on just this photo, then write it with no review, exactly as Save "
                "This Photo would. {summary} Change that under Save Options, in this button's "
                "menu.",
                summary=summary,
            ),
        )

    def _build_save_row(self) -> QHBoxLayout:
        """Per-photo actions at the bottom of the detail pane; batch actions live below."""
        row = QHBoxLayout()
        hint_label = QLabel(_("Hint for the AI"))
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
        hint_label.setToolTip(self._hint.toolTip())
        self._hint.returnPressed.connect(lambda: self._generate_current(trigger=TRIGGER_HINT))
        row.addWidget(hint_label)
        row.addWidget(self._hint, stretch=1)
        self._generate_one_button = QToolButton()
        self._generate_one_button.setObjectName("split")
        self._generate_one_button.setText(_("Generate This Photo"))
        self._generate_one_button.setToolTip(
            tooltip("Run the model on just this photo, regardless of which photos are checked."),
        )
        self._generate_one_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._generate_one_button.clicked.connect(
            lambda: self._generate_current(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._generate_one_menu = QMenu(self)
        self._generate_one_menu.setToolTipsVisible(True)
        skip_one = self._generate_one_menu.addAction(
            _("Generate This Photo (Skip Cache)"),
            lambda: self._generate_current(use_cache=False),
        )
        skip_one.setToolTip(tooltip("One-time: call the model even when a cached result exists."))
        # The same QAction hangs in both Generate menus, so the two can never disagree about it,
        # and the save options follow it: what the toggle writes with is one click away from the
        # toggle rather than something to know about the other button.
        self._generate_one_menu.addSeparator()
        self._generate_one_menu.addAction(self._autosave_action)
        self._generate_one_menu.addMenu(self._save_options_menu)
        self._generate_one_button.setMenu(self._generate_one_menu)
        # The save options live on the button's own arrow, so what a save writes is
        # discoverable right where the save happens (both Save buttons share one menu).
        self._save_button = QToolButton()
        self._save_button.setObjectName("split")
        self._save_button.setText(_("Save This Photo"))
        self._save_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._save_button.clicked.connect(
            lambda: self._save_current(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._save_button.setMenu(self._save_options_menu)
        row.addWidget(self._generate_one_button)
        row.addWidget(self._save_button)
        return row

    def _build_bottom_bar(self) -> QHBoxLayout:
        """Status on the left; the batch workflow (generate, then save) on the right."""
        self._status = QLabel(_("Drag photos or folders here to begin."))
        self._status.setObjectName("status")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setFormat("%v / %m")
        self._progress.setVisible(False)
        # Elapsed time (and, once a photo has finished, the estimate of what is left) beside the
        # bar. A long save used to show nothing at all, which looks like the app hung.
        self._timing = QLabel()
        self._timing.setObjectName("timing")
        self._timing.setVisible(False)
        # Repaints the readout between photos so the clock keeps moving during a slow one.
        self._timing_timer = QTimer(self)
        self._timing_timer.setInterval(_TIMING_TICK_MS)
        self._timing_timer.timeout.connect(self._update_timing)

        self._retry_button = QPushButton(_("Retry Failed"))
        self._retry_button.setToolTip(
            tooltip(
                "Re-run the model on every photo that failed to generate. Enabled once a photo "
                "has actually failed.",
            ),
        )
        self._retry_button.setEnabled(False)
        self._retry_button.clicked.connect(self._retry_failed)
        self._cancel_button = QPushButton(_("Cancel"))
        self._cancel_button.setToolTip(
            tooltip(
                "Stop the run in progress, generating or saving. The photo currently in flight "
                "finishes; the rest are left untouched so you can resume them later.",
            ),
        )
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_generation)

        # A split button: a click generates normally (cache included); the arrow offers the
        # one-time skip-cache run without changing the Settings toggle.
        self._generate_button = QToolButton()
        self._generate_button.setObjectName("primarysplit")
        self._generate_button.setText(_("Generate Selected"))
        self._generate_button.setToolTip(tooltip("Run the model on the checked photos."))
        self._generate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._generate_button.clicked.connect(
            lambda: self._generate(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        # Kept on self: Qt's menu() accessor returns a transient wrapper shiboken may delete.
        self._generate_menu = QMenu(self)
        self._generate_menu.setToolTipsVisible(True)
        skip_all = self._generate_menu.addAction(
            _("Generate Selected (Skip Cache)"),
            lambda: self._generate(use_cache=False),
        )
        skip_all.setToolTip(
            tooltip("One-time: call the model even for photos with cached results."),
        )
        self._generate_menu.addSeparator()
        self._generate_menu.addAction(self._autosave_action)
        self._generate_menu.addMenu(self._save_options_menu)
        self._generate_button.setMenu(self._generate_menu)

        self._save_selected_button = QToolButton()
        self._save_selected_button.setObjectName("primarysplit")
        self._save_selected_button.setText(_("Save Selected"))
        self._save_selected_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._save_selected_button.clicked.connect(
            lambda: self._save_selected(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._save_selected_button.setMenu(self._save_options_menu)

        row = QHBoxLayout()
        row.addWidget(self._status, stretch=1)
        row.addWidget(self._timing)
        row.addWidget(self._progress)
        row.addWidget(self._retry_button)
        row.addWidget(self._cancel_button)
        row.addWidget(self._generate_button)
        row.addWidget(self._save_selected_button)
        return row

    # --- drag and drop ---------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override.
        """Accept a drag that carries file/folder URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override.
        """Expand dropped files/folders and add the resulting photos to the tree."""
        self._count("drop")
        dropped = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        if dropped:
            self._add_inputs(dropped)

    # --- adding, removing, listing photos --------------------------------------------------

    def _choose_files(self) -> None:
        self._count("add_photos")
        filters = file_dialog_name_filters(self._extensions.text().strip())
        files, _filter = QFileDialog.getOpenFileNames(self, _("Add Photos"), "", ";;".join(filters))
        if files:
            self._add_inputs([Path(f) for f in files])

    def _choose_folder(self) -> None:
        self._count("add_folder")
        folder = QFileDialog.getExistingDirectory(self, _("Add a folder of photos"))
        if folder:
            self._add_inputs([Path(folder)])

    def _add_inputs(self, paths: list[Path]) -> list[Path]:
        """Add the photos under *paths* to the list, returning the ones that were not there yet."""
        found = expand_inputs(
            paths,
            self._extensions.text().strip(),
            recursive=self._recursive.isChecked(),
        )
        fresh = new_paths([Path(p) for p in self._items], found)
        if not fresh:
            self._update_status()
            return []
        for path in fresh:
            self._items[str(path)] = PhotoItem(path=path)
        self._usage.photos_added += len(fresh)
        self._rebuild_tree()
        self._update_status()
        self._start_metadata_scan()
        return fresh

    def _remove_selected(self) -> None:
        """Remove every selected row (files, and folders with their contents) from the list."""
        self._count("remove")
        removed: list[str] = []
        for tree_item in self._tree.selectedItems():
            path = tree_item.data(0, PATH_ROLE)
            if path is None:
                continue
            if bool(tree_item.data(0, IS_DIR_ROLE)):
                prefix = Path(path)
                removed += [k for k in self._items if Path(k).is_relative_to(prefix)]
            else:
                removed.append(path)
        self._remove_items(removed)

    def _remove_by_check(self, *, checked: bool) -> None:
        """Remove every photo whose checkbox matches *checked*, hidden by the search or not."""
        self._count("remove")
        keys = keys_by_check(self._items.values(), checked=checked)
        if not keys:
            self._status.setText(
                _("No checked photos to remove.")
                if checked
                else _("No unchecked photos to remove."),
            )
            return
        self._remove_items(keys)

    def _remove_items(self, keys: list[str]) -> None:
        """Drop the photos behind *keys* from the list and refresh the tree and grid."""
        if self._busy():
            # As for Clear: a save writes from its own job list, so the removed photos would still
            # be written, and its progress and tally would count photos the list no longer has.
            self._status.setText(_("Cancel the run in flight before removing photos."))
            return
        if not keys:
            return
        if self._scan_worker is not None:
            # Before anything slower: a scan in flight is reading the list as it stood when it
            # started, and the point of removing a folder is to stop paying for it.
            self._scan_worker.drop(Path(key) for key in keys)
        self._forget_items(keys)
        self._rebuild_tree()
        self._refresh_grid_after_removal()
        self._update_status()
        if self._scan_pending is not None and self._scan_thread is None:
            # The photos an uncheck was still waiting to have scanned are the ones just removed,
            # so nothing will finish a scan for it. Settle it now, against what is left, and after
            # the summary above so its own line is the one that stays up.
            self._end_scan(stopped=False)

    def _forget_items(self, keys: list[str]) -> None:
        """Drop each key from the list, from every cache holding its pixels, and from the view."""
        for key in keys:
            self._items.pop(key, None)
            self._scan_tried.discard(key)  # added again later, it deserves a fresh read
            self._preview_cache.pop(key, None)
            self._thumb_cache.pop(key, None)
            if self._viewer_cache is not None and self._viewer_cache[0] == key:
                self._viewer_cache = None
            if self._current is not None and str(self._current.path) == key:
                self._show_empty()

    def _refresh_grid_after_removal(self) -> None:
        """
        Rebuild a visible grid, or close it once its folder has emptied out.

        Left alone, the grid still holds the removed thumbnails: dead items whose clicks and
        checkboxes map to nothing.
        """
        if self._grid_folder is None or self._right.currentIndex() != _PAGE_GRID:
            return
        if paths_under(self._item_paths(), self._grid_folder):
            self._show_grid(self._grid_folder)
        else:
            self._grid_folder = None
            self._show_empty()

    def _clear(self) -> None:
        """Empty the list, unless a run is still working through it."""
        self._count("clear_list")
        if self._busy():
            # A save writes from its own job list, so clearing under it would keep writing photos
            # the window no longer knows about and then report every one of them as unsaved.
            self._status.setText(_("Cancel the run in flight before clearing the list."))
            return
        self._stop_thumbs()
        self._stop_scan()
        self._end_scan(stopped=True)  # the list is gone, so nothing is left to uncheck in it
        self._items.clear()
        self._scan_tried.clear()
        self._preview_cache.clear()
        self._thumb_cache.clear()
        self._viewer_cache = None
        self._grid.clear()
        self._grid_items = {}
        self._grid_folder = None
        self._current = None
        self._rebuild_tree()
        self._show_empty()
        self._status.setText(_("Drag photos or folders here to begin."))
        self._retry_button.setEnabled(False)

    def _deselect(self, paths: set[Path]) -> int:
        """Uncheck the matched photos and refresh the tree; return how many changed."""
        changed = deselect_paths(self._items, paths)
        if changed:
            self._rebuild_tree()
        return changed

    def _deselect_tagged(self, criterion: TaggedCriterion) -> None:
        """
        Uncheck photos that already carry the fields *criterion* asks for.

        The fields come from the same background scan that fills the Tagged column, so the action
        agrees with what the list shows. When the scan has not covered every photo yet the criterion
        is parked and applied the moment it does: reading a few thousand photos off a network share
        takes minutes, and doing it in the click handler froze the whole window.
        """
        self._count("uncheck_tagged")
        if not self._items:
            self._status.setText(_("Add photos before deselecting."))
            return
        if all(item.known_fields is not None for item in self._items.values()):
            self._apply_tagged_criterion(criterion)
            return
        self._scan_pending = criterion  # a second pick replaces one that is still waiting
        self._status.setText(
            _("Reading metadata before unchecking photos with {phrase}...").format(
                phrase=_(criterion.phrase),
            ),
        )
        unknown = [key for key, item in self._items.items() if item.known_fields is None]
        if all(key in self._scan_tried for key in unknown):
            # Nothing left that a scan can read: settle on what is known, unknown photos kept.
            self._end_scan(stopped=False)
            return
        self._start_metadata_scan()
        self._show_scan_row()  # the user is waiting on this scan, so do not hold the bar back

    def _apply_tagged_criterion(self, criterion: TaggedCriterion) -> None:
        """Uncheck every scanned photo matching *criterion* and report what that changed."""
        presence = {
            item.path: item.known_fields
            for item in self._items.values()
            if item.known_fields is not None
        }
        changed = self._deselect(criterion.matches(presence))
        phrase = _(criterion.phrase)
        if changed:
            self._status.setText(
                ngettext(
                    "Deselected {n} photo with {phrase}; {selected} still selected.",
                    "Deselected {n} photos with {phrase}; {selected} still selected.",
                    changed,
                ).format(n=changed, phrase=phrase, selected=self._selected_count()),
            )
        else:
            self._status.setText(_("No checked photos have {phrase}.").format(phrase=phrase))

    def _deselect_from_file(self) -> None:
        """Pick a skip-list file and uncheck the photos it names."""
        self._count("skip_file")
        if not self._items:
            self._status.setText(_("Add photos before deselecting."))
            return
        chosen, _filter = QFileDialog.getOpenFileName(self, _("Choose a skip-list file"))
        if chosen:
            self._apply_skip_file(Path(chosen))

    def _apply_skip_file(self, skip_file: Path) -> None:
        """Uncheck every photo whose name or path is listed in *skip_file*."""
        try:
            entries = load_skip_list(skip_file)
        except DiscoveryError as exc:
            QMessageBox.warning(self, _("Could not read the skip list"), str(exc))
            return
        if not entries:
            # The file read fine but had nothing usable (empty, blank lines, or only comments).
            # Say so, rather than the ambiguous "Deselected 0" a real no-match would also show.
            self._status.setText(
                _("That skip list had no usable entries (empty or only comments)."),
            )
            return
        matched = skip_list_matches([item.path for item in self._items.values()], entries)
        changed = self._deselect(matched)
        if changed:
            self._status.setText(
                ngettext(
                    "Deselected {n} photo from the skip list; {selected} still selected.",
                    "Deselected {n} photos from the skip list; {selected} still selected.",
                    changed,
                ).format(n=changed, selected=self._selected_count()),
            )
        else:
            self._status.setText(_("No photos in the list matched the skip list."))

    def _export_csv(self) -> None:
        """Write a CSV report of every photo in the list to a chosen path."""
        self._count("export_csv")
        if not self._items:
            self._status.setText(_("Add photos before exporting a CSV."))
            return
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            _("Export CSV report"),
            "photo-tagger-report.csv",
            _("CSV files (*.csv)"),
        )
        if not chosen:
            return
        target = Path(chosen)
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        # Fold any unsaved edits in the open photo into its row before exporting.
        self._commit_current()
        overwrite = self._save_options_menu.overwrites(FIELD_KEYWORDS)
        verbatim = self._verbatim_spellings()
        rows = [
            photo_item_to_report_row(item, overwrite=overwrite, verbatim=verbatim)
            for item in self._items.values()
        ]
        try:
            write_report(target, rows)
        except OSError as exc:
            QMessageBox.warning(self, _("Could not write the CSV"), str(exc))
            return
        self._status.setText(
            ngettext(
                "Exported {n} photo to {name}.",
                "Exported {n} photos to {name}.",
                len(rows),
            ).format(n=len(rows), name=target.name),
        )

    def _empty_tree(self) -> None:
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
        while self._tree.topLevelItemCount():
            self._tree.takeTopLevelItem(0)

    def _rebuild_tree(self) -> None:
        self._syncing = True
        # Build with sorting off so items do not shuffle on every insert; re-enabling at the
        # end re-applies whatever column/direction the header is currently set to.
        self._tree.setSortingEnabled(False)
        self._empty_tree()
        paths = self._visible_paths()
        if self._flat_list():
            self._add_flat_rows(paths)
        else:
            for node in build_tree(paths):
                self._add_folder_node(self._tree, node)
        self._tree.setSortingEnabled(True)
        self._sync_grid_checks()
        # Photos that just left the list have to leave the history with them, or Back would try to
        # open a row that no longer exists. Folders go too when their last photo did.
        self._history.prune(self._location_exists)
        self._refresh_nav()
        # Emptying the tree dropped the selection (_on_current_changed ignores it while _syncing,
        # so the right-hand pane kept whatever was open). Restore the highlight so bulk actions
        # (dragging photos in, Check All, Uncheck Already Tagged) do not kick the user out of
        # the photo or folder they are reviewing.
        if self._current is not None:
            self._select_tree_entry(str(self._current.path), is_dir=False)
        elif self._grid_folder is not None and self._right.currentIndex() == _PAGE_GRID:
            self._select_tree_entry(str(self._grid_folder), is_dir=True)
        self._syncing = False

    def _select_tree_entry(self, path: str, *, is_dir: bool) -> None:
        """Re-highlight the tree row for *path* (callers hold ``_syncing``)."""
        rows = self._folder_rows if is_dir else self._leaf_rows
        entry = rows.get(path)
        if entry is not None:
            self._tree.setCurrentItem(entry)

    def _add_flat_rows(self, paths: list[Path]) -> None:
        """Add every photo as a top-level row, so a column sort runs across the whole list."""
        labels = flat_labels(paths)
        for path in paths:
            leaf = self._add_leaf(self._tree, path, labels[path])
            # The label is relative to the folder the photos share, so the full path is worth
            # a tooltip here in a way it is not under a folder row that already names it.
            leaf.setToolTip(COL_NAME, str(path))

    def _add_folder_node(self, parent: object, node: FolderNode) -> None:
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
        self._render_status_cells(leaf, item)
        return leaf

    # --- tree interaction ------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        self._syncing = True
        if bool(item.data(0, IS_DIR_ROLE)):
            self._set_descendants_checked(item, item.checkState(0))
        else:
            path = item.data(0, PATH_ROLE)
            if path is not None:
                self._items[path].selected = item.checkState(0) == Qt.CheckState.Checked
        # Whichever kind of row was toggled, every folder above it may have gone mixed.
        self._sync_ancestors(item)
        self._sync_grid_checks()
        self._syncing = False
        self._update_status()

    def _set_descendants_checked(self, folder_item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for index in range(folder_item.childCount()):
            child = folder_item.child(index)
            child.setCheckState(0, state)
            if bool(child.data(0, IS_DIR_ROLE)):
                self._set_descendants_checked(child, state)
            else:
                path = child.data(0, PATH_ROLE)
                if path is not None:
                    self._items[path].selected = state == Qt.CheckState.Checked

    def _sync_ancestors(self, item: QTreeWidgetItem) -> None:
        """Re-derive the tristate of every folder above *item* (callers hold ``_syncing``)."""
        parent = item.parent()
        while parent is not None:
            self._sync_folder_check(parent)
            parent = parent.parent()

    def _sync_folder_check(self, folder_item: QTreeWidgetItem) -> None:
        states = {folder_item.child(i).checkState(0) for i in range(folder_item.childCount())}
        if states == {Qt.CheckState.Checked}:
            folder_item.setCheckState(0, Qt.CheckState.Checked)
        elif states == {Qt.CheckState.Unchecked}:
            folder_item.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            folder_item.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def _on_current_changed(self, current: QTreeWidgetItem | None, _previous: object) -> None:
        if self._syncing:
            # A programmatic rebuild (tree.clear() emits currentItemChanged(None)) is not the
            # user navigating; reacting would blank the pane they are working in.
            return
        path = current.data(0, PATH_ROLE) if current is not None else None
        if current is None or path is None:
            self._commit_current()
            self._stop_thumbs()
            self._show_empty()
            return
        self._open_location(Location(path=Path(path), is_dir=bool(current.data(0, IS_DIR_ROLE))))

    def _open_location(self, location: Location) -> None:
        """
        Show *location* in the right-hand pane and add it to the history trail.

        Both kinds of navigation land here: picking a row in the tree, and stepping through the
        trail with Back or Forward. A step has already moved the trail's cursor, so recording it
        again is a no-op and the two paths cannot fight over the history.
        """
        # Keep the open photo's in-progress edits (title, description, keywords, hint) when the
        # user browses away; they are restored when it is opened again. Without this, hinting
        # several photos before one Generate Selected would be impossible: every navigation
        # would drop the hint just typed.
        self._commit_current()
        self._history.visit(location)
        if location.is_dir:
            # A folder: show its thumbnail grid instead of a single photo's detail.
            self._current = None
            self._show_detail(enabled=False)
            self._show_grid(location.path)
        else:
            self._stop_thumbs()
            self._right.setCurrentIndex(_PAGE_DETAIL)
            self._current = self._items[str(location.path)]
            self._show_item(self._current)
        self._refresh_nav()

    # --- back, forward, and up -------------------------------------------------------------

    def _go_back(self) -> None:
        """Return to the place shown before this one."""
        self._count("go_back")
        self._walk_history(self._history.back)

    def _go_forward(self) -> None:
        """Undo a Back: return to the place it stepped away from."""
        self._count("go_forward")
        self._walk_history(self._history.forward)

    def _walk_history(self, step: Callable[[], Location | None]) -> None:
        """
        Show the next place *step* offers, skipping any that has since left the list.

        Places leave the history as their photos leave the list (see ``_rebuild_tree``), so the skip
        is only a safety net.
        """
        while (location := step()) is not None:
            if self._location_exists(location):
                self._show_location(location)
                return
        self._refresh_nav()

    def _go_up(self) -> None:
        """Show the grid of the folder holding what is open, the way Finder goes up a level."""
        self._count("go_up")
        location = self._history.current
        folder = self._enclosing_folder(location) if location is not None else None
        if folder is None:
            self._status.setText(_("Nothing above this in the list."))
            return
        self._show_location(Location(path=folder, is_dir=True))

    def _show_location(self, location: Location) -> None:
        """
        Open *location*, highlighting its tree row when it has one.

        The highlight is set without re-entering ``_on_current_changed``, since the pane is opened
        right below. A folder often has no row to highlight: the flat list has none at all, and the
        nested one can collapse a single-child chain away, so the selection is cleared instead.
        """
        self._syncing = True
        try:
            self._tree.setCurrentItem(self._row_for(location))
        finally:
            self._syncing = False
        self._open_location(location)

    def _row_for(self, location: Location) -> QTreeWidgetItem | None:
        """Return the tree row *location* stands for, or None when it has none."""
        rows = self._folder_rows if location.is_dir else self._leaf_rows
        return rows.get(str(location.path))

    def _location_exists(self, location: Location) -> bool:
        """
        Whether *location* is still something the window can open.

        Asked of the photos, not of the tree rows: a folder is a real place whenever the list holds
        anything under it, whether or not this view happens to draw a row for it.
        """
        if location.is_dir:
            return bool(paths_under(self._item_paths(), location.path))
        return str(location.path) in self._items

    def _enclosing_folder(self, location: Location) -> Path | None:
        """
        Return the folder whose grid sits above *location*, or None at the top of the list.

        In the nested tree that is the nearest ancestor with a row of its own, since single-child
        folder chains are collapsed away. The flat list draws no folder rows, so there the path
        itself answers: a photo's own folder always holds it.
        """
        folder = location.path.parent
        if folder == location.path:  # already at the filesystem root
            return None
        if self._flat_list():
            return folder
        while True:
            if str(folder) in self._folder_rows:
                return folder
            if folder.parent == folder:
                return None
            folder = folder.parent

    def _refresh_nav(self) -> None:
        """Enable Back and Forward, name where each one leads, and say what is open now."""
        behind = self._history.peek_back()
        ahead = self._history.peek_forward()
        self._apply_nav_state(
            self._back_button,
            self._back_action,
            target=behind,
            hint=(
                tooltip("Back to {place}.", place=location_crumb(behind))
                if behind is not None
                else tooltip("Nothing to go back to yet.")
            ),
        )
        self._apply_nav_state(
            self._forward_button,
            self._forward_action,
            target=ahead,
            hint=(
                tooltip("Forward to {place}.", place=location_crumb(ahead))
                if ahead is not None
                else tooltip("Nothing to go forward to. Go back first.")
            ),
        )
        here = self._history.current
        self._crumb.setText(location_crumb(here) if here is not None else "")

    @staticmethod
    def _apply_nav_state(
        button: QToolButton,
        action: QAction,
        *,
        target: Location | None,
        hint: str,
    ) -> None:
        """Point one arrow (button and menu entry alike) at *target*, or gray it out."""
        for control in (button, action):
            control.setEnabled(target is not None)
            control.setToolTip(hint)

    # --- folder thumbnail grid -------------------------------------------------------------

    def _show_grid(self, folder: Path) -> None:
        if folder != self._grid_folder:  # a refresh of the grid on show is not a new visit
            self._count("grid")
        self._stop_thumbs()
        self._grid_folder = folder
        self._grid.clear()
        self._grid_items = {}
        under = [self._items[str(path)] for path in paths_under(self._item_paths(), folder)]
        items = sort_photos(
            filter_photos(under, self._grid_filter),
            self._grid_sort,
            descending=self._grid_sort_desc,
        )
        pending: list[Path] = []
        self._syncing = True
        for item in items:
            key = str(item.path)
            grid_item = QListWidgetItem(item.path.name)
            grid_item.setData(PATH_ROLE, key)
            grid_item.setFlags(grid_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._grid.addItem(grid_item)
            self._grid_items[key] = grid_item
            self._update_grid_item(item, grid_item)
            if key not in self._thumb_cache:
                pending.append(item.path)
        self._syncing = False
        self._right.setCurrentIndex(_PAGE_GRID)
        self._status.setText(self._grid_status_text(folder, shown=len(items), total=len(under)))
        if pending:
            self._start_thumbs(pending)

    def _item_paths(self) -> list[Path]:
        """Return the path of every photo currently in the list."""
        return [item.path for item in self._items.values()]

    def _grid_status_text(self, folder: Path, *, shown: int, total: int) -> str:
        """Describe the grid's contents, noting when a filter is hiding some of the folder."""
        name = folder.name or folder
        if shown == total:
            return ngettext("{n} photo in {folder}.", "{n} photos in {folder}.", total).format(
                n=total,
                folder=name,
            )
        return _("Showing {shown} of {total} in {folder}.").format(
            shown=shown,
            total=total,
            folder=name,
        )

    def _update_grid_item(self, item: PhotoItem, grid_item: QListWidgetItem) -> None:
        """Refresh a grid thumbnail: image (or placeholder), state badges, and checkbox."""
        # Hold _syncing across the whole refresh: setIcon and setToolTip emit itemChanged just
        # like setCheckState, and an unguarded emission flips _grid_check_toggled, which then
        # swallows the user's next real click on a thumbnail (plus an O(n) tree walk per
        # streamed-in thumbnail).
        was_syncing = self._syncing
        self._syncing = True
        try:
            grid_item.setCheckState(check_state(item.selected))
            key = str(item.path)
            base = self._thumb_cache.get(key, self._placeholder_pixmap)
            badges = thumb_badges(item, has_sidecar=item.path.with_suffix(".xmp").exists())
            grid_item.setIcon(QIcon(_badged_pixmap(base, badges)))
            notes = [_(_BADGE_TEXT[name]) for name in badges]
            if item.status == FAILED and item.error:
                notes.append(item.error)
            grid_item.setToolTip("\n".join([item.path.name, *notes]))
        finally:
            self._syncing = was_syncing

    def _on_thumb_activated(self, item: QListWidgetItem) -> None:
        if self._grid_check_toggled:
            # This click landed on the checkbox: the toggle already happened, and jumping to
            # the detail page would yank the user out of the grid they are working in.
            self._grid_check_toggled = False
            return
        if selection_modifiers_active():
            # Shift/Cmd clicks are building a multi-selection; navigating away would kill it.
            return
        key = item.data(PATH_ROLE)
        leaf = self._leaf_for(Path(key))
        if leaf is not None:
            self._tree.setCurrentItem(leaf)  # routes to the detail page

    def _on_thumb_ready(self, path: str, data: bytes) -> None:
        item = self._items.get(path)
        if item is None:
            # Removed from the list while it was decoding, which a detached job makes likelier.
            # Caching it would strand a pixmap that nothing prunes.
            return
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        self._thumb_cache[path] = pixmap
        grid_item = self._grid_items.get(path)
        if grid_item is not None:
            self._update_grid_item(item, grid_item)

    def _start_thumbs(self, paths: list[Path]) -> None:
        self._thumb_thread = QThread(self)
        self._thumb_worker = ThumbnailWorker(paths)
        self._thumb_worker.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.ready.connect(self._on_thumb_ready)
        # Stop the thread's event loop once every thumbnail is decoded; without this the idle
        # thread would keep running until the next navigation or the window closed.
        self._thumb_worker.finished.connect(self._thumb_thread.quit)
        self._thumb_thread.start()

    def _stop_thumbs(self, *, blocking: bool = False) -> None:
        """
        Ask the thumbnail job to stop, waiting only briefly for it.

        The worker checks the stop flag between photos, so a decode already under way has to finish
        first: seconds, for a big RAW. Waiting that out froze the window on every click that leaves
        a loading folder, and with the Untagged filter on, every metadata scan batch re-enters here.
        Detach instead, the way _stop_scan does. A detached job only feeds _on_thumb_ready, which
        keys everything by path, so a late thumbnail is still the right thumbnail.

        closeEvent passes ``blocking``: the window owns the QThread, so leaving one running past the
        teardown would destroy it mid-run.
        """
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
            self._thumb_worker.deleteLater()
        if self._thumb_thread is not None:
            thread = self._thumb_thread
            thread.quit()
            if blocking:
                thread.wait()
                thread.deleteLater()
            elif thread.wait(_THUMB_STOP_TIMEOUT_MS):
                thread.deleteLater()
            else:
                logger.debug("gui_thumbnail_stop_detached")
                thread.finished.connect(thread.deleteLater)
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
        self._refresh_planned_copyright()
        self._title.setText(item.title)
        self._description.setPlainText(item.description)
        self._keywords.setPlainText(keywords_to_text(item.keywords))
        self._hint.setText(item.hint)
        self._review_note.setText(foreign_terms_note(item.foreign_terms))
        self._show_detail(enabled=True)
        self._update_error_banner(item)
        self._refresh_derived()

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

    def _update_error_banner(self, item: PhotoItem) -> None:
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

    def _ensure_loaded(self, item: PhotoItem) -> None:
        if item.loaded:
            return
        # Every caption value, not just the winning one: the pane names what a sidecar shadows.
        found = read_caption_values(item.path)
        context = read_image_context(
            item.path,
            field_tags=field_tags_to_read(self._copyright_dialog.fields),
        )
        item.existing_caption_sources = found
        title = first_caption(found[FIELD_TITLE])
        description = first_caption(found[FIELD_DESCRIPTION])
        item.existing_title = title
        item.existing_description = description
        item.existing_keywords = context.existing_keywords
        item.existing_copyright = context.existing_copyright
        item.field_sources = context.field_sources
        item.loaded = True
        if not item.has_proposal:
            # Seed the editable copy from the existing values so a file can be edited and saved
            # without generating first. Same rendering as the Existing column, so the two match.
            item.title = title or ""
            item.description = description or ""
            item.keywords = keyword_lines(context.existing_keywords)

    def _ensure_sources(self, item: PhotoItem) -> None:
        if item.sources_read:
            return
        item.existing_sources = read_metadata_sources(item.path)
        item.sources_read = True

    def _render_preview(self, item: PhotoItem) -> None:
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

    def _open_viewer(self) -> None:
        """Open the photo on show in a window of its own, big enough to zoom into."""
        self._count("viewer")
        item = self._current
        if item is None:
            return
        # Decoding a RAW file at _VIEWER_MAX takes about a second, which is short enough to just
        # wait through and long enough that the window must say it is working.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            pixmap = self._viewer_pixmap(item)
        finally:
            QApplication.restoreOverrideCursor()
        if pixmap is None or pixmap.isNull():
            self._status.setText(_("That photo cannot be opened for viewing."))
            return
        viewer = ImageViewerDialog(self, item.path, pixmap)
        viewer.resize(self.size())
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

    def _refresh_derived(self) -> None:
        """Recompute the keyword-change diff and hierarchy preview from the edited fields."""
        if self._current is None:
            return
        if not self._save_options_menu.write_keywords.isChecked():
            # Keywords are not being written, so the diff and hierarchy do not apply.
            self._diff.setHtml(_("(keywords will not be written)"))
            self._hierarchy.setPlainText(_(NONE_LABEL))
            self._details_toggle.setText(_("Keyword changes (not written)"))
            return
        edited = parse_keyword_lines(self._keywords.toPlainText())
        overwrite = self._save_options_menu.overwrites(FIELD_KEYWORDS)
        existing = self._current.existing_keywords
        verbatim = self._verbatim_spellings()
        paths = hierarchy_preview(existing, edited, overwrite=overwrite, verbatim=verbatim)
        self._hierarchy.setPlainText(paths or _(NONE_LABEL))
        diff = keyword_diff(existing, edited, overwrite=overwrite, verbatim=verbatim)
        self._diff.setHtml(keyword_diff_html(diff))
        # A collapsed section still tells the user whether saving changes anything.
        added = sum(1 for _kw, state in diff if state == ADDED)
        removed = sum(1 for _kw, state in diff if state == REMOVED)
        summary = f"+{added} / -{removed}" if added or removed else _("no change")
        self._details_toggle.setText(_("Keyword changes ({summary})").format(summary=summary))

    def _show_detail(self, *, enabled: bool) -> None:
        for widget in (
            self._title,
            self._description,
            self._keywords,
            self._hint,
            self._save_button,
            self._generate_one_button,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self._error_banner.hide()

    def _show_empty(self) -> None:
        """Show the idle placeholder page instead of an empty detail form, and clear selection."""
        self._current = None
        self._show_detail(enabled=False)
        self._empty_message.setText(_(_EMPTY_PICK) if self._items else _(_EMPTY_START))
        self._right.setCurrentIndex(_PAGE_EMPTY)
        # Nothing is open, so Back should return to the last place rather than step past it.
        self._history.leave()
        self._refresh_nav()

    def _commit_current(self) -> None:
        """Copy the visible editable fields back onto the selected item."""
        item = self._current
        if item is None:
            return
        item.title = self._title.text().strip()
        item.description = self._description.toPlainText().strip()
        item.keywords = parse_keyword_lines(self._keywords.toPlainText())
        item.hint = self._hint.text().strip()

    def _save_options(self) -> SaveOptions:
        """Snapshot the save toggles, so a background write never has to read a widget."""
        return self._save_options_menu.options(
            copyright_fields=self._copyright_dialog.fields,
            verbatim=self._verbatim_spellings(),
        )

    def _verbatim_spellings(self) -> dict[str, str] | None:
        """
        Return the vocabulary's own spelling of every term, or None when there is no vocabulary.

        A save merges keywords through the same step the CLI does, which title-cases anything fully
        lower case. A catalog that writes "gegenlicht" means it, so its spellings are handed over
        untouched; without this the window would write the near-duplicate the vocabulary prevents.
        """
        vocabulary = self._keyword_rules_dialog.vocabulary
        return vocabulary.exact if vocabulary else None

    def _write_fields_chosen(self) -> bool:
        """Report whether a save would write anything: a picked field, or the copyright block."""
        return self._save_options().writes_anything

    def _open_undo_journal(self) -> UndoJournal | None:
        """
        Open a journal for the batch about to be written, or None when recording is switched off.

        One journal per batch, the way the CLI writes one per run: undoing then puts back exactly
        the batch you regret rather than everything this window has ever written. The file is only
        created once something is actually recorded.
        """
        return open_journal(
            datetime.now(tz=UTC),
            enabled=self._undo_log_action.isChecked(),
        )

    def _single_save_journal(self) -> UndoJournal | None:
        """
        Return the journal that photo-by-photo saves share, opening it on the first one.

        A batch save is a run and gets a journal of its own, but clicking Save on one photo at a
        time is not fifty runs: journals are pruned to the fifty most recent, so a session spent
        reviewing photo by photo would push every command-line run out of the undo list. All of a
        session's single saves therefore land in one journal, which is also what "undo what I did
        just now" means when that is how you were working.
        """
        if not self._undo_log_action.isChecked():
            return None
        if self._session_journal is None:
            self._session_journal = self._open_undo_journal()
        return self._session_journal

    def _copyright_tags_for(self, options: SaveOptions, item: PhotoItem) -> dict[str, str]:
        """
        Resolve one photo's copyright tags for the inline (single photo) save path.

        The tags are read fresh, right before the write, which only happens when a copyright is
        configured at all. The batch path reads the whole batch at once instead.
        """
        fields = options.extra_fields
        if fields is None or fields.is_empty:
            return {}
        return resolve_extra_tags(
            options,
            item.path,
            read_field_sources([item.path], fields.source_tags),
        )

    def _write_item(self, item: PhotoItem) -> bool:
        """
        Write the item's checked fields to disk; return success.

        Unchecked fields stay as is. This is the inline path for saving one photo; a whole batch
        goes through :class:`SaveWorker` instead.
        """
        options = self._save_options()
        job = build_save_job(item, options)
        try:
            extra = self._copyright_tags_for(options, item)
            ok, targets = (
                (True, [])
                if writes_nothing(job, extra)
                else write_metadata_everywhere(
                    job.path,
                    job.keywords,
                    description=job.description,
                    title=job.title,
                    extra=extra,
                    backup=options.backup,
                    sidecar_mode=options.sidecar_mode,
                    sidecar_overrides=options.sidecar_overrides,
                )
            )
        except Exception as exc:  # noqa: BLE001 - exiftool itself failing to start must surface
            # as a failed save, not an uncaught exception Qt swallows into the log unseen.
            logger.exception("gui_save_single_failed", error=str(exc), file=str(job.path))
            ok, targets = False, []
        if (journal := self._single_save_journal()) is not None:
            journal.record_writes(job.path, targets, backup=options.backup)
        self._apply_write_result(item, job, ok=ok)
        return ok

    def _apply_write_result(self, item: PhotoItem, job: SaveJob, *, ok: bool) -> None:
        """Fold one finished write into its item: the status word and the Tagged column."""
        apply_write_result(item, job, ok=ok)
        self._refresh_status_cell(item)

    def _save_current(self) -> None:
        self._count("save_one")
        if self._current is None or self._busy():
            return
        if not self._write_fields_chosen():
            self._status.setText(_(PICK_A_FIELD))
            return
        self._commit_current()
        ok = self._write_item(self._current)
        self._usage.saved_single(self._current, ok=ok, now=time.monotonic())
        name = self._current.path.name
        self._status.setText(
            _("Saved {name}.").format(name=name)
            if ok
            else _("Failed to save {name}.").format(name=name),
        )
        self._resort()
        self._update_status()

    def _save_selected(self) -> None:
        self._count("save")
        if not self._write_fields_chosen():
            self._status.setText(_(PICK_A_FIELD))
            return
        self._commit_current()
        options = self._save_options()
        targets = photos_to_save(self._items.values(), options)
        if not targets:
            self._status.setText(
                _(
                    "No checked photos have a proposal to save. To stamp only a copyright, fill in "
                    "Settings > Copyright.",
                ),
            )
            return
        self._run_save(targets)

    def _run_save(self, items: list[PhotoItem]) -> None:
        """
        Write *items* on a background thread, reporting each file as it lands.

        Writing hundreds of photos inline blocked the event loop for minutes: the window stopped
        repainting and the OS showed its busy cursor, which is indistinguishable from a hang. The
        work now runs off the UI thread behind the same progress bar and clock a generation run
        uses.
        """
        if self._busy():
            return
        if self._usage.run is None:
            self._start_run(JOB_SAVE, TRIGGER_BATCH, items)
        else:
            # The save a finished generation chains: part of that run.
            self._usage.run.save_begun(time.monotonic())
        options = self._save_options()
        jobs = [build_save_job(item, save_options_for(item, options)) for item in items]
        self._save_jobs = {str(job.path): job for job in jobs}
        self._saved_ok = 0
        for item in items:
            # A stamp on a photo that was never generated keeps its status (see reports_status),
            # so a cancelled save cannot turn a FAILED photo into a PENDING one either.
            if reports_status(item, self._save_jobs[str(item.path)]):
                item.status = WORKING
                self._refresh_status_cell(item)
        self._cancelling = False
        self._set_running(running=True, total=len(jobs))
        self._status.setText(
            ngettext("Saving {n} photo...", "Saving {n} photos...", len(jobs)).format(n=len(jobs)),
        )

        self._save_thread = QThread(self)
        self._save_journal = self._open_undo_journal()
        self._save_worker = SaveWorker(jobs, options=options, journal=self._save_journal)
        self._save_worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.file_done.connect(self._on_save_done)
        self._save_worker.finished.connect(self._on_save_finished)
        self._save_thread.start()

    def _on_save_done(self, path: str, ok: bool) -> None:  # noqa: FBT001 - Qt signal argument
        """Fold one finished write into its item and tick the progress bar."""
        item = self._items.get(path)
        job = self._save_jobs.get(path)
        if item is None or job is None:
            return
        stamp_only = not reports_status(item, job)
        kept = self._save_worker is not None and path in self._save_worker.kept
        self._usage.saved(item, ok=ok, now=time.monotonic(), stamp_only=stamp_only, kept=kept)
        self._apply_write_result(item, job, ok=ok)
        self._saved_ok += int(ok)
        self._advance_progress()
        self._update_status()

    def _on_save_finished(self) -> None:
        """Report the batch tally and put the window back into its idle state."""
        if self._closing:
            # closeEvent already tore the thread down; this queued signal must not touch widgets
            # on a window that is going away.
            return
        total = len(self._save_jobs)
        if self._usage.run is not None:
            self._usage.run.save_ended(time.monotonic())
        # A cancelled save leaves the un-written photos marked WORKING; free them so they look
        # ready-again rather than stuck.
        self._reset_working()
        self._save_jobs = {}
        self._resort()
        self._teardown_save_thread()
        if self._save_journal is not None and self._save_journal.entries:
            logger.info(
                "gui_undo_journal_written",
                file=str(self._save_journal.path),
                entries=self._save_journal.entries,
            )
        self._save_journal = None
        # Whatever the generation that chained this save had to report, consumed either way so a
        # later manual save never repeats it.
        note, self._save_note = self._save_note, ""
        tally = (
            ngettext(
                "Cancelled after saving {saved} photo.",
                "Cancelled after saving {saved} photos.",
                self._saved_ok,
            ).format(saved=self._saved_ok)
            if self._cancelling
            else ngettext(
                "Saved {saved} of {n} checked photo.",
                "Saved {saved} of {n} checked photos.",
                total,
            ).format(saved=self._saved_ok, n=total)
        )
        # Last, so the tally is not overwritten by the per-file status refresh above.
        self._status.setText(" ".join(part for part in (tally, note) if part))
        cancelled = self._cancelling
        self._cancelling = False
        if cancelled:
            # _reset_working just put the photos this batch never reached back to READY, which is
            # exactly what a chained save looks for. Chaining on would rewrite the photos the
            # user stopped it from writing. The watch itself keeps running: whatever lands next
            # still queues normally.
            self._finish_run(telemetry.OUTCOME_CANCELLED)
            return
        # Same as a finished generation: the alert is for the run nobody was watching.
        self._alert_run_finished(self._status.text())
        self._continue_after_run()

    def _teardown_save_thread(self) -> None:
        """Join the save thread and release both it and its worker."""
        if self._save_thread is not None:
            self._save_thread.quit()
            self._save_thread.wait()
            self._save_thread.deleteLater()
            self._save_thread = None
        if self._save_worker is not None:
            self._save_worker.deleteLater()
        self._save_worker = None
        self._set_running(running=False)

    # --- generation ------------------------------------------------------------------------

    def _generate(self, *, use_cache: bool = True) -> None:
        self._count("generate" if use_cache else "skip_cache")
        selected = [item for item in self._items.values() if item.selected]
        if not selected:
            self._status.setText(_("Check at least one photo first."))
            return
        self._run_generation(selected, use_cache=use_cache)

    def _generate_current(self, *, use_cache: bool = True, trigger: str = TRIGGER_SINGLE) -> None:
        self._count("hint_enter" if trigger == TRIGGER_HINT else "generate_one")
        if not use_cache:
            self._count("skip_cache")
        if self._current is None:
            self._status.setText(_("Open a photo to generate it."))
            return
        self._run_generation([self._current], use_cache=use_cache, trigger=trigger)

    def _retry_failed(self) -> None:
        self._count("retry_failed")
        failed = [item for item in self._items.values() if item.status == FAILED]
        if not failed:
            self._status.setText(_("No failed photos to retry."))
            return
        self._run_generation(failed, trigger=TRIGGER_RETRY)

    def _run_generation(
        self,
        items: list[PhotoItem],
        *,
        use_cache: bool = True,
        extra_pass: str | None = None,
        refresh: bool = False,
        trigger: str = TRIGGER_BATCH,
    ) -> None:
        if self._busy() or not items:
            return
        # Fold the open photo's visible edits (most importantly a just-typed hint) into its
        # state before the run reads it.
        self._commit_current()
        if extra_pass is None:
            # An extra pass is the same batch having another go at part of itself, so it keeps
            # the batch (and the passes still owed) the chain is working through, and adds to
            # that batch's tallies rather than reporting the closing line off its own handful.
            self._generated_keys = [str(item.path) for item in items]
            self._passes_done = set()
            # The extra passes follow the batch's own choice: after Skip Cache, a retry must not
            # replay the cached answer the user chose to skip.
            self._batch_uses_cache = use_cache
            self._vocabulary_mapped = 0
            self._vocabulary_dropped = {}
            self._start_run(JOB_GENERATE, trigger, items, use_cache=use_cache)
        elif self._usage.run is not None:
            self._usage.run.pass_started(
                extra_pass,
                len(items),
                retry=extra_pass == RERUN_FAILED,
            )
        for item in items:
            item.status = WORKING
            self._refresh_status_cell(item)
        current = self._current
        if current is not None and current in items:
            # Clear a stale failure banner the moment its photo is re-queued.
            self._update_error_banner(current)
        self._cancelling = False
        self._set_running(running=True, total=len(items))
        # Kept so a pacing rest can report itself in the status line and hand it back afterwards.
        self._run_status_text = ngettext(
            "Generating {n} photo...",
            "Generating {n} photos...",
            len(items),
        ).format(n=len(items))
        self._status.setText(self._run_status_text)

        self._thread = QThread(self)
        self._worker = GenerateWorker(
            self._provider_name(),
            self._model.currentText().strip(),
            self._url.text().strip() or None,
            [item.path for item in items],
            api_key=self._api_key_value(),
            cache_file=self._active_cache_file() if use_cache else None,
            options=GenerateOptions(
                output_language=self._output_language,
                hints={str(item.path): item.hint for item in items if item.hint},
                vocabulary=self._keyword_rules_dialog.vocabulary,
                vocabulary_strict=self._keyword_rules_dialog.strict,
                repair_language=self._repair_language,
                pacing=self._pacing_dialog.plan,
                jpeg_dimensions=self._model_image_size,
                refresh=refresh,
            ),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.resting.connect(self._on_resting)
        self._worker.finished.connect(self._on_generate_finished)
        self._thread.start()

    def _on_resting(self, text: str) -> None:
        """Report a pacing rest, or put the run's own status line back once it ends."""
        self._status.setText(text or self._run_status_text)

    def _on_file_done(self, proposal: Proposal) -> None:
        item = self._items.get(str(proposal.path))
        if item is None:
            return
        # Normalizing the proposal runs the save's own merge, which must not re-capitalize a
        # catalog's terms, so the vocabulary spellings ride along.
        apply_proposal(item, proposal, verbatim=self._verbatim_spellings(), now=time.monotonic())
        self._vocabulary_mapped += proposal.vocabulary_mapped
        record_dropped_terms(self._vocabulary_dropped, proposal.vocabulary_dropped)
        self._usage.generated(str(item.path), proposal)
        self._refresh_status_cell(item)
        self._advance_progress()
        if self._current is item:
            self._show_item(item)
        self._update_status()

    def _on_file_failed(
        self,
        path: str,
        message: str,
        kind: str = FAILURE_OTHER,
        error_type: str = "",
    ) -> None:
        if self._usage.run is not None:
            self._usage.run.failed(path, kind, error_type)
        item = self._items.get(path)
        if item is None:
            return
        item.status = FAILED
        item.error = message
        self._refresh_status_cell(item)
        self._advance_progress()
        if self._current is item:
            self._update_error_banner(item)
        self._update_status()

    def _cancel_generation(self) -> None:
        """
        Ask whichever run is in flight to stop after the photo it is on.

        One Cancel button covers both runs (they never overlap): a model call and a metadata write
        are both uninterruptible once started, so cancelling takes effect at the next photo.
        """
        worker = self._worker or self._save_worker
        if worker is None:
            if self._pending_pass is not None or self._harmonize_thread is not None:
                # Between the chain's steps: nothing to stop mid-photo, so end it at the next step.
                self._count("cancel")
                self._chain_cancelled = True
                self._cancel_button.setEnabled(False)
                if self._harmonize_thread is None:
                    self._end_chain()
            return
        self._count("cancel")
        self._cancelling = True
        worker.stop()
        self._cancel_button.setEnabled(False)
        self._status.setText(_("Cancelling after the current photo finishes..."))

    def _on_generate_finished(self) -> None:
        if self._closing:
            # closeEvent already tore the thread down; this queued signal arriving afterwards
            # must not touch widgets on a window that is going away, or tear down twice.
            return
        if self._usage.run is not None and self._worker is not None:
            rested, rests = self._worker.rest_totals()
            self._usage.run.generation_ended(time.monotonic(), rested=rested, rests=rests)
        # A cancelled run leaves the un-started photos marked WORKING; free them so they look
        # queued-again rather than stuck, and report what actually got done.
        reset = self._reset_working()
        if self._cancelling:
            self._status.setText(
                ngettext(
                    "Cancelled. {n} photo not generated.",
                    "Cancelled. {n} photos not generated.",
                    reset,
                ).format(n=reset),
            )
        else:
            self._status.setText(self._generation_summary())
            # Only an unattended run is worth an alert: whoever pressed Cancel is already here.
            # An extra pass or a chained save still to come is part of the same job, so let the
            # end of that job raise the one alert rather than one per step.
            if self._next_pass() is None and not self._chained_save_targets():
                self._alert_run_finished(self._status.text())
        cancelled = self._cancelling
        self._cancelling = False
        self._resort()
        self._teardown_thread()
        if cancelled:
            # Same reasoning as the cancelled save: harmonizing and then saving unattended is
            # more work on the batch the user just stopped.
            self._finish_run(telemetry.OUTCOME_CANCELLED)
            return
        self._after_generation()

    def _generation_notes(self) -> str:
        """Report what a run has to say beyond 'it ran': failures, vocabulary, what to re-read."""
        parts = [
            failure_summary(sum(1 for item in self._batch_items() if item.status == FAILED)),
            vocabulary_summary(self._vocabulary_mapped, self._vocabulary_dropped),
            review_summary(sum(1 for item in self._items.values() if item.foreign_terms)),
        ]
        return " ".join(part for part in parts if part)

    def _generation_summary(self) -> str:
        """Build the closing line: the run finished, what the vocabulary did, what to re-read."""
        parts = (_("Generation finished."), self._generation_notes())
        return " ".join(part for part in parts if part)

    def _after_generation(self) -> None:
        """
        Ask again, harmonize, then save: the steps a finished generation still owes the batch.

        Each one hands back here (or to :meth:`_continue_after_run`) when it ends, so the chain is
        the same whether a step runs or is switched off. Harmonization is what ``--session-gap``
        does inside a CLI run; here it lands on the proposals, before the review and before any save
        that follows, so the keywords you see are the ones a save would write.
        """
        if self._chain_cancelled:
            self._end_chain()
            return
        if self._start_extra_pass():
            self._hold_for_chain()
            return  # the pass comes back through here when it ends
        if self._start_harmonize(announce=False):
            self._hold_for_chain()
            return  # _on_harmonize_finished picks the chain back up
        self._continue_after_run()

    def _hold_for_chain(self) -> None:
        """
        Keep the window busy while the chain waits (the pause before a pass, or harmonization).

        The action buttons stay off, since _busy() would ignore their clicks, and Cancel stays on to
        end the chain there.
        """
        for button in (
            self._generate_button,
            self._generate_one_button,
            self._save_button,
            self._save_selected_button,
            self._retry_button,
            self._test_button,
        ):
            button.setEnabled(False)
        self._harmonize_action.setEnabled(False)
        self._cancel_button.setEnabled(True)

    def _end_chain(self) -> None:
        """Close a chain that Cancel stopped between its steps: no pass, and no chained save."""
        self._chain_cancelled = False
        self._pending_pass = None
        self._set_running(running=False)
        self._status.setText(_("Cancelled. Nothing else was generated or saved."))
        self._finish_run(telemetry.OUTCOME_CANCELLED)
        self._continue_watch()

    def _batch_items(self) -> list[PhotoItem]:
        """List the photos of the run the chain is working through, skipping any since removed."""
        return [item for key in self._generated_keys if (item := self._items.get(key)) is not None]

    def _pass_targets(self, name: str) -> list[PhotoItem]:
        """Pick what *name* would re-run: nothing when it is spent, switched off, or has no work."""
        if name in self._passes_done or not self._wants_chained_save():
            return []
        return photos_for_pass(name, self._batch_items())

    def _next_pass(self) -> str | None:
        """Name the next extra pass this batch owes, in order, or None when it owes none."""
        return next((name for name in EXTRA_PASSES if self._pass_targets(name)), None)

    def _start_extra_pass(self) -> bool:
        """
        Queue the next extra pass this batch owes; report whether one was queued.

        Only for a run that writes by itself, where these are the two things a person would do
        before saving and nobody is there to do them: press **Retry Failed**, and generate again the
        odd photo whose text came back in two writing systems. Each pass runs once per batch, so a
        provider that is down, or a photo that really does show foreign text, cannot loop.
        """
        name = self._next_pass()
        if name is None:
            return False
        targets = self._pass_targets(name)
        self._passes_done.add(name)
        self._pending_pass = name
        self._status.setText(extra_pass_status(name, len(targets)))
        QTimer.singleShot(_RETRY_PASS_DELAY_MS, self._run_pending_pass)
        return True

    def _run_pending_pass(self) -> None:
        """Start the pass the pause was for, or pick the chain back up when it has no work left."""
        if self._closing or self._pending_pass is None:
            return  # closing, or Cancel already ended the chain during the pause
        name, self._pending_pass = self._pending_pass, None
        targets = photos_for_pass(name, self._batch_items()) if name else []
        if targets and not self._busy():
            # A mixed-script pass reads past the cache: the answer it is unhappy with is the one
            # stored there, so replaying it would not be asking again at all.
            self._run_generation(
                targets,
                use_cache=self._batch_uses_cache,
                extra_pass=name,
                refresh=name == RERUN_MIXED_SCRIPT,
            )
            return
        # Nothing left to do, or another run claimed the window during the pause. Either way the
        # chain carries on from where it was; a new run brings its own.
        self._after_generation()

    def _wants_chained_save(self) -> bool:
        """Report whether a setting asks a finished generation to write its photos by itself."""
        watching = self._watch_settings is not None and self._watch_settings.save
        return watching or self._autosave_action.isChecked()

    def _chained_save_targets(self) -> list[PhotoItem]:
        """
        Pick the photos a finished generation writes without being asked a second time.

        Save After Generating covers the run that just ended, so a proposal left over from an
        earlier one is still yours to review. A watch started with its own save box checked takes
        every unsaved proposal instead: it has no batch, only whatever has landed.
        """
        if not self._wants_chained_save() or not self._write_fields_chosen():
            return []
        watching = self._watch_settings is not None and self._watch_settings.save
        return photos_to_write(self._items.values() if watching else self._batch_items())

    def _continue_after_run(self) -> None:
        """After a finished run: write what nobody will review, then take what the watch found."""
        if self._busy():
            return
        if targets := self._chained_save_targets():
            # The save is about to replace the generation's closing line, so carry what that line
            # said into it: an unattended run is read once, when it is over.
            self._save_note = self._generation_notes()
            # Written now, so a later step in the chain cannot write them a second time.
            self._generated_keys = []
            self._run_save(targets)
            return
        if self._wants_chained_save() and not self._write_fields_chosen():
            # Silence here would mean coming back to an unattended run that wrote nothing.
            self._status.setText(_(PICK_A_FIELD))
        # Nothing more is chained, so whatever the run was, it is over.
        self._set_running(running=False)
        self._finish_run()
        self._continue_watch()

    def _reset_working(self) -> int:
        """
        Free any photo still marked WORKING; return how many were reset.

        A photo that already has a proposal goes back to READY (the proposal is still there to
        save), anything else back to PENDING. Used after a cancelled generation or save, so an
        interrupted photo reads as queued-again rather than stuck.
        """
        reset = 0
        for item in self._items.values():
            if item.status == WORKING:
                item.status = READY if item.has_proposal else PENDING
                self._refresh_status_cell(item)
                reset += 1
        return reset

    def _busy(self) -> bool:
        """
        Whether a run is in flight, counting the steps a finished generation still owes.

        A generation and a save never overlap. Neither may start in the pause before an extra pass
        or while harmonization runs: the chain would then carry on inside the new run, spend its
        pass there, and could lay the old run's harmonized keywords over the new run's photos. A
        click or a watch batch in that window waits, and the watch picks its queue up at the end.
        """
        return (
            self._thread is not None
            or self._save_thread is not None
            or self._pending_pass is not None
            or self._harmonize_thread is not None
        )

    def _has_failures(self) -> bool:
        """Report whether any photo is currently in the failed state."""
        return any(item.status == FAILED for item in self._items.values())

    def _set_running(self, *, running: bool, total: int = 0) -> None:
        """Switch the window between idle and busy: buttons, progress bar, and the clock."""
        self._generate_button.setEnabled(not running)
        self._generate_one_button.setEnabled(not running)
        self._save_button.setEnabled(not running)
        self._save_selected_button.setEnabled(not running)
        self._retry_button.setEnabled(not running and self._has_failures())
        self._test_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        # Harmonizing rewrites the proposals a run is still producing, so it waits its turn.
        self._harmonize_action.setEnabled(not running)
        if running:
            self._start_progress(total)
        else:
            self._stop_progress()

    def _start_progress(self, total: int) -> None:
        """Show the bar and start the clock for a run over *total* photos."""
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._run_started = time.monotonic()
        self._timing.setVisible(True)
        self._update_timing()
        self._timing_timer.start()

    def _stop_progress(self) -> None:
        """Hide the bar and the clock; the run is over."""
        self._timing_timer.stop()
        self._run_started = None
        self._progress.setVisible(False)
        self._timing.setVisible(False)
        self._timing.clear()

    def _update_timing(self) -> None:
        """
        Repaint the elapsed/remaining readout from the progress bar's own counters.

        Driven by a timer as well as by each finished photo, so the elapsed time keeps moving while
        a single slow photo is in flight. That movement is the point: it is what tells the user the
        program is working rather than wedged.
        """
        if self._run_started is None:
            return
        self._timing.setText(
            progress_timing_text(
                self._progress.value(),
                self._progress.maximum(),
                time.monotonic() - self._run_started,
            ),
        )

    def _advance_progress(self) -> None:
        """Tick the run progress bar for one finished (or failed) photo."""
        if self._run_started is None:
            return
        self._progress.setValue(self._progress.value() + 1)
        self._update_timing()

    def _teardown_thread(self) -> None:
        # deleteLater releases the C++ side of the thread and the (parentless, moved) worker;
        # dropping only the Python refs would leak both once per generation run.
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._set_running(running=False)

    # --- keyword rules: the vocabulary and the session gap -----------------------------------

    def _build_keyword_rules_dialog(self) -> KeywordRulesDialog:
        """Build the Keyword Rules dialog and put the configured vocabulary file in force."""
        dialog = KeywordRulesDialog(
            self,
            self._defaults.output,
            output_language=lambda: self._output_language,
        )
        dialog.status.connect(self._status.setText)
        dialog.used.connect(self._count)
        dialog.build_requested.connect(self._show_vocabulary_builder)
        # Read once the status bar exists: it is disk IO, and a refused file needs reporting.
        dialog.load_vocabulary(dialog.vocabulary_path, announce=False)
        return dialog

    def _show_keyword_rules(self) -> None:
        """Open the Keyword rules dialog on the values currently in force."""
        self._count("keyword_rules")
        self._keyword_rules_dialog.refresh()
        self._keyword_rules_dialog.exec()

    # --- pacing and copyright: the settings every run and every save follow -------------------

    def _show_pacing(self) -> None:
        """Open the Pacing dialog; a run already in flight keeps the plan it started with."""
        self._count("pacing")
        self._pacing_dialog.exec()

    def _on_copyright_changed(self) -> None:
        """Follow an edit in the Copyright dialog: the toggle, the Save tooltips, the open photo."""
        self._save_options_menu.set_copyright_usable(usable=self._copyright_dialog.usable)
        self._refresh_save_tooltips()
        self._refresh_planned_copyright()

    def _preview_year(self) -> int:
        """
        Return the year the preview stands for: the open photo's, else this one.

        A photo's capture date is already in hand once it has been opened, and showing the year of
        the photo on screen is what makes a wrong template obvious.
        """
        item = self._current
        year = capture_year(item.field_sources) if item is not None else None
        return year or datetime.now(tz=UTC).astimezone().year

    def _refresh_planned_copyright(self) -> None:
        """Say in the Copyright row what a save writes on the open photo, if anything."""
        if self._current is None:
            return
        options = self._save_options()
        fields = options.extra_fields
        sources = self._current.field_sources
        targets = save_targets(options, self._current.path)
        if keeps_copyright(fields, sources, targets):
            self._planned_copyright.setText(_("(keeps the notice it has)"))
            return
        planned = planned_copyright(fields, sources, targets)
        self._planned_copyright.setText(planned or _("(not written)"))

    def _show_copyright(self) -> None:
        """Open the Copyright dialog, its preview refreshed for the photo on screen."""
        self._count("copyright")
        self._copyright_dialog.refresh()
        self._copyright_dialog.exec()

    # --- shoot harmonization -----------------------------------------------------------------

    def _harmonize_now(self) -> None:
        """Menu action: harmonize the proposals on hand, explaining when there is nothing to do."""
        self._count("harmonize_now")
        if self._keyword_rules_dialog.session_gap <= 0:
            self._status.setText(_("Set a session gap first: Settings > Keyword Rules."))
            return
        self._start_harmonize()

    def _start_harmonize(self, *, announce: bool = True) -> bool:
        """
        Harmonize the generated proposals on a background thread; report whether it started.

        Grouping photos into shoots reads every capture time through exiftool, so it does not run in
        the click handler. Callers use the return value to know whether to wait for it.
        """
        gap = self._keyword_rules_dialog.session_gap
        if self._harmonize_thread is not None or self._busy() or gap <= 0:
            return False
        self._commit_current()
        keywords = {
            item.path: list(item.keywords) for item in self._items.values() if item.has_proposal
        }
        if not keywords:
            if announce:
                self._status.setText(harmonize_summary(HarmonizeResult()))
            return False
        self._harmonize_thread = QThread(self)
        self._harmonize_worker = HarmonizeWorker(
            keywords,
            gap_minutes=gap,
            output_language=self._output_language,
        )
        self._harmonize_worker.moveToThread(self._harmonize_thread)
        self._harmonize_thread.started.connect(self._harmonize_worker.run)
        self._harmonize_worker.done.connect(self._on_harmonize_done)
        self._harmonize_worker.finished.connect(self._on_harmonize_finished)
        self._harmonize_thread.start()
        return True

    def _on_harmonize_done(self, result: HarmonizeResult) -> None:
        """Apply the harmonized keywords to their photos and report what moved."""
        for key, keywords in result.keywords.items():
            item = self._items.get(key)
            if item is not None:
                item.keywords = list(keywords)
                # The shoot's own agreement, not a change made by hand.
                item.proposed_keywords = list(keywords)
        if self._usage.run is not None:
            self._usage.run.harmonized += len(result.keywords)
        if self._current is not None and str(self._current.path) in result.keywords:
            # The open photo's keyword box is showing what harmonization just replaced.
            self._keywords.setPlainText(keywords_to_text(self._current.keywords))
        self._status.setText(harmonize_summary(result))

    def _on_harmonize_finished(self) -> None:
        """Release the harmonization thread, then let the rest of the chain carry on."""
        if self._closing:
            return
        self._teardown_harmonize_thread()
        if self._chain_cancelled:
            self._end_chain()
            return
        self._continue_after_run()

    def _teardown_harmonize_thread(self) -> None:
        """
        Join the harmonization thread and release both it and its worker.

        Harmonizing reads every photo's capture time in one batched ExifTool call, which has no stop
        hook and takes as long as it takes: minutes over a network share. Closing the window used to
        sit out the whole of it with the UI frozen, so detach on a timeout the way the scan and the
        watch do. The worker only reads, and cutting its connections first means a detached one
        cannot report back into a window that has moved on.
        """
        if self._harmonize_worker is not None:
            self._harmonize_worker.disconnect(self)
            self._harmonize_worker.deleteLater()
        if self._harmonize_thread is not None:
            thread = self._harmonize_thread
            thread.quit()
            if thread.wait(_HARMONIZE_STOP_TIMEOUT_MS):
                thread.deleteLater()
            else:
                logger.warning("gui_harmonize_stop_timed_out")
                thread.finished.connect(thread.deleteLater)
            self._harmonize_thread = None
        self._harmonize_worker = None

    # --- building a vocabulary ---------------------------------------------------------------

    def _show_vocabulary_builder(self) -> None:
        """Open the builder, naming the photo count it would read from."""
        self._builder_dialog.show_for(len(self._items))

    # --- undo --------------------------------------------------------------------------------

    def _build_undo_dialog(self) -> QDialog:
        """Build the dialog behind Tools > Undo Writes."""
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Undo writes"))
        dialog.setMinimumWidth(660)
        box = QVBoxLayout(dialog)
        intro = QLabel(
            _(
                "Every recorded run, newest first, from this window and from the command line. "
                "Undoing deletes the sidecars a run created and restores the files it overwrote "
                "from their ExifTool backup.",
            ),
        )
        intro.setWordWrap(True)
        box.addWidget(intro)

        self._journal_list = QListWidget()
        self._journal_list.setMinimumHeight(150)
        box.addWidget(self._journal_list, stretch=1)

        self._undo_force = QCheckBox(_("Also revert files changed since the run"))
        self._undo_force.setToolTip(
            tooltip(
                "Off by default: a file that changed since the run was edited afterwards, and "
                "that edit is not this run's to undo.",
            ),
        )
        box.addWidget(self._undo_force)

        self._undo_details = readonly_box(120)
        self._undo_details.setPlaceholderText(_("What happened to each file appears here."))
        box.addWidget(self._undo_details)

        preview = QPushButton(_("Preview"))
        preview.setToolTip(
            tooltip("Report what undoing would put back, without touching anything."),
        )
        preview.clicked.connect(lambda: self._run_undo(dry_run=True))
        self._undo_button = QPushButton(_("Undo"))
        self._undo_button.setObjectName("primary")
        self._undo_button.clicked.connect(lambda: self._run_undo(dry_run=False))
        close = QPushButton(_(CLOSE_LABEL))
        close.clicked.connect(dialog.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(preview)
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(self._undo_button)
        box.addLayout(buttons)
        return dialog

    def _show_undo_dialog(self) -> None:
        """Open the undo dialog on a freshly-read list of recorded runs."""
        self._refresh_journals()
        self._undo_dialog.exec()

    def _refresh_journals(self) -> None:
        """List the recorded runs, newest first, with how many files each one wrote."""
        self._journal_list.clear()
        for path in list_journals():
            try:
                entries = len(read_journal(path))
            except UndoError as exc:  # pragma: no cover - one unreadable journal, not the list
                logger.warning("gui_journal_unreadable", file=str(path), error=str(exc))
                continue
            entry = QListWidgetItem(journal_label(path, entries))
            entry.setData(PATH_ROLE, str(path))
            self._journal_list.addItem(entry)
        listed = self._journal_list.count()
        if listed:
            self._journal_list.setCurrentRow(0)
        else:
            self._undo_details.setPlainText(_("No recorded runs to undo."))
        self._undo_button.setEnabled(bool(listed))

    def _run_undo(self, *, dry_run: bool) -> None:
        """Put back (or preview putting back) everything the selected run wrote."""
        self._count("undo_preview" if dry_run else "undo")
        if not dry_run and self._busy():
            # A save in flight is writing the very files an undo would be reverting.
            self._undo_details.setPlainText(_("Wait for the run in progress to finish first."))
            return
        entry = self._journal_list.currentItem()
        if entry is None:
            self._undo_details.setPlainText(_("Pick a run first."))
            return
        try:
            records = read_journal(Path(entry.data(PATH_ROLE)))
        except UndoError as exc:
            QMessageBox.warning(self._undo_dialog, _("Could not read that run"), str(exc))
            return
        if not records:
            self._undo_details.setPlainText(_("That run recorded no writes."))
            return
        if not dry_run and not self._confirm_undo(len(records)):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = undo_run(records, force=self._undo_force.isChecked(), dry_run=dry_run)
        finally:
            QApplication.restoreOverrideCursor()
        self._undo_details.setPlainText(undo_details_text(results, dry_run=dry_run))
        summary = undo_summary(results)
        self._status.setText(
            _("Preview: {summary}").format(summary=summary) if dry_run else summary,
        )
        if not dry_run:
            self._apply_undo_results(records, results)
            self._refresh_journals()

    def _confirm_undo(self, count: int) -> bool:
        """Ask before reverting: this rewrites files, and the run may not be the one in mind."""
        reply = QMessageBox.question(
            self._undo_dialog,
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

    def _apply_undo_results(
        self,
        records: list[WriteRecord],
        results: list[UndoResult],
    ) -> None:
        """Put the photos whose writes were reverted back into their pre-save state."""
        images = {record.target: record.image for record in records}
        touched = False
        for result in results:
            if result.action not in UNDO_OK_ACTIONS:
                continue
            item = self._items.get(images.get(str(result.target), ""))
            if item is None:
                continue
            # What is on the file changed under us, so everything read from it is stale: the
            # proposal is still worth keeping, the file's own metadata is not.
            item.status = READY if item.has_proposal else PENDING
            item.known_fields = None
            self._scan_tried.discard(str(item.path))  # the file changed, so read it again
            item.loaded = False
            item.sources_read = False
            self._refresh_status_cell(item)
            touched = True
        if not touched:
            return
        self._start_metadata_scan()
        if self._current is not None:
            self._show_item(self._current)

    # --- watching a folder ---------------------------------------------------------------------

    def _build_watch_dialog(self) -> QDialog:
        """Build the dialog behind Tools > Watch Folder."""
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Watch a folder"))
        dialog.setMinimumWidth(600)
        box = QVBoxLayout(dialog)
        intro = QLabel(
            _(
                "Point this at the folder your card reader, tethered capture, or sync client "
                "fills. Photos already there are picked up first, then each new one as it lands.",
            ),
        )
        intro.setWordWrap(True)
        box.addWidget(intro)
        box.addLayout(self._build_watch_form())

        self._watch_generate = QCheckBox(_("Generate each new photo"))
        self._watch_generate.setChecked(True)
        box.addWidget(self._watch_generate)
        self._watch_save = QCheckBox(_("Save it too, without reviewing"))
        self._watch_save.setToolTip(
            tooltip(
                "Off by default: the window is built around reviewing before writing. Turn it on "
                "for an unattended import, where it behaves like the CLI's watch command.",
            ),
        )
        box.addWidget(self._watch_save)

        start = QPushButton(_("Start Watching"))
        start.setObjectName("primary")
        start.clicked.connect(self._start_watch_from_dialog)
        cancel = QPushButton(_("Cancel"))
        cancel.clicked.connect(dialog.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(start)
        box.addLayout(buttons)
        return dialog

    def _build_watch_form(self) -> QFormLayout:
        """Build the watch dialog's form: which folder to poll, how often, and how long to wait."""
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._watch_folder = QLineEdit()
        self._watch_folder.setPlaceholderText(_("(pick a folder)"))
        choose = QPushButton(_(CHOOSE_LABEL))
        choose.clicked.connect(self._choose_watch_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self._watch_folder, stretch=1)
        folder_row.addWidget(choose)
        form.addRow(_("Folder"), folder_row)

        self._watch_recursive = QCheckBox(_("Include subfolders"))
        self._watch_recursive.setChecked(self._recursive.isChecked())
        form.addRow("", self._watch_recursive)

        self._watch_interval = QDoubleSpinBox()
        self._watch_interval.setRange(1.0, 3600.0)
        self._watch_interval.setDecimals(0)
        self._watch_interval.setValue(DEFAULT_INTERVAL_SECONDS)
        self._watch_interval.setSuffix(_(" seconds"))
        self._watch_interval.setToolTip(
            tooltip(
                "How often the folder is listed. Polling behaves the same on every platform and "
                "over network shares.",
            ),
        )
        form.addRow(_("Check every"), self._watch_interval)

        self._watch_settle = QDoubleSpinBox()
        self._watch_settle.setRange(0.0, 600.0)
        self._watch_settle.setDecimals(0)
        self._watch_settle.setValue(DEFAULT_SETTLE_SECONDS)
        self._watch_settle.setSuffix(_(" seconds"))
        self._watch_settle.setToolTip(
            tooltip(
                "How long a file must sit unchanged before it is picked up, so a photo still "
                "being copied is left alone until the copy finishes.",
            ),
        )
        form.addRow(_("Settle for"), self._watch_settle)
        return form

    def _choose_watch_folder(self) -> None:
        """Pick the folder to watch."""
        folder = QFileDialog.getExistingDirectory(
            self._watch_dialog,
            _("Choose a folder to watch"),
            self._watch_folder.text(),
        )
        if folder:
            self._watch_folder.setText(folder)

    def _toggle_watch(self) -> None:
        """Stop the running watch, or open the dialog to start one."""
        if self._watch_settings is not None:
            self._stop_watch()
            return
        self._watch_dialog.exec()

    def _start_watch_from_dialog(self) -> None:
        """Turn the dialog's choices into a watch and start it."""
        folder = self._watch_folder.text().strip()
        if not folder:
            QMessageBox.warning(
                self._watch_dialog,
                _("Pick a folder"),
                _("Choose the folder to watch first."),
            )
            return
        self._watch_dialog.accept()
        self._start_watch(
            WatchSettings(
                folders=(Path(folder),),
                extensions=self._extensions.text().strip(),
                recursive=self._watch_recursive.isChecked(),
                interval=self._watch_interval.value(),
                settle=self._watch_settle.value(),
                generate=self._watch_generate.isChecked(),
                save=self._watch_save.isChecked(),
            ),
        )

    def _start_watch(self, settings: WatchSettings) -> None:
        """Poll *settings*' folders on a background thread until the watch is stopped."""
        if self._watch_thread is not None:
            return
        self._count("watch_start")
        self._watch_settings = settings
        self._watch_added = 0
        self._watch_pending = []
        self._watch_thread = QThread(self)
        self._watch_worker = WatchWorker(settings)
        self._watch_worker.moveToThread(self._watch_thread)
        self._watch_thread.started.connect(self._watch_worker.run)
        self._watch_worker.batch.connect(self._on_watch_batch)
        self._watch_worker.finished.connect(self._on_watch_finished)
        self._watch_thread.start()
        self._watch_action.setText(_("Stop Watching"))
        self._status.setText(watch_status_text(settings, added=0))
        logger.info(
            "gui_watch_started",
            folders=[str(folder) for folder in settings.folders],
            interval=settings.interval,
            generate=settings.generate,
            save=settings.save,
        )

    def _on_watch_batch(self, paths: list[Path]) -> None:
        """Add the photos a poll found and queue them for generation when asked to."""
        settings = self._watch_settings
        if settings is None or self._closing:
            # A batch delivered while the window is going away has nothing left to be added to.
            return
        fresh = self._add_inputs(list(paths))
        if not fresh:
            return
        self._watch_added += len(fresh)
        self._status.setText(watch_status_text(settings, added=self._watch_added))
        if not settings.generate:
            return
        self._watch_pending += [str(path) for path in fresh]
        if not self._busy():
            self._generate_watched()

    def _generate_watched(self) -> None:
        """Run the model on the photos the watch queued, skipping any already dealt with."""
        if self._busy():
            # Draining the queue now would lose it: _run_generation refuses to start mid-run.
            return
        queued, self._watch_pending = self._watch_pending, []
        targets = [
            item
            for key in queued
            if (item := self._items.get(key)) is not None and item.status == PENDING
        ]
        if targets:
            self._run_generation(targets, trigger=TRIGGER_WATCH)

    def _continue_watch(self) -> None:
        """Take whatever the watch picked up while the last run was busy."""
        if self._watch_settings is None or self._busy() or not self._watch_pending:
            return
        self._generate_watched()

    def _on_watch_finished(self) -> None:
        """Clean up after a watch that ended on its own (an error, or a stop already asked for)."""
        if self._closing:
            return
        self._stop_watch()

    def _stop_watch(self) -> None:
        """End the watch and put the menu action back to offering a new one."""
        if self._watch_settings is None and self._watch_thread is None:
            # Already stopped: the worker's own finished signal arrives just after a Stop click,
            # and re-running this would overwrite whatever the status bar says by then.
            return
        self._count("watch_stop")
        if self._watch_worker is not None:
            self._watch_worker.stop()
        self._teardown_watch_thread()
        self._watch_settings = None
        self._watch_pending = []
        if not self._closing:
            self._watch_action.setText(_("Watch Folder..."))
            self._status.setText(_("Stopped watching."))

    def _teardown_watch_thread(self) -> None:
        """Join the watch thread and release both it and its worker."""
        if self._watch_worker is not None:
            self._watch_worker.deleteLater()
        if self._watch_thread is not None:
            thread = self._watch_thread
            thread.quit()
            if thread.wait(_WATCH_STOP_TIMEOUT_MS):
                thread.deleteLater()
            else:
                # Still inside a poll: a huge folder, or a network share that is not answering.
                # Detach rather than block the caller (a Stop click, or closeEvent); the worker
                # only lists directories, so leaving it to finish costs nothing.
                logger.warning("gui_watch_stop_timed_out")
                thread.finished.connect(thread.deleteLater)
            self._watch_thread = None
        self._watch_worker = None

    # --- providers and diagnostics ---------------------------------------------------------

    def _refresh_models(self) -> None:
        self._count("refresh_models")
        backend = get_backend(self._provider_name())
        base_url = self._url.text().strip() or backend.default_base_url
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            models = backend.list_models(base_url, backend.resolve_api_key(self._api_key_value()))
        except ProviderError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, _("Could not list models"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        current = self._model.currentText()
        self._model.clear()
        self._model.addItems(rank_vision_models(models))
        self._model.setCurrentText(current)
        self._status.setText(
            ngettext(
                "Found {n} model on {provider}.",
                "Found {n} models on {provider}.",
                len(models),
            ).format(n=len(models), provider=PROVIDER_LABELS.get(self._provider_name(), "")),
        )

    def _test_connection(self) -> None:
        self._count("test_connection")
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
        box = QMessageBox(self)
        box.setWindowTitle(_("Connection check"))
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(check_results_html(results))
        all_ok = all(r.ok for r in results)
        box.setIcon(QMessageBox.Icon.Information if all_ok else QMessageBox.Icon.Warning)
        box.exec()

    def _open_logs(self) -> None:
        """Reveal the log folder in the OS file browser so the user can read the run logs."""
        self._count("open_logs")
        LOG_FOLDER.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_FOLDER)))

    # --- helpers ---------------------------------------------------------------------------

    def _provider_name(self) -> ProviderName:
        """Return the selected provider; the combo carries the internal name as item data."""
        return cast("ProviderName", self._provider.currentData())

    def _api_key_value(self) -> str | None:
        """Return the typed API key, or None to fall back to the provider's env var/default."""
        return self._api_key.text().strip() or None

    def _connection(self) -> Connection:
        """Read the provider, model, server, and key off the header and the Connection dialog."""
        return Connection(
            provider=self._provider_name(),
            model=self._model.currentText().strip(),
            api_base_url=self._url.text().strip() or None,
            api_key=self._api_key_value(),
        )

    def _refresh_status_cell(self, item: PhotoItem) -> None:
        leaf = self._leaf_for(item.path)
        if leaf is not None:
            self._render_status_cells(leaf, item)
        # Keep the folder grid's badge overlay in step with the tree.
        grid_item = self._grid_items.get(str(item.path))
        if grid_item is not None:
            self._update_grid_item(item, grid_item)

    def _render_status_cells(self, leaf: QTreeWidgetItem, item: PhotoItem) -> None:
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

    def _resort(self) -> None:
        """Re-apply the active sort so changed statuses settle when sorting by the Status column."""
        header = self._tree.header()
        self._tree.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())

    def _leaf_for(self, path: Path) -> QTreeWidgetItem | None:
        """Return the tree row for the photo at *path*, or None when it is not listed."""
        return self._leaf_rows.get(str(path))

    def _selected_count(self) -> int:
        """How many photos are currently checked (used in deselect feedback)."""
        return sum(1 for item in self._items.values() if item.selected)

    def _update_status(self) -> None:
        if self._items:
            summary = status_summary(self._items.values())
            if self._search.text().strip():
                # _leaf_rows holds exactly the photos the last rebuild drew.
                shown = search_summary(len(self._leaf_rows), len(self._items))
                summary = f"{shown} · {summary}"
            self._status.setText(summary)
        # Retry only makes sense when something actually failed (and no run is in flight).
        if self._thread is None and self._save_thread is None:
            self._retry_button.setEnabled(self._has_failures())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override.
        """
        Confirm discarding unsaved proposals, then stop any in-flight generation or save.

        Asking the workers to stop first means closing mid-run only waits for the photo currently in
        flight, not the whole batch. Waiting on the threads keeps a running QThread from being
        destroyed under it.

        The QCloseEvent annotation is why QtGui imports that class at runtime rather than under
        TYPE_CHECKING. PySide6 creates its wrapper types on first use, and nothing else in this
        window touches QCloseEvent, so the first close used to build it (and QEvent, its base) from
        inside this very callback. Shiboken's introduceWrapperType does not check whether that
        creation succeeded, so a failure there segfaults the process rather than raising: quitting
        an untouched window sometimes died before this method's body ran. Importing the class up
        front builds the type during a plain import instead, well away from event delivery. The
        drag-and-drop handlers below take the same precaution.
        """
        unsaved = sum(
            1 for item in self._items.values() if item.has_proposal and item.status != SAVED
        )
        if unsaved:
            reply = QMessageBox.question(
                self,
                _("Unsaved changes"),
                _("Some photos have generated proposals that have not been saved. Close anyway?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._closing = True
        if self._worker is not None:
            self._worker.stop()
        if self._save_worker is not None:
            self._save_worker.stop()
        if self._watch_worker is not None:
            self._watch_worker.stop()
        self._stop_thumbs(blocking=True)
        self._stop_scan()
        self._teardown_thread()
        self._teardown_save_thread()
        self._teardown_harmonize_thread()
        self._teardown_watch_thread()
        self._builder_dialog.shutdown()
        self._report_close(unsaved=unsaved)
        super().closeEvent(event)

    def maybe_show_telemetry_notice(self) -> None:
        """
        Show the one-time telemetry disclosure on the first run telemetry is active.

        Offers to turn telemetry off right here; otherwise it stays on and can be toggled later from
        the Settings menu. Called from :func:`launch` after the window is shown, not from
        ``__init__``, so the headless test suite (which builds the window directly) never blocks on
        a modal dialog.
        """
        if not self._telemetry_consent().send:
            return
        if telemetry.first_run_notice(interface="gui") is None:  # already shown on an earlier run
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(_("Anonymous usage telemetry"))
        box.setText(_("Photo Tagger sends anonymous usage stats to guide development."))
        box.setInformativeText(
            _(
                "It sends the settings and features you use, how runs go, your system, and crash "
                "reports. It never sends your photos, file names, or anything you write.\n\n"
                "You can turn this off now, or anytime from Settings > Send Anonymous Telemetry.",
            ),
        )
        keep = box.addButton(_("Keep Enabled"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(_("Turn It Off"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is not keep:
            # Unchecking fires _on_telemetry_toggled, which persists the choice.
            self._telemetry_action.setChecked(False)

    # --- telemetry -----------------------------------------------------------------------------

    def _count(self, feature: str) -> None:
        """Count one use of a feature, for the session beacon (names from GUI_FEATURES)."""
        self._usage.count(feature, time.monotonic())

    def _touch(self, *_args: object) -> None:
        """Count input as time in use; connected to signals, whose arguments do not matter."""
        self._usage.clock.touch(time.monotonic())

    def _watch_activity(self) -> None:
        """
        Count keyboard and mouse use of the window as time in use.

        A filter on the few widgets a review happens in, plus focus changes anywhere, rather than
        one on the whole application: that would run Python for every event Qt delivers.
        """
        self._input_watcher = InputWatcher(self._touch, self)
        for widget in (
            self._tree,
            self._grid,
            self._title,
            self._description,
            self._keywords,
            self._hint,
        ):
            widget.installEventFilter(self._input_watcher)
            if hasattr(widget, "viewport"):
                widget.viewport().installEventFilter(self._input_watcher)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._touch)

    def _send(self, event: telemetry.Event, *, block: bool = False) -> object:
        """Send one beacon under this window's toggle; returns the sending thread, if any."""
        return telemetry.emit(
            event,
            enabled=self._telemetry_enabled,
            opted_in=self._telemetry_opted_in,
            block=block,
        )

    def _telemetry_settings(self) -> telemetry.Settings:
        """Snapshot the window's settings as the bounded values telemetry reports."""
        backend = get_backend(self._provider_name())
        typed = self._url.text().strip()
        watching = self._watch_settings or WatchSettings()
        vocabulary = self._keyword_rules_dialog.vocabulary
        return window_settings(
            self._current_config_values(),
            WindowFacts(
                url=typed or backend.default_base_url,
                custom_url=bool(typed),
                api_key=bool(backend.resolve_api_key(self._api_key_value())),
                cache=self._cache_action.isChecked(),
                jpeg_dimensions=self._model_image_size,
                vocabulary_terms=len(vocabulary.terms) if vocabulary else 0,
                copyright=self._copyright_dialog.fields,
                watch_interval=watching.interval,
                watch_settle=watching.settle,
                sidecar_overrides=len(self._sidecar_overrides),
            ),
        )

    def _run_context(self) -> RunContext:
        """
        Describe what a run starts with: model, languages, settings, and how they were set.

        Telemetry must never stop the window, so a snapshot that raises leaves only the bare facts.
        """
        try:
            return self._describe_run()
        except Exception as exc:  # noqa: BLE001 - any snapshot bug must not stop the window.
            logger.warning("telemetry_snapshot_failed", error=repr(exc))
            return RunContext(
                provider="",
                model="",
                output_language="",
                ui_language=i18n.current_language(),
                settings=telemetry.Settings(),
                setup=telemetry.Setup(),
            )

    def _describe_run(self) -> RunContext:
        """Build the run context from the window's current settings."""
        settings = self._telemetry_settings()
        keys, unknown = config_key_names(self._raw_config, known_config_keys())
        return RunContext(
            provider=self._provider_name(),
            model=self._model.currentText().strip(),
            output_language=self._output_language,
            ui_language=i18n.current_language(),
            settings=settings,
            setup=telemetry.Setup(
                config_file=config_file_kind(),
                config_keys=keys,
                unknown_config_keys=unknown,
                env_vars=env_var_names(),
                changed=changed_settings(settings, GUI_BASELINE),
            ),
        )

    def _display_facts(self) -> telemetry.Display:
        """Describe the toolkit and the screen the window runs on."""
        screen = QApplication.primaryScreen()
        size = screen.size() if screen is not None else None
        scheme = QApplication.styleHints().colorScheme()
        if self.isFullScreen():
            window = "full-screen"
        elif self.isMaximized():
            window = "maximized"
        else:
            window = f"{self.width()}x{self.height()}"
        return telemetry.Display(
            qt_version=qVersion(),
            pyside_version=pyside_version,
            screen=f"{size.width()}x{size.height()}@{screen.devicePixelRatio():g}" if size else "",
            screens=len(QApplication.screens()),
            color_scheme=_COLOR_SCHEMES.get(scheme, ""),
            window=window,
        )

    def _start_run(
        self,
        job: str,
        trigger: str,
        items: list[PhotoItem],
        *,
        use_cache: bool = True,
    ) -> None:
        """Open the run the window reports when it ends."""
        self._finish_run()
        self._run_keys = [str(item.path) for item in items]
        generating = job == JOB_GENERATE
        self._usage.start_run(
            RunTally(
                job=job,
                trigger=trigger,
                started=time.monotonic(),
                batch_size=len(items),
                context=self._run_context(),
                use_cache=use_cache,
                autosave=generating and self._wants_chained_save(),
                hinted=sum(1 for item in items if item.hint) if generating else 0,
                regenerated_flagged=(
                    sum(1 for item in items if item.foreign_terms) if generating else 0
                ),
            ),
        )

    def _finish_run(self, outcome: str | None = None) -> object:
        """Close the run in flight, if any, and send its beacon; returns the sending thread."""
        run = self._usage.run
        if run is None:
            return None
        now = time.monotonic()
        batch = [item for key in self._run_keys if (item := self._items.get(key)) is not None]
        info = self._usage.finish_run(
            RunEnd(
                now=now,
                outcome=outcome,
                attended=self._usage.clock.engaged_between(run.started, now),
                succeeded=batch_outcome(batch)[0] if run.job == JOB_GENERATE else run.saved,
                failed=batch_outcome(batch)[1] if run.job == JOB_GENERATE else 0,
                file_types=telemetry.file_types_summary(Path(key) for key in self._run_keys),
                flagged=sum(1 for item in batch if item.foreign_terms),
            ),
        )
        self._run_keys = []
        return self._send(info) if info is not None else None

    def report_launch(self, startup_seconds: float) -> None:
        """Send the launch beacon: how long the window took to open, and what it opened with."""
        context = self._run_context()
        self._send(
            telemetry.LaunchInfo(
                startup_seconds=startup_seconds,
                settings=context.settings,
                setup=context.setup,
                display=self._display_facts(),
            ),
        )

    def _report_close(self, *, unsaved: int) -> None:
        """
        Send the run the window closed on, if any, then the session, and wait for both.

        The process exits right after closeEvent, so a beacon still on its daemon thread would be
        lost; flush waits for both within one budget, instead of one after the other.
        """
        run = self._finish_run(telemetry.OUTCOME_CANCELLED)
        session = self._send(
            self._usage.session_info(
                time.monotonic(),
                context=self._run_context(),
                unsaved=unsaved,
                closed_with_unsaved=bool(unsaved),
                view_mode=VIEW_LIST if self._flat_list() else VIEW_TREE,
                display=self._display_facts(),
            ),
        )
        telemetry.flush([run, session])


# How the OS color scheme reads in the launch and session beacons.
_COLOR_SCHEMES = {Qt.ColorScheme.Dark: "dark", Qt.ColorScheme.Light: "light"}


def _make_placeholder() -> QPixmap:
    """Build a neutral grey tile shown in the grid until a thumbnail loads."""
    pixmap = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
    pixmap.fill(QColor(50, 50, 56))
    return pixmap


# Thumbnail badge rendering: fill color and glyph per badge name. The lifecycle badge (first
# three) draws top-right; the informational ones stack top-left.
_BADGE_STYLE = {
    BADGE_FAILED: ("#f85149", "✗"),  # red cross
    BADGE_SAVED: ("#3fb950", "✓"),  # green check
    BADGE_UNSAVED: ("#6366f1", "•"),  # indigo dot: generated, not saved yet
    BADGE_METADATA: ("#8a8a8a", "M"),  # file already carries metadata
    BADGE_COPYRIGHT: ("#8957e5", "C"),  # purple: the file carries a copyright notice
    BADGE_SIDECAR: ("#0e7490", "S"),  # an XMP sidecar exists
    BADGE_MIXED_SCRIPT: ("#d29922", "!"),  # amber: the text mixes writing systems
}
_BADGE_TEXT = {
    BADGE_FAILED: gettext_noop("generation failed"),
    BADGE_SAVED: gettext_noop("saved"),
    BADGE_UNSAVED: gettext_noop("generated, not saved yet"),
    BADGE_METADATA: gettext_noop("already has metadata"),
    BADGE_COPYRIGHT: gettext_noop("has a copyright notice"),
    BADGE_SIDECAR: gettext_noop("has an XMP sidecar"),
    BADGE_MIXED_SCRIPT: gettext_noop("mixes writing systems"),
}
_LIFECYCLE_BADGES = frozenset({BADGE_FAILED, BADGE_SAVED, BADGE_UNSAVED})


def _badged_pixmap(base: QPixmap, badges: list[str]) -> QPixmap:
    """Overlay the badge dots for *badges* onto a copy of *base*."""
    if not badges or base.isNull():
        return base
    pixmap = QPixmap(base)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    radius = max(12, pixmap.width() // 8)
    margin = max(3, radius // 4)
    left = margin
    font = painter.font()
    font.setPixelSize(int(radius * 0.7))
    font.setBold(True)
    painter.setFont(font)
    for name in badges:
        color, glyph = _BADGE_STYLE[name]
        x = pixmap.width() - radius - margin if name in _LIFECYCLE_BADGES else left
        if name not in _LIFECYCLE_BADGES:
            left += radius + margin
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(x, margin, radius, radius)
        painter.setPen(QColor("white"))
        painter.drawText(QRect(x, margin, radius, radius), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return pixmap
