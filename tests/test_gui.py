# mypy: ignore-errors
"""
Headless tests for the PySide6 desktop GUI.

Skipped entirely when PySide6 is absent (CLI-only setups and CI, which do not install the optional
``[gui]`` extra). When present, Qt's ``offscreen`` platform builds real widgets without a display
server, so the tree, the detail pane, and the generation worker can be exercised for real.

The ``# mypy: ignore-errors`` header opts this file out of the zuban type check, for the same reason
the gui package carries it: PySide6 ships no stubs and is not installed in the lint job, so a strict
run would only see an unresolved-import error here.
"""

import importlib
import json
import os
import pkgutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest


pytest.importorskip("PySide6")
# Must be set before the first QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidgetItem,
)

from photo_tagger import gui, i18n, image_io, metadata, telemetry
from photo_tagger.errors import ProviderError
from photo_tagger.gui import detail_pane, main_window, select_menu, viewer, widgets, workers
from photo_tagger.gui.common import COL_STATUS, COL_TAGGED, IS_DIR_ROLE, PATH_ROLE
from photo_tagger.gui_state import (
    DEFAULT_GUI_EXTENSIONS,
    EXTRA_PASSES,
    FAILED,
    FILTER_FAILED,
    FILTER_MIXED_SCRIPT,
    FILTER_NO_COPYRIGHT,
    FILTER_SAVED,
    FILTER_UNTAGGED,
    MAX_ZOOM,
    MIXED_SCRIPT_MARK,
    PENDING,
    READY,
    RERUN_FAILED,
    RERUN_MIXED_SCRIPT,
    SAVED,
    SELECT_CHECK,
    SELECT_UNCHECK,
    SORT_STATUS,
    TOOLTIP_WIDTH,
    WORKING,
    GenerateOptions,
    HarmonizeResult,
    Proposal,
    SaveJob,
    SaveOptions,
    TaggedCriterion,
    WatchSettings,
    build_copyright_fields,
    gui_cache_namespace,
    photo_matches_filter,
)
from photo_tagger.metadata import (
    FIELD_COPYRIGHT,
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    CaptionValue,
    FieldSources,
    ImageContext,
)
from photo_tagger.models import InferenceResult, KeywordSet
from photo_tagger.pacing import MAX_BATTERY_FLOOR, PacingPlan
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES
from photo_tagger.undo import list_journals, open_journal, read_journal
from photo_tagger.vocabulary import Vocabulary
from photo_tagger.vocabulary_build import KeywordCensus, TrimRules
from photo_tagger.vocabulary_organize import OrganizeStats


def _gui_modules() -> list[ModuleType]:
    """Return every module of the gui package, so a check or a patch can cover them all."""
    return [
        importlib.import_module(f"{gui.__name__}.{info.name}")
        for info in pkgutil.iter_modules(gui.__path__)
    ]


def _patch_gui(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    """
    Replace *name* in every GUI module that imports it.

    A module binds what it imports under its own name, so a patch only takes where it lands in the
    module that makes the call. Patching every one keeps a test from depending on which module that
    is, and failing when none has the name catches a patch that would otherwise quietly do nothing.
    """
    patched = [module for module in _gui_modules() if hasattr(module, name)]
    assert patched, f"no GUI module uses {name}"
    for module in patched:
        monkeypatch.setattr(module, name, value)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """One QApplication for the module (Qt allows only a single instance)."""
    return cast("QApplication", QApplication.instance() or QApplication([]))


@pytest.fixture(autouse=True)
def _cache_off_the_users_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep every test's cache writes inside tmp_path.

    The window falls back to ``~/.photo-tagger/cache.sqlite`` when the config names no cache file,
    so a test that starts a real run (``_run_generation`` spawns a worker that opens the cache)
    wrote its stub answer into whatever the person running the suite had tagged. Six such rows had
    accumulated in one real cache since July, one per test session.
    """
    _patch_gui(monkeypatch, "DEFAULT_CACHE_FILE", tmp_path / "test-cache.sqlite")


@pytest.fixture
def window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[gui.MainWindow]:
    """Yield a fresh main window, closing it (and its thread) after each test."""
    win = gui.MainWindow()
    # Adding photos kicks off the background exiftool scan for the Tagged column; keep tests
    # deterministic (and exiftool-free) by driving _on_scan_done directly where needed.
    monkeypatch.setattr(win, "_start_metadata_scan", lambda: None)
    # The teardown close() below must never block on a real modal: most tests leave an unsaved
    # proposal behind on purpose and are not testing the close-confirmation dialog itself. Default
    # to "close anyway"; a test that specifically exercises the prompt overrides this again.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Yes,
    )
    yield win
    win.close()


def _jpeg(path: Path) -> Path:
    """Write a tiny real JPEG so previews and exiftool have something to act on."""
    # Imported lazily so this module only needs Pillow when the GUI tests actually run.
    from PIL import Image  # noqa: PLC0415

    buf = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buf, format="JPEG")
    path.write_bytes(buf.getvalue())
    return path


def _stub_reads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    keywords: list[str],
    captions: dict[str, list[CaptionValue]] | None = None,
) -> None:
    """
    Patch the exiftool-backed reads so tests need no exiftool binary.

    *captions* overrides the per-field breakdown the detail pane reads, for a test that cares which
    file a value came from or what it shadows; by default both fields come from the sidecar alone.
    """
    _patch_gui(monkeypatch, "read_caption", lambda _p: ("Old Title", "Old caption."))
    found = captions or {
        FIELD_TITLE: [CaptionValue("Old Title", SOURCE_SIDECAR)],
        FIELD_DESCRIPTION: [CaptionValue("Old caption.", SOURCE_SIDECAR)],
    }
    _patch_gui(monkeypatch, "read_caption_values", lambda _p: found)
    _patch_gui(
        monkeypatch,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(existing_keywords=KeywordSet(subject=keywords)),
    )


def _stub_save_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the batch save's shared ExifTool out of the tests; the writes themselves are stubbed."""
    _patch_gui(monkeypatch, "managed_helper", lambda _et: nullcontext(None))


def _drain_save(window: gui.MainWindow, timeout: float = 10.0) -> None:
    """Pump the event loop until the background save finishes (or fail on the deadline)."""
    deadline = time.monotonic() + timeout
    while window._save_thread is not None:  # noqa: SLF001
        assert time.monotonic() < deadline, "the background save never finished"
        QApplication.processEvents()


def _add_dir(window: gui.MainWindow, files: dict[str, Path]) -> None:
    """Set jpg extensions and add the directory the files live in."""
    folder = next(iter(files.values())).parent
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([folder])  # noqa: SLF001


def _select(window: gui.MainWindow, item: QTreeWidgetItem | None) -> None:
    """Select a tree item, asserting it was actually found first."""
    assert item is not None
    window._tree.setCurrentItem(item)  # noqa: SLF001


def _check_state(window: gui.MainWindow, path: Path) -> Qt.CheckState:
    """Return the rendered checkbox state of *path*'s tree leaf (asserting it exists)."""
    leaf = window._tree.leaf_for(path)  # noqa: SLF001
    assert leaf is not None
    return leaf.checkState(0)


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------


def test_window_builds_with_capitalized_providers(window: gui.MainWindow) -> None:
    """The combo shows capitalized labels but carries the internal names as data."""
    assert "Photo Tagger" in window.windowTitle()
    combo = window._connection_bar._provider  # noqa: SLF001
    names = [combo.itemData(i) for i in range(combo.count())]
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert names == list(PROVIDER_NAMES)
    assert labels == [PROVIDER_LABELS[n] for n in PROVIDER_NAMES]
    assert "LM Studio" in labels


def test_provider_name_returns_internal_value(window: gui.MainWindow) -> None:
    """Selecting a label resolves back to the internal provider name."""
    window._connection_bar._provider.setCurrentIndex(list(PROVIDER_NAMES).index("openai"))  # noqa: SLF001
    assert window._connection_bar.connection().provider == "openai"  # noqa: SLF001


def test_refresh_models_populates_combo_vision_first(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh queries the backend and lists models, vision-likely ones first."""
    fake = SimpleNamespace(
        default_base_url="http://localhost/v1",
        resolve_api_key=lambda _k: None,
        list_models=lambda _base, _key: ["llama3", "qwen3-vl-30b"],
    )
    _patch_gui(monkeypatch, "get_backend", lambda _name: fake)
    window._connection_bar.refresh_models()  # noqa: SLF001
    combo = window._connection_bar._model  # noqa: SLF001
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items[0] == "qwen3-vl-30b"
    assert "llama3" in items


# ---------------------------------------------------------------------------
# API key field
# ---------------------------------------------------------------------------


def test_api_key_value_strips_and_blank_is_none(window: gui.MainWindow) -> None:
    """A typed key is trimmed; a blank field means "use the env var" (None)."""
    window._connection_bar._api_key.setText("  sk-typed  ")  # noqa: SLF001
    assert window._connection_bar.connection().api_key == "sk-typed"  # noqa: SLF001
    window._connection_bar._api_key.setText("   ")  # noqa: SLF001
    assert window._connection_bar.connection().api_key is None  # noqa: SLF001


def test_api_key_field_is_masked(window: gui.MainWindow) -> None:
    """The key field hides its contents so a shoulder-surfer cannot read it."""
    from PySide6.QtWidgets import QLineEdit  # noqa: PLC0415

    assert window._connection_bar._api_key.echoMode() == QLineEdit.EchoMode.Password  # noqa: SLF001


def test_refresh_models_passes_typed_api_key(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key typed into the field is handed to the backend's key resolution."""
    window._connection_bar._api_key.setText("sk-refresh")  # noqa: SLF001
    seen: dict[str, str | None] = {}
    fake = SimpleNamespace(
        default_base_url="http://localhost/v1",
        resolve_api_key=lambda key: seen.setdefault("key", key),
        list_models=lambda _base, _key: ["m"],
    )
    _patch_gui(monkeypatch, "get_backend", lambda _name: fake)
    window._connection_bar.refresh_models()  # noqa: SLF001
    assert seen["key"] == "sk-refresh"


def test_worker_passes_typed_api_key_to_create_agent(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate worker forwards its api_key to create_agent."""
    captured: dict[str, object] = {}

    def fake_create_agent(*_a: object, **kwargs: object) -> object:
        captured.update(kwargs)
        msg = "stop before per-photo work"
        raise ProviderError(msg)

    _patch_gui(monkeypatch, "create_agent", fake_create_agent)
    worker = workers.GenerateWorker("openai", "m", None, [Path("/a.jpg")], api_key="sk-worker")
    worker.file_failed.connect(lambda *_a: None)
    worker.run()
    assert captured["api_key"] == "sk-worker"


def test_worker_passes_output_language_to_create_agent(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate worker forwards its metadata language to create_agent."""
    captured: dict[str, object] = {}

    def fake_create_agent(*_a: object, **kwargs: object) -> object:
        captured.update(kwargs)
        msg = "stop before per-photo work"
        raise ProviderError(msg)

    _patch_gui(monkeypatch, "create_agent", fake_create_agent)
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(output_language="German"),
    )
    worker.file_failed.connect(lambda *_a: None)
    worker.run()
    assert captured["output_language"] == "German"


def test_run_generation_builds_worker_with_typed_api_key(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting a run hands the typed key to the worker it spawns."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)  # so the background worker does no real I/O
    _add_dir(window, {"a": img})
    window._connection_bar._api_key.setText("sk-run")  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._api_key == "sk-run"  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_run_generation_hands_the_metadata_language_to_the_worker(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting a run passes the window's metadata language to the worker it spawns."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)  # so the background worker does no real I/O
    _add_dir(window, {"a": img})
    window._settings_menu._output_language = "German"  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._output_language == "German"  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_run_generation_commits_and_passes_the_open_photos_hint(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hint still sitting in the field is committed and handed to the worker."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001 - open the photo so the field is live
    window._detail._hint.setText("The animal is a deer")  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    assert window._items[str(img)].hint == "The animal is a deer"  # noqa: SLF001
    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._hints == {str(img): "The animal is a deer"}  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


# ---------------------------------------------------------------------------
# Tree: nesting, selection, removal
# ---------------------------------------------------------------------------


def test_add_inputs_builds_a_nested_tree(window: gui.MainWindow, tmp_path: Path) -> None:
    """A folder with a subfolder nests under one top node."""
    (tmp_path / "sub").mkdir()
    files = {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "sub" / "b.jpg")}
    _add_dir(window, files)

    assert len(window._items) == 2  # noqa: SLF001, PLR2004 - two photos
    assert window._tree.topLevelItemCount() == 1  # noqa: SLF001
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    assert bool(top.data(0, IS_DIR_ROLE))
    # One file leaf and one "sub" folder under the top.
    kinds = {bool(top.child(i).data(0, IS_DIR_ROLE)) for i in range(top.childCount())}
    assert kinds == {True, False}


def test_unchecking_folder_deselects_descendants(window: gui.MainWindow, tmp_path: Path) -> None:
    """Unchecking the top folder deselects every photo beneath it."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "b.jpg")})
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    top.setCheckState(0, Qt.CheckState.Unchecked)
    assert all(not item.selected for item in window._items.values())  # noqa: SLF001


def test_unchecking_a_subfolder_repaints_its_ancestors(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A folder toggle has to walk up too: its parents are no longer fully checked."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(sub / "b.jpg")})
    top_row = window._tree._folder_rows[str(tmp_path)]  # noqa: SLF001
    assert top_row.checkState(0) == Qt.CheckState.Checked

    window._tree._folder_rows[str(sub)].setCheckState(0, Qt.CheckState.Unchecked)  # noqa: SLF001

    # "a" is still checked and "b" is not, so the top folder is neither on nor off.
    assert top_row.checkState(0) == Qt.CheckState.PartiallyChecked


def test_remove_selected_folder_drops_its_files(window: gui.MainWindow, tmp_path: Path) -> None:
    """Removing a folder node removes all photos under it (not just deselects)."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "b.jpg")})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001
    assert window._items == {}  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001


def test_remove_selected_file_drops_one(window: gui.MainWindow, tmp_path: Path) -> None:
    """Removing a single file leaf leaves the rest in place."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a, "b": _jpeg(tmp_path / "b.jpg")})
    leaf = window._tree.leaf_for(a)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001
    assert str(a) not in window._items  # noqa: SLF001
    assert len(window._items) == 1  # noqa: SLF001


def test_remove_menu_drops_checked_or_unchecked_photos(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """The Remove arrow drops photos by checkbox, whatever the tree has highlighted."""
    a, b, c = (_jpeg(tmp_path / f"{name}.jpg") for name in "abc")
    _add_dir(window, {"a": a, "b": b, "c": c})
    window._items[str(b)].selected = False  # noqa: SLF001
    menu = window._build_remove_menu()  # noqa: SLF001
    actions = {action.text(): action for action in menu.actions()}

    actions["Remove Unchecked"].trigger()
    assert set(window._items) == {str(a), str(c)}  # noqa: SLF001

    actions["Remove Unchecked"].trigger()
    assert window._status.text() == "No unchecked photos to remove."  # noqa: SLF001

    actions["Remove Checked"].trigger()
    assert window._items == {}  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001


def test_clear_empties_everything(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clear drops all items and the tree."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._clear()  # noqa: SLF001
    assert window._items == {}  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001


@pytest.mark.parametrize("attribute", ["_thread", "_save_thread"])
def test_clear_is_refused_while_a_run_is_in_flight(
    window: gui.MainWindow,
    tmp_path: Path,
    attribute: str,
) -> None:
    """
    Neither run may have the list pulled out from under it.

    A save keeps writing from its own job list, so clearing mid-save wrote photos the window no
    longer knew about and then reported "Saved 0" for every one of them.
    """
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    setattr(window, attribute, SimpleNamespace())  # a run is in flight
    try:
        window._clear()  # noqa: SLF001

        assert window._items != {}  # noqa: SLF001
        assert window._tree.topLevelItemCount() == 1  # noqa: SLF001
    finally:
        # Put it back before the fixture's close(), which would call quit() on the stand-in.
        setattr(window, attribute, None)


def test_row_index_covers_every_folder_and_leaf(window: gui.MainWindow, tmp_path: Path) -> None:
    """Both lookups read an index built during the rebuild, nested folders included."""
    nested = tmp_path / "sub"
    nested.mkdir()
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(nested / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    assert set(window._tree._leaf_rows) == {str(a), str(b)}  # noqa: SLF001
    assert set(window._tree._folder_rows) == {str(tmp_path), str(nested)}  # noqa: SLF001
    window._tree.select_row(str(nested), is_dir=True)  # noqa: SLF001
    assert window._tree.currentItem() is window._tree._folder_rows[str(nested)]  # noqa: SLF001


def test_a_rebuild_replaces_every_indexed_row(window: gui.MainWindow, tmp_path: Path) -> None:
    """
    A rebuild swaps in fresh rows and detaches the old ones cleanly.

    Anything still holding a row from the previous tree (a context menu, a caller's local) keeps a
    live, detached object. Under QTreeWidget.clear() that row would instead be freed in C++ while
    Python still pointed at it.
    """
    from shiboken6 import isValid  # noqa: PLC0415 - only importable with the [gui] extra

    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    stale_folder = window._tree._folder_rows[str(tmp_path)]  # noqa: SLF001
    stale_leaf = window._tree._leaf_rows[str(b)]  # noqa: SLF001

    window._remove_items([str(a)])  # noqa: SLF001

    assert set(window._tree._leaf_rows) == {str(b)}  # noqa: SLF001
    assert window._tree._leaf_rows[str(b)] is not stale_leaf  # noqa: SLF001
    assert window._tree._folder_rows[str(tmp_path)] is not stale_folder  # noqa: SLF001
    assert isValid(stale_folder), "the old row was freed while Python still held it"
    assert stale_folder.treeWidget() is None
    assert stale_leaf.treeWidget() is None


def test_the_tree_is_emptied_row_by_row_never_with_clear(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A rebuild takes the rows out; QTreeWidget.clear() is off limits.

    clear() frees rows that Python still references without emitting the removal, which leaves any
    QTreeWidgetItemIterator registered with QTreeModel pointing at freed memory.
    """
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    forbidden: list[int] = []
    monkeypatch.setattr(window._tree, "clear", lambda: forbidden.append(1))  # noqa: SLF001

    window._rebuild_tree()  # noqa: SLF001

    assert not forbidden, "_rebuild_tree fell back to QTreeWidget.clear()"
    assert window._tree.topLevelItemCount() == 1  # noqa: SLF001
    assert set(window._tree._leaf_rows) == {str(a), str(b)}  # noqa: SLF001


def test_emptying_the_tree_drops_the_rows_and_the_index(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Emptying the tree leaves nothing behind in the widget or the lookup."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._tree.empty()  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001
    assert not window._tree._folder_rows  # noqa: SLF001
    assert not window._tree._leaf_rows  # noqa: SLF001


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (main_window, "QCloseEvent"),
        (main_window, "QDragEnterEvent"),
        (main_window, "QDropEvent"),
        (viewer, "QKeyEvent"),
        (widgets, "QMouseEvent"),
    ],
)
def test_event_types_are_built_at_import_not_during_delivery(module: ModuleType, name: str) -> None:
    """
    The event classes the window annotates must exist before Qt delivers one.

    PySide6 builds its wrapper types on first use. Left to TYPE_CHECKING these were built from
    inside the C++ callback that delivers the event, and shiboken's introduceWrapperType does not
    check whether the creation succeeded, so quitting an untouched window could crash there.
    """
    import PySide6.QtGui  # noqa: PLC0415 - only importable with the [gui] extra

    assert getattr(module, name, None) is not None, (
        f"{module.__name__} must import {name} at runtime"
    )
    assert name in vars(PySide6.QtGui), f"{name} is still built lazily"


def test_gui_does_not_use_the_tree_item_iterator() -> None:
    """
    QTreeWidgetItemIterator is banned here: PySide never destroys one.

    Python does not own the C++ iterator, so it stays registered with QTreeModel forever with a
    pointer to whatever row it stopped on. The next teardown frees that row, and the removal after
    it dereferences the dangling pointer inside QTreeModel::beginRemoveItems.
    """
    assert not [m.__name__ for m in _gui_modules() if hasattr(m, "QTreeWidgetItemIterator")]


# ---------------------------------------------------------------------------
# Tree sorting (click a header to sort by name or status)
# ---------------------------------------------------------------------------


def _leaf_order(window: gui.MainWindow) -> list[str]:
    """Return the file-leaf names under the single top folder, in display order."""
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    return [top.child(i).text(0) for i in range(top.childCount())]


def test_tree_sorts_by_name(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clicking the Photos header orders leaves by name, not creation order."""
    for name in ("c.jpg", "a.jpg", "b.jpg"):  # created out of order
        _jpeg(tmp_path / name)
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001

    window._tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # noqa: SLF001

    assert _leaf_order(window) == ["a.jpg", "b.jpg", "c.jpg"]


def test_tree_sorts_by_status_using_lifecycle_rank(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Sorting the Status column uses the lifecycle rank, not the alphabetical label."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001
    window._items[str(a)].status = FAILED  # noqa: SLF001
    window._items[str(b)].status = PENDING  # noqa: SLF001
    window._items[str(c)].status = READY  # noqa: SLF001
    for path in (a, b, c):
        window._refresh_status_cell(window._items[str(path)])  # noqa: SLF001

    window._tree.sortByColumn(2, Qt.SortOrder.AscendingOrder)  # noqa: SLF001 - Status column

    # Ascending lifecycle: pending(b) < ready(c) < failed(a). Alphabetical-by-label would differ.
    assert _leaf_order(window) == ["b.jpg", "c.jpg", "a.jpg"]


def test_tree_keeps_folders_above_files(window: gui.MainWindow, tmp_path: Path) -> None:
    """A subfolder stays grouped above sibling files even when its name sorts after them."""
    (tmp_path / "zzz").mkdir()
    _jpeg(tmp_path / "a.jpg")  # top-level file, name sorts before "zzz"
    _jpeg(tmp_path / "zzz" / "deep.jpg")
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001

    window._tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # noqa: SLF001

    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    first = top.child(0)
    assert bool(first.data(0, IS_DIR_ROLE))
    assert first.text(0) == "zzz"


# ---------------------------------------------------------------------------
# Deselect: skip already-tagged, skip from a list (CLI --skip-tagged/--skip-from)
# ---------------------------------------------------------------------------


def _scan_reports(window: gui.MainWindow, by_name: dict[str, set[str]]) -> None:
    """Report a finished metadata scan over the whole list, fields keyed by filename."""
    window._on_scan_done(  # noqa: SLF001
        {key: set(by_name.get(Path(key).name, set())) for key in window._items},  # noqa: SLF001
    )


def _criterion(required: set[str], *, match_all: bool, phrase: str) -> TaggedCriterion:
    """Build one "already tagged" rule, the way a Select-menu preset holds it."""
    return TaggedCriterion(frozenset(required), match_all=match_all, phrase=phrase)


def test_deselect_tagged_field_aware_keeps_keyword_only_photos(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """'Title and description' skips photos that have both, keeping keyword-only ones selected."""
    a = _jpeg(tmp_path / "a.jpg")  # already has a title and a description
    b = _jpeg(tmp_path / "b.jpg")  # only keywords -> must stay selected for description generation
    _add_dir(window, {"a": a, "b": b})
    _scan_reports(
        window,
        {"a.jpg": {FIELD_TITLE, FIELD_DESCRIPTION}, "b.jpg": {FIELD_KEYWORDS}},
    )

    window._deselect_tagged(  # noqa: SLF001
        _criterion(
            {FIELD_TITLE, FIELD_DESCRIPTION},
            match_all=True,
            phrase="a title and a description",
        ),
    )

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    # The rendered tree checkbox, not just the model flag, must reflect the deselection.
    assert _check_state(window, a) == Qt.CheckState.Unchecked
    assert _check_state(window, b) == Qt.CheckState.Checked
    assert "Deselected 1" in window._status.text()  # noqa: SLF001
    assert "a title and a description" in window._status.text()  # noqa: SLF001


def test_deselect_tagged_any_metadata_uses_or_semantics(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """The 'any metadata' criterion deselects a photo that has even one indicator field."""
    a = _jpeg(tmp_path / "a.jpg")  # keywords only
    b = _jpeg(tmp_path / "b.jpg")  # nothing
    _add_dir(window, {"a": a, "b": b})
    _scan_reports(window, {"a.jpg": {FIELD_KEYWORDS}})

    window._deselect_tagged(  # noqa: SLF001
        _criterion(
            {FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS},
            match_all=False,
            phrase="any metadata",
        ),
    )

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001


def test_deselect_tagged_reports_when_none_match(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """When no checked photo has the field, the action says so instead of 'Deselected 0'."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    _scan_reports(window, {})  # scanned, and no photo has any field

    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_DESCRIPTION}, match_all=True, phrase="a description"),
    )

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "No checked photos have a description" in window._status.text()  # noqa: SLF001


def test_deselect_tagged_waits_for_the_metadata_scan(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A pick made before the Tagged column is known is parked until the scan reaches the end.

    Regression test: this used to run its own batched exiftool read straight from the click handler,
    which froze the window for minutes on a few thousand photos on a network share.
    """
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    started: list[int] = []
    monkeypatch.setattr(window, "_start_metadata_scan", lambda: started.append(1))

    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_TITLE}, match_all=True, phrase="a title"),
    )

    assert started  # it asked the scan to run instead of reading anything itself
    assert window._items[str(a)].selected is True  # noqa: SLF001 - nothing decided yet
    assert "Reading metadata" in window._status.text()  # noqa: SLF001

    _scan_reports(window, {"a.jpg": {FIELD_TITLE}})
    window._on_scan_finished()  # noqa: SLF001

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    assert "Deselected 1" in window._status.text()  # noqa: SLF001
    assert window._scan_pending is None  # noqa: SLF001


def test_a_second_pick_replaces_the_one_waiting_on_the_scan(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing your mind mid-scan applies the last pick, not both."""
    a = _jpeg(tmp_path / "a.jpg")  # keywords only
    _add_dir(window, {"a": a})
    monkeypatch.setattr(window, "_start_metadata_scan", lambda: None)

    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_TITLE}, match_all=True, phrase="a title"),
    )
    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_KEYWORDS}, match_all=True, phrase="keywords"),
    )
    _scan_reports(window, {"a.jpg": {FIELD_KEYWORDS}})
    window._on_scan_finished()  # noqa: SLF001

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert "keywords" in window._status.text()  # noqa: SLF001


def test_stopping_the_scan_abandons_the_waiting_uncheck(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop leaves every checkbox alone and says so, rather than acting on a half-read list."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    monkeypatch.setattr(window, "_start_metadata_scan", lambda: None)
    window._scan_worker = workers.MetadataScanWorker([])  # noqa: SLF001
    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_TITLE}, match_all=True, phrase="a title"),
    )

    window._cancel_scan()  # noqa: SLF001
    _scan_reports(window, {"a.jpg": {FIELD_TITLE}})  # the batch already in flight still lands
    window._on_scan_finished()  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "Stopped reading metadata" in window._status.text()  # noqa: SLF001
    assert window._scan_pending is None  # noqa: SLF001
    assert window._scan_row.isHidden()  # noqa: SLF001


def _stopping_scan(window: gui.MainWindow, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """
    Put *window* in a scan that Stop is winding down, and record the scans started after it.

    While a scan thread exists, the real _start_metadata_scan runs, which only notes the request.
    Once it is gone, a start is recorded and stands in for a new scan thread.
    """
    started: list[int] = []

    def start() -> None:
        if window._scan_thread is not None:  # noqa: SLF001
            gui.MainWindow._start_metadata_scan(window)  # noqa: SLF001
            return
        started.append(1)
        window._scan_thread = _FakeScanThread(finishes_in_time=True)  # noqa: SLF001

    monkeypatch.setattr(window, "_start_metadata_scan", start)
    window._scan_thread = _FakeScanThread(finishes_in_time=True)  # noqa: SLF001
    window._scan_worker = workers.MetadataScanWorker([])  # noqa: SLF001
    window._cancel_scan()  # noqa: SLF001
    return started


def test_an_uncheck_picked_while_the_scan_stops_waits_for_a_new_scan(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pick made after Stop is a new request: it is not dropped with the stopped scan."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    started = _stopping_scan(window, monkeypatch)
    criterion = _criterion({FIELD_TITLE}, match_all=True, phrase="a title")

    window._deselect_tagged(criterion)  # noqa: SLF001
    window._on_scan_finished()  # noqa: SLF001

    assert started == [1]
    assert window._scan_pending is criterion  # noqa: SLF001
    assert "Stopped reading metadata" not in window._status.text()  # noqa: SLF001


def test_a_photo_added_while_the_scan_stops_is_scanned(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop ends the scan it was pressed on, not the reading of photos added after it."""
    started = _stopping_scan(window, monkeypatch)

    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._on_scan_finished()  # noqa: SLF001

    assert started == [1]


def test_an_uncheck_stopped_stays_dropped_when_a_photo_restarts_the_scan(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo added after Stop restarts the scan, but not the uncheck Stop was pressed on."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    started: list[int] = []
    monkeypatch.setattr(window, "_start_metadata_scan", lambda: started.append(1))
    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_TITLE}, match_all=True, phrase="a title"),
    )
    started.clear()
    started = _stopping_scan(window, monkeypatch)

    other = tmp_path / "more"
    other.mkdir()
    _add_dir(window, {"b": _jpeg(other / "b.jpg")})
    window._on_scan_finished()  # noqa: SLF001

    assert started == [1]
    assert window._scan_pending is None  # noqa: SLF001
    assert "Stopped reading metadata" in window._status.text()  # noqa: SLF001


def test_a_stopped_scan_with_nothing_asked_after_it_stays_stopped(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a new request, Stop means stop: no scan starts again by itself."""
    started = _stopping_scan(window, monkeypatch)

    window._on_scan_finished()  # noqa: SLF001

    assert started == []


def test_removing_the_unscanned_photos_settles_a_waiting_uncheck(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Taking the unscanned photos off the list must not leave the pick waiting forever."""
    a = _jpeg(tmp_path / "a.jpg")  # scanned, has a title
    b = _jpeg(tmp_path / "b.jpg")  # the one the scan had not reached
    _add_dir(window, {"a": a, "b": b})
    window._on_scan_done({str(a): {FIELD_TITLE}})  # noqa: SLF001 - only a was read
    monkeypatch.setattr(window, "_start_metadata_scan", lambda: None)
    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_TITLE}, match_all=True, phrase="a title"),
    )

    window._remove_items([str(b)])  # noqa: SLF001

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._scan_pending is None  # noqa: SLF001
    assert "Deselected 1" in window._status.text()  # noqa: SLF001


def test_the_scan_row_only_shows_while_a_scan_runs(window: gui.MainWindow) -> None:
    """The delayed reveal must not pop a progress bar up after the scan it reports has ended."""
    window._show_scan_row()  # noqa: SLF001 - the timer firing with no scan left
    assert window._scan_row.isHidden()  # noqa: SLF001

    window._scan_thread = _FakeScanThread(finishes_in_time=True)  # noqa: SLF001
    window._show_scan_row()  # noqa: SLF001
    assert not window._scan_row.isHidden()  # noqa: SLF001

    window._hide_scan_row()  # noqa: SLF001
    window._scan_thread = None  # noqa: SLF001
    assert window._scan_row.isHidden()  # noqa: SLF001


def test_tagged_presets_all_run_without_error(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Every menu preset is wired to a real criterion the deselect handler accepts."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    _scan_reports(window, {})
    for _label, criterion in select_menu._TAGGED_PRESETS:  # noqa: SLF001
        window._deselect_tagged(criterion)  # noqa: SLF001
        assert criterion.phrase in window._status.text()  # noqa: SLF001


def test_deselect_tagged_with_no_photos_nags(window: gui.MainWindow) -> None:
    """With nothing added, the action reports it instead of starting a scan."""
    window._deselect_tagged(  # noqa: SLF001
        _criterion({FIELD_TITLE}, match_all=True, phrase="a title"),
    )
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_already_tagged_menu_actions_route_to_each_preset(window: gui.MainWindow) -> None:
    """Each preset in the Select menu asks for its own criterion (no late-binding bug)."""
    actions = window._select_menu._tagged_menu.actions()  # noqa: SLF001
    assert len(actions) == len(select_menu._TAGGED_PRESETS)  # noqa: SLF001

    captured: list[TaggedCriterion] = []
    window._select_menu.tagged_requested.connect(captured.append)  # noqa: SLF001
    for action in actions:
        action.trigger()

    assert captured == [criterion for _label, criterion in select_menu._TAGGED_PRESETS]  # noqa: SLF001


def test_select_menu_checks_and_unchecks_all(window: gui.MainWindow, tmp_path: Path) -> None:
    """Uncheck All clears every checkbox (model and rendered tree); Check All restores them."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})

    window._set_all_checked(checked=False)  # noqa: SLF001
    assert all(not item.selected for item in window._items.values())  # noqa: SLF001
    assert _check_state(window, a) == Qt.CheckState.Unchecked

    window._set_all_checked(checked=True)  # noqa: SLF001
    assert all(item.selected for item in window._items.values())  # noqa: SLF001
    assert _check_state(window, b) == Qt.CheckState.Checked


def test_set_all_checked_with_no_photos_nags(window: gui.MainWindow) -> None:
    """With nothing added, the bulk-select actions report it instead of rebuilding the tree."""
    window._set_all_checked(checked=False)  # noqa: SLF001
    assert "Add photos" in window._status.text()  # noqa: SLF001


def _add_two_shoots(window: gui.MainWindow, tmp_path: Path) -> tuple[Path, Path]:
    """Add two subfolders that each hold a photo of the same name, recursively."""
    first = tmp_path / "shoot1" / "DSC_0042.jpg"
    second = tmp_path / "shoot2" / "DSC_0042.jpg"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        _jpeg(path)
    window._extensions.setText("jpg")  # noqa: SLF001
    window._recursive.setChecked(True)  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001
    return first, second


def _top_rows(window: gui.MainWindow) -> list[QTreeWidgetItem]:
    """Return the tree's top-level rows."""
    tree = window._tree  # noqa: SLF001
    return [tree.topLevelItem(index) for index in range(tree.topLevelItemCount())]


def _search(window: gui.MainWindow, text: str) -> None:
    """Type into the search box and let its debounce fire at once."""
    window._search.setText(text)  # noqa: SLF001
    window._apply_search()  # noqa: SLF001


def test_search_narrows_the_tree_to_the_matches(window: gui.MainWindow, tmp_path: Path) -> None:
    """Typing a name leaves the matching photos (under their folders) and hides the rest."""
    first, second = _add_two_shoots(window, tmp_path)
    other = tmp_path / "shoot1" / "sunset.jpg"
    _jpeg(other)
    window._add_inputs([other])  # noqa: SLF001

    _search(window, "DSC")

    assert set(window._tree._leaf_rows) == {str(first), str(second)}  # noqa: SLF001
    # The folders holding a match survive; nothing else does.
    assert set(window._tree._folder_rows) == {  # noqa: SLF001
        str(tmp_path),
        str(first.parent),
        str(second.parent),
    }
    assert "Showing 2 of 3 photos" in window._status.text()  # noqa: SLF001


def test_search_matches_wildcards_and_clears(window: gui.MainWindow, tmp_path: Path) -> None:
    """A glob filters like the Select menu's patterns do, and emptying the box restores the list."""
    first, second = _add_two_shoots(window, tmp_path)
    edit = tmp_path / "shoot1" / "sunset_edit.jpg"
    _jpeg(edit)
    window._add_inputs([edit])  # noqa: SLF001

    _search(window, "*_edit.jpg")
    assert set(window._tree._leaf_rows) == {str(edit)}  # noqa: SLF001

    _search(window, "")
    assert set(window._tree._leaf_rows) == {str(first), str(second), str(edit)}  # noqa: SLF001
    assert "Showing" not in window._status.text()  # noqa: SLF001


def test_search_narrows_the_flat_list_too(window: gui.MainWindow, tmp_path: Path) -> None:
    """The two view modes filter the same way; only the grouping differs."""
    first, second = _add_two_shoots(window, tmp_path)
    other = tmp_path / "shoot1" / "sunset.jpg"
    _jpeg(other)
    window._add_inputs([other])  # noqa: SLF001
    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001

    _search(window, "DSC")

    assert set(window._tree._leaf_rows) == {str(first), str(second)}  # noqa: SLF001
    assert window._tree._folder_rows == {}  # noqa: SLF001


def test_search_hides_photos_without_dropping_them_from_a_run(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The search is a view filter, not a selection.

    A checked photo the box is hiding is still generated, the same way the thumbnail grid's own
    filter never changed what a run picks up. Nothing about the list itself is lost either.
    """
    first, _second = _add_two_shoots(window, tmp_path)
    hidden = tmp_path / "shoot1" / "sunset.jpg"
    _jpeg(hidden)
    window._add_inputs([hidden])  # noqa: SLF001

    _search(window, "DSC")

    assert str(hidden) not in window._tree._leaf_rows  # noqa: SLF001 - off screen
    assert window._items[str(hidden)].selected is True  # noqa: SLF001 - still checked
    started: list[list[str]] = []
    monkeypatch.setattr(
        gui.MainWindow,
        "_run_generation",
        lambda _self, items, **_kw: started.append([item.path.name for item in items]),
    )

    window._generate()  # noqa: SLF001

    assert sorted(started[0]) == ["DSC_0042.jpg", "DSC_0042.jpg", "sunset.jpg"]
    assert str(first) in window._tree._leaf_rows  # noqa: SLF001


def test_view_button_offers_the_two_layouts_with_tree_active(window: gui.MainWindow) -> None:
    """The quiet view button carries both layouts, exclusive, and starts on the folder tree."""
    labels = [action.text() for action in window._view_menu.actions()]  # noqa: SLF001
    assert labels == ["View as Tree", "View as List"]
    assert all(action.isCheckable() for action in window._view_menu.actions())  # noqa: SLF001
    assert window._view_actions[main_window.VIEW_TREE].isChecked()  # noqa: SLF001
    assert window._flat_list() is False  # noqa: SLF001
    # The button says nothing on its face, so the tooltip has to name the active layout.
    assert "View as Tree" in _unwrapped(window._view_button.toolTip())  # noqa: SLF001

    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001

    # Picking one mode drops the other: the two are one exclusive group.
    assert window._view_actions[main_window.VIEW_TREE].isChecked() is False  # noqa: SLF001
    assert window._flat_list() is True  # noqa: SLF001
    assert "View as List" in _unwrapped(window._view_button.toolTip())  # noqa: SLF001


def test_flat_list_puts_every_photo_at_one_level(window: gui.MainWindow, tmp_path: Path) -> None:
    """
    Flat drops the folder rows so a column sort runs across the whole list.

    The labels keep the folder each photo came from, or the two DSC_0042.jpg would be
    indistinguishable once the grouping is gone.
    """
    _add_two_shoots(window, tmp_path)

    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001

    rows = _top_rows(window)
    assert all(row.data(0, IS_DIR_ROLE) is None for row in rows)
    assert window._tree._folder_rows == {}  # noqa: SLF001
    assert sorted(row.text(0) for row in rows) == [
        str(Path("shoot1") / "DSC_0042.jpg"),
        str(Path("shoot2") / "DSC_0042.jpg"),
    ]


def test_flat_list_toggles_back_to_the_folder_tree(window: gui.MainWindow, tmp_path: Path) -> None:
    """Switching back restores the grouping, and a checkbox changed while flat survives it."""
    first, _second = _add_two_shoots(window, tmp_path)
    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001

    leaf = window._tree.leaf_for(first)  # noqa: SLF001
    assert leaf is not None
    leaf.setCheckState(0, Qt.CheckState.Unchecked)
    assert window._items[str(first)].selected is False  # noqa: SLF001

    window._view_actions[main_window.VIEW_TREE].setChecked(True)  # noqa: SLF001

    assert _check_state(window, first) == Qt.CheckState.Unchecked
    # The folder rows are back, and the one holding the unchecked photo says so.
    folder_row = window._tree._folder_rows[str(first.parent)]  # noqa: SLF001
    assert folder_row.checkState(0) == Qt.CheckState.Unchecked


def test_flat_list_keeps_the_open_photo_selected(window: gui.MainWindow, tmp_path: Path) -> None:
    """Flipping the view is not navigation: the photo being reviewed stays open and highlighted."""
    first, _second = _add_two_shoots(window, tmp_path)
    _select(window, window._tree.leaf_for(first))  # noqa: SLF001

    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001

    assert window._current is not None  # noqa: SLF001
    assert window._current.path == first  # noqa: SLF001
    assert window._tree.currentItem() is window._tree.leaf_for(first)  # noqa: SLF001


def test_flat_list_still_opens_the_enclosing_folder(window: gui.MainWindow, tmp_path: Path) -> None:
    """With no folder rows to click, Go > Enclosing Folder still reaches the thumbnail grid."""
    first, _second = _add_two_shoots(window, tmp_path)
    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001
    _select(window, window._tree.leaf_for(first))  # noqa: SLF001

    window._go_up()  # noqa: SLF001

    assert window._folder_grid.folder == first.parent  # noqa: SLF001
    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001


def test_history_survives_the_switch_to_the_flat_list(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A grid visited before the switch is still on the trail once its folder row is gone."""
    first, _second = _add_two_shoots(window, tmp_path)
    _select(window, window._tree._folder_rows[str(first.parent)])  # noqa: SLF001 - opens the grid
    _select(window, window._tree.leaf_for(first))  # noqa: SLF001

    window._view_actions[main_window.VIEW_LIST].setChecked(True)  # noqa: SLF001
    window._go_back()  # noqa: SLF001

    assert window._folder_grid.folder == first.parent  # noqa: SLF001
    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001
    # Nothing in the flat list stands for a folder, so no row is left highlighted.
    assert window._tree.currentItem() is None  # noqa: SLF001


def _select_submenu(window: gui.MainWindow, label: str) -> QMenu:
    """Return the Select menu's submenu titled *label* (asserting it is there)."""
    found = [
        action.menu()
        for action in window._select_menu.actions()  # noqa: SLF001
        if action.menu() is not None and action.menu().title() == label
    ]
    assert found, f"no {label} submenu"
    return found[0]


def test_select_by_file_type_lists_what_is_in_the_list(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """
    Check Only > File Type > dng leaves exactly the RAW files checked.

    The DNG+JPEG folder this exists for: the file-type entries are built when the menu opens, so
    they name the types actually present, with a count each.
    """
    raw = _jpeg(tmp_path / "IMG_0001.dng")  # a real JPEG under a RAW name: only the suffix matters
    photo = _jpeg(tmp_path / "IMG_0001.jpg")
    window._extensions.setText("jpg,dng")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001

    types = _select_submenu(window, "Check Only").actions()[0].menu()
    types.aboutToShow.emit()
    labels = [action.text() for action in types.actions()]
    assert labels == ["dng (1)", "jpg (1)"]
    next(action for action in types.actions() if action.text() == "dng (1)").trigger()

    assert window._items[str(raw)].selected is True  # noqa: SLF001
    assert window._items[str(photo)].selected is False  # noqa: SLF001
    assert _check_state(window, photo) == Qt.CheckState.Unchecked
    assert "Changed 1 photo (dng)" in window._status.text()  # noqa: SLF001


def test_select_by_status_reuses_the_grid_filters(window: gui.MainWindow, tmp_path: Path) -> None:
    """Uncheck > Status > Failed clears the failures and leaves the rest of the batch checked."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].status = FAILED  # noqa: SLF001

    status = _select_submenu(window, "Uncheck").actions()[1].menu()
    next(action for action in status.actions() if action.text() == "Failed").trigger()

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    assert "Changed 1 photo (Failed)" in window._status.text()  # noqa: SLF001


def test_select_by_status_checks_only_the_photos_without_a_copyright(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Check Only > Status > No copyright leaves checked just the photos with no notice."""
    bare = _jpeg(tmp_path / "a.jpg")
    owned = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": bare, "b": owned})
    window._on_scan_done({str(bare): {FIELD_TITLE}, str(owned): {FIELD_COPYRIGHT}})  # noqa: SLF001

    status = _select_submenu(window, "Check Only").actions()[1].menu()
    next(action for action in status.actions() if action.text() == "No copyright").trigger()

    assert window._items[str(bare)].selected is True  # noqa: SLF001
    assert window._items[str(owned)].selected is False  # noqa: SLF001
    assert "Changed 1 photo (No copyright)" in window._status.text()  # noqa: SLF001


def test_select_by_name_pattern_asks_and_remembers_the_pattern(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed pattern unchecks what it matches and comes back prefilled next time."""
    edit = _jpeg(tmp_path / "sunset_edit.jpg")
    keep = _jpeg(tmp_path / "sunset.jpg")
    _add_dir(window, {"edit": edit, "keep": keep})
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: ("*_edit.jpg", True))

    window._select_by_pattern(SELECT_UNCHECK)  # noqa: SLF001

    assert window._items[str(edit)].selected is False  # noqa: SLF001
    assert window._items[str(keep)].selected is True  # noqa: SLF001
    assert window._last_name_pattern == "*_edit.jpg"  # noqa: SLF001


def test_select_by_name_pattern_that_matches_nothing_says_so(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A criterion nothing matched reads differently from one that changed nothing."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: ("IMG", True))

    window._select_by_pattern(SELECT_CHECK)  # noqa: SLF001
    assert "No photos match IMG." in window._status.text()  # noqa: SLF001

    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: ("a.jpg", True))
    window._select_by_pattern(SELECT_CHECK)  # noqa: SLF001
    assert "Nothing to change for a.jpg" in window._status.text()  # noqa: SLF001


def test_select_by_pattern_cancelled_changes_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dismissing the dialog, or leaving it blank, leaves every checkbox alone."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    for reply in (("a.jpg", False), ("   ", True)):
        monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: reply)  # noqa: B023
        window._select_by_pattern(SELECT_UNCHECK)  # noqa: SLF001
        assert window._items[str(a)].selected is True  # noqa: SLF001


def test_select_by_criteria_with_no_photos_nags(window: gui.MainWindow) -> None:
    """Every criterion path reports an empty list rather than acting on nothing."""
    types = _select_submenu(window, "Check").actions()[0].menu()
    types.aboutToShow.emit()
    assert [action.isEnabled() for action in types.actions()] == [False]

    window._select_by_pattern(SELECT_CHECK)  # noqa: SLF001
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_apply_skip_file_unchecks_listed_photos(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A skip-list file deselects every photo it names, by bare filename or full path."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    skip = tmp_path / "skip.txt"
    skip.write_text(f"a.jpg\n{c}\n", encoding="utf-8")

    window._apply_skip_file(skip)  # noqa: SLF001

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    assert window._items[str(c)].selected is False  # noqa: SLF001
    # The rendered tree must repaint: a and c unchecked, b still checked.
    assert _check_state(window, a) == Qt.CheckState.Unchecked
    assert _check_state(window, b) == Qt.CheckState.Checked
    assert _check_state(window, c) == Qt.CheckState.Unchecked
    assert "Deselected 2" in window._status.text()  # noqa: SLF001


def test_apply_skip_file_reports_empty_list(window: gui.MainWindow, tmp_path: Path) -> None:
    """A comment-only (no usable entries) file says so rather than 'Deselected 0'."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    skip = tmp_path / "empty.txt"
    skip.write_text("# only a comment\n\n", encoding="utf-8")

    window._apply_skip_file(skip)  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "no usable entries" in window._status.text()  # noqa: SLF001


def test_apply_skip_file_reports_no_matches(window: gui.MainWindow, tmp_path: Path) -> None:
    """A non-empty list that matches nothing reports it, leaving every photo checked."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    skip = tmp_path / "skip.txt"
    skip.write_text("nonexistent.jpg\n", encoding="utf-8")

    window._apply_skip_file(skip)  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "matched" in window._status.text().lower()  # noqa: SLF001


def test_apply_skip_file_warns_on_unreadable_file(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable skip list pops a warning and changes no selection."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    warned: list[str] = []
    _patch_gui(
        monkeypatch,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args, **_k: warned.append(str(args[-1]))),
    )

    window._apply_skip_file(tmp_path / "missing.txt")  # noqa: SLF001

    assert warned  # a warning dialog was shown
    assert window._items[str(a)].selected is True  # noqa: SLF001


# ---------------------------------------------------------------------------
# Detail pane
# ---------------------------------------------------------------------------


def test_selecting_a_photo_loads_and_populates(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a file shows existing metadata and seeds the editable fields."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    leaf = window._tree.leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001 - triggers _on_current_changed

    assert window._detail._existing_title.text() == "Old Title"  # noqa: SLF001
    assert "Beach" in window._detail._existing_keywords.toPlainText()  # noqa: SLF001
    assert window._detail._title.text() == "Old Title"  # noqa: SLF001
    assert window._detail._keywords.toPlainText() == "Beach"  # noqa: SLF001


def test_edits_and_hint_survive_navigating_between_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-progress edits (here the hint) stick to their photo when the user browses away."""
    _stub_reads(monkeypatch, keywords=[])
    files = {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "b.jpg")}
    _add_dir(window, files)

    _select(window, window._tree.leaf_for(files["a"]))  # noqa: SLF001
    window._detail._hint.setText("The animal is a deer")  # noqa: SLF001
    _select(window, window._tree.leaf_for(files["b"]))  # noqa: SLF001

    assert window._items[str(files["a"])].hint == "The animal is a deer"  # noqa: SLF001
    assert window._detail._hint.text() == ""  # noqa: SLF001 - photo b has no hint of its own

    _select(window, window._tree.leaf_for(files["a"]))  # noqa: SLF001
    assert window._detail._hint.text() == "The animal is a deer"  # noqa: SLF001


def test_hierarchy_preview_updates_from_keywords(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing the keywords field refreshes the resulting-hierarchy preview."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._save_options_menu.overwrite_actions[FIELD_KEYWORDS].setChecked(True)  # noqa: SLF001
    window._detail._keywords.setPlainText("Duck<Bird<Animal")  # noqa: SLF001 - triggers textChanged
    assert window._detail._hierarchy.toPlainText() == "Animal\n└─ Bird\n   └─ Duck"  # noqa: SLF001


def test_save_current_writes_and_marks_saved(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save merges keywords, calls write_metadata, and marks the item saved."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    captured: dict[str, object] = {}

    def fake_write(*_a: object, **kwargs: object) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._detail._title.setText("New Title")  # noqa: SLF001
    window._detail._keywords.setPlainText("Eagle\nSky")  # noqa: SLF001
    # Scanned and found bare, so the column may trust what the save wrote.
    window._items[str(img)].known_fields = set()  # noqa: SLF001
    window._save_current()  # noqa: SLF001

    item = window._items[str(img)]  # noqa: SLF001
    assert item.status == SAVED
    assert captured["title"] == "New Title"
    # The save wrote title, description (seeded from the file), and keywords, so the Tagged
    # column reflects all three without waiting for a rescan.
    assert item.known_fields == {FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}
    leaf = window._tree.leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.text(COL_TAGGED) == "TDK"


def test_save_writes_both_targets_and_journals_each(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write To > Both saves the photo and a sidecar, and undo is told about each."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    targets: list[bool] = []

    def fake_write(image_path: Path, *_a: object, use_sidecar: bool = True, **_k: object) -> bool:
        targets.append(use_sidecar)
        # Create the sidecar the way exiftool would, so the journal can stat it.
        image_path.with_suffix(".xmp").write_text("written")
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    journal = open_journal(datetime.now(tz=UTC))
    assert journal is not None
    monkeypatch.setattr(window, "_single_save_journal", lambda: journal)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._save_options_menu.sidecar_actions["both"].setChecked(True)  # noqa: SLF001
    window._save_current()  # noqa: SLF001

    assert targets == [False, True]  # the photo, then its sidecar
    assert window._items[str(img)].status == SAVED  # noqa: SLF001
    written = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    assert [entry["target"] for entry in written] == [str(img), str(img.with_suffix(".xmp"))]


def test_save_keeps_a_caption_whose_overwrite_entry_is_off(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking Overwrite Existing > Description leaves it alone without touching keywords."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    captured: dict[str, object] = {}

    def fake_write(*args: object, **kwargs: object) -> bool:
        captured.update(kwargs)
        captured["keywords"] = args[1]
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._save_options_menu.overwrite_actions[FIELD_DESCRIPTION].setChecked(False)  # noqa: SLF001
    window._detail._description.setPlainText("A new caption.")  # noqa: SLF001
    window._detail._keywords.setPlainText("Eagle")  # noqa: SLF001
    window._save_current()  # noqa: SLF001

    assert captured["description"] is None  # the photo keeps "Old caption."
    assert captured["title"] == "Old Title"  # its own entry is still checked
    assert sorted(captured["keywords"].subject) == ["Beach", "Eagle"]  # type: ignore[attr-defined]


def test_detail_pane_names_the_caption_a_sidecar_shadows(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The camera's placeholder is named under the field instead of vanishing behind the sidecar."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(
        monkeypatch,
        keywords=[],
        captions={
            FIELD_TITLE: [CaptionValue("A Title", SOURCE_SIDECAR)],
            FIELD_DESCRIPTION: [
                CaptionValue("A real caption.", SOURCE_SIDECAR),
                CaptionValue("default", SOURCE_IMAGE),
            ],
        },
    )
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    assert window._detail._existing_description.toPlainText() == "A real caption."  # noqa: SLF001
    note = window._detail._description_source.text()  # noqa: SLF001
    assert 'shadows "default" in the image file' in note
    assert not window._detail._description_source.isHidden()  # noqa: SLF001
    # The title has one source, so its note says where it came from and nothing more.
    assert window._detail._title_source.text() == "from XMP sidecar"  # noqa: SLF001


def test_detail_columns_line_up_row_by_row(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Each row's Existing and New widget share a top edge, which is the point of the two columns.

    The source notes broke this once by being nested in their field's grid cell: the grid then sized
    the row by the read-only box's sizeHint rather than its minimum, stretching the Description row
    and leaving the New box floating in the middle of it.
    """
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(
        monkeypatch,
        keywords=["Beach"],
        captions={
            FIELD_TITLE: [CaptionValue("A Title", SOURCE_SIDECAR)],
            FIELD_DESCRIPTION: [
                CaptionValue("A real caption.", SOURCE_SIDECAR),
                CaptionValue("default", SOURCE_IMAGE),
            ],
        },
    )
    _add_dir(window, {"a": img})
    window.resize(1200, 900)
    window.show()
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    QApplication.processEvents()

    rows = (
        (window._detail._existing_title, window._detail._title),  # noqa: SLF001
        (window._detail._existing_description, window._detail._description),  # noqa: SLF001
        (window._detail._existing_keywords, window._detail._keywords),  # noqa: SLF001
    )
    assert [existing.y() for existing, _new in rows] == [new.y() for _existing, new in rows]


def test_detail_pane_hides_the_source_note_for_a_field_no_file_carries(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty note would still cost a line of the pane, so it is hidden instead."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(
        monkeypatch,
        keywords=[],
        captions={FIELD_TITLE: [], FIELD_DESCRIPTION: []},
    )
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    # isHidden, not isVisible: nothing is visible in a window the test never shows, but only an
    # explicit hide sets isHidden.
    assert window._detail._title_source.isHidden()  # noqa: SLF001
    assert window._detail._description_source.isHidden()  # noqa: SLF001


def test_opening_a_photo_seeds_the_new_column_like_the_existing_one(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo opened without generating starts with its own keywords, chains and all."""
    img = _jpeg(tmp_path / "a.jpg")
    _patch_gui(monkeypatch, "read_caption", lambda _p: (None, None))
    _patch_gui(
        monkeypatch,
        "read_caption_values",
        lambda _p: {FIELD_TITLE: [], FIELD_DESCRIPTION: []},
    )
    _patch_gui(
        monkeypatch,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(
            existing_keywords=KeywordSet(
                subject=["Animal", "Bird", "Duck", "Sky"],
                hierarchical=["Animal|Bird|Duck"],
            ),
        ),
    )
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    # Both columns render the same way, so they can be read line by line.
    assert window._detail._keywords.toPlainText() == window._detail._existing_keywords.toPlainText()  # noqa: SLF001
    assert window._detail._keywords.toPlainText() == "Duck<Bird<Animal\nSky"  # noqa: SLF001


def test_selecting_shows_metadata_source(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Source field reports where the existing metadata was read from."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _patch_gui(monkeypatch, "read_metadata_sources", lambda _p: ["XMP sidecar"])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    assert window._detail._existing_source.text() == "(from XMP sidecar)"  # noqa: SLF001


# ---------------------------------------------------------------------------
# Empty / placeholder state (the idle right pane)
# ---------------------------------------------------------------------------


def test_empty_state_is_shown_at_startup(window: gui.MainWindow) -> None:
    """A fresh window shows the placeholder page, not an empty detail form."""
    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001
    assert "Add photos" in window._empty_message.text()  # noqa: SLF001


def test_empty_state_nudges_to_pick_a_photo_once_loaded(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """With photos loaded but none open, the placeholder asks the user to pick one."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._on_current_changed(None, None)  # noqa: SLF001 - selection cleared, no photo open

    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001
    assert "Select a photo" in window._empty_message.text()  # noqa: SLF001


def test_clear_returns_to_the_empty_state(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clearing the list returns the right pane to the getting-started placeholder."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._clear()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001
    assert "Add photos" in window._empty_message.text()  # noqa: SLF001


def test_removing_the_open_photo_returns_to_the_empty_state(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the photo being inspected drops back to the placeholder, not a stale form."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001 - open it -> detail page
    assert window._right.currentIndex() == main_window._PAGE_DETAIL  # noqa: SLF001

    window._tree.setCurrentItem(window._tree.leaf_for(img))  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001

    assert window._current is None  # noqa: SLF001
    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------


def _patch_save_dialog(monkeypatch: pytest.MonkeyPatch, path: Path | str) -> None:
    """Replace the save-file dialog so it returns *path* without opening a real dialog."""
    _patch_gui(
        monkeypatch,
        "QFileDialog",
        SimpleNamespace(getSaveFileName=lambda *_a, **_k: (str(path), "CSV files (*.csv)")),
    )


def test_export_csv_without_items_shows_hint(window: gui.MainWindow) -> None:
    """Exporting with an empty list nudges the user instead of writing a file."""
    window._export_csv()  # noqa: SLF001
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_export_csv_writes_report(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export writes one CSV row per photo, folding in the edited working fields."""
    import csv as csv_module  # noqa: PLC0415 - test-local parser.

    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._detail._title.setText("New Title")  # noqa: SLF001
    window._detail._keywords.setPlainText("Eagle\nSky")  # noqa: SLF001

    target = tmp_path / "report.csv"
    _patch_save_dialog(monkeypatch, target)
    window._export_csv()  # noqa: SLF001

    with target.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv_module.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["filename"] == "a.jpg"
    assert rows[0]["title"] == "New Title"
    assert "Eagle" in rows[0]["keywords"]
    assert "Beach" in rows[0]["keywords"]  # merged with the existing keyword
    assert rows[0]["existing_keywords"] == "Beach"
    assert "Exported 1" in window._status.text()  # noqa: SLF001


def test_export_csv_appends_missing_suffix(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chosen path without a .csv suffix gets one so the file is unambiguous."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _patch_save_dialog(monkeypatch, tmp_path / "report")  # no extension
    window._export_csv()  # noqa: SLF001
    assert (tmp_path / "report.csv").exists()


def test_export_csv_cancel_writes_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the dialog (empty path) leaves the filesystem untouched."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _patch_save_dialog(monkeypatch, "")
    window._export_csv()  # noqa: SLF001
    assert list(tmp_path.glob("*.csv")) == []


def test_export_csv_reports_write_error(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure surfaces a warning dialog rather than crashing the GUI."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _patch_save_dialog(monkeypatch, tmp_path / "report.csv")

    def boom(*_a: object, **_k: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    _patch_gui(monkeypatch, "write_report", boom)
    messages: list[str] = []
    _patch_gui(
        monkeypatch,
        "QMessageBox",
        SimpleNamespace(warning=lambda _parent, _title, text, *_a, **_k: messages.append(text)),
    )
    window._export_csv()  # noqa: SLF001
    assert messages
    assert "disk full" in messages[0]


def test_save_selected_writes_only_checked_generated_items(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save selected writes checked photos with a proposal; an unchecked one is skipped."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _stub_save_helper(monkeypatch)
    written: list[str] = []
    monkeypatch.setattr(
        metadata,
        "write_metadata",
        lambda path, *_a, **_k: written.append(path.name) or True,
    )
    _add_dir(window, {"a": a, "b": b})
    # Both generated, but only "a" is checked; "b" is generated yet unchecked -> skipped.
    for name, selected in ((a, True), (b, False)):
        item = window._items[str(name)]  # noqa: SLF001
        item.has_proposal = True
        item.selected = selected
        item.title = "T"
        item.keywords = ["Eagle"]

    window._save_selected()  # noqa: SLF001
    _drain_save(window)
    assert written == ["a.jpg"]
    assert window._items[str(a)].status == SAVED  # noqa: SLF001
    assert "Saved 1 of 1" in window._status.text()  # noqa: SLF001


def test_save_marks_failed_when_write_fails(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed write_metadata sets the item status to failed."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    monkeypatch.setattr(metadata, "write_metadata", lambda *_a, **_k: False)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._save_current()  # noqa: SLF001
    assert window._items[str(img)].status == FAILED  # noqa: SLF001


def test_save_marks_failed_when_write_raises(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    write_metadata raising (e.g. exiftool missing) fails the save instead of crashing silently.

    Regression test: the single-photo save path had no try/except around write_metadata, unlike the
    batch SaveWorker path, so this exact scenario used to propagate all the way to Qt's exception
    hook: the click handler died mid-save with no status update and no error shown.
    """
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])

    def boom(*_a: object, **_k: object) -> bool:
        message = "exiftool missing"
        raise FileNotFoundError(message)

    monkeypatch.setattr(metadata, "write_metadata", boom)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._save_current()  # noqa: SLF001 - must not raise
    assert window._items[str(img)].status == FAILED  # noqa: SLF001


def _two_ready_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Add two checked photos that both carry a proposal, ready for a batch save."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _stub_save_helper(monkeypatch)
    _add_dir(window, {"a": a, "b": b})
    for path in (a, b):
        item = window._items[str(path)]  # noqa: SLF001
        item.has_proposal = True
        item.title = "T"
        item.keywords = ["Eagle"]
        # Scanned and found bare, so the Tagged column may trust what a save writes.
        item.known_fields = set()
    return a, b


def test_batch_save_runs_in_the_background_with_progress(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch save shows the bar and clock, locks the buttons, and frees them when done."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    monkeypatch.setattr(metadata, "write_metadata", lambda *_a, **_k: True)

    window._save_selected()  # noqa: SLF001
    assert window._save_thread is not None  # noqa: SLF001 - the writes happen off the UI thread
    assert window._run_bar._progress.maximum() == 2  # noqa: SLF001, PLR2004 - both photos
    assert not window._run_bar._timing.isHidden()  # noqa: SLF001
    assert not window._run_bar.generate_button.isEnabled()  # noqa: SLF001
    assert not window._run_bar.save_button.isEnabled()  # noqa: SLF001
    assert window._run_bar.cancel_button.isEnabled()  # noqa: SLF001 - a long save is cancellable

    _drain_save(window)
    assert window._run_bar._progress.isHidden()  # noqa: SLF001
    assert window._run_bar.save_button.isEnabled()  # noqa: SLF001
    assert window._items[str(a)].status == SAVED  # noqa: SLF001
    assert window._items[str(b)].status == SAVED  # noqa: SLF001
    assert window._items[str(a)].known_fields == {FIELD_TITLE, FIELD_KEYWORDS}  # noqa: SLF001


def test_batch_save_reports_a_failed_write_in_the_tally(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One photo failing to write leaves the rest saved and is counted in the final message."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    monkeypatch.setattr(metadata, "write_metadata", lambda path, *_a, **_k: path.name != "b.jpg")

    window._save_selected()  # noqa: SLF001
    _drain_save(window)
    assert window._items[str(a)].status == SAVED  # noqa: SLF001
    assert window._items[str(b)].status == FAILED  # noqa: SLF001
    assert "Saved 1 of 2" in window._status.text()  # noqa: SLF001


def test_batch_save_refuses_to_start_while_generating(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two runs never overlap: a save asked for mid-generation is ignored."""
    a, _b = _two_ready_photos(window, tmp_path, monkeypatch)
    _stub_generation(monkeypatch)
    written: list[str] = []
    monkeypatch.setattr(
        metadata,
        "write_metadata",
        lambda path, *_a, **_k: written.append(path.name),
    )

    window._run_generation([window._items[str(a)]])  # noqa: SLF001
    window._save_selected()  # noqa: SLF001
    assert window._save_thread is None  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started
    assert written == []


def test_cancelled_save_frees_the_unwritten_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a save leaves the photos it never reached ready to save again."""
    a, _b = _two_ready_photos(window, tmp_path, monkeypatch)
    window._items[str(a)].status = WORKING  # noqa: SLF001 - never reached before the cancel
    window._cancelling = True  # noqa: SLF001

    window._on_save_finished()  # noqa: SLF001

    assert window._items[str(a)].status == READY  # noqa: SLF001 - proposal still there to save
    assert "Cancelled after saving 0 photos" in window._status.text()  # noqa: SLF001


def test_cancel_stops_a_running_save(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one Cancel button also covers a save: the worker is asked to stop."""
    _two_ready_photos(window, tmp_path, monkeypatch)
    monkeypatch.setattr(metadata, "write_metadata", lambda *_a, **_k: True)

    window._save_selected()  # noqa: SLF001
    window._cancel_generation()  # noqa: SLF001
    assert window._cancelling  # noqa: SLF001
    assert not window._run_bar.cancel_button.isEnabled()  # noqa: SLF001
    _drain_save(window)


def test_save_worker_shares_one_exiftool_across_the_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every write in a batch reuses the same helper instead of spawning one process per photo."""
    helper = object()
    opened = 0

    @contextmanager
    def fake_helper(_et: object) -> Iterator[object]:
        nonlocal opened
        opened += 1
        yield helper

    _patch_gui(monkeypatch, "managed_helper", fake_helper)
    seen: list[object] = []
    monkeypatch.setattr(
        metadata,
        "write_metadata",
        lambda *_a, et=None, **_k: bool(seen.append(et)),
    )
    jobs = [
        SaveJob(
            path=tmp_path / name,
            keywords=KeywordSet(subject=["K"]),
            title="T",
            description=None,
        )
        for name in ("a.jpg", "b.jpg")
    ]
    worker = workers.SaveWorker(jobs, options=SaveOptions(backup=True, sidecar_mode="all"))
    results: list[tuple[str, bool]] = []
    worker.file_done.connect(lambda path, ok: results.append((Path(path).name, ok)))

    worker.run()

    assert opened == 1
    assert seen == [helper, helper]
    assert len(results) == 2  # noqa: PLR2004 - one result per job


def test_save_worker_reports_failures_when_exiftool_cannot_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken exiftool fails every photo rather than leaving the window waiting forever."""

    def boom(_et: object) -> object:
        message = "exiftool missing"
        raise OSError(message)

    _patch_gui(monkeypatch, "managed_helper", boom)
    jobs = [SaveJob(path=tmp_path / "a.jpg", keywords=KeywordSet(), title="T", description=None)]
    worker = workers.SaveWorker(jobs, options=SaveOptions(backup=True, sidecar_mode="all"))
    results: list[bool] = []
    finished: list[bool] = []
    worker.file_done.connect(lambda _path, ok: results.append(ok))
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert results == [False]
    assert finished == [True]


def test_save_worker_stops_before_the_next_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop request takes effect at the next photo; the write in flight still completes."""
    _patch_gui(monkeypatch, "managed_helper", lambda _et: nullcontext(None))
    jobs = [
        SaveJob(
            path=tmp_path / name,
            keywords=KeywordSet(subject=["K"]),
            title=None,
            description=None,
        )
        for name in ("a.jpg", "b.jpg", "c.jpg")
    ]
    worker = workers.SaveWorker(jobs, options=SaveOptions(backup=True, sidecar_mode="all"))
    written: list[str] = []

    def fake_write(path: Path, *_a: object, **_k: object) -> bool:
        written.append(path.name)
        worker.stop()  # asked to cancel while the first photo is being written
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    worker.run()
    assert written == ["a.jpg"]


# ---------------------------------------------------------------------------
# Per-field write toggles (Title / Description / Keywords)
# ---------------------------------------------------------------------------


def _capture_write(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch write_metadata to record its positional keywords and keyword arguments."""
    captured: dict[str, object] = {}

    def fake_write(path: Path, keywords: object, **kwargs: object) -> bool:
        captured["path"] = path
        captured["keywords"] = keywords
        captured.update(kwargs)
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    return captured


def test_save_writes_only_title_and_description_when_keywords_unchecked(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking 'Keywords' writes title + description and hands write_metadata no keywords."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._detail._title.setText("New Title")  # noqa: SLF001
    window._detail._description.setPlainText("New description.")  # noqa: SLF001
    window._detail._keywords.setPlainText("Eagle")  # noqa: SLF001
    window._save_options_menu.write_keywords.setChecked(False)  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["title"] == "New Title"
    assert captured["description"] == "New description."
    # An empty KeywordSet means write_metadata emits no keyword tags, so existing ones survive.
    assert captured["keywords"].subject == []  # type: ignore[attr-defined]


def test_save_skips_title_when_title_unchecked(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking 'Title' nulls the title kwarg while keywords and description still write."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._detail._title.setText("New Title")  # noqa: SLF001
    window._save_options_menu.write_title.setChecked(False)  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["title"] is None


def test_save_with_no_write_fields_nags_and_writes_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every write toggle off, Save reports it and never touches the file."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    wrote: list[int] = []
    monkeypatch.setattr(metadata, "write_metadata", lambda *_a, **_k: wrote.append(1) or True)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    options = window._save_options_menu  # noqa: SLF001
    for checkbox in (options.write_title, options.write_description, options.write_keywords):
        checkbox.setChecked(False)

    window._save_current()  # noqa: SLF001

    assert wrote == []
    assert "at least one field" in window._status.text()  # noqa: SLF001


def test_save_keeps_the_exiftool_backup_by_default(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out of the box the GUI matches the CLI default and lets ExifTool keep *_original."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert window._save_options_menu.backup.isChecked()  # noqa: SLF001
    assert captured["backup"] is True


def test_unchecking_backup_writes_without_an_original_copy(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning the backup off passes backup=False, so ExifTool overwrites in place."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._save_options_menu.backup.setChecked(False)  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["backup"] is False


def test_unchecking_write_keywords_disables_overwrite_and_blanks_diff(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning keywords off grays out Overwrite and shows the diff is moot."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    window._save_options_menu.write_keywords.setChecked(False)  # noqa: SLF001 - fires _on_write_field_toggled

    assert not window._save_options_menu.overwrite_actions[FIELD_KEYWORDS].isEnabled()  # noqa: SLF001
    assert "keywords will not be written" in window._detail._diff.toPlainText().lower()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Logs and failure reporting
# ---------------------------------------------------------------------------


def test_file_failure_surfaces_reason_on_the_open_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed photo records its reason, shows the error banner, and tooltips the tree cell."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001 - open the photo

    window._on_file_failed(str(img), "model unreachable")  # noqa: SLF001

    item = window._items[str(img)]  # noqa: SLF001
    assert item.status == FAILED
    assert item.error == "model unreachable"
    assert not window._detail._error_banner.isHidden()  # noqa: SLF001
    assert "model unreachable" in window._detail._error_banner.text()  # noqa: SLF001
    leaf = window._tree.leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.toolTip(2) == "model unreachable"  # Status column


def test_opening_a_healthy_photo_hides_the_error_banner(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The banner only shows for the failed photo; opening a healthy one clears it."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.leaf_for(a))  # noqa: SLF001
    window._on_file_failed(str(a), "boom")  # noqa: SLF001
    assert not window._detail._error_banner.isHidden()  # noqa: SLF001

    _select(window, window._tree.leaf_for(b))  # noqa: SLF001 - healthy photo
    assert window._detail._error_banner.isHidden()  # noqa: SLF001


def test_retry_failed_targets_only_failed_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry failed re-runs every failed photo and leaves the rest alone."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].status = FAILED  # noqa: SLF001
    window._items[str(b)].status = READY  # noqa: SLF001

    captured: list[tuple[list[Path], str]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **kwargs: captured.append(([i.path for i in items], kwargs["trigger"])),
    )
    window._retry_failed()  # noqa: SLF001

    assert captured == [([a], "retry")]


def test_retry_failed_with_nothing_failed_nags(window: gui.MainWindow, tmp_path: Path) -> None:
    """With no failures, Retry failed reports it instead of starting a run."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._retry_failed()  # noqa: SLF001
    assert window._thread is None  # noqa: SLF001
    assert "No failed photos" in window._status.text()  # noqa: SLF001


def test_retrying_the_open_photo_clears_its_banner(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-queuing the open failed photo hides the stale banner and marks it working."""
    img = _jpeg(tmp_path / "a.jpg")
    # Stub the whole generation path so the background worker does no real I/O.
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    assert not window._detail._error_banner.isHidden()  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    # Checked synchronously, before the worker thread's queued signals are processed: the
    # pre-run bookkeeping marks the photo working and clears its stale failure banner.
    assert window._items[str(img)].status == WORKING  # noqa: SLF001
    assert window._detail._error_banner.isHidden()  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_open_logs_creates_the_folder_and_reveals_it(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open logs makes sure the folder exists and hands it to the OS file browser."""
    folder = tmp_path / "logs"
    _patch_gui(monkeypatch, "LOG_FOLDER", folder)
    opened: list[str] = []
    _patch_gui(
        monkeypatch,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened.append(url.toLocalFile()) or True),
    )
    window._open_logs()  # noqa: SLF001
    assert folder.is_dir()
    assert opened == [str(folder)]


def _raise_oserror(*_args: object, **_kwargs: object) -> None:
    """Stand in for a Popen that cannot start the program it was given."""
    message = "Exec format error"
    raise OSError(message)


def test_reveal_falls_back_to_the_folder_when_the_browser_will_not_start(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file browser that fails to spawn opens the folder instead of crashing the window."""
    photo = _jpeg(tmp_path / "a.jpg")
    opened: list[str] = []
    _patch_gui(
        monkeypatch,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened.append(url.toLocalFile()) or True),
    )
    _patch_gui(monkeypatch, "reveal_command", lambda _path, _platform: ["/bin/nope"])
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _raise_oserror,
    )

    window._reveal(photo)  # noqa: SLF001

    assert opened == [str(tmp_path)]


# ---------------------------------------------------------------------------
# Folder thumbnail grid
# ---------------------------------------------------------------------------


def test_selecting_a_folder_shows_the_thumbnail_grid(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a folder switches the right pane to a grid of its photos."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    grid = window._folder_grid  # noqa: SLF001
    monkeypatch.setattr(grid, "_start_thumbs", lambda _paths: None)  # no background thread
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - the folder node

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001
    assert window._folder_grid._list.count() == 2  # noqa: SLF001, PLR2004 - two photos in the folder


def test_clicking_a_thumbnail_opens_the_detail(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activating a grid item selects that photo and shows its detail page."""
    a = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    grid_item = window._folder_grid._list.item(0)  # noqa: SLF001

    window._folder_grid._on_thumb_activated(grid_item)  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_DETAIL  # noqa: SLF001
    assert window._current is window._items[str(a)]  # noqa: SLF001


def test_thumbnail_worker_emits_bytes(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker decodes each path to bytes and stops when asked."""
    _patch_gui(
        monkeypatch,
        "prepare_image_for_agent",
        lambda *_a, **_k: SimpleNamespace(data=b"jpeg-bytes"),
    )
    emitted: list[tuple[str, bytes]] = []
    worker = workers.ThumbnailWorker([Path("/a.jpg"), Path("/b.jpg")])
    worker.ready.connect(lambda path, data: emitted.append((path, data)))
    worker.run()
    assert emitted == [("/a.jpg", b"jpeg-bytes"), ("/b.jpg", b"jpeg-bytes")]


def test_on_thumb_ready_sets_icon_and_caches(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished thumbnail updates the grid item and is cached."""
    a = _jpeg(tmp_path / "a.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001
    window._folder_grid._on_thumb_ready(str(a), a.read_bytes())  # noqa: SLF001
    assert str(a) in window._folder_grid._thumbs  # noqa: SLF001


def test_a_thumbnail_for_a_removed_photo_is_dropped(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detached job outlives the list it was started for; its late results are not cached."""
    a = _jpeg(tmp_path / "a.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": _jpeg(tmp_path / "b.jpg")})
    window._remove_items([str(a)])  # noqa: SLF001

    window._folder_grid._on_thumb_ready(str(a), a.read_bytes())  # noqa: SLF001

    assert str(a) not in window._folder_grid._thumbs  # noqa: SLF001


# Generous next to the 250 ms _stop_thumbs allows, but far under the 10 s decode the test holds
# open: the assertion is "it detached", not a timing measurement.
_STOP_THUMBS_BUDGET_S = 2.0


def test_stopping_thumbnails_does_not_block_on_a_slow_decode(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Navigation must not wait out the decode in flight.

    The worker only checks its stop flag between photos, so _stop_thumbs detaches rather than
    blocking the UI thread for however long one photo takes.
    """
    decoding = threading.Event()
    release = threading.Event()

    def slow_decode(*_args: object, **_kwargs: object) -> SimpleNamespace:
        decoding.set()
        release.wait(10.0)
        return SimpleNamespace(data=b"")

    _patch_gui(monkeypatch, "prepare_image_for_agent", slow_decode)
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - opens the grid, starts the job
    assert window._folder_grid._thumb_thread is not None  # noqa: SLF001
    detached = window._folder_grid._thumb_thread  # noqa: SLF001
    # The stop flag is only read between photos, so the freeze needs a decode already running.
    assert decoding.wait(5.0), "the worker never reached the decode"

    started = time.monotonic()
    try:
        window._folder_grid.stop_thumbs()  # noqa: SLF001
        elapsed = time.monotonic() - started
        assert elapsed < _STOP_THUMBS_BUDGET_S, f"_stop_thumbs blocked for {elapsed:.1f}s"
        assert window._folder_grid._thumb_thread is None  # noqa: SLF001
    finally:
        release.set()
        detached.wait(5000)


def _grid_order(window: gui.MainWindow) -> list[str]:
    """Return the thumbnail filenames in the grid, in display order."""
    view = window._folder_grid.view  # noqa: SLF001
    return [view.item(i).text() for i in range(view.count())]


def test_grid_sorts_by_name_by_default(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freshly shown grid orders thumbnails by name, not the order they were added."""
    files = {name: _jpeg(tmp_path / f"{name}.jpg") for name in ("c", "a", "b")}
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, files)

    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    assert _grid_order(window) == ["a.jpg", "b.jpg", "c.jpg"]


def test_grid_sort_direction_toggle_reverses_order(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flipping the direction toggle re-sorts the visible grid from A-Z to Z-A in place."""
    files = {name: _jpeg(tmp_path / f"{name}.jpg") for name in ("c", "a", "b")}
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, files)
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    window._folder_grid._sort_dir.setChecked(True)  # noqa: SLF001 - ascending -> descending

    assert _grid_order(window) == ["c.jpg", "b.jpg", "a.jpg"]


def test_grid_sort_by_status_uses_lifecycle_rank(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Choosing the Status sort orders thumbnails by lifecycle, mirroring the tree's Status sort."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": b, "c": c})
    window._items[str(a)].status = FAILED  # noqa: SLF001
    window._items[str(b)].status = PENDING  # noqa: SLF001
    window._items[str(c)].status = READY  # noqa: SLF001
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    combo = window._folder_grid._sort_combo  # noqa: SLF001
    combo.setCurrentIndex(combo.findData(SORT_STATUS))

    # Ascending lifecycle: pending(b) < ready(c) < failed(a), not the alphabetical file order.
    assert _grid_order(window) == ["b.jpg", "c.jpg", "a.jpg"]


def test_grid_sort_survives_navigating_between_folders(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chosen sort persists as the user moves between folders within a session."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    _jpeg(tmp_path / "one" / "b.jpg")
    _jpeg(tmp_path / "one" / "a.jpg")
    _jpeg(tmp_path / "two" / "d.jpg")
    _jpeg(tmp_path / "two" / "c.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001
    _select(window, window._tree.topLevelItem(0).child(0))  # noqa: SLF001 - first subfolder
    window._folder_grid._sort_dir.setChecked(True)  # noqa: SLF001 - descending

    _select(window, window._tree.topLevelItem(0).child(1))  # noqa: SLF001 - second subfolder

    assert _grid_order(window) == ["d.jpg", "c.jpg"]


def test_grid_filter_shows_only_matching_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Show filter hides photos not in the chosen state, leaving the rest in the list."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": b, "c": c})
    window._items[str(a)].status = SAVED  # noqa: SLF001
    window._items[str(b)].status = SAVED  # noqa: SLF001
    window._items[str(c)].status = FAILED  # noqa: SLF001
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    combo = window._folder_grid._filter_combo  # noqa: SLF001
    combo.setCurrentIndex(combo.findData(FILTER_SAVED))

    assert _grid_order(window) == ["a.jpg", "b.jpg"]
    assert len(window._items) == 3  # noqa: SLF001, PLR2004 - filtered from the grid, not removed


def test_grid_filter_status_message_reports_hidden_count(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a filter active, the status bar shows how many of the folder's photos are visible."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].status = FAILED  # noqa: SLF001
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    combo = window._folder_grid._filter_combo  # noqa: SLF001
    combo.setCurrentIndex(combo.findData(FILTER_FAILED))

    text = window._status.text()  # noqa: SLF001
    assert "1" in text
    assert "2" in text


def test_grid_filter_untagged_refreshes_after_metadata_scan(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Untagged filter re-evaluates when the background scan reports, not just at build time."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    combo = window._folder_grid._filter_combo  # noqa: SLF001
    combo.setCurrentIndex(combo.findData(FILTER_UNTAGGED))
    # Nothing is scanned yet (known_fields is None), so no photo is hidden or shown as untagged.
    assert _grid_order(window) == []

    window._on_scan_done({str(a): set(), str(b): {FIELD_TITLE}})  # noqa: SLF001

    assert _grid_order(window) == ["a.jpg"]  # only the scanned-empty photo


def test_grid_filter_no_copyright_refreshes_after_metadata_scan(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No copyright reads the same scan as Untagged, so it re-filters when the scan reports."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    combo = window._folder_grid._filter_combo  # noqa: SLF001
    combo.setCurrentIndex(combo.findData(FILTER_NO_COPYRIGHT))
    assert _grid_order(window) == []

    window._on_scan_done({str(a): {FIELD_TITLE}, str(b): {FIELD_COPYRIGHT}})  # noqa: SLF001

    assert _grid_order(window) == ["a.jpg"]


def test_a_thumbnail_with_a_notice_gets_the_copyright_badge(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C badge draws next to M, and hovering the thumbnail names it."""
    a = _jpeg(tmp_path / "a.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a})
    window._on_scan_done({str(a): {FIELD_TITLE, FIELD_COPYRIGHT}})  # noqa: SLF001
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    tip = window._folder_grid._list.item(0).toolTip()  # noqa: SLF001
    assert tip == "a.jpg\nalready has metadata\nhas a copyright notice"


def test_grid_combos_are_sized_to_their_widest_label(window: gui.MainWindow) -> None:
    """Both grid combos reserve room for their longest entry so no label is clipped."""
    for combo in (window._folder_grid._filter_combo, window._folder_grid._sort_combo):  # noqa: SLF001
        widest = max(len(combo.itemText(i)) for i in range(combo.count()))
        assert combo.minimumContentsLength() == widest


# ---------------------------------------------------------------------------
# Back / Forward / Enclosing Folder
# ---------------------------------------------------------------------------


def _open_grid(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...] = ("a.jpg", "b.jpg"),
) -> dict[str, Path]:
    """Add a folder of photos and open its thumbnail grid, with no background jobs."""
    files = {name: _jpeg(tmp_path / name) for name in names}
    _stub_reads(monkeypatch, keywords=[])
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, files)
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - the folder node
    return files


def test_back_returns_from_a_photo_to_the_grid_it_was_opened_from(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grid -> thumbnail -> Back lands on the grid again, with the folder row selected."""
    files = _open_grid(window, tmp_path, monkeypatch)

    window._folder_grid._on_thumb_activated(window._folder_grid._cells[str(files["a.jpg"])])  # noqa: SLF001
    assert window._right.currentIndex() == main_window._PAGE_DETAIL  # noqa: SLF001

    window._go_back()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001
    assert window._folder_grid.folder == tmp_path  # noqa: SLF001
    current = window._tree.currentItem()  # noqa: SLF001
    assert current is not None
    assert current.data(0, PATH_ROLE) == str(tmp_path)


def test_back_and_forward_walk_between_two_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back returns to the photo seen before this one, and Forward comes back to it."""
    files = _open_grid(window, tmp_path, monkeypatch)
    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001
    _select(window, window._tree.leaf_for(files["b.jpg"]))  # noqa: SLF001

    window._go_back()  # noqa: SLF001
    assert window._current is window._items[str(files["a.jpg"])]  # noqa: SLF001

    window._go_forward()  # noqa: SLF001
    assert window._current is window._items[str(files["b.jpg"])]  # noqa: SLF001


def test_back_keeps_the_edits_typed_before_leaving(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stepping back commits the open photo's fields, like any other navigation."""
    files = _open_grid(window, tmp_path, monkeypatch)
    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001
    _select(window, window._tree.leaf_for(files["b.jpg"]))  # noqa: SLF001
    window._detail._hint.setText("the second frame")  # noqa: SLF001

    window._go_back()  # noqa: SLF001

    assert window._items[str(files["b.jpg"])].hint == "the second frame"  # noqa: SLF001


def test_nav_arrows_are_disabled_until_there_is_somewhere_to_go(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both arrows start gray, Back arms on the second place, Forward only after a Back."""
    assert not window._back_button.isEnabled()  # noqa: SLF001
    assert not window._forward_button.isEnabled()  # noqa: SLF001

    files = _open_grid(window, tmp_path, monkeypatch)
    assert not window._back_button.isEnabled()  # noqa: SLF001 - the grid is the first place

    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001
    assert window._back_button.isEnabled()  # noqa: SLF001
    assert not window._forward_button.isEnabled()  # noqa: SLF001

    window._go_back()  # noqa: SLF001
    assert window._forward_button.isEnabled()  # noqa: SLF001


def test_nav_arrows_name_their_target_and_the_crumb_says_where_you_are(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hints name the place they lead to, so the arrows are not two mystery buttons."""
    files = _open_grid(window, tmp_path, monkeypatch)
    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001

    assert tmp_path.name in window._back_button.toolTip()  # noqa: SLF001
    assert "a.jpg" in window._crumb.text()  # noqa: SLF001

    window._go_back()  # noqa: SLF001

    assert "a.jpg" in window._forward_button.toolTip()  # noqa: SLF001
    assert window._crumb.text() == tmp_path.name  # noqa: SLF001


def test_back_skips_a_photo_that_left_the_list(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a photo takes it out of the trail, so Back never opens a dead row."""
    files = _open_grid(window, tmp_path, monkeypatch, names=("a.jpg", "b.jpg", "c.jpg"))
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _select(window, window._tree.leaf_for(files[name]))  # noqa: SLF001

    window._remove_items([str(files["b.jpg"])])  # noqa: SLF001
    window._go_back()  # noqa: SLF001

    assert window._current is window._items[str(files["a.jpg"])]  # noqa: SLF001


def test_removing_the_open_photo_leaves_back_pointing_at_it_no_more(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pane empties, and Back returns to the grid rather than the removed photo."""
    files = _open_grid(window, tmp_path, monkeypatch)
    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001

    window._remove_items([str(files["a.jpg"])])  # noqa: SLF001
    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001

    window._go_back()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001


def test_clearing_the_list_empties_the_trail(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear List leaves nowhere to go back to, so both arrows go gray again."""
    files = _open_grid(window, tmp_path, monkeypatch)
    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001

    window._clear()  # noqa: SLF001

    assert not window._back_button.isEnabled()  # noqa: SLF001
    assert not window._forward_button.isEnabled()  # noqa: SLF001
    assert window._crumb.text() == ""  # noqa: SLF001


def test_back_on_an_empty_trail_does_nothing(window: gui.MainWindow) -> None:
    """The keyboard shortcut still fires when there is nothing behind; it must be a no-op."""
    window._go_back()  # noqa: SLF001
    window._go_forward()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001


def test_enclosing_folder_opens_the_grid_of_the_open_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo picked straight from the tree can still jump up to its contact sheet."""
    files = _open_grid(window, tmp_path, monkeypatch)
    _select(window, window._tree.leaf_for(files["a.jpg"]))  # noqa: SLF001

    window._go_up()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001
    assert window._folder_grid.folder == tmp_path  # noqa: SLF001


def test_enclosing_folder_at_the_top_says_so(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top folder has nothing above it in the list, and the status bar explains that."""
    _open_grid(window, tmp_path, monkeypatch)

    window._go_up()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001 - stayed put
    assert window._status.text() != ""  # noqa: SLF001


def test_enclosing_folder_with_nothing_open_is_a_no_op(window: gui.MainWindow) -> None:
    """Nothing is open, so there is no enclosing folder to show."""
    window._go_up()  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_EMPTY  # noqa: SLF001


def test_go_menu_holds_the_navigation_actions(window: gui.MainWindow) -> None:
    """The three moves are in the menu bar, which is also what arms their shortcuts."""
    actions = [action for action in window._go_menu.actions() if not action.isSeparator()]  # noqa: SLF001
    assert actions == [
        window._back_action,  # noqa: SLF001
        window._forward_action,  # noqa: SLF001
        window._up_action,  # noqa: SLF001
    ]
    assert all(not action.shortcut().isEmpty() for action in actions)


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------


def _stub_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gui(monkeypatch, "create_agent", lambda *_a, **_k: object())
    _patch_gui(
        monkeypatch,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(existing_keywords=KeywordSet(subject=["Beach"])),
    )
    _patch_gui(monkeypatch, "read_caption", lambda _p: ("Old", "Old caption."))
    _patch_gui(
        monkeypatch,
        "prepare_image_for_agent",
        lambda *_a, **_k: SimpleNamespace(data=b"x"),
    )
    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="T", description="D", keywords=["Eagle"]),
    )


def test_worker_emits_a_proposal_per_photo(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker reads context, runs the model, and emits one proposal per photo."""
    _stub_generation(monkeypatch)
    proposals: list[Proposal] = []
    finished: list[bool] = []
    worker = workers.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
    worker.file_done.connect(proposals.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(proposals) == 1
    assert proposals[0].title == "T"
    assert proposals[0].keywords == ["Eagle"]
    assert finished == [True]


def test_worker_marks_all_failed_when_agent_cannot_build(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the agent cannot be created, every photo is reported as failed."""

    def boom(*_a: object, **_k: object) -> object:
        msg = "unreachable"
        raise ProviderError(msg)

    _patch_gui(monkeypatch, "create_agent", boom)
    failed: list[str] = []
    worker = workers.GenerateWorker("openai", "m", None, [Path("/a.jpg"), Path("/b.jpg")])
    worker.file_failed.connect(lambda path, _msg: failed.append(path))

    worker.run()

    assert failed == ["/a.jpg", "/b.jpg"]


def test_worker_reports_finished_even_when_opening_the_cache_raises(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Whatever goes wrong, the window hears the run is over.

    `finished` is what re-enables the buttons and clears the "working..." rows, so a worker that
    dies without emitting it leaves the window unusable for the rest of the session. Opening the
    cache sat outside the block that guarantees the signal.
    """
    _stub_generation(monkeypatch)

    def boom(_self: object) -> object:
        msg = "namespace blew up"
        raise RuntimeError(msg)

    monkeypatch.setattr(workers.GenerateWorker, "_open_cache", boom)
    done: list[bool] = []
    worker = workers.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
    worker.finished.connect(lambda: done.append(True))

    with pytest.raises(RuntimeError, match="namespace blew up"):
        worker.run()

    assert done == [True]


def test_worker_reports_a_single_file_failure(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-photo error is reported without aborting the batch."""
    _stub_generation(monkeypatch)

    def boom(*_a: object, **_k: object) -> object:
        msg = "decode failed"
        raise ValueError(msg)

    _patch_gui(monkeypatch, "prepare_image_for_agent", boom)
    failed: list[str] = []
    worker = workers.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
    worker.file_failed.connect(lambda path, _msg: failed.append(path))

    worker.run()

    assert failed == ["/a.jpg"]


def test_generate_current_targets_only_the_open_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate this photo runs for the open photo, ignoring which photos are checked."""
    a = _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a})  # both files added; all checked by default
    _select(window, window._tree.leaf_for(a))  # noqa: SLF001 - open "a"

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_kwargs: captured.append([i.path for i in items]),
    )
    window._generate_current()  # noqa: SLF001

    assert captured == [[a]]


def test_generate_targets_checked_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate selected runs for the checked photos only."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(b)].selected = False  # noqa: SLF001 - uncheck b

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_kwargs: captured.append([i.path for i in items]),
    )
    window._generate()  # noqa: SLF001

    assert captured == [[a]]


def test_generate_current_needs_an_open_photo(window: gui.MainWindow) -> None:
    """With no photo open, Generate this photo nags instead of starting a run."""
    window._generate_current()  # noqa: SLF001
    assert window._thread is None  # noqa: SLF001 - no generation started
    assert "Open a photo" in window._status.text()  # noqa: SLF001


def test_worker_stops_after_current_photo_when_cancelled(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling stop() halts the batch at the next photo boundary, not mid-photo."""
    _stub_generation(monkeypatch)
    proposals: list[Proposal] = []
    worker = workers.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg"), Path("/b.jpg")])

    def record_then_cancel(proposal: Proposal) -> None:
        # Cancel as soon as the first photo's proposal lands; the second photo must not run.
        proposals.append(proposal)
        worker.stop()

    worker.file_done.connect(record_then_cancel)

    worker.run()

    assert len(proposals) == 1


def test_cancel_generation_stops_worker_and_disables_button(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel asks the worker to stop, flags the run as cancelling, and grays the button."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    window._run_generation([window._items[str(img)]])  # noqa: SLF001
    assert window._run_bar.cancel_button.isEnabled()  # noqa: SLF001 - enabled while running

    window._cancel_generation()  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._stop is True  # noqa: SLF001
    assert window._cancelling is True  # noqa: SLF001
    assert not window._run_bar.cancel_button.isEnabled()  # noqa: SLF001
    assert "Cancelling" in window._status.text()  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_finish_after_cancel_resets_working_photos_to_pending(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A cancelled run frees the un-started (still-WORKING) photos and reports the tally."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].status = WORKING  # noqa: SLF001 - never reached before cancel
    window._items[str(b)].status = READY  # noqa: SLF001 - already generated, must survive
    window._cancelling = True  # noqa: SLF001

    window._on_generate_finished()  # noqa: SLF001

    assert window._items[str(a)].status == PENDING  # noqa: SLF001
    assert window._items[str(b)].status == READY  # noqa: SLF001
    assert "Cancelled" in window._status.text()  # noqa: SLF001
    assert "1 photo not generated" in window._status.text()  # noqa: SLF001


def test_on_file_done_applies_proposal_to_item(window: gui.MainWindow, tmp_path: Path) -> None:
    """A finished proposal updates the matching item and its tree status."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    proposal = Proposal(
        path=img,
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="Generated",
        description="Desc.",
        keywords=["Eagle"],
    )
    window._on_file_done(proposal)  # noqa: SLF001
    item = window._items[str(img)]  # noqa: SLF001
    assert item.status == READY
    assert item.title == "Generated"


# ---------------------------------------------------------------------------
# menu bar + telemetry toggle
# ---------------------------------------------------------------------------


def test_menu_bar_has_file_tools_settings_help(window: gui.MainWindow) -> None:
    """The window carries a real menu bar with File, Tools, Settings, and Help menus."""
    titles = [action.text() for action in window.menuBar().actions()]
    assert "File" in titles
    assert "Tools" in titles
    assert "Settings" in titles
    assert "Help" in titles


def test_tools_menu_offers_every_library_job(window: gui.MainWindow) -> None:
    """Building, harmonizing, watching, and undoing all live one click away."""
    entries = [action.text() for action in window._tools_menu.actions()]  # noqa: SLF001
    assert entries == [
        "Build Vocabulary...",
        "Harmonize Shoots Now",
        "",  # separator
        "Watch Folder...",
        "",  # separator
        "Undo Writes...",
    ]


def test_telemetry_toggle_defaults_on_and_persists(window: gui.MainWindow) -> None:
    """The Settings toggle reflects the default (on) and persists the choice when changed."""
    action = window._settings_menu._telemetry_action  # noqa: SLF001
    assert action.isCheckable()
    assert action.isChecked() is True  # default: telemetry on, no saved preference yet
    assert window._settings_menu._telemetry_enabled is True  # noqa: SLF001

    action.setChecked(False)  # user turns it off from the menu
    assert window._settings_menu._telemetry_enabled is False  # noqa: SLF001
    assert telemetry.read_gui_pref() is False  # the choice is persisted for next launch


def test_telemetry_toggle_shows_do_not_track_and_checking_it_opts_in(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Under DO_NOT_TRACK the box starts unchecked, since nothing is sent, and checking it opts in.

    A box left checked would claim a send that DO_NOT_TRACK quietly stops.
    """
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    win = gui.MainWindow()
    try:
        action = win._settings_menu._telemetry_action  # noqa: SLF001
        assert action.isChecked() is False
        assert action.isEnabled() is True
        assert "DO_NOT_TRACK" in action.toolTip()

        action.setChecked(True)  # the user opts Photo Tagger in anyway
        assert telemetry.read_gui_pref() is True
        consent = win._settings_menu.consent()  # noqa: SLF001
        assert consent.send is True
        assert consent.reason == telemetry.CONSENT_GUI_OPT_IN
        assert "turned this on here" in " ".join(action.toolTip().split())
    finally:
        win.close()


def test_telemetry_toggle_is_grayed_out_under_an_environment_opt_out(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PHOTO_TAGGER_TELEMETRY=0 cannot be undone from the menu, so the toggle cannot pretend to."""
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", "0")
    win = gui.MainWindow()
    try:
        action = win._settings_menu._telemetry_action  # noqa: SLF001
        assert action.isChecked() is False
        assert action.isEnabled() is False
        assert "PHOTO_TAGGER_TELEMETRY" in action.toolTip()
    finally:
        win.close()


def test_notify_toggles_default_off_and_gate_each_other(window: gui.MainWindow) -> None:
    """Alerts start off, and the sound sub-toggle is only reachable once alerts are on."""
    alert = window._settings_menu._notify_action  # noqa: SLF001
    sound = window._settings_menu._notify_sound_action  # noqa: SLF001
    assert alert.isChecked() is False
    assert sound.isEnabled() is False

    alert.setChecked(True)
    assert sound.isEnabled() is True

    alert.setChecked(False)
    assert sound.isEnabled() is False


def test_a_saved_sound_survives_while_alerts_are_off(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sound box keeps its saved value, so turning alerts back on brings the sound with it."""
    config = tmp_path / "config.toml"
    config.write_text("[display]\nnotify = false\nnotify_sound = true\n", encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))
    win = gui.MainWindow()
    try:
        assert win._settings_menu._notify_action.isChecked() is False  # noqa: SLF001
        sound = win._settings_menu._notify_sound_action  # noqa: SLF001
        assert sound.isChecked() is True
        assert sound.isEnabled() is False
    finally:
        win.close()


def test_finished_generation_alerts_with_the_status_line(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The notification repeats what the window says, so the two never disagree."""
    raised: list[tuple[str, str, bool, bool]] = []
    _patch_gui(
        monkeypatch,
        "notify",
        lambda title, message, **kwargs: raised.append(
            (title, message, kwargs["enabled"], kwargs["sound"]),
        ),
    )
    window._settings_menu._notify_action.setChecked(True)  # noqa: SLF001
    window._settings_menu._notify_sound_action.setChecked(True)  # noqa: SLF001

    window._on_generate_finished()  # noqa: SLF001

    assert raised == [("Photo Tagger", "Generation finished.", True, True)]


def test_cancelled_generation_raises_no_alert(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whoever pressed Cancel is already at the keyboard and needs no calling back."""
    raised: list[str] = []
    _patch_gui(monkeypatch, "notify", lambda _t, message, **_k: raised.append(message))
    window._settings_menu._notify_action.setChecked(True)  # noqa: SLF001
    window._cancelling = True  # noqa: SLF001

    window._on_generate_finished()  # noqa: SLF001

    assert raised == []


def test_finished_save_alerts_with_its_tally(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished write run announces how many photos it wrote."""
    raised: list[str] = []
    _patch_gui(monkeypatch, "notify", lambda _t, message, **_k: raised.append(message))
    window._settings_menu._notify_action.setChecked(True)  # noqa: SLF001

    window._on_save_finished()  # noqa: SLF001

    assert raised == ["Saved 0 of 0 checked photos."]


def test_alert_is_skipped_while_the_toggle_is_off(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default window still calls notify, but with enabled=False, which spawns nothing."""
    calls: list[bool] = []
    _patch_gui(monkeypatch, "notify", lambda _t, _m, **kwargs: calls.append(kwargs["enabled"]))

    window._on_generate_finished()  # noqa: SLF001

    assert calls == [False]


def test_language_menu_lists_system_default_and_catalogs(window: gui.MainWindow) -> None:
    """The Settings > Language menu offers System Default plus every shipped language."""
    labels = [action.text() for action in window._settings_menu._language_menu.actions()]  # noqa: SLF001
    assert labels[0] == "System Default"
    assert "English" in labels
    assert "Português (Brasil)" in labels
    # No explicit choice saved: the system-default entry starts checked.
    assert window._settings_menu._language_menu.actions()[0].isChecked()  # noqa: SLF001


def test_language_choice_persists_to_the_config_file(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Picking a language writes the config key; picking System Default removes it."""
    target = tmp_path / "config.toml"
    target.write_text('# keep me\nextensions = "jpg"\n', encoding="utf-8")
    _patch_gui(monkeypatch, "find_config_file", lambda: target)

    window._settings_menu.set_language("pt_BR")  # noqa: SLF001
    text = target.read_text(encoding="utf-8")
    assert 'language = "pt_BR"' in text
    assert "# keep me" in text
    assert "Restart" in window._status.text()  # noqa: SLF001

    window._settings_menu.set_language("auto")  # noqa: SLF001
    assert "language" not in target.read_text(encoding="utf-8")


def test_output_language_menu_lists_english_first_then_other(window: gui.MainWindow) -> None:
    """
    The menu offers one-click language entries: English (checked) first, Other...

    last.
    """
    actions = window._settings_menu._output_language_menu.actions()  # noqa: SLF001
    labels = [action.text() for action in actions if not action.isSeparator()]
    assert labels[0] == "English"
    assert labels[-1] == "Other..."
    assert "Brazilian Portuguese" in labels
    # No explicit choice saved: the default entry starts checked.
    assert window._settings_menu._output_language_actions["English"].isChecked()  # noqa: SLF001


def test_output_language_defaults_to_english_and_persists(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The metadata language starts at English; a choice persists and English unpins it."""
    assert window._settings_menu.output_language == "English"  # noqa: SLF001
    target = tmp_path / "config.toml"
    target.write_text('# keep me\nextensions = "jpg"\n', encoding="utf-8")
    _patch_gui(monkeypatch, "find_config_file", lambda: target)

    window._settings_menu.set_output_language("Brazilian Portuguese")  # noqa: SLF001
    text = target.read_text(encoding="utf-8")
    assert 'output_language = "Brazilian Portuguese"' in text
    assert "# keep me" in text
    assert window._settings_menu.output_language == "Brazilian Portuguese"  # noqa: SLF001
    assert window._settings_menu._output_language_actions["Brazilian Portuguese"].isChecked()  # noqa: SLF001
    assert "Brazilian Portuguese" in window._status.text()  # noqa: SLF001

    window._settings_menu.set_output_language("English")  # noqa: SLF001
    assert "output_language" not in target.read_text(encoding="utf-8")
    assert window._settings_menu._output_language_actions["English"].isChecked()  # noqa: SLF001


def test_output_language_custom_value_gets_its_own_checked_entry(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    A language typed via Other...

    joins the menu as a new checked entry above Other.
    """
    target = tmp_path / "config.toml"
    _patch_gui(monkeypatch, "find_config_file", lambda: target)

    window._settings_menu.set_output_language("Swahili")  # noqa: SLF001

    action = window._settings_menu._output_language_actions["Swahili"]  # noqa: SLF001
    assert action.isChecked()
    menu_actions = window._settings_menu._output_language_menu.actions()  # noqa: SLF001
    assert action in menu_actions
    assert menu_actions.index(action) < len(menu_actions) - 1  # above the Other... row
    assert 'output_language = "Swahili"' in target.read_text(encoding="utf-8")


def test_output_language_blank_means_the_default(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    A blank Other...

    entry falls back to English instead of sending an empty language.
    """
    target = tmp_path / "config.toml"
    _patch_gui(monkeypatch, "find_config_file", lambda: target)
    window._settings_menu.set_output_language("German")  # noqa: SLF001
    window._settings_menu.set_output_language("   ")  # noqa: SLF001
    assert window._settings_menu.output_language == "English"  # noqa: SLF001
    assert window._settings_menu._output_language_actions["English"].isChecked()  # noqa: SLF001
    assert "output_language" not in target.read_text(encoding="utf-8")


def test_telemetry_toggle_reflects_saved_off_preference(qapp: QApplication) -> None:
    """A window built after the user disabled telemetry comes up unchecked."""
    telemetry.write_gui_pref(enabled=False)
    win = gui.MainWindow()
    try:
        assert win._settings_menu._telemetry_enabled is False  # noqa: SLF001
        assert win._settings_menu._telemetry_action.isChecked() is False  # noqa: SLF001
    finally:
        win.close()


def test_close_flushes_telemetry_beacon(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Closing the window sends the session beacon and waits for it.

    The process exits right after closeEvent; a beacon still on its daemon thread would die with the
    process, so the GUI beacon would be silently lost on every run.
    """
    sent: list[object] = []
    flushed: list[list[object]] = []

    def fake_emit(event: object, **_kwargs: object) -> object:
        sent.append(event)
        return f"thread for {type(event).__name__}"

    monkeypatch.setattr(telemetry, "emit", fake_emit)
    monkeypatch.setattr(telemetry, "flush", lambda threads: flushed.append(list(threads)))
    window.close()

    assert [type(event) for event in sent] == [telemetry.SessionInfo]
    assert sent[0].interface == "gui"  # type: ignore[attr-defined]
    assert flushed == [[None, "thread for SessionInfo"]], "close waits for what it sent"


def test_close_beacon_reports_session_tagged_count_and_fields(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session beacon reports the generated-photo count and both languages."""
    runs: list[telemetry.SessionInfo] = []
    monkeypatch.setattr(telemetry, "emit", lambda run, **_: runs.append(run))
    monkeypatch.setattr(i18n, "current_language", lambda: "pt_BR")

    img_a = _jpeg(tmp_path / "a.jpg")
    img_b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": img_a, "b": img_b})
    for img in (img_a, img_b):
        window._on_file_done(  # noqa: SLF001
            Proposal(
                path=img,
                existing_title=None,
                existing_description=None,
                existing_keywords=KeywordSet(),
                title="T",
                description="D",
                keywords=["K"],
            ),
        )
    window._settings_menu._output_language = "German"  # noqa: SLF001

    window.close()

    assert len(runs) == 1
    assert runs[0].photos_generated == 2  # noqa: PLR2004 - two distinct photos this session
    assert runs[0].output_language == "German"
    assert runs[0].ui_language == "pt_BR"


def test_close_asks_for_confirmation_when_proposals_are_unsaved(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An unsaved proposal prompts for confirmation; declining keeps the window open.

    Regression test: closeEvent used to discard every unsaved title/description/keyword edit with no
    confirmation at all.
    """
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    item = window._items[str(img)]  # noqa: SLF001
    item.has_proposal = True
    item.status = READY

    asked: list[object] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **_k: asked.append(args) or QMessageBox.StandardButton.No,
    )

    assert window.close() is False  # declining keeps the window open
    assert asked  # the confirmation dialog was shown


def test_close_proceeds_when_user_confirms_discarding_unsaved_proposals(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirming the close-anyway prompt lets the window close as normal."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    item = window._items[str(img)]  # noqa: SLF001
    item.has_proposal = True
    item.status = READY
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Yes,
    )

    assert window.close() is True


def test_close_skips_confirmation_when_nothing_unsaved(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved (or never-generated) photo needs no prompt at all."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    item = window._items[str(img)]  # noqa: SLF001
    item.has_proposal = True
    item.status = SAVED
    asked: list[object] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **_k: asked.append(args) or QMessageBox.StandardButton.Yes,
    )

    assert window.close() is True
    assert not asked


# ---------------------------------------------------------------------------
# Redesigned chrome: progress bar, details disclosure, save options, menus
# ---------------------------------------------------------------------------


def test_progress_bar_tracks_the_run(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The progress bar appears for a run, ticks once per finished photo, and hides after."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    assert window._run_bar._progress.isHidden()  # noqa: SLF001 - idle: no bar

    window._run_generation([window._items[str(img)]])  # noqa: SLF001
    assert not window._run_bar._progress.isHidden()  # noqa: SLF001
    assert window._run_bar._progress.maximum() == 1  # noqa: SLF001
    assert window._run_bar._progress.value() == 0  # noqa: SLF001

    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=img,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="T",
            description="D",
            keywords=[],
        ),
    )
    assert window._run_bar._progress.value() == 1  # noqa: SLF001

    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started
    assert window._run_bar._progress.isHidden()  # noqa: SLF001


def test_timing_readout_shows_elapsed_then_an_estimate(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clock beside the bar runs during a generation and disappears when it is over."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    assert window._run_bar._timing.isHidden()  # noqa: SLF001 - idle: no clock

    window._run_generation([window._items[str(img)]])  # noqa: SLF001
    assert not window._run_bar._timing.isHidden()  # noqa: SLF001
    # Backdate the start so the readout has something to show without a real wait.
    window._run_bar._started = time.monotonic() - 30  # noqa: SLF001
    window._run_bar._update_timing()  # noqa: SLF001
    assert window._run_bar._timing.text() == "0:30 elapsed"  # noqa: SLF001 - nothing finished yet

    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started
    assert window._run_bar._timing.isHidden()  # noqa: SLF001
    assert window._run_bar._started is None  # noqa: SLF001


def test_timing_readout_estimates_the_remaining_time_mid_run(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Once photos have finished, the readout extrapolates how long the rest will take."""
    window._run_bar.start_clock(4)  # noqa: SLF001
    window._run_bar._started = time.monotonic() - 60  # noqa: SLF001
    window._run_bar._progress.setValue(1)  # noqa: SLF001
    window._run_bar._update_timing()  # noqa: SLF001
    assert window._run_bar._timing.text() == "1:00 elapsed · 3:00 left"  # noqa: SLF001
    window._run_bar.stop_clock()  # noqa: SLF001


def test_advance_progress_ignores_ticks_when_idle(window: gui.MainWindow) -> None:
    """A late signal from a torn-down run must not move a bar that belongs to nothing."""
    before = window._run_bar._progress.value()  # noqa: SLF001
    window._run_bar.advance()  # noqa: SLF001
    assert window._run_bar._progress.value() == before  # noqa: SLF001


def test_details_disclosure_expands_and_collapses(window: gui.MainWindow) -> None:
    """The keyword-change details start collapsed and follow the disclosure toggle."""
    assert window._detail._details_panel.isHidden()  # noqa: SLF001
    window._detail._details_toggle.setChecked(True)  # noqa: SLF001
    assert not window._detail._details_panel.isHidden()  # noqa: SLF001
    assert window._detail._details_toggle.arrowType() == Qt.ArrowType.DownArrow  # noqa: SLF001
    window._detail._details_toggle.setChecked(False)  # noqa: SLF001
    assert window._detail._details_panel.isHidden()  # noqa: SLF001
    assert window._detail._details_toggle.arrowType() == Qt.ArrowType.RightArrow  # noqa: SLF001


def test_details_toggle_summarizes_keyword_changes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collapsed disclosure still shows how many keywords a save would add or remove."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    window._detail._keywords.setPlainText("Beach\nEagle")  # noqa: SLF001 - one new keyword
    assert "+1" in window._detail._details_toggle.text()  # noqa: SLF001

    window._detail._keywords.setPlainText("Beach")  # noqa: SLF001 - back to the existing set
    assert "no change" in window._detail._details_toggle.text()  # noqa: SLF001


def test_save_options_default_to_writing_all_fields(window: gui.MainWindow) -> None:
    """The Save options menu defaults match the CLI: write every field, merge, sidecar."""
    options = window._save_options_menu  # noqa: SLF001
    for action in (options.write_title, options.write_description, options.write_keywords):
        assert action.isCheckable()
        assert action.isChecked()
    overwrite = window._save_options_menu.overwrite_actions  # noqa: SLF001
    assert not overwrite[FIELD_KEYWORDS].isChecked()  # merge, not overwrite
    # A title and a description hold one value each, so the generated one replaces it.
    assert overwrite[FIELD_TITLE].isChecked()
    assert overwrite[FIELD_DESCRIPTION].isChecked()
    assert window._save_options_menu.sidecar_mode() == "all"  # noqa: SLF001 - sidecar, not embed
    assert window._save_options_menu.backup.isChecked()  # noqa: SLF001 - keep ExifTool's *_original


def test_file_menu_offers_csv_export(window: gui.MainWindow) -> None:
    """The CSV export moved off the toolbar and lives in the File menu."""
    texts = [action.text() for action in window._file_menu.actions()]  # noqa: SLF001
    assert "Export CSV Report..." in texts


def test_scan_options_hold_extensions_and_recursion(window: gui.MainWindow) -> None:
    """The Add menu's scan options carry the folder-scan settings with the GUI defaults."""
    assert window._extensions.text() == DEFAULT_GUI_EXTENSIONS  # noqa: SLF001
    assert window._recursive.isChecked()  # noqa: SLF001


def test_retry_button_enabled_only_with_failures(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Retry failed starts disabled, lights up on a failure, and resets with the list."""
    assert not window._run_bar.retry_button.isEnabled()  # noqa: SLF001 - nothing failed yet
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    assert not window._run_bar.retry_button.isEnabled()  # noqa: SLF001

    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    assert window._run_bar.retry_button.isEnabled()  # noqa: SLF001

    window._clear()  # noqa: SLF001
    assert not window._run_bar.retry_button.isEnabled()  # noqa: SLF001


def test_generate_menus_offer_one_time_skip_cache(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Generate split buttons' arrow menus run once without the cache."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001 - so "this photo" has a target

    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **kwargs: calls.append((len(items), kwargs.get("use_cache", True))),
    )
    window._run_bar.generate_menu.actions()[0].trigger()  # noqa: SLF001
    window._detail.generate_menu.actions()[0].trigger()  # noqa: SLF001

    assert calls == [(1, False), (1, False)]


def test_grid_thumbnails_share_the_context_menu(window: gui.MainWindow) -> None:
    """The folder grid asks for the same custom context menu as the tree."""
    policy = window._folder_grid._list.contextMenuPolicy()  # noqa: SLF001
    assert policy == Qt.ContextMenuPolicy.CustomContextMenu


def test_help_menu_links_the_documentation(window: gui.MainWindow) -> None:
    """Help offers a Documentation entry pointing at the hosted docs."""
    texts = [action.text() for action in window._help_menu.actions()]  # noqa: SLF001
    assert "Documentation" in texts


def test_save_buttons_share_the_options_menu(window: gui.MainWindow) -> None:
    """Both Save buttons carry the same arrow menu holding the write toggles."""
    options = window._save_options_menu  # noqa: SLF001
    actions = options.actions()
    assert options.write_title in actions
    assert options.backup in actions
    # The Write To and Overwrite Existing choices hang off submenus of that same menu.
    submenus = [action.menu() for action in actions if action.menu() is not None]
    assert any(options.sidecar_actions["raw"] in menu.actions() for menu in submenus)
    keywords = options.overwrite_actions[FIELD_KEYWORDS]
    assert any(keywords in menu.actions() for menu in submenus)


def _unwrapped(tip: str) -> str:
    """Join a wrapped tooltip back into one line, so tests can assert on its wording."""
    return " ".join(tip.split())


def test_long_tooltips_are_wrapped_into_lines(window: gui.MainWindow) -> None:
    """Tooltips are hard-wrapped, since Qt draws a long plain-text one as a single wide line."""
    tip = window._save_options_menu.backup.toolTip()  # noqa: SLF001
    assert "\n" in tip
    assert max(len(line) for line in tip.splitlines()) <= TOOLTIP_WIDTH


def test_save_tooltips_follow_the_chosen_options(window: gui.MainWindow) -> None:
    """Toggling a save option rewrites the Save buttons' current-options summary."""
    assert "title, description, keywords" in _unwrapped(window._run_bar.save_button.toolTip())  # noqa: SLF001
    assert "XMP sidecar" in _unwrapped(window._detail.save_button.toolTip())  # noqa: SLF001

    window._save_options_menu.write_description.setChecked(False)  # noqa: SLF001
    window._save_options_menu.sidecar_actions["none"].setChecked(True)  # noqa: SLF001

    tip = _unwrapped(window._run_bar.save_button.toolTip())  # noqa: SLF001
    assert "title, keywords" in tip
    assert "into the image file" in tip
    assert "keeping a *_original backup" in tip

    window._save_options_menu.sidecar_actions["raw"].setChecked(True)  # noqa: SLF001
    assert "to a sidecar for RAW" in _unwrapped(window._run_bar.save_button.toolTip())  # noqa: SLF001

    window._save_options_menu.sidecar_actions["both"].setChecked(True)  # noqa: SLF001
    tip = _unwrapped(window._run_bar.save_button.toolTip())  # noqa: SLF001
    assert "into the image file and an XMP sidecar" in tip

    window._save_options_menu.backup.setChecked(False)  # noqa: SLF001
    assert "with no *_original backup" in _unwrapped(window._run_bar.save_button.toolTip())  # noqa: SLF001


def test_grid_checkbox_unchecks_photo_and_tree(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking a thumbnail deselects the photo and repaints its tree row."""
    a = _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    grid_item = window._folder_grid._cells[str(a)]  # noqa: SLF001
    assert grid_item.checkState() == Qt.CheckState.Checked
    grid_item.setCheckState(Qt.CheckState.Unchecked)  # fires itemChanged

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert _check_state(window, a) == Qt.CheckState.Unchecked
    # The parent folder is now partially checked.
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    assert top.checkState(0) == Qt.CheckState.PartiallyChecked


def test_tree_uncheck_reflects_in_grid_checkboxes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking the folder in the tree unchecks every visible thumbnail."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    top.setCheckState(0, Qt.CheckState.Unchecked)

    states = {item.checkState() for item in window._folder_grid._cells.values()}  # noqa: SLF001
    assert states == {Qt.CheckState.Unchecked}


def test_grid_checkbox_click_does_not_open_the_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toggling a thumbnail checkbox swallows the click; a later plain click still opens."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": img})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    grid_item = window._folder_grid._cells[str(img)]  # noqa: SLF001

    grid_item.setCheckState(Qt.CheckState.Unchecked)  # the toggle half of the click
    window._folder_grid._on_thumb_activated(grid_item)  # noqa: SLF001 - the click half

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001 - stayed in the grid

    window._folder_grid._on_thumb_activated(grid_item)  # noqa: SLF001 - a later plain click
    assert window._right.currentIndex() == main_window._PAGE_DETAIL  # noqa: SLF001


def test_modifier_click_on_thumbnail_keeps_the_grid(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shift/cmd click is building a selection, so it must not navigate away."""
    img = _jpeg(tmp_path / "a.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _patch_gui(monkeypatch, "selection_modifiers_active", lambda: True)
    _add_dir(window, {"a": img})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    window._folder_grid._on_thumb_activated(window._folder_grid._cells[str(img)])  # noqa: SLF001

    assert window._right.currentIndex() == main_window._PAGE_GRID  # noqa: SLF001


def test_bulk_context_menu_checks_unchecks_and_removes(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Right-clicking inside a multi-selection offers bulk check/uncheck/remove."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    for path in (a, b):
        leaf = window._tree.leaf_for(path)  # noqa: SLF001
        assert leaf is not None
        leaf.setSelected(True)

    leaf_a = window._tree.leaf_for(a)  # noqa: SLF001
    menu = window._build_tree_context_menu(leaf_a)  # noqa: SLF001
    assert menu is not None
    actions = {action.text(): action for action in menu.actions() if action.text()}
    assert "Uncheck 2 Photos" in actions
    assert "Check 2 Photos" in actions

    actions["Uncheck 2 Photos"].trigger()
    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is False  # noqa: SLF001
    assert window._items[str(c)].selected is True  # noqa: SLF001 - not in the selection
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    assert top.checkState(0) == Qt.CheckState.PartiallyChecked

    actions["Remove From List"].trigger()
    assert str(a) not in window._items  # noqa: SLF001
    assert str(b) not in window._items  # noqa: SLF001
    assert str(c) in window._items  # noqa: SLF001


def test_bulk_menu_check_only_and_skip_cache(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check Only keeps just the selection checked; the bulk generate can skip the cache."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    for path in (a, b):
        leaf = window._tree.leaf_for(path)  # noqa: SLF001
        assert leaf is not None
        leaf.setSelected(True)

    menu = window._build_tree_context_menu(window._tree.leaf_for(a))  # noqa: SLF001
    assert menu is not None
    actions = {action.text(): action for action in menu.actions() if action.text()}

    actions["Check Only 2 Photos"].trigger()
    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    assert window._items[str(c)].selected is False  # noqa: SLF001 - not in the selection

    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **kwargs: calls.append((len(items), kwargs.get("use_cache", True))),
    )
    actions["Generate 2 Photos (Skip Cache)"].trigger()
    assert calls == [(2, False)]


def test_select_menu_inverts_the_checkboxes(window: gui.MainWindow, tmp_path: Path) -> None:
    """Invert Checked flips every checkbox, model and rendered tree alike."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].selected = False  # noqa: SLF001
    texts = [action.text() for action in window._select_menu.actions()]  # noqa: SLF001
    assert "Invert Checked" in texts

    window._invert_checked()  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert window._items[str(b)].selected is False  # noqa: SLF001
    assert _check_state(window, a) == Qt.CheckState.Checked
    assert _check_state(window, b) == Qt.CheckState.Unchecked


def test_single_row_right_click_keeps_the_per_photo_menu(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """With one row selected, the menu is the per-photo one (Generate, reveal, ...)."""
    a = _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a})
    leaf = window._tree.leaf_for(a)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001 - selects just this row

    menu = window._build_tree_context_menu(leaf)  # noqa: SLF001
    assert menu is not None
    texts = [action.text() for action in menu.actions() if action.text()]
    assert texts[0] == "Generate"


def test_remove_selected_drops_every_selected_row(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Delete acts on the whole selection, not only the current row."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    for path in (a, c):
        leaf = window._tree.leaf_for(path)  # noqa: SLF001
        assert leaf is not None
        leaf.setSelected(True)

    window._remove_selected()  # noqa: SLF001

    assert sorted(window._items) == [str(b)]  # noqa: SLF001


def test_generating_status_pluralizes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One photo reads "1 photo", several read "N photos" (no "(s)" anywhere)."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": a, "b": b})

    window._run_generation([window._items[str(a)]])  # noqa: SLF001
    assert window._status.text() == "Generating 1 photo..."  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001

    window._run_generation(  # noqa: SLF001
        [window._items[str(a)], window._items[str(b)]],  # noqa: SLF001
    )
    assert window._status.text() == "Generating 2 photos..."  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Tree columns, colors, and context menu
# ---------------------------------------------------------------------------


def test_type_column_shows_extension_and_sidecar(window: gui.MainWindow, tmp_path: Path) -> None:
    """The Type column shows the extension, plus +xmp when a sidecar sits next to the file."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    (tmp_path / "a.xmp").write_text("<x/>", encoding="utf-8")
    _add_dir(window, {"a": a, "b": b})
    leaf_a = window._tree.leaf_for(a)  # noqa: SLF001
    leaf_b = window._tree.leaf_for(b)  # noqa: SLF001
    assert leaf_a is not None
    assert leaf_b is not None
    assert leaf_a.text(1) == "jpg+xmp"
    assert leaf_b.text(1) == "jpg"


def test_scan_results_fill_the_tagged_column(window: gui.MainWindow, tmp_path: Path) -> None:
    """A finished metadata scan paints the Tagged column and records the fields."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    leaf = window._tree.leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.text(3) == ""  # unknown until the scan reports

    window._on_scan_done({str(img): {FIELD_TITLE, FIELD_KEYWORDS}})  # noqa: SLF001

    assert leaf.text(3) == "TK"
    assert window._items[str(img)].known_fields == {FIELD_TITLE, FIELD_KEYWORDS}  # noqa: SLF001


def test_metadata_scan_worker_emits_each_batch_with_progress(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The worker reports a batch at a time, keyed by path string, and ticks progress for each.

    Emitting per batch is what fills the Tagged column as the read runs, instead of leaving the list
    blank until a whole library has been read.
    """
    paths = [Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")]
    _patch_gui(
        monkeypatch,
        "iter_field_presence",
        # The middle batch stands in for one exiftool could not read at all.
        lambda scanned: (
            metadata.PresenceBatch(1, {} if path.name == "b.jpg" else {path: {FIELD_TITLE}})
            for path in scanned
        ),
    )
    results: list[dict] = []
    ticks: list[tuple[int, int]] = []
    worker = workers.MetadataScanWorker(paths)
    worker.done.connect(results.append)
    worker.progress.connect(lambda done, total: ticks.append((done, total)))

    worker.run()

    assert results == [{"/a.jpg": {FIELD_TITLE}}, {"/c.jpg": {FIELD_TITLE}}]
    assert ticks == [(0, 3), (1, 3), (2, 3), (3, 3)]  # the unread batch still advances the bar


def test_metadata_scan_worker_stops_after_the_batch_in_flight(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping ends the read at the next batch boundary, leaving the rest of the list unread."""
    worker = workers.MetadataScanWorker([Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")])

    def batches(scanned: list[Path]) -> Iterator[metadata.PresenceBatch]:
        for path in scanned:
            worker.stop()  # the UI thread pressing Stop while the read is under way
            yield metadata.PresenceBatch(1, {path: set()})

    _patch_gui(monkeypatch, "iter_field_presence", batches)
    results: list[dict] = []
    worker.done.connect(results.append)

    worker.run()

    assert results == [{"/a.jpg": set()}]


def test_metadata_scan_worker_skips_photos_dropped_while_it_reads(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A folder removed mid-scan stops being read, and leaves the progress total behind it.

    Regression test: the scan used to work from the list it was handed, so removing a folder of
    19000 photos out of 19500 still read all 19500 of them, and the bar still counted to 19500.
    """
    keep = [Path("/keep/0.jpg"), Path("/keep/1.jpg")]
    gone = [Path(f"/gone/{index}.jpg") for index in range(3)]
    worker = workers.MetadataScanWorker([keep[0], *gone, keep[1]])

    def batches(scanned: Iterator[Path]) -> Iterator[metadata.PresenceBatch]:
        for path in scanned:
            if path == keep[0]:
                worker.drop(gone)  # the UI thread removing that folder from the list
            yield metadata.PresenceBatch(1, {path: set()})

    _patch_gui(monkeypatch, "iter_field_presence", batches)
    read: list[str] = []
    ticks: list[tuple[int, int]] = []
    worker.done.connect(read.extend)
    worker.progress.connect(lambda done, total: ticks.append((done, total)))

    worker.run()

    assert read == ["/keep/0.jpg", "/keep/1.jpg"]
    assert ticks == [(0, 5), (1, 2), (2, 2)]  # the removed folder leaves the total at once


def test_remove_items_tells_a_running_scan_to_skip_them(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Removing photos hands them to the scan in flight, so it stops reading them."""
    a, b = _jpeg(tmp_path / "a.jpg"), _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    worker = workers.MetadataScanWorker([a, b])
    window._scan_worker = worker  # noqa: SLF001

    window._remove_items([str(a)])  # noqa: SLF001

    assert list(worker._live_paths()) == [b]  # noqa: SLF001


class _FakeScanThread:
    """
    Stands in for a QThread without starting one, so tests stay fast and deterministic.

    Exposes only what MainWindow._stop_scan touches: quit(), wait(ms), deleteLater(), and a finished
    signal with connect().
    """

    def __init__(self, *, finishes_in_time: bool) -> None:
        self._finishes_in_time = finishes_in_time
        self.quit_called = False
        self.deleted = False
        self.finished_slots: list[object] = []
        self.finished = SimpleNamespace(connect=self.finished_slots.append)

    def quit(self) -> None:
        self.quit_called = True

    def wait(self, _timeout_ms: int) -> bool:
        return self._finishes_in_time

    def deleteLater(self) -> None:  # noqa: N802 - Qt naming convention
        self.deleted = True


def test_stop_scan_deletes_immediately_when_it_finishes_in_time(
    window: gui.MainWindow,
) -> None:
    """The common case (the scan finishes within the grace period) cleans up right away."""
    fake_thread = _FakeScanThread(finishes_in_time=True)
    window._scan_thread = fake_thread  # noqa: SLF001
    window._scan_worker = workers.MetadataScanWorker([])  # noqa: SLF001

    window._stop_scan()  # noqa: SLF001

    assert window._scan_thread is None  # noqa: SLF001
    assert window._scan_worker is None  # noqa: SLF001
    assert fake_thread.quit_called
    assert fake_thread.deleted
    assert not fake_thread.finished_slots


def test_stop_scan_detaches_instead_of_blocking_when_still_running(
    window: gui.MainWindow,
) -> None:
    """
    _stop_scan must not block the caller forever waiting on a scan that will not finish soon.

    Regression test: a single batched exiftool call cannot be cancelled mid-flight, so a large
    folder (or a hung exiftool) used to freeze closeEvent/_clear until the whole scan finished.
    """
    fake_thread = _FakeScanThread(finishes_in_time=False)
    window._scan_thread = fake_thread  # noqa: SLF001
    window._scan_worker = workers.MetadataScanWorker([])  # noqa: SLF001

    window._stop_scan()  # noqa: SLF001 - must return promptly, not block for _SCAN_STOP_TIMEOUT_MS

    assert window._scan_thread is None  # noqa: SLF001 - detached, not waited on further
    assert window._scan_worker is None  # noqa: SLF001
    assert fake_thread.quit_called
    assert not fake_thread.deleted  # never force-deleted while still running
    assert fake_thread.finished_slots  # cleanup deferred to whenever it actually finishes


def test_a_detached_scan_finishing_leaves_a_newer_scan_alone(
    window: gui.MainWindow,
) -> None:
    """
    A scan the timeout detached must not take the next scan down with it.

    It keeps running after _stop_scan lets go, and _on_scan_finished tears down whatever scan is
    current when it fires, so a late finish used to quit and discard an unrelated live one.
    """
    stale = workers.MetadataScanWorker([])
    stale.done.connect(window._on_scan_done)  # noqa: SLF001
    stale.finished.connect(window._on_scan_finished)  # noqa: SLF001
    window._scan_thread = _FakeScanThread(finishes_in_time=False)  # noqa: SLF001
    window._scan_worker = stale  # noqa: SLF001
    window._stop_scan()  # noqa: SLF001 - detaches; `stale` runs on

    live_thread = _FakeScanThread(finishes_in_time=True)
    window._scan_thread = live_thread  # noqa: SLF001
    window._scan_worker = workers.MetadataScanWorker([])  # noqa: SLF001

    stale.finished.emit()

    assert window._scan_thread is live_thread  # noqa: SLF001
    assert not live_thread.quit_called


def test_on_scan_finished_does_not_restart_a_scan_once_closing(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan that finishes after closeEvent already ran must not kick off a new one."""
    started: list[object] = []
    monkeypatch.setattr(window, "_start_metadata_scan", lambda: started.append(1))
    window._closing = True  # noqa: SLF001

    window._on_scan_finished()  # noqa: SLF001

    assert not started


def test_failed_status_is_painted_red(window: gui.MainWindow, tmp_path: Path) -> None:
    """A failed photo's Status cell turns red so it stands out in a long list."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    leaf = window._tree.leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    brush = leaf.data(2, Qt.ItemDataRole.ForegroundRole)
    assert brush is not None
    assert brush.color().name() == "#f85149"


def test_context_menu_on_failed_photo_offers_retry(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Right-clicking a failed photo leads with Retry, plus reveal and remove actions."""
    import sys as sys_module  # noqa: PLC0415 - platform-dependent expected label.

    from photo_tagger.gui_state import reveal_label  # noqa: PLC0415

    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    window._items[str(img)].status = FAILED  # noqa: SLF001

    menu = window._build_tree_context_menu(window._tree.leaf_for(img))  # noqa: SLF001
    assert menu is not None
    texts = [action.text() for action in menu.actions() if action.text()]

    assert texts[0] == "Retry Generation"
    assert "Generate (Skip Cache)" in texts
    assert reveal_label(sys_module.platform) in texts
    assert "Remove From List" in texts


def test_context_menu_on_folder_skips_generate_actions(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A folder's context menu has no per-photo generate entries."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    menu = window._build_tree_context_menu(window._tree.topLevelItem(0))  # noqa: SLF001
    assert menu is not None
    texts = [action.text() for action in menu.actions() if action.text()]
    assert all("Generate" not in text and "Retry" not in text for text in texts)
    assert "Remove From List" in texts


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------


def test_worker_reuses_cached_results(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run over the same photo hits the cache instead of calling the model."""
    _stub_generation(monkeypatch)
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"

    first = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    done_first: list[Proposal] = []
    first.file_done.connect(done_first.append)
    first.run()
    assert len(done_first) == 1
    assert cache_file.exists()

    def boom(**_k: object) -> object:
        msg = "the model was called despite a cache hit"
        raise AssertionError(msg)

    _patch_gui(monkeypatch, "analyze_image_with_ai", boom)
    second = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    done_second: list[Proposal] = []
    second.file_done.connect(done_second.append)
    second.run()

    assert len(done_second) == 1
    assert done_second[0].title == "T"


def test_worker_puts_the_hint_in_that_photos_prompt(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hinted photo's prompt carries the photographer's note; other photos' prompts do not."""
    _stub_generation(monkeypatch)
    prompts: list[str] = []

    def record(**kwargs: object) -> InferenceResult:
        prompts.append(str(kwargs["user_prompt"]))
        return InferenceResult(title="T", description="D", keywords=[])

    _patch_gui(monkeypatch, "analyze_image_with_ai", record)
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg"), Path("/b.jpg")],
        options=GenerateOptions(hints={str(Path("/a.jpg")): "The animal is a deer"}),
    )
    worker.file_done.connect(lambda _p: None)
    worker.run()

    assert "Photographer's note about this photo: The animal is a deer" in prompts[0]
    assert "Photographer's note" not in prompts[1]


def test_hinted_photo_skips_the_cache_lookup(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hint forces a fresh model call instead of replaying whatever is already cached."""
    _stub_generation(monkeypatch)
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"

    first = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    first.file_done.connect(lambda _p: None)
    first.run()  # caches the (wrong) title "T"

    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="Deer", description="D", keywords=["Deer"]),
    )
    second = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [img],
        cache_file=cache_file,
        options=GenerateOptions(hints={str(img): "The animal is a deer"}),
    )
    corrected: list[Proposal] = []
    second.file_done.connect(corrected.append)
    second.run()

    assert corrected[0].title == "Deer"  # the stale "T" was not replayed
    assert corrected[0].from_cache is False


def test_a_refresh_run_replaces_the_answer_it_read_past(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The mixed-script pass asks again because the cached answer is the unusable one.

    Reading past it is half the job: what comes back is a plain generic answer for this photo, so it
    has to land under the same key. Otherwise the next run of the folder replays the mixed text and
    re-flags a photo the window already fixed and wrote.
    """
    _stub_generation(monkeypatch)
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"
    first = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    first.file_done.connect(lambda _p: None)
    first.run()  # caches the mixed-script title "T"

    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="Ocarina", description="D", keywords=["Ocarina"]),
    )
    again = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [img],
        cache_file=cache_file,
        options=GenerateOptions(refresh=True),
    )
    fresh: list[Proposal] = []
    again.file_done.connect(fresh.append)
    again.run()
    assert fresh[0].title == "Ocarina"
    assert fresh[0].from_cache is False

    _patch_gui(monkeypatch, "analyze_image_with_ai", lambda **_k: pytest.fail("no call expected"))
    later = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    replayed: list[Proposal] = []
    later.file_done.connect(replayed.append)
    later.run()

    assert replayed[0].title == "Ocarina"  # the refreshed answer, not the one it replaced
    assert replayed[0].from_cache is True


def test_a_cached_mixed_answer_is_repaired_when_the_setting_is_on(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning the repair on fixes a mixed answer cached before, and the fix replaces it."""
    _stub_generation(monkeypatch)
    _patch_gui(monkeypatch, "analyze_image_with_ai", lambda **_k: _GUI_MIXED)
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"
    first = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    first.file_done.connect(lambda _p: None)
    first.run()  # caches the mixed answer, with the repair off

    _patch_gui(monkeypatch, "analyze_image_with_ai", lambda **_k: _GUI_REPAIRED)
    options = GenerateOptions(repair_language=True)
    again = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [img],
        cache_file=cache_file,
        options=options,
    )
    fixed: list[Proposal] = []
    again.file_done.connect(fixed.append)
    again.run()
    assert fixed[0].title == _GUI_REPAIRED.title
    assert fixed[0].from_cache is False

    _patch_gui(monkeypatch, "analyze_image_with_ai", lambda **_k: pytest.fail("no call expected"))
    later = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [img],
        cache_file=cache_file,
        options=options,
    )
    replayed: list[Proposal] = []
    later.file_done.connect(replayed.append)
    later.run()

    assert replayed[0].title == _GUI_REPAIRED.title
    assert replayed[0].from_cache is True


def test_hinted_photo_does_not_poison_the_shared_cache(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A hint is a one-off correction; it must never become the cached "generic" answer.

    Regression test: a hinted result used to be stored under the same content-only cache key a
    hint-less run reads from, so a later hint-less regeneration of the same photo silently replayed
    someone's one-time correction as if it were the model's generic read.
    """
    _stub_generation(monkeypatch)
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"

    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="Deer", description="D", keywords=["Deer"]),
    )
    first = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [img],
        cache_file=cache_file,
        options=GenerateOptions(hints={str(img): "The animal is a deer"}),
    )
    corrected: list[Proposal] = []
    first.file_done.connect(corrected.append)
    first.run()
    assert corrected[0].title == "Deer"

    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="Generic Animal", description="D", keywords=[]),
    )
    second = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    replayed: list[Proposal] = []
    second.file_done.connect(replayed.append)
    second.run()

    # The hinted "Deer" answer was never cached, so the hint-less run calls the model again
    # instead of silently replaying someone's one-off correction.
    assert replayed[0].title == "Generic Animal"
    assert replayed[0].from_cache is False


def test_worker_cache_survives_metadata_rewrites(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding metadata rewrites the file, but the content-hash key still hits the cache."""
    _stub_generation(monkeypatch)
    _patch_gui(
        monkeypatch,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(
            existing_keywords=KeywordSet(),
            content_hash="pixels-unchanged",
        ),
    )
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"

    first = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    first.file_done.connect(lambda _p: None)
    first.run()

    # Simulate "Embed in Photo": the file bytes change, the image data does not.
    img.write_bytes(img.read_bytes() + b"embedded-metadata")

    def boom(**_k: object) -> object:
        msg = "the model was called although the pixels did not change"
        raise AssertionError(msg)

    _patch_gui(monkeypatch, "analyze_image_with_ai", boom)
    second = workers.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    done: list[Proposal] = []
    second.file_done.connect(done.append)
    second.run()

    assert len(done) == 1
    assert done[0].from_cache is True


def test_cache_toggle_controls_the_active_cache_file(window: gui.MainWindow) -> None:
    """Caching defaults on; unchecking the Settings toggle disables it for new runs."""
    assert window._settings_menu._cache_action.isChecked()  # noqa: SLF001
    assert window._settings_menu.active_cache_file() is not None  # noqa: SLF001
    window._settings_menu._cache_action.setChecked(False)  # noqa: SLF001
    assert window._settings_menu.active_cache_file() is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Saving settings as the config file
# ---------------------------------------------------------------------------


def test_save_config_writes_the_gui_choices(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save Settings as Defaults writes a TOML file load_defaults understands, sans API key."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    target = tmp_path / "config.toml"
    # No config in effect, so the save creates the user config from the template.
    _patch_gui(monkeypatch, "find_config_file", lambda: None)
    _patch_gui(monkeypatch, "user_config_path", lambda: target)
    window._connection_bar._model.setCurrentText("qwen/qwen3-vl-30b")  # noqa: SLF001
    window._connection_bar._api_key.setText("sk-secret")  # noqa: SLF001 - must NOT be written
    window._extensions.setText("jpg,cr3")  # noqa: SLF001
    window._save_options_menu.sidecar_actions["raw"].setChecked(True)  # noqa: SLF001
    window._save_options_menu.backup.setChecked(False)  # noqa: SLF001

    window._save_config()  # noqa: SLF001

    text = target.read_text(encoding="utf-8")
    assert "sk-secret" not in text
    data = tomllib.loads(text)
    assert data["provider"]["model_name"] == "qwen/qwen3-vl-30b"
    assert data["extensions"] == "jpg,cr3"
    assert data["output"]["sidecar_mode"] == "raw"
    assert data["output"]["backup_xmp"] is False
    assert data["telemetry"]["enabled"] is True
    assert data["display"]["notify"] is False


def test_save_config_refuses_a_copyright_template_with_an_error(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A template the CLI would refuse is never saved, or every later CLI run would stop."""
    target = tmp_path / "config.toml"
    _patch_gui(monkeypatch, "find_config_file", lambda: None)
    _patch_gui(monkeypatch, "user_config_path", lambda: target)
    warned: list[str] = []
    _patch_gui(
        monkeypatch,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args, **_k: warned.append(str(args[-1]))),
    )
    _fill_copyright(window, notice="Copyright (c) {shot_at}")
    assert window._copyright_dialog.fields is None  # noqa: SLF001 - the dialog shows the error

    window._save_config()  # noqa: SLF001

    assert warned
    assert not target.exists()


def test_save_config_merges_into_the_config_in_effect(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving over an existing config updates GUI keys but keeps comments and other tables."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    target = tmp_path / "config.toml"
    target.write_text(
        '# hands off\nextensions = "cr3"\n\n[filter]\nskip_tagged = true\n',
        encoding="utf-8",
    )
    _patch_gui(monkeypatch, "find_config_file", lambda: target)
    window._extensions.setText("jpg")  # noqa: SLF001

    window._save_config()  # noqa: SLF001

    text = target.read_text(encoding="utf-8")
    assert "# hands off" in text
    data = tomllib.loads(text)
    assert data["extensions"] == "jpg"
    assert data["filter"]["skip_tagged"] is True  # untouched table survives
    assert "preserved" in window._status.text()  # noqa: SLF001


def test_cached_proposal_shows_in_the_status_column(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A proposal replayed from the cache labels its row 'ready (cached)'."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=img,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="T",
            description="D",
            keywords=[],
            from_cache=True,
        ),
    )
    leaf = window._tree.leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.text(2) == "ready (cached)"


def test_description_boxes_grow_only_with_content(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short description keeps its box short; a long one grows it up to the cap."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    short = window._detail._description.minimumHeight()  # noqa: SLF001 - setFixedHeight sets min=max
    window._detail._description.setPlainText("line\n" * 40)  # noqa: SLF001

    grown = window._detail._description.minimumHeight()  # noqa: SLF001
    assert grown > short
    assert grown <= 140  # noqa: PLR2004 - the documented cap


# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------


def _keyword_file(tmp_path: Path, text: str = "Animal|Bird|Osprey\nSunset\n") -> Path:
    """Write a small vocabulary file, the shape a Lightroom export or a build produces."""
    path = tmp_path / "keywords.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_worker_snaps_keywords_onto_the_vocabulary(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated keyword comes back spelled and filed the way the catalog has it."""
    _stub_generation(monkeypatch)
    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="T", description="D", keywords=["ospreys", "Tractor"]),
    )
    vocabulary = Vocabulary.from_entries(["Animal|Bird|Osprey"])
    proposals: list[Proposal] = []
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(vocabulary=vocabulary),
    )
    worker.file_done.connect(proposals.append)

    worker.run()

    assert proposals[0].keywords == ["Osprey<Bird<Animal", "Tractor"]
    assert proposals[0].vocabulary_mapped == 1
    assert proposals[0].vocabulary_dropped == []


def test_worker_strict_vocabulary_drops_what_the_catalog_lacks(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode reports the rejects, so the vocabulary can grow on purpose."""
    _stub_generation(monkeypatch)
    _patch_gui(
        monkeypatch,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="T", description="D", keywords=["Osprey", "Tractor"]),
    )
    proposals: list[Proposal] = []
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(
            vocabulary=Vocabulary.from_entries(["Osprey"]),
            vocabulary_strict=True,
        ),
    )
    worker.file_done.connect(proposals.append)

    worker.run()

    assert proposals[0].keywords == ["Osprey"]
    assert proposals[0].vocabulary_dropped == ["Tractor"]


def test_worker_lists_the_vocabulary_in_the_prompt(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model is told the catalog's terms, so it prefers them before any snapping."""
    _stub_generation(monkeypatch)
    prompts: list[str] = []

    def capture(**kwargs: object) -> InferenceResult:
        prompts.append(str(kwargs["user_prompt"]))
        return InferenceResult(title="T", description="D", keywords=[])

    _patch_gui(monkeypatch, "analyze_image_with_ai", capture)
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(vocabulary=Vocabulary.from_entries(["Animal|Bird|Osprey"])),
    )

    worker.run()

    assert "Controlled Vocabulary" in prompts[0]
    assert "Animal > Bird > Osprey" in prompts[0]


def test_vocabulary_is_part_of_the_cache_namespace(qapp: QApplication) -> None:
    """Swapping vocabularies starts a fresh cache slice instead of replaying the old keywords."""
    plain = workers.GenerateWorker("lmstudio", "m", None, [])
    with_vocabulary = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [],
        options=GenerateOptions(vocabulary=Vocabulary.from_entries(["Osprey"])),
    )
    namespace = gui_cache_namespace
    assert namespace("m", "English", plain._prompt) != namespace(  # noqa: SLF001
        "m",
        "English",
        with_vocabulary._prompt,  # noqa: SLF001
    )


def test_run_generation_hands_the_vocabulary_to_the_worker(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the Keyword rules dialog set is what the run uses."""
    photo = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": photo})
    window._keyword_rules_dialog.load_vocabulary(_keyword_file(tmp_path))  # noqa: SLF001
    window._keyword_rules_dialog._strict_box.setChecked(True)  # noqa: SLF001
    monkeypatch.setattr(QThread, "start", lambda *_a, **_k: None)

    window._run_generation([window._items[str(photo)]])  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._vocabulary is not None  # noqa: SLF001
    assert worker._vocabulary.match("ospreys") == "Osprey"  # noqa: SLF001
    assert worker._vocabulary_strict is True  # noqa: SLF001


def test_loading_an_unusable_vocabulary_says_why(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A file with no keywords in it is refused, and the window says so rather than writing."""
    empty = tmp_path / "empty.txt"
    empty.write_text("# just a comment\n", encoding="utf-8")

    window._keyword_rules_dialog.load_vocabulary(empty)  # noqa: SLF001

    assert window._keyword_rules_dialog.vocabulary is None  # noqa: SLF001
    assert "no keywords" in window._status.text()  # noqa: SLF001
    assert "no keywords" in window._keyword_rules_dialog._vocabulary_label.text()  # noqa: SLF001


def test_clearing_the_vocabulary_writes_keywords_as_they_come(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Clearing is how you get the model's own wording back."""
    window._keyword_rules_dialog.load_vocabulary(_keyword_file(tmp_path))  # noqa: SLF001
    assert window._keyword_rules_dialog.vocabulary is not None  # noqa: SLF001

    window._keyword_rules_dialog.load_vocabulary(None)  # noqa: SLF001

    assert window._keyword_rules_dialog.vocabulary_path is None  # noqa: SLF001
    assert window._keyword_rules_dialog._vocabulary_field.text() == ""  # noqa: SLF001
    assert "No vocabulary" in window._keyword_rules_dialog._vocabulary_label.text()  # noqa: SLF001


def test_keyword_rules_build_button_opens_the_vocabulary_builder(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vocabulary can be built from where it is chosen, without a trip through the Tools menu."""
    opened: list[bool] = []
    monkeypatch.setattr(window._builder_dialog, "exec", lambda: opened.append(True))  # noqa: SLF001
    buttons = window._keyword_rules_dialog.findChildren(QPushButton)  # noqa: SLF001

    next(button for button in buttons if button.text() == "Build...").click()

    assert opened == [True]


def test_generation_summary_reports_what_the_vocabulary_changed(
    window: gui.MainWindow,
) -> None:
    """The closing status line of a run names the rejected terms, not just a count."""
    window._vocabulary_mapped = 2  # noqa: SLF001
    window._vocabulary_dropped = {"Tractor": 3}  # noqa: SLF001

    summary = window._generation_summary()  # noqa: SLF001

    assert summary.startswith("Generation finished.")
    assert "2 keywords rewritten" in summary
    assert "Tractor" in summary


def test_keyword_rules_persist_to_the_config_file(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Save Settings as Defaults carries the rules over to CLI runs too."""
    listing = _keyword_file(tmp_path)
    window._keyword_rules_dialog.load_vocabulary(listing)  # noqa: SLF001
    window._keyword_rules_dialog._strict_box.setChecked(True)  # noqa: SLF001
    window._keyword_rules_dialog._session_gap_box.setValue(45.0)  # noqa: SLF001

    values = window._current_config_values()  # noqa: SLF001

    assert values.vocabulary == listing
    assert values.vocabulary_strict is True
    assert values.session_gap_minutes == 45.0  # noqa: PLR2004 - the value just set


# ---------------------------------------------------------------------------
# Shoot harmonization
# ---------------------------------------------------------------------------


def _drain_harmonize(window: gui.MainWindow, timeout: float = 10.0) -> None:
    """Pump the event loop until the background harmonization finishes."""
    deadline = time.monotonic() + timeout
    while window._harmonize_thread is not None:  # noqa: SLF001
        assert time.monotonic() < deadline, "the harmonization never finished"
        QApplication.processEvents()


def test_harmonizing_makes_a_shoot_agree_with_itself(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two frames of one bird stop landing in the catalog as two keywords."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    window._items[str(a)].keywords = ["Osprey"]  # noqa: SLF001
    window._items[str(b)].keywords = ["Ospreys"]  # noqa: SLF001
    monkeypatch.setattr("photo_tagger.sessions.read_capture_times", lambda *_a, **_k: {})
    window._keyword_rules_dialog._session_gap_box.setValue(60.0)  # noqa: SLF001

    assert window._start_harmonize() is True  # noqa: SLF001
    _drain_harmonize(window)

    assert window._items[str(b)].keywords == ["Osprey"]  # noqa: SLF001
    assert "harmonized" in window._status.text()  # noqa: SLF001


def test_harmonize_now_needs_a_session_gap(window: gui.MainWindow) -> None:
    """Without a gap there are no shoots to harmonize, so the window says what to set."""
    window._keyword_rules_dialog._session_gap_box.setValue(0.0)  # noqa: SLF001

    window._harmonize_now()  # noqa: SLF001

    assert window._harmonize_thread is None  # noqa: SLF001
    assert "session gap" in window._status.text()  # noqa: SLF001


def test_harmonizing_with_nothing_generated_says_so(window: gui.MainWindow) -> None:
    """Harmonization works on proposals, so an empty list is a nudge rather than a no-op."""
    window._keyword_rules_dialog._session_gap_box.setValue(60.0)  # noqa: SLF001

    assert window._start_harmonize() is False  # noqa: SLF001
    assert "generate some photos first" in window._status.text()  # noqa: SLF001


def test_a_finished_run_harmonizes_when_a_gap_is_set(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI harmonizes inside the run; the window does it to the proposals, before review."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    window._items[str(a)].keywords = ["Sunset"]  # noqa: SLF001
    window._items[str(b)].keywords = ["Sunsets"]  # noqa: SLF001
    monkeypatch.setattr("photo_tagger.sessions.read_capture_times", lambda *_a, **_k: {})
    window._keyword_rules_dialog._session_gap_box.setValue(60.0)  # noqa: SLF001

    window._after_generation()  # noqa: SLF001
    _drain_harmonize(window)

    assert window._items[str(b)].keywords == ["Sunset"]  # noqa: SLF001


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def _save_two_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Save two photos for real enough that a sidecar lands on disk and is recorded."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)

    def fake_write(path: Path, *_a: object, **_k: object) -> bool:
        path.with_suffix(".xmp").write_text("<xmp/>", encoding="utf-8")
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    window._save_selected()  # noqa: SLF001
    _drain_save(window)
    return a, b


def test_a_save_records_what_it_wrote(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window's own writes go into a journal, so its undo covers more than CLI runs."""
    a, b = _save_two_photos(window, tmp_path, monkeypatch)

    journals = list_journals()
    assert len(journals) == 1
    records = read_journal(journals[0])
    assert {Path(record.target).name for record in records} == {"a.xmp", "b.xmp"}
    assert {Path(record.image) for record in records} == {a, b}
    assert all(record.created for record in records)  # neither sidecar existed before


def test_switching_off_the_undo_log_records_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording is a setting, and turning it off leaves no journal behind."""
    window._settings_menu._undo_log_action.setChecked(False)  # noqa: SLF001

    _save_two_photos(window, tmp_path, monkeypatch)

    assert list_journals() == []


def test_undo_puts_back_what_a_save_wrote(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undoing a run deletes the sidecars it created and frees its photos to be saved again."""
    a, b = _save_two_photos(window, tmp_path, monkeypatch)
    window._undo_dialog.refresh()  # noqa: SLF001

    window._undo_dialog._run(dry_run=False)  # noqa: SLF001

    assert not a.with_suffix(".xmp").exists()
    assert not b.with_suffix(".xmp").exists()
    assert "Put back 2 files." in window._status.text()  # noqa: SLF001
    assert window._items[str(a)].status == READY  # noqa: SLF001 - the proposal is still there
    assert window._items[str(a)].known_fields is None  # noqa: SLF001 - and will be re-scanned


def test_undo_preview_touches_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview is how you check you picked the right run before anything is rewritten."""
    a, _b = _save_two_photos(window, tmp_path, monkeypatch)
    window._undo_dialog.refresh()  # noqa: SLF001

    window._undo_dialog._run(dry_run=True)  # noqa: SLF001

    assert a.with_suffix(".xmp").exists()
    assert window._items[str(a)].status == SAVED  # noqa: SLF001
    assert "Preview:" in window._status.text()  # noqa: SLF001
    assert "What undoing would do:" in window._undo_dialog._details.toPlainText()  # noqa: SLF001


def test_undo_list_shows_every_recorded_run(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row per run, newest first, saying when it ran and how much it wrote."""
    _save_two_photos(window, tmp_path, monkeypatch)

    window._undo_dialog.refresh()  # noqa: SLF001

    assert window._undo_dialog._journals.count() == 1  # noqa: SLF001
    assert "2 files" in window._undo_dialog._journals.item(0).text()  # noqa: SLF001
    assert window._undo_dialog._undo_button.isEnabled()  # noqa: SLF001


def test_undo_list_is_empty_before_anything_is_saved(window: gui.MainWindow) -> None:
    """A fresh machine has nothing to undo, and the button says so by staying off."""
    window._undo_dialog.refresh()  # noqa: SLF001

    assert window._undo_dialog._journals.count() == 0  # noqa: SLF001
    assert not window._undo_dialog._undo_button.isEnabled()  # noqa: SLF001
    assert "No recorded runs" in window._undo_dialog._details.toPlainText()  # noqa: SLF001


def test_undo_without_a_selected_run_asks_for_one(window: gui.MainWindow) -> None:
    """Undo acts on one run, so it will not guess which."""
    window._undo_dialog._run(dry_run=False)  # noqa: SLF001
    assert "Pick a run first." in window._undo_dialog._details.toPlainText()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Save After Generating (the batch nobody sits through)
# ---------------------------------------------------------------------------


def _ready_pair_for_autosave(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Two generated photos waiting to be written, with the toggle on and only *a* in the run."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    for path in (a, b):
        window._items[str(path)].status = READY  # noqa: SLF001
    window._generated_keys = [str(a)]  # noqa: SLF001 - b was generated by an earlier run
    window._autosave_action.setChecked(True)  # noqa: SLF001
    return a, b


def test_save_after_generating_writes_the_run_it_just_finished(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unattended batch: the photos a run generated are written without a second click."""
    a, _b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))

    window._continue_after_run()  # noqa: SLF001

    # Only the run's own photo: a proposal left over from an earlier one is still for review.
    assert saved == [[a]]
    assert window._generated_keys == []  # noqa: SLF001 - and it cannot be written twice


def test_generating_leaves_saving_to_the_user_by_default(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review-before-write stays the default, so a plain run writes nothing by itself."""
    a, _b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._autosave_action.setChecked(False)  # noqa: SLF001
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))

    window._continue_after_run()  # noqa: SLF001

    assert saved == []
    assert window._items[str(a)].status == READY  # noqa: SLF001


def test_a_cancelled_run_writes_nothing_even_with_save_after_generating(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel means stop, not "write the part that finished"."""
    _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))
    window._cancelling = True  # noqa: SLF001

    window._on_generate_finished()  # noqa: SLF001

    assert saved == []


def test_both_generate_menus_hold_the_one_set_of_save_options(
    window: gui.MainWindow,
) -> None:
    """A button that writes has to show what it writes with, not point at the other button."""
    options = window._save_options_menu  # noqa: SLF001
    for menu in (window._run_bar.generate_menu, window._detail.generate_menu):  # noqa: SLF001
        assert options.menuAction() in menu.actions()
    # The very same menu, so a field unchecked here is unchecked for every save there is.
    assert window._detail.save_button.menu() is options  # noqa: SLF001
    assert window._run_bar.save_button.menu() is options  # noqa: SLF001
    assert options.title() == "Save Options"


def test_the_generate_tooltip_names_what_a_chained_save_writes(
    window: gui.MainWindow,
) -> None:
    """The mode's whole risk is writing something you did not mean to, so the button says it."""
    window._autosave_action.setChecked(True)  # noqa: SLF001
    window._save_options_menu.write_description.setChecked(False)  # noqa: SLF001

    tip = _unwrapped(window._run_bar.generate_button.toolTip())  # noqa: SLF001

    assert "as Save Selected would" in tip
    assert "Currently writes title, keywords" in tip
    assert "Save Options" in tip


def test_the_toggle_renames_both_generate_buttons(
    window: gui.MainWindow,
) -> None:
    """A button that is about to write has to say so before it is pressed."""
    assert window._run_bar.generate_button.text() == "Generate Selected"  # noqa: SLF001
    assert window._detail.generate_button.text() == "Generate This Photo"  # noqa: SLF001

    window._autosave_action.setChecked(True)  # noqa: SLF001

    assert window._run_bar.generate_button.text() == "Generate and Save Selected"  # noqa: SLF001
    assert window._detail.generate_button.text() == "Generate and Save This Photo"  # noqa: SLF001
    # And the tooltip spells out what that write covers, the way the Save buttons do.
    assert "keywords" in window._run_bar.generate_button.toolTip()  # noqa: SLF001

    window._autosave_action.setChecked(False)  # noqa: SLF001

    assert window._run_bar.generate_button.text() == "Generate Selected"  # noqa: SLF001


def test_turning_the_toggle_on_with_nothing_to_write_says_so(
    window: gui.MainWindow,
) -> None:
    """Promising an unattended write the save options cannot make is worth catching early."""
    options = window._save_options_menu  # noqa: SLF001
    for action in (options.write_title, options.write_description, options.write_keywords):
        action.setChecked(False)

    window._autosave_action.setChecked(True)  # noqa: SLF001

    assert "Pick at least one field" in window._status.text()  # noqa: SLF001


def test_an_unwritable_run_reports_instead_of_saving_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coming back to a finished batch that wrote nothing must not be a silent surprise."""
    _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))
    options = window._save_options_menu  # noqa: SLF001
    for action in (options.write_title, options.write_description, options.write_keywords):
        action.setChecked(False)

    window._continue_after_run()  # noqa: SLF001

    assert saved == []
    assert "Pick at least one field" in window._status.text()  # noqa: SLF001


def test_a_chained_save_carries_what_the_run_wants_reviewed(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save's tally replaces the generation's line, so it has to say what that line said."""
    a, _b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._items[str(a)].foreign_terms = ["陶笛"]  # noqa: SLF001
    monkeypatch.setattr(window, "_run_save", lambda _items: None)

    window._continue_after_run()  # noqa: SLF001
    window._on_save_finished()  # noqa: SLF001

    assert "mixes writing systems" in window._status.text()  # noqa: SLF001
    assert window._save_note == ""  # noqa: SLF001 - consumed, so the next save starts clean


def test_a_chained_save_raises_one_alert_for_the_whole_job(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate-and-save is one job to the user, so the finished write is the one alert."""
    _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    raised: list[str] = []
    _patch_gui(monkeypatch, "notify", lambda _t, message, **_k: raised.append(message))
    monkeypatch.setattr(window, "_run_save", lambda _items: None)
    window._settings_menu._notify_action.setChecked(True)  # noqa: SLF001

    window._on_generate_finished()  # noqa: SLF001
    assert raised == []  # the run is not over: its photos are still to be written

    window._on_save_finished()  # noqa: SLF001
    assert raised == ["Saved 0 of 0 checked photos."]


def test_an_unattended_run_retries_what_failed_before_saving(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo lost to a model server that blinked must not sit unwritten until someone looks."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001 - both were in the run
    window._items[str(b)].status = FAILED  # noqa: SLF001
    runs: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_k: runs.append([i.path for i in items]),
    )
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))

    window._after_generation()  # noqa: SLF001 - the step a finished run takes next

    assert "Retrying 1 photo that failed" in window._status.text()  # noqa: SLF001
    assert saved == []  # the save waits for the retry
    window._run_pending_pass()  # noqa: SLF001 - fire the queued pass without its wait
    assert runs == [[b]]


def test_an_unattended_run_asks_again_for_mixed_writing_systems(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second answer usually comes back in one script, and nobody is there to ask for it."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    window._items[str(b)].foreign_terms = ["陶笛"]  # noqa: SLF001
    runs: list[tuple[list[Path], bool]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **kw: runs.append(([i.path for i in items], kw.get("refresh", False))),
    )
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))

    window._after_generation()  # noqa: SLF001

    assert "Retrying 1 photo that mixed writing systems" in window._status.text()  # noqa: SLF001
    assert saved == []  # the save waits for the answer
    window._run_pending_pass()  # noqa: SLF001

    # Refreshed, not cached: the answer it is unhappy with is the one in the cache.
    assert runs == [([b], True)]


def test_the_failed_pass_runs_before_the_mixed_script_one(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo that failed has no text yet, so its retry can only make more work for the other."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    window._items[str(a)].foreign_terms = ["陶笛"]  # noqa: SLF001
    window._items[str(b)].status = FAILED  # noqa: SLF001

    assert window._next_pass() == RERUN_FAILED  # noqa: SLF001
    window._passes_done.add(RERUN_FAILED)  # noqa: SLF001 - as starting it would
    assert window._next_pass() == RERUN_MIXED_SCRIPT  # noqa: SLF001


def test_a_reviewed_run_leaves_the_mixed_script_call_to_you(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed script is right when the photo shows the text, and only you can tell."""
    a, _b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._autosave_action.setChecked(False)  # noqa: SLF001
    window._generated_keys = [str(a)]  # noqa: SLF001
    window._items[str(a)].foreign_terms = ["陶笛"]  # noqa: SLF001

    assert window._next_pass() is None  # noqa: SLF001


def test_the_retry_pass_keeps_the_batch_tallies(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The closing line is the batch's, so a second pass over two photos must not reset it."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    _stub_generation(monkeypatch)  # so the background worker does no real I/O
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    window._vocabulary_mapped = 7  # noqa: SLF001 - what the first pass rewrote

    window._run_generation([window._items[str(b)]], extra_pass=True)  # noqa: SLF001

    assert window._vocabulary_mapped == 7  # noqa: SLF001, PLR2004 - still the batch's tally
    assert window._generated_keys == [str(a), str(b)]  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the pass started


def test_the_retry_pass_happens_once_per_batch(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that is simply down must not turn one batch into an endless loop."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    window._items[str(b)].status = FAILED  # noqa: SLF001
    monkeypatch.setattr(window, "_run_generation", lambda _items, **_k: None)

    assert window._start_extra_pass() is True  # noqa: SLF001
    assert window._next_pass() is None  # noqa: SLF001 - still failing, but that was its one go

    # A fresh run is a fresh batch, so it gets its own passes again.
    window._passes_done = set()  # noqa: SLF001 - what _run_generation does for a first pass
    assert window._next_pass() == RERUN_FAILED  # noqa: SLF001


def test_a_reviewed_run_leaves_retrying_to_the_retry_button(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the toggle off you are at the keyboard, so a second round of calls is your call."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._autosave_action.setChecked(False)  # noqa: SLF001
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    window._items[str(b)].status = FAILED  # noqa: SLF001

    assert window._start_extra_pass() is False  # noqa: SLF001


def test_an_extra_pass_carries_on_when_its_photos_are_gone(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chain must reach the save even if the photos it queued left the list meanwhile."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    saved: list[list[Path]] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append([i.path for i in items]))
    window._passes_done = set(EXTRA_PASSES)  # noqa: SLF001 - both passes already ran
    window._pending_pass = "failed"  # noqa: SLF001 - the pass the pause was for

    window._run_pending_pass()  # noqa: SLF001

    assert saved == [[a, b]]


def test_a_batch_that_still_has_failures_says_so(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save's tally counts what it wrote, so the photos it never got need their own line."""
    a, b = _ready_pair_for_autosave(window, tmp_path, monkeypatch)
    window._generated_keys = [str(a), str(b)]  # noqa: SLF001
    window._items[str(b)].status = FAILED  # noqa: SLF001
    monkeypatch.setattr(window, "_run_save", lambda _items: None)

    window._continue_after_run()  # noqa: SLF001
    window._on_save_finished()  # noqa: SLF001

    assert "1 photo failed" in window._status.text()  # noqa: SLF001


def test_save_after_generating_persists_to_the_config_file(
    window: gui.MainWindow,
) -> None:
    """Set once for overnight runs, and the next launch still knows it."""
    window._autosave_action.setChecked(True)  # noqa: SLF001

    assert window._current_config_values().save_after_generating is True  # noqa: SLF001


def test_a_window_built_from_that_config_comes_up_ready_to_write(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saved choice is back before the first click, button label and all."""
    _patch_gui(monkeypatch, "load_config", lambda: {"gui": {"save_after_generating": True}})
    win = gui.MainWindow()
    try:
        assert win._autosave_action.isChecked() is True  # noqa: SLF001
        assert win._run_bar.generate_button.text() == "Generate and Save Selected"  # noqa: SLF001
    finally:
        win.close()


def test_a_window_follows_the_config_file_sidecar_overrides(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save writes into each DNG as the CLI would, and the tooltip says so."""
    _patch_gui(
        monkeypatch,
        "load_config",
        lambda: {"output": {"sidecar_overrides": ["dng=photo"]}},
    )
    win = gui.MainWindow()
    try:
        assert win._save_options().sidecar_overrides == {".dng": "photo"}  # noqa: SLF001
        assert "but DNG into the image file" in win._save_options_summary()  # noqa: SLF001
    finally:
        win.close()


def test_a_window_opens_despite_a_bad_sidecar_override(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo the CLI refuses leaves the window usable, following the Write To mode alone."""
    _patch_gui(
        monkeypatch,
        "load_config",
        lambda: {"output": {"sidecar_overrides": ["dng=embed"]}},
    )
    win = gui.MainWindow()
    try:
        assert win._save_options().sidecar_overrides == {}  # noqa: SLF001
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Watching a folder
# ---------------------------------------------------------------------------


def _settled(path: Path) -> Path:
    """Write a photo whose mtime is old enough for the watcher to call it finished."""
    _jpeg(path)
    stamp = time.time() - 60
    os.utime(path, (stamp, stamp))
    return path


def test_watch_worker_emits_a_batch_and_stops(qapp: QApplication, tmp_path: Path) -> None:
    """The worker hands finished photos to the window and ends when it is told to."""
    photo = _settled(tmp_path / "a.jpg")
    worker = workers.WatchWorker(
        WatchSettings(folders=(tmp_path,), extensions="jpg", interval=0.0, settle=0.0),
    )
    batches: list[list[Path]] = []
    worker.batch.connect(lambda paths: (batches.append(paths), worker.stop()))

    worker.run()

    assert batches == [[photo]]


def test_watching_adds_new_photos_and_generates_them(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo that lands in the folder joins the list and is generated for review."""
    photo = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    window._extensions.setText("jpg")  # noqa: SLF001
    generated: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_k: generated.append([item.path for item in items]),
    )
    window._watch_settings = WatchSettings(folders=(tmp_path,), extensions="jpg")  # noqa: SLF001

    window._on_watch_batch([photo])  # noqa: SLF001

    assert str(photo) in window._items  # noqa: SLF001
    assert generated == [[photo]]
    assert "1 photo added" in window._status.text()  # noqa: SLF001


def test_watching_queues_photos_that_land_mid_run(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo arriving during a run waits for it rather than starting a second one."""
    photo = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    window._extensions.setText("jpg")  # noqa: SLF001
    generated: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_k: generated.append([item.path for item in items]),
    )
    monkeypatch.setattr(window, "_busy", lambda: True)
    window._watch_settings = WatchSettings(folders=(tmp_path,), extensions="jpg")  # noqa: SLF001

    window._on_watch_batch([photo])  # noqa: SLF001
    assert generated == []
    assert window._watch_pending == [str(photo)]  # noqa: SLF001

    monkeypatch.setattr(window, "_busy", lambda: False)
    window._continue_watch()  # noqa: SLF001
    assert generated == [[photo]]


def test_watching_can_save_without_reviewing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unattended import: opt in, and each generated photo is written straight away."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    for path in (a, b):
        window._items[str(path)].status = READY  # noqa: SLF001
    saved: list[int] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append(len(items)))
    window._watch_settings = WatchSettings(folders=(tmp_path,), save=True)  # noqa: SLF001

    window._continue_after_run()  # noqa: SLF001

    assert saved == [2]


def test_cancelling_a_save_does_not_restart_it_under_a_watch(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Cancel must stop the batch, not pause it for one event-loop turn.

    _reset_working puts the photos the save never reached back to READY, which is precisely what
    _continue_watch treats as "unsaved, write them", so an unattended watch used to rewrite them
    immediately and the cancel wrote every file anyway.
    """
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    for path in (a, b):
        window._items[str(path)].status = WORKING  # noqa: SLF001 - a save is mid-batch
    saved: list[int] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append(len(items)))
    window._watch_settings = WatchSettings(folders=(tmp_path,), save=True)  # noqa: SLF001
    window._cancelling = True  # noqa: SLF001 - the user pressed Cancel

    window._on_save_finished()  # noqa: SLF001

    assert saved == []
    assert not window._cancelling  # noqa: SLF001 - and the flag is cleared for the next run


def test_watching_leaves_saving_to_the_user_by_default(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review-before-write is the window's whole point, so a watch does not write by itself."""
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)
    for path in (a, b):
        window._items[str(path)].status = READY  # noqa: SLF001
    saved: list[int] = []
    monkeypatch.setattr(window, "_run_save", lambda items: saved.append(len(items)))
    window._watch_settings = WatchSettings(folders=(tmp_path,))  # noqa: SLF001

    window._continue_after_run()  # noqa: SLF001

    assert saved == []


def test_starting_and_stopping_a_watch_flips_the_menu_action(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """One menu entry both starts and stops the watch, and says which it will do."""
    window._start_watch(  # noqa: SLF001
        WatchSettings(folders=(tmp_path,), extensions="jpg", interval=1.0),
    )
    assert window._watch_action.text() == "Stop Watching"  # noqa: SLF001

    window._toggle_watch()  # noqa: SLF001

    assert window._watch_settings is None  # noqa: SLF001
    assert window._watch_thread is None  # noqa: SLF001
    assert window._watch_action.text() == "Watch Folder..."  # noqa: SLF001
    assert "Stopped watching." in window._status.text()  # noqa: SLF001


def test_the_watch_dialog_starts_the_watch_it_describes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the dialog holds when Start is pressed is the watch that runs."""
    started: list[WatchSettings] = []
    monkeypatch.setattr(window, "_start_watch", started.append)
    window._extensions.setText("jpg,cr3")  # noqa: SLF001
    dialog = window._watch_dialog  # noqa: SLF001
    dialog._folder.setText(str(tmp_path))  # noqa: SLF001
    dialog._save.setChecked(True)  # noqa: SLF001
    monkeypatch.setattr(dialog, "exec", lambda: QDialog.DialogCode.Accepted)

    window._toggle_watch()  # noqa: SLF001

    assert len(started) == 1
    assert started[0].folders == (tmp_path,)
    assert started[0].extensions == "jpg,cr3"
    assert started[0].save is True


def test_cancelling_the_watch_dialog_starts_nothing(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel means no watch, whatever the dialog holds."""
    started: list[WatchSettings] = []
    monkeypatch.setattr(window, "_start_watch", started.append)
    monkeypatch.setattr(window._watch_dialog, "exec", lambda: QDialog.DialogCode.Rejected)  # noqa: SLF001

    window._toggle_watch()  # noqa: SLF001

    assert started == []


def test_starting_a_watch_needs_a_folder(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty folder field is a mistake worth naming, not an empty watch."""
    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, _text: warned.append(title),
    )
    window._watch_dialog._folder.setText("")  # noqa: SLF001

    window._watch_dialog._on_start()  # noqa: SLF001

    assert warned == ["Pick a folder"]
    assert window._watch_settings is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Building a vocabulary
# ---------------------------------------------------------------------------


def _census() -> KeywordCensus:
    """Build a small keyword count, like reading a library's photos would produce."""
    census = KeywordCensus()
    census.add(["Animal", "Bird"], weight=5)
    census.add(["Sunset"], weight=4)
    census.add(["Fluke"], weight=1)
    return census


def test_vocabulary_build_worker_writes_a_file_and_a_report(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The build keeps what earns its place and says what it dropped, without touching a photo."""
    _patch_gui(monkeypatch, "census_from_photos", lambda *_a, **_k: _census())
    output = tmp_path / "vocabulary.txt"
    report = tmp_path / "dropped.csv"
    worker = workers.VocabularyBuildWorker(
        [Path("/a.jpg")],
        None,
        output,
        rules=TrimRules(min_uses=2),
        report_file=report,
    )
    done: list[tuple[str, int]] = []
    worker.done.connect(lambda message, kept: done.append((message, kept)))

    worker.run()

    text = output.read_text(encoding="utf-8")
    assert "# photo-tagger vocabulary: 3 keywords kept, 1 dropped." in text
    assert "Animal|Bird" in text
    assert "Fluke" not in text
    assert "Fluke,1,rare,used 1x" in report.read_text(encoding="utf-8")
    assert done[0][1] == 3  # noqa: PLR2004 - Animal, Bird, Sunset


def test_vocabulary_build_worker_reports_an_empty_library(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A library with no keywords cannot seed a vocabulary, and no file is written."""
    _patch_gui(monkeypatch, "census_from_photos", lambda *_a, **_k: KeywordCensus())
    output = tmp_path / "vocabulary.txt"
    worker = workers.VocabularyBuildWorker([Path("/a.jpg")], None, output, rules=TrimRules())
    failures: list[str] = []
    worker.failed.connect(failures.append)

    worker.run()

    assert "No keywords found" in failures[0]
    assert not output.exists()


def test_vocabulary_build_worker_organizes_when_asked(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional model pass runs with the chosen provider, and says so in the file."""
    _patch_gui(monkeypatch, "census_from_photos", lambda *_a, **_k: _census())
    passed: dict[str, object] = {}

    def fake_organize(result: object, **kwargs: object) -> tuple[object, OrganizeStats]:
        passed.update(kwargs)
        return result, OrganizeStats(model_name="qwen-vl", categories=["Animal"], grouped=1)

    _patch_gui(monkeypatch, "organize", fake_organize)
    output = tmp_path / "vocabulary.txt"
    worker = workers.VocabularyBuildWorker(
        [Path("/a.jpg")],
        None,
        output,
        rules=TrimRules(min_uses=2),
        organize_workers=3,
        provider="lmstudio",
        model="qwen-vl",
    )

    worker.run()

    assert passed["workers"] == 3  # noqa: PLR2004 - the value the dialog was set to
    assert passed["provider_name"] == "lmstudio"
    assert "# Organized by qwen-vl" in output.read_text(encoding="utf-8")


def test_vocabulary_build_worker_reads_an_export_too(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog that is not on this machine is still a source, through its keyword export."""
    _patch_gui(monkeypatch, "census_from_photos", lambda *_a, **_k: _census())
    export = tmp_path / "export.txt"
    export.write_text("Osprey\nOsprey\n", encoding="utf-8")
    output = tmp_path / "vocabulary.txt"
    worker = workers.VocabularyBuildWorker(
        [Path("/a.jpg")],
        export,
        output,
        rules=TrimRules(min_uses=2),
    )

    worker.run()

    text = output.read_text(encoding="utf-8")
    assert "Osprey" in text
    assert "keyword export export.txt" in text
    assert "photo(s)" in text


def test_starting_a_build_needs_a_source(window: gui.MainWindow) -> None:
    """With neither the photo list nor an export there is nothing to count."""
    window._builder_dialog._from_photos.setChecked(False)  # noqa: SLF001
    window._builder_dialog._export.setText("")  # noqa: SLF001

    window._builder_dialog._start()  # noqa: SLF001

    assert window._builder_dialog._thread is None  # noqa: SLF001
    assert "Pick a source" in window._builder_dialog._status.text()  # noqa: SLF001


def test_starting_a_build_uses_the_dialog_settings(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rules the dialog shows are the rules the build runs with."""
    photo = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": photo})
    monkeypatch.setattr(QThread, "start", lambda *_a, **_k: None)
    window._builder_dialog._output.setText(str(tmp_path / "v.txt"))  # noqa: SLF001
    window._builder_dialog._min_uses.setValue(5)  # noqa: SLF001
    window._builder_dialog._max_terms.setValue(0)  # noqa: SLF001 - "no cap"
    window._builder_dialog._digits.setChecked(True)  # noqa: SLF001

    window._builder_dialog._start()  # noqa: SLF001

    worker = window._builder_dialog._worker  # noqa: SLF001
    assert worker is not None
    assert worker._rules == TrimRules(min_uses=5, max_terms=None, allow_digits=True)  # noqa: SLF001
    assert worker._paths == [photo]  # noqa: SLF001
    assert worker._provider is None  # noqa: SLF001 - organizing was left off
    assert not window._builder_dialog._button.isEnabled()  # noqa: SLF001


def test_a_finished_build_offers_to_use_the_file(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Building a vocabulary and then having to go and choose it would be two steps too many."""
    listing = _keyword_file(tmp_path)
    window._builder_dialog._built = listing  # noqa: SLF001

    # The window fixture answers every question dialog with Yes.
    window._builder_dialog._on_done("Wrote 2 keyword(s) to keywords.txt, dropped 0.", 2)  # noqa: SLF001

    assert window._keyword_rules_dialog.vocabulary_path == listing  # noqa: SLF001
    assert window._keyword_rules_dialog.vocabulary is not None  # noqa: SLF001
    assert window._keyword_rules_dialog.vocabulary.match("ospreys") == "Osprey"  # noqa: SLF001


def test_changing_the_metadata_language_reloads_the_vocabulary(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plural folding is English-only, so the lookup index has to be rebuilt for a new language."""
    # The file has to exist: find_config_file ignores a $PHOTO_TAGGER_CONFIG that points at
    # nothing, and the window then falls back to the real ~/.config/photo-tagger/config.toml,
    # which this test would rewrite on the developer's own machine.
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))
    window._keyword_rules_dialog.load_vocabulary(_keyword_file(tmp_path, "Landschaft\n"))  # noqa: SLF001
    assert window._keyword_rules_dialog.vocabulary is not None  # noqa: SLF001
    assert window._keyword_rules_dialog.vocabulary.fold_plurals is True  # noqa: SLF001

    window._settings_menu.set_output_language("German")  # noqa: SLF001

    assert window._keyword_rules_dialog.vocabulary is not None  # noqa: SLF001
    assert window._keyword_rules_dialog.vocabulary.fold_plurals is False  # noqa: SLF001


def test_undo_waits_for_a_run_in_progress(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save in flight is writing the very files an undo would revert, so it has to finish."""
    a, _b = _save_two_photos(window, tmp_path, monkeypatch)
    window._undo_dialog.refresh()  # noqa: SLF001
    monkeypatch.setattr(window._undo_dialog, "_busy", lambda: True)  # noqa: SLF001

    window._undo_dialog._run(dry_run=False)  # noqa: SLF001

    assert a.with_suffix(".xmp").exists()
    assert "Wait for the run in progress" in window._undo_dialog._details.toPlainText()  # noqa: SLF001


def test_saving_writes_the_vocabulary_spelling_verbatim(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lower-case catalog must survive the save, or the window seeds the duplicate it prevents."""
    photo = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": photo})
    window._keyword_rules_dialog.load_vocabulary(_keyword_file(tmp_path, "gegenlicht\n"))  # noqa: SLF001
    captured = _capture_write(monkeypatch)
    _select(window, window._tree.leaf_for(photo))  # noqa: SLF001
    window._detail._keywords.setPlainText("gegenlicht")  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    keywords = captured["keywords"]
    assert keywords is not None
    assert keywords.subject == ["gegenlicht"]


def test_photo_by_photo_saves_share_one_journal(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Saving one photo at a time is one session, not one run per click.

    Journals are pruned to the fifty most recent, so a journal per click would push every command-
    line run out of the undo list during an ordinary review session.
    """
    a, b = _two_ready_photos(window, tmp_path, monkeypatch)

    def fake_write(path: Path, *_a: object, **_k: object) -> bool:
        path.with_suffix(".xmp").write_text("<xmp/>", encoding="utf-8")
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    for path in (a, b):
        _select(window, window._tree.leaf_for(path))  # noqa: SLF001
        window._save_current()  # noqa: SLF001

    journals = list_journals()
    assert len(journals) == 1
    assert len(read_journal(journals[0])) == 2  # noqa: PLR2004 - one entry per saved photo


def test_a_batch_save_gets_its_own_journal(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch is a run, so undoing it puts back that batch and nothing else."""
    photo = _jpeg(tmp_path / "single.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"single": photo})
    window._items[str(photo)].has_proposal = True  # noqa: SLF001
    window._items[str(photo)].title = "T"  # noqa: SLF001

    def fake_write(path: Path, *_a: object, **_k: object) -> bool:
        path.with_suffix(".xmp").write_text("<xmp/>", encoding="utf-8")
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    _stub_save_helper(monkeypatch)
    _select(window, window._tree.leaf_for(photo))  # noqa: SLF001
    window._save_current()  # noqa: SLF001 - one photo by hand first

    window._save_selected()  # noqa: SLF001 - then the batch
    _drain_save(window)

    assert len(list_journals()) == 2  # noqa: PLR2004 - the session's single saves, and the batch


def _viewer(qapp: QApplication, size: tuple[int, int] = (400, 300)) -> viewer.ImageViewerDialog:
    """Build a viewer over a solid-color pixmap of *size*, laid out but never shown modally."""
    from PySide6.QtGui import QColor, QPixmap  # noqa: PLC0415

    pixmap = QPixmap(*size)
    pixmap.fill(QColor("red"))
    dialog = viewer.ImageViewerDialog(None, Path("/photos/a.jpg"), pixmap)
    dialog.resize(200, 200)
    dialog.show()
    QApplication.processEvents()
    return dialog


def test_viewer_opens_fitted_to_the_window(qapp: QApplication) -> None:
    """A photo larger than the window starts scaled down so all of it is visible."""
    dialog = _viewer(qapp)
    try:
        assert dialog._fit_to_window  # noqa: SLF001
        assert dialog._zoom < 1.0  # noqa: SLF001 - 400x300 does not fit a 200x200 window
        assert dialog.windowTitle() == "a.jpg"
    finally:
        dialog.close()


def test_viewer_zoom_leaves_fit_mode_and_scales_the_canvas(qapp: QApplication) -> None:
    """Zooming in enlarges the drawn pixmap and stops the window resize from refitting."""
    dialog = _viewer(qapp)
    try:
        dialog._set_zoom(1.0)  # noqa: SLF001
        assert not dialog._fit_to_window  # noqa: SLF001
        assert dialog._canvas.pixmap().width() == 400  # noqa: SLF001, PLR2004 - 1:1 with the source
        assert dialog._zoom_label.text() == "100%"  # noqa: SLF001

        dialog._zoom_by(1)  # noqa: SLF001
        assert dialog._canvas.pixmap().width() > 400  # noqa: SLF001, PLR2004 - a notch larger
    finally:
        dialog.close()


def test_viewer_double_click_toggles_between_fit_and_actual_size(qapp: QApplication) -> None:
    """The double-click shortcut swings to 1:1 and back to fit."""
    dialog = _viewer(qapp)
    try:
        dialog._toggle_fit()  # noqa: SLF001
        assert dialog._zoom == 1.0  # noqa: SLF001
        dialog._toggle_fit()  # noqa: SLF001
        assert dialog._fit_to_window  # noqa: SLF001
        assert dialog._zoom < 1.0  # noqa: SLF001
    finally:
        dialog.close()


def test_viewer_zoom_stops_at_the_ceiling(qapp: QApplication) -> None:
    """Holding zoom in cannot grow the pixmap without bound."""
    dialog = _viewer(qapp)
    try:
        for _ in range(30):
            dialog._zoom_by(1)  # noqa: SLF001
        assert dialog._zoom == MAX_ZOOM  # noqa: SLF001
    finally:
        dialog.close()


def test_preview_click_opens_the_viewer(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clicking the detail pane's preview opens the full-size viewer on the open photo."""
    a = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a})
    _select(window, window._tree.leaf_for(a))  # noqa: SLF001

    opened: list[Path] = []
    monkeypatch.setattr(
        viewer.ImageViewerDialog,
        "exec",
        lambda self: opened.append(self.windowTitle()),
    )
    window._detail._preview.clicked.emit()  # noqa: SLF001

    assert opened == ["a.jpg"]


def test_preview_click_without_an_open_photo_does_nothing(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The placeholder preview is clickable but has nothing to open."""
    opened: list[str] = []
    monkeypatch.setattr(viewer.ImageViewerDialog, "exec", lambda _self: opened.append("opened"))
    window._detail._preview.clicked.emit()  # noqa: SLF001
    assert opened == []


def test_viewer_falls_back_to_the_preview_when_the_full_decode_fails(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo that will not decode at viewer size still opens, on the small preview."""
    a = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a})
    item = window._items[str(a)]  # noqa: SLF001

    def _fail_big(path: Path, max_size: int = 0, **_kwargs: object) -> object:
        if max_size == detail_pane._VIEWER_MAX:  # noqa: SLF001
            msg = "no decoder"
            raise OSError(msg)
        return _real_prepare(path, max_size=max_size)

    _real_prepare = image_io.prepare_image_for_agent
    _patch_gui(monkeypatch, "prepare_image_for_agent", _fail_big)

    pixmap = window._detail._viewer_pixmap(item)  # noqa: SLF001

    assert pixmap is not None
    assert not pixmap.isNull()
    assert window._detail._viewer_cache is None  # noqa: SLF001 - a fallback is not cached as the full size


def test_viewer_cache_holds_one_photo_at_a_time(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The viewer's decode is kept for the last photo only, so long sessions do not grow."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a, "b": b})

    window._detail._viewer_pixmap(window._items[str(a)])  # noqa: SLF001
    assert window._detail._viewer_cache[0] == str(a)  # noqa: SLF001

    window._detail._viewer_pixmap(window._items[str(b)])  # noqa: SLF001
    assert window._detail._viewer_cache[0] == str(b)  # noqa: SLF001

    # Removing the cached photo drops its pixmap with it.
    window._remove_items([str(b)])  # noqa: SLF001
    assert window._detail._viewer_cache is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Mixed-language output
# ---------------------------------------------------------------------------


_GUI_MIXED = InferenceResult(
    title="Decorated陶笛",
    description="A陶笛 on a white surface.",
    keywords=["陶笛<Wind Instrument"],
)
_GUI_REPAIRED = InferenceResult(
    title="Decorated Ocarina",
    description="An ocarina on a white surface.",
    keywords=["Ocarina<Wind Instrument"],
)


def test_worker_marks_a_proposal_that_mixes_writing_systems(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag reaches the window with the proposal, naming the word to look at."""
    _stub_generation(monkeypatch)
    _patch_gui(monkeypatch, "analyze_image_with_ai", lambda **_k: _GUI_MIXED)
    worker = workers.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
    done: list[Proposal] = []
    worker.file_done.connect(done.append)
    worker.run()

    assert done[0].foreign_terms == ["陶笛"]


def test_worker_repairs_mixed_output_when_the_setting_is_on(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the toggle on, a second call rewrites the answer and clears the flag."""
    _stub_generation(monkeypatch)
    answers = iter([_GUI_MIXED, _GUI_REPAIRED])
    prompts: list[str] = []

    def record(**kwargs: object) -> InferenceResult:
        prompts.append(str(kwargs["user_prompt"]))
        return next(answers)

    _patch_gui(monkeypatch, "analyze_image_with_ai", record)
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(repair_language=True),
    )
    done: list[Proposal] = []
    worker.file_done.connect(done.append)
    worker.run()

    assert "陶笛" in prompts[1]
    assert done[0].title == "Decorated Ocarina"
    assert done[0].foreign_terms == []


def test_worker_keeps_the_answer_when_the_repair_call_fails(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead server on the follow-up costs the fix, never the proposal."""
    _stub_generation(monkeypatch)
    answers = iter([_GUI_MIXED])

    def flaky(**_k: object) -> InferenceResult:
        try:
            return next(answers)
        except StopIteration:
            msg = "the model server died on the follow-up"
            raise RuntimeError(msg) from None

    _patch_gui(monkeypatch, "analyze_image_with_ai", flaky)
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(repair_language=True),
    )
    done: list[Proposal] = []
    worker.file_done.connect(done.append)
    worker.run()

    assert done[0].title == "Decorated陶笛"
    assert done[0].foreign_terms == ["陶笛"]


def test_worker_keeps_the_original_when_the_repair_mixes_scripts_again(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proposal keeps the text the check flagged, and counts what both calls cost."""
    _stub_generation(monkeypatch)
    first = replace(_GUI_MIXED, total_tokens=100)
    second = replace(_GUI_MIXED, title="Painted陶器", total_tokens=40)
    answers = iter([first, second])
    _patch_gui(monkeypatch, "analyze_image_with_ai", lambda **_k: next(answers))
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(repair_language=True),
    )
    done: list[Proposal] = []
    worker.file_done.connect(done.append)
    worker.run()

    assert done[0].title == "Decorated陶笛"
    assert done[0].total_tokens == first.total_tokens + second.total_tokens


def test_the_detail_pane_says_when_the_text_mixes_writing_systems(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """The note sits over the editable fields and clears for a photo with nothing wrong."""
    flagged = _jpeg(tmp_path / "a.jpg")
    clean = _jpeg(tmp_path / "b.jpg")
    window._add_inputs([flagged, clean])  # noqa: SLF001 - exercising the window's own state
    items = window._items  # noqa: SLF001
    items[str(flagged)].foreign_terms = ["陶笛"]

    window._show_item(items[str(flagged)])  # noqa: SLF001
    assert "陶笛" in window._detail._review_note.text()  # noqa: SLF001

    window._show_item(items[str(clean)])  # noqa: SLF001
    assert window._detail._review_note.text() == ""  # noqa: SLF001


def test_a_hand_fix_of_the_flagged_words_clears_the_flag(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Fixing the words by hand takes the note away, and the photo out of the mixed-script pass."""
    img = _jpeg(tmp_path / "a.jpg")
    window._add_inputs([img])  # noqa: SLF001 - exercising the window's own state
    item = window._items[str(img)]  # noqa: SLF001
    item.title = "Decorated陶笛"
    item.foreign_terms = ["陶笛"]
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    assert window._current is item  # noqa: SLF001

    window._detail._title.setText("Decorated Ocarina")  # noqa: SLF001
    window._commit_current()  # noqa: SLF001

    assert item.foreign_terms == []
    assert window._detail._review_note.text() == ""  # noqa: SLF001


def test_the_closing_line_counts_only_the_run_s_mixed_photos(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A photo flagged by an earlier run is not this run's to report."""
    old = _jpeg(tmp_path / "a.jpg")
    new = _jpeg(tmp_path / "b.jpg")
    window._add_inputs([old, new])  # noqa: SLF001
    window._items[str(old)].foreign_terms = ["陶笛"]  # noqa: SLF001
    window._generated_keys = [str(new)]  # noqa: SLF001

    assert "mix" not in window._generation_notes()  # noqa: SLF001


def test_the_repair_toggle_lands_in_the_saved_settings(window: gui.MainWindow) -> None:
    """Flipping the Settings entry is what a config save then writes."""
    window._settings_menu._repair_action.setChecked(True)  # noqa: SLF001
    assert window._current_config_values().repair_language is True  # noqa: SLF001
    window._settings_menu._repair_action.setChecked(False)  # noqa: SLF001
    assert window._current_config_values().repair_language is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# Copyright and credit
# ---------------------------------------------------------------------------


def _taken(year: int, **more: str) -> FieldSources:
    """
    Return what a photo taken in *year* carries, plus *more* tags, in the photo and its sidecar.

    Both files, so the values count whichever of them the Write To mode saves to.
    """
    values = {"EXIF:DateTimeOriginal": f"{year}:05:04 10:00:00"}
    values |= {tag.replace("_", ":", 1): value for tag, value in more.items()}
    return FieldSources(image=values, sidecar=values)


def _fill_copyright(
    window: gui.MainWindow,
    *,
    artist: str = "Ada Lovelace",
    notice: str = "Copyright (c) {year} {artist}",
    rights: str = "",
    marked: bool | None = True,
) -> None:
    """Type into the Copyright dialog's fields the way a user would."""
    dialog = window._copyright_dialog  # noqa: SLF001
    dialog._artist_field.setText(artist)  # noqa: SLF001
    dialog._notice_field.setText(notice)  # noqa: SLF001
    dialog._rights_field.setText(rights)  # noqa: SLF001
    dialog._marked_box.setCurrentIndex(dialog._marked_box.findData(marked))  # noqa: SLF001


def test_copyright_dialog_previews_the_resolved_tags(window: gui.MainWindow) -> None:
    """Typing a template shows what it becomes, which is how a wrong one is caught."""
    _fill_copyright(window, rights="© {year} {artist}")

    preview = window._copyright_dialog._preview.toPlainText()  # noqa: SLF001
    year = datetime.now(tz=UTC).astimezone().year
    assert f"EXIF:Copyright = Copyright (c) {year} Ada Lovelace" in preview
    assert f"XMP-dc:Rights = © {year} Ada Lovelace" in preview
    assert "XMP-xmpRights:Marked = True" in preview


def test_copyright_dialog_reports_a_template_it_cannot_fill(window: gui.MainWindow) -> None:
    """A placeholder it does not know says so, and nothing is written."""
    _fill_copyright(window, notice="Copyright (c) {year} {shot_at}")

    assert "{shot_at}" in window._copyright_dialog._preview.toPlainText()  # noqa: SLF001
    assert window._copyright_dialog.fields is None  # noqa: SLF001
    assert not window._writes_copyright()  # noqa: SLF001


def test_copyright_reaches_the_save_options_summary(window: gui.MainWindow) -> None:
    """The Save button says the copyright is part of what it writes."""
    assert "copyright" not in window._save_options_summary()  # noqa: SLF001

    _fill_copyright(window)

    assert "copyright" in window._save_options_summary()  # noqa: SLF001


def test_save_writes_the_copyright_dated_by_the_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single save stamps the credit, with the year read off that photo."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags, **_k: dict.fromkeys(paths, _taken(2011)),
    )
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    _fill_copyright(window)

    window._save_current()  # noqa: SLF001

    extra = captured["extra"]
    assert extra["EXIF:Copyright"] == "Copyright (c) 2011 Ada Lovelace"  # type: ignore[index]
    assert extra["XMP-dc:Creator"] == "Ada Lovelace"  # type: ignore[index]


def test_a_photo_generated_before_it_was_opened_reads_its_copyright_tags(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposal fills the existing metadata, not the tags the row needs, so opening reads them."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    sources = _taken(2011, XMP_Rights="(c) 2019 Someone Else")
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags: dict.fromkeys(paths, sources),
    )
    _add_dir(window, {"a": img})
    _fill_copyright(window)
    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=img,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="T",
            description="D",
            keywords=[],
        ),
    )

    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    assert window._detail._planned_copyright.text() == "(keeps the notice it has)"  # noqa: SLF001


def test_the_copyright_row_compares_the_file_with_what_a_save_writes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing shows the notice on the file; New follows the settings, dated by the photo."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    context = ImageContext(
        existing_copyright="(c) 2019 Someone Else",
        field_sources=_taken(2011, XMP_Rights="(c) 2019 Someone Else"),
    )
    _patch_gui(monkeypatch, "read_image_context", lambda _p, **_k: context)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    assert window._detail._existing_copyright.text() == "(c) 2019 Someone Else"  # noqa: SLF001
    assert window._detail._planned_copyright.text() == "(not written)"  # noqa: SLF001

    _fill_copyright(window)
    assert window._detail._planned_copyright.text() == "(keeps the notice it has)"  # noqa: SLF001

    window._copyright_dialog._preserve_copyright_box.setChecked(False)  # noqa: SLF001
    assert window._detail._planned_copyright.text() == "Copyright (c) 2011 Ada Lovelace"  # noqa: SLF001

    window._save_options_menu.write_copyright.setChecked(False)  # noqa: SLF001
    assert window._detail._planned_copyright.text() == "(not written)"  # noqa: SLF001


def test_the_copyright_row_says_none_for_a_photo_without_one(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare photo reads '(none)' on the Existing side, like the other empty fields."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    assert window._detail._existing_copyright.text() == "(none)"  # noqa: SLF001


def test_a_saved_notice_takes_the_photo_out_of_no_copyright(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save records the notice it wrote, so the filter lets go of the photo with no rescan."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _capture_write(monkeypatch)
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags, **_k: dict.fromkeys(paths, _taken(2011)),
    )
    _add_dir(window, {"a": img})
    window._on_scan_done({str(img): set()})  # noqa: SLF001
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    _fill_copyright(window)

    window._save_current()  # noqa: SLF001

    assert not photo_matches_filter(window._items[str(img)], FILTER_NO_COPYRIGHT)  # noqa: SLF001


def test_save_reads_the_photo_right_before_the_write(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single save reads what the templates need fresh, and keeps a notice the photo has."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    asked: list[tuple[str, ...]] = []

    def fake_read(paths: list[Path], tags: tuple[str, ...], **_k: object) -> dict[Path, object]:
        asked.append(tags)
        return dict.fromkeys(paths, _taken(2019, EXIF_Copyright="(c) 2011 Grace"))

    _patch_gui(monkeypatch, "read_field_sources", fake_read)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    _fill_copyright(window)

    window._save_current()  # noqa: SLF001

    assert "EXIF:DateTimeOriginal" in asked[0]
    extra = captured["extra"]
    assert "EXIF:Copyright" not in extra  # type: ignore[operator]
    assert extra["EXIF:Artist"] == "Ada Lovelace"  # type: ignore[index]


def test_save_writes_nothing_extra_without_a_copyright(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The usual save is unchanged: no copyright configured, no extra tags."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["extra"] == {}


def test_a_copyright_alone_is_enough_to_save(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every generated field off, the save still stamps the credit rather than nagging."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags, **_k: dict.fromkeys(paths, _taken(2025)),
    )
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    _fill_copyright(window)
    options = window._save_options_menu  # noqa: SLF001
    for action in (options.write_title, options.write_description, options.write_keywords):
        action.setChecked(False)

    window._save_current()  # noqa: SLF001

    assert captured["title"] is None
    assert captured["extra"]["EXIF:Artist"] == "Ada Lovelace"  # type: ignore[index]


def test_save_selected_stamps_a_copyright_on_photos_never_generated(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-tagged photo gets the notice alone; the generated one gets its text as well."""
    fresh = _jpeg(tmp_path / "a.jpg")
    tagged = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=["Old"])
    _stub_save_helper(monkeypatch)
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags, **_k: dict.fromkeys(paths, _taken(2011)),
    )
    written: dict[str, dict[str, object]] = {}

    def fake_write(path: Path, keywords: KeywordSet, **kwargs: object) -> bool:
        written[path.name] = {"keywords": keywords, **kwargs}
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    _add_dir(window, {"a": fresh, "b": tagged})
    item = window._items[str(fresh)]  # noqa: SLF001
    item.has_proposal = True
    item.title = "New Title"
    _select(window, window._tree.leaf_for(tagged))  # noqa: SLF001 - opened, so its fields are seeded
    _fill_copyright(window)

    window._save_selected()  # noqa: SLF001
    _drain_save(window)

    assert written["a.jpg"]["title"] == "New Title"
    assert written["b.jpg"]["title"] is None
    assert written["b.jpg"]["description"] is None
    assert written["b.jpg"]["keywords"].is_empty()  # type: ignore[attr-defined]
    extra = written["b.jpg"]["extra"]
    assert extra["EXIF:Copyright"] == "Copyright (c) 2011 Ada Lovelace"  # type: ignore[index]
    # A stamp says nothing about generation, so the tagged photo keeps its status word.
    assert window._items[str(tagged)].status != SAVED  # noqa: SLF001
    assert window._items[str(fresh)].status == SAVED  # noqa: SLF001


def test_save_selected_without_a_copyright_still_needs_a_proposal(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to stamp, a photo never generated has nothing to save, and the line says how."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})

    window._save_selected()  # noqa: SLF001

    assert window._save_thread is None  # noqa: SLF001
    assert "Settings > Copyright" in window._status.text()  # noqa: SLF001


def test_save_worker_dates_each_photo_in_the_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch save reads every capture year once, then stamps each photo with its own."""
    _patch_gui(monkeypatch, "managed_helper", lambda _et: nullcontext(None))
    old, new = tmp_path / "old.jpg", tmp_path / "new.jpg"
    reads: list[list[Path]] = []

    def fake_sources(paths: list[Path], _tags: object, **_kwargs: object) -> dict[Path, object]:
        reads.append(list(paths))
        return {old: _taken(2011), new: _taken(2025)}

    _patch_gui(monkeypatch, "read_field_sources", fake_sources)
    written: dict[str, object] = {}

    def fake_write(path: Path, _keywords: object, **kwargs: object) -> bool:
        written[path.name] = kwargs["extra"]
        return True

    monkeypatch.setattr(metadata, "write_metadata", fake_write)
    fields, _error = build_copyright_fields(
        artist="Ada",
        notice="(c) {year} {artist}",
        rights="",
        marked=None,
    )
    jobs = [
        SaveJob(path=path, keywords=KeywordSet(), title=None, description=None)
        for path in (old, new)
    ]

    workers.SaveWorker(jobs, options=SaveOptions(extra_fields=fields)).run()

    assert reads == [[old, new]]  # one batched read for the whole save
    assert written["old.jpg"]["EXIF:Copyright"] == "(c) 2011 Ada"  # type: ignore[index]
    assert written["new.jpg"]["EXIF:Copyright"] == "(c) 2025 Ada"  # type: ignore[index]


def test_save_worker_counts_a_photo_that_keeps_its_notice_as_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copyright-only save of a photo that has one writes nothing, and reports success."""
    _patch_gui(monkeypatch, "managed_helper", lambda _et: nullcontext(None))
    image = tmp_path / "a.jpg"
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags, **_k: dict.fromkeys(paths, _taken(2011, EXIF_Copyright="(c) 2011")),
    )

    def _refuse(*_a: object, **_k: object) -> bool:
        message = "nothing to write, so no write"
        raise AssertionError(message)

    monkeypatch.setattr(metadata, "write_metadata", _refuse)
    fields, _error = build_copyright_fields(artist="", notice="(c) {year}", rights="", marked=None)
    job = SaveJob(path=image, keywords=KeywordSet(), title=None, description=None)
    worker = workers.SaveWorker([job], options=SaveOptions(extra_fields=fields))
    done: list[tuple[str, bool]] = []
    worker.file_done.connect(lambda path, ok: done.append((path, ok)))

    worker.run()

    assert done == [(str(image), True)]


def test_save_worker_skips_the_capture_read_without_a_copyright(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save that writes no copyright costs exactly what it always did."""
    _patch_gui(monkeypatch, "managed_helper", lambda _et: nullcontext(None))

    def _refuse(*_a: object, **_k: object) -> dict[Path, int]:
        message = "no copyright, no capture-date read"
        raise AssertionError(message)

    _patch_gui(monkeypatch, "read_field_sources", _refuse)
    monkeypatch.setattr(metadata, "write_metadata", lambda *_a, **_k: True)
    jobs = [SaveJob(path=tmp_path / "a.jpg", keywords=KeywordSet(), title="T", description=None)]

    workers.SaveWorker(jobs, options=SaveOptions()).run()


def test_saving_settings_persists_the_copyright(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save Settings as Defaults writes the [copyright] table CLI runs read too."""
    config = tmp_path / "config.toml"
    _patch_gui(monkeypatch, "user_config_path", lambda: config)
    _patch_gui(monkeypatch, "find_config_file", lambda: None)
    _fill_copyright(window, rights="© {year} {artist}")
    window._copyright_dialog._preserve_copyright_box.setChecked(False)  # noqa: SLF001

    window._save_config()  # noqa: SLF001

    import tomllib  # noqa: PLC0415 - test-local parser.

    saved = tomllib.loads(config.read_text(encoding="utf-8"))["copyright"]
    assert saved["artist"] == "Ada Lovelace"
    assert saved["copyright_notice"] == "Copyright (c) {year} {artist}"
    assert saved["rights"] == "© {year} {artist}"
    assert saved["marked"] is True
    assert saved["preserve_copyright"] is False
    assert window._copyright_dialog.fields is not None  # noqa: SLF001
    assert not window._copyright_dialog.fields.preserve  # noqa: SLF001


def test_copyright_fields_fill_the_dialog(window: gui.MainWindow) -> None:
    """
    The form grows its fields, whatever the platform style says.

    QMacStyle leaves them at their size hint, which cut the notice field down to a couple of hundred
    pixels and hid the example template it carries.
    """
    form = window._copyright_dialog.findChild(QFormLayout)  # noqa: SLF001
    assert form is not None
    assert form.fieldGrowthPolicy() == QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow


def _copyright_button(window: gui.MainWindow, text: str) -> QPushButton:
    """Find one of the Copyright dialog's Insert buttons by the text it types."""
    buttons = window._copyright_dialog.findChildren(QPushButton)  # noqa: SLF001
    return next(button for button in buttons if button.text() == text)


def _focus_copyright(window: gui.MainWindow, field: QLineEdit) -> None:
    """Put the keyboard cursor in one of the dialog's template fields."""
    dialog = window._copyright_dialog  # noqa: SLF001
    dialog.show()
    dialog.activateWindow()
    QApplication.processEvents()
    field.setFocus()
    QApplication.processEvents()


def test_insert_buttons_build_a_notice_without_typing_it(window: gui.MainWindow) -> None:
    """The three buttons cover the parts of a notice a keyboard makes awkward."""
    _focus_copyright(window, window._copyright_dialog._notice_field)  # noqa: SLF001

    _copyright_button(window, "©").click()
    _copyright_button(window, "{year}").click()
    _copyright_button(window, "{artist}").click()

    assert window._copyright_dialog._notice_field.text() == "©{year}{artist}"  # noqa: SLF001
    window._copyright_dialog.close()  # noqa: SLF001


def test_insert_goes_to_the_field_holding_the_cursor(window: gui.MainWindow) -> None:
    """A copyright sign belongs in XMP as often as in the notice, so the cursor decides."""
    _focus_copyright(window, window._copyright_dialog._rights_field)  # noqa: SLF001

    _copyright_button(window, "©").click()

    assert window._copyright_dialog._rights_field.text() == "©"  # noqa: SLF001
    assert window._copyright_dialog._notice_field.text() == ""  # noqa: SLF001
    window._copyright_dialog.close()  # noqa: SLF001


def test_insert_falls_back_to_the_notice_field(window: gui.MainWindow) -> None:
    """With the cursor nowhere, the notice is the field a template belongs in."""
    _copyright_button(window, "{year}").click()

    assert window._copyright_dialog._notice_field.text() == "{year}"  # noqa: SLF001


def test_insert_leaves_the_typing_where_it_was(window: gui.MainWindow) -> None:
    """The button takes no focus, so the next keystroke still lands in the field."""
    _focus_copyright(window, window._copyright_dialog._rights_field)  # noqa: SLF001

    _copyright_button(window, "©").click()

    assert QApplication.focusWidget() is window._copyright_dialog._rights_field  # noqa: SLF001
    window._copyright_dialog.close()  # noqa: SLF001


def test_select_by_status_isolates_the_mixed_script_photos(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """The route the run's closing line names: Check Only > Status > Mixed writing systems."""
    flagged = _jpeg(tmp_path / "a.jpg")
    clean = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": flagged, "b": clean})
    window._items[str(flagged)].foreign_terms = ["陶笛"]  # noqa: SLF001

    status = _select_submenu(window, "Check Only").actions()[1].menu()
    entry = next(a for a in status.actions() if a.text() == "Mixed writing systems")
    entry.trigger()

    assert window._items[str(flagged)].selected is True  # noqa: SLF001
    assert window._items[str(clean)].selected is False  # noqa: SLF001


def test_the_grid_can_show_only_the_mixed_script_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The folder grid offers the same filter, so a visual pass over a shoot works too."""
    flagged = _jpeg(tmp_path / "a.jpg")
    clean = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window._folder_grid, "_start_thumbs", lambda _paths: None)  # noqa: SLF001
    _add_dir(window, {"a": flagged, "b": clean})
    window._items[str(flagged)].foreign_terms = ["陶笛"]  # noqa: SLF001
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    combo = window._folder_grid._filter_combo  # noqa: SLF001
    combo.setCurrentIndex(combo.findData(FILTER_MIXED_SCRIPT))

    assert _grid_order(window) == ["a.jpg"]


def test_the_status_column_marks_a_mixed_script_photo(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A flagged row can be found by scrolling, and hovering it names the words."""
    flagged = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": flagged})
    item = window._items[str(flagged)]  # noqa: SLF001
    item.status = READY
    item.foreign_terms = ["陶笛"]
    window._refresh_status_cell(item)  # noqa: SLF001

    leaf = window._tree.leaf_for(flagged)  # noqa: SLF001
    assert leaf is not None
    assert leaf.text(COL_STATUS) == "ready ⚠"
    assert "陶笛" in leaf.toolTip(COL_STATUS)


def test_copyright_preview_grows_with_its_content(window: gui.MainWindow) -> None:
    """
    The preview is a box that fits its lines, not a label the dialog squeezes.

    A label grew past the space the dialog gave it as fields were filled in, and its last tag line
    was cut off mid-text.
    """
    preview = window._copyright_dialog._preview  # noqa: SLF001
    _fill_copyright(window, artist="", notice="", marked=None)
    short = preview.height()

    _fill_copyright(window, rights="© {year} {artist}")

    assert preview.height() > short
    assert preview.toPlainText().splitlines()[-1].startswith("XMP-xmpRights:Marked")


def test_copyright_preview_flags_a_notice_exif_cannot_promise(window: gui.MainWindow) -> None:
    """A copyright sign in the EXIF notice is a bet on the reader, and the preview says so."""
    _fill_copyright(window, notice="Copyright © {year} {artist}")

    assert "EXIF:Copyright" in window._copyright_dialog._warning.text()  # noqa: SLF001
    assert "Exif 3.0" in window._copyright_dialog._warning.text()  # noqa: SLF001


def test_copyright_preview_stays_quiet_for_a_plain_notice(window: gui.MainWindow) -> None:
    """The usual ASCII notice gets no warning, so the one above means something."""
    _fill_copyright(window, notice="Copyright (c) {year} {artist}")

    assert window._copyright_dialog._warning.text() == ""  # noqa: SLF001


def test_write_copyright_toggle_skips_the_block(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking Write Copyright skips the credit without clearing what the dialog holds."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _patch_gui(
        monkeypatch,
        "read_field_sources",
        lambda paths, _tags, **_k: dict.fromkeys(paths, _taken(2025)),
    )
    _add_dir(window, {"a": img})
    _select(window, window._tree.leaf_for(img))  # noqa: SLF001
    _fill_copyright(window)
    window._save_options_menu.write_copyright.setChecked(False)  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["extra"] == {}
    # The dialog still holds it, ready for the next save.
    assert window._copyright_dialog._notice_field.text() != ""  # noqa: SLF001


def test_write_copyright_toggle_reaches_the_save_summary(window: gui.MainWindow) -> None:
    """The Save button stops claiming a copyright it is no longer writing."""
    _fill_copyright(window)
    assert "copyright" in window._save_options_summary()  # noqa: SLF001

    window._save_options_menu.write_copyright.setChecked(False)  # noqa: SLF001

    assert "copyright" not in window._save_options_summary()  # noqa: SLF001


def test_write_copyright_is_grayed_out_until_there_is_one(window: gui.MainWindow) -> None:
    """A toggle for something that cannot happen yet says so instead of looking like a choice."""
    _fill_copyright(window, artist="", notice="", marked=None)
    assert not window._save_options_menu.write_copyright.isEnabled()  # noqa: SLF001
    assert "Settings > Copyright" in window._save_options_menu.write_copyright.toolTip()  # noqa: SLF001

    _fill_copyright(window)

    assert window._save_options_menu.write_copyright.isEnabled()  # noqa: SLF001
    assert window._save_options_menu.write_copyright.isChecked()  # noqa: SLF001


def test_a_copyright_alone_needs_its_toggle_on(window: gui.MainWindow) -> None:
    """With every field off and the copyright skipped, a save would write nothing and says so."""
    _fill_copyright(window)
    options = window._save_options_menu  # noqa: SLF001
    for action in (options.write_title, options.write_description, options.write_keywords):
        action.setChecked(False)
    assert window._write_fields_chosen()  # noqa: SLF001

    window._save_options_menu.write_copyright.setChecked(False)  # noqa: SLF001

    assert not window._write_fields_chosen()  # noqa: SLF001


_PACED_PHOTOS = 3


def test_worker_rests_between_photos_when_paced(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paced run rests at photo boundaries and tells the window each time."""
    _stub_generation(monkeypatch)
    rests: list[str] = []
    paths = [Path(f"/photo{i}.jpg") for i in range(_PACED_PHOTOS)]
    # Every window is over as soon as it starts, so each photo boundary earns a rest. The rests
    # themselves are near-instant: half of nothing worked is nothing owed.
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        paths,
        options=GenerateOptions(pacing=PacingPlan(duty_percent=50, work_seconds=0.0)),
    )
    proposals: list[Proposal] = []
    worker.file_done.connect(proposals.append)
    worker.resting.connect(rests.append)

    worker.run()

    assert len(proposals) == _PACED_PHOTOS
    # One announcement per rest, each followed by the blank that hands the status line back.
    assert rests.count("") == _PACED_PHOTOS - 1
    assert any("works 50%" in text for text in rests)


def test_worker_stop_wakes_a_run_that_is_resting(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop ends the rest at once instead of leaving the user to wait it out."""
    _stub_generation(monkeypatch)
    paths = [Path("/a.jpg"), Path("/b.jpg")]
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        paths,
        options=GenerateOptions(
            pacing=PacingPlan(duty_percent=1, work_seconds=0.0, max_pause_seconds=3600),
        ),
    )
    proposals: list[Proposal] = []
    worker.file_done.connect(proposals.append)
    worker.resting.connect(lambda _text: worker.stop())

    worker.run()

    # The first photo went through; the rest that followed was cancelled, so the second never ran.
    assert len(proposals) == 1


def test_worker_without_pacing_never_rests(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run with no plan never rests, so nothing new sits between two photos."""
    _stub_generation(monkeypatch)
    rests: list[str] = []
    worker = workers.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg"), Path("/b.jpg")])
    worker.resting.connect(rests.append)

    worker.run()

    assert rests == []


def test_window_paces_a_run_from_the_config_file(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A [pacing] table paces the window's runs, exactly as it paces the CLI's."""
    config = tmp_path / "config.toml"
    config.write_text("[pacing]\nduty = 70\nbattery_floor = 30\n", encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))

    window = gui.MainWindow()
    try:
        assert window._pacing_dialog.plan.duty_percent == 70  # noqa: SLF001, PLR2004 - window state.
        assert window._pacing_dialog.plan.battery_floor == 30  # noqa: SLF001, PLR2004 - window state.
    finally:
        window.close()


@pytest.mark.parametrize(
    ("label", "dialog"),
    [
        ("Keyword Rules...", "_keyword_rules_dialog"),
        ("Copyright...", "_copyright_dialog"),
        ("Pacing...", "_pacing_dialog"),
    ],
)
def test_settings_menu_entries_open_their_dialogs(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    dialog: str,
) -> None:
    """Each Settings entry reaches the window's own dialog, the one that holds the setting."""
    opened: list[str] = []
    monkeypatch.setattr(getattr(window, dialog), "exec", lambda: opened.append(dialog))
    actions = window._settings_menu.actions()  # noqa: SLF001

    next(action for action in actions if action.text() == label).trigger()

    assert opened == [dialog]


def test_pacing_dialog_edits_the_plan_the_next_run_follows(window: gui.MainWindow) -> None:
    """Editing a spin box changes the plan, so the next Generate rests by it."""
    window._pacing_dialog._duty_spin.setValue(70)  # noqa: SLF001 - the dialog's own widget.
    window._pacing_dialog._battery_floor_spin.setValue(30)  # noqa: SLF001 - the dialog's own widget.
    window._pacing_dialog._work_minutes_spin.setValue(5.0)  # noqa: SLF001 - the dialog's own widget.

    plan = window._pacing_dialog.plan  # noqa: SLF001 - window state.
    assert plan.duty_percent == 70  # noqa: PLR2004 - the value under test.
    assert plan.battery_floor == 30  # noqa: PLR2004 - the value under test.
    assert plan.work_seconds == 300  # noqa: PLR2004 - five minutes, in seconds.
    assert plan.rests is True


def test_pacing_dialog_starts_from_the_config_file(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spin boxes open on what the config file says, not on the built-in defaults."""
    config = tmp_path / "config.toml"
    config.write_text("[pacing]\nduty = 80\nwork_minutes = 2.5\n", encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))

    paced = gui.MainWindow()
    try:
        assert paced._pacing_dialog._duty_spin.value() == 80  # noqa: SLF001, PLR2004 - the dialog's widget.
        assert paced._pacing_dialog._work_minutes_spin.value() == 2.5  # noqa: SLF001, PLR2004 - ditto.
    finally:
        paced.close()


def test_pacing_dialog_greys_out_the_windows_until_something_rests(
    window: gui.MainWindow,
) -> None:
    """
    An unconfigured window shows the rest fields as inert, because nothing rests.

    The two windows describe the shape of a rest, so with a full duty cycle and no battery floor
    they have nothing to shape, and a dialog that let them be edited would imply a run pauses when
    it does not.
    """
    assert window._pacing_dialog._work_minutes_spin.isEnabled() is False  # noqa: SLF001 - the dialog's widget.
    assert window._pacing_dialog._max_pause_spin.isEnabled() is False  # noqa: SLF001 - the dialog's widget.

    window._pacing_dialog._duty_spin.setValue(70)  # noqa: SLF001 - the dialog's widget.

    assert window._pacing_dialog._work_minutes_spin.isEnabled() is True  # noqa: SLF001 - the dialog's widget.
    assert window._pacing_dialog._max_pause_spin.isEnabled() is True  # noqa: SLF001 - the dialog's widget.


def test_pacing_dialog_battery_floor_alone_enables_the_windows(window: gui.MainWindow) -> None:
    """Watching the battery is a reason to rest too, so the windows come alive for it."""
    window._pacing_dialog._battery_floor_spin.setValue(30)  # noqa: SLF001 - the dialog's widget.

    assert window._pacing_dialog._work_minutes_spin.isEnabled() is True  # noqa: SLF001 - the dialog's widget.


def test_pacing_dialog_summary_follows_the_settings(window: gui.MainWindow) -> None:
    """
    The line under the fields describes the plan as it stands, not as it shipped.

    Regression test: it was a fixed sentence saying a run works flat out, which stayed on screen
    after the duty cycle had been turned down and flatly contradicted the fields above it.
    """
    assert "works flat out" in window._pacing_dialog._pacing_preview.text()  # noqa: SLF001 - the dialog's widget.

    window._pacing_dialog._duty_spin.setValue(90)  # noqa: SLF001 - the dialog's widget.

    summary = window._pacing_dialog._pacing_preview.text()  # noqa: SLF001 - the dialog's widget.
    assert "works flat out" not in summary
    assert "Works 90% of the time" in summary


def test_pacing_dialog_keeps_the_resume_level_above_the_floor(window: gui.MainWindow) -> None:
    """
    A resume level at or below the floor cannot be expressed, so it cannot surprise anyone.

    Regression test: both fields accepted 10, and the run then resumed at 25 because the plan
    substitutes a level above the floor. The dialog said one thing and the run did another.
    """
    floor = window._pacing_dialog._battery_floor_spin  # noqa: SLF001 - the dialog's widget.
    resume = window._pacing_dialog._battery_resume_spin  # noqa: SLF001 - the dialog's widget.
    assert resume.isEnabled() is False  # nothing to resume from yet

    floor.setValue(10)

    assert resume.isEnabled() is True
    assert resume.minimum() == 11  # noqa: PLR2004 - one above the floor.
    assert resume.value() == 25  # noqa: PLR2004 - the margin the CLI would have picked.
    assert window._pacing_dialog.plan.resume_percent == 25  # noqa: SLF001, PLR2004 - what the run does.

    # Typing something too low is not possible: the spin box floor holds it up.
    resume.setValue(5)
    assert resume.value() == 11  # noqa: PLR2004 - clamped, not silently reinterpreted.


def test_pacing_dialog_carries_the_resume_level_up_with_the_floor(window: gui.MainWindow) -> None:
    """Raising the floor past the resume level takes the resume level with it."""
    window._pacing_dialog._battery_floor_spin.setValue(10)  # noqa: SLF001 - the dialog's widget.
    window._pacing_dialog._battery_resume_spin.setValue(20)  # noqa: SLF001 - the dialog's widget.

    window._pacing_dialog._battery_floor_spin.setValue(50)  # noqa: SLF001 - the dialog's widget.

    assert window._pacing_dialog._battery_resume_spin.value() == 65  # noqa: SLF001, PLR2004 - floor plus margin.


def test_pacing_dialog_clears_the_resume_level_when_the_floor_goes(window: gui.MainWindow) -> None:
    """Dropping the floor back to zero puts the resume field back to its inert state."""
    window._pacing_dialog._battery_floor_spin.setValue(30)  # noqa: SLF001 - the dialog's widget.

    window._pacing_dialog._battery_floor_spin.setValue(0)  # noqa: SLF001 - the dialog's widget.

    assert window._pacing_dialog._battery_resume_spin.value() == 0  # noqa: SLF001 - nothing to resume from.
    assert window._pacing_dialog._battery_resume_spin.isEnabled() is False  # noqa: SLF001 - the dialog's widget.


def test_pacing_dialog_warns_about_a_floor_near_full_instead_of_moving_it(
    window: gui.MainWindow,
) -> None:
    """
    A floor in the nineties is kept as typed and flagged, not quietly replaced.

    A plugged-in laptop is rarely brim full (optimized charging parks it around 80), so "rest below
    99%" means resting after almost every window. That is worth saying and not worth overruling:
    silently returning a different number is the same trick the resume level used to play.
    """
    floor = window._pacing_dialog._battery_floor_spin  # noqa: SLF001 - the dialog's widget.
    assert floor.maximum() == MAX_BATTERY_FLOOR

    floor.setValue(99)

    assert floor.value() == 99  # noqa: PLR2004 - exactly what was typed.
    assert MIXED_SCRIPT_MARK in window._pacing_dialog._pacing_preview.text()  # noqa: SLF001 - the dialog's widget.


def test_pacing_dialog_does_not_warn_about_an_ordinary_floor(window: gui.MainWindow) -> None:
    """A floor anyone would actually use carries no warning."""
    window._pacing_dialog._battery_floor_spin.setValue(30)  # noqa: SLF001 - the dialog's widget.

    assert MIXED_SCRIPT_MARK not in window._pacing_dialog._pacing_preview.text()  # noqa: SLF001 - the widget.


def test_worker_sends_the_configured_image_size(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The size the model sees comes from the configuration, not from the preview constant.

    Regression test: the window hardcoded its 640 px preview size for generation, so `[inference]
    jpeg_dimensions` was silently ignored and a GUI run and a CLI run over the same photo asked the
    model different questions.
    """
    _stub_generation(monkeypatch)
    sizes: list[int] = []

    def record_size(_path: Path, **kwargs: int) -> SimpleNamespace:
        sizes.append(kwargs["max_size"])
        return SimpleNamespace(data=b"x")

    _patch_gui(monkeypatch, "prepare_image_for_agent", record_size)
    worker = workers.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        options=GenerateOptions(jpeg_dimensions=1024),
    )

    worker.run()

    assert sizes == [1024]


def test_window_reads_the_image_size_from_the_config(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured jpeg_dimensions is what the window's runs use."""
    config = tmp_path / "config.toml"
    config.write_text("[inference]\njpeg_dimensions = 896\n", encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))

    window = gui.MainWindow()
    try:
        assert window._model_image_size == 896  # noqa: SLF001, PLR2004 - window state.
    finally:
        window.close()


def test_the_steps_a_generation_still_owes_keep_the_window_busy(window: gui.MainWindow) -> None:
    """
    The pause before an extra pass, and harmonization, belong to the run that just ended.

    A click or a watch batch landing there used to start a run the chain then carried on inside.
    """
    assert window._busy() is False  # noqa: SLF001
    window._pending_pass = "failed"  # noqa: SLF001
    assert window._busy() is True  # noqa: SLF001
    window._pending_pass = None  # noqa: SLF001
    window._harmonize_thread = object()  # type: ignore[assignment]  # noqa: SLF001
    try:
        assert window._busy() is True  # noqa: SLF001
    finally:
        window._harmonize_thread = None  # noqa: SLF001


def test_an_extra_pass_follows_the_batch_choice_to_skip_the_cache(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After Skip Cache, the retry pass must not replay the cached answer the user skipped."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    item = window._items[str(img)]  # noqa: SLF001
    item.status = FAILED
    window._generated_keys = [str(img)]  # noqa: SLF001
    window._batch_uses_cache = False  # noqa: SLF001
    window._pending_pass = "failed"  # noqa: SLF001 - the retry pass (RERUN_FAILED)
    runs: list[dict[str, object]] = []
    monkeypatch.setattr(window, "_run_generation", lambda _items, **kwargs: runs.append(kwargs))

    window._run_pending_pass()  # noqa: SLF001

    assert runs
    assert runs[0]["use_cache"] is False


def test_cancel_during_the_pause_ends_the_chain(window: gui.MainWindow) -> None:
    """Cancel between the chain's steps runs no pass and no save, and frees the buttons."""
    window._pending_pass = "failed"  # noqa: SLF001
    window._hold_for_chain()  # noqa: SLF001
    assert not window._run_bar.generate_button.isEnabled()  # noqa: SLF001
    assert window._run_bar.cancel_button.isEnabled()  # noqa: SLF001

    window._cancel_generation()  # noqa: SLF001

    assert window._pending_pass is None  # noqa: SLF001
    assert window._busy() is False  # noqa: SLF001
    assert window._run_bar.generate_button.isEnabled()  # noqa: SLF001
    window._run_pending_pass()  # noqa: SLF001 - the timer firing later changes nothing
    assert window._busy() is False  # noqa: SLF001


def test_remove_waits_for_the_run_in_flight(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save writes from its own job list, so Remove waits for it, as Clear does."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    window._pending_pass = "failed"  # noqa: SLF001 - any busy phase will do

    window._remove_items([str(img)])  # noqa: SLF001

    assert str(img) in window._items  # noqa: SLF001
    assert "Cancel the run in flight" in window._status.text()  # noqa: SLF001
    window._pending_pass = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# usage telemetry: runs, sessions, features
# ---------------------------------------------------------------------------


def _capture_beacons(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every event the window hands to telemetry.emit."""
    sent: list[object] = []
    monkeypatch.setattr(telemetry, "emit", lambda event, **_k: sent.append(event))
    return sent


def _drain_generation(window: gui.MainWindow, timeout: float = 10.0) -> None:
    """Pump the event loop until the background generation finishes (or fail on the deadline)."""
    deadline = time.monotonic() + timeout
    while window._thread is not None:  # noqa: SLF001
        assert time.monotonic() < deadline, "the background generation never finished"
        QApplication.processEvents()


def test_a_generation_sends_one_run_beacon_when_it_ends(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate on two photos: one run beacon at the end, with its trigger and its counts."""
    sent = _capture_beacons(monkeypatch)
    _stub_generation(monkeypatch)
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].hint = "The bird is an eagle"  # noqa: SLF001

    window._generate()  # noqa: SLF001
    _drain_generation(window)

    runs = [event for event in sent if isinstance(event, telemetry.RunInfo)]
    assert len(runs) == 1
    run = runs[0]
    assert (run.job, run.trigger, run.batch_size, run.success_count) == ("generate", "batch", 2, 2)
    assert run.outcome == telemetry.OUTCOME_OK
    assert run.gui is not None
    assert run.gui.hinted == 1
    assert run.settings is not None
    assert run.settings.cache is True
    assert window._usage.features["generate"] == 1  # noqa: SLF001


def test_a_batch_save_counts_what_was_changed_by_hand(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save Selected is a run of its own, and its review counts the edited title."""
    sent = _capture_beacons(monkeypatch)
    a = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _stub_save_helper(monkeypatch)
    monkeypatch.setattr(metadata, "write_metadata", lambda *_a, **_k: True)
    _add_dir(window, {"a": a})
    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=a,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="Eagle",
            description="An eagle.",
            keywords=["Eagle"],
        ),
    )
    window._items[str(a)].title = "Bald Eagle"  # noqa: SLF001 - the user's own correction

    window._save_selected()  # noqa: SLF001
    _drain_save(window)

    runs = [event for event in sent if isinstance(event, telemetry.RunInfo)]
    assert [run.job for run in runs] == ["save"]
    assert runs[0].review is not None
    assert (runs[0].review.saved, runs[0].review.edited_title) == (1, 1)


def test_closing_mid_run_reports_the_run_cancelled_and_then_the_session(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A window closed while a run is open sends that run as cancelled before the session."""
    sent: list[object] = []
    monkeypatch.setattr(telemetry, "emit", lambda event, **_k: sent.append(event) or event)
    flushed: list[list[object]] = []
    monkeypatch.setattr(telemetry, "flush", lambda threads: flushed.append(list(threads)))
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    window._start_run("generate", "batch", [window._items[str(a)]])  # noqa: SLF001

    window.close()

    assert [type(event) for event in sent] == [telemetry.RunInfo, telemetry.SessionInfo]
    assert sent[0].outcome == telemetry.OUTCOME_CANCELLED  # type: ignore[attr-defined]
    assert flushed == [sent]


def test_the_launch_beacon_describes_the_screen(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window opening reports its startup time, settings, and display facts."""
    sent = _capture_beacons(monkeypatch)
    window.report_launch(1.25)
    launch = sent[0]
    assert isinstance(launch, telemetry.LaunchInfo)
    assert launch.startup_seconds == 1.25  # noqa: PLR2004 - what was passed
    assert launch.display is not None
    assert launch.display.qt_version
    assert "x" in launch.display.screen
    assert launch.settings is not None


def test_features_are_counted_where_they_are_used(window: gui.MainWindow, tmp_path: Path) -> None:
    """Using a feature counts it, and counts as time in use."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._set_all_checked(checked=False)  # noqa: SLF001
    window._invert_checked()  # noqa: SLF001
    assert window._usage.features["select"] == 2  # noqa: SLF001, PLR2004 - two uses
    assert window._usage.clock.split(time.monotonic() + 1).engaged > 0  # noqa: SLF001


def test_keyboard_input_on_the_list_counts_as_time_in_use(window: gui.MainWindow) -> None:
    """A key press in the photo list reaches the activity clock through the input watcher."""
    from PySide6.QtCore import QEvent  # noqa: PLC0415 - the GUI tests import Qt lazily
    from PySide6.QtGui import QKeyEvent  # noqa: PLC0415

    before = window._usage.clock.split(time.monotonic()).engaged  # noqa: SLF001
    QApplication.sendEvent(
        window._tree,  # noqa: SLF001
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier),
    )
    after = window._usage.clock.split(time.monotonic() + 1).engaged  # noqa: SLF001
    assert after > before


def test_harmonized_keywords_do_not_count_as_edits(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Shoot harmonization rewrites the proposal itself, so a save does not see a manual edit."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=a,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="T",
            description="D",
            keywords=["Ospreys"],
        ),
    )
    window._on_harmonize_done(  # noqa: SLF001
        HarmonizeResult(keywords={str(a): ["Osprey"]}, sessions=1),
    )
    item = window._items[str(a)]  # noqa: SLF001
    assert item.keywords == ["Osprey"]
    assert item.proposed_keywords == ["Osprey"]


def test_a_photo_the_scan_could_not_read_is_not_scanned_again_and_again(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable photo stays unknown, but it does not restart the scan in an endless loop."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    window._stop_scan()  # noqa: SLF001
    window._items[str(img)].known_fields = None  # noqa: SLF001
    window._scan_tried = {str(img)}  # noqa: SLF001 - an earlier scan tried it and read nothing
    _patch_gui(monkeypatch, "MetadataScanWorker", lambda _paths: pytest.fail("scanned again"))

    # The real method: the fixture stubs it out for every other test.
    gui.MainWindow._start_metadata_scan(window)  # noqa: SLF001

    assert window._scan_thread is None  # noqa: SLF001


def test_clear_forgets_which_photos_the_scan_tried(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Photos added again after Clear are read again, not skipped as already tried."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    window._scan_tried = {str(img)}  # noqa: SLF001

    window._clear()  # noqa: SLF001

    assert window._scan_tried == set()  # noqa: SLF001


def test_a_stopped_scan_marks_nothing_as_tried(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop means the photos it had not reached were never tried, so a later scan reads them."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    window._items[str(img)].known_fields = None  # noqa: SLF001
    window._scan_batch = {str(img)}  # noqa: SLF001
    window._scan_cancelled = True  # noqa: SLF001

    window._on_scan_finished()  # noqa: SLF001

    assert str(img) not in window._scan_tried  # noqa: SLF001


def test_a_finished_scan_marks_what_it_could_not_read(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan that ran to its end and still read nothing for a photo leaves it out of the next."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    window._items[str(img)].known_fields = None  # noqa: SLF001
    window._scan_batch = {str(img)}  # noqa: SLF001

    window._on_scan_finished()  # noqa: SLF001

    assert str(img) in window._scan_tried  # noqa: SLF001


def test_a_photo_added_again_while_the_scan_ran_is_read_again(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan in flight skips a removed photo, so the finish must not count it as tried."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    key = str(img)
    window._scan_batch = {key}  # noqa: SLF001 - the scan in flight had it

    window._forget_items([key])  # noqa: SLF001 - removed mid-scan...
    _add_dir(window, {"a": img})  # ...and added back before the scan ended
    window._items[key].known_fields = None  # noqa: SLF001
    window._on_scan_finished()  # noqa: SLF001

    assert key not in window._scan_tried  # noqa: SLF001


def test_a_photo_undone_while_the_scan_ran_is_read_again(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the scan in flight read of a photo an undo then rewrote is stale, not a fresh try."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    key = str(img)
    window._scan_batch = {key}  # noqa: SLF001

    window._on_undone([key])  # noqa: SLF001
    _scan_reports(window, {"a.jpg": {FIELD_TITLE}})  # read before the undo, delivered after it
    window._on_scan_finished()  # noqa: SLF001

    assert window._items[key].known_fields is None  # noqa: SLF001
    assert key not in window._scan_tried  # noqa: SLF001


def test_uncheck_already_tagged_settles_when_only_unreadable_photos_are_left(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo no scan can read must not keep the pick waiting: it settles on what is known."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    window._items[str(img)].known_fields = None  # noqa: SLF001
    window._scan_tried = {str(img)}  # noqa: SLF001

    window._deselect_tagged(select_menu._TAGGED_PRESETS[0][1])  # noqa: SLF001

    assert window._scan_pending is None  # noqa: SLF001
    assert "Reading metadata" not in window._status.text()  # noqa: SLF001
