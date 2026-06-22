# mypy: ignore-errors
"""
Headless tests for the PySide6 desktop GUI.

Skipped entirely when PySide6 is absent (CLI-only setups and CI, which do not install the optional
``[gui]`` extra). When present, Qt's ``offscreen`` platform builds real widgets without a display
server, so the tree, the detail pane, and the generation worker can be exercised for real.

The ``# mypy: ignore-errors`` header opts this file out of the zuban type check, for the same reason
gui.py does: PySide6 ships no stubs and is not installed in the lint job, so a strict run would only
see an unresolved-import error here.
"""

import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


pytest.importorskip("PySide6")
# Must be set before the first QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from photo_tagger import gui
from photo_tagger.errors import ProviderError
from photo_tagger.gui_state import FAILED, PROVIDER_LABELS, READY, SAVED, Proposal
from photo_tagger.metadata import ImageContext
from photo_tagger.models import InferenceResult, KeywordSet
from photo_tagger.providers import PROVIDER_NAMES


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """One QApplication for the module (Qt allows only a single instance)."""
    return cast("QApplication", QApplication.instance() or QApplication([]))


@pytest.fixture
def window(qapp: QApplication) -> Iterator[gui.MainWindow]:
    """Yield a fresh main window, closing it (and its thread) after each test."""
    win = gui.MainWindow()
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


def _stub_reads(monkeypatch: pytest.MonkeyPatch, *, keywords: list[str]) -> None:
    """Patch the exiftool-backed reads so tests need no exiftool binary."""
    monkeypatch.setattr(gui, "read_caption", lambda _p: ("Old Title", "Old caption."))
    monkeypatch.setattr(
        gui,
        "read_image_context",
        lambda _p: ImageContext(existing_keywords=KeywordSet(subject=keywords)),
    )


def _add_dir(window: gui.MainWindow, files: dict[str, Path]) -> None:
    """Set jpg extensions and add the directory the files live in."""
    folder = next(iter(files.values())).parent
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([folder])  # noqa: SLF001


def _select(window: gui.MainWindow, item: QTreeWidgetItem | None) -> None:
    """Select a tree item, asserting it was actually found first."""
    assert item is not None
    window._tree.setCurrentItem(item)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------


def test_window_builds_with_capitalized_providers(window: gui.MainWindow) -> None:
    """The combo shows capitalized labels but carries the internal names as data."""
    assert "Photo Tagger" in window.windowTitle()
    combo = window._provider  # noqa: SLF001
    names = [combo.itemData(i) for i in range(combo.count())]
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert names == list(PROVIDER_NAMES)
    assert labels == [PROVIDER_LABELS[n] for n in PROVIDER_NAMES]
    assert "LM Studio" in labels


def test_provider_name_returns_internal_value(window: gui.MainWindow) -> None:
    """Selecting a label resolves back to the internal provider name."""
    window._provider.setCurrentIndex(list(PROVIDER_NAMES).index("openai"))  # noqa: SLF001
    assert window._provider_name() == "openai"  # noqa: SLF001


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
    monkeypatch.setattr(gui, "get_backend", lambda _name: fake)
    window._refresh_models()  # noqa: SLF001
    items = [window._model.itemText(i) for i in range(window._model.count())]  # noqa: SLF001
    assert items[0] == "qwen3-vl-30b"
    assert "llama3" in items


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
    assert bool(top.data(0, gui._IS_DIR_ROLE))  # noqa: SLF001 - the top node is a folder
    # One file leaf and one "sub" folder under the top.
    kinds = {bool(top.child(i).data(0, gui._IS_DIR_ROLE)) for i in range(top.childCount())}  # noqa: SLF001
    assert kinds == {True, False}


def test_unchecking_folder_deselects_descendants(window: gui.MainWindow, tmp_path: Path) -> None:
    """Unchecking the top folder deselects every photo beneath it."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "b.jpg")})
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    top.setCheckState(0, Qt.CheckState.Unchecked)
    assert all(not item.selected for item in window._items.values())  # noqa: SLF001


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
    leaf = window._leaf_for(a)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001
    assert str(a) not in window._items  # noqa: SLF001
    assert len(window._items) == 1  # noqa: SLF001


def test_clear_empties_everything(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clear drops all items and the tree."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._clear()  # noqa: SLF001
    assert window._items == {}  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001


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
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001 - triggers _on_current_changed

    assert window._existing_title.text() == "Old Title"  # noqa: SLF001
    assert "Beach" in window._existing_keywords.toPlainText()  # noqa: SLF001
    assert window._title.text() == "Old Title"  # noqa: SLF001
    assert window._keywords.toPlainText() == "Beach"  # noqa: SLF001


def test_hierarchy_preview_updates_from_keywords(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing the keywords field refreshes the resulting-hierarchy preview."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._overwrite.setChecked(True)  # noqa: SLF001
    window._keywords.setPlainText("Duck<Bird<Animal")  # noqa: SLF001 - triggers textChanged
    assert "Animal|Bird|Duck" in window._hierarchy.toPlainText()  # noqa: SLF001


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

    monkeypatch.setattr(gui, "write_metadata", fake_write)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._keywords.setPlainText("Eagle\nSky")  # noqa: SLF001
    window._save_current()  # noqa: SLF001

    assert window._items[str(img)].status == SAVED  # noqa: SLF001
    assert captured["title"] == "New Title"


def test_selecting_shows_metadata_source(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Source field reports where the existing metadata was read from."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    monkeypatch.setattr(gui, "read_metadata_sources", lambda _p: ["XMP sidecar"])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    assert window._existing_source.text() == "XMP sidecar"  # noqa: SLF001


def test_save_selected_writes_only_checked_generated_items(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save selected writes checked photos with a proposal; an unchecked one is skipped."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    written: list[str] = []
    monkeypatch.setattr(
        gui,
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
    assert written == ["a.jpg"]
    assert window._items[str(a)].status == SAVED  # noqa: SLF001


def test_save_marks_failed_when_write_fails(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed write_metadata sets the item status to failed."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    monkeypatch.setattr(gui, "write_metadata", lambda *_a, **_k: False)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._save_current()  # noqa: SLF001
    assert window._items[str(img)].status == FAILED  # noqa: SLF001


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
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)  # no background thread
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - the folder node

    assert window._right.currentIndex() == gui._PAGE_GRID  # noqa: SLF001
    assert window._grid.count() == 2  # noqa: SLF001, PLR2004 - two photos in the folder


def test_clicking_a_thumbnail_opens_the_detail(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activating a grid item selects that photo and shows its detail page."""
    a = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    grid_item = window._grid.item(0)  # noqa: SLF001

    window._on_thumb_activated(grid_item)  # noqa: SLF001

    assert window._right.currentIndex() == gui._PAGE_DETAIL  # noqa: SLF001
    assert window._current is window._items[str(a)]  # noqa: SLF001


def test_thumbnail_worker_emits_bytes(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker decodes each path to bytes and stops when asked."""
    monkeypatch.setattr(
        gui,
        "prepare_image_for_agent",
        lambda *_a, **_k: SimpleNamespace(data=b"jpeg-bytes"),
    )
    emitted: list[tuple[str, bytes]] = []
    worker = gui.ThumbnailWorker([Path("/a.jpg"), Path("/b.jpg")])
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
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001
    window._on_thumb_ready(str(a), a.read_bytes())  # noqa: SLF001
    assert str(a) in window._thumb_cache  # noqa: SLF001


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------


def _stub_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gui, "create_agent", lambda *_a, **_k: object())
    monkeypatch.setattr(
        gui,
        "read_image_context",
        lambda _p: ImageContext(existing_keywords=KeywordSet(subject=["Beach"])),
    )
    monkeypatch.setattr(gui, "read_caption", lambda _p: ("Old", "Old caption."))
    monkeypatch.setattr(
        gui,
        "prepare_image_for_agent",
        lambda *_a, **_k: SimpleNamespace(data=b"x"),
    )
    monkeypatch.setattr(
        gui,
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
    worker = gui.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
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

    monkeypatch.setattr(gui, "create_agent", boom)
    failed: list[str] = []
    worker = gui.GenerateWorker("openai", "m", None, [Path("/a.jpg"), Path("/b.jpg")])
    worker.file_failed.connect(lambda path, _msg: failed.append(path))

    worker.run()

    assert failed == ["/a.jpg", "/b.jpg"]


def test_worker_reports_a_single_file_failure(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-photo error is reported without aborting the batch."""
    _stub_generation(monkeypatch)

    def boom(*_a: object, **_k: object) -> object:
        msg = "decode failed"
        raise ValueError(msg)

    monkeypatch.setattr(gui, "prepare_image_for_agent", boom)
    failed: list[str] = []
    worker = gui.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
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
    _select(window, window._leaf_for(a))  # noqa: SLF001 - open "a"

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items: captured.append([i.path for i in items]),
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
        lambda items: captured.append([i.path for i in items]),
    )
    window._generate()  # noqa: SLF001

    assert captured == [[a]]


def test_generate_current_needs_an_open_photo(window: gui.MainWindow) -> None:
    """With no photo open, Generate this photo nags instead of starting a run."""
    window._generate_current()  # noqa: SLF001
    assert window._thread is None  # noqa: SLF001 - no generation started
    assert "Open a photo" in window._status.text()  # noqa: SLF001


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
