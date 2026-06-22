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
from photo_tagger.gui_state import FAILED, PROVIDER_LABELS, READY, SAVED, WORKING, Proposal
from photo_tagger.metadata import FIELD_DESCRIPTION, FIELD_KEYWORDS, FIELD_TITLE, ImageContext
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


def _check_state(window: gui.MainWindow, path: Path) -> Qt.CheckState:
    """Return the rendered checkbox state of *path*'s tree leaf (asserting it exists)."""
    leaf = window._leaf_for(path)  # noqa: SLF001
    assert leaf is not None
    return leaf.checkState(0)


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
# API key field
# ---------------------------------------------------------------------------


def test_api_key_value_strips_and_blank_is_none(window: gui.MainWindow) -> None:
    """A typed key is trimmed; a blank field means "use the env var" (None)."""
    window._api_key.setText("  sk-typed  ")  # noqa: SLF001
    assert window._api_key_value() == "sk-typed"  # noqa: SLF001
    window._api_key.setText("   ")  # noqa: SLF001
    assert window._api_key_value() is None  # noqa: SLF001


def test_api_key_field_is_masked(window: gui.MainWindow) -> None:
    """The key field hides its contents so a shoulder-surfer cannot read it."""
    from PySide6.QtWidgets import QLineEdit  # noqa: PLC0415

    assert window._api_key.echoMode() == QLineEdit.EchoMode.Password  # noqa: SLF001


def test_refresh_models_passes_typed_api_key(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key typed into the field is handed to the backend's key resolution."""
    window._api_key.setText("sk-refresh")  # noqa: SLF001
    seen: dict[str, str | None] = {}
    fake = SimpleNamespace(
        default_base_url="http://localhost/v1",
        resolve_api_key=lambda key: seen.setdefault("key", key),
        list_models=lambda _base, _key: ["m"],
    )
    monkeypatch.setattr(gui, "get_backend", lambda _name: fake)
    window._refresh_models()  # noqa: SLF001
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

    monkeypatch.setattr(gui, "create_agent", fake_create_agent)
    worker = gui.GenerateWorker("openai", "m", None, [Path("/a.jpg")], api_key="sk-worker")
    worker.file_failed.connect(lambda *_a: None)
    worker.run()
    assert captured["api_key"] == "sk-worker"


def test_run_generation_builds_worker_with_typed_api_key(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting a run hands the typed key to the worker it spawns."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)  # so the background worker does no real I/O
    _add_dir(window, {"a": img})
    window._api_key.setText("sk-run")  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._api_key == "sk-run"  # noqa: SLF001
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
# Deselect: skip already-tagged, skip from a list (CLI --skip-tagged/--skip-from)
# ---------------------------------------------------------------------------


def _stub_field_presence(monkeypatch: pytest.MonkeyPatch, by_name: dict[str, set[str]]) -> None:
    """Patch find_field_presence to report fields per filename, on the passed-in path objects."""
    monkeypatch.setattr(
        gui,
        "find_field_presence",
        lambda paths: {p: set(by_name.get(p.name, set())) for p in paths},
    )


def test_deselect_tagged_field_aware_keeps_keyword_only_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Title and description' skips photos that have both, keeping keyword-only ones selected."""
    a = _jpeg(tmp_path / "a.jpg")  # already has a title and a description
    b = _jpeg(tmp_path / "b.jpg")  # only keywords -> must stay selected for description generation
    _add_dir(window, {"a": a, "b": b})
    _stub_field_presence(
        monkeypatch,
        {"a.jpg": {FIELD_TITLE, FIELD_DESCRIPTION}, "b.jpg": {FIELD_KEYWORDS}},
    )

    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION}),
        match_all=True,
        phrase="a title and a description",
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'any metadata' criterion deselects a photo that has even one indicator field."""
    a = _jpeg(tmp_path / "a.jpg")  # keywords only
    b = _jpeg(tmp_path / "b.jpg")  # nothing
    _add_dir(window, {"a": a, "b": b})
    _stub_field_presence(monkeypatch, {"a.jpg": {FIELD_KEYWORDS}})

    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}),
        match_all=False,
        phrase="any metadata",
    )

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001


def test_deselect_tagged_reports_when_none_match(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no checked photo has the field, the action says so instead of 'Deselected 0'."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    _stub_field_presence(monkeypatch, {})  # no photo has any field

    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_DESCRIPTION}),
        match_all=True,
        phrase="a description",
    )

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "No checked photos have a description" in window._status.text()  # noqa: SLF001


def test_tagged_presets_all_run_without_error(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every menu preset is wired to a real criterion the deselect handler accepts."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    _stub_field_presence(monkeypatch, {})
    for _label, required, match_all, phrase in gui._TAGGED_PRESETS:  # noqa: SLF001
        window._deselect_tagged(required, match_all=match_all, phrase=phrase)  # noqa: SLF001
        assert phrase in window._status.text()  # noqa: SLF001


def test_deselect_tagged_with_no_photos_nags(window: gui.MainWindow) -> None:
    """With nothing added, the action reports it instead of calling exiftool."""
    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_TITLE}),
        match_all=True,
        phrase="a title",
    )
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_already_tagged_menu_actions_route_to_each_preset(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each menu entry triggers _deselect_tagged with its own preset (no late-binding bug)."""
    from PySide6.QtWidgets import QPushButton  # noqa: PLC0415

    # The "Already tagged" button is the only one carrying a popup menu.
    menu_buttons = [b for b in window.findChildren(QPushButton) if b.menu() is not None]
    assert len(menu_buttons) == 1
    actions = menu_buttons[0].menu().actions()
    assert len(actions) == len(gui._TAGGED_PRESETS)  # noqa: SLF001

    captured: list[tuple[frozenset[str], bool, str]] = []
    monkeypatch.setattr(
        window,
        "_deselect_tagged",
        lambda req, *, match_all, phrase: captured.append((req, match_all, phrase)),
    )
    for action in actions:
        action.trigger()

    assert captured == [(r, m, p) for _label, r, m, p in gui._TAGGED_PRESETS]  # noqa: SLF001


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
    monkeypatch.setattr(
        gui,
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

    monkeypatch.setattr(gui, "write_metadata", fake_write)
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
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._description.setPlainText("New description.")  # noqa: SLF001
    window._keywords.setPlainText("Eagle")  # noqa: SLF001
    window._write_keywords.setChecked(False)  # noqa: SLF001

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
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._write_title.setChecked(False)  # noqa: SLF001

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
    monkeypatch.setattr(gui, "write_metadata", lambda *_a, **_k: wrote.append(1) or True)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    for checkbox in (window._write_title, window._write_description, window._write_keywords):  # noqa: SLF001
        checkbox.setChecked(False)

    window._save_current()  # noqa: SLF001

    assert wrote == []
    assert "at least one field" in window._status.text()  # noqa: SLF001


def test_unchecking_write_keywords_disables_overwrite_and_blanks_diff(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning keywords off grays out Overwrite and shows the diff is moot."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001

    window._write_keywords.setChecked(False)  # noqa: SLF001 - fires _on_write_keywords_toggled

    assert not window._overwrite.isEnabled()  # noqa: SLF001
    assert "keywords will not be written" in window._diff.toPlainText().lower()  # noqa: SLF001


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
    _select(window, window._leaf_for(img))  # noqa: SLF001 - open the photo

    window._on_file_failed(str(img), "model unreachable")  # noqa: SLF001

    item = window._items[str(img)]  # noqa: SLF001
    assert item.status == FAILED
    assert item.error == "model unreachable"
    assert not window._error_banner.isHidden()  # noqa: SLF001
    assert "model unreachable" in window._error_banner.text()  # noqa: SLF001
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.toolTip(1) == "model unreachable"


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
    _select(window, window._leaf_for(a))  # noqa: SLF001
    window._on_file_failed(str(a), "boom")  # noqa: SLF001
    assert not window._error_banner.isHidden()  # noqa: SLF001

    _select(window, window._leaf_for(b))  # noqa: SLF001 - healthy photo
    assert window._error_banner.isHidden()  # noqa: SLF001


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

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items: captured.append([i.path for i in items]),
    )
    window._retry_failed()  # noqa: SLF001

    assert captured == [[a]]


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
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    assert not window._error_banner.isHidden()  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    # Checked synchronously, before the worker thread's queued signals are processed: the
    # pre-run bookkeeping marks the photo working and clears its stale failure banner.
    assert window._items[str(img)].status == WORKING  # noqa: SLF001
    assert window._error_banner.isHidden()  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_open_logs_creates_the_folder_and_reveals_it(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open logs makes sure the folder exists and hands it to the OS file browser."""
    folder = tmp_path / "logs"
    monkeypatch.setattr(gui, "_LOG_FOLDER", folder)
    opened: list[str] = []
    monkeypatch.setattr(
        gui,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened.append(url.toLocalFile()) or True),
    )
    window._open_logs()  # noqa: SLF001
    assert folder.is_dir()
    assert opened == [str(folder)]


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
