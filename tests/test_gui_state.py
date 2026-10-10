"""Tests for the Qt-free GUI helpers (no PySide6, no display required)."""

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from photo_tagger.errors import ConfigFileError
from photo_tagger.gui_state import (
    _MAX_LISTED_DROPPED,
    ADDED,
    BADGE_FAILED,
    BADGE_METADATA,
    BADGE_SAVED,
    BADGE_SIDECAR,
    BADGE_UNSAVED,
    DEFAULT_GUI_EXTENSIONS,
    FAILED,
    FILTER_ALL,
    FILTER_FAILED,
    FILTER_GENERATED,
    FILTER_PENDING,
    FILTER_SAVED,
    FILTER_SELECTED,
    FILTER_UNTAGGED,
    MAX_ZOOM,
    MIN_ZOOM,
    OUTPUT_LANGUAGE_SUGGESTIONS,
    PENDING,
    READY,
    REMOVED,
    SAVED,
    SELECT_CHECK,
    SELECT_ONLY,
    SELECT_UNCHECK,
    SORT_NAME,
    SORT_STATUS,
    SORT_TAGGED,
    SORT_TYPE,
    STATUS_SORT_ORDER,
    TOOLTIP_WIDTH,
    UNCHANGED,
    WORKING,
    ZOOM_STEP,
    FolderNode,
    GuiConfigValues,
    HarmonizeResult,
    Location,
    NavigationHistory,
    PhotoItem,
    Proposal,
    SaveOptions,
    SelectionChange,
    WatchSettings,
    anchored_scroll,
    apply_proposal,
    apply_selection,
    apply_vocabulary,
    build_save_job,
    build_tree,
    caption_source_note,
    chain_to_display,
    clamp_zoom,
    config_text_with_language,
    config_text_with_output_language,
    config_toml_text,
    deselect_paths,
    ensure_path_dirs,
    estimate_remaining,
    expand_inputs,
    extension_counts,
    fields_written,
    file_dialog_name_filters,
    file_type_label,
    filter_photos,
    fit_zoom,
    flat_labels,
    format_duration,
    format_existing_keywords,
    group_by_parent,
    harmonize_sessions,
    harmonize_summary,
    hierarchy_preview,
    hierarchy_tree_text,
    journal_label,
    journal_time,
    keyword_diff,
    keyword_lines,
    keywords_to_save,
    keywords_to_text,
    load_vocabulary_file,
    location_crumb,
    login_shell_path,
    matches_extension,
    matches_name_pattern,
    merged_config_text,
    navigation_shortcuts,
    new_paths,
    normalize_keyword_lines,
    parse_keyword_lines,
    paths_matching_fields,
    paths_under,
    photo_item_to_report_row,
    photo_matches_filter,
    photo_sort_key,
    progress_timing_text,
    rank_vision_models,
    record_dropped_terms,
    reveal_command,
    reveal_label,
    search_summary,
    sort_photos,
    source_label,
    status_sort_rank,
    status_summary,
    step_zoom,
    tagged_legend,
    tagged_summary,
    tagged_tooltip,
    thumb_badges,
    tooltip,
    undo_action_label,
    undo_summary,
    vocabulary_status,
    vocabulary_summary,
    watch_status_text,
    wrap_tooltip,
    zoom_label,
)
from photo_tagger.i18n import activate
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    CaptionValue,
)
from photo_tagger.models import KeywordSet
from photo_tagger.pipeline import MAX_TRACKED_DROPPED_TERMS
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES
from photo_tagger.undo import CHANGED, DELETED, RESTORED, UndoResult
from photo_tagger.vocabulary import Vocabulary


def test_expand_inputs_walks_folders_and_keeps_files(tmp_path: Path) -> None:
    """Directories are extension-filtered and recursed; explicit files pass through."""
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "b.cr3").write_text("x")
    (tmp_path / "skip.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpg").write_text("x")

    flat = expand_inputs([tmp_path], "jpg", recursive=False)
    assert flat == [(tmp_path / "a.jpg").resolve()]

    deep = expand_inputs([tmp_path], "jpg,cr3", recursive=True)
    assert set(deep) == {
        (tmp_path / "a.jpg").resolve(),
        (tmp_path / "b.cr3").resolve(),
        (sub / "c.jpg").resolve(),
    }


def test_expand_inputs_empty_extensions_yields_nothing(tmp_path: Path) -> None:
    """A blank extension string returns no files rather than raising."""
    (tmp_path / "a.jpg").write_text("x")
    assert expand_inputs([tmp_path], "  ", recursive=False) == []


def test_new_paths_filters_already_present() -> None:
    """Only paths not already tracked are returned, in order, de-duplicated."""
    existing = [Path("/a.jpg"), Path("/b.jpg")]
    found = [Path("/b.jpg"), Path("/c.jpg"), Path("/c.jpg"), Path("/d.jpg")]
    assert new_paths(existing, found) == [Path("/c.jpg"), Path("/d.jpg")]


def test_group_by_parent_groups_and_preserves_order() -> None:
    """Files group under their parent folder in first-seen order."""
    paths = [Path("/x/a.jpg"), Path("/y/b.jpg"), Path("/x/c.jpg")]
    assert group_by_parent(paths) == [
        (Path("/x"), [Path("/x/a.jpg"), Path("/x/c.jpg")]),
        (Path("/y"), [Path("/y/b.jpg")]),
    ]


def test_parse_keyword_lines_strips_and_drops_blanks() -> None:
    """One keyword per line; whitespace trimmed and empty lines removed."""
    assert parse_keyword_lines("  Bird \n\n Sky\n   \nForest") == ["Bird", "Sky", "Forest"]


def test_keywords_to_text_round_trips() -> None:
    """Rendering then parsing returns the same list."""
    keywords = ["Bird", "Sky", "Forest"]
    assert parse_keyword_lines(keywords_to_text(keywords)) == keywords


def test_keywords_to_save_merges_with_existing() -> None:
    """Without overwrite, edited keywords merge into the existing set."""
    existing = KeywordSet(subject=["Beach"], weighted=["Beach"])
    merged = keywords_to_save(existing, ["Bird"], overwrite=False)
    assert merged.subject == ["Beach", "Bird"]


def test_keywords_to_save_overwrite_drops_existing() -> None:
    """With overwrite, the existing keywords are discarded before merging."""
    existing = KeywordSet(subject=["Beach"], weighted=["Beach"])
    merged = keywords_to_save(existing, ["Bird"], overwrite=True)
    assert merged.subject == ["Bird"]


def test_apply_proposal_seeds_the_editable_copy() -> None:
    """A proposal fills existing metadata and seeds the editable title/desc/keywords."""
    item = PhotoItem(path=Path("/a.jpg"))
    proposal = Proposal(
        path=Path("/a.jpg"),
        existing_title="Old",
        existing_description="Old caption.",
        existing_keywords=KeywordSet(subject=["Beach"]),
        title="Golden Eagle",
        description="An eagle soars.",
        keywords=["Eagle", "Sky"],
        camera_info={"EXIF:Model": "Canon EOS R5"},
        gps_position="53 N, 9 E",
        input_tokens=10,
        total_tokens=15,
        seconds=0.4,
    )
    apply_proposal(item, proposal)
    assert item.status == READY
    assert item.has_proposal is True
    assert item.loaded is True
    assert item.existing_title == "Old"
    assert item.existing_keywords.subject == ["Beach"]
    assert item.title == "Golden Eagle"
    assert item.keywords == ["Eagle", "Sky"]
    # The read context and usage carried by the proposal land on the item for the CSV export.
    assert item.camera_info == {"EXIF:Model": "Canon EOS R5"}
    assert item.gps_position == "53 N, 9 E"
    assert item.total_tokens == 15  # noqa: PLR2004 - fixture value
    # The editable copy is independent of the proposal's list.
    item.keywords.append("Extra")
    assert proposal.keywords == ["Eagle", "Sky"]


def test_photo_item_to_report_row_merges_keywords_and_reads_context() -> None:
    """The GUI row reflects a Save (merged keywords) plus existing metadata and EXIF."""
    item = PhotoItem(
        path=Path("/photos/a.jpg"),
        status=READY,
        title="Golden Eagle",
        description="An eagle soars.",
        keywords=["Eagle", "Bird<Animal"],
        existing_title="Old",
        existing_description="Old caption.",
        existing_keywords=KeywordSet(subject=["Beach"]),
        camera_info={"EXIF:Model": "Canon EOS R5", "EXIF:LensModel": "RF 100mm"},
        location_tags={"IPTC:City": "Berlin", "IPTC:Country-PrimaryLocationName": "Germany"},
        gps_position="53 N, 9 E",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        seconds=0.4,
    )

    row = photo_item_to_report_row(item, overwrite=False).as_dict()

    assert row["filename"] == "a.jpg"
    assert row["status"] == READY
    assert row["title"] == "Golden Eagle"
    # Merge (not overwrite): the existing "Beach" survives alongside the new flat keyword.
    assert "Beach" in row["keywords"]
    assert "Eagle" in row["keywords"]
    # The "Bird<Animal" entry becomes a Lightroom hierarchy path, not a flat keyword.
    assert "Animal|Bird" in row["hierarchical_keywords"]
    assert row["existing_keywords"] == "Beach"
    assert row["existing_title"] == "Old"
    assert row["camera_model"] == "Canon EOS R5"
    assert row["lens_model"] == "RF 100mm"
    assert row["city"] == "Berlin"
    assert row["country"] == "Germany"
    assert row["total_tokens"] == "15"
    # The GUI has no cache or retry pass, so those columns stay blank.
    assert row["from_cache"] == ""
    assert row["retry"] == ""


def test_photo_item_to_report_row_overwrite_drops_existing_keywords() -> None:
    """With overwrite, the existing keywords are replaced rather than merged in."""
    item = PhotoItem(
        path=Path("/photos/a.jpg"),
        keywords=["Eagle"],
        existing_keywords=KeywordSet(subject=["Beach"]),
    )
    row = photo_item_to_report_row(item, overwrite=True).as_dict()
    assert row["keywords"] == "Eagle"
    # existing_keywords column still reports what was on the file before the (hypothetical) save.
    assert row["existing_keywords"] == "Beach"


def test_build_tree_groups_files_under_their_folder() -> None:
    """A single folder of files becomes one top node labelled with its absolute path."""
    forest = build_tree([Path("/photos/a.jpg"), Path("/photos/b.jpg")])
    assert len(forest) == 1
    assert forest[0].path == Path("/photos")
    # str(Path(...)) rather than a literal: Windows renders the same path with backslashes.
    assert forest[0].label == str(Path("/photos"))
    assert forest[0].files == [Path("/photos/a.jpg"), Path("/photos/b.jpg")]
    assert forest[0].folders == []


def test_build_tree_nests_subfolders_under_parent() -> None:
    """Subfolders nest under the parent folder with a relative label."""
    forest = build_tree([Path("/p/a.jpg"), Path("/p/sub/b.jpg")])
    assert len(forest) == 1
    top = forest[0]
    assert top.label == str(Path("/p"))
    assert top.files == [Path("/p/a.jpg")]
    assert [f.label for f in top.folders] == ["sub"]
    assert top.folders[0].files == [Path("/p/sub/b.jpg")]


def test_build_tree_collapses_single_child_chains() -> None:
    """A chain of single, file-less folders collapses into one labelled node."""
    forest = build_tree([Path("/p/x/y/b.jpg")])
    assert len(forest) == 1
    assert forest[0].label == str(Path("/p/x/y"))
    assert forest[0].files == [Path("/p/x/y/b.jpg")]


def test_build_tree_keeps_a_branching_single_root_together() -> None:
    """A folder whose subfolders each hold files stays one top node with both children."""
    forest = build_tree([Path("/p/a/x.jpg"), Path("/p/b/y.jpg")])
    assert len(forest) == 1
    assert forest[0].label == str(Path("/p"))
    assert sorted(f.label for f in forest[0].folders) == ["a", "b"]


def test_build_tree_splits_disjoint_roots() -> None:
    """Files under unrelated roots become separate top-level nodes (no '/' wrapper)."""
    forest = build_tree([Path("/r1/x.jpg"), Path("/r2/y.jpg")])
    assert sorted(node.label for node in forest) == [str(Path("/r1")), str(Path("/r2"))]


def test_build_tree_empty() -> None:
    """No paths yields no nodes."""
    assert build_tree([]) == []


def test_paths_under_filters_by_folder() -> None:
    """paths_under keeps only the paths beneath a folder, preserving order."""
    paths = [Path("/a/1.jpg"), Path("/b/2.jpg"), Path("/a/sub/3.jpg")]
    assert paths_under(paths, Path("/a")) == [Path("/a/1.jpg"), Path("/a/sub/3.jpg")]
    assert paths_under(paths, Path("/b")) == [Path("/b/2.jpg")]


# ---------------------------------------------------------------------------
# Navigation history
# ---------------------------------------------------------------------------


def _photo_at(name: str, folder: str = "/shoot") -> Location:
    """Build a photo location, the thing the detail pane shows."""
    return Location(path=Path(folder) / name, is_dir=False)


def _folder_at(path: str) -> Location:
    """Build a folder location, the thing the thumbnail grid shows."""
    return Location(path=Path(path), is_dir=True)


def test_history_starts_empty() -> None:
    """A fresh history has nowhere to go and nothing open."""
    history = NavigationHistory()
    assert history.current is None
    assert history.peek_back() is None
    assert history.peek_forward() is None
    assert history.back() is None
    assert history.forward() is None


def test_history_back_returns_to_the_grid_a_photo_was_opened_from() -> None:
    """The folder grid -> photo -> Back round trip, which is what the arrows are for."""
    history = NavigationHistory()
    grid = _folder_at("/shoot")
    history.visit(grid)
    history.visit(_photo_at("a.jpg"))

    assert history.peek_back() == grid
    assert history.back() == grid
    assert history.current == grid


def test_history_walks_a_trail_of_photos_both_ways() -> None:
    """Back retraces photo by photo, and Forward replays the steps it undid."""
    history = NavigationHistory()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        history.visit(_photo_at(name))

    assert history.back() == _photo_at("b.jpg")
    assert history.back() == _photo_at("a.jpg")
    assert history.back() is None  # nothing before the first photo
    assert history.forward() == _photo_at("b.jpg")
    assert history.forward() == _photo_at("c.jpg")
    assert history.forward() is None


def test_history_visiting_after_going_back_drops_the_forward_trail() -> None:
    """Browsing on from a place stepped back to ends the forward trail there."""
    history = NavigationHistory()
    history.visit(_photo_at("a.jpg"))
    history.visit(_photo_at("b.jpg"))
    history.back()

    history.visit(_photo_at("c.jpg"))

    assert history.peek_forward() is None
    assert history.back() == _photo_at("a.jpg")


def test_history_ignores_revisiting_the_open_place() -> None:
    """Re-recording the place already shown is a no-op, so Back never lands on it twice."""
    history = NavigationHistory()
    history.visit(_photo_at("a.jpg"))
    history.visit(_photo_at("b.jpg"))

    history.visit(_photo_at("b.jpg"))

    assert history.peek_back() == _photo_at("a.jpg")
    assert history.back() == _photo_at("a.jpg")


def test_history_leave_makes_back_return_to_the_place_just_closed() -> None:
    """Emptying the pane keeps the place in the trail, so Back reopens it."""
    history = NavigationHistory()
    history.visit(_photo_at("a.jpg"))

    history.leave()

    assert history.current is None
    assert history.back() == _photo_at("a.jpg")


def test_history_leave_on_an_empty_pane_changes_nothing() -> None:
    """Leaving twice must not stack the same place onto the trail."""
    history = NavigationHistory()
    history.visit(_photo_at("a.jpg"))
    history.leave()
    history.leave()

    assert history.back() == _photo_at("a.jpg")
    assert history.back() is None


def test_history_prune_forgets_places_that_left_the_list() -> None:
    """A removed photo is dropped from both trails, so Back skips over it."""
    history = NavigationHistory()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        history.visit(_photo_at(name))
    history.back()  # b.jpg open, c.jpg ahead

    history.prune(lambda location: location.path.name != "c.jpg")

    assert history.current == _photo_at("b.jpg")
    assert history.peek_forward() is None
    assert history.back() == _photo_at("a.jpg")


def test_history_prune_of_the_open_place_leaves_the_pane_empty() -> None:
    """Removing the open photo empties the pane; Back then lands on the one before it."""
    history = NavigationHistory()
    history.visit(_photo_at("a.jpg"))
    history.visit(_photo_at("b.jpg"))

    history.prune(lambda location: location.path.name != "b.jpg")

    assert history.current is None
    assert history.back() == _photo_at("a.jpg")


def test_history_forgets_the_oldest_place_past_its_limit() -> None:
    """The trail is capped, so a long session cannot grow it without bound."""
    history = NavigationHistory(limit=2)
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        history.visit(_photo_at(name))

    assert history.back() == _photo_at("c.jpg")
    assert history.back() == _photo_at("b.jpg")
    assert history.back() is None  # a.jpg fell off the end


def test_location_crumb_names_a_photo_with_its_folder() -> None:
    """A photo's crumb carries the folder, since file names alone repeat across shoots."""
    assert location_crumb(_photo_at("DSC_0042.NEF", "/pics/Shoot 1")) == "Shoot 1 / DSC_0042.NEF"


def test_navigation_shortcuts_follow_the_platform() -> None:
    """Each platform gets its own idiom: browser keys on macOS, file manager keys elsewhere."""
    assert navigation_shortcuts("darwin") == ("Ctrl+[", "Ctrl+]", "Ctrl+Up")
    assert navigation_shortcuts("linux") == ("Alt+Left", "Alt+Right", "Alt+Up")
    assert navigation_shortcuts("win32") == ("Alt+Left", "Alt+Right", "Alt+Up")


def test_location_crumb_names_a_folder_by_itself() -> None:
    """A folder's crumb is its own name, and the full path when it has none."""
    assert location_crumb(_folder_at("/pics/Shoot 1")) == "Shoot 1"
    # A filesystem root has no name, so the crumb falls back to the path. Windows renders that
    # root as "\", hence str() rather than a "/" literal.
    root = Path("/")
    assert location_crumb(Location(path=root, is_dir=True)) == str(root)


def test_rank_vision_models_surfaces_likely_first_without_dropping_any() -> None:
    """Likely vision models sort first; every input model is still present."""
    models = ["llama3", "qwen3-vl-30b", "gpt-4o", "llava-7b"]
    ranked = rank_vision_models(models)
    assert ranked[:2] == ["qwen3-vl-30b", "llava-7b"]
    assert set(ranked) == set(models)


def test_provider_labels_cover_every_provider() -> None:
    """Every backend name has a display label and they are distinct."""
    assert set(PROVIDER_LABELS) == set(PROVIDER_NAMES)
    assert len(set(PROVIDER_LABELS.values())) == len(PROVIDER_NAMES)


def test_default_gui_extensions_includes_jpg_and_jpeg() -> None:
    """The broad default lists jpg and jpeg separately (matching is not variant-aware)."""
    exts = DEFAULT_GUI_EXTENSIONS.split(",")
    assert "jpg" in exts
    assert "jpeg" in exts


def test_format_existing_keywords_uses_leaf_first_chains() -> None:
    """Hierarchies render once, leaf-first with '<'; only truly flat keywords are listed apart."""
    kw = KeywordSet(
        subject=["Animal", "Bird", "Sky"],
        hierarchical=["Animal|Bird"],
    )
    text = format_existing_keywords(kw)
    assert "Bird<Animal" in text
    assert "Sky" in text.splitlines()
    # The flat copies of the chain segments are folded into the chain line, not repeated.
    assert "Animal" not in text.splitlines()
    assert format_existing_keywords(KeywordSet()) == ""


def test_format_existing_keywords_keeps_only_deepest_chain() -> None:
    """Cumulative Lightroom paths (A|B plus A|B|C) collapse into the single deepest chain."""
    kw = KeywordSet(
        subject=["Animal", "Bird", "Duck"],
        hierarchical=["Animal|Bird", "Animal|Bird|Duck"],
    )
    assert format_existing_keywords(kw) == "Duck<Bird<Animal"


def test_caption_source_note_names_the_file_a_value_came_from() -> None:
    """With one source there is nothing to compare, so the note is just where it was read."""
    note = caption_source_note([CaptionValue("A caption.", SOURCE_SIDECAR)])
    assert note == "from XMP sidecar"
    assert caption_source_note([]) == ""


def test_caption_source_note_names_the_value_a_sidecar_shadows() -> None:
    """
    A caption left inside the photo is named, not hidden.

    photo-tagger writes the sidecar and reads it back first, so the camera's placeholder is still on
    the photo and still what a plain `exiftool photo.dng` prints.
    """
    note = caption_source_note(
        [
            CaptionValue("A man and a woman sit on a beach.", SOURCE_SIDECAR),
            CaptionValue("default", SOURCE_IMAGE),
        ],
    )
    assert note.splitlines() == [
        "from XMP sidecar",
        'shadows "default" in the image file',
    ]


def test_caption_source_note_stays_quiet_when_both_files_agree() -> None:
    """The same value in both files shadows nothing, so there is nothing to report."""
    note = caption_source_note(
        [CaptionValue("Same.", SOURCE_SIDECAR), CaptionValue("Same.", SOURCE_IMAGE)],
    )
    assert note == "from XMP sidecar"


def test_source_label_passes_an_unknown_source_through() -> None:
    """A source name with no label of its own is shown as it came, not dropped."""
    assert source_label(SOURCE_IMAGE) == "image file"
    assert source_label("something else") == "something else"


def test_keyword_lines_sort_by_leaf() -> None:
    """Both columns sort on the rendered line, which is leaf-first, so they can be compared."""
    kw = KeywordSet(
        subject=["Zebra", "Animal", "Bird", "Duck", "Apple"],
        hierarchical=["Animal|Bird|Duck"],
    )
    assert keyword_lines(kw) == ["Apple", "Duck<Bird<Animal", "Zebra"]


def test_normalize_keyword_lines_drops_leaves_a_chain_already_covers() -> None:
    """
    A model returns each leaf twice, bare and in its chain; the field shows it once.

    The bare copies never changed what a save writes (merging expands every chain into its levels),
    so they were noise in a column meant to be read against the one beside it.
    """
    raw = [
        "Beach",
        "Ocean",
        "Sand",
        "Sun Hat",
        "Sun Hat<Headwear<Clothing",
        "Beach<Sandy Area<Outdoor Area",
        "Ocean<Water Body<Natural Feature",
    ]
    assert normalize_keyword_lines(raw) == [
        "Beach<Sandy Area<Outdoor Area",
        "Ocean<Water Body<Natural Feature",
        "Sand",
        "Sun Hat<Headwear<Clothing",
    ]


def test_normalize_keyword_lines_writes_what_the_raw_list_would_have() -> None:
    """Normalizing is a display change only: the saved keyword set has to come out the same."""
    raw = ["Beach", "Sand", "Beach<Sandy Area<Outdoor Area", "Man", "Man<Adult<People"]
    before = keywords_to_save(KeywordSet(), raw, overwrite=True)
    after = keywords_to_save(KeywordSet(), normalize_keyword_lines(raw), overwrite=True)
    assert sorted(after.subject) == sorted(before.subject)
    assert sorted(after.hierarchical) == sorted(before.hierarchical)


def test_normalize_keyword_lines_keeps_the_vocabulary_spelling() -> None:
    """The merge it runs through is the save's own, so a catalog's lower-case term is untouched."""
    assert normalize_keyword_lines(["gegenlicht"], verbatim={"gegenlicht": "gegenlicht"}) == [
        "gegenlicht",
    ]


def test_apply_proposal_normalizes_the_keywords_it_seeds() -> None:
    """What the editable field shows after a generation is the set a save would write."""
    item = PhotoItem(path=Path("/photos/a.jpg"))
    proposal = Proposal(
        path=item.path,
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="T",
        description="D",
        keywords=["Duck", "Bird", "Duck<Bird<Animal"],
    )
    apply_proposal(item, proposal)
    assert item.keywords == ["Duck<Bird<Animal"]


def test_hierarchy_preview_renders_a_guided_tree() -> None:
    """The preview folds the cumulative paths a save writes into one tree with branch guides."""
    preview = hierarchy_preview(KeywordSet(), ["Duck<Bird<Animal"], overwrite=True)
    assert preview == "Animal\n└─ Bird\n   └─ Duck"


def test_hierarchy_tree_text_merges_shared_roots() -> None:
    """Two chains under one root share the root line, with tree-style connectors."""
    text = hierarchy_tree_text(["Animal|Bird", "Animal|Bird|Duck", "Animal|Cat"])
    assert text == "Animal\n├─ Bird\n│  └─ Duck\n└─ Cat"


def test_chain_to_display_reverses_to_leaf_first() -> None:
    """A root-first '|' path flips to the editable field's leaf-first '<' form."""
    assert chain_to_display("Animal|Bird|Duck") == "Duck<Bird<Animal"
    assert chain_to_display("Flat") == "Flat"


def test_reveal_command_per_platform(tmp_path: Path) -> None:
    """MacOS and Windows get a reveal argv; Linux falls back to opening the folder (None)."""
    photo = tmp_path / "a.jpg"
    with patch("photo_tagger.gui_state.shutil.which", lambda name: f"/bin/{name}"):
        assert reveal_command(photo, "darwin") == ["/bin/open", "-R", str(photo)]
        assert reveal_command(photo, "win32") == ["/bin/explorer", f"/select,{photo}"]
        assert reveal_command(photo, "linux") is None


def test_reveal_command_is_none_when_the_browser_is_not_on_path(tmp_path: Path) -> None:
    """
    The browser is resolved here, so the caller opens the folder instead of failing to spawn.

    Leaving the lookup to the OS is also how the current working directory (whichever folder the
    user last opened photos from) gets searched before PATH on Windows.
    """
    with patch("photo_tagger.gui_state.shutil.which", return_value=None):
        assert reveal_command(tmp_path / "a.jpg", "darwin") is None


def test_reveal_label_names_the_platform_browser() -> None:
    """The context-menu label matches each platform's file browser name."""
    assert reveal_label("darwin") == "Reveal in Finder"
    assert reveal_label("win32") == "Show in Explorer"
    assert reveal_label("linux") == "Show in File Manager"


def test_file_type_label_flags_sidecars(tmp_path: Path) -> None:
    """The Type label is the lowercased extension, with +xmp when a sidecar exists."""
    photo = tmp_path / "IMG_0001.CR3"
    photo.write_text("x", encoding="utf-8")
    assert file_type_label(photo) == "cr3"
    (tmp_path / "IMG_0001.xmp").write_text("<x/>", encoding="utf-8")
    assert file_type_label(photo) == "cr3+xmp"


def test_file_dialog_name_filters_puts_configured_types_first() -> None:
    """The default filter is the user's configured types, sorted, deduped, and lowercased."""
    filters = file_dialog_name_filters("jpg, cr3 ,JPG")
    assert filters[0] == "Your file types (*.cr3 *.jpg)"
    assert filters[1].startswith("All known image formats (*.arw ")
    assert "JPEG (*.jpg *.jpeg)" in filters
    assert "Camera Raw (*.arw *.cr2 *.cr3 *.dng *.nef *.orf *.raf *.rw2)" in filters
    assert filters[-1] == "All files (*)"


def test_file_dialog_name_filters_without_configured_types_still_offers_known_formats() -> None:
    """With no parseable extensions, the known-format hints and 'All files' remain."""
    filters = file_dialog_name_filters("  ,, ")
    assert filters[0].startswith("All known image formats (")
    assert "PNG (*.png)" in filters
    assert filters[-1] == "All files (*)"


def test_tagged_summary_letters_and_empty_marker() -> None:
    """Present fields compress to their letters in T/D/K order; none becomes a dash."""
    assert tagged_summary({FIELD_KEYWORDS, FIELD_TITLE}) == "TK"
    assert tagged_summary(set()) == "-"


def test_tagged_legend_maps_every_letter() -> None:
    """The header legend pairs each letter with its field name, in display order."""
    assert tagged_legend() == "T = title, D = description, K = keywords"


def test_tagged_tooltip_spells_out_the_present_letters() -> None:
    """The cell tooltip names only the fields on the file; an empty set says nothing."""
    assert tagged_tooltip({FIELD_KEYWORDS, FIELD_TITLE}) == (
        "Already on the file: T = title, K = keywords"
    )
    assert tagged_tooltip(set()) == "Already on the file: nothing"


def test_tagged_tooltip_is_fully_translated() -> None:
    """The pt_BR catalog covers the whole tooltip, field names included (regression test)."""
    activate("pt_BR")
    try:
        assert tagged_tooltip({FIELD_KEYWORDS, FIELD_TITLE}) == (
            "Já no arquivo: T = título, K = palavras-chave"
        )
    finally:
        activate("en")


def test_wrap_tooltip_breaks_long_text_into_lines() -> None:
    """A long tooltip becomes several lines, none wider than the tooltip column."""
    text = "Let ExifTool save the untouched file as *_original before writing. " * 3
    wrapped = wrap_tooltip(text)
    lines = wrapped.splitlines()
    assert len(lines) > 1
    assert max(len(line) for line in lines) <= TOOLTIP_WIDTH
    assert wrapped.split() == text.split()  # only whitespace changed


def test_wrap_tooltip_leaves_short_text_and_existing_breaks_alone() -> None:
    """Text that already fits is untouched, and hand-written line breaks survive re-wrapping."""
    assert wrap_tooltip("Short enough.") == "Short enough."

    paragraphs = "First line.\n\nSecond line."
    assert wrap_tooltip(paragraphs) == paragraphs
    # Wrapping is idempotent, so a tooltip built from an already-wrapped one does not re-flow.
    once = wrap_tooltip("Wrap me. " * 20)
    assert wrap_tooltip(once) == once


def test_tooltip_translates_formats_and_wraps() -> None:
    """Tooltip() runs the message through gettext, fills placeholders, then wraps the result."""
    activate("pt_BR")
    try:
        text = tooltip(
            "Language of the app itself (menus, buttons, messages). The language of "
            "the generated metadata is set under Metadata Language.",
        )
    finally:
        activate("en")
    assert text.startswith("Idioma do próprio aplicativo")
    assert max(len(line) for line in text.splitlines()) <= TOOLTIP_WIDTH

    filled = tooltip("Already on the file: {fields}", fields="T = title")
    assert filled == "Already on the file: T = title"


def test_fields_written_reports_only_nonempty_values() -> None:
    """Empty values write nothing, so only the populated fields count as newly present."""
    written = fields_written("Duck", "", KeywordSet(subject=["Bird"]))
    assert written == {FIELD_TITLE, FIELD_KEYWORDS}
    assert fields_written(None, None, KeywordSet()) == set()


def _config_values(**overrides: object) -> GuiConfigValues:
    """Build GuiConfigValues with sensible test defaults, overridable per test."""
    base: dict[str, object] = {
        "provider_name": "lmstudio",
        "model_name": "qwen/qwen3-vl-30b",
        "api_base_url": "http://localhost:1234/v1",
        "extensions": "jpg,cr3",
        "recursive": True,
        "write_title": True,
        "write_description": True,
        "write_keywords": True,
        "preserve_keywords": True,
        "sidecar_mode": "all",
        "backup_xmp": True,
        "telemetry_enabled": True,
    }
    base.update(overrides)
    return GuiConfigValues(**base)  # type: ignore[arg-type]


def test_config_toml_text_round_trips_through_load_defaults() -> None:
    """The GUI-written TOML parses and lands on the right Defaults fields."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    from photo_tagger.cli_options import load_defaults  # noqa: PLC0415

    text = config_toml_text(
        _config_values(
            model_name='qwen "vl" model',
            write_description=False,
            sidecar_mode="raw",
            backup_xmp=False,
            telemetry_enabled=False,
        ),
    )
    defaults = load_defaults(tomllib.loads(text))

    assert defaults.provider.model_name == 'qwen "vl" model'  # quotes survive escaping
    assert defaults.provider.api_base_url == "http://localhost:1234/v1"
    assert defaults.extensions == "jpg,cr3"
    assert defaults.recursive is True
    assert defaults.output.write_description is False
    assert defaults.output.sidecar_mode == "raw"
    assert defaults.output.backup_xmp is False
    assert defaults.telemetry.enabled is False


def test_config_writers_persist_the_end_of_run_alert_choices() -> None:
    """A fresh file and a merged one both carry the notification and its sound."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    from photo_tagger.cli_options import load_defaults  # noqa: PLC0415

    values = _config_values(notify=True, notify_sound=True)
    for text in (config_toml_text(values), merged_config_text("[display]\n", values)):
        display = load_defaults(tomllib.loads(text)).display
        assert display.notify is True
        assert display.notify_sound is True


def test_config_writers_persist_the_per_field_overwrite_choices() -> None:
    """A fresh file and a merged one both carry all three keep-or-replace answers."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    values = _config_values(preserve_title=True, preserve_description=True)
    for text in (config_toml_text(values), merged_config_text("[output]\n", values)):
        output = tomllib.loads(text)["output"]
        assert output["preserve_title"] is True
        assert output["preserve_description"] is True
        assert output["preserve_keywords"] is True


def test_config_toml_text_omits_blank_url() -> None:
    """A blank base URL is left out so the provider default applies."""
    text = config_toml_text(_config_values(api_base_url=None))
    assert "api_base_url" not in text


def test_merged_config_text_preserves_comments_and_unknown_keys() -> None:
    """Merging updates only the GUI-managed keys; comments and other settings survive."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    existing = (
        "# my hand-written config\n"
        'extensions = "cr3"\n'
        "\n"
        "[provider]\n"
        'provider_name = "ollama"\n'
        '# model_name = "qwen/qwen3-vl-30b"\n'
        "\n"
        "[filter]\n"
        "skip_tagged = true\n"
        "\n"
        "[artifacts]\n"
        'summary_file = "summary.txt"  # keep me\n'
    )
    merged = merged_config_text(existing, _config_values(model_name="llava", backup_xmp=False))

    # Comments and untouched tables survive verbatim.
    assert "# my hand-written config" in merged
    assert "# keep me" in merged
    data = tomllib.loads(merged)
    assert data["filter"]["skip_tagged"] is True
    assert data["artifacts"]["summary_file"] == "summary.txt"
    # GUI-managed keys are updated in place.
    assert data["extensions"] == "jpg,cr3"
    assert data["provider"]["provider_name"] == "lmstudio"
    assert data["provider"]["model_name"] == "llava"
    assert data["output"]["sidecar_mode"] == "all"
    assert data["output"]["backup_xmp"] is False


def test_merged_config_text_drops_blank_url() -> None:
    """Clearing the URL removes the key so the provider default applies again."""
    existing = '[provider]\napi_base_url = "http://old:1234/v1"\n'
    merged = merged_config_text(existing, _config_values(api_base_url=None))
    assert "api_base_url" not in merged


@pytest.mark.parametrize(
    "writer",
    [
        lambda text: merged_config_text(text, _config_values()),
        lambda text: config_text_with_language(text, "pt_BR"),
        lambda text: config_text_with_output_language(text, "German"),
    ],
    ids=["settings", "ui-language", "metadata-language"],
)
def test_config_writers_refuse_a_file_that_is_not_toml(writer: Callable[[str], str]) -> None:
    """
    A hand-edited config with a syntax error is reported, not rewritten from scratch.

    tomlkit raises a ParseError, which used to escape the Qt slot that saves settings and take the
    window down with it. The window catches ConfigFileError and shows it in the usual warning.
    """
    with pytest.raises(ConfigFileError, match="not valid TOML"):
        writer("this is not toml = = =")


@pytest.mark.parametrize(
    ("existing", "writer"),
    [
        ('provider = "lmstudio"\n', lambda text: merged_config_text(text, _config_values())),
        ("inference = 5\n", lambda text: config_text_with_output_language(text, "English")),
    ],
    ids=["provider", "inference"],
)
def test_config_writers_refuse_a_key_that_is_not_a_table(
    existing: str,
    writer: Callable[[str], str],
) -> None:
    """
    A GUI-managed key that holds a scalar is left alone rather than overwritten.

    Assigning into it raised a bare TypeError (or AttributeError) out of the save slot. Refusing
    says what is wrong and keeps whatever the user meant by the key.
    """
    with pytest.raises(ConfigFileError, match="not a"):
        writer(existing)


def test_apply_proposal_carries_the_cache_flag() -> None:
    """A cached proposal marks the item so the tree can label it 'ready (cached)'."""
    item = PhotoItem(path=Path("/a.jpg"))
    proposal = Proposal(
        path=Path("/a.jpg"),
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="T",
        description="D",
        keywords=[],
        from_cache=True,
    )
    apply_proposal(item, proposal)
    assert item.from_cache is True


def test_thumb_badges_lifecycle_and_info_markers() -> None:
    """One lifecycle badge at most, plus metadata and sidecar markers when they apply."""
    failed = PhotoItem(path=Path("/a.jpg"), status=FAILED, known_fields={"title"})
    assert thumb_badges(failed, has_sidecar=True) == [BADGE_FAILED, BADGE_METADATA, BADGE_SIDECAR]

    saved = PhotoItem(path=Path("/a.jpg"), status=SAVED)
    assert thumb_badges(saved, has_sidecar=False) == [BADGE_SAVED]

    ready = PhotoItem(path=Path("/a.jpg"), status=READY, has_proposal=True)
    assert thumb_badges(ready, has_sidecar=False) == [BADGE_UNSAVED]

    pending = PhotoItem(path=Path("/a.jpg"), known_fields=set())
    assert thumb_badges(pending, has_sidecar=False) == []


def test_keyword_diff_merge_marks_added_and_unchanged() -> None:
    """Without overwrite, kept keywords are unchanged and new ones are added; none removed."""
    existing = KeywordSet(subject=["Beach", "Bird"])
    diff = keyword_diff(existing, ["Bird", "Eagle"], overwrite=False)
    assert ("Beach", UNCHANGED) in diff
    assert ("Bird", UNCHANGED) in diff
    assert ("Eagle", ADDED) in diff
    assert all(state != REMOVED for _, state in diff)


def test_keyword_diff_overwrite_marks_removed() -> None:
    """With overwrite, keywords not in the new set are marked removed."""
    existing = KeywordSet(subject=["Beach", "Bird"])
    diff = keyword_diff(existing, ["Bird", "Eagle"], overwrite=True)
    assert ("Eagle", ADDED) in diff
    assert ("Bird", UNCHANGED) in diff
    assert ("Beach", REMOVED) in diff


def test_folder_node_is_constructible() -> None:
    """FolderNode is a plain dataclass usable directly in tests and rendering."""
    node = FolderNode(path=Path("/x"), label="/x", folders=[], files=[Path("/x/a.jpg")])
    assert node.label == "/x"


def test_deselect_paths_unchecks_matches_and_counts_only_changes() -> None:
    """Only listed, still-selected items flip; the count is the newly-skipped tally."""
    a, b, c = Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")
    items = {str(p): PhotoItem(path=p) for p in (a, b, c)}
    items[str(b)].selected = False  # already off: it must not count again

    changed = deselect_paths(items, [a, b])

    assert changed == 1
    assert items[str(a)].selected is False
    assert items[str(b)].selected is False
    assert items[str(c)].selected is True  # untouched


def test_deselect_paths_ignores_unknown_paths() -> None:
    """A path not in the list deselects nothing and counts zero."""
    a = Path("/a.jpg")
    items = {str(a): PhotoItem(path=a)}
    assert deselect_paths(items, [Path("/missing.jpg")]) == 0
    assert items[str(a)].selected is True


def _mixed_folder() -> list[PhotoItem]:
    """Build a folder's worth of photos: two DNG+JPEG pairs, everything checked."""
    names = ("IMG_0001.dng", "IMG_0001.jpg", "IMG_0002.DNG", "sunset_edit.jpg")
    return [PhotoItem(path=Path("/photos") / name) for name in names]


def test_apply_selection_check_leaves_the_rest_alone() -> None:
    """Checking a criterion touches only what it matched, and counts only real changes."""
    items = _mixed_folder()
    for item in items:
        item.selected = False

    result = apply_selection(items, lambda item: matches_extension(item, "dng"), SELECT_CHECK)

    assert result == SelectionChange(matched=2, changed=2)
    assert [item.selected for item in items] == [True, False, True, False]


def test_apply_selection_uncheck_counts_only_what_moved() -> None:
    """A photo already unchecked is matched but not counted: the tally is what changed."""
    items = _mixed_folder()
    items[0].selected = False  # already off

    result = apply_selection(items, lambda item: matches_extension(item, "dng"), SELECT_UNCHECK)

    assert result == SelectionChange(matched=2, changed=1)
    assert [item.selected for item in items] == [False, True, False, True]


def test_apply_selection_only_unchecks_everything_it_missed() -> None:
    """Check Only is the "just these" move: the matches go on, every other photo goes off."""
    items = _mixed_folder()
    # Start on the opposite footing: the RAWs are checked and the JPEGs are not.
    items[1].selected = False
    items[3].selected = False

    result = apply_selection(items, lambda item: matches_extension(item, "jpg"), SELECT_ONLY)

    assert result == SelectionChange(matched=2, changed=4)
    assert [item.selected for item in items] == [False, True, False, True]


def test_apply_selection_reports_a_criterion_that_matched_nothing() -> None:
    """Nothing matched and nothing changed are different answers the status line tells apart."""
    items = _mixed_folder()

    result = apply_selection(items, lambda item: matches_extension(item, "cr3"), SELECT_CHECK)

    assert result == SelectionChange(matched=0, changed=0)
    assert all(item.selected for item in items)


def test_extension_counts_orders_by_count_then_name() -> None:
    """The file-type menu lists the commonest type first, ties broken alphabetically."""
    items = [*_mixed_folder(), PhotoItem(path=Path("/photos/no_extension"))]

    assert extension_counts(items) == [("dng", 2), ("jpg", 2), ("", 1)]


def test_matches_extension_ignores_case_and_a_leading_dot() -> None:
    """A .DNG is a dng, and the caller may pass the extension either way."""
    item = PhotoItem(path=Path("/photos/IMG_0002.DNG"))
    assert matches_extension(item, "dng") is True
    assert matches_extension(item, ".DNG") is True
    assert matches_extension(item, "jpg") is False


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        # No wildcard reads as "contains", which is what a name box invites.
        ("IMG", True),
        ("img_00", True),
        ("sunset", False),
        # Anything with a wildcard is matched as the glob it looks like.
        ("IMG_*.dng", True),
        ("*.jpg", False),
        ("IMG_000?.dng", True),
        # A separator in the pattern matches the whole path, so it can name a folder.
        ("/photos/*.dng", True),
        ("/other/*.dng", False),
        # Either separator spells the same folder, so a Windows habit still finds the photo.
        ("\\photos\\*.dng", True),
        ("\\other\\*.dng", False),
        # An empty pattern matches nothing, rather than every photo in the list.
        ("", False),
    ],
)
def test_matches_name_pattern(pattern: str, *, expected: bool) -> None:
    """Name patterns are case-insensitive, and plain text means "contains"."""
    item = PhotoItem(path=Path("/photos/IMG_0001.dng"))
    assert matches_name_pattern(item, pattern) is expected


@pytest.mark.parametrize(
    ("windows_path", "pattern"),
    [
        # What the Windows runners build from Path("/photos/IMG_0001.dng"), typed either way.
        (r"\photos\IMG_0001.dng", "/photos/*.dng"),
        (r"\photos\IMG_0001.dng", r"\photos\*.dng"),
        # The same for a real drive-lettered path.
        (r"C:\photos\IMG_0001.dng", "C:/photos/*.dng"),
        (r"C:\photos\IMG_0001.dng", r"C:\photos\*.dng"),
    ],
)
def test_matches_name_pattern_on_a_windows_path(windows_path: str, pattern: str) -> None:
    """
    A Windows path stringifies with backslashes, which must not decide what the pattern finds.

    Pinned with an explicit PureWindowsPath so the POSIX runners catch a regression here too, rather
    than only the Windows job.
    """
    item = PhotoItem(path=cast("Path", PureWindowsPath(windows_path)))
    assert matches_name_pattern(item, pattern) is True


def test_search_summary_counts_what_is_hidden() -> None:
    """A filtered list says so, since the counts beside it are of the whole list."""
    assert search_summary(2, 4) == "Showing 2 of 4 photos"
    assert search_summary(0, 1) == "Showing 0 of 1 photo"


def test_flat_labels_keep_plain_names_inside_one_folder() -> None:
    """With nothing to disambiguate, the flat list reads exactly like the nested one."""
    paths = [Path("/photos/a.jpg"), Path("/photos/b.jpg")]
    assert flat_labels(paths) == {paths[0]: "a.jpg", paths[1]: "b.jpg"}


def test_flat_labels_show_the_path_below_the_shared_folder() -> None:
    """Two folders holding the same filename stay told apart once the grouping is gone."""
    first, second = Path("/photos/shoot1/DSC_0042.NEF"), Path("/photos/shoot2/DSC_0042.NEF")

    labels = flat_labels([first, second])

    assert labels == {
        first: str(Path("shoot1") / "DSC_0042.NEF"),
        second: str(Path("shoot2") / "DSC_0042.NEF"),
    }


def test_flat_labels_fall_back_to_the_full_path_without_a_shared_root() -> None:
    """Paths with no common root at all (mixed absolute and relative) still label uniquely."""
    absolute, relative = Path("/photos/a.jpg"), Path("shoot/b.jpg")

    labels = flat_labels([absolute, relative])

    assert labels == {absolute: str(absolute), relative: str(relative)}


def test_flat_labels_of_an_empty_list() -> None:
    """An empty list labels nothing, rather than reaching for a common root that is not there."""
    assert flat_labels([]) == {}


def test_paths_matching_fields_all_requires_every_field() -> None:
    """match_all selects only photos that carry every required field."""
    a, b, c = Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")
    presence = {a: {"title", "description"}, b: {"keywords"}, c: {"title"}}
    # A title-and-description criterion skips a, but keeps the keyword-only b and title-only c.
    matched = paths_matching_fields(presence, {"title", "description"}, match_all=True)
    assert matched == {a}


def test_paths_matching_fields_any_requires_one_field() -> None:
    """Without match_all, having any one required field is enough (the 'any metadata' rule)."""
    a, b = Path("/a.jpg"), Path("/b.jpg")
    presence = {a: {"keywords"}, b: set()}
    matched = paths_matching_fields(
        presence,
        {"title", "description", "keywords"},
        match_all=False,
    )
    assert matched == {a}


def test_paths_matching_fields_empty_required_matches_nothing() -> None:
    """An empty required set never matches, so it cannot deselect every photo by accident."""
    presence = {Path("/a.jpg"): {"title"}}
    assert paths_matching_fields(presence, set(), match_all=True) == set()
    assert paths_matching_fields(presence, set(), match_all=False) == set()


def test_status_sort_rank_follows_lifecycle_order() -> None:
    """Ranks increase along the lifecycle, so a status sort is meaningful, not alphabetical."""
    ranks = [status_sort_rank(s) for s in (PENDING, WORKING, READY, SAVED, FAILED)]
    assert ranks == [0, 1, 2, 3, 4]


def test_status_sort_rank_unknown_status_sorts_last() -> None:
    """An unrecognized status ranks after every known one rather than raising."""
    assert status_sort_rank("bogus") == len(STATUS_SORT_ORDER)
    assert status_sort_rank("bogus") > status_sort_rank(FAILED)


def test_sort_photos_by_name_flips_with_direction() -> None:
    """Sorting by name is case-insensitive ascending, and descending reverses it end to end."""
    items = [PhotoItem(path=Path(f"/d/{n}")) for n in ("c.jpg", "A.jpg", "b.jpg")]
    assert [i.path.name for i in sort_photos(items, SORT_NAME, descending=False)] == [
        "A.jpg",
        "b.jpg",
        "c.jpg",
    ]
    assert [i.path.name for i in sort_photos(items, SORT_NAME, descending=True)] == [
        "c.jpg",
        "b.jpg",
        "A.jpg",
    ]


def test_sort_photos_by_status_uses_lifecycle_rank() -> None:
    """Status sorts by lifecycle rank (pending < ready < failed), not the alphabetical label."""
    a = PhotoItem(path=Path("/d/a.jpg"), status=FAILED)
    b = PhotoItem(path=Path("/d/b.jpg"), status=PENDING)
    c = PhotoItem(path=Path("/d/c.jpg"), status=READY)
    order = sort_photos([a, b, c], SORT_STATUS, descending=False)
    assert [i.path.name for i in order] == ["b.jpg", "c.jpg", "a.jpg"]


def test_sort_photos_by_type_then_name() -> None:
    """Type sorts by extension label, with the filename breaking ties within one type."""
    items = [
        PhotoItem(path=Path("/d/b.png")),
        PhotoItem(path=Path("/d/a.png")),
        PhotoItem(path=Path("/d/c.jpg")),
    ]
    order = sort_photos(items, SORT_TYPE, descending=False)
    assert [i.path.name for i in order] == ["c.jpg", "a.png", "b.png"]


def test_sort_photos_by_tagged_puts_untagged_first() -> None:
    """Tagged sorts by the compressed letters, so the '-' of an untagged photo sorts first."""
    tagged = PhotoItem(path=Path("/d/a.jpg"), known_fields={FIELD_TITLE, FIELD_KEYWORDS})
    untagged = PhotoItem(path=Path("/d/b.jpg"), known_fields=set())
    order = sort_photos([tagged, untagged], SORT_TAGGED, descending=False)
    assert [i.path.name for i in order] == ["b.jpg", "a.jpg"]


def test_photo_sort_key_is_uniformly_shaped_across_criteria() -> None:
    """Every criterion yields a same-shape string tuple, so one criterion sorts a whole list."""
    item = PhotoItem(path=Path("/d/a.jpg"), status=READY, known_fields={FIELD_TITLE})
    for criterion in (SORT_NAME, SORT_TYPE, SORT_STATUS, SORT_TAGGED):
        key = photo_sort_key(item, criterion)
        assert len(key) == 3  # noqa: PLR2004 - primary, name, full-path tiebreaks
        assert all(isinstance(part, str) for part in key)


def test_filter_photos_all_keeps_everything_in_order() -> None:
    """FILTER_ALL is a pass-through: every photo survives, order preserved."""
    items = [
        PhotoItem(path=Path("/d/a.jpg"), status=PENDING),
        PhotoItem(path=Path("/d/b.jpg"), status=SAVED),
    ]
    assert filter_photos(items, FILTER_ALL) == items


def test_filter_photos_by_status_and_selection() -> None:
    """The status and selection filters each keep only the photos in that state."""
    saved = PhotoItem(path=Path("/d/a.jpg"), status=SAVED, selected=False)
    failed = PhotoItem(path=Path("/d/b.jpg"), status=FAILED, selected=True)
    pending = PhotoItem(path=Path("/d/c.jpg"), status=PENDING, selected=True)
    items = [saved, failed, pending]
    assert filter_photos(items, FILTER_SAVED) == [saved]
    assert filter_photos(items, FILTER_FAILED) == [failed]
    assert filter_photos(items, FILTER_PENDING) == [pending]
    assert filter_photos(items, FILTER_SELECTED) == [failed, pending]


def test_filter_photos_generated_uses_has_proposal() -> None:
    """The Generated filter keys off has_proposal, not the lifecycle status."""
    generated = PhotoItem(path=Path("/d/a.jpg"), has_proposal=True)
    plain = PhotoItem(path=Path("/d/b.jpg"), has_proposal=False)
    assert filter_photos([generated, plain], FILTER_GENERATED) == [generated]


def test_filter_untagged_matches_only_scanned_empty() -> None:
    """Untagged keeps a scanned-but-empty photo, and never a tagged or still-unscanned one."""
    empty = PhotoItem(path=Path("/d/a.jpg"), known_fields=set())
    tagged = PhotoItem(path=Path("/d/b.jpg"), known_fields={FIELD_TITLE})
    unscanned = PhotoItem(path=Path("/d/c.jpg"), known_fields=None)
    assert filter_photos([empty, tagged, unscanned], FILTER_UNTAGGED) == [empty]


def test_photo_matches_filter_unknown_criterion_matches_all() -> None:
    """An unrecognized criterion behaves like FILTER_ALL rather than hiding every photo."""
    assert photo_matches_filter(PhotoItem(path=Path("/d/a.jpg")), "bogus") is True


def test_status_summary_counts_states() -> None:
    """The summary reports total, selected, generated, saved, and failed counts."""
    items = [
        PhotoItem(path=Path("/a.jpg"), has_proposal=True, status=SAVED),
        PhotoItem(path=Path("/b.jpg"), has_proposal=True, status=READY),
        PhotoItem(path=Path("/c.jpg"), selected=False, status="failed"),
    ]
    summary = status_summary(items)
    assert "3 files" in summary
    assert "2 selected" in summary
    assert "2 generated" in summary
    assert "1 saved" in summary
    assert "1 failed" in summary


def test_status_summary_uses_singular_for_one_file() -> None:
    """A single photo reads '1 file', not '1 files'."""
    assert "1 file ·" in status_summary([PhotoItem(path=Path("/a.jpg"))])


def test_config_text_with_language_sets_and_preserves() -> None:
    """Setting a language keeps every other line (comments included) intact."""
    existing = '# my config\nextensions = "jpg"\n\n[provider]\nmodel_name = "m"\n'
    result = config_text_with_language(existing, "pt_BR")
    assert 'language = "pt_BR"' in result
    assert "# my config" in result
    assert 'model_name = "m"' in result


def test_config_text_with_language_auto_removes_the_key() -> None:
    """Choosing the system default removes the pinned key instead of writing 'auto'."""
    existing = 'language = "pt_BR"\nextensions = "jpg"\n'
    result = config_text_with_language(existing, "auto")
    assert "language" not in result
    assert 'extensions = "jpg"' in result


def test_config_text_with_language_works_on_an_empty_file() -> None:
    """A missing config starts from empty text and still gets the key."""
    assert 'language = "en"' in config_text_with_language("", "en")
    assert config_text_with_language("", "auto") == ""


def test_config_text_with_output_language_sets_and_preserves() -> None:
    """Setting a metadata language keeps comments and unrelated [inference] keys intact."""
    existing = '# my config\nextensions = "jpg"\n\n[inference]\ntemperature = 0.3\n'
    result = config_text_with_output_language(existing, "German")
    assert 'output_language = "German"' in result
    assert "# my config" in result
    assert "temperature = 0.3" in result


def test_config_text_with_output_language_default_removes_the_key() -> None:
    """Choosing English again (any casing) unpins the key but keeps the table's other keys."""
    existing = '[inference]\noutput_language = "German"\ntemperature = 0.3\n'
    result = config_text_with_output_language(existing, "english")
    assert "output_language" not in result
    assert "temperature = 0.3" in result


def test_config_text_with_output_language_drops_an_emptied_table() -> None:
    """Removing the key also removes an [inference] table that held nothing else."""
    existing = 'extensions = "jpg"\n\n[inference]\noutput_language = "German"\n'
    result = config_text_with_output_language(existing, "English")
    assert "inference" not in result
    assert 'extensions = "jpg"' in result


def test_config_text_with_output_language_works_on_an_empty_file() -> None:
    """A missing config starts from empty text; the default writes nothing at all."""
    result = config_text_with_output_language("", "Spanish")
    assert "[inference]" in result
    assert 'output_language = "Spanish"' in result
    assert config_text_with_output_language("", "English") == ""


def test_output_language_suggestions_start_with_the_default() -> None:
    """The menu builder relies on the default language leading the suggestion list."""
    from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE  # noqa: PLC0415

    assert OUTPUT_LANGUAGE_SUGGESTIONS[0] == DEFAULT_OUTPUT_LANGUAGE
    # The values reach the prompt verbatim, so the list must stay in English regardless of the
    # UI language; the GUI translates the labels at display time only.
    assert "Brazilian Portuguese" in OUTPUT_LANGUAGE_SUGGESTIONS


def test_ensure_path_dirs_prepends_missing_and_dedups() -> None:
    """Absent dirs are prepended in order; ones already on PATH are not duplicated."""
    base = os.pathsep.join(["/usr/bin", "/bin"])
    out = ensure_path_dirs(base, ["/opt/homebrew/bin", "/usr/bin"])
    assert out == os.pathsep.join(["/opt/homebrew/bin", "/usr/bin", "/bin"])


def test_ensure_path_dirs_returns_input_unchanged_when_all_present() -> None:
    """When every dir is already present, the original string is returned as-is."""
    base = os.pathsep.join(["/opt/homebrew/bin", "/usr/bin"])
    assert ensure_path_dirs(base, ["/usr/bin"]) is base


def test_ensure_path_dirs_handles_empty_path() -> None:
    """An empty starting PATH yields just the added dirs (no leading separator)."""
    assert ensure_path_dirs("", ["/opt/homebrew/bin"]) == "/opt/homebrew/bin"


def test_login_shell_path_parses_shell_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The login shell's reported PATH is split into entries, blanks dropped."""
    monkeypatch.setenv("SHELL", "/bin/zsh")
    out = os.pathsep.join(["/run/current-system/sw/bin", "/usr/bin", ""])
    monkeypatch.setattr(
        "photo_tagger.gui_state.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout=out),
    )
    assert login_shell_path() == ["/run/current-system/sw/bin", "/usr/bin"]


def test_login_shell_path_empty_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no $SHELL there is nothing to ask, so it returns []."""
    monkeypatch.delenv("SHELL", raising=False)
    assert login_shell_path() == []


def test_login_shell_path_empty_on_shell_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shell that errors or times out degrades to [] rather than raising."""
    monkeypatch.setenv("SHELL", "/bin/zsh")

    def boom(*_a: object, **_k: object) -> object:
        raise OSError

    monkeypatch.setattr("photo_tagger.gui_state.subprocess.run", boom)
    assert login_shell_path() == []


def test_format_duration_switches_to_hours() -> None:
    """Below an hour it reads m:ss; past it, h:mm:ss."""
    assert format_duration(0) == "0:00"
    assert format_duration(9.7) == "0:09"
    assert format_duration(75) == "1:15"
    assert format_duration(3675) == "1:01:15"


def test_format_duration_clamps_negatives() -> None:
    """A negative duration reads 0:00 instead of counting backwards."""
    assert format_duration(-5) == "0:00"


def test_estimate_remaining_extrapolates_from_finished_items() -> None:
    """Three photos in 30s means 10s each, so the seven left are 70s away."""
    assert estimate_remaining(3, 10, 30.0) == pytest.approx(70.0)


def test_estimate_remaining_is_unknown_before_the_first_and_after_the_last() -> None:
    """With nothing finished there is no rate, and a finished batch has nothing left."""
    assert estimate_remaining(0, 10, 5.0) is None
    assert estimate_remaining(10, 10, 100.0) is None


def test_progress_timing_text_omits_the_estimate_until_something_finishes() -> None:
    """The readout starts as bare elapsed time, then gains the remaining estimate."""
    assert progress_timing_text(0, 7, 4.0) == "0:04 elapsed"
    assert progress_timing_text(1, 7, 10.0) == "0:10 elapsed · 1:00 left"


def _proposed_item() -> PhotoItem:
    """Build a photo with existing metadata and an edited proposal, ready to save."""
    return PhotoItem(
        path=Path("/photos/a.jpg"),
        existing_keywords=KeywordSet(subject=["Old"]),
        title="New Title",
        description="New caption.",
        keywords=["Duck"],
        has_proposal=True,
    )


def test_build_save_job_writes_every_checked_field() -> None:
    """With all three toggles on, the job carries the title, description, and merged keywords."""
    job = build_save_job(_proposed_item(), SaveOptions())
    assert job.title == "New Title"
    assert job.description == "New caption."
    assert sorted(job.keywords.subject) == ["Duck", "Old"]
    assert job.fields == {FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}


def test_build_save_job_leaves_unchecked_fields_out() -> None:
    """Switched-off fields become empty, which write_metadata leaves off the file."""
    options = SaveOptions(write_title=False, write_description=False)
    job = build_save_job(_proposed_item(), options)
    assert job.title is None
    assert job.description is None
    assert job.fields == {FIELD_KEYWORDS}


def test_build_save_job_overwrite_drops_the_existing_keywords() -> None:
    """Overwrite replaces the keywords on the file instead of merging with them."""
    job = build_save_job(_proposed_item(), SaveOptions(overwrite_keywords=True))
    assert job.keywords.subject == ["Duck"]


def test_build_save_job_keeps_a_caption_the_photo_already_has() -> None:
    """With its Overwrite entry off, a field the photo already fills is left alone."""
    item = _proposed_item()
    item.existing_title = "Camera Title"
    item.existing_description = "default"
    options = SaveOptions(overwrite_title=False, overwrite_description=False)
    job = build_save_job(item, options)
    assert job.title is None
    assert job.description is None
    # The keywords still merge: each field answers for itself.
    assert sorted(job.keywords.subject) == ["Duck", "Old"]


def test_build_save_job_fills_an_empty_caption_even_when_preserving() -> None:
    """Keeping what a photo has is not the same as writing nothing: an empty field is filled."""
    options = SaveOptions(overwrite_title=False, overwrite_description=False)
    job = build_save_job(_proposed_item(), options)
    assert job.title == "New Title"
    assert job.description == "New caption."


def test_build_save_job_without_keywords_writes_none() -> None:
    """Keywords off means an empty set, so the keyword field is not touched."""
    job = build_save_job(_proposed_item(), SaveOptions(write_keywords=False))
    assert job.keywords.is_empty()


def test_save_options_any_field_needs_one_toggle() -> None:
    """A save with all three fields off would write nothing at all."""
    assert SaveOptions().any_field
    assert SaveOptions(write_title=False, write_keywords=False).any_field
    assert not SaveOptions(
        write_title=False,
        write_description=False,
        write_keywords=False,
    ).any_field


# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------


def _vocabulary() -> Vocabulary:
    """Build a small catalog with one hierarchy, like a Lightroom export gives."""
    return Vocabulary.from_entries(["Animal|Bird|Osprey", "Sunset"])


def test_apply_vocabulary_without_one_passes_the_keywords_through() -> None:
    """No vocabulary means the model's own wording is what the review pane shows."""
    outcome = apply_vocabulary(["Ospreys", "Tractor"], None)
    assert outcome.keywords == ["Ospreys", "Tractor"]
    assert outcome.mapped == 0
    assert outcome.dropped == []


def test_apply_vocabulary_rewrites_onto_the_catalogs_spelling_and_hierarchy() -> None:
    """A match comes back as the catalog spells it, carrying the catalog's own parents."""
    outcome = apply_vocabulary(["ospreys", "Tractor"], _vocabulary())
    assert outcome.keywords == ["Osprey<Bird<Animal", "Tractor"]
    assert outcome.mapped == 1
    assert outcome.dropped == []


def test_apply_vocabulary_strict_drops_what_the_catalog_lacks() -> None:
    """Strict mode is what stops a run seeding the catalog with new keywords."""
    outcome = apply_vocabulary(["Ospreys", "Tractor"], _vocabulary(), strict=True)
    assert outcome.keywords == ["Osprey<Bird<Animal"]
    assert outcome.dropped == ["Tractor"]


def test_load_vocabulary_file_reports_an_unusable_file(tmp_path: Path) -> None:
    """Choosing the wrong file is a normal mistake, so it is a message, not an exception."""
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n", encoding="utf-8")

    vocabulary, error = load_vocabulary_file(empty)

    assert vocabulary is None
    assert "no keywords" in error


def test_load_vocabulary_file_reads_a_keyword_list(tmp_path: Path) -> None:
    """A plain list of terms and paths loads the same way the CLI's --vocabulary does."""
    listing = tmp_path / "keywords.txt"
    listing.write_text("Animal|Bird|Osprey\nSunset\n", encoding="utf-8")

    vocabulary, error = load_vocabulary_file(listing)

    assert error == ""
    assert vocabulary is not None
    assert vocabulary.match("ospreys") == "Osprey"


def test_vocabulary_status_describes_each_state(tmp_path: Path) -> None:
    """The label under the picker says what is in force, including why a file was refused."""
    assert "No vocabulary" in vocabulary_status(None, None)
    assert vocabulary_status(tmp_path / "k.txt", None, "broken file") == "broken file"
    loaded = vocabulary_status(tmp_path / "k.txt", _vocabulary())
    assert "4 keywords" in loaded  # Animal, Bird, Osprey, Sunset
    assert "k.txt" in loaded


def test_vocabulary_summary_is_silent_when_nothing_changed() -> None:
    """A vocabulary the batch already fits has nothing to report."""
    assert vocabulary_summary(0, {}) == ""


def test_vocabulary_summary_reports_rewrites_and_rejections() -> None:
    """The summary is what tells you whether the vocabulary needs a new keyword."""
    summary = vocabulary_summary(3, {"Tractor": 4, "Barn": 1})
    assert "3 keywords rewritten" in summary
    assert "2 keywords dropped" in summary
    assert "Tractor, Barn" in summary  # most frequent first


def test_vocabulary_summary_caps_the_named_terms() -> None:
    """Naming every rejection would fill the status bar; the log keeps the full list."""
    dropped = {f"Term{index}": 10 - index for index in range(8)}
    summary = vocabulary_summary(0, dropped)
    assert "8 keywords dropped" in summary
    # Five names, then an ellipsis, which is six comma-separated pieces.
    assert summary.count(",") == _MAX_LISTED_DROPPED
    assert "..." in summary


# ---------------------------------------------------------------------------
# Shoot harmonization
# ---------------------------------------------------------------------------


def _photo(path: Path, minutes: int) -> Path:
    """Create a file whose mtime sits *minutes* into a fixed morning, for session grouping."""
    path.write_text("x")
    stamp = datetime(2026, 5, 1, 9, 0, tzinfo=UTC).timestamp() + minutes * 60
    os.utime(path, (stamp, stamp))
    return path


def test_harmonize_sessions_is_off_for_a_zero_gap(tmp_path: Path) -> None:
    """Zero minutes means every photo stands on its own, exactly as before."""
    photo = _photo(tmp_path / "a.jpg", 0)
    assert harmonize_sessions({photo: ["Osprey"]}, gap_minutes=0) == HarmonizeResult()


def test_harmonize_sessions_makes_one_shoot_agree_with_itself(tmp_path: Path) -> None:
    """Two frames of one bird stop landing in the catalog as two keywords."""
    first = _photo(tmp_path / "a.jpg", 0)
    second = _photo(tmp_path / "b.jpg", 3)
    third = _photo(tmp_path / "c.jpg", 6)

    result = harmonize_sessions(
        {first: ["Osprey"], second: ["Osprey"], third: ["Ospreys"]},
        gap_minutes=30,
    )

    assert result.sessions == 1
    # Only the odd one out changed, onto the spelling the shoot used most.
    assert result.keywords == {str(third): ["Osprey"]}


def test_harmonize_sessions_keeps_separate_shoots_apart(tmp_path: Path) -> None:
    """A shoot two hours later is a different shoot, and settles its own wording."""
    morning = _photo(tmp_path / "a.jpg", 0)
    morning_two = _photo(tmp_path / "b.jpg", 4)
    evening = _photo(tmp_path / "c.jpg", 300)

    result = harmonize_sessions(
        {morning: ["Sunrise"], morning_two: ["Sunrise"], evening: ["Sunsets"]},
        gap_minutes=60,
    )

    # The evening frame is alone in its session, so its own spelling is the majority.
    assert (result.sessions, result.keywords) == (2, {})


def test_harmonize_summary_describes_each_outcome() -> None:
    """The status line has to say something useful whether or not anything moved."""
    assert "generate some photos first" in harmonize_summary(HarmonizeResult())
    assert "already agreed" in harmonize_summary(HarmonizeResult(sessions=2))
    changed = HarmonizeResult(keywords={"/a.jpg": ["Osprey"]}, sessions=3)
    summary = harmonize_summary(changed)
    assert "3 shoots" in summary
    assert "1 photo" in summary


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def test_journal_time_parses_the_run_start_from_the_name() -> None:
    """Journals are named after their UTC start time, which is what the list shows."""
    assert journal_time(Path("20260501143005-4242.jsonl")) == datetime(
        2026,
        5,
        1,
        14,
        30,
        5,
        tzinfo=UTC,
    )
    assert journal_time(Path("not-a-journal.jsonl")) is None
    # Right width, impossible date: still not one of ours.
    assert journal_time(Path("99999999999999-1.jsonl")) is None


def test_journal_time_reads_a_microsecond_stamp() -> None:
    """Journals now carry microseconds so two runs in one second stay apart; both widths read."""
    assert journal_time(Path("20260501143005123456-4242.jsonl")) == datetime(
        2026,
        5,
        1,
        14,
        30,
        5,
        123456,
        tzinfo=UTC,
    )


def test_journal_label_names_the_run_and_its_size() -> None:
    """One row per recorded run: when it ran, and how much it wrote."""
    label = journal_label(Path("20260501143005-4242.jsonl"), 128)
    assert "128 files" in label
    assert "2026-05-01" in label


def test_journal_label_falls_back_to_the_file_name() -> None:
    """A journal named by hand still lists, rather than vanishing from the dialog."""
    assert journal_label(Path("mine.jsonl"), 1).startswith("mine")


def test_undo_action_label_translates_the_outcomes() -> None:
    """The dialog reads in words, not in the log's identifiers."""
    assert undo_action_label(RESTORED) == "restored"
    assert undo_action_label(CHANGED) == "changed since the run"
    assert undo_action_label("something-new") == "something-new"


def test_undo_summary_counts_what_was_put_back() -> None:
    """The status line separates what was reverted from what was deliberately left alone."""
    assert "recorded no writes" in undo_summary([])
    all_good = [UndoResult(Path("/a.xmp"), RESTORED), UndoResult(Path("/b.xmp"), DELETED)]
    assert undo_summary(all_good) == "Put back 2 files."
    mixed = [*all_good, UndoResult(Path("/c.xmp"), CHANGED, "changed since the run")]
    assert undo_summary(mixed) == "Put back 2 files; left 1 alone."


# ---------------------------------------------------------------------------
# Watching a folder
# ---------------------------------------------------------------------------


def test_watch_status_text_names_the_folders(tmp_path: Path) -> None:
    """The status bar says where the watch is looking, then what it has picked up."""
    settings = WatchSettings(folders=(tmp_path / "Inbox",))
    assert watch_status_text(settings, added=0) == "Watching Inbox for new photos..."
    assert watch_status_text(settings, added=3) == "Watching Inbox: 3 photos added so far."


def test_watch_settings_default_to_reviewing_before_writing() -> None:
    """Generating is the point of watching; saving unattended has to be asked for."""
    settings = WatchSettings()
    assert settings.generate is True
    assert settings.save is False


def test_config_toml_text_carries_the_keyword_rules(tmp_path: Path) -> None:
    """The vocabulary, the session gap, and undo logging are run settings, so they persist."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    from photo_tagger.cli_options import load_defaults  # noqa: PLC0415

    listing = tmp_path / "keywords.txt"
    text = config_toml_text(
        _config_values(
            vocabulary=listing,
            vocabulary_strict=True,
            session_gap_minutes=45.0,
            undo_log=False,
        ),
    )
    defaults = load_defaults(tomllib.loads(text))

    assert (
        defaults.output.vocabulary,
        defaults.output.vocabulary_strict,
        defaults.output.session_gap_minutes,
        defaults.artifacts.undo_log,
    ) == (listing, True, 45.0, False)


def test_config_toml_text_omits_an_unset_vocabulary() -> None:
    """No vocabulary chosen writes no key, so the CLI does not try to open a blank path."""
    assert "vocabulary = " not in config_toml_text(_config_values())


def test_merged_config_text_drops_a_cleared_vocabulary(tmp_path: Path) -> None:
    """Clearing the vocabulary in the window removes it from the config file too."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    existing = '[output]\nvocabulary = "old.txt"\nvocabulary_strict = true\n'
    kept = merged_config_text(existing, _config_values(vocabulary=tmp_path / "new.txt"))
    assert tomllib.loads(kept)["output"]["vocabulary"] == str(tmp_path / "new.txt")

    cleared = merged_config_text(existing, _config_values())
    assert "vocabulary" not in tomllib.loads(cleared)["output"]


def test_harmonize_sessions_skips_a_shoot_with_nothing_generated(tmp_path: Path) -> None:
    """A shoot whose photos carry no keywords has nothing to agree on, and is left alone."""
    photo = _photo(tmp_path / "a.jpg", 0)

    result = harmonize_sessions({photo: []}, gap_minutes=30)

    assert (result.sessions, result.keywords) == (1, {})


def test_keywords_to_save_keeps_the_vocabulary_spelling_verbatim() -> None:
    """A catalog that writes its keywords in lower case means it, merge step or not."""
    vocabulary = Vocabulary.from_entries(["gegenlicht"])

    written = keywords_to_save(
        KeywordSet(),
        ["gegenlicht"],
        overwrite=False,
        verbatim=vocabulary.exact,
    )

    assert written.subject == ["gegenlicht"]
    # Without the vocabulary there is nobody to defer to, so the merge step capitalizes as before.
    assert keywords_to_save(KeywordSet(), ["gegenlicht"], overwrite=False).subject == ["Gegenlicht"]


def test_build_save_job_writes_the_declared_spelling() -> None:
    """The save toggles carry the vocabulary through to what ExifTool is handed."""
    vocabulary = Vocabulary.from_entries(["gegenlicht"])
    item = PhotoItem(path=Path("/a.jpg"), keywords=["gegenlicht"])

    job = build_save_job(item, SaveOptions(verbatim=vocabulary.exact))

    assert job.keywords.subject == ["gegenlicht"]


def test_keyword_previews_show_what_the_save_will_write() -> None:
    """The diff and the tree must not promise a spelling the save then changes."""
    vocabulary = Vocabulary.from_entries(["landschaft|gegenlicht"])
    edited = ["gegenlicht<landschaft"]

    diff = keyword_diff(KeywordSet(), edited, overwrite=False, verbatim=vocabulary.exact)
    tree = hierarchy_preview(KeywordSet(), edited, overwrite=False, verbatim=vocabulary.exact)

    assert [keyword for keyword, _state in diff] == ["landschaft", "gegenlicht"]
    assert tree == "landschaft\n└─ gegenlicht"


def test_report_row_uses_the_declared_spelling(tmp_path: Path) -> None:
    """The CSV reports what a save writes, so it follows the vocabulary too."""
    vocabulary = Vocabulary.from_entries(["gegenlicht"])
    item = PhotoItem(path=tmp_path / "a.jpg", keywords=["gegenlicht"], has_proposal=True)

    row = photo_item_to_report_row(item, overwrite=False, verbatim=vocabulary.exact)

    assert row.keywords == ["gegenlicht"]


def test_record_dropped_terms_counts_and_stops_at_the_cap() -> None:
    """The tally names the busiest rejects; a run against the wrong file cannot grow it forever."""
    tally: dict[str, int] = {}

    record_dropped_terms(tally, ["Tractor", "Barn", "Tractor"])
    assert tally == {"Tractor": 2, "Barn": 1}

    record_dropped_terms(tally, [f"Term{index}" for index in range(MAX_TRACKED_DROPPED_TERMS)])
    assert len(tally) == MAX_TRACKED_DROPPED_TERMS
    # A term already counted still counts, cap or no cap.
    record_dropped_terms(tally, ["Tractor"])
    assert tally["Tractor"] == 3  # noqa: PLR2004 - two, then one more


def test_fit_zoom_scales_a_large_photo_down_to_the_viewport() -> None:
    """The tighter of the two axes decides, so the whole photo lands inside the window."""
    half = 0.5
    assert fit_zoom((4000, 3000), (2000, 2000)) == half
    assert fit_zoom((3000, 4000), (2000, 2000)) == half


def test_fit_zoom_never_upscales_a_small_photo() -> None:
    """A photo smaller than the window shows at its own size rather than blurred up to fill it."""
    assert fit_zoom((320, 240), (2000, 2000)) == 1.0


def test_fit_zoom_falls_back_on_a_degenerate_size() -> None:
    """A failed decode or a window without a layout yet reads as 1:1, not a division by zero."""
    assert fit_zoom((0, 0), (800, 600)) == 1.0
    assert fit_zoom((800, 600), (0, 0)) == 1.0


def test_step_zoom_moves_one_notch_each_way() -> None:
    """A notch in and a notch out are inverses, so the zoom returns to where it started."""
    assert step_zoom(1.0, 1) == ZOOM_STEP
    assert step_zoom(1.0, -1) == pytest.approx(1 / ZOOM_STEP)
    assert step_zoom(step_zoom(1.0, 1), -1) == pytest.approx(1.0)


def test_step_zoom_stops_at_the_limits() -> None:
    """Holding the zoom keys cannot push past the range the viewer can actually draw."""
    assert step_zoom(MAX_ZOOM, 5) == MAX_ZOOM
    assert step_zoom(MIN_ZOOM, -5) == MIN_ZOOM


def test_clamp_zoom_holds_the_range() -> None:
    """Any caller-supplied factor lands inside the limits."""
    assert clamp_zoom(0.0) == MIN_ZOOM
    assert clamp_zoom(1000.0) == MAX_ZOOM
    assert clamp_zoom(1.0) == 1.0


def test_zoom_label_reads_as_a_percentage() -> None:
    """The toolbar shows whole percent, rounded."""
    assert zoom_label(1.0) == "100%"
    assert zoom_label(0.336) == "34%"
    assert zoom_label(8.0) == "800%"


def test_anchored_scroll_keeps_the_viewport_centered() -> None:
    """Zooming in doubles the offset of what was centered, plus half a viewport of new image."""
    assert anchored_scroll(100, 400, 2.0) == 400  # noqa: PLR2004 - 2*100 + 1*400/2
    # Unchanged zoom must not move the view at all.
    assert anchored_scroll(137, 400, 1.0) == 137  # noqa: PLR2004 - the same scroll position


def test_anchored_scroll_clamps_below_zero() -> None:
    """Zooming out far enough asks for a negative offset; a scrollbar has no such position."""
    assert anchored_scroll(0, 400, 0.25) == 0
