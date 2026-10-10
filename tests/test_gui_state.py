"""Tests for the Qt-free GUI helpers (no PySide6, no display required)."""

import os
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from photo_tagger import telemetry
from photo_tagger.diagnostics import CheckResult
from photo_tagger.errors import ConfigFileError
from photo_tagger.fields import ExtraFields
from photo_tagger.gui_state import (
    _MAX_LISTED_DROPPED,
    ADDED,
    BADGE_COPYRIGHT,
    BADGE_FAILED,
    BADGE_METADATA,
    BADGE_MIXED_SCRIPT,
    BADGE_SAVED,
    BADGE_SIDECAR,
    BADGE_UNSAVED,
    DEFAULT_GUI_EXTENSIONS,
    FAILED,
    FILTER_ALL,
    FILTER_CHOICES,
    FILTER_FAILED,
    FILTER_GENERATED,
    FILTER_MIXED_SCRIPT,
    FILTER_NO_COPYRIGHT,
    FILTER_PENDING,
    FILTER_SAVED,
    FILTER_SELECTED,
    FILTER_UNTAGGED,
    MAX_ZOOM,
    MIN_ZOOM,
    MIXED_SCRIPT_MARK,
    OUTPUT_LANGUAGE_SUGGESTIONS,
    PENDING,
    READY,
    REMOVED,
    RERUN_FAILED,
    RERUN_MIXED_SCRIPT,
    SAVED,
    SELECT_CHECK,
    SELECT_ONLY,
    SELECT_UNCHECK,
    SORT_CHOICES,
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
    SaveJob,
    SaveOptions,
    SelectionChange,
    TaggedCriterion,
    WatchSettings,
    anchored_scroll,
    apply_proposal,
    apply_selection,
    apply_vocabulary,
    apply_write_result,
    ascii_warning,
    build_copyright_fields,
    build_save_job,
    build_tree,
    caption_source_note,
    capture_year,
    chain_to_display,
    check_results_html,
    clamp_zoom,
    config_text_with_language,
    config_text_with_output_language,
    config_toml_text,
    copyright_summary,
    deselect_paths,
    ensure_path_dirs,
    estimate_remaining,
    expand_inputs,
    extension_counts,
    extra_pass_status,
    failure_summary,
    field_tags_to_read,
    fields_written,
    file_dialog_name_filters,
    file_type_label,
    filter_photos,
    fit_zoom,
    flat_labels,
    foreign_terms_note,
    format_duration,
    format_existing_keywords,
    gui_cache_namespace,
    harmonize_sessions,
    harmonize_summary,
    hierarchy_preview,
    hierarchy_tree_text,
    journal_label,
    journal_time,
    keeps_copyright,
    keys_by_check,
    keyword_diff,
    keyword_diff_html,
    keyword_lines,
    keywords_to_save,
    keywords_to_text,
    load_sidecar_overrides,
    load_vocabulary_file,
    location_crumb,
    login_shell_path,
    mark_mixed_script,
    matches_extension,
    matches_name_pattern,
    merged_config_text,
    navigation_shortcuts,
    new_paths,
    normalize_keyword_lines,
    pacing_status_text,
    pacing_summary,
    parse_keyword_lines,
    paths_matching_fields,
    paths_under,
    photo_item_to_report_row,
    photo_matches_filter,
    photo_sort_key,
    photos_for_pass,
    photos_to_save,
    photos_to_write,
    planned_copyright,
    progress_timing_text,
    rank_vision_models,
    recheck_foreign_terms,
    reports_status,
    resolve_extra_tags,
    reveal_command,
    reveal_label,
    reverted_images,
    review_summary,
    save_options_for,
    save_tally,
    search_summary,
    sidecar_overrides_summary,
    sort_photos,
    source_label,
    status_sort_rank,
    status_summary,
    step_zoom,
    tagged_legend,
    tagged_summary,
    tagged_tooltip,
    telemetry_toggle_note,
    thumb_badges,
    tooltip,
    undo_action_label,
    undo_details_text,
    undo_summary,
    vocabulary_status,
    vocabulary_summary,
    watch_status_text,
    with_notice_written,
    wrap_tooltip,
    write_save_job,
    writes_nothing,
    zoom_label,
)
from photo_tagger.i18n import activate
from photo_tagger.metadata import (
    FIELD_COPYRIGHT,
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    CaptionValue,
    FieldSources,
)
from photo_tagger.models import KeywordSet
from photo_tagger.pacing import PacingPlan
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES
from photo_tagger.undo import CHANGED, DELETED, RESTORED, UndoResult, WriteRecord
from photo_tagger.vocabulary import Vocabulary
from photo_tagger.write_plan import WriteTarget


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


def test_check_results_html_marks_each_check_and_escapes_its_text() -> None:
    """A passed and a failed check each get their mark, and an error with markup stays text."""
    rendered = check_results_html(
        [
            CheckResult("ExifTool", ok=True, detail="13.10"),
            CheckResult("Model", ok=False, detail="<none> served"),
        ],
    )
    first, second = rendered.split("<br><br>")
    assert "&#10004;" in first
    assert "<b>ExifTool</b>" in first
    assert "&#10008;" in second
    assert "&lt;none&gt; served" in second


def test_check_results_html_is_empty_without_checks() -> None:
    """No checks render to nothing, rather than a stray separator."""
    assert check_results_html([]) == ""


def test_gui_cache_namespace_follows_the_image_size() -> None:
    """Two sizes never read each other's results: the answer depends on what the model saw."""
    small = gui_cache_namespace("m", "English", jpeg_dimensions=640)
    large = gui_cache_namespace("m", "English", jpeg_dimensions=1024)
    assert small != large


def test_gui_cache_namespace_follows_the_metadata_language() -> None:
    """Switching the metadata language never replays answers written in the old one."""
    assert gui_cache_namespace("m", "English") != gui_cache_namespace("m", "German")


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
    with patch("photo_tagger.programs.shutil.which", lambda name, **_kwargs: f"/bin/{name}"):
        assert reveal_command(photo, "darwin") == ["/bin/open", "-R", str(photo)]
        assert reveal_command(photo, "win32") == ["/bin/explorer", f"/select,{photo}"]
        assert reveal_command(photo, "linux") is None


def test_reveal_command_is_none_when_the_browser_is_not_on_path(tmp_path: Path) -> None:
    """
    The browser is resolved here, so the caller opens the folder instead of failing to spawn.

    Leaving the lookup to the OS is also how the current working directory (whichever folder the
    user last opened photos from) gets searched before PATH on Windows.
    """
    with patch("photo_tagger.programs.shutil.which", return_value=None):
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
    """Present fields compress to their letters in T/D/K/C order; none becomes a dash."""
    assert tagged_summary({FIELD_KEYWORDS, FIELD_TITLE}) == "TK"
    assert tagged_summary({FIELD_COPYRIGHT, FIELD_DESCRIPTION}) == "DC"
    assert tagged_summary(set()) == "-"


def test_tagged_legend_maps_every_letter() -> None:
    """The header legend pairs each letter with its field name, in display order."""
    assert tagged_legend() == "T = title, D = description, K = keywords, C = copyright"


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
        assert tagged_tooltip({FIELD_COPYRIGHT}) == "Já no arquivo: C = direitos autorais"
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


def test_config_writers_persist_the_pacing_plan() -> None:
    """A fresh file and a merged one both carry [pacing], which the CLI reads too."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    from photo_tagger.cli_options import load_defaults, to_pacing_plan  # noqa: PLC0415

    values = _config_values(duty=70, battery_floor=30, work_minutes=5.0)
    for text in (config_toml_text(values), merged_config_text("[pacing]\n", values)):
        plan = to_pacing_plan(load_defaults(tomllib.loads(text)).pacing)
        assert plan.duty_percent == 70  # noqa: PLR2004 - the value under test.
        assert plan.battery_floor == 30  # noqa: PLR2004 - the value under test.
        assert plan.work_seconds == 300  # noqa: PLR2004 - five minutes, in seconds.


def test_config_writers_persist_save_after_generating() -> None:
    """A fresh file and a merged one both carry it, under the window's own [gui] table."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    values = _config_values(save_after_generating=True)
    for text in (config_toml_text(values), merged_config_text("[gui]\n", values)):
        assert tomllib.loads(text)["gui"]["save_after_generating"] is True


def test_config_writers_persist_the_repair_choice() -> None:
    """A fresh file and a merged one both carry it, into the [inference] table the CLI reads."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    from photo_tagger.cli_options import load_defaults  # noqa: PLC0415

    values = _config_values(repair_language=True)
    for text in (config_toml_text(values), merged_config_text("[inference]\n", values)):
        assert load_defaults(tomllib.loads(text)).inference.repair_language is True


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


def test_apply_proposal_carries_the_existing_copyright() -> None:
    """A generated photo is never read again, so its notice must come in with the proposal."""
    item = PhotoItem(path=Path("/a.jpg"))
    proposal = Proposal(
        path=Path("/a.jpg"),
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="T",
        description="D",
        keywords=[],
        existing_copyright="© 2019 Ada",
    )
    apply_proposal(item, proposal)
    assert item.existing_copyright == "© 2019 Ada"


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

    # A notice alone is not the metadata M means, the same as for the Untagged filter. It has
    # a badge of its own.
    stamped = PhotoItem(path=Path("/a.jpg"), known_fields={FIELD_COPYRIGHT})
    assert thumb_badges(stamped, has_sidecar=False) == [BADGE_COPYRIGHT]
    both = PhotoItem(path=Path("/a.jpg"), known_fields={FIELD_TITLE, FIELD_COPYRIGHT})
    assert thumb_badges(both, has_sidecar=True) == [BADGE_METADATA, BADGE_COPYRIGHT, BADGE_SIDECAR]

    # Mixed script is informational: it marks the generated thumbnail, it does not replace
    # the lifecycle badge that says the proposal is there.
    mixed = PhotoItem(
        path=Path("/a.jpg"),
        status=READY,
        has_proposal=True,
        foreign_terms=["陶笛"],
    )
    assert thumb_badges(mixed, has_sidecar=False) == [BADGE_UNSAVED, BADGE_MIXED_SCRIPT]


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


def test_keyword_diff_html_styles_each_state_and_escapes_keywords() -> None:
    """Added, removed, and kept keywords each read differently, and a keyword is never markup."""
    rendered = keyword_diff_html([("Eagle", ADDED), ("Beach", REMOVED), ("<Bird>", UNCHANGED)])
    added, removed, kept = rendered.split("<br>")
    assert added == '<span style="color:#3fb950">+&nbsp;Eagle</span>'
    assert "line-through" in removed
    assert "Beach" in removed
    assert "&lt;Bird&gt;" in kept
    assert "color:#8a8a8a" in kept


def test_keyword_diff_html_says_so_when_nothing_changes() -> None:
    """An empty diff still tells the reader something."""
    assert keyword_diff_html([]) == "(no change)"


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


def test_keys_by_check_splits_the_list_by_checkbox() -> None:
    """Checked and unchecked photos come back as separate key lists, in list order."""
    a, b, c = Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")
    items = [PhotoItem(path=p) for p in (a, b, c)]
    items[1].selected = False

    assert keys_by_check(items, checked=True) == [str(a), str(c)]
    assert keys_by_check(items, checked=False) == [str(b)]


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


def test_tagged_criterion_matches_carries_its_own_rule() -> None:
    """A criterion applies the same AND/OR rule wherever it ends up, menu or parked scan."""
    a, b = Path("/a.jpg"), Path("/b.jpg")
    presence = {a: {"title", "description"}, b: {"title"}}
    both = TaggedCriterion(frozenset({"title", "description"}), match_all=True, phrase="both")
    either = TaggedCriterion(frozenset({"title", "description"}), match_all=False, phrase="either")

    assert both.matches(presence) == {a}
    assert either.matches(presence) == {a, b}


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


def test_sort_choices_offer_each_sort_once_name_first() -> None:
    """The Sort by menu lists every sort once, with a label, and opens on the default."""
    criteria = [criterion for _label, criterion in SORT_CHOICES]
    assert criteria == [SORT_NAME, SORT_TYPE, SORT_STATUS, SORT_TAGGED]
    assert all(label for label, _criterion in SORT_CHOICES)


def test_filter_choices_offer_each_filter_once_all_first() -> None:
    """The Show menu lists every filter once, with its own label, and opens on All."""
    criteria = [criterion for _label, criterion in FILTER_CHOICES]
    assert criteria[0] == FILTER_ALL
    assert len(set(criteria)) == len(criteria)
    assert set(criteria) == {
        FILTER_ALL,
        FILTER_SELECTED,
        FILTER_PENDING,
        FILTER_GENERATED,
        FILTER_SAVED,
        FILTER_FAILED,
        FILTER_UNTAGGED,
        FILTER_NO_COPYRIGHT,
        FILTER_MIXED_SCRIPT,
    }
    labels = [label for label, _criterion in FILTER_CHOICES]
    assert all(labels)
    assert len(set(labels)) == len(labels)


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


def test_filter_untagged_ignores_a_copyright_alone() -> None:
    """A camera stamps a notice on every frame, and that photo still waits for its tags."""
    stamped = PhotoItem(path=Path("/d/a.jpg"), known_fields={FIELD_COPYRIGHT})
    assert filter_photos([stamped], FILTER_UNTAGGED) == [stamped]


def test_filter_no_copyright_matches_only_scanned_photos_without_one() -> None:
    """Other tags do not count, and a photo the scan has not reached yet is left out."""
    bare = PhotoItem(path=Path("/d/a.jpg"), known_fields=set())
    titled = PhotoItem(path=Path("/d/b.jpg"), known_fields={FIELD_TITLE})
    owned = PhotoItem(path=Path("/d/c.jpg"), known_fields={FIELD_TITLE, FIELD_COPYRIGHT})
    unscanned = PhotoItem(path=Path("/d/d.jpg"), known_fields=None)
    photos = [bare, titled, owned, unscanned]
    assert filter_photos(photos, FILTER_NO_COPYRIGHT) == [bare, titled]


def test_filter_mixed_script_finds_the_photos_the_run_flagged() -> None:
    """The count in the status line has to turn into rows, which is what this filter is."""
    flagged = PhotoItem(path=Path("/d/a.jpg"), foreign_terms=["陶笛"])
    clean = PhotoItem(path=Path("/d/b.jpg"), has_proposal=True)
    assert filter_photos([flagged, clean], FILTER_MIXED_SCRIPT) == [flagged]


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
        "photo_tagger.gui_state.environment.subprocess.run",
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

    monkeypatch.setattr("photo_tagger.gui_state.environment.subprocess.run", boom)
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


def test_pacing_status_text_says_why_the_run_stopped() -> None:
    """A battery rest names the charge it waits for; a duty rest names the share it works."""
    battery = pacing_status_text("battery", 900.0, duty_percent=100, resume_percent=45)
    assert battery == "Resting up to 15:00 so the battery charges back to 45%."
    duty = pacing_status_text("duty", 257.0, duty_percent=70, resume_percent=0)
    assert duty == "Resting 4:17: this run works 70% of the time."


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


def test_review_summary_is_silent_when_no_photo_is_flagged() -> None:
    """A run where every answer stayed in one script has nothing to say about it."""
    assert review_summary(0) == ""


def test_review_summary_counts_the_photos_to_go_back_to() -> None:
    """With thousands of rows in the list, the count is what says to go click something."""
    assert "1 photo mixes writing systems" in review_summary(1)
    assert "4 photos mix writing systems" in review_summary(4)


def test_review_summary_says_how_to_reach_those_photos() -> None:
    """A count with no route left people scrolling a long list for a row they could not spot."""
    assert "Select > Check Only > Status" in review_summary(3)


def test_mark_mixed_script_marks_only_a_flagged_row() -> None:
    """The Status cell carries the mark so a flagged photo is visible without the filter on."""
    assert mark_mixed_script("ready", ["陶笛"]) == "ready ⚠"
    assert mark_mixed_script("ready", []) == "ready"
    # A photo with no status text yet still has to show the mark rather than a leading space.
    assert mark_mixed_script("", ["陶笛"]) == "⚠"


def test_foreign_terms_note_names_the_words_to_check() -> None:
    """The detail pane points at the exact terms, and says nothing when there are none."""
    assert foreign_terms_note([]) == ""
    note = foreign_terms_note(["陶笛", "居酒屋"])
    assert "陶笛, 居酒屋" in note


def test_recheck_foreign_terms_clears_a_flag_the_edit_fixed() -> None:
    """A hand fix of the flagged words takes the photo out of the filter and the re-run pass."""
    item = PhotoItem(path=Path("/a.jpg"), title="Decorated陶笛", foreign_terms=["陶笛"])

    assert recheck_foreign_terms(item) is False  # nothing changed yet
    item.title = "Decorated Ocarina"

    assert recheck_foreign_terms(item) is True
    assert item.foreign_terms == []


def test_recheck_foreign_terms_names_what_is_still_mixed() -> None:
    """A partial fix keeps the flag, naming only the words still left."""
    item = PhotoItem(
        path=Path("/a.jpg"),
        title="Decorated Ocarina",
        keywords=["居酒屋<Restaurant", "Ocarina"],
        foreign_terms=["陶笛", "居酒屋"],
    )

    assert recheck_foreign_terms(item) is True
    assert item.foreign_terms == ["居酒屋"]


def test_recheck_foreign_terms_never_flags_what_a_person_typed() -> None:
    """Only a flag the model's answer raised is checked again: typed text is the person's own."""
    item = PhotoItem(path=Path("/a.jpg"), title="Decorated陶笛 Ocarina in a Garden")

    assert recheck_foreign_terms(item) is False
    assert item.foreign_terms == []


def test_apply_proposal_carries_the_review_flag() -> None:
    """A proposal that mixes writing systems arrives at the detail pane already marked."""
    item = PhotoItem(path=Path("/a.jpg"))
    proposal = Proposal(
        path=Path("/a.jpg"),
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="Decorated陶笛",
        description="A陶笛 on a table.",
        keywords=[],
        foreign_terms=["陶笛"],
    )
    apply_proposal(item, proposal)
    assert item.foreign_terms == ["陶笛"]
    assert photo_item_to_report_row(item, overwrite=False).as_dict()["foreign_terms"] == "陶笛"


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


def test_reverted_images_names_each_photo_an_undo_put_back_once() -> None:
    """A photo whose sidecar and file were both put back is named once; a refused one is not."""
    records = [
        WriteRecord(image="/p/a.cr3", target="/p/a.xmp", created=True),
        WriteRecord(image="/p/a.cr3", target="/p/a.cr3", created=False),
        WriteRecord(image="/p/b.cr3", target="/p/b.xmp", created=True),
    ]
    results = [
        UndoResult(Path("/p/a.xmp"), DELETED),
        UndoResult(Path("/p/a.cr3"), RESTORED),
        UndoResult(Path("/p/b.xmp"), CHANGED, "edited since"),
    ]
    assert reverted_images(records, results) == ["/p/a.cr3"]


def test_reverted_images_ignores_a_result_no_record_explains() -> None:
    """An outcome for a file the journal never listed names no photo."""
    results = [UndoResult(Path("/elsewhere/c.xmp"), DELETED)]
    assert reverted_images([], results) == []


def test_undo_details_text_lists_what_happened_to_each_file() -> None:
    """After an undo, each file gets one line, with the reason when it was left alone."""
    results = [
        UndoResult(Path("/photos/a.xmp"), RESTORED),
        UndoResult(Path("/photos/b.xmp"), CHANGED, "edited on 2 May"),
    ]
    assert undo_details_text(results, dry_run=False).splitlines() == [
        "What was undone:",
        f"  {undo_action_label(RESTORED)}: a.xmp",
        f"  {undo_action_label(CHANGED)}: b.xmp (edited on 2 May)",
    ]


def test_undo_details_text_preview_leaves_out_the_reasons() -> None:
    """A preview says what would happen; the reasons only matter once something was refused."""
    results = [UndoResult(Path("/photos/b.xmp"), CHANGED, "edited on 2 May")]
    assert undo_details_text(results, dry_run=True).splitlines() == [
        "What undoing would do:",
        f"  {undo_action_label(CHANGED)}: b.xmp",
    ]


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


def test_photos_to_write_takes_only_the_unsaved_proposals() -> None:
    """A save nobody reviews writes what was generated, and nothing else on the list."""
    ready = PhotoItem(path=Path("/a.jpg"), status=READY, has_proposal=True)
    saved = PhotoItem(path=Path("/b.jpg"), status=SAVED, has_proposal=True)
    failed = PhotoItem(path=Path("/c.jpg"), status=FAILED, has_proposal=True)
    pending = PhotoItem(path=Path("/d.jpg"))

    assert photos_to_write([ready, saved, failed, pending]) == [ready]


def test_photos_for_pass_picks_what_each_extra_pass_re_runs() -> None:
    """One pass takes the photos with no answer, the other the answers in two scripts."""
    failed = PhotoItem(path=Path("/a.jpg"), status=FAILED)
    mixed = PhotoItem(
        path=Path("/b.jpg"),
        status=READY,
        has_proposal=True,
        foreign_terms=["陶笛"],
    )
    clean = PhotoItem(path=Path("/c.jpg"), status=READY, has_proposal=True)
    batch = [failed, mixed, clean]

    assert photos_for_pass(RERUN_FAILED, batch) == [failed]
    assert photos_for_pass(RERUN_MIXED_SCRIPT, batch) == [mixed]


def test_extra_pass_status_says_which_photos_it_is_asking_about() -> None:
    """The status bar has to explain a run that started itself, plural included."""
    assert extra_pass_status(RERUN_FAILED, 1) == "Retrying 1 photo that failed..."
    assert extra_pass_status(RERUN_FAILED, 4) == "Retrying 4 photos that failed..."
    mixed = extra_pass_status(RERUN_MIXED_SCRIPT, 2)
    assert mixed == "Retrying 2 photos that mixed writing systems..."


def test_failure_summary_names_the_button_that_tries_again() -> None:
    """A batch left alone ends on its save's tally, so the photos it never got have to show."""
    assert failure_summary(0) == ""
    assert failure_summary(1) == "1 photo failed (Retry Failed to try again)."
    assert failure_summary(3) == "3 photos failed (Retry Failed to try again)."


def test_save_tally_counts_what_a_save_wrote_out_of_what_it_set_out_to() -> None:
    """A finished save names the checked photos it was given; the plural follows that total."""
    assert save_tally(1, 1, cancelled=False) == "Saved 1 of 1 checked photo."
    assert save_tally(2, 3, cancelled=False) == "Saved 2 of 3 checked photos."


def test_save_tally_after_cancel_counts_only_what_was_written() -> None:
    """A cancelled save drops the total, which no longer says what it will do."""
    assert save_tally(1, 5, cancelled=True) == "Cancelled after saving 1 photo."
    assert save_tally(0, 5, cancelled=True) == "Cancelled after saving 0 photos."


def test_save_tally_keeps_the_chained_generations_note_after_it() -> None:
    """The note a generation left for its save follows the tally; an empty note adds nothing."""
    note = "1 photo failed (Retry Failed to try again)."
    assert save_tally(2, 2, cancelled=False, note=note) == f"Saved 2 of 2 checked photos. {note}"
    assert not save_tally(2, 2, cancelled=False, note="").endswith(" ")


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


# ---------------------------------------------------------------------------
# Copyright and credit
# ---------------------------------------------------------------------------


_NOTICE = "Copyright (c) {year} {artist}"


def _taken(year: int) -> FieldSources:
    """Return what a photo taken in *year* carries in the file itself, with nothing else on it."""
    return FieldSources(image={"EXIF:DateTimeOriginal": f"{year}:05:04 10:00:00"})


_EMBEDDED = [WriteTarget(path=Path("/a.jpg"), use_sidecar=False, existed=True)]
_SIDECAR = [WriteTarget(path=Path("/a.xmp"), use_sidecar=True, existed=False)]


def _copyright_options(**kwargs: object) -> SaveOptions:
    """Return a SaveOptions carrying a usable copyright block."""
    fields, error = build_copyright_fields(
        artist="Ada Lovelace",
        notice=_NOTICE,
        rights="",
        marked=True,
        **kwargs,  # type: ignore[arg-type]
    )
    assert error == ""
    return SaveOptions(extra_fields=fields)


def test_save_options_writes_anything_covers_a_copyright_only_save() -> None:
    """Every generated field off still writes, when a copyright is configured."""
    bare = SaveOptions(write_title=False, write_description=False, write_keywords=False)
    assert not bare.writes_anything
    assert not bare.writes_copyright

    stamped = replace(bare, extra_fields=_copyright_options().extra_fields)
    assert stamped.writes_copyright
    assert stamped.writes_anything
    # The three field toggles are still off, which is what any_field reports.
    assert not stamped.any_field


def test_save_options_ignores_an_empty_copyright() -> None:
    """A dialog nobody filled in is not a reason to write."""
    options = SaveOptions(extra_fields=ExtraFields())
    assert not options.writes_copyright
    assert not options.writes_notice


def test_save_job_counts_the_notice_among_the_fields_it_writes() -> None:
    """After the save, No copyright lets go of the photo without a rescan."""
    job = build_save_job(_proposed_item(), _copyright_options())
    assert FIELD_COPYRIGHT in job.fields
    assert FIELD_COPYRIGHT not in build_save_job(_proposed_item(), SaveOptions()).fields


def test_save_job_does_not_count_a_name_alone_as_a_notice() -> None:
    """A credit with no notice and no rights leaves the photo without a copyright."""
    options = SaveOptions(extra_fields=ExtraFields(artist="Ada Lovelace", marked=True))
    assert options.writes_copyright
    assert not options.writes_notice
    assert FIELD_COPYRIGHT not in build_save_job(_proposed_item(), options).fields
    rights_only = SaveOptions(extra_fields=ExtraFields(rights="© {year}"))
    assert rights_only.writes_notice


def test_photos_to_save_adds_the_photos_never_generated_only_for_a_copyright() -> None:
    """Without a notice to stamp, a photo that was never generated has nothing to save."""
    generated = PhotoItem(path=Path("/a.jpg"), has_proposal=True)
    tagged = PhotoItem(path=Path("/b.jpg"))
    unchecked = PhotoItem(path=Path("/c.jpg"), selected=False, has_proposal=True)
    photos = [generated, tagged, unchecked]
    assert photos_to_save(photos, SaveOptions()) == [generated]
    assert photos_to_save(photos, _copyright_options()) == [generated, tagged]


def test_save_options_for_keeps_only_the_copyright_on_a_photo_never_generated() -> None:
    """Its text fields are the file's own values, so they are not written back."""
    options = _copyright_options()
    narrowed = save_options_for(PhotoItem(path=Path("/a.jpg")), options)
    assert not narrowed.any_field
    assert narrowed.writes_copyright
    assert save_options_for(PhotoItem(path=Path("/a.jpg"), has_proposal=True), options) is options


def test_resolve_extra_tags_dates_the_notice_by_the_photo() -> None:
    """The year comes from the batch read, so one photo's notice is not another's."""
    old, new = Path("/photos/old.jpg"), Path("/photos/new.jpg")
    options = _copyright_options()
    sources = {old: _taken(2011), new: _taken(2025)}

    assert resolve_extra_tags(options, old, sources)["EXIF:Copyright"] == (
        "Copyright (c) 2011 Ada Lovelace"
    )
    assert resolve_extra_tags(options, new, sources)["EXIF:Copyright"] == (
        "Copyright (c) 2025 Ada Lovelace"
    )


def test_resolve_extra_tags_is_empty_without_a_copyright() -> None:
    """The usual save resolves nothing, so it pays no read and writes no extra tag."""
    assert resolve_extra_tags(SaveOptions(), Path("/photos/a.jpg"), {}) == {}


def test_resolve_extra_tags_leaves_an_undated_photo_undated() -> None:
    """A photo missing from the read gets a notice without a year, never today's."""
    tags = resolve_extra_tags(_copyright_options(), Path("/photos/a.jpg"), {})
    assert tags["EXIF:Copyright"] == "Copyright (c) Ada Lovelace"


def test_build_copyright_fields_reports_a_template_it_cannot_fill() -> None:
    """The dialog gets the reason as text, rather than an exception out of a Qt slot."""
    fields, error = build_copyright_fields(
        artist="",
        notice="Copyright (c) {year} {shot_at}",
        rights="",
        marked=None,
    )
    assert fields is None
    assert "{shot_at}" in error


def test_build_copyright_fields_carries_the_preserve_choice() -> None:
    """The dialog's checkbox decides whether a save replaces what a photo has."""
    fields, _error = build_copyright_fields(
        artist="Ada",
        notice="",
        rights="",
        marked=None,
        preserve=False,
    )
    assert fields is not None
    assert not fields.preserve


def test_build_copyright_fields_trims_what_was_typed() -> None:
    """Stray spaces around a name are not part of the credit."""
    fields, error = build_copyright_fields(
        artist="  Ada Lovelace  ",
        notice="",
        rights="",
        marked=None,
    )
    assert error == ""
    assert fields is not None
    assert fields.resolve({})["EXIF:Artist"] == "Ada Lovelace"


def test_load_sidecar_overrides_parses_the_config_entries() -> None:
    """The window writes a DNG where the CLI would, from the same config key."""
    assert load_sidecar_overrides(["dng=photo"]) == ({".dng": "photo"}, "")


def test_load_sidecar_overrides_reports_a_bad_entry_instead_of_raising() -> None:
    """A config the CLI refuses must not keep the window from opening; the reason is logged."""
    overrides, error = load_sidecar_overrides(["dng=embed"])

    assert overrides == {}
    assert "EXT=TARGET" in error


def test_sidecar_overrides_summary_names_each_type() -> None:
    """The Save tooltip says where each overridden type goes, since the menu cannot show it."""
    summary = sidecar_overrides_summary({".tif": "both", ".dng": "photo"})

    assert summary == "DNG into the image file, TIF into the image file and an XMP sidecar"
    assert sidecar_overrides_summary({}) == ""


def test_build_copyright_fields_keeps_config_only_assignments() -> None:
    """set_tags has no dialog row, so the window carries it through from the config file."""
    options = _copyright_options(assignments=["XMP-xmpRights:WebStatement=https://x.test/terms"])
    assert options.extra_fields is not None
    resolved = options.extra_fields.resolve({})
    assert resolved["XMP-xmpRights:WebStatement"] == "https://x.test/terms"


def test_copyright_summary_lists_every_tag_for_a_real_year() -> None:
    """The preview is what tells the user their {year} template came out right."""
    options = _copyright_options()
    summary = copyright_summary(options.extra_fields, 2019)
    assert "2019" in summary
    assert "EXIF:Copyright = Copyright (c) 2019 Ada Lovelace" in summary
    assert "XMP-xmpRights:Marked = True" in summary


def test_planned_copyright_shows_the_xmp_wording_for_the_photo() -> None:
    """The detail pane compares this with the file's notice, which is read XMP first."""
    fields = ExtraFields(notice="(c) {year} Ada", rights="© {year} Ada")
    assert planned_copyright(fields, _taken(2019)) == "© 2019 Ada"
    assert planned_copyright(ExtraFields(notice="(c) {year} Ada"), _taken(2019)) == "(c) 2019 Ada"


def test_planned_copyright_is_none_without_a_notice() -> None:
    """No settings, or a name alone, put no notice on the photo."""
    assert planned_copyright(None, _taken(2019)) is None
    assert planned_copyright(ExtraFields(), _taken(2019)) is None
    assert planned_copyright(ExtraFields(artist="Ada Lovelace"), _taken(2019)) is None


def test_keeps_copyright_when_the_photo_has_a_notice_and_preserve_is_on() -> None:
    """The detail pane says the notice stays, instead of showing one that is never written."""
    fields = ExtraFields(notice="(c) {year} Ada")
    has_one = FieldSources(image={"XMP:Rights": "(c) 2011 Grace"})
    assert keeps_copyright(fields, has_one, _EMBEDDED)
    assert planned_copyright(fields, has_one, _EMBEDDED) is None
    assert keeps_copyright(ExtraFields(marked=True), has_one, _EMBEDDED)
    assert not keeps_copyright(fields, FieldSources(), _EMBEDDED)
    assert not keeps_copyright(ExtraFields(notice="(c)", preserve=False), has_one, _EMBEDDED)
    assert not keeps_copyright(ExtraFields(artist="Ada"), has_one, _EMBEDDED)
    assert not keeps_copyright(None, has_one, _EMBEDDED)


def test_keeps_copyright_looks_only_at_the_files_the_save_writes() -> None:
    """A notice in the photo does not stop a save that writes only its sidecar, and back."""
    fields = ExtraFields(notice="(c) {year} Ada")
    in_photo = FieldSources(image={"XMP:Rights": "(c) 2011 Grace"})
    in_sidecar = FieldSources(sidecar={"XMP:Rights": "(c) 2011 Grace"})
    assert not keeps_copyright(fields, in_photo, _SIDECAR)
    assert not keeps_copyright(fields, in_sidecar, _EMBEDDED)
    assert keeps_copyright(fields, in_sidecar, _SIDECAR)


def test_capture_year_reads_the_first_usable_date() -> None:
    """The preview year follows the same chain as {year}, and nothing else."""
    shot_in = 2011
    values = {"EXIF:DateTimeOriginal": "0000:00:00", "IPTC:DateCreated": str(shot_in)}
    assert capture_year(FieldSources(sidecar=values)) == shot_in
    assert capture_year(FieldSources(image={"File:FileModifyDate": "2025:01:01"})) is None


def test_writes_nothing_only_when_the_job_and_the_copyright_are_both_empty() -> None:
    """A photo that keeps its notice and gets no generated field is done, not failed."""
    bare = SaveJob(path=Path("/a.jpg"), keywords=KeywordSet(), title=None, description=None)
    assert writes_nothing(bare, {})
    assert not writes_nothing(bare, {"EXIF:Artist": "Ada"})
    assert not writes_nothing(replace(bare, title="T"), {})


def test_write_save_job_touches_no_file_when_there_is_nothing_to_write() -> None:
    """A photo that keeps its notice and gets no generated field is a save that worked."""
    bare = SaveJob(path=Path("/a.jpg"), keywords=KeywordSet(), title=None, description=None)
    with patch("photo_tagger.gui_state.saving.write_metadata_everywhere") as write:
        assert write_save_job(bare, {}, SaveOptions()) == (True, [])
    write.assert_not_called()


def test_write_save_job_hands_the_save_options_to_the_writer() -> None:
    """The single save and the batch write a photo the same way: through these options."""
    job = SaveJob(path=Path("/a.jpg"), keywords=KeywordSet(), title="T", description=None)
    options = SaveOptions(sidecar_mode="both", backup=False)
    with patch(
        "photo_tagger.gui_state.saving.write_metadata_everywhere",
        return_value=(True, []),
    ) as write:
        assert write_save_job(job, {"EXIF:Artist": "Ada"}, options) == (True, [])
    kwargs = write.call_args.kwargs
    assert (kwargs["sidecar_mode"], kwargs["backup"], kwargs["title"]) == ("both", False, "T")
    assert kwargs["extra"] == {"EXIF:Artist": "Ada"}


def test_field_tags_to_read_covers_the_defaults_and_the_templates() -> None:
    """The load reads what the default chains need, and what a custom template names."""
    tags = field_tags_to_read(ExtraFields(notice="{EXIF:Model}"))
    assert "EXIF:DateTimeOriginal" in tags
    assert "XMP:Rights" in tags
    assert "EXIF:Model" in tags
    assert field_tags_to_read(None) == field_tags_to_read(ExtraFields())


def test_copyright_summary_says_when_nothing_is_written() -> None:
    """An empty dialog explains itself instead of showing an empty preview."""
    assert "Nothing is written" in copyright_summary(None, 2025)
    assert "Nothing is written" in copyright_summary(ExtraFields(), 2025)


def test_config_toml_text_writes_the_copyright_table() -> None:
    """Save Settings as Defaults hands the CLI the same [copyright] block it reads."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    text = config_toml_text(
        _config_values(
            artist="Ada Lovelace",
            copyright_notice=_NOTICE,
            rights="© {year} {artist}",
            marked=True,
        ),
    )
    parsed = tomllib.loads(text)
    assert parsed["copyright"] == {
        "artist": "Ada Lovelace",
        "copyright_notice": _NOTICE,
        "rights": "© {year} {artist}",
        "marked": True,
    }


def test_config_toml_text_leaves_out_a_rights_flag_nobody_picked() -> None:
    """A credit on its own writes no Marked key, so the photo keeps whatever it says."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    text = config_toml_text(_config_values(artist="Ada Lovelace"))
    assert tomllib.loads(text)["copyright"] == {"artist": "Ada Lovelace"}


def test_config_toml_text_omits_an_unused_copyright_table() -> None:
    """A user who never opened the dialog gets no empty table in their config."""
    assert "[copyright]" not in config_toml_text(_config_values())


def test_merged_config_text_updates_and_clears_copyright_keys() -> None:
    """Emptying a field removes its key, so the CLI stops writing it too."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    text = merged_config_text(
        '[copyright]\nartist = "Old Name"\ncopyright_notice = "(c) old"\nmarked = true\n',
        _config_values(artist="Ada Lovelace", marked=False),
    )
    parsed = tomllib.loads(text)
    assert parsed["copyright"] == {"artist": "Ada Lovelace", "marked": False}


def test_config_text_writes_preserve_copyright_only_when_it_is_off() -> None:
    """Keeping is the default, so only the choice to replace takes a line, and a merge clears it."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    text = config_toml_text(_config_values(preserve_copyright=False))
    assert tomllib.loads(text)["copyright"] == {"preserve_copyright": False}
    merged = merged_config_text(text, _config_values(artist="Ada"))
    assert tomllib.loads(merged)["copyright"] == {"artist": "Ada"}
    fresh = merged_config_text("", _config_values(preserve_copyright=False))
    assert tomllib.loads(fresh)["copyright"] == {"preserve_copyright": False}


def test_merged_config_text_drops_a_copyright_table_left_empty() -> None:
    """Clearing every field removes the table rather than leaving a hollow one behind."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    text = merged_config_text('[copyright]\nartist = "Old Name"\n', _config_values())
    assert "copyright" not in tomllib.loads(text)


def test_merged_config_text_leaves_a_file_without_copyright_alone() -> None:
    """A config that never mentioned copyright does not grow an empty table on save."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    text = merged_config_text("[provider]\nmodel_name = 'm'\n", _config_values())
    assert "copyright" not in tomllib.loads(text)


def test_ascii_warning_flags_a_notice_exif_cannot_promise() -> None:
    """A © in the notice rides in a field EXIF types as ASCII, and the preview says so."""
    fields, _error = build_copyright_fields(
        artist="José",
        notice="Copyright © {year} {artist}",
        rights="",
        marked=None,
    )
    warning = ascii_warning(fields, 2025)
    assert "EXIF:Artist, EXIF:Copyright" in warning
    assert "Exif 3.0" in warning
    # The tag list itself stays a plain list of what gets written.
    assert "Exif 3.0" not in copyright_summary(fields, 2025)


def test_ascii_warning_stays_quiet_for_a_plain_notice() -> None:
    """A © in XMP alone is correct there, so nothing is flagged."""
    fields, _error = build_copyright_fields(
        artist="Ada",
        notice="Copyright (c) {year} {artist}",
        rights="© {year} {artist}",
        marked=None,
    )
    assert "Exif 3.0" not in copyright_summary(fields, 2025)


def test_config_writers_persist_the_write_copyright_toggle() -> None:
    """Save Settings as Defaults carries the toggle, so CLI runs skip the block too."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    values = _config_values(write_copyright=False)
    assert tomllib.loads(config_toml_text(values))["output"]["write_copyright"] is False
    merged = merged_config_text("[output]\nwrite_title = true\n", values)
    assert tomllib.loads(merged)["output"]["write_copyright"] is False


_BOTH_REASONS = 2  # one line for the duty cycle, one for the battery


def test_pacing_summary_describes_a_plan_that_never_rests() -> None:
    """With nothing asked for, the summary says the run does not stop."""
    assert pacing_summary(PacingPlan()) == "Left as it is, a run works flat out."


def test_pacing_summary_turns_a_duty_cycle_into_minutes() -> None:
    """A percentage does not say how long a run stops for, so the summary works it out."""
    summary = pacing_summary(PacingPlan(duty_percent=90, work_seconds=600))
    assert summary == (
        "Works 90% of the time: about 1 minute of rest after every 10 minutes of work."
    )


def test_pacing_summary_reports_a_rest_too_short_to_matter() -> None:
    """A duty cycle set close to 100 earns seconds, and saying so is the point."""
    summary = pacing_summary(PacingPlan(duty_percent=99, work_seconds=600))
    assert "about 6 seconds of rest" in summary


def test_pacing_summary_describes_the_battery_floor() -> None:
    """The battery half names the charge it waits for and the cap on the wait."""
    plan = PacingPlan(battery_floor=30, battery_resume=60, max_pause_seconds=900)
    assert pacing_summary(plan) == (
        "Whenever the battery falls to 30%, rests until it is back to 60%, for at most 15 minutes."
    )


def test_pacing_summary_covers_both_reasons_at_once() -> None:
    """A plan that rests for both reasons describes both, one per line."""
    plan = PacingPlan(duty_percent=70, battery_floor=30, work_seconds=600)
    lines = pacing_summary(plan).split("\n")
    assert len(lines) == _BOTH_REASONS
    assert lines[0].startswith("Works 70% of the time")
    assert lines[1].startswith("Whenever the battery falls to 30%")


def test_pacing_summary_flags_a_floor_near_full() -> None:
    """A floor up in the nineties is kept and flagged, since it rests after nearly every window."""
    summary = pacing_summary(PacingPlan(battery_floor=95))
    assert "falls to 95%" in summary
    assert summary.splitlines()[-1].startswith(MIXED_SCRIPT_MARK)


def test_pacing_summary_leaves_an_ordinary_floor_unflagged() -> None:
    """A floor anyone would use carries no warning at all."""
    assert MIXED_SCRIPT_MARK not in pacing_summary(PacingPlan(battery_floor=30))


# --- telemetry toggle ----------------------------------------------------------------------------


def test_telemetry_toggle_note_explains_what_the_environment_decided() -> None:
    """The toggle only carries a note when an environment variable, not the box, decides."""
    assert "DO_NOT_TRACK" in telemetry_toggle_note(
        telemetry.CONSENT_DO_NOT_TRACK,
        do_not_track=True,
    )
    assert "PHOTO_TAGGER_NO_TELEMETRY" in telemetry_toggle_note(
        telemetry.CONSENT_ENV_OPT_OUT,
        do_not_track=False,
    )
    assert "turned this on here" in telemetry_toggle_note(
        telemetry.CONSENT_GUI_OPT_IN,
        do_not_track=True,
    )
    assert "PHOTO_TAGGER_TELEMETRY=1" in telemetry_toggle_note(
        telemetry.CONSENT_ENV_OPT_IN,
        do_not_track=True,
    )


@pytest.mark.parametrize(
    "reason",
    [
        telemetry.CONSENT_DEFAULT,
        telemetry.CONSENT_SETTING_OFF,
        telemetry.CONSENT_GUI_OPT_IN,
        telemetry.CONSENT_ENV_OPT_IN,
    ],
)
def test_telemetry_toggle_note_stays_quiet_without_do_not_track(reason: str) -> None:
    """With DO_NOT_TRACK unset, an opt-in overrules nothing, so there is nothing to explain."""
    assert telemetry_toggle_note(reason, do_not_track=False) == ""


def test_apply_proposal_records_the_proposal_it_showed(tmp_path: Path) -> None:
    """The first-shown proposal, its time, and its hint are kept to count later edits against."""
    item = PhotoItem(path=tmp_path / "a.jpg")
    proposal = Proposal(
        path=item.path,
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="Heron",
        description="A heron.",
        keywords=["heron"],
        hinted=True,
    )
    apply_proposal(item, proposal, now=12.5)
    assert (item.proposed_title, item.proposed_description) == ("Heron", "A heron.")
    assert item.proposed_keywords == item.keywords
    assert (item.proposed_at, item.hinted) == (12.5, True)
    item.title = "Grey Heron"
    assert item.proposed_title == "Heron", "editing the field leaves the baseline alone"


def _job(
    path: Path,
    *,
    title: str | None = None,
    copyright_notice: bool = False,
) -> SaveJob:
    """Build a save job for *path*: copyright only unless a title is given."""
    return SaveJob(
        path=path,
        keywords=KeywordSet(),
        title=title,
        description=None,
        copyright=copyright_notice,
    )


def test_with_notice_written_follows_what_the_photo_got(tmp_path: Path) -> None:
    """A run writes a notice, but a chain with no literal can find nothing on one photo."""
    job = _job(tmp_path / "a.jpg", copyright_notice=True)

    assert with_notice_written(job, {"XMP-dc:Rights": "(c) 2011 Ada"}).copyright is True
    assert with_notice_written(job, {"XMP-dc:Creator": "Ada"}).copyright is False
    # A run that writes no notice never claims one, whatever the tags hold.
    plain = _job(tmp_path / "a.jpg")
    assert with_notice_written(plain, {"EXIF:Copyright": "(c)"}).copyright is False


def test_a_photo_left_without_a_notice_stays_in_the_no_copyright_filter(tmp_path: Path) -> None:
    """The C badge and the filter follow the notice the photo got, not the one the run meant."""
    item = PhotoItem(path=tmp_path / "a.jpg", known_fields=set())
    job = with_notice_written(_job(item.path, copyright_notice=True), {"XMP-dc:Creator": "Ada"})

    apply_write_result(item, job, ok=True)

    assert FIELD_COPYRIGHT not in (item.known_fields or set())


def test_a_copyright_only_save_leaves_the_generation_status_alone(tmp_path: Path) -> None:
    """
    A stamp says nothing about generation: a failed stamp is not a failed photo, and vice versa.

    Marking a never-generated photo FAILED would send it to Retry Failed, and the model with it.
    """
    stamp = _job(tmp_path / "a.jpg", copyright_notice=True)
    fresh = PhotoItem(path=tmp_path / "a.jpg")
    apply_write_result(fresh, stamp, ok=False)
    assert fresh.status == PENDING

    failed = PhotoItem(path=tmp_path / "a.jpg", status=FAILED)
    apply_write_result(failed, stamp, ok=True)
    assert failed.status == FAILED


def test_a_text_save_sets_the_status(tmp_path: Path) -> None:
    """Writing the generated text is what a status word reports."""
    item = PhotoItem(path=tmp_path / "a.jpg")
    apply_write_result(item, _job(tmp_path / "a.jpg", title="T"), ok=True)
    assert item.status == SAVED


def test_a_save_keeps_an_unscanned_photo_unknown(tmp_path: Path) -> None:
    """The scan has not read it, so a few written fields must not pass for all it carries."""
    item = PhotoItem(path=tmp_path / "a.jpg")
    apply_write_result(item, _job(tmp_path / "a.jpg", copyright_notice=True), ok=True)
    assert item.known_fields is None

    scanned = PhotoItem(path=tmp_path / "b.jpg", known_fields={FIELD_TITLE})
    apply_write_result(scanned, _job(tmp_path / "b.jpg", copyright_notice=True), ok=True)
    assert scanned.known_fields is not None
    assert FIELD_TITLE in scanned.known_fields
    assert len(scanned.known_fields) > 1


def test_a_hand_edit_on_a_photo_never_generated_is_a_save(tmp_path: Path) -> None:
    """Save This Photo writes a hand edit, which is news for the status word."""
    item = PhotoItem(path=tmp_path / "a.jpg")
    assert reports_status(item, _job(tmp_path / "a.jpg", title="Typed")) is True
    assert reports_status(item, _job(tmp_path / "a.jpg", copyright_notice=True)) is False
