"""Tests for the Qt-free GUI helpers (no PySide6, no display required)."""

from pathlib import Path

from photo_tagger.gui_state import (
    ADDED,
    DEFAULT_GUI_EXTENSIONS,
    PROVIDER_LABELS,
    READY,
    REMOVED,
    SAVED,
    UNCHANGED,
    FolderNode,
    PhotoItem,
    Proposal,
    apply_proposal,
    build_tree,
    descendant_files,
    expand_inputs,
    format_existing_keywords,
    group_by_parent,
    hierarchy_preview,
    keyword_diff,
    keywords_to_save,
    keywords_to_text,
    new_paths,
    parse_keyword_lines,
    paths_under,
    rank_vision_models,
    status_summary,
)
from photo_tagger.models import KeywordSet
from photo_tagger.providers import PROVIDER_NAMES


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
    )
    apply_proposal(item, proposal)
    assert item.status == READY
    assert item.has_proposal is True
    assert item.loaded is True
    assert item.existing_title == "Old"
    assert item.existing_keywords.subject == ["Beach"]
    assert item.title == "Golden Eagle"
    assert item.keywords == ["Eagle", "Sky"]
    # The editable copy is independent of the proposal's list.
    item.keywords.append("Extra")
    assert proposal.keywords == ["Eagle", "Sky"]


def test_build_tree_groups_files_under_their_folder() -> None:
    """A single folder of files becomes one top node labelled with its absolute path."""
    forest = build_tree([Path("/photos/a.jpg"), Path("/photos/b.jpg")])
    assert len(forest) == 1
    assert forest[0].path == Path("/photos")
    assert forest[0].label == "/photos"
    assert forest[0].files == [Path("/photos/a.jpg"), Path("/photos/b.jpg")]
    assert forest[0].folders == []


def test_build_tree_nests_subfolders_under_parent() -> None:
    """Subfolders nest under the parent folder with a relative label."""
    forest = build_tree([Path("/p/a.jpg"), Path("/p/sub/b.jpg")])
    assert len(forest) == 1
    top = forest[0]
    assert top.label == "/p"
    assert top.files == [Path("/p/a.jpg")]
    assert [f.label for f in top.folders] == ["sub"]
    assert top.folders[0].files == [Path("/p/sub/b.jpg")]


def test_build_tree_collapses_single_child_chains() -> None:
    """A chain of single, file-less folders collapses into one labelled node."""
    forest = build_tree([Path("/p/x/y/b.jpg")])
    assert len(forest) == 1
    assert forest[0].label == "/p/x/y"
    assert forest[0].files == [Path("/p/x/y/b.jpg")]


def test_build_tree_keeps_a_branching_single_root_together() -> None:
    """A folder whose subfolders each hold files stays one top node with both children."""
    forest = build_tree([Path("/p/a/x.jpg"), Path("/p/b/y.jpg")])
    assert len(forest) == 1
    assert forest[0].label == "/p"
    assert sorted(f.label for f in forest[0].folders) == ["a", "b"]


def test_build_tree_splits_disjoint_roots() -> None:
    """Files under unrelated roots become separate top-level nodes (no '/' wrapper)."""
    forest = build_tree([Path("/r1/x.jpg"), Path("/r2/y.jpg")])
    assert sorted(node.label for node in forest) == ["/r1", "/r2"]


def test_build_tree_empty() -> None:
    """No paths yields no nodes."""
    assert build_tree([]) == []


def test_descendant_files_recurses() -> None:
    """descendant_files gathers files from a node and all its subfolders."""
    forest = build_tree([Path("/p/a.jpg"), Path("/p/sub/b.jpg")])
    assert descendant_files(forest[0]) == [Path("/p/a.jpg"), Path("/p/sub/b.jpg")]


def test_paths_under_filters_by_folder() -> None:
    """paths_under keeps only the paths beneath a folder, preserving order."""
    paths = [Path("/a/1.jpg"), Path("/b/2.jpg"), Path("/a/sub/3.jpg")]
    assert paths_under(paths, Path("/a")) == [Path("/a/1.jpg"), Path("/a/sub/3.jpg")]
    assert paths_under(paths, Path("/b")) == [Path("/b/2.jpg")]


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


def test_format_existing_keywords_shows_flat_and_hierarchy() -> None:
    """Existing keywords render as a flat list plus a hierarchy section when present."""
    kw = KeywordSet(subject=["Bird", "Sky"], hierarchical=["Animal|Bird"])
    text = format_existing_keywords(kw)
    assert "Bird, Sky" in text
    assert "Hierarchy:" in text
    assert "Animal|Bird" in text
    assert format_existing_keywords(KeywordSet()) == ""


def test_hierarchy_preview_shows_resulting_paths() -> None:
    """The preview shows the Lightroom paths that saving the edited keywords would write."""
    preview = hierarchy_preview(KeywordSet(), ["Duck<Bird<Animal"], overwrite=True)
    assert "Animal|Bird|Duck" in preview


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
