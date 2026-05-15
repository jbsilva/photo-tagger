"""Tests for input discovery and skip-list handling."""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from photo_tagger.discovery import (
    apply_skip_file,
    apply_skip_tagged,
    filter_skipped_files,
    load_skip_list,
    make_skip_list_appender,
    parse_extensions,
    resolve_image_batch,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_parse_extensions_drops_blanks_and_dots() -> None:
    """Blank entries and stray dots collapse to nothing."""
    assert parse_extensions("") == set()
    assert parse_extensions(", , .") == set()
    assert parse_extensions("cr3,.jpg, , png") == {".cr3", ".jpg", ".png"}


def test_resolve_image_batch_exits_on_unknown_extension() -> None:
    """Empty / dotted-only extension strings exit before touching the filesystem."""
    with pytest.raises(SystemExit):
        resolve_image_batch([], "", recursive=False)


def test_resolve_image_batch_exits_when_inputs_missing() -> None:
    """No --input paths is a hard error so we never silently skip the run."""
    with pytest.raises(SystemExit):
        resolve_image_batch(None, "cr3", recursive=False)


def test_resolve_image_batch_exits_when_directory_yields_nothing(tmp_path: Path) -> None:
    """A non-empty directory with no matching extensions also exits."""
    (tmp_path / "note.txt").write_text("hello")
    with pytest.raises(SystemExit):
        resolve_image_batch([tmp_path], "cr3", recursive=False)


def test_resolve_image_batch_returns_files(tmp_path: Path) -> None:
    """Happy path returns the resolved file list, deduplicated."""
    a = tmp_path / "a.cr3"
    a.write_text("x")
    b = tmp_path / "b.jpg"
    b.write_text("x")

    result = resolve_image_batch([tmp_path, a], "cr3,jpg", recursive=False)
    assert {p.resolve() for p in result} == {a.resolve(), b.resolve()}


def test_load_skip_list_strips_blank_and_comment_lines(tmp_path: Path) -> None:
    """Blank lines and lines starting with '#' are dropped from the skip list."""
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text("\n# comment\nkeep.cr3\n  also.jpg\n")
    assert load_skip_list(skip_file) == {"keep.cr3", "also.jpg"}


def test_load_skip_list_warns_when_empty(tmp_path: Path) -> None:
    """An empty skip file returns an empty set instead of crashing."""
    skip_file = tmp_path / "empty.txt"
    skip_file.write_text("# only comments\n\n")
    assert load_skip_list(skip_file) == set()


def test_load_skip_list_exits_on_io_error(tmp_path: Path) -> None:
    """A missing skip file is a hard error (we do not silently skip filtering)."""
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(SystemExit):
        load_skip_list(missing)


def test_filter_skipped_files_supports_name_and_path(tmp_path: Path) -> None:
    """Both bare filenames and full paths in the skip list match."""
    a = tmp_path / "a.cr3"
    b = tmp_path / "b.cr3"
    c = tmp_path / "c.cr3"
    for path in (a, b, c):
        path.write_text("x")

    expected_skipped = 2
    kept, skipped = filter_skipped_files([a, b, c], {"a.cr3", str(b)})
    assert kept == [c]
    assert skipped == expected_skipped


def test_filter_skipped_files_passthrough_when_empty(tmp_path: Path) -> None:
    """An empty skip set returns the input list unchanged."""
    a = tmp_path / "a.cr3"
    a.write_text("x")
    kept, skipped = filter_skipped_files([a], set())
    assert kept == [a]
    assert skipped == 0


def test_apply_skip_file_returns_input_when_no_skip_file(tmp_path: Path) -> None:
    """A None skip-file means no filtering happens at all."""
    a = tmp_path / "a.cr3"
    a.write_text("x")
    assert apply_skip_file([a], None) == [a]


def test_apply_skip_file_filters(tmp_path: Path) -> None:
    """Files listed in the skip file are removed from the batch."""
    a = tmp_path / "a.cr3"
    b = tmp_path / "b.cr3"
    a.write_text("x")
    b.write_text("x")
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text("a.cr3\n")
    assert apply_skip_file([a, b], skip_file) == [b]


def test_apply_skip_tagged_disabled_returns_input(tmp_path: Path) -> None:
    """skip_tagged=False is a no-op even when find_tagged_images would match."""
    a = tmp_path / "a.cr3"
    a.write_text("x")
    # Patch as a sanity check: nothing should call find_tagged_images when disabled.
    with patch("photo_tagger.discovery.find_tagged_images") as finder:
        result = apply_skip_tagged([a], skip_tagged=False)
    assert result == [a]
    finder.assert_not_called()


def test_apply_skip_tagged_empty_input_skips_metadata_call(tmp_path: Path) -> None:
    """An empty batch returns immediately without invoking exiftool."""
    with patch("photo_tagger.discovery.find_tagged_images") as finder:
        result = apply_skip_tagged([], skip_tagged=True)
    assert result == []
    finder.assert_not_called()


def test_apply_skip_tagged_drops_tagged_paths(tmp_path: Path) -> None:
    """Paths returned by find_tagged_images are filtered out of the batch."""
    a = tmp_path / "a.cr3"
    b = tmp_path / "b.cr3"
    c = tmp_path / "c.cr3"
    for path in (a, b, c):
        path.write_text("x")
    with patch("photo_tagger.discovery.find_tagged_images", return_value={a, c}):
        kept = apply_skip_tagged([a, b, c], skip_tagged=True)
    assert kept == [b]


def test_apply_skip_tagged_returns_all_when_none_tagged(tmp_path: Path) -> None:
    """If no images are detected as tagged, the original batch passes through."""
    a = tmp_path / "a.cr3"
    a.write_text("x")
    with patch("photo_tagger.discovery.find_tagged_images", return_value=set()):
        kept = apply_skip_tagged([a], skip_tagged=True)
    assert kept == [a]


def test_make_skip_list_appender_returns_none_for_no_path() -> None:
    """No path means no callback (callers treat None as 'do nothing')."""
    assert make_skip_list_appender(None) is None


def test_make_skip_list_appender_creates_and_appends(tmp_path: Path) -> None:
    """A non-existent skip file is created and gets one line per success."""
    skip_file = tmp_path / "processed.txt"
    appender = make_skip_list_appender(skip_file)
    assert appender is not None
    appender(tmp_path / "IMG_0001.CR3")
    appender(tmp_path / "IMG_0002.CR3")
    assert skip_file.read_text(encoding="utf-8").splitlines() == [
        "IMG_0001.CR3",
        "IMG_0002.CR3",
    ]


def test_make_skip_list_appender_skips_existing_entries(tmp_path: Path) -> None:
    """Filenames already in the file (from earlier runs) are not appended a second time."""
    skip_file = tmp_path / "processed.txt"
    skip_file.write_text("IMG_0001.CR3\n# user note\n\n")
    appender = make_skip_list_appender(skip_file)
    assert appender is not None
    appender(tmp_path / "IMG_0001.CR3")  # already there, should not duplicate
    appender(tmp_path / "IMG_0002.CR3")
    lines = skip_file.read_text(encoding="utf-8").splitlines()
    assert lines.count("IMG_0001.CR3") == 1
    assert "IMG_0002.CR3" in lines


def test_make_skip_list_appender_does_not_repeat_within_a_run(tmp_path: Path) -> None:
    """Repeated calls with the same name in one run only append it once."""
    skip_file = tmp_path / "processed.txt"
    appender = make_skip_list_appender(skip_file)
    assert appender is not None
    target = tmp_path / "IMG_0001.CR3"
    appender(target)
    appender(target)
    assert skip_file.read_text(encoding="utf-8").splitlines() == ["IMG_0001.CR3"]
