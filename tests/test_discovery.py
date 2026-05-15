"""Tests for input discovery and skip-list handling."""

from typing import TYPE_CHECKING

import pytest

from photo_tagger.discovery import (
    apply_skip_file,
    filter_skipped_files,
    load_skip_list,
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
