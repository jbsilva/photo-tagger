"""Tests for input discovery and skip-list handling."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from photo_tagger.discovery import (
    apply_date_filter,
    apply_skip_file,
    apply_skip_tagged,
    filter_skipped_files,
    load_skip_list,
    make_skip_list_appender,
    parse_extensions,
    resolve_image_batch,
    resolve_image_files,
    skip_list_matches,
)
from photo_tagger.errors import DiscoveryError


if TYPE_CHECKING:
    from pathlib import Path


class _ResolveRaisingPath:
    """Path-like whose .resolve() raises OSError, to exercise the resolve guard."""

    def __init__(self, name: str) -> None:
        self._name = name

    def resolve(self) -> Path:
        msg = "cannot resolve"
        raise OSError(msg)

    def is_dir(self) -> bool:
        return False

    def is_file(self) -> bool:
        return True

    def exists(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name


def test_parse_extensions_drops_blanks_and_dots() -> None:
    """Blank entries and stray dots collapse to nothing."""
    assert parse_extensions("") == set()
    assert parse_extensions(", , .") == set()
    assert parse_extensions("cr3,.jpg, , png") == {".cr3", ".jpg", ".png"}


def test_resolve_image_batch_exits_on_unknown_extension() -> None:
    """Empty / dotted-only extension strings exit before touching the filesystem."""
    with pytest.raises(DiscoveryError):
        resolve_image_batch([], "", recursive=False)


def test_resolve_image_batch_exits_when_inputs_missing() -> None:
    """No --input paths is a hard error so we never silently skip the run."""
    with pytest.raises(DiscoveryError):
        resolve_image_batch(None, "cr3", recursive=False)


def test_resolve_image_batch_exits_when_directory_yields_nothing(tmp_path: Path) -> None:
    """A non-empty directory with no matching extensions also exits."""
    (tmp_path / "note.txt").write_text("hello")
    with pytest.raises(DiscoveryError):
        resolve_image_batch([tmp_path], "cr3", recursive=False)


def test_resolve_image_batch_matches_extensions_case_insensitively(tmp_path: Path) -> None:
    """Canon-style .CR3 files are accepted with --ext cr3 (the matching is case-insensitive)."""
    canon_upper = tmp_path / "IMG_0001.CR3"
    canon_upper.write_text("x")
    other = tmp_path / "note.txt"
    other.write_text("x")

    result = resolve_image_batch([tmp_path], "cr3", recursive=False)
    assert {p.resolve() for p in result} == {canon_upper.resolve()}


def test_resolve_image_batch_mixed_case_extensions_and_files(tmp_path: Path) -> None:
    """Mixed-case directory entries all match a single lowercase extension spec."""
    lower = tmp_path / "a.cr3"
    upper = tmp_path / "b.CR3"
    titlecase = tmp_path / "c.Cr3"
    for path in (lower, upper, titlecase):
        path.write_text("x")

    result = resolve_image_batch([tmp_path], "CR3", recursive=False)
    assert {p.resolve() for p in result} == {
        lower.resolve(),
        upper.resolve(),
        titlecase.resolve(),
    }


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
    with pytest.raises(DiscoveryError):
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


def test_skip_list_matches_returns_matched_paths(tmp_path: Path) -> None:
    """The matched set covers both bare-filename and full-path entries, ignoring case."""
    a = tmp_path / "a.cr3"
    b = tmp_path / "b.cr3"
    c = tmp_path / "c.cr3"
    matched = skip_list_matches([a, b, c], {"A.CR3", str(b)})
    assert matched == {a, b}


def test_skip_list_matches_empty_set_matches_nothing(tmp_path: Path) -> None:
    """An empty skip set yields an empty match set rather than every file."""
    assert skip_list_matches([tmp_path / "a.cr3"], set()) == set()


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


def _set_mtime(path: Path, when: datetime) -> None:
    """Stamp *path*'s mtime to *when* (UTC) so date-filter tests are deterministic."""
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def test_apply_date_filter_returns_input_when_no_bounds(tmp_path: Path) -> None:
    """Neither bound set means no filtering happens at all."""
    a = tmp_path / "a.cr3"
    a.write_text("x")
    assert apply_date_filter([a]) == [a]


def test_apply_date_filter_drops_files_older_than_newer_than_bound(tmp_path: Path) -> None:
    """--newer-than rejects files whose mtime is on or before the bound."""
    old = tmp_path / "old.cr3"
    new = tmp_path / "new.cr3"
    old.write_text("x")
    new.write_text("x")
    boundary = datetime(2024, 1, 1, tzinfo=UTC)
    _set_mtime(old, boundary - timedelta(days=1))
    _set_mtime(new, boundary + timedelta(days=1))

    kept = apply_date_filter([old, new], newer_than=boundary)
    assert kept == [new]


def test_apply_date_filter_drops_files_newer_than_older_than_bound(tmp_path: Path) -> None:
    """--older-than rejects files whose mtime is on or after the bound."""
    old = tmp_path / "old.cr3"
    new = tmp_path / "new.cr3"
    old.write_text("x")
    new.write_text("x")
    boundary = datetime(2024, 6, 1, tzinfo=UTC)
    _set_mtime(old, boundary - timedelta(days=30))
    _set_mtime(new, boundary + timedelta(days=30))

    kept = apply_date_filter([old, new], older_than=boundary)
    assert kept == [old]


def test_apply_date_filter_supports_window(tmp_path: Path) -> None:
    """Combining --newer-than and --older-than selects an open interval."""
    before = tmp_path / "before.cr3"
    inside = tmp_path / "inside.cr3"
    after = tmp_path / "after.cr3"
    for path in (before, inside, after):
        path.write_text("x")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)
    _set_mtime(before, start - timedelta(days=1))
    _set_mtime(inside, datetime(2024, 6, 15, tzinfo=UTC))
    _set_mtime(after, end + timedelta(days=1))

    kept = apply_date_filter([before, inside, after], newer_than=start, older_than=end)
    assert kept == [inside]


def test_apply_date_filter_logs_warning_for_unreadable_stat(tmp_path: Path) -> None:
    """A path that fails stat() is logged and skipped, not raised."""
    ghost = tmp_path / "ghost.cr3"  # never created
    kept = apply_date_filter([ghost], newer_than=datetime(2020, 1, 1, tzinfo=UTC))
    assert kept == []


def test_make_skip_list_appender_is_thread_safe(tmp_path: Path) -> None:
    """Under concurrent writes the file contains exactly one entry per unique name."""
    skip_file = tmp_path / "processed.txt"
    appender = make_skip_list_appender(skip_file)
    assert appender is not None

    image_count = 50
    duplicates_per_image = 4
    paths = [tmp_path / f"IMG_{i:04d}.CR3" for i in range(image_count)]
    schedule = paths * duplicates_per_image

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(appender, schedule))

    lines = skip_file.read_text(encoding="utf-8").splitlines()
    # No interleaved/partial lines, no duplicate names, no missing names.
    assert sorted(lines) == sorted(p.name for p in paths)
    assert len(set(lines)) == image_count


def test_resolve_image_files_keeps_path_when_resolve_fails() -> None:
    """If Path.resolve() raises OSError the original path is used instead of crashing."""
    fake = _ResolveRaisingPath("weird.cr3")
    result = resolve_image_files(cast("list[Path]", [fake]), {".cr3"}, recursive=False)
    assert result == cast("list[Path]", [fake])


def test_resolve_image_files_warns_on_non_file_non_dir(tmp_path: Path) -> None:
    """An input that is neither a file nor a directory is logged and dropped."""
    ghost = tmp_path / "missing.cr3"  # never created
    assert resolve_image_files([ghost], {".cr3"}, recursive=False) == []


def test_apply_skip_file_warns_when_no_files_match(tmp_path: Path) -> None:
    """A non-empty skip list that matches nothing logs a warning and keeps every file."""
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text("other.cr3\n", encoding="utf-8")
    image = tmp_path / "kept.cr3"

    assert apply_skip_file([image], skip_file) == [image]


def test_apply_skip_file_keeps_all_when_skip_list_is_empty(tmp_path: Path) -> None:
    """A comments-only skip file yields no entries, so every file passes through silently."""
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text("# just a comment\n", encoding="utf-8")
    image = tmp_path / "kept.cr3"

    assert apply_skip_file([image], skip_file) == [image]


def test_apply_date_filter_keeps_all_when_window_excludes_nothing(tmp_path: Path) -> None:
    """When every file falls inside the window, nothing is skipped."""
    image = tmp_path / "img.cr3"
    image.write_text("x")
    epoch = datetime(2000, 1, 1, tzinfo=UTC)

    assert apply_date_filter([image], newer_than=epoch) == [image]


def test_make_skip_list_appender_survives_unreadable_and_unwritable_target(tmp_path: Path) -> None:
    """A skip path pointing at a directory: the preload and the append both log and recover."""
    skip_dir = tmp_path / "skip_as_dir"
    skip_dir.mkdir()  # read_text() and open("a") both raise OSError on a directory

    appender = make_skip_list_appender(skip_dir)
    assert appender is not None

    # The write failure is swallowed; calling the appender must not raise.
    appender(tmp_path / "IMG_0001.CR3")
