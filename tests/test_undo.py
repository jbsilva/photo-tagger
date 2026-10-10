"""Tests for the undo journal and the restore logic behind ``photo-tagger undo``."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from photo_tagger.undo import (
    BACKUP_SUFFIX,
    CHANGED,
    DELETED,
    FAILED,
    MISSING,
    NO_BACKUP,
    RESTORED,
    UndoError,
    UndoJournal,
    UndoResult,
    WriteRecord,
    latest_journal,
    list_journals,
    open_journal,
    prune_journals,
    read_journal,
    runs_dir,
    undo_run,
)


_STARTED_AT = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _record_for(target: Path, *, created: bool, backup: Path | None = None) -> WriteRecord:
    """Build a record describing *target* exactly as it is on disk right now."""
    stat = target.stat()
    return WriteRecord(
        image=str(target.with_suffix(".cr3")),
        target=str(target),
        created=created,
        backup=str(backup) if backup is not None else None,
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


def test_runs_dir_follows_the_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Journals live under the same state directory as the rest of the app's state."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert runs_dir() == tmp_path / "photo-tagger" / "runs"


def test_journal_is_created_on_the_first_record(tmp_path: Path) -> None:
    """A run that writes nothing leaves no journal behind."""
    journal = UndoJournal(tmp_path / "runs" / "run.jsonl")
    assert not journal.path.exists()
    assert journal.entries == 0

    target = tmp_path / "a.xmp"
    target.write_text("xmp")
    journal.record(tmp_path / "a.cr3", target, created=True)

    assert journal.entries == 1
    payload = json.loads(journal.path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["target"] == str(target)
    assert payload["created"] is True
    assert payload["backup"] is None
    assert payload["size"] == len("xmp")


def test_journal_records_an_exiftool_backup_when_one_exists(tmp_path: Path) -> None:
    """An overwritten file is recorded with the backup that can restore it."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")

    journal = UndoJournal(tmp_path / "run.jsonl")
    journal.record(tmp_path / "a.cr3", target, created=False)

    assert json.loads(journal.path.read_text(encoding="utf-8"))["backup"] == str(backup)


def test_journal_ignores_a_stale_backup_when_the_write_made_none(tmp_path: Path) -> None:
    """
    A write with backups off must never adopt an ``*_original`` it did not create.

    ExifTool gets ``-overwrite_original`` and writes no backup, so a file under that name is left
    over from an earlier run or from the user's own ExifTool use. Recording it would point undo at
    content this run never wrote, and restoring it would destroy every edit made since.
    """
    target = tmp_path / "a.xmp"
    target.write_text("written now")
    stale = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    stale.write_text("from some run months ago")

    journal = UndoJournal(tmp_path / "run.jsonl")
    journal.record(tmp_path / "a.cr3", target, created=False, backed_up=False)

    assert json.loads(journal.path.read_text(encoding="utf-8"))["backup"] is None


def test_journal_skips_a_target_it_cannot_stat(tmp_path: Path) -> None:
    """A vanished target records nothing rather than a half-truth."""
    journal = UndoJournal(tmp_path / "run.jsonl")
    journal.record(tmp_path / "a.cr3", tmp_path / "gone.xmp", created=True)
    assert journal.entries == 0
    assert not journal.path.exists()


def test_journal_stops_after_a_write_failure(tmp_path: Path) -> None:
    """An unwritable journal is disabled instead of logging once per photo."""
    target = tmp_path / "a.xmp"
    target.write_text("x")
    journal = UndoJournal(tmp_path / "run.jsonl")

    with patch("pathlib.Path.open", side_effect=OSError("read-only")):
        journal.record(tmp_path / "a.cr3", target, created=True)
    assert journal.entries == 0

    # No further attempt is made, so a now-writable path still records nothing.
    journal.record(tmp_path / "a.cr3", target, created=True)
    assert journal.entries == 0


def test_open_journal_is_off_when_disabled() -> None:
    """--no-undo-log means no journal object at all."""
    assert open_journal(_STARTED_AT, enabled=False) is None


def test_open_journal_names_the_file_after_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name starts with the UTC start time so journals sort chronologically by name."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    journal = open_journal(_STARTED_AT)
    assert journal is not None
    # Microseconds included: one process can open two journals inside a second (the GUI opens one
    # per save), and a shared name would merge two runs into one undo.
    assert journal.path.name.startswith("20260501090000000000-")
    assert journal.path.suffix == ".jsonl"


def test_two_journals_opened_in_the_same_second_do_not_collide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undoing one batch must not put back the batch saved a moment before it."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    first = open_journal(_STARTED_AT)
    second = open_journal(_STARTED_AT.replace(microsecond=500))
    assert first is not None
    assert second is not None
    assert first.path != second.path


def test_list_and_latest_journal_are_newest_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listing sorts by name, which is the start timestamp."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    folder = runs_dir()
    folder.mkdir(parents=True)
    older = folder / "20260101090000-1.jsonl"
    newer = folder / "20260501090000-2.jsonl"
    for path in (older, newer):
        path.write_text("", encoding="utf-8")

    assert list_journals() == [newer, older]
    assert latest_journal() == newer


def test_list_journals_is_empty_without_a_runs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine that never tagged anything has nothing to list and does not fail."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert list_journals() == []
    assert latest_journal() is None


def test_prune_journals_keeps_the_newest_and_drops_the_ancient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both retention limits apply: a count of recent runs, and an age cutoff."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr("photo_tagger.undo._KEEP_RUNS", 2)
    folder = runs_dir()
    folder.mkdir(parents=True)
    recent = [folder / f"2026050{i}090000-1.jsonl" for i in (1, 2, 3)]
    for path in recent:
        path.write_text("", encoding="utf-8")
    ancient = folder / "20200101090000-1.jsonl"
    ancient.write_text("", encoding="utf-8")
    old_stamp = (datetime.now(tz=UTC) - timedelta(days=365)).timestamp()
    import os  # noqa: PLC0415 - only this test needs to backdate a file

    os.utime(ancient, (old_stamp, old_stamp))

    # The third-newest (past --keep) plus the ancient one (past the age cutoff).
    expected_removed = 2
    assert prune_journals() == expected_removed
    assert sorted(p.name for p in list_journals()) == [
        "20260502090000-1.jsonl",
        "20260503090000-1.jsonl",
    ]


def test_read_journal_skips_unusable_lines(tmp_path: Path) -> None:
    """A truncated final line costs one entry, not the whole undo."""
    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"image": "a.cr3", "target": "a.xmp", "created": true}\n'
        "\n"
        "not json at all\n"
        '{"no": "target"}\n'
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    records = read_journal(path)

    assert [record.target for record in records] == ["a.xmp"]
    assert records[0].created is True


def test_read_journal_reports_an_unreadable_file(tmp_path: Path) -> None:
    """A missing journal is a clean domain error."""
    with pytest.raises(UndoError, match="Could not read"):
        read_journal(tmp_path / "nope.jsonl")


def test_undo_restores_an_overwritten_file(tmp_path: Path) -> None:
    """The ExifTool backup goes back over the file the run wrote."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")
    record = _record_for(target, created=False, backup=backup)

    results = undo_run([record])

    assert [result.action for result in results] == [RESTORED]
    assert target.read_text(encoding="utf-8") == "old"
    assert not backup.exists()


def test_undo_deletes_a_sidecar_the_run_created(tmp_path: Path) -> None:
    """A file that did not exist before the run is removed, not restored."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    record = _record_for(target, created=True)

    results = undo_run([record])

    assert [result.action for result in results] == [DELETED]
    assert not target.exists()


def test_undo_ignores_a_stale_backup_next_to_a_created_file(tmp_path: Path) -> None:
    """A leftover *_original from an older run must not resurrect an older sidecar."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    stale = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    stale.write_text("from a previous run")
    record = _record_for(target, created=True, backup=stale)

    assert [result.action for result in undo_run([record])] == [DELETED]
    assert not target.exists()
    assert stale.exists()


def test_undo_leaves_a_changed_file_alone(tmp_path: Path) -> None:
    """An edit after the run means the file is no longer ours to revert."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")
    record = _record_for(target, created=False, backup=backup)
    target.write_text("edited in Lightroom since")

    results = undo_run([record])

    assert [result.action for result in results] == [CHANGED]
    assert target.read_text(encoding="utf-8") == "edited in Lightroom since"


def test_undo_force_reverts_a_changed_file(tmp_path: Path) -> None:
    """--force is the escape hatch for a file that changed after the run."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")
    record = _record_for(target, created=False, backup=backup)
    target.write_text("edited since")

    assert [result.action for result in undo_run([record], force=True)] == [RESTORED]
    assert target.read_text(encoding="utf-8") == "old"


def test_undo_reports_a_target_that_is_gone(tmp_path: Path) -> None:
    """Nothing to put back is not an error, but it is not a success either."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    record = _record_for(target, created=True)
    target.unlink()

    assert [result.action for result in undo_run([record])] == [MISSING]
    assert [result.action for result in undo_run([record], force=True)] == [MISSING]


def test_undo_reports_a_write_made_without_a_backup(tmp_path: Path) -> None:
    """--no-backup-xmp leaves nothing to restore, and undo says so instead of guessing."""
    target = tmp_path / "photo.jpg"
    target.write_text("tagged")
    record = _record_for(target, created=False)

    results = undo_run([record])

    assert [result.action for result in results] == [NO_BACKUP]
    assert target.read_text(encoding="utf-8") == "tagged"


def test_undo_reports_a_vanished_backup(tmp_path: Path) -> None:
    """A backup deleted since the run cannot restore anything."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")
    record = _record_for(target, created=False, backup=backup)
    backup.unlink()

    assert [result.action for result in undo_run([record])] == [NO_BACKUP]


def test_undo_reports_a_failed_restore(tmp_path: Path) -> None:
    """An OS error while reverting is reported per file, not raised."""
    target = tmp_path / "a.xmp"
    target.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")
    record = _record_for(target, created=False, backup=backup)

    with patch("pathlib.Path.replace", side_effect=OSError("permission denied")):
        results = undo_run([record])

    assert results[0].action == FAILED
    assert "permission denied" in results[0].detail


def test_undo_dry_run_changes_nothing(tmp_path: Path) -> None:
    """A dry run reports the same verdicts without touching the files."""
    restored = tmp_path / "a.xmp"
    restored.write_text("new")
    backup = tmp_path / ("a.xmp" + BACKUP_SUFFIX)
    backup.write_text("old")
    created = tmp_path / "b.xmp"
    created.write_text("new")
    records = [
        _record_for(restored, created=False, backup=backup),
        _record_for(created, created=True),
    ]

    results = undo_run(records, dry_run=True)

    assert sorted(result.action for result in results) == [DELETED, RESTORED]
    assert restored.read_text(encoding="utf-8") == "new"
    assert created.exists()


def test_undo_walks_the_journal_backwards(tmp_path: Path) -> None:
    """The last write is undone first, so a file written twice ends up at its oldest state."""
    target = tmp_path / "a.xmp"
    target.write_text("second")
    order: list[str] = []
    records = [
        _record_for(target, created=True),
        WriteRecord(image="b.cr3", target=str(tmp_path / "b.xmp"), created=True),
    ]

    def spy(record: WriteRecord, **_kwargs: object) -> UndoResult:
        order.append(record.target)
        return UndoResult(Path(record.target), DELETED)

    with patch("photo_tagger.undo._undo_one", side_effect=spy):
        undo_run(records)

    assert order == [str(tmp_path / "b.xmp"), str(target)]


def test_list_journals_survives_an_unreadable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state directory we cannot read means "no journals", not a crash."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with patch("pathlib.Path.glob", side_effect=OSError("permission denied")):
        assert list_journals() == []


def test_prune_journals_skips_an_entry_it_cannot_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A journal that vanishes mid-prune is left to whoever removed it."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    folder = runs_dir()
    folder.mkdir(parents=True)
    (folder / "20260501090000-1.jsonl").write_text("", encoding="utf-8")

    with patch("pathlib.Path.stat", side_effect=OSError("gone")):
        assert prune_journals() == 0


def test_journal_keeps_its_backup_for_a_second_write_of_the_same_file(tmp_path: Path) -> None:
    """
    A file saved twice in one run keeps the backup of its first write.

    ExifTool keeps that backup and writes the file anyway, so the second write finds it there. It is
    still this run's backup, and dropping it would leave undo nothing to restore.
    """
    target = tmp_path / "a.jpg"
    target.write_text("first save")
    (tmp_path / ("a.jpg" + BACKUP_SUFFIX)).write_text("before the run")
    journal = UndoJournal(tmp_path / "run.jsonl")
    journal.record(target, target, created=False, backed_up=True)
    target.write_text("second save")

    journal.record(target, target, created=False, backed_up=False)

    records = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["backup"] is not None
