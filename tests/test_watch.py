"""Tests for the folder watcher behind ``photo-tagger watch``."""

import os
import time
from pathlib import Path

import pytest

from photo_tagger.watch import _PollState, _settled_files, watch_batches


def _make_photo(path: Path, *, age_seconds: float = 10.0, content: str = "x") -> Path:
    """Create a photo whose mtime is *age_seconds* in the past, so it counts as settled."""
    path.write_text(content)
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_watch_yields_photos_that_were_already_there(tmp_path: Path) -> None:
    """Pointing at a folder that has waited all afternoon tags it, rather than ignoring it."""
    first = _make_photo(tmp_path / "a.jpg")
    second = _make_photo(tmp_path / "b.jpg")

    batches = list(watch_batches([tmp_path], "jpg", interval_seconds=0, max_polls=2))

    assert batches == [[first, second]]


def test_watch_yields_each_photo_once(tmp_path: Path) -> None:
    """A photo already tagged in an earlier batch is not offered again."""
    _make_photo(tmp_path / "a.jpg")
    new_photo: list[Path] = []

    seen: list[list[Path]] = []
    for batch in watch_batches([tmp_path], "jpg", interval_seconds=0, max_polls=4):
        seen.append(batch)
        if not new_photo:
            new_photo.append(_make_photo(tmp_path / "b.jpg"))

    assert seen == [[tmp_path / "a.jpg"], [tmp_path / "b.jpg"]]


def test_watch_waits_for_a_file_to_stop_changing(tmp_path: Path) -> None:
    """A photo still being copied is left alone until its size and mtime hold still."""
    growing = tmp_path / "copying.jpg"
    _make_photo(growing, content="partial")
    state = _PollState()
    now = time.time()

    assert _settled_files([growing], state, settle_seconds=1.0, now=now) == []

    _make_photo(growing, content="partial plus more")
    assert _settled_files([growing], state, settle_seconds=1.0, now=now) == []

    # Unchanged since the previous poll: now it is a finished file.
    assert _settled_files([growing], state, settle_seconds=1.0, now=now) == [growing]


def test_watch_waits_for_a_freshly_written_file_to_age(tmp_path: Path) -> None:
    """Two identical polls are not enough when the file was written a moment ago."""
    fresh = _make_photo(tmp_path / "fresh.jpg", age_seconds=0.0)
    state = _PollState()
    now = time.time()

    assert _settled_files([fresh], state, settle_seconds=30.0, now=now) == []
    assert _settled_files([fresh], state, settle_seconds=30.0, now=now) == []
    # The same file, checked far enough in the future, is settled.
    assert _settled_files([fresh], state, settle_seconds=30.0, now=now + 60) == [fresh]


def test_watch_tags_a_file_stamped_in_the_future(tmp_path: Path) -> None:
    """
    A clock ahead of ours must not park a photo forever.

    Cameras, card readers and NAS boxes stamp files in the future when their clock runs fast, and so
    does any copy that preserves the source mtime. `now - mtime` is then negative and never reaches
    settle_seconds, so the photo was re-polled on every interval and never tagged.
    """
    ahead = _make_photo(tmp_path / "ahead.jpg", age_seconds=-300.0)
    state = _PollState()
    now = time.time()

    assert _settled_files([ahead], state, settle_seconds=30.0, now=now) == []
    # Two identical polls are the real evidence that it stopped changing.
    assert _settled_files([ahead], state, settle_seconds=30.0, now=now) == [ahead]


def test_watch_skips_a_file_that_vanishes_between_listing_and_stat(tmp_path: Path) -> None:
    """A path deleted mid-poll is dropped rather than raising."""
    state = _PollState()
    assert _settled_files([tmp_path / "gone.jpg"], state, settle_seconds=0.0, now=time.time()) == []
    assert state.fingerprints == {}


def test_watch_stops_tracking_a_file_once_it_is_yielded(tmp_path: Path) -> None:
    """The fingerprint map holds only what is still pending, not everything ever seen."""
    photo = _make_photo(tmp_path / "a.jpg")
    state = _PollState()
    now = time.time()

    _settled_files([photo], state, settle_seconds=1.0, now=now)
    assert photo in state.fingerprints

    assert _settled_files([photo], state, settle_seconds=1.0, now=now) == [photo]
    assert state.fingerprints == {}
    assert state.seen == {photo}


def test_watch_honours_the_extension_filter_and_recursion(tmp_path: Path) -> None:
    """Only matching extensions are offered, and subfolders only with recursive=True."""
    _make_photo(tmp_path / "keep.jpg")
    _make_photo(tmp_path / "skip.txt")
    nested = tmp_path / "sub"
    nested.mkdir()
    _make_photo(nested / "deep.jpg")

    flat = list(watch_batches([tmp_path], "jpg", interval_seconds=0, max_polls=2))
    assert flat == [[tmp_path / "keep.jpg"]]

    deep = list(
        watch_batches([tmp_path], "jpg", recursive=True, interval_seconds=0, max_polls=2),
    )
    assert deep == [[tmp_path / "keep.jpg", nested / "deep.jpg"]]


def test_watch_sleeps_between_polls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The interval applies between polls, not before the first one."""
    _make_photo(tmp_path / "a.jpg")
    slept: list[float] = []
    monkeypatch.setattr("photo_tagger.watch.time.sleep", slept.append)

    list(watch_batches([tmp_path], "jpg", interval_seconds=7.5, max_polls=3))

    assert slept == [7.5, 7.5]


def test_watch_stops_when_asked(tmp_path: Path) -> None:
    """A caller with a Stop button ends the watch at the next poll instead of running forever."""
    _make_photo(tmp_path / "a.jpg")
    stopped: list[bool] = []

    seen: list[list[Path]] = []
    # No max_polls: only the predicate ends this loop, which is how the GUI runs it.
    for batch in watch_batches(
        [tmp_path],
        "jpg",
        interval_seconds=0,
        should_stop=lambda: bool(stopped),
    ):
        seen.append(batch)
        stopped.append(True)

    assert seen == [[tmp_path / "a.jpg"]]


def test_watch_cuts_the_wait_short_when_asked_to_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wait is taken in slices, so stopping does not have to sit out a whole interval."""
    _make_photo(tmp_path / "a.jpg")
    slept: list[float] = []
    stop = False

    def fake_sleep(seconds: float) -> None:
        nonlocal stop
        slept.append(seconds)
        stop = True  # asked to stop one slice into the wait

    monkeypatch.setattr("photo_tagger.watch.time.sleep", fake_sleep)

    list(watch_batches([tmp_path], "jpg", interval_seconds=30.0, should_stop=lambda: stop))

    assert slept == [0.2]  # one slice, not the full 30 seconds


def test_watch_waits_the_whole_interval_while_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watcher nobody has stopped still sleeps the full interval, one slice at a time."""
    _make_photo(tmp_path / "a.jpg")
    slept: list[float] = []
    monkeypatch.setattr("photo_tagger.watch.time.sleep", slept.append)

    list(
        watch_batches(
            [tmp_path],
            "jpg",
            interval_seconds=0.5,
            max_polls=2,
            should_stop=lambda: False,
        ),
    )

    assert slept == [0.2, 0.2, pytest.approx(0.1)]
