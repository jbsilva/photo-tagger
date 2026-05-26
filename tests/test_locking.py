"""Tests for the cross-process FileLock used to gate concurrent runs."""

import os
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING

import pytest

from photo_tagger.locking import FileLock, LockHeldError


if TYPE_CHECKING:
    from pathlib import Path


def test_file_lock_acquires_and_releases(tmp_path: Path) -> None:
    """A FileLock used as a context manager creates the file and clears the lock on exit."""
    lock_path = tmp_path / "photo-tagger.lock"
    with FileLock(lock_path):
        assert lock_path.exists()
    # After release, a fresh acquisition must succeed immediately and re-create the file.
    with FileLock(lock_path) as held:
        assert held is not None


def test_file_lock_writes_owning_pid(tmp_path: Path) -> None:
    """The lock file content is the holder's PID so humans can see who owns it."""
    lock_path = tmp_path / "photo-tagger.lock"
    with FileLock(lock_path):
        contents = lock_path.read_text(encoding="utf-8").strip()
    assert contents == str(os.getpid())


def test_file_lock_creates_file_owner_only(tmp_path: Path) -> None:
    """The lock file is created with 0o600 so only the owner can read/write it."""
    lock_path = tmp_path / "photo-tagger.lock"
    expected_mode = 0o600
    with FileLock(lock_path):
        mode = lock_path.stat().st_mode & 0o777
    assert mode == expected_mode


_HOLDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {src_root!r})
    from photo_tagger.locking import FileLock

    with FileLock(Path(sys.argv[1])):
        # Print a sentinel to stdout so the parent knows the lock is held, then
        # wait for the parent to either send SIGTERM or for the timeout to elapse.
        print("ACQUIRED", flush=True)
        time.sleep(float(sys.argv[2]))
    """,
)


def test_file_lock_blocks_second_acquirer(tmp_path: Path) -> None:
    """A second process trying to acquire the same lock raises LockHeldError."""
    lock_path = tmp_path / "photo-tagger.lock"
    src_root = str((__import__("photo_tagger").__file__ or "").rsplit("/", 2)[0])
    script = _HOLDER_SCRIPT.format(src_root=src_root)
    holder = subprocess.Popen(  # noqa: S603 - inputs are test-controlled.
        [sys.executable, "-c", script, str(lock_path), "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for the child to print ACQUIRED so we know it owns the lock.
        assert holder.stdout is not None
        first = holder.stdout.readline()
        assert first.strip() == b"ACQUIRED", f"child failed to acquire: stdout={first!r}"

        with pytest.raises(LockHeldError), FileLock(lock_path) as held:
            # Acquisition must raise LockHeldError; reaching this line is the test failure.
            pytest.fail(f"expected LockHeldError, got lock {held!r}")
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_file_lock_is_reentrant_after_release(tmp_path: Path) -> None:
    """Re-acquiring after a clean release works without delay."""
    lock_path = tmp_path / "photo-tagger.lock"
    for _ in range(3):
        with FileLock(lock_path):
            time.sleep(0.001)
