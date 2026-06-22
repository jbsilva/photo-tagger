"""
Cross-process file lock used to prevent concurrent runs from racing on the same outputs.

Uses the ``filelock`` library for cross-platform support (POSIX + Windows). The lock is advisory: it
only stops other photo-tagger runs that opt into the same lock file. The OS releases the lock
automatically when the process exits or the file descriptor is closed, so a crashed run does not
leave a stale lock behind.
"""

import os
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Self

from filelock import (
    FileLock as _FileLock,
    Timeout,
)
from loguru import logger


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class LockHeldError(RuntimeError):
    """Raised when another process already holds the lock."""


class FileLock(AbstractContextManager["FileLock"]):
    """
    Non-blocking exclusive lock on a sentinel file.

    Use as a context manager. ``__enter__`` raises :class:`LockHeldError` immediately if the lock is
    already held by another process; the caller is expected to translate that into a CLI-friendly
    error.

    Delegates to ``filelock.FileLock`` which works on Linux, macOS, and Windows.
    """

    __slots__ = ("_lock", "_path")

    _path: Path
    _lock: _FileLock

    def __init__(self, path: Path) -> None:
        """Track the lock file path; do not open it yet."""
        self._path = path
        self._lock = _FileLock(path, mode=0o600)

    def __enter__(self) -> Self:
        """Acquire the lock or raise LockHeldError if another process holds it."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except Timeout as exc:
            msg = f"lock file {self._path} is held by another process"
            raise LockHeldError(msg) from exc
        # Write our PID so a human inspecting the lock file knows who owns it.
        try:
            self._path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        except OSError:
            self._lock.release()
            raise
        logger.debug("lock_acquired", file=str(self._path), pid=os.getpid())
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """
        Release the lock.

        Safe to call multiple times.
        """
        self._lock.release()
        logger.debug("lock_released", file=str(self._path))
