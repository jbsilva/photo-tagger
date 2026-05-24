"""
Cross-process file lock used to prevent concurrent runs from racing on the same outputs.

POSIX-only (fcntl). The lock is advisory: it only stops other photo-tagger runs that opt
into the same lock file. The OS releases the lock automatically when the process exits
or the file descriptor is closed, so a crashed run does not leave a stale lock behind.
"""

import fcntl
import os
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Self

from loguru import logger


if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class LockHeldError(RuntimeError):
    """Raised when another process already holds the lock."""


class FileLock(AbstractContextManager["FileLock"]):
    """
    Non-blocking exclusive flock on a sentinel file.

    Use as a context manager. ``__enter__`` raises :class:`LockHeldError` immediately if
    the lock is already held by another process; the caller is expected to translate
    that into a CLI-friendly error.
    """

    __slots__ = ("_fd", "_path")

    _path: Path
    _fd: int | None

    def __init__(self, path: Path) -> None:
        """Track the lock file path; do not open it yet."""
        self._path = path
        self._fd = None

    def __enter__(self) -> Self:
        """Acquire the lock or raise LockHeldError if another process holds it."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            msg = f"lock file {self._path} is held by another process"
            raise LockHeldError(msg) from exc
        # Truncate + write our PID so a human inspecting the lock file knows who owns it.
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        logger.debug("lock_acquired", file=str(self._path), pid=os.getpid())
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Release the lock and close the fd. Safe to call multiple times."""
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            logger.debug("lock_released", file=str(self._path))
