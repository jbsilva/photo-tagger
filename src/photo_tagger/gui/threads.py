# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""Start and stop the QThreads that the window's background jobs run on."""

from loguru import logger
from PySide6.QtCore import QObject, QThread


def start_in_thread(parent: QObject, worker: QObject, *, quit_when_done: bool = False) -> QThread:
    """
    Move *worker* onto a new thread that *parent* owns, start it, and return the thread.

    Connect the worker's own signals before calling this: its run() starts as soon as the thread
    does, and a signal emitted before its connection is lost. *quit_when_done* ends the thread's
    event loop when the worker finishes, for a job that nobody tears down by hand.
    """
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if quit_when_done:
        worker.finished.connect(thread.quit)
    thread.start()
    return thread


def join_thread(thread: QThread) -> None:
    """
    End *thread*'s event loop, wait for it to stop, and free it.

    For a job that has to be over before the caller goes on: a generation or a save the window is
    tearing down, whose worker stops at the next photo once asked to.
    """
    thread.quit()
    thread.wait()
    thread.deleteLater()


def release_thread(
    thread: QThread,
    *,
    timeout_ms: int,
    detached_event: str,
    level: str = "WARNING",
) -> None:
    """
    End *thread*'s event loop and free it: now if it stops within *timeout_ms*, else once it does.

    A worker inside an uninterruptible call (a batched ExifTool read, a model request) cannot be
    told to stop, and waiting it out would freeze whoever asked: a click, or the window closing. So
    a thread that is still busy is detached, which logs *detached_event* at *level*, and is freed
    once it has stopped (see _Reaper).
    """
    thread.quit()
    if thread.wait(timeout_ms):
        thread.deleteLater()
        return
    # Logged as the caller, so the record names the code that let the thread go, not this helper.
    logger.opt(depth=1).log(level, detached_event)
    _Reaper(thread)


class _Reaper(QObject):
    """
    Free a detached thread once it has stopped, never while it is still winding down.

    A thread emits finished before it destroys the worker that deleteLater left on it, and that
    teardown calls back into the thread's Python wrapper. With deleteLater connected straight to
    finished, the main thread could free that wrapper at the same moment, and the process crashed
    with a segmentation fault. The reaper lives on the main thread, so its slot runs there and can
    wait for the thread to exit before it frees it.
    """

    def __init__(self, thread: QThread) -> None:
        super().__init__(thread)  # a child of the thread, so it goes when the thread does
        thread.finished.connect(self._reap)

    def _reap(self) -> None:
        join_thread(self.parent())
