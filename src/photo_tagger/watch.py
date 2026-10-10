"""
Watch folders for new photos and hand each settled batch to the caller.

Tagging on import is the natural place for this tool in a photographer's workflow: point it at the
folder the card reader (or the tethered capture, or the sync client) fills, and let it work as the
files land. Everything the batch runner needs already exists; what was missing is the loop.

Polling, not filesystem events: a poll is a directory listing every few seconds, it behaves the same
on every platform and over network shares, and it needs no extra dependency. A file must also *hold
still* before it is tagged, since a photo half-copied onto the disk is not a photo yet.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.config import DEFAULT_INTERVAL_SECONDS, DEFAULT_SETTLE_SECONDS
from photo_tagger.discovery import parse_extensions, resolve_image_files


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


# The wait between polls is taken in slices no longer than this, so a caller that asks to stop is
# not held for a whole interval. The CLI stops on Ctrl-C and never passes a predicate; the GUI does,
# because closing a window that takes ten seconds to react reads as a freeze.
_STOP_CHECK_SECONDS = 0.2


@dataclass(slots=True, frozen=True)
class _Fingerprint:
    """The size and mtime of a candidate, as seen at one poll."""

    size: int
    mtime: float


@dataclass(slots=True)
class _PollState:
    """What the watcher remembers between polls."""

    seen: set[Path] = field(default_factory=set)
    fingerprints: dict[Path, _Fingerprint] = field(default_factory=dict)


def _fingerprint(path: Path) -> _Fingerprint | None:
    """Return the current size and mtime of *path*, or None when it cannot be read."""
    try:
        stat = path.stat()
    except OSError:
        # Deleted or still appearing between the listing and the stat; the next poll sorts it out.
        return None
    return _Fingerprint(stat.st_size, stat.st_mtime)


def _has_settled(mtime: float, now: float, settle_seconds: float) -> bool:
    """
    Report whether a file's last change is far enough back to call it finished.

    A camera, card reader or NAS whose clock runs ahead of ours stamps files in the future, and so
    does any copy that preserves the source mtime (``cp -p``, rsync). Plain subtraction then never
    reaches *settle_seconds*, so such a photo was re-polled forever and silently never tagged. The
    caller has already seen two identical polls of it, which is the real evidence that it stopped
    changing, so take that rather than a clock we do not share.
    """
    age = now - mtime
    return age >= settle_seconds or age < 0


def _settled_files(
    candidates: list[Path],
    state: _PollState,
    *,
    settle_seconds: float,
    now: float,
) -> list[Path]:
    """
    Return the candidates that have finished being written, updating *state*.

    A file qualifies when this poll sees exactly what the previous poll saw and its last change is
    at least *settle_seconds* old. Both conditions matter: the comparison catches a slow copy that
    is still growing, and the age catches a fast one that finished between two polls of a folder
    whose files all arrive at once.
    """
    ready: list[Path] = []
    current: dict[Path, _Fingerprint] = {}
    for path in candidates:
        fingerprint = _fingerprint(path)
        if fingerprint is None:
            continue
        current[path] = fingerprint
        previous = state.fingerprints.get(path)
        if previous == fingerprint and _has_settled(fingerprint.mtime, now, settle_seconds):
            ready.append(path)
    # Keep only what is still pending: anything yielded is now the caller's problem, and anything
    # that vanished between polls should not be remembered either.
    yielded = set(ready)
    state.fingerprints = {
        path: fingerprint for path, fingerprint in current.items() if path not in yielded
    }
    state.seen.update(ready)
    return ready


def _sleep_between_polls(seconds: float, should_stop: Callable[[], bool] | None) -> None:
    """Wait *seconds* before the next poll, cutting the wait short once *should_stop* says so."""
    if should_stop is None:
        time.sleep(seconds)
        return
    remaining = seconds
    while remaining > 0 and not should_stop():
        nap = min(_STOP_CHECK_SECONDS, remaining)
        time.sleep(nap)
        remaining -= nap


def watch_batches(  # noqa: PLR0913 - every kwarg is one independent knob.
    inputs: list[Path],
    image_extensions: str,
    *,
    recursive: bool = False,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    max_polls: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[list[Path]]:
    """
    Yield each batch of new, finished photos found under *inputs*, forever.

    Files already present when the watch starts count as new: pointing at a folder that has waited
    all afternoon should tag it, not ignore it. Every file is yielded at most once per watcher, so
    the caller's own skip list and cache are only a backstop.

    Polls with nothing to report yield nothing at all, so the caller's loop body runs only when
    there is work. *max_polls* bounds the loop for tests; production passes None and stops on
    Ctrl-C.

    *should_stop* ends the watch at the next poll boundary, and is checked while waiting too, so a
    caller with a Stop button does not have to wait out an interval. The CLI leaves it None.
    """
    extensions = parse_extensions(image_extensions)
    state = _PollState()
    polls = 0
    while max_polls is None or polls < max_polls:
        if polls:
            _sleep_between_polls(interval_seconds, should_stop)
        if should_stop is not None and should_stop():
            logger.info("watch_stopped", polls=polls, waiting=len(state.fingerprints))
            return
        polls += 1
        candidates = [
            path
            for path in resolve_image_files(inputs, extensions, recursive=recursive)
            if path not in state.seen
        ]
        ready = _settled_files(
            candidates,
            state,
            settle_seconds=settle_seconds,
            now=time.time(),
        )
        if ready:
            logger.info("watch_batch_ready", files=len(ready), waiting=len(state.fingerprints))
            yield sorted(ready)
