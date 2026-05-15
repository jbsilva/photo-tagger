"""
Progress bar built on rich.

Wraps rich.Progress so the CLI gets one-line live status (e.g. ``43/500 ETA 12m``) on
interactive terminals and a no-op callback on non-tty stdouts (CI, file redirects).
"""

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    ProgressCallback = Callable[[Path, bool], None]


@contextmanager
def batch_progress(total: int, *, enabled: bool = True) -> "Iterator[ProgressCallback | None]":
    """
    Yield a per-image progress callback for :func:`photo_tagger.pipeline.run_batch`.

    Args:
        total: Number of images about to be processed (used to size the bar).
        enabled: If False (or stderr is not a tty), yield ``None`` so the caller can
            short-circuit. CI runs and piped output stay quiet that way.

    """
    if not enabled or not sys.stderr.isatty() or total <= 0:
        yield None
        return

    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    )
    with Progress(*columns, transient=False) as progress:
        task_id = progress.add_task("photos", total=total)

        def advance(path: "Path", ok: bool) -> None:  # noqa: FBT001 - matches pipeline contract.
            # The bar is the dominant feedback while running; keep the per-photo log
            # in the file sink (DEBUG) and just nudge the bar here.
            del path, ok
            progress.advance(task_id)

        yield advance
