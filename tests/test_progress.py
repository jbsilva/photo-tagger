"""Tests for the progress bar module."""

from pathlib import Path
from unittest.mock import patch

from photo_tagger.progress import batch_progress


def test_batch_progress_yields_none_when_disabled() -> None:
    """``enabled=False`` produces a ``None`` callback so the pipeline skips progress tracking."""
    with batch_progress(10, enabled=False) as cb:
        assert cb is None


def test_batch_progress_yields_none_when_total_is_zero() -> None:
    """An empty batch disables the bar even if enabled=True and stderr is a tty."""
    with patch("sys.stderr") as mock_stderr:
        mock_stderr.isatty.return_value = True
        with batch_progress(0, enabled=True) as cb:
            assert cb is None


def test_batch_progress_yields_none_when_not_tty() -> None:
    """Non-tty stderr (pipes, CI) disables the progress bar."""
    with patch("sys.stderr") as mock_stderr:
        mock_stderr.isatty.return_value = False
        with batch_progress(5, enabled=True) as cb:
            assert cb is None


def test_batch_progress_yields_callable_on_tty() -> None:
    """An interactive tty stderr with a positive total yields a working callback."""
    import io  # noqa: PLC0415

    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    with patch("photo_tagger.progress.sys") as mock_sys:
        mock_sys.stderr = _FakeTTY()
        with batch_progress(3, enabled=True) as cb:
            assert callable(cb)
            # Calling the callback must not raise.
            cb(Path("img.cr3"), True)  # noqa: FBT003 - matches pipeline contract.
            cb(Path("img2.cr3"), False)  # noqa: FBT003
