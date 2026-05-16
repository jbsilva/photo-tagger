"""Tests for setup_logging."""

from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.logging_setup import setup_logging


if TYPE_CHECKING:
    from pathlib import Path


def test_setup_logging_creates_log_folder(tmp_path: Path) -> None:
    """File logging creates the log folder and writes at least one file."""
    folder = tmp_path / "logs"
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=folder)
    logger.info("hello")
    logger.complete()
    assert folder.exists()
    files = list(folder.glob("*-photo_tagger.log"))
    assert files, "expected a log file to be created"


def test_setup_logging_off_disables_handlers(tmp_path: Path) -> None:
    """OFF on both sinks leaves loguru with no handlers attached."""
    folder = tmp_path / "logs"
    setup_logging(file_log_level="OFF", console_log_level="OFF", log_folder=folder)
    # No file handler => the folder is never created.
    assert not folder.exists()
