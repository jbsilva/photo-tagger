"""Loguru configuration shared by the CLI and any embedding caller."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from photo_tagger.config import LogLevel


_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name:<8}:{function:<25}:{line:>4} | "
    "{message:<40} | "
    "{extra}"
)

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<level>{message:<40.50}</level> | "
    "<yellow>{extra}</yellow>"
)


def setup_logging(
    file_log_level: LogLevel = "DEBUG",
    console_log_level: LogLevel = "INFO",
    log_folder: Path = Path("logs"),
) -> None:
    """
    Configure Loguru for both console and file logging.

    Args:
        file_log_level: Log level for file (use 'OFF' to disable)
        console_log_level: Log level for console (use 'OFF' to disable)
        log_folder: Directory where log files are stored

    """
    logger.remove()
    if file_log_level != "OFF":
        log_folder.mkdir(parents=True, exist_ok=True)
        log_file = log_folder / Path(
            datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S-photo_tagger.log"),
        )
        logger.add(
            log_file,
            level=file_log_level,
            format=_FILE_FORMAT,
            rotation="500 MB",
            retention="10 days",
            compression="zip",
        )
    if console_log_level != "OFF":
        logger.add(
            sys.stderr,
            level=console_log_level,
            colorize=True,
            format=_CONSOLE_FORMAT,
        )
