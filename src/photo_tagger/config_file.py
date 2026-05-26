"""
TOML-based configuration file loader.

Searches for a config file in standard locations and returns a flat dict of overrides that the CLI
module applies to its default dataclass instances before cyclopts parses the command line. CLI flags
always take precedence over the config file because they replace the defaults after they have been
set.

Search order (first match wins):
1. ``$PHOTO_TAGGER_CONFIG`` environment variable (explicit path)
2. ``.photo-tagger.toml`` in the current working directory (project-local)
3. ``~/.config/photo-tagger/config.toml`` (XDG-style user default)
"""

import os
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from loguru import logger


_USER_CONFIG = Path.home() / ".config" / "photo-tagger" / "config.toml"
_LOCAL_CONFIG = Path(".photo-tagger.toml")


def find_config_file() -> Path | None:
    """Return the first existing config path from the search order, or None."""
    if env_path := os.getenv("PHOTO_TAGGER_CONFIG"):
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate
        logger.warning("config_file_env_not_found", path=str(candidate))
        return None

    for candidate in (_LOCAL_CONFIG, _USER_CONFIG):
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Read the TOML config file and return its contents as a nested dict.

    Returns an empty dict when *path* is None or the file cannot be read.
    """
    if path is None:
        path = find_config_file()
    if path is None:
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("config_file_load_failed", path=str(path), error=str(exc))
        return {}
    logger.debug("config_file_loaded", path=str(path))
    return data


def apply_overrides[T](instance: T, overrides: dict[str, Any]) -> T:
    """
    Return a copy of *instance* with dataclass fields replaced by *overrides*.

    Only keys that match an existing field name are applied; unknown keys are
    silently skipped so the TOML file can contain forward-compatible entries
    that older versions ignore.
    """
    valid = {f.name for f in fields(instance)}  # type: ignore[arg-type]
    applicable = {k: v for k, v in overrides.items() if k in valid}
    if not applicable:
        return instance
    return replace(instance, **applicable)  # type: ignore[misc,no-any-return]
