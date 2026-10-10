"""
TOML-based configuration file loader.

Searches for a config file in standard locations and returns its contents as a nested dict. The CLI
feeds it through cyclopts' config layer (``cli_options.ConfigFileSource``) so flags the user does
not pass pick up config values, while the GUI folds it into concrete defaults via
``cli_options.load_defaults``.

Search order (first match wins):
1. ``$PHOTO_TAGGER_CONFIG`` environment variable (explicit path)
2. ``.photo-tagger.toml`` in the current working directory (project-local)
3. ``~/.config/photo-tagger/config.toml`` (XDG-style user default)
"""

import os
import tomllib
import types
import typing
from dataclasses import fields, replace
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin

from loguru import logger


_USER_CONFIG = Path.home() / ".config" / "photo-tagger" / "config.toml"
_LOCAL_CONFIG = Path(".photo-tagger.toml")


def user_config_path() -> Path:
    """Return the XDG-style user config path; the GUI writes its saved defaults here."""
    return _USER_CONFIG


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


def config_file_kind() -> str:
    """Name where the config came from, never its path: ``env``, ``cwd``, ``user``, or ``none``."""
    path = find_config_file()
    if path is None:
        return "none"
    if os.getenv("PHOTO_TAGGER_CONFIG"):
        return "env"
    return "cwd" if path == _LOCAL_CONFIG else "user"


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
    if path == _LOCAL_CONFIG and data.get("exiftool_path"):
        # A CWD-local config names the "ExifTool" binary this run will execute. Running inside a
        # directory someone else prepared (an unpacked archive, a cloned repo) would silently run
        # whatever that file points at, so make the redirect visible.
        logger.warning(
            "exiftool_path_from_local_config",
            exiftool_path=str(data["exiftool_path"]),
            config=str(path.resolve()),
        )
    logger.debug("config_file_loaded", path=str(path))
    return data


def configured_exiftool_path() -> str | None:
    """Return the top-level ``exiftool_path`` config value, or None when unset."""
    value = load_config().get("exiftool_path")
    return str(value) if value else None


def export_exiftool_path() -> None:
    """Bridge a config ``exiftool_path`` into ``PHOTO_TAGGER_EXIFTOOL`` (an exported var wins)."""
    if path := configured_exiftool_path():
        os.environ.setdefault("PHOTO_TAGGER_EXIFTOOL", path)


def _coerce_field(annotation: object, value: object) -> object:
    """
    Convert a raw TOML value to the type required by *annotation*.

    TOML only knows strings, ints, floats, bools, datetimes, arrays, and tables. Dataclass fields
    typed as ``Path`` or ``Path | None`` therefore arrive as plain strings and must be wrapped
    explicitly.
    """
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    if not isinstance(value, str):
        return value

    origin = get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        if Path in get_args(annotation):
            return Path(value)
    elif annotation is Path:
        return Path(value)

    return value


def apply_overrides[T](instance: T, overrides: dict[str, Any]) -> T:
    """
    Return a copy of *instance* with dataclass fields replaced by *overrides*.

    Only keys that match an existing field name are applied; unknown keys are silently skipped so
    the TOML file can contain forward-compatible entries that older versions ignore.
    """
    valid = {f.name for f in fields(instance)}  # type: ignore[arg-type]
    applicable = {k: v for k, v in overrides.items() if k in valid}
    if not applicable:
        return instance

    try:
        hints = typing.get_type_hints(type(instance), include_extras=True)
    except NameError, AttributeError, TypeError:
        hints = {}

    coerced = {k: _coerce_field(hints.get(k, type(v)), v) for k, v in applicable.items()}
    return replace(instance, **coerced)  # type: ignore[misc,no-any-return]
