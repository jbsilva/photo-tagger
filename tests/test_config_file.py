"""Tests for the TOML config file loader and dataclass override logic."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest
from loguru import logger

from photo_tagger.config_file import (
    apply_overrides,
    config_file_kind,
    find_config_file,
    load_config,
)


@pytest.fixture(autouse=True)
def _isolate_config_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Prevent tests from picking up the developer's real config files.

    ``find_config_file`` consults ``$PHOTO_TAGGER_CONFIG`` and the module-level ``_USER_CONFIG``
    path under the user's home directory. If either exists on the machine running the tests,
    assertions about "no config found" silently fail. Redirect both to a tmp location so every test
    starts from a clean slate.
    """
    monkeypatch.delenv("PHOTO_TAGGER_CONFIG", raising=False)
    monkeypatch.setattr(
        "photo_tagger.config_file._USER_CONFIG",
        tmp_path / "nonexistent-home" / "config.toml",
    )


# --- apply_overrides ---------------------------------------------------------------------------


@dataclass
class _SampleDC:
    name: str = "default"
    count: int = 0
    flag: bool = False


@dataclass
class _PathDC:
    output: Path | None = None


def test_apply_overrides_replaces_matching_fields() -> None:
    """Known field names are replaced; the original instance is untouched."""
    original = _SampleDC()
    updated = apply_overrides(original, {"name": "custom", "count": 42})
    assert updated.name == "custom"
    assert updated.count == 42  # noqa: PLR2004 - asserting override value
    assert updated.flag is False
    # Original is unmodified.
    assert original.name == "default"


def test_apply_overrides_ignores_unknown_keys() -> None:
    """Keys that don't match a field are silently dropped."""
    result = apply_overrides(_SampleDC(), {"unknown_key": 99, "name": "ok"})
    assert result.name == "ok"
    assert result.count == 0


def test_apply_overrides_returns_original_when_nothing_applies() -> None:
    """An empty or fully-unknown override dict returns the original unchanged."""
    original = _SampleDC()
    assert apply_overrides(original, {}) is original
    assert apply_overrides(original, {"nope": 1}) is original


def test_apply_overrides_coerces_str_to_path() -> None:
    """TOML string values are converted to Path for Path | None fields."""
    result = apply_overrides(_PathDC(), {"output": "reports/run.json"})
    assert result.output == Path("reports/run.json")
    assert isinstance(result.output, Path)


def test_apply_overrides_coerces_annotated_path_and_leaves_other_unions() -> None:
    """_coerce_field unwraps Annotated, converts a bare Path, and leaves a non-Path union alone."""

    @dataclass
    class _AnnotatedDC:
        folder: Annotated[Path, "meta"] = Path()
        name: Annotated[str | None, "meta"] = None

    result = apply_overrides(_AnnotatedDC(), {"folder": "logs/today", "name": "keep"})
    assert result.folder == Path("logs/today")
    assert isinstance(result.folder, Path)
    assert result.name == "keep"


def test_apply_overrides_falls_back_when_type_hints_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If get_type_hints raises, coercion falls back to each value's runtime type."""

    def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        msg = "unresolved annotation"
        raise NameError(msg)

    monkeypatch.setattr("photo_tagger.config_file.typing.get_type_hints", boom)
    result = apply_overrides(_SampleDC(), {"count": 5})
    assert result.count == 5  # noqa: PLR2004 - asserting override value


# --- find_config_file --------------------------------------------------------------------------


def test_find_config_file_prefers_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PHOTO_TAGGER_CONFIG takes priority over automatic search paths."""
    cfg = tmp_path / "custom.toml"
    cfg.write_text("[provider]\nmodel_name = 'test'\n")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(cfg))
    assert find_config_file() == cfg


def test_find_config_file_env_var_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-existent env var path logs a warning and returns None."""
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(tmp_path / "nope.toml"))
    assert find_config_file() is None


def test_find_config_file_returns_none_when_nothing_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No config in CWD, home, or env returns None."""
    monkeypatch.delenv("PHOTO_TAGGER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_config_file() is None


def test_find_config_file_discovers_local_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """.photo-tagger.toml in CWD is picked up when no env var is set."""
    monkeypatch.delenv("PHOTO_TAGGER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / ".photo-tagger.toml"
    cfg.write_text("[inference]\ntemperature = 0.5\n")
    found = find_config_file()
    assert found is not None
    assert found.resolve() == cfg.resolve()


def test_config_file_kind_names_the_location_never_the_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Telemetry learns which lookup found the file (env, cwd, user, none), not where it is."""
    monkeypatch.chdir(tmp_path)
    assert config_file_kind() == "none"
    local = tmp_path / ".photo-tagger.toml"
    local.write_text("recursive = true\n")
    assert config_file_kind() == "cwd"
    local.unlink()
    user = tmp_path / "home" / "config.toml"
    user.parent.mkdir()
    user.write_text("recursive = true\n")
    monkeypatch.setattr("photo_tagger.config_file._USER_CONFIG", user)
    assert config_file_kind() == "user"
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(user))
    assert config_file_kind() == "env"


# --- load_config -------------------------------------------------------------------------------


def test_load_config_reads_toml(tmp_path: Path) -> None:
    """A valid TOML file is parsed into a nested dict."""
    cfg = tmp_path / "config.toml"
    cfg.write_text("[provider]\nmodel_name = 'my-model'\nretries = 3\n")
    data = load_config(cfg)
    assert data["provider"]["model_name"] == "my-model"
    assert data["provider"]["retries"] == 3  # noqa: PLR2004 - asserting parsed value


def test_load_config_returns_empty_for_none() -> None:
    """Passing None (no config found) returns an empty dict."""
    assert load_config(None) == {}


def test_load_config_returns_empty_on_parse_error(tmp_path: Path) -> None:
    """A broken TOML file degrades to no config instead of crashing."""
    cfg = tmp_path / "bad.toml"
    cfg.write_text("this is not valid TOML [[[")
    assert load_config(cfg) == {}


def test_load_config_returns_empty_on_io_error(tmp_path: Path) -> None:
    """An unreadable path degrades gracefully."""
    assert load_config(tmp_path / "nonexistent.toml") == {}


def test_load_config_flags_exiftool_path_from_the_cwd_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A CWD-local config that redirects the ExifTool binary is honored but loudly flagged."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".photo-tagger.toml").write_text('exiftool_path = "./evil"\n')

    events: list[str] = []
    handler = logger.add(lambda m: events.append(m.record["message"]), level="WARNING")
    try:
        data = load_config()
    finally:
        logger.remove(handler)

    assert data["exiftool_path"] == "./evil"
    assert "exiftool_path_from_local_config" in events


def test_load_config_does_not_flag_exiftool_path_from_an_explicit_file(tmp_path: Path) -> None:
    """The same key in a user-chosen config (home or $PHOTO_TAGGER_CONFIG) raises no warning."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('exiftool_path = "/usr/local/bin/exiftool"\n')

    events: list[str] = []
    handler = logger.add(lambda m: events.append(m.record["message"]), level="WARNING")
    try:
        load_config(cfg)
    finally:
        logger.remove(handler)

    assert "exiftool_path_from_local_config" not in events
