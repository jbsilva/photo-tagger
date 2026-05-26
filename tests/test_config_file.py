"""Tests for the TOML config file loader and dataclass override logic."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from photo_tagger.config_file import apply_overrides, find_config_file, load_config


@pytest.fixture(autouse=True)
def _isolate_config_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Prevent tests from picking up the developer's real config files.

    ``find_config_file`` consults ``$PHOTO_TAGGER_CONFIG`` and the module-level ``_USER_CONFIG``
    path under the user's home directory. If either exists on the machine running the tests,
    assertions about "no config found" silently fail.
    Redirect both to a tmp location so every test starts from a clean slate.
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
