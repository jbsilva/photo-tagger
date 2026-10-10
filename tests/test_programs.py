"""Tests for finding helper programs without searching the current directory."""

import os
import stat
from typing import TYPE_CHECKING

import pytest

from photo_tagger.programs import find_program


if TYPE_CHECKING:
    from pathlib import Path


def _tool(folder: Path, name: str) -> Path:
    """Create an executable file *name* in *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_program_ignores_the_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy in the folder photo-tagger runs from never wins over the one on PATH."""
    photos = tmp_path / "photos"
    decoy = _tool(photos, "exiftool")
    real = _tool(tmp_path / "bin", "exiftool")
    monkeypatch.chdir(photos)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", str(real.parent)]))

    assert find_program("exiftool") == str(real)
    assert find_program("exiftool") != str(decoy)


def test_find_program_is_none_when_only_the_current_directory_has_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative PATH entry stands for the current directory, so it is not searched either."""
    _tool(tmp_path, "notify-send")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", ".")

    assert find_program("notify-send") is None


def test_find_program_checks_a_path_with_a_folder_as_given(tmp_path: Path) -> None:
    """An explicit exiftool_path is used as it stands."""
    tool = _tool(tmp_path / "opt", "exiftool")
    assert find_program(str(tool)) == str(tool)
