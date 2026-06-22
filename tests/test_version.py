"""The package version is single-sourced from installed distribution metadata."""

import re
import tomllib
from pathlib import Path

import photo_tagger


_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_version_is_a_release_identifier() -> None:
    """__version__ is a non-empty string shaped like a release identifier."""
    assert isinstance(photo_tagger.__version__, str)
    assert re.match(r"^\d+\.\d+", photo_tagger.__version__)


def test_version_matches_pyproject() -> None:
    """
    The runtime version agrees with the canonical pyproject.toml value.

    This is the metadata-path counterpart to ``scripts/check_version_sync.py``:
    it catches a stale editable install whose recorded version drifted from
    ``[project].version`` after a bump.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    assert photo_tagger.__version__ == data["project"]["version"]
