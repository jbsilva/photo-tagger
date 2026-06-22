#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
# ruff: noqa: T201
"""
Check that the project version is consistent across all source-of-truth files.

Exits 0 when every file agrees with pyproject.toml, 1 on any mismatch.
"""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
_CANONICAL = "pyproject.toml"


def _read(rel_path: str) -> str | None:
    """Read a file under ROOT, returning None if it is missing or unreadable."""
    try:
        return (ROOT / rel_path).read_text()
    except OSError:
        return None


def _grep(rel_path: str, pattern: str, flags: int = 0) -> str | None:
    """Return the first regex capture from a file under ROOT, or None."""
    text = _read(rel_path)
    if text is not None and (m := re.search(pattern, text, flags)):
        return m.group(1)
    return None


def _uv_lock_version() -> str | None:
    """Find the photo-tagger version inside uv.lock's package table."""
    text = _read("uv.lock")
    if text is None:
        return None
    in_pkg = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            in_pkg = False
        elif stripped == 'name = "photo-tagger"':
            in_pkg = True
        elif in_pkg and (m := re.match(r'^version\s*=\s*"([^"]+)"', line)):
            return m.group(1)
    return None


def _collect_versions() -> dict[str, str | None]:
    """
    Extract the version string from each source-of-truth file.

    The package no longer hardcodes its version: ``photo_tagger.__version__`` is
    read from the installed distribution metadata, so ``pyproject.toml`` is the
    only code-side source. The remaining entries are docs that must be bumped by
    hand on release, which is exactly what this check guards.
    """
    return {
        _CANONICAL: _grep(_CANONICAL, r'^version\s*=\s*"([^"]+)"', re.MULTILINE),
        "SECURITY.md": _grep("SECURITY.md", r"The current release is (\S+)\."),
        "CHANGELOG.md": _grep("CHANGELOG.md", r"^## \[(\d+\.\d+\.\d+)]", re.MULTILINE),
        "uv.lock": _uv_lock_version(),
    }


def main() -> int:
    """Compare versions across all source-of-truth files and report mismatches."""
    versions = _collect_versions()
    canonical = versions[_CANONICAL]
    if canonical is None:
        print(f"ERROR: could not parse version from {_CANONICAL}", file=sys.stderr)
        return 1

    ok = True
    for name, ver in versions.items():
        if ver is None:
            print(f"WARN: could not extract version from {name}", file=sys.stderr)
            ok = False
        elif ver != canonical:
            print(
                f"MISMATCH: {name} has {ver!r}, expected {canonical!r}",
                file=sys.stderr,
            )
            ok = False

    if ok:
        print(f"All versions match: {canonical}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
