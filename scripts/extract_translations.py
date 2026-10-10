#!/usr/bin/env python3
"""
Refresh the .pot template from the source and merge it into every locale's .po catalog.

Wraps ``pybabel extract`` + ``pybabel update`` so the header metadata (copyright holder, bug-report
address, translator) is filled in consistently instead of Babel's FIRST AUTHOR / ORGANIZATION
placeholders. Run it after changing translatable strings:

uv run scripts/extract_translations.py

Then recompile the runtime catalogs:

uv run scripts/compile_translations.py
"""

import os
import re
from pathlib import Path

from babel.messages.frontend import CommandLineInterface


# Paths are kept relative to the repo root (main() changes into it) so the "#: file:line"
# references in the committed catalogs stay machine-independent.
ROOT = Path(__file__).parent.parent
SOURCE_DIR = Path("src") / "photo_tagger"
LOCALE_DIR = SOURCE_DIR / "locale"
DOMAIN = "photo_tagger"
POT_FILE = LOCALE_DIR / f"{DOMAIN}.pot"

TRANSLATOR = "Julio Batista Silva <python@juliobs.com>"
BUGS_ADDRESS = "python@juliobs.com"
COPYRIGHT_HOLDER = "Julio Batista Silva"

# Babel's default template says "FIRST AUTHOR <EMAIL@ADDRESS>, YEAR."; this replaces that line.
# PROJECT, YEAR, and ORGANIZATION are substituted by Babel at extraction time.
HEADER_COMMENT = f"""\
# Translations template for PROJECT.
# Copyright (C) YEAR ORGANIZATION
# This file is distributed under the same license as the PROJECT project.
# {TRANSLATOR}, YEAR.
#"""


# Babel stamps the extraction time into every catalog, so a run that finds nothing new still
# changes each file. That line alone is no change worth committing.
_CREATION_DATE = re.compile(r'^"POT-Creation-Date: [^"]*"$', re.MULTILINE)


def _catalogs() -> list[Path]:
    """List the template and every locale's catalog, the files a run rewrites."""
    return [POT_FILE, *sorted(LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.po"))]


def _keep_if_only_dated(path: Path, before: str | None) -> None:
    """Put *before* back when the rewrite of *path* changed nothing but its creation date."""
    if before is None:
        return
    after = path.read_text(encoding="utf-8")
    if _CREATION_DATE.sub("", after) == _CREATION_DATE.sub("", before):
        path.write_text(before, encoding="utf-8")


def _pybabel(*argv: str) -> None:
    # Babel's CommandLineInterface.run has no type annotations, hence the targeted ignore.
    CommandLineInterface().run(["pybabel", *argv])  # type: ignore[no-untyped-call]


def _ensure_trailing_newline(path: Path) -> None:
    """Babel ends catalogs with a blank line; the end-of-file-fixer hook wants a single newline."""
    text = path.read_text(encoding="utf-8")
    normalized = text.rstrip("\n") + "\n"
    if normalized != text:
        path.write_text(normalized, encoding="utf-8")


def main() -> int:
    """Extract the template, then merge it into the per-locale catalogs; return the exit code."""
    os.chdir(ROOT)
    before = {path: path.read_text(encoding="utf-8") for path in _catalogs() if path.exists()}
    # Besides the gettext defaults, two of our own helpers take a translatable literal:
    # gettext_noop (mark now, translate later) and gui_state.tooltip (translate and word-wrap).
    _pybabel(
        "extract",
        "-k",
        "gettext_noop",
        "-k",
        "tooltip",
        "--project",
        "photo-tagger",
        "--copyright-holder",
        COPYRIGHT_HOLDER,
        "--msgid-bugs-address",
        BUGS_ADDRESS,
        "--last-translator",
        TRANSLATOR,
        "--header-comment",
        HEADER_COMMENT,
        "-o",
        str(POT_FILE),
        str(SOURCE_DIR),
    )
    _pybabel("update", "-i", str(POT_FILE), "-d", str(LOCALE_DIR), "-D", DOMAIN)
    for path in _catalogs():
        _ensure_trailing_newline(path)
        _keep_if_only_dated(path, before.get(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
