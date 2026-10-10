#!/usr/bin/env python3
# ruff: noqa: T201
"""
Check that the user-facing docs still describe the CLI the code actually exposes.

Docs rot in two directions, and both are caught here by walking the live cyclopts app instead of a
hand-maintained list:

* A new flag or command ships without a mention anywhere (silently undocumented).
* A flag is renamed or dropped while the docs keep advertising the old name (silently wrong).

Run it with ``uv run scripts/check_docs_sync.py``; it needs the project environment because it
imports the app. Exits 0 when the docs agree with the CLI, 1 otherwise.
"""

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from photo_tagger.main import app


if TYPE_CHECKING:
    from cyclopts import App


ROOT = Path(__file__).resolve().parent.parent

# The page that promises to document every flag. Anything the CLI accepts must appear here.
FLAG_REFERENCE = "docs/usage/cli-reference.md"

# Pages that must name every subcommand, so neither the entry page nor the reference can forget one.
COMMAND_PAGES = ("README.md", FLAG_REFERENCE)

# Everything a user reads to learn the CLI. A flag-shaped token in any of these must be real.
USER_FACING_DOCS = (
    "README.md",
    "docs/index.md",
    "docs/telemetry.md",
    "docs/troubleshooting.md",
    "docs/usage",
    "docs/getting-started",
)

# Cyclopts adds these to every app; they are CLI plumbing, not features to document.
BUILTIN_COMMANDS = frozenset({"help-print", "version-print"})
BUILTIN_FLAGS = frozenset({"--help", "--version"})

_FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")

# Docs also show commands from other tools, whose flags are none of our business. A line that
# invokes one of these and does not mention photo-tagger is skipped by the stale-flag scan.
_FOREIGN_COMMAND_RE = re.compile(
    r"\b(uv|uvx|pip|pipx|conda|mamba|pixi|brew|pybabel|exiftool|zensical|prek|git|gpgconf)\b",
)


def _flag_names(target: App) -> list[tuple[str, ...]]:
    """List each parameter of *target* as the tuple of long option names it answers to."""
    return [
        names
        for argument in target.assemble_argument_collection()
        if (names := tuple(n for n in argument.names if n.startswith("--")))
    ]


def _cli_surface() -> tuple[set[str], list[tuple[str, ...]]]:
    """
    Walk the app and its subcommands, returning the command names and every parameter's names.

    Command names come from the app's own registry, which is what a user types. A sub-app reports
    the name of the function behind it instead, and the two differ whenever a command is registered
    under a name of its own (``set``, from ``set_fields``).

    ``watch`` re-exposes the whole tagging option set, so parameters are deduplicated: one flag
    should be reported once, however many commands accept it.
    """
    commands = {name for name in app if not name.startswith("-")} - BUILTIN_COMMANDS
    parameters = dict.fromkeys(_flag_names(app))
    for subapp in app.subapps:
        if subapp.name[0] in BUILTIN_COMMANDS:
            continue
        parameters.update(dict.fromkeys(_flag_names(subapp)))
    return commands, list(parameters)


def _doc_paths() -> list[Path]:
    """Expand USER_FACING_DOCS into concrete Markdown files."""
    paths = []
    for entry in USER_FACING_DOCS:
        target = ROOT / entry
        paths.extend(sorted(target.rglob("*.md")) if target.is_dir() else [target])
    return paths


def _missing_commands(commands: set[str]) -> list[str]:
    """Report subcommands that a page listing the commands never mentions."""
    problems = []
    for page in COMMAND_PAGES:
        text = (ROOT / page).read_text()
        problems.extend(
            f"{page}: subcommand 'photo-tagger {command}' is not documented"
            for command in sorted(commands)
            if f"photo-tagger {command}" not in text
        )
    return problems


def _missing_flags(parameters: list[tuple[str, ...]]) -> list[str]:
    """Report parameters the flag reference documents under none of their names."""
    text = (ROOT / FLAG_REFERENCE).read_text()
    return [
        f"{FLAG_REFERENCE}: {names[0]} is not documented"
        for names in parameters
        # Not \b: it sits between "--vocabulary" and the "-strict" of a longer flag, so
        # documenting only --vocabulary-strict would have counted for --vocabulary too.
        if not any(re.search(rf"{re.escape(name)}(?![\w-])", text) for name in names)
    ]


def _stale_flags(known: set[str]) -> list[str]:
    """Report flag-shaped tokens in the user-facing docs that the CLI no longer accepts."""
    problems = []
    for path in _doc_paths():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if _FOREIGN_COMMAND_RE.search(line) and "photo-tagger" not in line:
                continue
            for token in sorted(set(_FLAG_RE.findall(line)) - known):
                rel = path.relative_to(ROOT)
                problems.append(f"{rel}:{number}: {token} is not a photo-tagger flag")
    return problems


def main() -> int:
    """Compare the docs against the live CLI and report every disagreement."""
    commands, parameters = _cli_surface()
    known_flags = {name for names in parameters for name in names} | BUILTIN_FLAGS

    problems = _missing_commands(commands) + _missing_flags(parameters) + _stale_flags(known_flags)
    if problems:
        print("Docs are out of sync with the CLI:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"Docs cover {len(parameters)} flags and {len(commands)} subcommands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
