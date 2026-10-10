"""The ``undo`` command: put back what a run wrote, from its journal."""

from pathlib import Path  # noqa: TC003 - cyclopts resolves the flag annotations at runtime.
from typing import Annotated

from cyclopts import Parameter, validators
from rich.console import Console

from photo_tagger import telemetry
from photo_tagger.undo import (
    DELETED,
    RESTORED,
    UndoError,
    UndoResult,
    latest_journal,
    list_journals,
    read_journal,
    undo_run,
)


def _render_journal_list(console: Console) -> None:
    """Print the recorded runs, newest first, with how many files each one wrote."""
    journals = list_journals()
    if not journals:
        console.print("No recorded runs to undo.")
        return
    console.print("Recorded runs (newest first):\n")
    for path in journals:
        try:
            entries = len(read_journal(path))
        except UndoError:  # pragma: no cover - listing must survive one unreadable journal
            entries = 0
        console.print(f"  {path.name}  {entries} file(s)  {path}")


# Undo outcomes that mean the file is back as it was. Anything else needs the user's attention,
# so it decides the exit code.
_UNDO_OK_ACTIONS = frozenset({RESTORED, DELETED})


def _render_undo_results(results: list[UndoResult], console: Console, *, dry_run: bool) -> bool:
    """Print one line per entry and return True when every one of them was put back."""
    verb = "Would undo" if dry_run else "Undoing"
    console.print(f"{verb} {len(results)} write(s)\n")
    for result in results:
        ok = result.action in _UNDO_OK_ACTIONS
        colour = "green" if ok else "yellow"
        mark = f"[{colour}]{result.action:<9}[/{colour}]"
        detail = f"  ({result.detail})" if result.detail else ""
        console.print(f"  {mark}  {result.target}{detail}")
    skipped = [result for result in results if result.action not in _UNDO_OK_ACTIONS]
    if skipped:
        console.print(f"\n[yellow]{len(skipped)} entry/entries left alone.[/yellow]")
    else:
        console.print("\n[green]Every recorded write was put back.[/green]")
    return not skipped


def undo(
    *,
    run: Annotated[
        Path | None,
        Parameter(
            name=("--run",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help="Undo this journal instead of the most recent run",
        ),
    ] = None,
    show_list: Annotated[
        bool,
        Parameter(name=("--list",), help="List the recorded runs and exit"),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(name=("--dry-run",), help="Report what would be put back, without touching it"),
    ] = False,
    force: Annotated[
        bool,
        Parameter(
            name=("--force",),
            help=(
                "Revert files that changed after the run wrote them. Without this they are left "
                "alone, because a change means someone edited the file since"
            ),
        ),
    ] = False,
) -> None:
    """
    Put back what the last tagging run wrote.

    Every run records the files it writes (unless ``--no-undo-log`` was passed), so a batch tagged
    with the wrong prompt or the wrong vocabulary can be reverted in one command: sidecars the run
    created are deleted, and files it overwrote are restored from ExifTool's ``*_original`` backup.

    A file that changed since the run is left alone unless ``--force`` says otherwise, and a file
    written with ``--no-backup-xmp`` cannot be restored at all: there is no copy of what it held.
    Saves from the desktop GUI land in the same journal folder, unless its undo recording is off.

    Exit status: 1 when there is nothing to undo or when any entry was left alone; 0 when every
    recorded write was put back.
    """
    console = Console()
    if show_list:
        _render_journal_list(console)
        return

    journal_path = run or latest_journal()
    if journal_path is None:
        console.print("No recorded runs to undo.")
        raise SystemExit(1)

    try:
        records = read_journal(journal_path)
    except UndoError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    if not records:
        console.print(f"{journal_path} records no writes.")
        raise SystemExit(1)

    console.print(f"Undoing run {journal_path.name}")
    results = undo_run(records, force=force, dry_run=dry_run)
    put_back = sum(result.action in _UNDO_OK_ACTIONS for result in results)
    telemetry.note(entries=len(results), put_back=put_back, left_alone=len(results) - put_back)
    if not _render_undo_results(results, console, dry_run=dry_run):
        raise SystemExit(1)
