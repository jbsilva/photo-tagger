"""
The ``show`` and ``set`` commands: read a photo's fields, and write them without the model.

``show`` reports the title, description, keywords, and authorship fields a photo carries, and where
each one comes from. ``set`` writes the copyright and credit fields, or any ``--set`` tag.
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - cyclopts resolves the flag annotations at runtime.
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter, validators
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from photo_tagger import telemetry
from photo_tagger.cli_options import (
    DEFAULT_COPYRIGHT_OPTIONS,
    DEFAULT_EXTENSIONS,
    DEFAULT_RECURSIVE,
    CopyrightConfig,
    OutputConfig,
    resolve_sidecar_mode,
    resolve_sidecar_overrides,
    to_extra_fields,
    validate_sidecar_overrides,
)

# Runtime import: cyclopts evaluates the Annotated[SidecarMode, ...] flag of the set command.
from photo_tagger.config import SidecarMode  # noqa: TC001
from photo_tagger.config_file import export_exiftool_path
from photo_tagger.discovery import resolve_image_batch
from photo_tagger.errors import (
    DiscoveryError,
    FieldError,
)
from photo_tagger.fields import SHOWN_FIELDS
from photo_tagger.metadata import (
    SOURCE_SIDECAR,
    FieldSources,
    managed_helper,
    read_all_tags,
    read_field_sources,
    read_fields,
    sidecar_can_hold,
    write_metadata_everywhere,
)
from photo_tagger.models import KeywordSet
from photo_tagger.undo import open_journal
from photo_tagger.write_plan import plan_writes


if TYPE_CHECKING:
    from exiftool import ExifToolHelper  # type: ignore[attr-defined]

    from photo_tagger.fields import ExtraFields
    from photo_tagger.metadata import FieldReading
    from photo_tagger.undo import UndoJournal


def _quiet_logging() -> None:
    """
    Leave only warnings and errors on stderr, for the commands whose output is the report.

    ``show`` and ``set`` print what they found or wrote; the usual INFO stream would bury it. Going
    quiet is not the same as going silent, though: a failed ExifTool read must not come out as a
    photo with no metadata, so warnings still reach the terminal.
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING", format="{message}")


def _two_column_table(first: str, second: str) -> Table:
    """Build the borderless two-column table both ``show`` layouts print."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(first, style="cyan", no_wrap=True)
    table.add_column(second, overflow="fold")
    return table


def _render_readings(console: Console, readings: list[FieldReading]) -> None:
    """Print one photo's fields, naming a label once and the tag behind every value."""
    table = _two_column_table("Field", "Value")
    table.add_column("Tag", style="dim", no_wrap=True)
    previous = ""
    for reading in readings:
        label = "" if reading.label == previous else reading.label
        previous = reading.label
        tag = f"{reading.tag} (sidecar)" if reading.source == SOURCE_SIDECAR else reading.tag
        # Values come off the file, so they are printed as text: a keyword like "[b]" is a
        # keyword, not rich markup.
        table.add_row(label, Text(reading.value), tag)
    console.print(table)


def _render_tag_dump(console: Console, by_file: dict[str, dict[str, str]]) -> None:
    """
    Print every tag ExifTool reported, one table per file it came from.

    Each file is its own heading here, unlike the curated view: a sidecar and its photo can hold
    different values for the same tag, and ``--all`` is where somebody goes to see which is which.
    """
    for source, tags in by_file.items():
        console.print(source, style="bold")
        table = _two_column_table("Tag", "Value")
        for tag, value in sorted(tags.items()):
            table.add_row(tag, Text(value))
        console.print(table)


def _emit_json_line(payload: dict[str, object]) -> None:
    """Write one JSON object to stdout, unwrapped, so a pipe gets exactly one line per photo."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _show_one(
    console: Console,
    image_file: Path,
    *,
    all_tags: bool,
    as_json: bool,
    et: ExifToolHelper | None = None,
) -> None:
    """Report a single photo, as a table or as one JSON line."""
    if all_tags:
        by_file = read_all_tags(image_file, et=et)
        if as_json:
            _emit_json_line({"file": str(image_file), "tags": by_file})
        else:
            _render_tag_dump(console, by_file)
        return

    readings = read_fields(image_file, SHOWN_FIELDS, et=et)
    if as_json:
        _emit_json_line({"file": str(image_file), "fields": [asdict(r) for r in readings]})
        return
    console.print(str(image_file), style="bold")
    if readings:
        _render_readings(console, readings)
    else:
        console.print("  no metadata", style="dim")


def show(
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            validator=validators.Path(exists=True),
            help="One or more paths: files and/or directories (repeat this option)",
        ),
    ] = None,
    *,
    image_extensions: Annotated[
        str,
        Parameter(name=("--ext", "--extensions"), help="Extensions to scan for (case insensitive)"),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(name=("--recursive", "-r"), help="Scan subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    all_tags: Annotated[
        bool,
        Parameter(
            name=("--all",),
            help="Print every tag ExifTool reports, not the curated list. Binary payloads are cut",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        Parameter(name=("--json",), help="Emit one JSON object per photo instead of a table"),
    ] = False,
) -> None:
    """
    Print the metadata a photo already carries, without changing anything.

    The read side of ``set``: title, description, keywords, creator, copyright, camera, and place,
    each with the ExifTool tag it came from and a note when the value lives in the XMP sidecar
    rather than the photo. A field two tags disagree about is listed twice, which is the whole point
    of showing the tag: it is how a stale caption in the photo, shadowed by its sidecar, becomes
    visible.

    Add ``--all`` for every tag ExifTool can read, and ``--json`` to pipe any of it elsewhere.

    Examples:
        photo-tagger show -i IMG_0001.CR3

        photo-tagger show -i ~/Pictures/Shoot -r --json | jq '.fields[] | select(.label)'
    """
    export_exiftool_path()
    _quiet_logging()
    console = Console()
    # resolve_image_batch raises rather than returning an empty batch, so "nothing matched" and
    # "that path is unusable" arrive here as the same clean error.
    try:
        image_files = resolve_image_batch(inputs, image_extensions, recursive=recursive)
    except DiscoveryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    telemetry.note(photos=len(image_files))
    # One ExifTool process for the whole report: a folder of five thousand photos must not pay a
    # process spawn each.
    with managed_helper(None) as helper:
        for image_file in image_files:
            _show_one(console, image_file, all_tags=all_tags, as_json=as_json, et=helper)


@dataclass(slots=True)
class _FieldsTally:
    """What a ``set`` run did to each photo."""

    written: int = 0
    failed: list[Path] = field(default_factory=list)
    # A template found nothing on the photo, so that field was left unwritten.
    skipped: int = 0
    # Everything asked for was already on the photo, and --preserve-copyright kept it.
    kept: int = 0


def _write_fields(
    image_files: list[Path],
    fields: ExtraFields,
    *,
    output: OutputConfig,
    journal: UndoJournal | None,
) -> _FieldsTally:
    """
    Stamp every photo with *fields*, filled from its own tags, where *output* says to write.

    One ExifTool process for the batch, and one batched read of the tags the templates need before
    it, so a thousand photos cost two round trips plus one write each.
    """
    tally = _FieldsTally()
    with managed_helper(None) as helper:
        sources = read_field_sources(image_files, fields.source_tags, et=helper)
        for image_file in image_files:
            missing: list[str] = []
            extra = _resolve_for(fields, image_file, sources[image_file], output, skipped=missing)
            tally.skipped += bool(missing)
            if not extra:
                # Nothing to write because the photo keeps what it has, not because a template
                # found nothing: that one is counted as skipped.
                tally.kept += not missing
            elif _write_extra(image_file, extra, output=output, journal=journal, et=helper):
                tally.written += 1
            else:
                tally.failed.append(image_file)
    return tally


def _resolve_for(
    fields: ExtraFields,
    image_file: Path,
    sources: FieldSources,
    output: OutputConfig,
    *,
    skipped: list[str] | None = None,
) -> dict[str, str]:
    """Render one photo's fields, preserving only what the files this save writes carry."""
    targets = plan_writes(
        image_file,
        resolve_sidecar_mode(output),
        resolve_sidecar_overrides(output),
    )
    return fields.resolve(
        sources.values,
        existing=sources.written(targets),
        photo=str(image_file),
        skipped=skipped,
    )


def _write_extra(
    image_file: Path,
    extra: dict[str, str],
    *,
    output: OutputConfig,
    journal: UndoJournal | None,
    et: ExifToolHelper,
) -> bool:
    """Write one photo's resolved fields and journal the files it touched."""
    backup = output.backup_xmp
    ok, targets = write_metadata_everywhere(
        image_file,
        KeywordSet(),
        extra=extra,
        backup=backup,
        sidecar_mode=resolve_sidecar_mode(output),
        sidecar_overrides=resolve_sidecar_overrides(output),
        et=et,
    )
    if journal is not None:
        journal.record_writes(image_file, targets, backup=backup)
    return ok


def set_fields(  # noqa: PLR0913 - cyclopts entry point; each arg is a CLI flag.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            validator=validators.Path(exists=True),
            help="One or more paths: files and/or directories (repeat this option)",
        ),
    ] = None,
    *,
    image_extensions: Annotated[
        str,
        Parameter(name=("--ext", "--extensions"), help="Extensions to scan for (case insensitive)"),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(name=("--recursive", "-r"), help="Scan subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    sidecar_mode: Annotated[
        SidecarMode | None,
        Parameter(
            name=("--sidecar-mode",),
            help=(
                "Where the fields go: 'all' (the default) writes an XMP sidecar next to every "
                "photo, 'none' embeds them in the photo itself, 'raw' splits by file type, 'both' "
                "writes each photo and a sidecar beside it"
            ),
        ),
    ] = None,
    use_sidecar: Annotated[
        bool | None,
        Parameter(
            name=("--write-sidecar",),
            negative="--embed-in-photo",
            help="Shorthand for --sidecar-mode: 'all' or 'none'. An explicit mode wins over both",
        ),
    ] = None,
    sidecar_overrides: Annotated[
        list[str] | None,
        Parameter(
            name=("--sidecar-override",),
            validator=validate_sidecar_overrides,
            help=(
                "Send one file type somewhere else than --sidecar-mode says, as EXT=TARGET with "
                "TARGET one of 'sidecar', 'photo', or 'both' (repeat this option)"
            ),
        ),
    ] = None,
    backup_xmp: Annotated[
        bool,
        Parameter(
            name=("--backup-xmp",),
            negative="--no-backup-xmp",
            help="Create an ExifTool backup (_original) before overwriting metadata",
        ),
    ] = True,
    undo_log: Annotated[
        bool,
        Parameter(
            name=("--undo-log",),
            negative="--no-undo-log",
            help="Record what this run writes so 'photo-tagger undo' can put it back",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        Parameter(name=("--dry-run",), help="Print the tags each photo would get, and write none"),
    ] = False,
    copyright_config: Annotated[CopyrightConfig, Parameter(name="*")] = DEFAULT_COPYRIGHT_OPTIONS,
) -> None:
    """
    Write copyright, creator, and any other tag, without calling a model.

    The fields no model can guess. ``tag`` writes them too, in the same pass as the generated
    metadata; this is for the photos that are already tagged, or for a library you want credited
    without paying for inference.

    The year comes from each photo, not from today: ``{year}`` in a notice becomes the year that
    photo was taken (its capture date in EXIF, XMP, or IPTC), so one command stamps a decade of
    shoots correctly. A photo that already has a name or a notice keeps it, unless you pass
    ``--no-preserve-copyright``.

    No keyword, title, or description is read or rewritten. Like every save, a write also names
    photo-tagger in the file (ProcessingSoftware, and an xmpMM:History event). Writes are recorded,
    so ``photo-tagger undo`` puts back a run that went to the wrong folder.

    Exit status: 1 when nothing was asked for, no photo matched, or any write failed.

    Examples:
        photo-tagger set -i ~/Pictures/2025 -r --embed-in-photo \
            --artist "Ada Lovelace" \
            --copyright "Copyright (c) {year} {artist}, All rights reserved" \
            --marked

        photo-tagger set -i photo.jpg --set "XMP-xmpRights:WebStatement=https://example.com/terms"
    """
    export_exiftool_path()
    _quiet_logging()
    console = Console()
    try:
        fields = to_extra_fields(copyright_config)
    except FieldError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    if fields.is_empty:
        console.print(
            "[yellow]Nothing to set. Pass --artist, --copyright, --rights, --marked, or "
            "--set TAG=VALUE.[/yellow]",
        )
        raise SystemExit(1)

    try:
        image_files = resolve_image_batch(inputs, image_extensions, recursive=recursive)
    except DiscoveryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    output = OutputConfig(
        backup_xmp=backup_xmp,
        sidecar_mode=sidecar_mode,
        use_sidecar=use_sidecar,
        sidecar_overrides=sidecar_overrides,
    )
    telemetry.note(photos=len(image_files))
    _refuse_fields_no_target_can_hold(console, image_files, fields, output)
    if dry_run:
        _preview_fields(console, image_files, fields, output)
        return

    tally = _write_fields(
        image_files,
        fields,
        output=output,
        journal=open_journal(datetime.now(tz=UTC), enabled=undo_log),
    )
    telemetry.note(written=tally.written, failed=len(tally.failed))
    _report_fields(console, tally, len(image_files))


def _report_fields(console: Console, tally: _FieldsTally, total: int) -> None:
    """Print what a ``set`` run did, and exit 1 when any write failed."""
    console.print(f"[green]Wrote {tally.written} of {total} photo(s).[/green]")
    if tally.kept:
        console.print(
            f"{tally.kept} photo(s) already carried everything asked for, and kept it. Pass "
            "--no-preserve-copyright to replace it.",
        )
    if tally.skipped:
        console.print(
            f"[yellow]{tally.skipped} photo(s) got no notice: the template found nothing on "
            'them. End the placeholder with a fallback, such as {year|""}, to write one '
            "anyway.[/yellow]",
        )
    if tally.failed:
        console.print(f"[red]{len(tally.failed)} failed; see the log for what ExifTool said.[/red]")
        raise SystemExit(1)


def _refuse_fields_no_target_can_hold(
    console: Console,
    image_files: list[Path],
    fields: ExtraFields,
    output: OutputConfig,
) -> None:
    """
    Stop before the first photo when every write goes to a sidecar and no tag fits in one.

    A sidecar holds XMP only, so a --set of EXIF or IPTC tags would fail on every photo, one log
    line at a time, and end in a count of failures that does not say why.
    """
    mode = resolve_sidecar_mode(output)
    overrides = resolve_sidecar_overrides(output)
    only_sidecars = all(
        target.use_sidecar
        for image_file in image_files
        for target in plan_writes(image_file, mode, overrides)
    )
    if only_sidecars and not sidecar_can_hold(fields.preview(datetime.now(tz=UTC).year)):
        console.print(
            "[red]Every tag asked for is EXIF or IPTC, and an XMP sidecar holds neither. Write "
            "into the photos with --embed-in-photo or --sidecar-mode both.[/red]",
        )
        raise SystemExit(1)


def _preview_fields(
    console: Console,
    image_files: list[Path],
    fields: ExtraFields,
    output: OutputConfig,
) -> None:
    """Print the tags each photo would get under ``--dry-run``, without opening it for writing."""
    sources = read_field_sources(image_files, fields.source_tags)
    for image_file in image_files:
        console.print(str(image_file), style="bold")
        table = _two_column_table("Tag", "Value")
        for tag, value in _resolve_for(fields, image_file, sources[image_file], output).items():
            table.add_row(tag, Text(value))
        console.print(table)
