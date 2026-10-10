#!/usr/bin/env python3
"""
Photo Tagger: CLI app to describe photos and add keywords using AI.

Optionally non-destructive: create/update XMP sidecar files with Lightroom-compatible tags.
Alternatively, pass --embed-in-photo to write metadata directly into the original file.
Unfortunately, Lightroom uses XMP sidecar files only for proprietary raw formats (e.g., CR3, NEF).
For JPEG, DNG and other formats, you'll often prefer embedding the metadata directly into the file.
A folder holding both is what --sidecar-mode raw is for: sidecars for the RAW files, embedded
metadata for the rest, in one pass. Mind that DNG counts as RAW there; add
--sidecar-override dng=photo to write into each DNG, as Lightroom Classic does.
This can be done with ExifTool manually as well:
    exiftool -tagsFromFile image.xmp -all:all image.jpg

Requirements:
 - Exiftool installed and available in PATH.
 - Ollama or LM Studio server running and containing a vision-language model.
"""

import contextlib
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - cyclopts resolves the flag annotations at runtime.
from typing import TYPE_CHECKING, Annotated

from cyclopts import App, Parameter, validators
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from photo_tagger import __version__, telemetry
from photo_tagger.ai import create_agent
from photo_tagger.cache import build_cache_namespace, open_cache
from photo_tagger.cli_options import (
    DEFAULT_EXTENSIONS,
    DEFAULT_RECURSIVE,
    DEFAULT_WORKERS,
    ArtifactConfig,
    ConfigFileSource,
    CopyrightConfig,
    DisplayConfig,
    FilterConfig,
    InferenceConfig,
    LogConfig,
    OutputConfig,
    PacingConfig,
    ProviderConfig,
    RunOptions,
    TelemetryConfig,
    VocabularyBuildConfig,
    WatchConfig,
    resolve_notify,
    resolve_sidecar_mode,
    resolve_sidecar_overrides,
    to_extra_fields,
    to_pacing_plan,
    to_processing_options,
    to_trim_rules,
    validate_pacing,
    validate_sidecar_overrides,
)
from photo_tagger.cli_reports import (
    CsvImageResultSink,
    NDJSONEmitter,
    alert_message,
    combine_image_result_callbacks,
    open_csv_report,
    write_summary_file,
)
from photo_tagger.cli_telemetry import (
    Beacons,
    PhotoStats,
    exit_status,
    invocation_setup,
    run_from_totals,
    show_notice_if_due,
)
from photo_tagger.config import DEFAULT_USER_PROMPT, SidecarMode
from photo_tagger.config_file import configured_exiftool_path, load_config
from photo_tagger.diagnostics import render_report, run_checks
from photo_tagger.discovery import (
    apply_date_filter,
    apply_skip_file,
    apply_skip_tagged,
    make_skip_list_appender,
    resolve_image_batch,
)
from photo_tagger.errors import (
    BatchError,
    DiscoveryError,
    FieldError,
    ObservabilityError,
    PhotoTaggerError,
)
from photo_tagger.fields import SHOWN_FIELDS
from photo_tagger.locking import FileLock, LockHeldError
from photo_tagger.logging_setup import setup_logging
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
from photo_tagger.notify import APP_NAME, notify
from photo_tagger.observability import configure_tracing
from photo_tagger.pacing import Pacer, build_pacer
from photo_tagger.pipeline import BatchTotals, ImageOutcome, ProcessingOptions, run_batch
from photo_tagger.progress import batch_progress
from photo_tagger.prompt_context import prompt_with_hint

# Runtime import (not type-only): cyclopts evaluates the Annotated[ProviderName, ...] field
# on the doctor command to validate the --provider choices, so it must exist at definition time.
from photo_tagger.providers import ProviderName  # noqa: TC001
from photo_tagger.sessions import plan_sessions
from photo_tagger.telemetry_facts import (
    joined,
)
from photo_tagger.undo import (
    DELETED,
    RESTORED,
    UndoError,
    UndoResult,
    latest_journal,
    list_journals,
    open_journal,
    read_journal,
    undo_run,
)
from photo_tagger.vocabulary import load_vocabulary, prompt_with_vocabulary
from photo_tagger.vocabulary_build import (
    KeywordCensus,
    census_from_export,
    census_from_photos,
    render_drop_report,
    render_vocabulary,
    trim,
    vocabulary_header,
)
from photo_tagger.vocabulary_organize import OrganizeStats, organize
from photo_tagger.watch import (
    watch_batches,
)
from photo_tagger.write_plan import plan_writes


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]
    from pydantic_ai import Agent

    from photo_tagger.cache import InferenceCache
    from photo_tagger.csv_report import CsvReportWriter
    from photo_tagger.fields import ExtraFields
    from photo_tagger.metadata import FieldReading
    from photo_tagger.models import GeneratedMetadata
    from photo_tagger.undo import UndoJournal


# Any TOML config is layered onto flags the user does not pass by the ConfigFileSource hook at
# parse time, giving per-field precedence (CLI flag > config file > built-in). The hook re-reads
# the file per invocation, so nothing config-dependent is captured at import.
# The hook also records the command and the options typed on it, which telemetry reports.
_CONFIG_SOURCE = ConfigFileSource()
app = App(name="photo-tagger", version=__version__, config=_CONFIG_SOURCE)


# Built-in default option groups, hoisted to module scope so the function-default expressions on
# `tag` are simple name lookups, which keeps ruff's B008 (no function call in a default argument)
# satisfied. Do NOT fold the config file into these instances: cyclopts rebuilds a group from its
# class defaults whenever any of the group's flags is passed, which would drop the config values
# of every sibling field in that group.
_DEFAULT_PROVIDER = ProviderConfig()
_DEFAULT_OUTPUT = OutputConfig()
_DEFAULT_INFERENCE = InferenceConfig()
_DEFAULT_LOG = LogConfig()
_DEFAULT_DISPLAY = DisplayConfig()
_DEFAULT_ARTIFACTS = ArtifactConfig()
_DEFAULT_FILTER = FilterConfig()
_DEFAULT_TELEMETRY = TelemetryConfig()
_DEFAULT_PACING = PacingConfig()
_DEFAULT_WATCH = WatchConfig()
# The `set` command's own group. `tag` and `watch` get these flags through OutputConfig, which
# inherits them; only a command that takes them *without* the rest of the output group needs this.
_DEFAULT_COPYRIGHT = CopyrightConfig()


def _apply_exiftool_path(path: str | None) -> None:
    """Bridge a config ``exiftool_path`` into ``PHOTO_TAGGER_EXIFTOOL`` (an exported var wins)."""
    if path:
        os.environ.setdefault("PHOTO_TAGGER_EXIFTOOL", path)


def _enable_tracing(log: LogConfig) -> None:
    """
    Switch on OpenTelemetry tracing when ``--otel`` asked for it, or stop with a clear reason.

    Called before any agent exists, because instrumentation is applied to the agent class rather
    than to instances. A failure here ends the run: somebody who passed ``--otel`` wants the traces,
    and silently tagging photos without them wastes the batch they were trying to observe.
    """
    try:
        configure_tracing(
            enabled=log.otel,
            endpoint=log.otel_endpoint,
            include_content=log.otel_content,
        )
    except ObservabilityError as exc:
        logger.error("otel_setup_failed", error=str(exc))
        raise SystemExit(1) from exc


@app.command
def doctor(
    *,
    provider: Annotated[
        ProviderName,
        Parameter(name=("--provider",), help="Backend provider to check"),
    ] = _DEFAULT_PROVIDER.provider_name,
    model: Annotated[
        str,
        Parameter(name=("--model", "-m"), help="Model name expected to be served"),
    ] = _DEFAULT_PROVIDER.model_name,
    url: Annotated[
        str | None,
        Parameter(name=("--url", "-u"), help="Provider API base URL"),
    ] = _DEFAULT_PROVIDER.api_base_url,
    api_key: Annotated[
        str | None,
        Parameter(name=("--api-key", "-k"), help="Provider API key (prefer env vars)"),
    ] = _DEFAULT_PROVIDER.api_key,
) -> None:
    """
    Check that ExifTool and the model provider are reachable, then exit.

    Prints a short checklist and exits 0 when everything is in order, 1 if any check fails. Run this
    first when a tagging run cannot start: it isolates a missing ExifTool, an unreachable provider,
    or a model name typo from the rest of the pipeline. Honors the same config file and env vars as
    ``tag``.
    """
    # Resolve the exiftool path (and its own warning, e.g. a CWD config redirecting the binary)
    # before silencing loguru, or that warning never reaches the user in the one command meant
    # to surface exactly this kind of misconfiguration.
    _apply_exiftool_path(configured_exiftool_path())
    # Silence loguru so only the checklist reaches the terminal; failures are
    # captured in the report itself, not the logs.
    logger.remove()
    results = run_checks(provider, model, api_base_url=url, api_key=api_key)
    telemetry.note(
        checks_failed=sum(not result.ok for result in results),
        problems=joined(result.problem for result in results),
    )
    if not render_report(results):
        raise SystemExit(1)


@app.command
def gui() -> None:
    """
    Launch the desktop GUI (requires the optional ``[gui]`` extra).

    The GUI is a review-before-write frontend over the same pipeline as the ``tag`` command: add
    photos or folders to a checkable list, generate proposals with the model, review and edit each
    photo's title, description, and keywords side by side with the existing metadata, then save.
    Install it with ``pip install 'photo-tagger[gui]'``.

    PySide6 is imported lazily here so the base CLI never depends on Qt; a missing dependency is
    reported with an install hint rather than a traceback.
    """
    try:
        gui_module = importlib.import_module("photo_tagger.gui")
    except ImportError as exc:
        # Only a genuinely missing Qt module earns the install hint. Anything else (a broken
        # shiboken build, an import error inside our own gui code) must surface as itself, or the
        # hint sends the user reinstalling an extra that is not the problem.
        # Without the parentheses the "or" takes the result of the split, and the module name
        # arrives whole. SonarQube reads them as redundant (S1110) and is wrong.
        missing = (exc.name or "").split(".")[0]  # NOSONAR S1110
        if missing not in ("PySide6", "shiboken6"):
            raise
        sys.stderr.write(
            "The desktop GUI needs PySide6, which is not installed.\n"
            "Install the optional extra with:\n"
            "    pip install 'photo-tagger[gui]'\n",
        )
        raise SystemExit(1) from exc
    raise SystemExit(gui_module.launch())


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


@app.command
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
    _apply_exiftool_path(configured_exiftool_path())
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


@app.command(name="set")
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
    copyright_config: Annotated[CopyrightConfig, Parameter(name="*")] = _DEFAULT_COPYRIGHT,
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
    _apply_exiftool_path(configured_exiftool_path())
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


def _census_source(inputs: list[Path] | None, from_export: Path | None) -> str:
    """Name where a vocabulary census came from, for telemetry: photos, an export, or both."""
    if inputs and from_export:
        return "both"
    return "export" if from_export else "photos"


def _build_census(
    inputs: list[Path] | None,
    from_export: Path | None,
    *,
    image_extensions: str,
    recursive: bool,
) -> tuple[KeywordCensus, str]:
    """Count keywords from the photos, from an export, or from both; also name the source."""
    census = KeywordCensus()
    sources: list[str] = []
    if from_export is not None:
        # utf-8-sig: a keyword export saved by a Windows editor starts with a BOM, which would
        # otherwise be counted as part of the first keyword.
        census.merge(census_from_export(from_export.read_text(encoding="utf-8-sig")))
        sources.append(f"keyword export {from_export.name} (counts are tree occurrences)")
    if inputs:
        image_files = resolve_image_batch(inputs, image_extensions, recursive=recursive)
        photo_census = census_from_photos(image_files)
        census.merge(photo_census)
        sources.append(f"{photo_census.photos} photo(s)")
    return census, " and ".join(sources)


@app.command
def vocabulary(  # noqa: PLR0913 - every parameter is a separate concern of the command.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            help="Photos or folders to read existing keywords from (repeat this option)",
        ),
    ] = None,
    *,
    output: Annotated[
        Path,
        Parameter(
            name=("--output", "-o"),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help="Where to write the vocabulary file",
        ),
    ],
    from_export: Annotated[
        Path | None,
        Parameter(
            name=("--from-export",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help=(
                "Read a Lightroom keyword export (.txt or .csv) instead of, or as well as, the "
                "photos. Counts from an export are tree occurrences, not photos"
            ),
        ),
    ] = None,
    image_extensions: Annotated[
        str,
        Parameter(name=("--ext", "--extensions"), help="Extensions to scan for (case insensitive)"),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(name=("--recursive", "-r"), help="Scan subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    build: Annotated[VocabularyBuildConfig, Parameter(name="*")] = VocabularyBuildConfig(),  # noqa: B008
    provider: Annotated[ProviderConfig, Parameter(name="*")] = _DEFAULT_PROVIDER,
) -> None:
    """
    Build a controlled vocabulary from the keywords a library already uses.

    ``--vocabulary-strict`` is what stops a catalog sprawling, and it needs a keyword file worth
    enforcing. This writes one: point it at your photos and it reads the keywords they already carry
    (through exiftool, so any application that writes XMP or IPTC works, not only Lightroom), counts
    how often each is used, and keeps the ones that earn their place.

    ``--from-export`` reads a Lightroom keyword export instead, for a catalog that is not on this
    machine. Its counts are occurrences in the keyword tree rather than photos, which is a weaker
    signal; prefer the photos when you have them.

    ``--organize`` adds a model pass over the keywords that survived, for the two things counting
    cannot settle: which of them are synonyms of each other, and what hierarchy they should have. It
    never decides what to keep and never invents a keyword.

    Nothing is written to your photos or your catalog. The output is a file to review and edit, plus
    an optional ``--report`` naming every keyword that was dropped and why.
    """
    _apply_exiftool_path(configured_exiftool_path())
    if not inputs and from_export is None:
        logger.error("vocabulary_no_source")
        raise SystemExit(1)

    console = Console()
    rules = to_trim_rules(build)
    try:
        census, source = _build_census(
            inputs,
            from_export,
            image_extensions=image_extensions,
            recursive=recursive,
        )
    except (DiscoveryError, OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError, not an OSError: a keyword export saved as UTF-16
        # (or a binary file chosen by mistake) is a bad input, not a crash. --vocabulary has
        # always said so through VocabularyError; this says it the same way.
        logger.error("vocabulary_source_unreadable", error=str(exc))
        console.print(f"[red]Could not read the keyword source: {exc}[/red]")
        raise SystemExit(1) from exc

    if not census.uses:
        console.print("[yellow]No keywords found: nothing to build a vocabulary from.[/yellow]")
        raise SystemExit(1)

    result = trim(census, rules)
    telemetry.note(
        source=_census_source(inputs, from_export),
        keywords_read=len(census.uses),
        keywords_kept=len(result.kept),
        keywords_dropped=len(result.dropped),
    )
    stats: OrganizeStats | None = None
    if build.organize:
        console.print(f"Organizing {len(result.kept)} keyword(s) with {provider.model_name}...")
        try:
            result, stats = organize(
                result,
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                api_base_url=provider.api_base_url,
                api_key=provider.api_key,
                workers=build.organize_workers,
            )
        except PhotoTaggerError as exc:
            # The trimmed list is already worth writing, but silently downgrading to it would hide
            # that the organize pass the user asked for never ran.
            logger.error("vocabulary_organize_failed", error=str(exc))
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1) from exc

    header = vocabulary_header(source, len(result.kept), len(result.dropped), rules, stats)
    try:
        output.write_text(
            render_vocabulary(result, header=header, flat=build.flat),
            encoding="utf-8",
        )
        if build.report_file is not None:
            build.report_file.write_text(render_drop_report(result), encoding="utf-8")
    except OSError as exc:
        logger.error("vocabulary_write_failed", error=str(exc))
        raise SystemExit(1) from exc

    console.print(f"Read {len(census.uses)} keyword(s) from {source}.")
    console.print(f"[green]Wrote {len(result.kept)} keyword(s) to {output}[/green]")
    if build.report_file is not None:
        console.print(f"Dropped {len(result.dropped)}; see {build.report_file}")
    console.print(
        "\nReview the file, then tag with it:\n"
        f"  photo-tagger -i PHOTOS --vocabulary {output} --vocabulary-strict",
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


@app.command
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


def _read_prompt_file(prompt_file: Path | None) -> str:
    """Return the contents of *prompt_file* (stripped) or DEFAULT_USER_PROMPT when None."""
    if prompt_file is None:
        return DEFAULT_USER_PROMPT
    try:
        text = prompt_file.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        logger.error("prompt_file_read_failed", file=str(prompt_file), error=str(exc))
        raise SystemExit(1) from exc
    if not text:
        logger.error("prompt_file_empty", file=str(prompt_file))
        raise SystemExit(1)
    logger.info("prompt_file_loaded", file=str(prompt_file), chars=len(text))
    return text


def _parse_filter_date(value: str | None, *, flag: str) -> datetime | None:
    """
    Parse an ISO 8601 string from *flag* into a timezone-aware datetime, or None.

    A naive timestamp like ``2024-01-01`` is read as **local** time (matching ``git log --since``
    and ``find -newer`` conventions). The system local zone is attached, and the value is returned
    aware so downstream comparisons with UTC mtimes do the right thing.
    """
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        logger.error("date_filter_parse_failed", flag=flag, value=value, error=str(exc))
        raise SystemExit(1) from exc
    if parsed.tzinfo is not None:
        return parsed
    # datetime.astimezone() on a naive value attaches the system local zone
    # (per the stdlib docs), turning it into a tz-aware value without shifting
    # the wall-clock fields. That is exactly the "user meant their local time"
    # semantics we want here.
    return parsed.astimezone()


def _maybe_str(value: Path | None) -> str | None:
    """Return ``str(value)`` or ``None`` so loguru extras keep a clean type."""
    return str(value) if value is not None else None


def _start_run(log: LogConfig) -> None:
    """Set up the log sinks, tracing, and the ExifTool path a tag or watch run needs."""
    try:
        setup_logging(
            file_log_level=log.file_log_level,
            console_log_level=log.console_log_level,
            log_folder=log.log_folder,
        )
    except OSError as exc:
        # No sink is guaranteed to be active yet (setup_logging removes the default one before it
        # can fail), so this cannot rely on the logger; write directly to stderr like the gui
        # command's own pre-logging error path does.
        sys.stderr.write(f"Could not set up logging at {log.log_folder}: {exc}\n")
        raise SystemExit(1) from exc
    _enable_tracing(log)
    _apply_exiftool_path(configured_exiftool_path())


def _require_inputs(inputs: list[Path] | None) -> list[Path]:
    """Return *inputs*, or stop with the hint to pass some when there are none."""
    if not inputs:
        logger.error("no_inputs_provided", hint="Pass one or more --input/-i paths to watch")
        raise SystemExit(1)
    return inputs


@app.default
def tag(  # noqa: PLR0913 - cyclopts entry point; each arg is a CLI flag group.
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
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(
            name=("--recursive", "-r"),
            help="Process files in subdirectories recursively",
        ),
    ] = DEFAULT_RECURSIVE,
    workers: Annotated[
        int,
        Parameter(
            name=("--workers", "-w"),
            help=(
                "Number of photos to process concurrently. Defaults to 1 (serial). The model "
                "server is the bottleneck; raising this past what your provider can serve in "
                "parallel will not help"
            ),
        ),
    ] = DEFAULT_WORKERS,
    filter_: Annotated[FilterConfig, Parameter(name="*")] = _DEFAULT_FILTER,
    display: Annotated[DisplayConfig, Parameter(name="*")] = _DEFAULT_DISPLAY,
    artifacts: Annotated[ArtifactConfig, Parameter(name="*")] = _DEFAULT_ARTIFACTS,
    provider: Annotated[ProviderConfig, Parameter(name="*")] = _DEFAULT_PROVIDER,
    output: Annotated[OutputConfig, Parameter(name="*")] = _DEFAULT_OUTPUT,
    inference: Annotated[InferenceConfig, Parameter(name="*")] = _DEFAULT_INFERENCE,
    log: Annotated[LogConfig, Parameter(name="*")] = _DEFAULT_LOG,
    telemetry_config: Annotated[TelemetryConfig, Parameter(name="*")] = _DEFAULT_TELEMETRY,
    pacing: Annotated[
        PacingConfig,
        Parameter(name="*", validator=validate_pacing),
    ] = _DEFAULT_PACING,
) -> None:
    """
    Tag images with AI and write Lightroom-compatible metadata (sidecar or embedded).

    Requirements:
    - ExifTool installed and on PATH.
    - Vision-language model API reachable (e.g., Ollama server).

    Inputs:
    - One or more --input/-i paths (files and/or directories; repeatable).
    - Files are processed as is. Directories use --ext (add --recursive for subfolders).
    - You can mix files and directories; order is preserved, duplicates skipped.

    Behavior:
    - Loads image (RAW supported), converts to in-memory JPEG, queries the model.
    - Generates title, description, and keywords; merges with existing XMP by default
        (use --overwrite-keywords to replace).
    - Writes metadata to an XMP sidecar (default), embeds it directly with --embed-in-photo,
        splits the two per file type with --sidecar-mode raw (sidecars for RAW, embedded for JPEG
        and friends), or does both with --sidecar-mode both. --sidecar-override dng=photo sends
        one file type elsewhere (into each DNG here, as Lightroom Classic does). Use
        --no-write-title/--no-write-description to skip fields; --no-backup-xmp to avoid backups.

    Skipping:
    - --skip-from FILE: skip files listed in FILE (one name or path per line).
    - --append-to-skip-file FILE: append each successfully processed file's path to FILE so a
        later run with --skip-from FILE resumes where this one stopped.
    - --skip-tagged: skip files that already have keywords, a description, or a title in
        the image or its XMP sidecar.

    Dry runs:
    - --dry-run: query the model and log the generated title, description, and keywords
        for each image but do not write any metadata. Useful for prompt iteration.

    Performance:
    - --workers N: process N photos concurrently using a thread pool. Each worker opens
        its own ExifToolHelper. The model server is the dominant bottleneck.
    - --max-keywords N: cap the number of AI-generated keywords kept before merging.

    Pacing (for long runs on a laptop):
    - --duty N: work only N percent of the time, resting in between. Keeps the fans down,
        leaves the machine usable, and holds the average power draw below what the charger
        supplies. A run rests at a photo boundary, never mid-photo.
    - --battery-floor N: rest until the battery charges back up whenever it falls to N
        percent, so a long run on a laptop does not flatten it. Only while plugged in.
    - --work-minutes / --max-pause-minutes: how long a work window runs, and the longest a
        single rest may last.

    Customization:
    - --prompt-file PATH: replace the default user prompt with the file's contents.
        Existing photo metadata (keywords, GPS, location) is still appended automatically.

    Exit status: returns 1 if no inputs are given, discovery finds no matching images, or any
        file fails. Returns 0 if discovery finds images but every one of them is subsequently
        excluded by --skip-from, --skip-tagged, or the date filters: that is "nothing left to
        do", not an error, and scripted incremental reruns rely on it not failing the batch.

    Examples:
        photo-tagger -i ./photos/IMG_0001.CR3

        photo-tagger \
            --extensions cr3,jpg \
            --provider lmstudio \
            --url http://localhost:1234/v1 \
            --recursive \
            --skip-from processed.txt \
            --append-to-skip-file processed.txt \
            Pictures/Camera

        photo-tagger -i Pictures/Mixed --skip-tagged
    """
    run = RunOptions(
        provider=provider,
        inference=inference,
        output=output,
        artifacts=artifacts,
        display=display,
        filter_=filter_,
        log=log,
        pacing=pacing,
        telemetry_enabled=telemetry_config.enabled,
        workers=workers,
        extensions=image_extensions,
        recursive=recursive,
        inputs=inputs,
    )
    beacons = Beacons.for_invocation("tag", run, _CONFIG_SOURCE.invocation)
    try:
        _start_run(log)
        with _optional_lock(artifacts.lock_file):
            try:
                _tag_inside_lock(run, beacons)
            except PhotoTaggerError as exc:
                raise SystemExit(1) from exc
    except (SystemExit, KeyboardInterrupt) as exc:
        # A run that stopped before its first photo (no model, no photos, a held lock) sends no
        # batch beacon of its own, so it is reported here.
        beacons.stopped(exc)
        raise


@contextlib.contextmanager
def _optional_lock(lock_file: Path | None) -> Generator[None]:
    """
    Hold *lock_file* for the run, when one was given, or stop with exit code 1.

    A lock another run holds, or one that cannot be opened (a folder that is not writable, a full
    disk), ends the run with a log line instead of a traceback in the user's terminal.
    """
    with contextlib.ExitStack() as stack:
        if lock_file is not None:
            try:
                stack.enter_context(FileLock(lock_file))
            except LockHeldError as exc:
                logger.error("lock_held_by_other_process", file=str(lock_file), error=str(exc))
                raise SystemExit(1) from exc
            except OSError as exc:
                logger.error("lock_file_open_failed", file=str(lock_file), error=str(exc))
                raise SystemExit(1) from exc
        yield


@dataclass(slots=True)
class _RunSetup:
    """
    The collaborators a tagging run builds once and reuses for every batch it processes.

    ``tag`` processes exactly one batch, ``watch`` processes a new one whenever photos land, and
    both want the same agent, cache, and report writers for the whole invocation: reopening them per
    batch would repeat the CSV header and re-prune the cache.

    The undo journal is the exception. A watch can run for days, and one journal for all of it would
    make "undo the import" mean every photo since Tuesday; each batch is its own run, so
    :meth:`start_journal` opens one per batch. ``tag`` has a single batch and never calls it.
    """

    run: RunOptions
    options: ProcessingOptions
    agent: Agent[None, GeneratedMetadata]
    user_prompt: str
    workers: int
    beacons: Beacons
    # Resolved from the two --notify flags, not the group itself: --notify-sound implies --notify,
    # and the batch runner should not have to know that.
    notify_enabled: bool
    notify_sound: bool
    cache: InferenceCache | None = None
    csv_writer: CsvReportWriter | None = None
    # None unless the run was asked to slow down. Shared by every batch a watch processes, so the
    # work window carries across batches instead of restarting with each one.
    pacer: Pacer | None = None
    on_image_result: Callable[[ImageOutcome], None] | None = None
    on_success: Callable[[Path], None] | None = None
    journal: UndoJournal | None = None
    # Whether writes are recorded at all, so a rotated journal knows to stay switched off.
    journal_enabled: bool = True

    def start_journal(self) -> None:
        """Open a journal for the batch about to run, reporting what the previous one recorded."""
        self._report_journal()
        self.journal = open_journal(datetime.now(tz=UTC), enabled=self.journal_enabled)

    def _report_journal(self) -> None:
        """Log the journal's path once it holds something, so the user can name it to undo."""
        if self.journal is not None and self.journal.entries:
            logger.info(
                "undo_journal_written",
                file=str(self.journal.path),
                entries=self.journal.entries,
            )

    def close(self) -> None:
        """Release everything the run opened, in the order it was opened."""
        if self.cache is not None:
            self.cache.close()
        if self.csv_writer is not None:
            self.csv_writer.close()
        self._report_journal()


def _build_run_setup(run: RunOptions, beacons: Beacons) -> _RunSetup:
    """Build the agent, cache, report sinks, pacer, and undo journal shared by every batch."""
    provider, inference, output, artifacts = run.provider, run.inference, run.output, run.artifacts
    # Raises VocabularyError (a PhotoTaggerError) on an unusable file, which the commands turn into
    # a clean exit 1. Loading it up front means a typo in the path fails before any model call.
    vocabulary = (
        load_vocabulary(output.vocabulary, output_language=inference.output_language)
        if output.vocabulary is not None
        else None
    )
    if vocabulary is not None:
        beacons.settings = replace(beacons.settings, vocabulary_terms=len(vocabulary.terms))
    # A --hint and the --vocabulary listing ride inside the user prompt, so the cache namespace
    # below picks them up too: a hinted run never replays results generated without the hint (and
    # vice versa), and the same holds for a vocabulary.
    user_prompt = prompt_with_vocabulary(
        prompt_with_hint(_read_prompt_file(artifacts.prompt_file), inference.hint),
        vocabulary,
    )
    # Fold the prompt + language + sampling/JPEG settings into the cache namespace so a
    # different configuration writes to a fresh slice instead of replaying
    # stale entries generated under earlier settings.
    cache_namespace = build_cache_namespace(
        provider.model_name,
        user_prompt=user_prompt,
        temperature=inference.temperature,
        frequency_penalty=inference.frequency_penalty,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
        output_language=inference.output_language,
    )
    csv_writer = open_csv_report(artifacts.csv_file)
    ndjson_emitter = NDJSONEmitter(sys.stdout) if run.display.json_output else None
    csv_sink = CsvImageResultSink(csv_writer) if csv_writer is not None else None
    # A dry run writes nothing, so there is nothing for undo to put back.
    journal_enabled = artifacts.undo_log and not output.dry_run
    # Anything below can raise before a _RunSetup exists, and then the caller has nothing to
    # close: the report file and the cache's SQLite handle would stay open until the interpreter
    # got round to them. create_agent validates the model over HTTP, which is the failure that
    # actually happens here; it runs first, so a wrong model name never opens the cache at all.
    cache: InferenceCache | None = None
    notify_enabled, notify_sound = resolve_notify(run.display)
    try:
        agent = create_agent(
            provider.provider_name,
            provider.model_name,
            api_base_url=provider.api_base_url,
            api_key=provider.api_key,
            retries=provider.retries,
            output_language=inference.output_language,
        )
        cache = open_cache(artifacts.cache_file, namespace=cache_namespace)
        return _RunSetup(
            run=run,
            options=to_processing_options(
                output,
                inference,
                vocabulary=vocabulary,
                retries=provider.retries,
            ),
            agent=agent,
            user_prompt=user_prompt,
            workers=max(1, run.workers),
            beacons=beacons,
            notify_enabled=notify_enabled,
            notify_sound=notify_sound,
            cache=cache,
            csv_writer=csv_writer,
            pacer=build_pacer(to_pacing_plan(run.pacing)),
            on_image_result=combine_image_result_callbacks(ndjson_emitter, csv_sink),
            on_success=make_skip_list_appender(artifacts.append_to_skip_file),
            journal=open_journal(datetime.now(tz=UTC), enabled=journal_enabled),
            journal_enabled=journal_enabled,
        )
    except Exception:
        if cache is not None:
            cache.close()
        if csv_writer is not None:
            csv_writer.close()
        raise


def _log_startup(setup: _RunSetup) -> None:
    """Single-shot info log so the run's full configuration is captured up-front."""
    run = setup.run
    provider, inference, artifacts = run.provider, run.inference, run.artifacts
    options = setup.options
    logger.info(
        "starting_photo_tagger",
        inputs=[str(p) for p in (run.inputs or [])],
        extensions=run.extensions,
        model=provider.model_name,
        provider=provider.provider_name,
        api_base_url=provider.api_base_url,
        api_key_present=bool(provider.api_key),
        recursive=run.recursive,
        workers=setup.workers,
        prompt_file=_maybe_str(artifacts.prompt_file),
        summary_file=_maybe_str(artifacts.summary_file),
        csv_file=_maybe_str(artifacts.csv_file),
        cache_file=_maybe_str(artifacts.cache_file),
        lock_file=_maybe_str(artifacts.lock_file),
        json_output=run.display.json_output,
        progress_bar=run.display.progress_bar,
        notify=setup.notify_enabled,
        notify_sound=setup.notify_sound,
        telemetry=setup.beacons.enabled,
        skip_from=_maybe_str(artifacts.skip_from),
        append_to_skip_file=_maybe_str(artifacts.append_to_skip_file),
        skip_tagged=run.filter_.skip_tagged,
        newer_than=run.filter_.newer_than,
        older_than=run.filter_.older_than,
        preserve_keywords=options.preserve_existing_kw,
        preserve_title=options.preserve_existing_title,
        preserve_description=options.preserve_existing_description,
        write_description=options.write_description,
        write_title=options.write_title,
        write_keywords=options.write_keywords,
        backup_xmp=options.backup_xmp,
        sidecar_mode=options.sidecar_mode,
        dry_run=options.dry_run,
        max_keywords=options.max_new_keywords,
        vocabulary_terms=len(options.vocabulary.terms) if options.vocabulary else 0,
        vocabulary_strict=options.vocabulary_strict,
        extra_fields=not (options.extra_fields is None or options.extra_fields.is_empty),
        session_gap_minutes=run.output.session_gap_minutes,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        timeout_seconds=options.timeout_seconds,
        frequency_penalty=options.frequency_penalty,
        jpeg_dimensions=options.jpeg_dimensions,
        jpeg_quality=options.jpeg_quality,
        output_language=inference.output_language,
        hint=inference.hint,
        retries=provider.retries,
        log_folder=str(run.log.log_folder),
    )


def _process_batch(image_files: list[Path], setup: _RunSetup) -> None:
    """
    Tag one resolved batch, writing the summary file and firing the telemetry beacon.

    Raises :class:`~photo_tagger.errors.BatchError` when any photo fails, which ``tag`` turns into
    exit 1 and ``watch`` logs before waiting for the next batch.
    """
    started_at = datetime.now(tz=UTC)
    stats = PhotoStats()
    plan = plan_sessions(image_files, gap_minutes=setup.run.output.session_gap_minutes)

    def _on_complete(totals: BatchTotals) -> None:
        # Runs once before run_batch raises, so the summary file is written, the desktop alert
        # raised, and the telemetry beacon fired whether the batch succeeded fully or only
        # partially. A run that half failed is exactly the one worth being called back for.
        notify(
            APP_NAME,
            alert_message(totals),
            enabled=setup.notify_enabled,
            sound=setup.notify_sound,
            # `tag` exits right after this, and `watch` goes back to sleep; either way nothing
            # else is waiting on the few seconds the notifier takes.
            block=True,
        )
        write_summary_file(
            setup.run.artifacts.summary_file,
            totals,
            started_at=started_at,
            model_name=setup.run.provider.model_name,
            provider_name=setup.run.provider.provider_name,
            user_prompt_chars=len(setup.user_prompt),
        )
        setup.beacons.record(
            run_from_totals(
                totals,
                image_files,
                beacons=setup.beacons,
                started_at=started_at,
                stats=stats,
                plan=plan,
            ),
        )

    with batch_progress(len(image_files), enabled=setup.run.display.progress_bar) as progress:
        run_batch(
            image_files,
            setup.agent,
            setup.options,
            on_success=setup.on_success,
            on_complete=_on_complete,
            user_prompt=setup.user_prompt,
            workers=setup.workers,
            progress=progress,
            cache=setup.cache,
            on_image_result=combine_image_result_callbacks(setup.on_image_result, stats),
            session_plan=plan,
            journal=setup.journal,
            pacer=setup.pacer,
        )


def _filter_batch(
    image_files: list[Path],
    run: RunOptions,
    *,
    newer_than: datetime | None,
    older_than: datetime | None,
) -> list[Path]:
    """Apply the skip list, the date window, and --skip-tagged to a resolved batch."""
    image_files = apply_skip_file(image_files, run.artifacts.skip_from)
    image_files = apply_date_filter(image_files, newer_than=newer_than, older_than=older_than)
    return apply_skip_tagged(image_files, skip_tagged=run.filter_.skip_tagged)


def _tag_inside_lock(run: RunOptions, beacons: Beacons) -> None:
    """Body of ``tag`` that runs once the optional file lock has been acquired."""
    show_notice_if_due(enabled=beacons.enabled)
    newer_than = _parse_filter_date(run.filter_.newer_than, flag="--newer-than")
    older_than = _parse_filter_date(run.filter_.older_than, flag="--older-than")
    # Discovery first: building the run truncates the CSV report, opens the cache and validates the
    # model against the server. A batch that turns out to be empty should cost none of that, and
    # least of all last run's report.
    image_files = _filter_batch(
        resolve_image_batch(run.inputs, run.extensions, recursive=run.recursive),
        run,
        newer_than=newer_than,
        older_than=older_than,
    )
    if not image_files:
        logger.info("no_files_to_process_after_skipping")
        beacons.emit(replace(beacons.base_run(), outcome=telemetry.OUTCOME_EMPTY))
        return
    setup = _build_run_setup(run, beacons)
    _log_startup(setup)

    try:
        _process_batch(image_files, setup)
    finally:
        setup.close()


@app.command
def watch(  # noqa: PLR0913 - cyclopts entry point; each arg is a CLI flag group.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            validator=validators.Path(exists=True),
            help="One or more folders (or files) to watch (repeat this option)",
        ),
    ] = None,
    *,
    image_extensions: Annotated[
        str,
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(name=("--recursive", "-r"), help="Watch subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    watch_config: Annotated[WatchConfig, Parameter(name="*")] = _DEFAULT_WATCH,
    filter_: Annotated[FilterConfig, Parameter(name="*")] = _DEFAULT_FILTER,
    display: Annotated[DisplayConfig, Parameter(name="*")] = _DEFAULT_DISPLAY,
    artifacts: Annotated[ArtifactConfig, Parameter(name="*")] = _DEFAULT_ARTIFACTS,
    provider: Annotated[ProviderConfig, Parameter(name="*")] = _DEFAULT_PROVIDER,
    output: Annotated[OutputConfig, Parameter(name="*")] = _DEFAULT_OUTPUT,
    inference: Annotated[InferenceConfig, Parameter(name="*")] = _DEFAULT_INFERENCE,
    log: Annotated[LogConfig, Parameter(name="*")] = _DEFAULT_LOG,
    telemetry_config: Annotated[TelemetryConfig, Parameter(name="*")] = _DEFAULT_TELEMETRY,
    pacing: Annotated[
        PacingConfig,
        Parameter(name="*", validator=validate_pacing),
    ] = _DEFAULT_PACING,
) -> None:
    """
    Watch folders and tag photos as they arrive, until you stop it with Ctrl-C.

    This is the import-time workflow: point it at the folder your card reader, tethered capture, or
    sync client fills, and leave it running. Photos already in the folder are tagged first, then
    each new one as it lands.

    A file is only tagged once it has stopped changing (see ``--settle``), so a photo still being
    copied is left alone until the copy finishes. Scanning is a plain directory listing every
    ``--interval`` seconds, which behaves the same on every platform and over network shares.

    Every flag ``photo-tagger`` itself takes works here too and applies to each batch, with one
    agent, cache, and report file shared by the whole session. Each batch records its own undo
    journal, so ``photo-tagger undo`` puts back the last import rather than every import since the
    watch started. A batch where some photo fails is logged and the watch continues; the failure
    does not stop the loop.

    Examples:
        photo-tagger watch -i ~/Pictures/Inbox

        photo-tagger watch -i ~/Pictures/Inbox -r --interval 10 --skip-tagged
    """
    run = RunOptions(
        provider=provider,
        inference=inference,
        output=output,
        artifacts=artifacts,
        display=display,
        filter_=filter_,
        log=log,
        pacing=pacing,
        telemetry_enabled=telemetry_config.enabled,
        workers=watch_config.workers,
        extensions=image_extensions,
        recursive=recursive,
        inputs=inputs,
        watch=watch_config,
    )
    beacons = Beacons.for_invocation("watch", run, _CONFIG_SOURCE.invocation)
    try:
        _start_run(log)
        folders = _require_inputs(inputs)
        with _optional_lock(artifacts.lock_file):
            try:
                _watch_forever(replace(run, inputs=folders), watch_config, beacons)
            except PhotoTaggerError as exc:
                raise SystemExit(1) from exc
    except (SystemExit, KeyboardInterrupt) as exc:
        beacons.stopped(exc)
        raise


def _watch_forever(run: RunOptions, watch_config: WatchConfig, beacons: Beacons) -> None:
    """Poll *run*'s inputs for new photos and tag each settled batch until interrupted."""
    show_notice_if_due(enabled=beacons.enabled)
    newer_than = _parse_filter_date(run.filter_.newer_than, flag="--newer-than")
    older_than = _parse_filter_date(run.filter_.older_than, flag="--older-than")
    setup = _build_run_setup(run, beacons)
    inputs = run.inputs or []
    logger.info(
        "watching_for_new_photos",
        inputs=[str(path) for path in inputs],
        extensions=run.extensions,
        recursive=run.recursive,
        interval=watch_config.interval,
        settle=watch_config.settle,
    )
    batches = 0
    tagged = 0
    try:
        for batch in watch_batches(
            inputs,
            run.extensions,
            recursive=run.recursive,
            interval_seconds=watch_config.interval,
            settle_seconds=watch_config.settle,
        ):
            image_files = _filter_batch(batch, run, newer_than=newer_than, older_than=older_than)
            if not image_files:
                continue
            batches += 1
            tagged += len(image_files)
            # Each batch is its own run to undo: a watch left running all week must not fold a
            # month of imports into one journal.
            setup.start_journal()
            try:
                _process_batch(image_files, setup)
            except BatchError as exc:
                # run_batch turns a Ctrl-C into an interrupted batch, and that must end the watch.
                if _was_interrupted(exc):
                    raise KeyboardInterrupt from exc
                # One bad batch must not end the watch: the next photo to land deserves its turn.
                logger.error("watch_batch_had_failures", error=str(exc))
    except KeyboardInterrupt:
        logger.info("watch_stopped_by_user", batches=batches, photos=tagged)
    finally:
        setup.close()
        beacons.emit(beacons.session())


def _was_interrupted(error: BatchError) -> bool:
    """Say whether *error* reports a batch that a Ctrl-C cut short."""
    totals = error.args[0] if error.args else None
    return isinstance(totals, BatchTotals) and totals.interrupted


def _telemetry_enabled_for(tokens: list[str]) -> bool:
    """
    Best-effort telemetry choice for an invocation without a parsed ``--no-telemetry``.

    A crash may happen before (or during) CLI parsing, and the other commands have no such flag, so
    scan the raw *tokens* for it and read the config file's [telemetry] table. The environment
    switches are enforced inside emit and emit_crash themselves.
    """
    if "--no-telemetry" in tokens:
        return False
    table = load_config().get("telemetry", {})
    return bool(table.get("enabled", True)) if isinstance(table, dict) else True


# Exit statuses the command beacon names: cyclopts' own for a bad flag, and the shell's for Ctrl-C.
_USAGE_EXIT = 2
_INTERRUPTED_EXIT = 130


def _command_name(tokens: list[str]) -> str:
    """Name what *tokens* run: a subcommand, help, the version, or the default ``tag``."""
    if {"--help", "-h"}.intersection(tokens):
        return "help"
    if "--version" in tokens:
        return "version"
    commands = {name for name in app if not name.startswith("-")}
    return tokens[0] if tokens and tokens[0] in commands else "tag"


def _command_outcome(exit_code: int, error: BaseException | None) -> str:
    """Name how a command ended, from its exit status."""
    if isinstance(error, KeyboardInterrupt):
        return telemetry.OUTCOME_INTERRUPTED
    if exit_code == 0:
        return telemetry.OUTCOME_OK
    return telemetry.OUTCOME_USAGE_ERROR if exit_code == _USAGE_EXIT else telemetry.OUTCOME_FAILED


def _report_command(
    tokens: list[str],
    *,
    exit_code: int,
    error: BaseException | None,
    started: float,
) -> None:
    """
    Send the beacon for an invocation that sent none of its own.

    That is every command but ``tag``, ``watch``, and ``gui``, plus a ``tag`` or ``watch`` that
    stopped at the command line. The GUI is left out: its window reports itself, under its own
    toggle. Never raises, because it runs while the real exit is on its way out.
    """
    command = _command_name(tokens)
    if telemetry.sent_any() or command == "gui":
        return
    try:
        enabled = _telemetry_enabled_for(tokens)
        if not telemetry.should_send(config_enabled=enabled):
            return
        show_notice_if_due(enabled=enabled)
        interrupted = isinstance(error, KeyboardInterrupt)
        telemetry.emit(
            telemetry.CommandInfo(
                command=command,
                outcome=_command_outcome(exit_code, error),
                exit_code=exit_code,
                duration_seconds=time.monotonic() - started,
                error_kind=type(error).__name__ if error is not None and not interrupted else "",
                setup=invocation_setup(_CONFIG_SOURCE.invocation),
                details=telemetry.command_notes(),
            ),
            enabled=enabled,
            block=True,
        )
    except Exception as exc:  # noqa: BLE001 - a beacon must never change how the CLI exits.
        logger.debug("telemetry_command_report_failed", error=str(exc))


def main(argv: list[str] | None = None) -> None:
    """
    Console entry point: run the CLI, reporting an unhandled crash before re-raising.

    Expected exits (SystemExit from clean error handling, Ctrl-C) pass through untouched; anything
    else is a genuine crash, so an anonymous beacon (exception type and stack trace as code
    locations, never the message) is sent before the traceback surfaces as usual. An invocation that
    sent no beacon of its own gets a short command beacon on the way out.
    """
    tokens = sys.argv[1:] if argv is None else argv
    started = time.monotonic()
    exit_code = 0
    error: BaseException | None = None
    try:
        app(tokens)
    except SystemExit as exc:
        exit_code = exit_status(exc)
        error = exc.__cause__ or exc.__context__
        raise
    except KeyboardInterrupt as exc:
        exit_code, error = _INTERRUPTED_EXIT, exc
        raise
    except Exception as exc:
        if _command_name(tokens) == "gui":
            # The window reports its own crashes, under its own consent and notice.
            raise
        enabled = _telemetry_enabled_for(tokens)
        # A crash on the very first run is still the first beacon, so the disclosure comes first.
        show_notice_if_due(enabled=enabled)
        telemetry.emit_crash(
            exc,
            interface="cli",
            command=_command_name(tokens),
            enabled=enabled,
            block=True,
        )
        raise
    finally:
        _report_command(tokens, exit_code=exit_code, error=error, started=started)


if __name__ == "__main__":  # pragma: no cover - module entry point, not exercised by tests
    main()
