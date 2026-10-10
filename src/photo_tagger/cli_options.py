"""
CLI option groups and their config-file-aware defaults.

Each dataclass below is a group of related ``photo-tagger`` flags. Cyclopts' ``Parameter(name="*")``
flattens the fields onto the top-level CLI, so users still pass ``--temperature 0.5``, ``--no-
backup-xmp``, etc. The grouping keeps the ``tag`` entry point's signature small enough to satisfy
Sonar's S107 parameter-count rule without burying the option metadata inside ``main``.

Splitting this out of ``main`` keeps the CLI *schema* (what flags exist, their help, their defaults)
separate from the orchestration logic that consumes it.

A TOML config file, if found, supplies values for flags the user does not pass on the command line.
It is fed through cyclopts' own config layer (:class:`ConfigFileSource`) rather than baked into the
default instances, because cyclopts builds a *fresh* group instance whenever any flag from that
group appears on the command line; baked-in defaults would silently drop the config values of every
sibling field in that group. The hook gives true per-field precedence: CLI flag > config file >
built-in default. It also means config values pass through the same conversion and validation as
flags, so a mistyped value fails with a clean CLI error instead of a traceback later.

``load_defaults`` still resolves the config into concrete instances for callers that never parse a
command line (the GUI).
"""

import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args

from cyclopts import App, ArgumentCollection, Parameter, validators
from cyclopts.argument import update_argument_collection
from cyclopts.config import Dict as _CycloptsDictConfig

from photo_tagger import telemetry
from photo_tagger.config import (
    DEFAULT_BATTERY_FLOOR,
    DEFAULT_BATTERY_RESUME,
    DEFAULT_DIMENSIONS,
    DEFAULT_DUTY_PERCENT,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_PAUSE_MINUTES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_RETRIES,
    DEFAULT_SIDECAR_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORK_MINUTES,
    FileTypeTarget,
    LogLevel,
    SidecarMode,
)
from photo_tagger.config_file import apply_overrides, find_config_file, load_config
from photo_tagger.fields import ExtraFields, build_extra_fields
from photo_tagger.metadata import parse_sidecar_overrides
from photo_tagger.pacing import MAX_BATTERY_FLOOR, SECONDS_PER_MINUTE, PacingPlan
from photo_tagger.pipeline import ProcessingOptions

# Runtime import (not type-only): cyclopts evaluates the Annotated[ProviderName, ...] field
# below to validate the --provider choices, so the name must exist at class-definition time.
from photo_tagger.providers import ProviderName, get_backend
from photo_tagger.telemetry_facts import endpoint_kind, extension_list
from photo_tagger.vocabulary import folds_plurals
from photo_tagger.vocabulary_build import TrimRules
from photo_tagger.watch import DEFAULT_INTERVAL_SECONDS, DEFAULT_SETTLE_SECONDS


if TYPE_CHECKING:
    from photo_tagger.vocabulary import Vocabulary


# Built-in defaults for the top-level (non-grouped) flags, shared by `main` and `load_defaults`.
# They sit above the option groups because `WatchConfig` uses `DEFAULT_WORKERS` as a field default.
DEFAULT_EXTENSIONS = "cr3,jpg"
DEFAULT_WORKERS = 1
DEFAULT_RECURSIVE = False


@dataclass
class ProviderConfig:
    """Backend provider, model id, and credential overrides."""

    model_name: Annotated[
        str,
        Parameter(name=("--model", "-m"), help="Vision-language model name"),
    ] = DEFAULT_MODEL_NAME
    provider_name: Annotated[
        ProviderName,
        Parameter(
            name=("--provider",),
            help=(
                "Backend provider: 'ollama', 'lmstudio', 'llamacpp', or 'openai' (any "
                "OpenAI-compatible API)"
            ),
        ),
    ] = "lmstudio"
    api_base_url: Annotated[
        str | None,
        Parameter(name=("--url", "-u"), help="Provider API base URL"),
    ] = None
    api_key: Annotated[
        str | None,
        Parameter(
            name=("--api-key", "-k"),
            help=(
                "Provider API key. Prefer env vars (OLLAMA_API_KEY, LM_STUDIO_API_KEY,"
                " LLAMA_CPP_API_KEY, OPENAI_API_KEY) over this flag. Note: CLI args are visible"
                " in process listings!"
            ),
        ),
    ] = None
    retries: Annotated[
        int,
        Parameter(name=("--retries",), help="Number of automatic validation retries"),
    ] = DEFAULT_RETRIES


@dataclass
class InferenceConfig:
    """Sampling, prompt (language, hint), and image-encoding knobs sent to the model."""

    output_language: Annotated[
        str,
        Parameter(
            name=("--output-language", "--lang"),
            help=(
                "Language for the generated title, description, and keywords (any language "
                "name the model understands, e.g. 'German' or 'Brazilian Portuguese')"
            ),
        ),
    ] = DEFAULT_OUTPUT_LANGUAGE
    repair_language: Annotated[
        bool,
        Parameter(
            name=("--repair-language",),
            negative="--no-repair-language",
            help=(
                "When the generated text mixes writing systems, spend one more model call asking "
                "for it in the output language (e.g. 'Ocarina' for a stray '陶笛'). Off by "
                "default; the photos are flagged either way"
            ),
        ),
    ] = False
    hint: Annotated[
        str | None,
        Parameter(
            name=("--hint",),
            help=(
                "A note about every photo in this run that the model must trust over its own "
                "reading of the image, e.g. 'The animal in these photos is a deer'. Useful when "
                "the model keeps misidentifying a subject"
            ),
        ),
    ] = None
    temperature: Annotated[
        float,
        Parameter(name=("--temperature",), help="Sampling temperature (0.0-1.0)"),
    ] = DEFAULT_TEMPERATURE
    max_tokens: Annotated[
        int,
        Parameter(name=("--max-tokens",), help="Maximum tokens to generate"),
    ] = DEFAULT_MAX_TOKENS
    timeout_seconds: Annotated[
        float,
        Parameter(
            name=("--timeout-seconds",),
            help="Timeout of one model request, in seconds; aborts it and lets the retries step in",
        ),
    ] = DEFAULT_TIMEOUT_SECONDS
    frequency_penalty: Annotated[
        float,
        Parameter(
            name=("--frequency-penalty",),
            help="Penalty on repeated tokens (0.0-2.0); discourages chant-style output loops",
        ),
    ] = DEFAULT_FREQUENCY_PENALTY
    jpeg_dimensions: Annotated[
        int,
        Parameter(
            name=("--jpeg-dimensions",),
            help=(
                "Longest edge in pixels of the JPEG sent to the model, and the biggest lever on "
                "speed: prefill scales with the pixel count. 640 (faster), 768 (more detail), "
                "896 (large text), 1024 (fine text). Changing it starts a fresh cache namespace"
            ),
        ),
    ] = DEFAULT_DIMENSIONS
    jpeg_quality: Annotated[
        int,
        Parameter(
            name=("--jpeg-quality",),
            help="JPEG quality (1-100) for the image sent to the model",
        ),
    ] = DEFAULT_JPEG_QUALITY


@dataclass
class CopyrightConfig:
    """
    Authorship and rights fields, plus the raw escape hatch for any other tag.

    Nothing here involves the model: these are the fields only the photographer knows. The ``set``
    command takes this group on its own, to stamp photos without calling a model; ``tag`` and
    ``watch`` inherit it through :class:`OutputConfig` and write it in the same ExifTool call as the
    generated metadata, so a run credits every photo it touches.
    """

    artist: Annotated[
        str | None,
        Parameter(
            name=("--artist", "--creator"),
            help=(
                "Who took the photo. Written to EXIF:Artist, XMP-dc:Creator, and IPTC:By-line, so "
                "every application that shows a credit finds one"
            ),
        ),
    ] = None
    copyright_notice: Annotated[
        str | None,
        Parameter(
            name=("--copyright",),
            help=(
                "The copyright notice, written to EXIF:Copyright, IPTC:CopyrightNotice, and "
                "XMP-dc:Rights. Use {year} for the year the photo was taken and {artist} for the "
                "name, e.g. 'Copyright (c) {year} {artist}, All rights reserved'. A placeholder "
                'can read any tag and fall back: {EXIF:Artist|XMP-dc:Creator|"Unknown"}'
            ),
        ),
    ] = None
    rights: Annotated[
        str | None,
        Parameter(
            name=("--rights",),
            help=(
                "XMP-dc:Rights, for when it should read differently from --copyright: XMP is UTF-8 "
                "and safe for '© {year}' where EXIF gets '(c) {year}'. Defaults to --copyright"
            ),
        ),
    ] = None
    marked: Annotated[
        bool | None,
        Parameter(
            name=("--marked",),
            negative="--not-marked",
            help=(
                "Set XMP-xmpRights:Marked, the flag that says whether the photo is under copyright "
                "(--marked) or in the public domain (--not-marked). Left alone unless asked for"
            ),
        ),
    ] = None
    preserve_copyright: Annotated[
        bool,
        Parameter(
            name=("--preserve-copyright",),
            help=(
                "Keep the name and the notice a photo already has. --no-preserve-copyright "
                "replaces them"
            ),
        ),
    ] = True
    set_tags: Annotated[
        list[str] | None,
        Parameter(
            name=("--set",),
            help=(
                "Write any other tag, as TAG=VALUE, using ExifTool's own tag names (repeat this "
                "option). For example --set 'XMP-xmpRights:WebStatement=https://example.com/terms'."
                " An empty value deletes the tag. Applied after everything else, so it wins"
            ),
        ),
    ] = None


def validate_sidecar_overrides(_type: type, entries: list[str] | None) -> None:
    """Refuse a malformed --sidecar-override at parse time, before the first photo is read."""
    parse_sidecar_overrides(entries)


@dataclass
class OutputConfig(CopyrightConfig):
    """
    How metadata is merged with existing tags and where it is written.

    Inherits :class:`CopyrightConfig` so ``tag`` and ``watch`` accept the authorship flags without a
    tenth option-group parameter, which would put both commands over the parameter count a single
    function can be read at. The same reason ``--otel`` lives in :class:`LogConfig`. It is the right
    group either way: a copyright notice is part of what a run writes.
    """

    preserve_keywords: Annotated[
        bool,
        Parameter(
            name=("--preserve-keywords",),
            negative="--overwrite-keywords",
            help="Preserve existing keywords in XMP files (merge) vs overwrite them",
        ),
    ] = True
    preserve_title: Annotated[
        bool,
        Parameter(
            name=("--preserve-title",),
            negative="--overwrite-title",
            help="Keep a title the photo already has (the default replaces it)",
        ),
    ] = False
    preserve_description: Annotated[
        bool,
        Parameter(
            name=("--preserve-description",),
            negative="--overwrite-description",
            help="Keep a description the photo already has (the default replaces it)",
        ),
    ] = False
    write_description: Annotated[
        bool,
        Parameter(
            name=("--write-description",),
            negative="--no-write-description",
            help="Also generate and write a short description (IFD0/XMP)",
        ),
    ] = True
    write_title: Annotated[
        bool,
        Parameter(
            name=("--write-title",),
            negative="--no-write-title",
            help="Also generate and write a title (XMP-dc:Title / IPTC:ObjectName)",
        ),
    ] = True
    write_keywords: Annotated[
        bool,
        Parameter(
            name=("--write-keywords",),
            negative="--no-write-keywords",
            help=(
                "Write keywords (merged with existing ones per --preserve-keywords). Pass "
                "--no-write-keywords to leave existing keywords untouched, e.g. to refresh only "
                "the title and description"
            ),
        ),
    ] = True
    write_copyright: Annotated[
        bool,
        Parameter(
            name=("--write-copyright",),
            negative="--no-write-copyright",
            help=(
                "Write the copyright block (--artist, --copyright, --rights, --marked, --set). On "
                "by default, and writes nothing unless one of those is set. Pass "
                "--no-write-copyright to skip it for one run without clearing your config file"
            ),
        ),
    ] = True
    backup_xmp: Annotated[
        bool,
        Parameter(
            name=("--backup-xmp",),
            negative="--no-backup-xmp",
            help="Create an ExifTool backup (_original) before overwriting metadata",
        ),
    ] = True
    sidecar_mode: Annotated[
        SidecarMode | None,
        Parameter(
            name=("--sidecar-mode",),
            help=(
                "Where metadata goes: 'all' (the default) writes an XMP sidecar next to every "
                "photo, 'none' embeds it in the photo itself, 'raw' splits by file type (a sidecar "
                "for RAW files, embedded for everything else), and 'both' writes each photo and a "
                "sidecar beside it. 'raw' is the answer for a folder of RAW+JPEG pairs, where the "
                "RAWs should keep their bytes untouched but the JPEGs should carry their own "
                "metadata"
            ),
        ),
    ] = None
    use_sidecar: Annotated[
        bool | None,
        Parameter(
            name=("--write-sidecar",),
            negative="--embed-in-photo",
            help=(
                "Shorthand for --sidecar-mode: --write-sidecar is 'all' (the default), "
                "--embed-in-photo is 'none'. An explicit --sidecar-mode wins over both"
            ),
        ),
    ] = None
    sidecar_overrides: Annotated[
        list[str] | None,
        Parameter(
            name=("--sidecar-override",),
            validator=validate_sidecar_overrides,
            help=(
                "Send one file type somewhere else than --sidecar-mode says, as EXT=TARGET with "
                "TARGET one of 'sidecar', 'photo', or 'both' (repeat this option). For example, "
                "--sidecar-override dng=photo writes into each DNG, which is where Lightroom "
                "Classic keeps a DNG's metadata"
            ),
        ),
    ] = None
    dry_run: Annotated[
        bool,
        Parameter(
            name=("--dry-run",),
            help=(
                "Run the model and log the proposed metadata for each photo, but do not "
                "write XMP. Useful for previewing prompts before committing to a batch"
            ),
        ),
    ] = False
    max_keywords: Annotated[
        int | None,
        Parameter(
            name=("--max-keywords",),
            help=(
                "Cap the number of AI-generated keywords kept per photo before merging with "
                "existing tags. Lightroom users with already-curated catalogs typically want a "
                "lower cap (e.g. 10) so the merged keyword cloud stays readable"
            ),
        ),
    ] = None
    vocabulary: Annotated[
        Path | None,
        Parameter(
            name=("--vocabulary",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help=(
                "Restrict generated keywords to the terms in this file: a Lightroom keyword-list "
                "export, or a plain list of terms and 'Animal|Bird|Osprey' paths. Terms are "
                "matched ignoring case, punctuation, and plurals, and every match is rewritten to "
                "the file's own spelling and hierarchy, so a run cannot seed your catalog with "
                "near-duplicates of keywords you already have"
            ),
        ),
    ] = None
    vocabulary_strict: Annotated[
        bool,
        Parameter(
            name=("--vocabulary-strict",),
            help=(
                "Drop generated keywords that --vocabulary does not cover instead of writing them "
                "as-is. The run summary lists every dropped term and how often it came up, so the "
                "vocabulary can grow on purpose rather than by accident"
            ),
        ),
    ] = False
    session_gap_minutes: Annotated[
        float,
        Parameter(
            name=("--session-gap",),
            validator=validators.Number(gte=0),
            help=(
                "Group photos into shoots separated by this many idle minutes (by capture time, "
                "falling back to file mtime) and make each shoot's keywords agree with itself: "
                "the spelling and the hierarchy most of the session used win for all of it. "
                "Nothing in a session is written until every photo in it has been analyzed. "
                "0 (the default) processes each photo on its own, as before"
            ),
        ),
    ] = 0.0


@dataclass
class LogConfig:
    """
    Where a run's diagnostics go: the loguru sinks, and optional OpenTelemetry tracing.

    The ``--otel`` flags share this group rather than forming their own because a tenth option group
    would push ``watch`` past the parameter count a single function can be read at. They are the
    same kind of thing as the log sinks anyway: output about the run, for whoever is debugging it.
    """

    file_log_level: Annotated[
        LogLevel,
        Parameter(name="--file-log-level", help="Log level for file (use 'OFF' to disable)"),
    ] = "DEBUG"
    console_log_level: Annotated[
        LogLevel,
        Parameter(
            name="--console-log-level",
            help="Log level for console (use 'OFF' to disable)",
        ),
    ] = "INFO"
    log_folder: Annotated[
        Path,
        Parameter(
            name=("--log-folder",),
            validator=validators.Path(file_okay=False, dir_okay=True),
            help="Folder where log files are stored",
        ),
    ] = field(default_factory=lambda: Path("logs"))
    otel: Annotated[
        bool,
        Parameter(
            name=("--otel",),
            negative="--no-otel",
            help=(
                "Export an OpenTelemetry trace of every model call (timings, tokens, failures) to "
                "your own collector: Logfire, Jaeger, Grafana Tempo, anything speaking OTLP over "
                "HTTP. Off by default, and needs an endpoint from --otel-endpoint or "
                "OTEL_EXPORTER_OTLP_ENDPOINT. The photos themselves are never exported"
            ),
        ),
    ] = False
    otel_endpoint: Annotated[
        str | None,
        Parameter(
            name=("--otel-endpoint",),
            help=(
                "Where to send traces, e.g. http://localhost:4318/v1/traces. Defaults to "
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, then OTEL_EXPORTER_OTLP_ENDPOINT plus "
                "/v1/traces"
            ),
        ),
    ] = None
    otel_content: Annotated[
        bool,
        Parameter(
            name=("--otel-content",),
            negative="--no-otel-content",
            help=(
                "Include the prompt and the generated title, description, and keywords in the "
                "exported traces. Off by default because that text describes your photos; turn it "
                "on when debugging a prompt against a collector you trust. Photo bytes stay out "
                "either way"
            ),
        ),
    ] = False


@dataclass
class FilterConfig:
    """Filters applied to the resolved file list before the pipeline runs."""

    skip_tagged: Annotated[
        bool,
        Parameter(
            name=("--skip-tagged",),
            help=(
                "Skip files whose image or XMP sidecar already has keywords, a description, "
                "or a title (set by an earlier run, Lightroom, or another tool)"
            ),
        ),
    ] = False
    newer_than: Annotated[
        str | None,
        Parameter(
            name=("--newer-than",),
            help=(
                "Drop files whose mtime is on or before this ISO 8601 timestamp "
                "(e.g. 2024-01-01 or 2024-01-01T14:30). Naive timestamps are treated as "
                "local time"
            ),
        ),
    ] = None
    older_than: Annotated[
        str | None,
        Parameter(
            name=("--older-than",),
            help=(
                "Drop files whose mtime is on or after this ISO 8601 timestamp. "
                "Combine with --newer-than to select a window"
            ),
        ),
    ] = None


@dataclass
class DisplayConfig:
    """
    How a run makes itself visible: the progress bar, per-image NDJSON, the alert when it ends.

    The desktop notification lives here rather than in a group of its own because it answers the
    same question the other two do, and because one more option group would put ``tag`` and
    ``watch`` over Sonar's parameter cap for no gain in clarity.
    """

    progress_bar: Annotated[
        bool,
        Parameter(
            name=("--progress",),
            negative="--no-progress",
            help=(
                "Show a live rich progress bar (default on interactive terminals). Disabled "
                "automatically when stderr is not a tty (CI, redirected output)"
            ),
        ),
    ] = True
    json_output: Annotated[
        bool,
        Parameter(
            name=("--json",),
            help=(
                "Emit one NDJSON line per processed photo to stdout (file, status, title, "
                "description, keywords, token usage, seconds, cache flag). Useful for "
                "piping into other tools. Logs and progress stay on stderr"
            ),
        ),
    ] = False
    notify: Annotated[
        bool,
        Parameter(
            name=("--notify",),
            negative="--no-notify",
            help=(
                "Raise a desktop notification when the batch finishes, so a long run can be "
                "started and walked away from. Uses whatever notifier the OS already ships "
                "(osascript, notify-send, a PowerShell toast); best-effort, and skipped in "
                "silence on a machine that has none"
            ),
        ),
    ] = False
    notify_sound: Annotated[
        bool,
        Parameter(
            name=("--notify-sound",),
            negative="--no-notify-sound",
            help=(
                "Play a sound with the notification, falling back to the terminal bell where "
                "the platform offers nothing better. Implies --notify"
            ),
        ),
    ] = False


@dataclass
class ArtifactConfig:
    """Optional sidecar files the run reads (prompt, skip list) or writes (summary, cache)."""

    skip_from: Annotated[
        Path | None,
        Parameter(
            name=("--skip-from",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help="Path to newline-delimited text file listing filenames to skip",
        ),
    ] = None
    append_to_skip_file: Annotated[
        Path | None,
        Parameter(
            name=("--append-to-skip-file",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "Append the path of each successfully-processed file to this file. "
                "Created if it does not exist. Pass the same path to --skip-from on later "
                "runs to resume work without redoing finished photos"
            ),
        ),
    ] = None
    prompt_file: Annotated[
        Path | None,
        Parameter(
            name=("--prompt-file",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help=(
                "Override the default user prompt with the contents of this file. The "
                "prompt is used as-is; existing photo metadata (keywords, GPS, location) "
                "is appended automatically as before"
            ),
        ),
    ] = None
    summary_file: Annotated[
        Path | None,
        Parameter(
            name=("--summary-file",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "Write a JSON summary of the run (success counts, failed files, token usage, "
                "wall time) to this path on completion. Created if missing"
            ),
        ),
    ] = None
    csv_file: Annotated[
        Path | None,
        Parameter(
            name=("--csv-file",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "Write a CSV report with one row per photo: filename, generated title, "
                "description, and keywords, the keywords already on the file, the camera/location "
                "EXIF read as context, and per-photo token usage and timing. Rows stream as photos "
                "finish, so a stopped run still leaves a valid file. Created if missing"
            ),
        ),
    ] = None
    cache_file: Annotated[
        Path | None,
        Parameter(
            name=("--cache-file",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "SQLite cache of model outputs keyed by image-content hash and model name. "
                "The hash covers the image data only and ignores metadata, so embedding tags "
                "does not change it: a second run over the same folder still hits the cache "
                "instead of calling the model again. (Formats ExifTool cannot content-hash "
                "fall back to a whole-file hash, which re-embedding does change.) Created if "
                "missing; safe to delete to clear the cache"
            ),
        ),
    ] = None
    undo_log: Annotated[
        bool,
        Parameter(
            name=("--undo-log",),
            negative="--no-undo-log",
            help=(
                "Record every file this run writes to a journal under the state directory, so "
                "'photo-tagger undo' can put it back. On by default: the runs worth undoing are "
                "the ones nobody planned to. Journals are small and pruned automatically"
            ),
        ),
    ] = True
    lock_file: Annotated[
        Path | None,
        Parameter(
            name=("--lock-file",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "Acquire an exclusive file lock on this path before running. Refuses "
                "to start if another photo-tagger process holds the same lock, preventing "
                "two runs from racing on the same folder"
            ),
        ),
    ] = None


@dataclass
class WatchConfig:
    """
    How the ``watch`` command scans for photos, waits for each one, and works through a batch.

    Only the ``watch`` command takes these, and like the vocabulary build flags they have no entry
    in ``_CONFIG_TABLES``: ``interval`` and ``settle`` belong to one command, not to a run.
    ``workers`` is the exception. It is the same knob ``tag`` takes, and it still reads its config
    key from the file's top level, because config keys resolve by CLI option name rather than by
    which group holds the field. It lives here so ``watch`` stays inside the parameter budget.
    """

    interval: Annotated[
        float,
        Parameter(
            name=("--interval",),
            validator=validators.Number(gt=0),
            help="Seconds between folder scans",
        ),
    ] = DEFAULT_INTERVAL_SECONDS
    settle: Annotated[
        float,
        Parameter(
            name=("--settle",),
            validator=validators.Number(gte=0),
            help=(
                "Seconds a file must sit unchanged before it is tagged, so a photo still being "
                "copied is left alone until the copy finishes"
            ),
        ),
    ] = DEFAULT_SETTLE_SECONDS
    workers: Annotated[
        int,
        Parameter(
            name=("--workers", "-w"),
            help="Photos to process concurrently per batch",
        ),
    ] = DEFAULT_WORKERS


@dataclass
class VocabularyBuildConfig:
    """Filters the ``vocabulary`` command applies when turning a keyword census into a file."""

    min_uses: Annotated[
        int,
        Parameter(
            name=("--min-uses",),
            help=(
                "Keep a keyword only when the library uses it at least this many times. The "
                "single most useful knob: in a catalog an AI has been writing to, most keywords "
                "are used once and are one-offs, not vocabulary"
            ),
        ),
    ] = 2
    max_terms: Annotated[
        int | None,
        Parameter(
            name=("--max-terms",),
            help=(
                "Cap the file at this many keywords, dropping the least used first. The default "
                "keeps it under the 5000 terms above which vocabulary matching gives up its fuzzy "
                "pass. Pass 0 for no cap"
            ),
        ),
    ] = 4800
    allow_digits: Annotated[
        bool,
        Parameter(
            name=("--allow-digits",),
            help=(
                "Keep keywords containing digits. They are dropped by default because they are "
                "nearly always measurements or model numbers ('19.5V', '0 Percent Battery') "
                "rather than subjects"
            ),
        ),
    ] = False
    output_language: Annotated[
        str,
        Parameter(
            name=("--output-language", "--lang"),
            help=(
                "Language the catalog's keywords are in. Only English plurals can be folded "
                "onto their singular; naming any other language keeps 'Alles' and 'Alle' apart "
                "instead of dropping one of them as a variant"
            ),
        ),
    ] = DEFAULT_OUTPUT_LANGUAGE
    flat: Annotated[
        bool,
        Parameter(
            name=("--flat",),
            help=(
                "Write bare keywords instead of their hierarchies. Worth using when the source "
                "hierarchy is not trustworthy, since a vocabulary imposes its own on every photo "
                "it matches"
            ),
        ),
    ] = False
    report_file: Annotated[
        Path | None,
        Parameter(
            name=("--report",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "Write a CSV of every dropped keyword, its usage count, and the rule that cut it. "
                "This is what makes the thresholds tunable rather than a guess"
            ),
        ),
    ] = None
    organize: Annotated[
        bool,
        Parameter(
            name=("--organize",),
            help=(
                "Ask the model for the two things counting cannot settle: which keywords are "
                "synonyms of each other (folded into one, the others kept as {synonyms} so they "
                "still match), and what hierarchy the list should have. Never decides what to "
                "keep, and never invents a keyword. Needs a reachable provider"
            ),
        ),
    ] = False
    organize_workers: Annotated[
        int,
        Parameter(
            name=("--organize-workers",),
            help="Model requests to run at once while organizing. The list is sent in chunks",
        ),
    ] = 1


@dataclass
class TelemetryConfig:
    """Anonymous, opt-out usage telemetry toggle."""

    enabled: Annotated[
        bool,
        Parameter(
            name=("--telemetry",),
            negative="--no-telemetry",
            help=(
                "Send anonymous usage stats (the settings and features used, model, batch size, "
                "OS, CPU/GPU model, RAM size, timing, outcomes) and anonymous crash reports (error "
                "type and stack trace, code locations only) to help guide development. No photos, "
                "file paths, filenames, tags, prompts, error messages, or personal data are ever "
                "sent. Disable with --no-telemetry, PHOTO_TAGGER_TELEMETRY=0, or the cross-tool "
                "DO_NOT_TRACK=1. PHOTO_TAGGER_TELEMETRY=1 sends anyway when DO_NOT_TRACK is set"
            ),
        ),
    ] = True


@dataclass
class PacingConfig:
    """
    How hard a run is allowed to push the machine (see :mod:`photo_tagger.pacing`).

    Everything here is off by default. A run that is not asked to slow down never does.
    """

    duty: Annotated[
        int,
        Parameter(
            name=("--duty",),
            validator=validators.Number(gte=1, lte=100),
            help=(
                "Percentage of the time the run may work, resting for the remainder. 100 (the "
                "default) works flat out. Use it to keep the fans quiet, to leave the machine "
                "usable, or to draw less power on average than the charger supplies"
            ),
        ),
    ] = DEFAULT_DUTY_PERCENT
    battery_floor: Annotated[
        int,
        Parameter(
            name=("--battery-floor",),
            validator=validators.Number(gte=0, lte=MAX_BATTERY_FLOOR),
            help=(
                "Battery percentage at which the run rests until the charge comes back. 0 never "
                "watches the battery. Above 90 the run warns and carries on: a plugged-in laptop "
                "is rarely brim full, so a floor up there rests after nearly every window. Only "
                "ever rests while a charger is plugged in, since on battery power a rest "
                "recharges nothing"
            ),
        ),
    ] = DEFAULT_BATTERY_FLOOR
    battery_resume: Annotated[
        int,
        Parameter(
            name=("--battery-resume",),
            validator=validators.Number(gte=0, lte=100),
            help=(
                "Battery percentage that ends a rest started by --battery-floor. Defaults to 15 "
                "points above the floor"
            ),
        ),
    ] = DEFAULT_BATTERY_RESUME
    work_minutes: Annotated[
        float,
        Parameter(
            name=("--work-minutes",),
            validator=validators.Number(gt=0),
            help=(
                "Minutes of work between rests, for a run that --duty or --battery-floor has "
                "asked to rest; on its own it changes nothing. A rest falls at the first photo "
                "boundary after the window, never in the middle of a photo, and never after "
                "every photo: short rests charge nothing and give a model server the chance to "
                "unload"
            ),
        ),
    ] = DEFAULT_WORK_MINUTES
    max_pause_minutes: Annotated[
        float,
        Parameter(
            name=("--max-pause-minutes",),
            validator=validators.Number(gt=0),
            help=(
                "Longest a single rest may last, in minutes, for a run that --duty or "
                "--battery-floor has asked to rest. Bounds the wait for a battery that is "
                "charging slowly, and keeps the run under the idle timeout that makes a model "
                "server unload its model"
            ),
        ),
    ] = DEFAULT_MAX_PAUSE_MINUTES


# The config-file tables and the option group each one feeds. Also drives the translation of
# config field names into CLI option names inside `ConfigFileSource`.
_CONFIG_TABLES: dict[str, type] = {
    "provider": ProviderConfig,
    "output": OutputConfig,
    # After [output], which inherits these same fields: later tables win, and [copyright] is the
    # table the docs name and the GUI saves, so a change made there must not lose to a stale
    # [output] key.
    "copyright": CopyrightConfig,
    "inference": InferenceConfig,
    "log": LogConfig,
    "display": DisplayConfig,
    "artifacts": ArtifactConfig,
    "filter": FilterConfig,
    "telemetry": TelemetryConfig,
    "pacing": PacingConfig,
}

# Top-level config keys that are plain flags on `tag` (their config key equals the CLI name).
# `exiftool_path` is deliberately absent: it has no flag and is bridged into the environment.
_TOP_LEVEL_KEYS = ("extensions", "workers", "recursive")


def known_config_keys() -> dict[str, frozenset[str]]:
    """Map each config table to the keys photo-tagger reads, with "" for the top level."""
    known = {table: frozenset(typing.get_type_hints(cls)) for table, cls in _CONFIG_TABLES.items()}
    known[""] = frozenset((*_TOP_LEVEL_KEYS, "exiftool_path", "language"))
    known["gui"] = frozenset({"save_after_generating"})
    return known


def _cli_option_names(cls: type) -> dict[str, str]:
    """Map each of *cls*'s field names to its primary long CLI option, without the dashes."""
    names: dict[str, str] = {}
    for field_name, hint in typing.get_type_hints(cls, include_extras=True).items():
        for meta in get_args(hint)[1:]:
            if not isinstance(meta, Parameter):
                continue
            declared = (meta.name,) if isinstance(meta.name, str) else (meta.name or ())
            if long_name := next((n for n in declared if n.startswith("--")), None):
                names[field_name] = long_name.removeprefix("--")
            break
    return names


def cli_config_overrides(file_config: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten the nested TOML *file_config* into a dict keyed by CLI option names.

    Config files use dataclass field names (``[provider] model_name = ...``) while cyclopts matches
    config keys against the declared option names (``--model``), so each known field is translated.
    Unknown keys are dropped, keeping older versions tolerant of forward-compatible entries.
    """
    flat: dict[str, Any] = {}
    for table, cls in _CONFIG_TABLES.items():
        entries = file_config.get(table)
        if not isinstance(entries, dict):
            continue
        names = _cli_option_names(cls)
        for key, value in entries.items():
            if (option := names.get(key)) is not None:
                flat[option] = value
    for key in _TOP_LEVEL_KEYS:
        if key in file_config:
            flat[key] = file_config[key]
    return flat


def _apply_shadowed_keys(
    values: dict[str, Any],
    arguments: ArgumentCollection,
    *,
    source: str,
) -> None:
    """
    Set the flags cyclopts' own config layer refuses to look at, because a command shares the name.

    Cyclopts drops every flat-config key that names a subcommand before it matches anything (see
    ``cyclopts.config._common.ConfigBase.__call__``), which is right for hierarchical configs and
    wrong for ours: our keys are option names, so ``[output] vocabulary`` was silently ignored the
    moment a ``vocabulary`` command existed. These keys are therefore handed to the argument they
    name directly, mirroring what cyclopts would have done with them.

    This is cyclopts' own matching step, called directly on the keys its filter removed, rather than
    a second implementation of it: conversion, validation, and "a flag the user passed wins" all
    behave exactly as they do for every other key.
    """
    if not values:
        return
    update_argument_collection(
        values,
        source,
        arguments,
        # No table nesting: these keys are option names, and a command that has no such flag
        # (doctor, gui) must skip it rather than fail.
        root_keys=(),
        allow_unknown=True,
    )


@dataclass(slots=True, frozen=True)
class Invocation:
    """What the command line asked for: the command, and the names of the options typed on it."""

    command: str = "tag"
    cli_flags: tuple[str, ...] = ()


def _typed_flags(arguments: ArgumentCollection) -> tuple[str, ...]:
    """
    Name the options typed on the command line, never their values.

    Every spelling of an option is reported by its own long name, so an alias (``--lang``), a short
    name, or a positional path counts as the one option it sets. A negative keeps its "no", named by
    the option's first negative, so ``--no-backup-xmp`` stays apart from ``--backup-xmp``.
    """
    names: set[str] = set()
    for argument in arguments:
        negatives = tuple(argument.negatives)
        for token in argument.tokens:
            if token.source == "cli":
                keyword = (token.keyword or "").split("=", 1)[0]
                names.add(negatives[0] if keyword in negatives else argument.name)
    return tuple(sorted(names))


# The two-state shorthand for --sidecar-mode, as typed on the command line.
_SIDECAR_SHORTHAND = frozenset({"--write-sidecar", "--embed-in-photo"})


def _drop_modes_the_shorthand_overrules(overrides: dict[str, Any], typed: tuple[str, ...]) -> None:
    """
    Leave the config file's sidecar mode out when the command line typed the shorthand for one.

    An explicit --sidecar-mode wins over --embed-in-photo, which is right when both are typed. The
    GUI writes a mode whenever it saves its settings as defaults, though, so without this a typed
    --embed-in-photo would lose to a mode the user never typed at all: what is typed wins over what
    is saved.
    """
    if _SIDECAR_SHORTHAND.intersection(typed) and "--sidecar-mode" not in typed:
        overrides.pop("sidecar-mode", None)


class ConfigFileSource:
    """
    Cyclopts config hook that fills flags the user did not pass from the TOML config file.

    Cyclopts constructs a fresh option-group instance whenever any flag from that group appears on
    the command line, so config values baked into the *default instances* would be dropped for the
    group's other fields. Routing the file through cyclopts' config layer instead yields per-field
    precedence: CLI flag > config file > built-in default.

    The file is re-read on every invocation, so tests (and long-lived processes) observe changes to
    ``$PHOTO_TAGGER_CONFIG`` without re-importing ``main``. The hook also records which command ran
    and which options were typed (:attr:`invocation`), for telemetry.
    """

    def __init__(self) -> None:
        """Start with an empty invocation until cyclopts calls the hook."""
        self.invocation = Invocation()

    def __call__(self, app: App, commands: tuple[str, ...], arguments: ArgumentCollection) -> None:
        """Feed the flattened config to cyclopts for any argument without CLI tokens."""
        self.invocation = Invocation(
            command=commands[-1] if commands else "tag",
            cli_flags=_typed_flags(arguments),
        )
        overrides = cli_config_overrides(load_config())
        _drop_modes_the_shorthand_overrules(overrides, self.invocation.cli_flags)
        if not overrides:
            return
        source = find_config_file()
        source_name = str(source) if source is not None else "config"
        # Keys that share a name with a command never reach the delegate below, so take them out
        # first and apply them by hand.
        shadowed = {key: value for key, value in overrides.items() if key in app}
        for key in shadowed:
            del overrides[key]
        delegate = _CycloptsDictConfig(
            data=overrides,
            # The keys are global option names, not per-command tables, and commands that lack a
            # given flag (gui, doctor) must ignore it rather than error.
            use_commands_as_keys=False,
            allow_unknown=True,
            source=source_name,
        )
        delegate(app, commands, arguments)
        _apply_shadowed_keys(shadowed, arguments, source=source_name)


def resolve_sidecar_mode(output: OutputConfig) -> SidecarMode:
    """
    Fold the two ways of asking for a write target into the one the pipeline uses.

    ``--sidecar-mode`` is the real option; ``--write-sidecar`` / ``--embed-in-photo`` stay as the
    two-state shorthand they have always been. Both default to None (nothing asked), so an explicit
    mode can win over the boolean without having to guess which of the two the user typed.
    """
    if output.sidecar_mode is not None:
        return output.sidecar_mode
    if output.use_sidecar is None:
        return DEFAULT_SIDECAR_MODE
    return "all" if output.use_sidecar else "none"


def resolve_sidecar_overrides(output: OutputConfig) -> dict[str, FileTypeTarget]:
    """
    Parse ``--sidecar-override`` into the per-extension answers the writer uses.

    The CLI has already refused a malformed entry at parse time. The GUI reads the config file
    without that step, so it catches the :class:`ValueError` this raises in that case.
    """
    return parse_sidecar_overrides(output.sidecar_overrides)


def resolve_notify(display: DisplayConfig) -> tuple[bool, bool]:
    """
    Fold the two notification flags into (raise an alert, make it audible).

    ``--notify-sound`` on its own is read as asking for both: the sound is how the notification
    announces itself, so nobody means it to play without one.
    """
    return (display.notify or display.notify_sound, display.notify_sound)


def to_extra_fields(config: CopyrightConfig) -> ExtraFields:
    """
    Translate the CLI's copyright group into the writer's own value object.

    Mirrors :func:`to_trim_rules`: the schema module owns the flags, :mod:`photo_tagger.fields` owns
    the behavior. Raises :class:`~photo_tagger.errors.FieldError` on a malformed ``--set`` or an
    placeholder it cannot fill, so the run stops before it stamps the first photo.
    """
    return build_extra_fields(
        artist=config.artist,
        notice=config.copyright_notice,
        rights=config.rights,
        marked=config.marked,
        assignments=config.set_tags,
        preserve=config.preserve_copyright,
    )


def to_processing_options(
    output: OutputConfig,
    inference: InferenceConfig,
    *,
    vocabulary: Vocabulary | None = None,
    retries: int = DEFAULT_RETRIES,
) -> ProcessingOptions:
    """
    Combine the CLI's output + inference groups into the pipeline's options dataclass.

    *vocabulary* is passed in already loaded rather than read from ``output.vocabulary`` here:
    parsing the file is IO that can fail, and this module is pure schema.
    """
    return ProcessingOptions(
        preserve_existing_kw=output.preserve_keywords,
        preserve_existing_title=output.preserve_title,
        preserve_existing_description=output.preserve_description,
        write_description=output.write_description,
        write_title=output.write_title,
        write_keywords=output.write_keywords,
        backup_xmp=output.backup_xmp,
        sidecar_mode=resolve_sidecar_mode(output),
        sidecar_overrides=resolve_sidecar_overrides(output),
        dry_run=output.dry_run,
        temperature=inference.temperature,
        max_tokens=inference.max_tokens,
        timeout_seconds=inference.timeout_seconds,
        retries=retries,
        frequency_penalty=inference.frequency_penalty,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
        max_new_keywords=output.max_keywords,
        vocabulary=vocabulary,
        vocabulary_strict=output.vocabulary_strict,
        output_language=inference.output_language,
        repair_language=inference.repair_language,
        # OutputConfig is a CopyrightConfig (see its docstring), so the authorship flags arrive
        # with the rest of the output group. --no-write-copyright switches the whole block off
        # without anyone having to empty the fields that hold it.
        extra_fields=to_extra_fields(output) if output.write_copyright else ExtraFields(),
    )


def to_pacing_plan(config: PacingConfig) -> PacingPlan:
    """
    Translate the CLI's pacing group into the plan the pacer reads.

    The flags are in minutes because that is how a person thinks about resting a laptop; the plan is
    in seconds because that is how the code measures one.
    """
    return PacingPlan(
        duty_percent=config.duty,
        battery_floor=config.battery_floor,
        battery_resume=config.battery_resume,
        work_seconds=config.work_minutes * SECONDS_PER_MINUTE,
        max_pause_seconds=config.max_pause_minutes * SECONDS_PER_MINUTE,
    )


def _marked_label(*, marked: bool | None) -> str:
    """Name the XMP Marked choice the way telemetry reports it."""
    if marked is None:
        return "unset"
    return "copyrighted" if marked else "public-domain"


def to_telemetry_settings(  # noqa: PLR0913 - one parameter per option group tag() takes.
    *,
    provider: ProviderConfig,
    inference: InferenceConfig,
    output: OutputConfig,
    artifacts: ArtifactConfig,
    display: DisplayConfig,
    filter_: FilterConfig,
    log: LogConfig,
    pacing: PacingConfig,
    workers: int,
    extensions: str,
    recursive: bool,
    vocabulary_terms: int = 0,
    watch: WatchConfig | None = None,
) -> telemetry.Settings:
    """
    Reduce a run's option groups to the bounded values telemetry reports.

    Mirrors :func:`to_processing_options`: this module owns the flags, telemetry owns what is sent.
    Nothing here keeps a path or free text: a file becomes "one is set", a hint its length.
    """
    backend = get_backend(provider.provider_name)
    url = provider.api_base_url or backend.default_base_url
    notify, notify_sound = resolve_notify(display)
    watching = watch or WatchConfig()
    return telemetry.Settings(
        endpoint=endpoint_kind(url),
        secure=url.lower().startswith("https://"),
        custom_url=provider.api_base_url is not None,
        api_key=bool(backend.resolve_api_key(provider.api_key)),
        retries=provider.retries,
        timeout_seconds=inference.timeout_seconds,
        temperature=inference.temperature,
        max_tokens=inference.max_tokens,
        frequency_penalty=inference.frequency_penalty,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
        repair_language=inference.repair_language,
        hint=bool(inference.hint),
        hint_chars=len(inference.hint or ""),
        custom_prompt=artifacts.prompt_file is not None,
        workers=workers,
        cache=artifacts.cache_file is not None,
        max_keywords=output.max_keywords or 0,
        vocabulary=output.vocabulary is not None,
        vocabulary_terms=vocabulary_terms,
        vocabulary_strict=output.vocabulary_strict,
        session_gap_minutes=output.session_gap_minutes,
        sidecar_mode=resolve_sidecar_mode(output),
        sidecar_overrides=len(resolve_sidecar_overrides(output)),
        write_title=output.write_title,
        write_description=output.write_description,
        write_keywords=output.write_keywords,
        write_copyright=output.write_copyright,
        overwrite_title=not output.preserve_title,
        overwrite_description=not output.preserve_description,
        overwrite_keywords=not output.preserve_keywords,
        backup=output.backup_xmp,
        undo_log=artifacts.undo_log,
        dry_run=output.dry_run,
        artist=bool(output.artist),
        copyright_notice=bool(output.copyright_notice),
        rights=bool(output.rights),
        marked=_marked_label(marked=output.marked),
        overwrite_copyright=not output.preserve_copyright,
        set_tags=len(output.set_tags or []),
        paced=to_pacing_plan(pacing).rests,
        duty=pacing.duty,
        battery_floor=pacing.battery_floor,
        battery_resume=pacing.battery_resume,
        work_minutes=pacing.work_minutes,
        max_pause_minutes=pacing.max_pause_minutes,
        notify=notify,
        notify_sound=notify_sound,
        extensions=extension_list(extensions.split(",")),
        recursive=recursive,
        skip_tagged=filter_.skip_tagged,
        date_filter=bool(filter_.newer_than or filter_.older_than),
        skip_list=artifacts.skip_from is not None,
        resume_list=artifacts.append_to_skip_file is not None,
        summary_file=artifacts.summary_file is not None,
        csv_report=artifacts.csv_file is not None,
        json_output=display.json_output,
        progress_bar=display.progress_bar,
        lock_file=artifacts.lock_file is not None,
        otel=log.otel,
        otel_content=log.otel_content,
        console_log_level=log.console_log_level,
        file_log_level=log.file_log_level,
        watch_interval=watching.interval,
        watch_settle=watching.settle,
    )


def to_trim_rules(config: VocabularyBuildConfig) -> TrimRules:
    """
    Translate the CLI's vocabulary-build group into the builder's own rules.

    Mirrors :func:`to_processing_options`: the schema module owns the flags, the builder owns the
    behavior, and neither imports the other's vocabulary. ``--max-terms 0`` reads as "no cap",
    because a cyclopts flag cannot take ``None`` on the command line.
    """
    return TrimRules(
        min_uses=config.min_uses,
        max_terms=config.max_terms or None,
        allow_digits=config.allow_digits,
        # Folding plurals is English morphology, and _collapse_variants drops the loser of every
        # collision. Left on for a German catalog it keys "Alles" as "alle" and deletes one of two
        # real keywords. load_vocabulary and the session builder already gate on the language;
        # this is the third caller and was the only one that did not.
        fold_plurals=folds_plurals(config.output_language),
    )


@dataclass(slots=True, frozen=True)
class Defaults:
    """The fully-resolved default option groups, after folding in the TOML config."""

    provider: ProviderConfig
    output: OutputConfig
    inference: InferenceConfig
    log: LogConfig
    display: DisplayConfig
    artifacts: ArtifactConfig
    filter: FilterConfig
    telemetry: TelemetryConfig
    pacing: PacingConfig
    extensions: str
    workers: int
    recursive: bool
    exiftool_path: str | None


def load_defaults(config: dict[str, Any] | None = None) -> Defaults:
    """
    Build the default option groups, layering any TOML config over the built-ins.

    The CLI resolves its config through :class:`ConfigFileSource` at parse time instead; this is for
    callers that never parse a command line, such as the GUI.
    """
    file_config = load_config() if config is None else config
    return Defaults(
        provider=apply_overrides(ProviderConfig(), file_config.get("provider", {})),
        # The authorship fields are part of OutputConfig but keep their own [copyright] table, so
        # both feed the same instance here. [copyright] wins, as the later one (see _CONFIG_TABLES).
        output=apply_overrides(
            OutputConfig(),
            {**file_config.get("output", {}), **file_config.get("copyright", {})},
        ),
        inference=apply_overrides(InferenceConfig(), file_config.get("inference", {})),
        log=apply_overrides(LogConfig(), file_config.get("log", {})),
        display=apply_overrides(DisplayConfig(), file_config.get("display", {})),
        artifacts=apply_overrides(ArtifactConfig(), file_config.get("artifacts", {})),
        filter=apply_overrides(FilterConfig(), file_config.get("filter", {})),
        telemetry=apply_overrides(TelemetryConfig(), file_config.get("telemetry", {})),
        pacing=apply_overrides(PacingConfig(), file_config.get("pacing", {})),
        extensions=file_config.get("extensions", DEFAULT_EXTENSIONS),
        workers=file_config.get("workers", DEFAULT_WORKERS),
        recursive=file_config.get("recursive", DEFAULT_RECURSIVE),
        exiftool_path=file_config.get("exiftool_path"),
    )


def validate_pacing(_type: type, config: PacingConfig) -> None:
    """
    Refuse a resume level at or below the floor, rather than quietly ignoring it.

    The pair only means anything as a band: resuming where the rest began would rest again a photo
    later, so :class:`~photo_tagger.pacing.PacingPlan` falls back to a level above the floor. That
    fallback is right for the flag nobody set, and wrong as an answer to two numbers somebody typed
    on purpose, which is why this is a parse-time error rather than a silent correction.
    """
    if 0 < config.battery_resume <= config.battery_floor:
        msg = (
            f"--battery-resume {config.battery_resume} must be above --battery-floor "
            f"{config.battery_floor}, or 0 to sit a margin above it. As written, the run would "
            f"rest again as soon as it resumed."
        )
        raise ValueError(msg)
