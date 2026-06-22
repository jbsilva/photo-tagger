"""
CLI option groups and their config-file-aware defaults.

Each dataclass below is a group of related ``photo-tagger`` flags. Cyclopts' ``Parameter(name="*")``
flattens the fields onto the top-level CLI, so users still pass ``--temperature 0.5``, ``--no-
backup-xmp``, etc. The grouping keeps the ``tag`` entry point's signature small enough to satisfy
Sonar's S107 parameter-count rule without burying the option metadata inside ``main``.

Splitting this out of ``main`` keeps the CLI *schema* (what flags exist, their help, their defaults)
separate from the orchestration logic that consumes it.

A TOML config file, if found, overrides the built-in defaults so flags the user does not pass on the
command line pick up persisted values instead. CLI flags always win because cyclopts applies them
after these defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from cyclopts import Parameter, validators

from photo_tagger.config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RETRIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    LogLevel,
)
from photo_tagger.config_file import apply_overrides, load_config
from photo_tagger.pipeline import ProcessingOptions

# Runtime import (not type-only): cyclopts evaluates the Annotated[ProviderName, ...] field
# below to validate the --provider choices, so the name must exist at class-definition time.
from photo_tagger.providers import ProviderName  # noqa: TC001


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
            help="Backend provider: 'ollama', 'lmstudio', or 'openai' (any OpenAI-compatible API)",
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
                " OPENAI_API_KEY) over this flag. Note: CLI args are visible in process listings!"
            ),
        ),
    ] = None
    retries: Annotated[
        int,
        Parameter(name=("--retries",), help="Number of automatic validation retries"),
    ] = DEFAULT_RETRIES


@dataclass
class InferenceConfig:
    """Sampling and image-encoding knobs sent to the model."""

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
            help="Per-image inference timeout in seconds; aborts and lets the retry loop step in",
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
            help="Max dimension in pixels for the resized JPEG sent to the model",
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
class OutputConfig:
    """How metadata is merged with existing tags and where it is written."""

    preserve_keywords: Annotated[
        bool,
        Parameter(
            name=("--preserve-keywords",),
            negative="--overwrite-keywords",
            help="Preserve existing keywords in XMP files (merge) vs overwrite them",
        ),
    ] = True
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
    backup_xmp: Annotated[
        bool,
        Parameter(
            name=("--backup-xmp",),
            negative="--no-backup-xmp",
            help="Create an ExifTool backup (_original) before overwriting metadata",
        ),
    ] = True
    use_sidecar: Annotated[
        bool,
        Parameter(
            name=("--write-sidecar",),
            negative="--embed-in-photo",
            help="Write metadata to XMP sidecars (default) instead of embedding in the image",
        ),
    ] = True
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


@dataclass
class LogConfig:
    """Loguru sink levels and the directory used for rotating log files."""

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
        Parameter(name=("--log-folder",), help="Folder where log files are stored"),
    ] = field(default_factory=lambda: Path("logs"))


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
                "(e.g. 2024-01-01 or 2024-01-01T14:30). Naive timestamps are treated as UTC"
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
    """Stdout/stderr presentation toggles (progress bar, per-image NDJSON)."""

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


@dataclass
class ArtifactConfig:
    """Optional sidecar files the run reads (prompt) or writes (summary, cache)."""

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
                "SQLite cache of model outputs keyed by image content hash and model name. "
                "Reruns that point at the same photos with the same model skip the model "
                "call entirely. Created if missing; safe to delete to clear the cache"
            ),
        ),
    ] = None
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


def to_processing_options(output: OutputConfig, inference: InferenceConfig) -> ProcessingOptions:
    """Combine the CLI's output + inference groups into the pipeline's options dataclass."""
    return ProcessingOptions(
        preserve_existing_kw=output.preserve_keywords,
        write_description=output.write_description,
        write_title=output.write_title,
        write_keywords=output.write_keywords,
        backup_xmp=output.backup_xmp,
        use_sidecar=output.use_sidecar,
        dry_run=output.dry_run,
        temperature=inference.temperature,
        max_tokens=inference.max_tokens,
        timeout_seconds=inference.timeout_seconds,
        frequency_penalty=inference.frequency_penalty,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
        max_new_keywords=output.max_keywords,
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
    extensions: str
    workers: int
    recursive: bool


def load_defaults(config: dict[str, Any] | None = None) -> Defaults:
    """
    Build the default option groups, layering any TOML config over the built-ins.

    Hoisting this out of ``main`` keeps the function-default expressions on the ``tag`` entry point
    simple name lookups, which satisfies ruff's B008 (no function call in a default argument).
    """
    file_config = load_config() if config is None else config
    return Defaults(
        provider=apply_overrides(ProviderConfig(), file_config.get("provider", {})),
        output=apply_overrides(OutputConfig(), file_config.get("output", {})),
        inference=apply_overrides(InferenceConfig(), file_config.get("inference", {})),
        log=apply_overrides(LogConfig(), file_config.get("log", {})),
        display=apply_overrides(DisplayConfig(), file_config.get("display", {})),
        artifacts=apply_overrides(ArtifactConfig(), file_config.get("artifacts", {})),
        filter=apply_overrides(FilterConfig(), file_config.get("filter", {})),
        extensions=file_config.get("extensions", "cr3,jpg"),
        workers=file_config.get("workers", 1),
        recursive=file_config.get("recursive", False),
    )
