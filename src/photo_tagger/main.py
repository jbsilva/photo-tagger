#!/usr/bin/env python3
"""
Photo Tagger: CLI app to describe photos and add keywords using AI.

Optionally non-destructive: create/update XMP sidecar files with Lightroom-compatible tags.
Alternatively, pass --embed-in-photo to write metadata directly into the original file.
Unfortunately, Lightroom uses XMP sidecar files only for proprietary raw formats (e.g., CR3, NEF).
For JPEG, DNG and other formats, you'll often prefer embedding the metadata directly into the file.
This can be done with ExifTool manually as well:
    exiftool -tagsFromFile image.xmp -all:all image.jpg

Requirements:
 - Exiftool installed and available in PATH.
 - Ollama or LM Studio server running and containing a vision-language model.

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter, validators
from loguru import logger

from photo_tagger.ai import create_agent
from photo_tagger.config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RETRIES,
    DEFAULT_TEMPERATURE,
    LogLevel,
)
from photo_tagger.discovery import (
    apply_skip_file,
    apply_skip_tagged,
    make_skip_list_appender,
    resolve_image_batch,
)
from photo_tagger.logging_setup import setup_logging
from photo_tagger.pipeline import ProcessingOptions, run_batch


__version__ = "0.1.0"

app = App(name="photo-tagger", version=__version__)


# --- CLI option groups ---------------------------------------------------------------------------
# Each dataclass below is a group of related CLI flags. Cyclopts' `Parameter(name="*")` flattens
# the fields onto the top-level CLI, so users still pass `--temperature 0.5`, `--no-backup-xmp`,
# etc. The grouping exists purely to keep the `tag` function signature small enough that Sonar's
# S107 parameter-count rule is satisfied without burying the option metadata in a side module.


@dataclass
class ProviderConfig:
    """Backend provider, model id, and credential overrides."""

    model_name: Annotated[
        str,
        Parameter(name=("--model", "-m"), help="Vision-language model name"),
    ] = DEFAULT_MODEL_NAME
    provider_name: Annotated[
        Literal["ollama", "lmstudio"],
        Parameter(name=("--provider",), help="Backend provider: 'ollama' or 'lmstudio'"),
    ] = "lmstudio"
    api_base_url: Annotated[
        str | None,
        Parameter(name=("--url", "-u"), help="Provider API base URL"),
    ] = None
    api_key: Annotated[
        str | None,
        Parameter(name=("--api-key", "-k"), help="Provider API key. Will try env vars if not set"),
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
    ] = "DEBUG"
    log_folder: Annotated[
        Path,
        Parameter(name=("--log-folder",), help="Folder where log files are stored"),
    ] = field(default_factory=lambda: Path("logs"))


# Default group instances. Hoisted to module scope so the function-default expressions in
# `tag` are simple name lookups and ruff's B008 (call-in-default) is satisfied.
_DEFAULT_PROVIDER = ProviderConfig()
_DEFAULT_OUTPUT = OutputConfig()
_DEFAULT_INFERENCE = InferenceConfig()
_DEFAULT_LOG = LogConfig()


def _to_processing_options(output: OutputConfig, inference: InferenceConfig) -> ProcessingOptions:
    """Combine the CLI's output + inference groups into the pipeline's options dataclass."""
    return ProcessingOptions(
        preserve_existing_kw=output.preserve_keywords,
        write_description=output.write_description,
        write_title=output.write_title,
        backup_xmp=output.backup_xmp,
        use_sidecar=output.use_sidecar,
        dry_run=output.dry_run,
        temperature=inference.temperature,
        max_tokens=inference.max_tokens,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
    )


def _log_startup(  # noqa: PLR0913 - the log line names every config explicitly.
    *,
    inputs: list[Path] | None,
    skip_from: Path | None,
    append_to_skip_file: Path | None,
    skip_tagged: bool,
    image_extensions: str,
    recursive: bool,
    provider: ProviderConfig,
    options: ProcessingOptions,
    log: LogConfig,
) -> None:
    """Single-shot info log so the run's full configuration is captured up-front."""
    logger.info(
        "starting_photo_tagger",
        inputs=[str(p) for p in (inputs or [])],
        extensions=image_extensions,
        model=provider.model_name,
        provider=provider.provider_name,
        api_base_url=provider.api_base_url,
        api_key_present=bool(provider.api_key),
        recursive=recursive,
        skip_from=str(skip_from) if skip_from else None,
        append_to_skip_file=str(append_to_skip_file) if append_to_skip_file else None,
        skip_tagged=skip_tagged,
        preserve_keywords=options.preserve_existing_kw,
        write_description=options.write_description,
        write_title=options.write_title,
        backup_xmp=options.backup_xmp,
        use_sidecar=options.use_sidecar,
        dry_run=options.dry_run,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        jpeg_dimensions=options.jpeg_dimensions,
        jpeg_quality=options.jpeg_quality,
        retries=provider.retries,
        log_folder=str(log.log_folder),
    )


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
    skip_from: Annotated[
        Path | None,
        Parameter(
            name=("--skip-from",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help="Path to newline-delimited text file listing filenames to skip",
        ),
    ] = None,
    append_to_skip_file: Annotated[
        Path | None,
        Parameter(
            name=("--append-to-skip-file",),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help=(
                "Append the name of each successfully-processed file to this path. "
                "Created if it does not exist. Pass the same path to --skip-from on later "
                "runs to resume work without redoing finished photos"
            ),
        ),
    ] = None,
    *,
    image_extensions: Annotated[
        str,
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = "cr3",
    recursive: Annotated[
        bool,
        Parameter(
            name=("--recursive", "-r"),
            help="Process files in subdirectories recursively",
        ),
    ] = False,
    skip_tagged: Annotated[
        bool,
        Parameter(
            name=("--skip-tagged",),
            help=(
                "Skip files whose image or XMP sidecar already has keywords, a description, "
                "or a title (set by an earlier run, Lightroom, or another tool)"
            ),
        ),
    ] = False,
    provider: Annotated[ProviderConfig, Parameter(name="*")] = _DEFAULT_PROVIDER,
    output: Annotated[OutputConfig, Parameter(name="*")] = _DEFAULT_OUTPUT,
    inference: Annotated[InferenceConfig, Parameter(name="*")] = _DEFAULT_INFERENCE,
    log: Annotated[LogConfig, Parameter(name="*")] = _DEFAULT_LOG,
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
    - Writes metadata to an XMP sidecar (default) or embeds it directly when --embed-in-photo
        is used. Use --no-write-title/--no-write-description to skip fields; --no-backup-xmp
        to avoid backups.

    Skipping:
    - --skip-from FILE: skip filenames listed in FILE (one per line).
    - --append-to-skip-file FILE: append each successfully processed filename to FILE so a
        later run with --skip-from FILE resumes where this one stopped.
    - --skip-tagged: skip files that already have keywords, a description, or a title in
        the image or its XMP sidecar.

    Dry runs:
    - --dry-run: query the model and log the generated title, description, and keywords
        for each image but do not write any metadata. Useful for prompt iteration.

    Exit status: returns 1 if no inputs, no images found, or any file fails.

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
    setup_logging(
        file_log_level=log.file_log_level,
        console_log_level=log.console_log_level,
        log_folder=log.log_folder,
    )

    options = _to_processing_options(output, inference)
    _log_startup(
        inputs=inputs,
        skip_from=skip_from,
        append_to_skip_file=append_to_skip_file,
        skip_tagged=skip_tagged,
        image_extensions=image_extensions,
        recursive=recursive,
        provider=provider,
        options=options,
        log=log,
    )

    image_files = apply_skip_file(
        resolve_image_batch(inputs, image_extensions, recursive=recursive),
        skip_from,
    )
    image_files = apply_skip_tagged(image_files, skip_tagged=skip_tagged)
    if not image_files:
        logger.info("no_files_to_process_after_skipping")
        return

    agent = create_agent(
        provider.provider_name,
        provider.model_name,
        api_base_url=provider.api_base_url,
        api_key=provider.api_key,
        retries=provider.retries,
    )
    run_batch(image_files, agent, options, on_success=make_skip_list_appender(append_to_skip_file))


if __name__ == "__main__":
    app()
