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
# ruff: noqa: PLR0913

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
from photo_tagger.discovery import apply_skip_file, resolve_image_batch
from photo_tagger.logging_setup import setup_logging
from photo_tagger.pipeline import ProcessingOptions, run_batch


__version__ = "0.1.0"

app = App(name="photo-tagger", version=__version__)


def _log_startup(
    *,
    inputs: list[Path] | None,
    skip_from: Path | None,
    image_extensions: str,
    model_name: str,
    provider_name: str,
    api_base_url: str | None,
    api_key: str | None,
    recursive: bool,
    options: ProcessingOptions,
    retries: int,
    log_folder: Path,
) -> None:
    """Single-shot info log so the run's full configuration is captured up-front."""
    logger.info(
        "starting_photo_tagger",
        inputs=[str(p) for p in (inputs or [])],
        extensions=image_extensions,
        model=model_name,
        provider=provider_name,
        api_base_url=api_base_url,
        api_key_present=bool(api_key),
        recursive=recursive,
        skip_from=str(skip_from) if skip_from else None,
        preserve_keywords=options.preserve_existing_kw,
        write_description=options.write_description,
        write_title=options.write_title,
        backup_xmp=options.backup_xmp,
        use_sidecar=options.use_sidecar,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        jpeg_dimensions=options.jpeg_dimensions,
        jpeg_quality=options.jpeg_quality,
        retries=retries,
        log_folder=str(log_folder),
    )


@app.default
def tag(
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
    *,
    image_extensions: Annotated[
        str,
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = "cr3",
    model_name: Annotated[
        str,
        Parameter(name=("--model", "-m"), help="Vision-language model name"),
    ] = DEFAULT_MODEL_NAME,
    provider_name: Annotated[
        Literal["ollama", "lmstudio"],
        Parameter(name=("--provider",), help="Backend provider: 'ollama' or 'lmstudio'"),
    ] = "lmstudio",
    api_base_url: Annotated[
        str | None,
        Parameter(name=("--url", "-u"), help="Provider API base URL"),
    ] = None,
    api_key: Annotated[
        str | None,
        Parameter(name=("--api-key", "-k"), help="Provider API key. Will try env vars if not set"),
    ] = None,
    recursive: Annotated[
        bool,
        Parameter(
            name=("--recursive", "-r"),
            help="Process files in subdirectories recursively",
        ),
    ] = False,
    preserve_keywords: Annotated[
        bool,
        Parameter(
            name=("--preserve-keywords",),
            negative="--overwrite-keywords",
            help="Preserve existing keywords in XMP files (merge) vs overwrite them",
        ),
    ] = True,
    write_description: Annotated[
        bool,
        Parameter(
            name=("--write-description",),
            negative="--no-write-description",
            help="Also generate and write a short description (IFD0/XMP)",
        ),
    ] = True,
    write_title: Annotated[
        bool,
        Parameter(
            name=("--write-title",),
            negative="--no-write-title",
            help="Also generate and write a title (XMP-dc:Title / IPTC:ObjectName)",
        ),
    ] = True,
    backup_xmp: Annotated[
        bool,
        Parameter(
            name=("--backup-xmp",),
            negative="--no-backup-xmp",
            help="Create an ExifTool backup (_original) before overwriting metadata",
        ),
    ] = True,
    use_sidecar: Annotated[
        bool,
        Parameter(
            name=("--write-sidecar",),
            negative="--embed-in-photo",
            help="Write metadata to XMP sidecars (default) instead of embedding in the image",
        ),
    ] = True,
    temperature: Annotated[
        float,
        Parameter(name=("--temperature",), help="Sampling temperature (0.0-1.0)"),
    ] = DEFAULT_TEMPERATURE,
    max_tokens: Annotated[
        int,
        Parameter(name=("--max-tokens",), help="Maximum tokens to generate"),
    ] = DEFAULT_MAX_TOKENS,
    jpeg_dimensions: Annotated[
        int,
        Parameter(
            name=("--jpeg-dimensions",),
            help="Max dimension in pixels for the resized JPEG sent to the model",
        ),
    ] = DEFAULT_DIMENSIONS,
    jpeg_quality: Annotated[
        int,
        Parameter(
            name=("--jpeg-quality",),
            help="JPEG quality (1-100) for the image sent to the model",
        ),
    ] = DEFAULT_JPEG_QUALITY,
    retries: Annotated[
        int,
        Parameter(name=("--retries",), help="Number of automatic validation retries"),
    ] = DEFAULT_RETRIES,
    file_log_level: Annotated[
        LogLevel,
        Parameter(name="--file-log-level", help="Log level for file (use 'OFF' to disable)"),
    ] = "DEBUG",
    log_folder: Annotated[
        Path,
        Parameter(name=("--log-folder",), help="Folder where log files are stored"),
    ] = Path("logs"),
    console_log_level: Annotated[
        LogLevel,
        Parameter(
            name="--console-log-level",
            help="Log level for console (use 'OFF' to disable)",
        ),
    ] = "DEBUG",
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

    Exit status: returns 1 if no inputs, no images found, or any file fails.

    Examples:
        photo-tagger -i ./photos/IMG_0001.CR3

        photo-tagger \
            --extensions cr3,jpg \
            --provider lmstudio \
            --url http://localhost:1234/v1 \
            --recursive \
            --skip-from processed.txt \
            Pictures/Camera

    """
    setup_logging(
        file_log_level=file_log_level,
        console_log_level=console_log_level,
        log_folder=log_folder,
    )

    options = ProcessingOptions(
        preserve_existing_kw=preserve_keywords,
        write_description=write_description,
        write_title=write_title,
        backup_xmp=backup_xmp,
        use_sidecar=use_sidecar,
        temperature=temperature,
        max_tokens=max_tokens,
        jpeg_dimensions=jpeg_dimensions,
        jpeg_quality=jpeg_quality,
    )

    _log_startup(
        inputs=inputs,
        skip_from=skip_from,
        image_extensions=image_extensions,
        model_name=model_name,
        provider_name=provider_name,
        api_base_url=api_base_url,
        api_key=api_key,
        recursive=recursive,
        options=options,
        retries=retries,
        log_folder=log_folder,
    )

    image_files = apply_skip_file(
        resolve_image_batch(inputs, image_extensions, recursive=recursive),
        skip_from,
    )
    if not image_files:
        logger.info("no_files_to_process_after_skipping")
        return

    agent = create_agent(
        provider_name,
        model_name,
        api_base_url=api_base_url,
        api_key=api_key,
        retries=retries,
    )
    run_batch(image_files, agent, options)


if __name__ == "__main__":
    app()
