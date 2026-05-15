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

import contextlib
import os
import time
import urllib.parse
from http import HTTPStatus
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import rawpy
from cyclopts import App, Parameter, validators
from loguru import logger
from PIL import Image
from pydantic_ai import Agent, AgentRunResult, BinaryContent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from photo_tagger.config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LMSTUDIO_API_KEY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RETRIES,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_USER_PROMPT,
    PROVIDER_URLS,
    LogLevel,
)
from photo_tagger.keywords import merge_keywords
from photo_tagger.logging_setup import setup_logging
from photo_tagger.metadata import (
    build_contextual_prompt,
    read_existing_keywords,
    read_gps_coordinates,
    read_location_tags,
    write_metadata,
)
from photo_tagger.models import GeneratedMetadata


# Cyclopts app
__version__ = "0.1.0"
app = App(
    name="photo-tagger",
    version=__version__,
)


def _validate_lmstudio_model(api_base_url: str, model_name: str, api_key: str | None) -> None:
    """Fail fast when LM Studio cannot resolve the requested model name."""
    url = urllib.parse.urljoin(api_base_url.rstrip("/") + "/", "models")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        logger.error("lmstudio_model_listing_invalid_scheme", url=url, scheme=parsed.scheme)
        raise SystemExit(1)
    if not parsed.netloc:
        logger.error("lmstudio_model_listing_missing_host", url=url)
        raise SystemExit(1)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.error("lmstudio_model_listing_error", error=str(exc), url=url)
        raise SystemExit(1) from exc

    if response.status_code != HTTPStatus.OK:
        logger.error(
            "lmstudio_model_listing_failed",
            status=response.status_code,
            url=url,
            body=response.text,
        )
        raise SystemExit(1)

    try:
        listing = response.json()
    except ValueError as exc:
        logger.error("lmstudio_model_listing_invalid_json", error=str(exc), url=url)
        raise SystemExit(1) from exc

    models = [
        str(entry["id"])
        for entry in listing.get("data", [])
        if isinstance(entry, dict) and "id" in entry
    ]

    if model_name not in models:
        logger.error(
            "lmstudio_model_not_available",
            requested=model_name,
            available=models,
        )
        raise SystemExit(1)

    logger.debug("lmstudio_model_validated", model=model_name)


def _pil_from_image_path(image_path: Path) -> Image.Image:
    """Open an image from a path with PIL, using rawpy unless format is known non-RAW."""
    non_raw_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".jpe",
        ".jp2",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
        ".psd",
        ".ico",
        ".ppm",
        ".pgm",
        ".pbm",
    }
    suffix = image_path.suffix.lower()
    if suffix in non_raw_exts:
        logger.info("skipping_rawpy_for_known_format", extension=suffix)
    else:
        try:
            with rawpy.imread(str(image_path)) as raw:  # type: ignore[no-untyped-call]
                rgb = raw.postprocess()  # 8-bit RGB np.ndarray
            logger.info("image_opened_with_rawpy")
            return Image.fromarray(rgb)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rawpy_failed_falling_back_to_pil", error=str(exc))

    logger.info("opening_image_with_pil", extension=suffix or "")
    return Image.open(image_path)


def prepare_image_for_agent(
    image_path: Path,
    jpg_quality: int = DEFAULT_JPEG_QUALITY,
    max_size: int = DEFAULT_DIMENSIONS,
) -> BinaryContent:
    """
    Prepare an image for Pydantic AI agent processing.

    Handles RAW (CR2, CR3, NEF, ARW, RW2, RAF, DNG) and standard (JPG, PNG, WEBP, BMP) formats.
    Converts RAW to RGB, resizes if needed, and returns as BinaryContent ready for the agent.
    The entire process is done in memory; no temporary files are created.

    Args:
        image_path: Path to the input image file
        jpg_quality: JPEG compression quality (1-100, recommended: 80)
        max_size: Maximum dimension in pixels for resizing (recommended: <=1280)

    Returns:
        BinaryContent object ready for Pydantic AI agent

    """
    try:
        img = _pil_from_image_path(image_path)

        # Composite alpha onto white background if present
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            logger.info("compositing_alpha_to_white")
            alpha = img.convert("RGBA")
            bg = Image.new("RGBA", alpha.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, alpha).convert("RGB")
        else:
            logger.info("converting_image_to_rgb")
            img = img.convert("RGB")

        # Resize in-place maintaining aspect ratio (downscale only) with high-quality filter
        logger.info("resizing_image", max_size=max_size)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Encode to JPEG bytes in memory
        logger.info("encoding_image_to_jpeg", quality=jpg_quality)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpg_quality)
        jpeg_bytes = buf.getvalue()

    except Exception as e:
        logger.exception("image_preparation_failed", error=str(e))
        raise
    else:
        logger.debug(
            "image_prepared_for_agent",
            width=img.width,
            height=img.height,
            size_kb=len(jpeg_bytes) // 1024,
        )
        # Return as BinaryContent for Pydantic AI
        return BinaryContent(data=jpeg_bytes, media_type="image/jpeg")


def analyze_image_with_ai(
    image_bytes: BinaryContent,
    agent: Agent,
    *,
    user_prompt: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, str, list[str]]:
    """
    Generate a short title, description and keywords using a vision-language model.

    Args:
        image_bytes: Image data as BinaryContent (JPEG format)
        agent: Configured Pydantic AI Agent with Ollama provider
        user_prompt: Optional user prompt with contextual metadata
        temperature: Sampling temperature for generation (0.0-1.0)
        max_tokens: Maximum tokens to generate in response (recommended: <=400)

    Returns:
        Tuple of (title, description, keywords) where:
        - title: Short title string (<= 64 chars)
        - description: Short description string (<= 160 chars)
        - keywords: List of keyword strings (Title Case, hierarchical with '<' if applicable)

    """
    logger.info("analyzing_image_with_ai")
    _t0 = time.perf_counter()
    prompt = user_prompt or DEFAULT_USER_PROMPT

    result: AgentRunResult[GeneratedMetadata] = agent.run_sync(
        [
            prompt,
            image_bytes,
        ],
        model_settings=ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        output_type=GeneratedMetadata,
    )
    _elapsed = time.perf_counter() - _t0
    logger.info(
        "ai_inference_completed",
        seconds=round(_elapsed, 3),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.debug(
        "ai_generated_metadata",
        title=result.output.title,
        description=result.output.description,
        keywords=result.output.keywords,
    )
    return result.output.title, result.output.description, result.output.keywords


def process_photo(
    image_path: Path,
    agent: Agent,
    *,
    preserve_existing_kw: bool = True,
    write_description: bool = True,
    write_title: bool = True,
    backup_xmp: bool = True,
    use_sidecar: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    jpeg_dimensions: int = DEFAULT_DIMENSIONS,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> bool:
    """
    Convert image to JPEG bytes in memory, analyze with AI, then write metadata to XMP or embed it.

    Args:
        image_path: Path to the image file to process
        agent: Configured Pydantic AI Agent with Ollama provider
        preserve_existing_kw: If True, merge with existing keywords rather than overwriting
        write_description: If True, also generate and write a short description
        write_title: If True, also generate and write a short title
        backup_xmp: If True, let ExifTool create _original backups when updating metadata
        use_sidecar: If True, write to an XMP sidecar; otherwise embed metadata in the image file
        temperature: Sampling temperature for generation (0.0-1.0)
        max_tokens: Maximum tokens to generate in the model response
        jpeg_dimensions: Max dimension in pixels for the resized JPEG sent to the model
        jpeg_quality: JPEG quality (1-100) for the image sent to the model

    Returns:
        True if entire workflow succeeded, False if any step failed

    Note:
        - No temporary files are created; all conversion happens in memory
        - Supports Lightroom hierarchical keyword format (pipe separators)

    """
    logger.info("processing_photo")

    # Step 1: Prepare image (convert to JPEG bytes)
    jpeg_bytes = prepare_image_for_agent(
        image_path,
        jpg_quality=jpeg_quality,
        max_size=jpeg_dimensions,
    )

    # Step 2: Gather existing metadata context
    existing_keywords_full = read_existing_keywords(image_path)
    if any(existing_keywords_full.values()):
        logger.info(
            "existing_keywords_found",
            count=len(existing_keywords_full["subject"]),
        )

    location_tags = read_location_tags(image_path)
    gps_info = read_gps_coordinates(image_path)

    # Step 3: Generate metadata with a concise contextual prompt
    unique_flat_keywords = list(dict.fromkeys(existing_keywords_full.get("subject", [])))
    contextual_prompt = build_contextual_prompt(
        DEFAULT_USER_PROMPT,
        unique_flat_keywords,
        location_tags,
        gps_info,
    )

    title, description, keywords = analyze_image_with_ai(
        image_bytes=jpeg_bytes,
        agent=agent,
        user_prompt=contextual_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Step 4: Prepare keywords for merging/writing
    existing_keywords: dict[str, list[str]]
    if preserve_existing_kw:
        existing_keywords = existing_keywords_full
    else:
        existing_keywords = {
            "subject": [],
            "hierarchical": [],
            "weighted": [],
        }

    # Step 5: Merge AI-generated keywords with existing keywords
    merged_keywords = merge_keywords(existing_keywords, keywords)

    # Step 6: Persist metadata (sidecar or embedded)
    return write_metadata(
        image_path,
        merged_keywords,
        description=description if write_description else None,
        title=title if write_title else None,
        backup=backup_xmp,
        use_sidecar=use_sidecar,
    )


def _execute_process(
    image_file: Path,
    *,
    agent: Agent,
    preserve_existing_kw: bool,
    write_description: bool,
    write_title: bool,
    backup_xmp: bool,
    use_sidecar: bool,
    temperature: float,
    max_tokens: int,
    jpeg_dimensions: int,
    jpeg_quality: int,
    index: str,
    retry: bool = False,
) -> bool:
    """Run process_photo once with consistent logging and error handling."""
    context_kwargs: dict[str, Any] = {"file": image_file.name}
    if retry:
        context_kwargs["retry"] = True

    with logger.contextualize(**context_kwargs):
        try:
            ok = process_photo(
                image_file,
                agent=agent,
                preserve_existing_kw=preserve_existing_kw,
                write_description=write_description,
                write_title=write_title,
                backup_xmp=backup_xmp,
                use_sidecar=use_sidecar,
                temperature=temperature,
                max_tokens=max_tokens,
                jpeg_dimensions=jpeg_dimensions,
                jpeg_quality=jpeg_quality,
            )
        except Exception as exc:  # noqa: BLE001
            event = "processing_retry_exception" if retry else "processing_exception"
            logger.exception(event, error=str(exc))
            return False

        if ok:
            event = "retry_success" if retry else "processing_success"
            logger.info(event, index=index)
            return True

        event = "retry_failed" if retry else "processing_failed"
        if retry:
            logger.error(event, index=index)
        else:
            logger.error(event, index=index, queued_for_retry=True)
        return False


def _parse_extensions(image_extensions: str) -> set[str]:
    """
    Normalize comma-separated extensions into a set like {".cr3", ".jpg"}.

    Examples:
        >>> _parse_extensions("cr3, jpg ,PNG")
        {'.cr3', '.jpg', '.PNG'}

    """
    return {
        f".{ext.strip().lstrip('.')}"
        for ext in image_extensions.split(",")
        if ext.strip().lstrip(".")
    }


def _resolve_image_files(
    inputs: list[Path],
    ext_set: set[str],
    *,
    recursive: bool,
) -> list[Path]:
    """
    Resolve provided inputs into a list of files.

    - Directories are expanded by extension (honoring --recursive)
    - Explicit files are accepted as-is (extension filter not applied)
    - Order is preserved and duplicates removed
    """
    pattern = "**/*" if recursive else "*"

    files_from_dirs: list[Path] = []
    files_explicit: list[Path] = []

    for path in inputs:
        path_resolved = path
        with contextlib.suppress(Exception):
            path_resolved = path.resolve()
        if path_resolved.is_dir():
            for ext in ext_set:
                files_from_dirs.extend(path_resolved.glob(f"{pattern}{ext}"))
        elif path_resolved.is_file():
            files_explicit.append(path_resolved)
        else:
            logger.warning("input_not_file_or_dir", path=str(path))

    combined: list[Path] = []
    seen = set()
    for f in chain(files_explicit, files_from_dirs):
        key = str(f.resolve()) if f.exists() else str(f)
        if key not in seen:
            combined.append(f)
            seen.add(key)

    return combined


def _resolve_image_batch(
    inputs: list[Path] | None,
    image_extensions: str,
    *,
    recursive: bool,
) -> list[Path]:
    ext_set = _parse_extensions(image_extensions)
    if not ext_set:
        logger.error("no_valid_extensions_provided", raw_input=image_extensions)
        raise SystemExit(1)
    logger.debug("parsed_extensions", extensions=sorted(ext_set))

    if not inputs:
        logger.error(
            "no_inputs_provided",
            hint=("Pass one or more --input/-i paths (files or directories)"),
        )
        raise SystemExit(1)

    image_files = _resolve_image_files(inputs, ext_set, recursive=recursive)
    if not image_files:
        logger.error(
            "no_image_files_found",
            inputs=[str(p) for p in inputs],
            recursive=recursive,
            extensions=sorted(ext_set),
        )
        raise SystemExit(1)

    logger.info("image_files_discovered", count=len(image_files))
    return image_files


def _load_skip_list(skip_file: Path) -> set[str]:
    try:
        content = skip_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("skip_file_read_failed", file=str(skip_file), error=str(exc))
        raise SystemExit(1) from exc

    entries: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.add(stripped)

    if entries:
        logger.info("skip_entries_loaded", count=len(entries), file=str(skip_file))
    else:
        logger.warning("skip_file_has_no_entries", file=str(skip_file))
    return entries


def _filter_skipped_files(
    image_files: list[Path],
    skip_entries: set[str],
) -> tuple[list[Path], int]:
    if not skip_entries:
        return image_files, 0

    name_keys = {
        entry.casefold() for entry in skip_entries if os.sep not in entry and "/" not in entry
    }
    path_keys = {entry.casefold() for entry in skip_entries if entry.casefold() not in name_keys}

    filtered: list[Path] = []
    skipped = 0
    for path in image_files:
        name_key = path.name.casefold()
        path_key = str(path).casefold()
        if name_key in name_keys or path_key in path_keys:
            logger.debug("skipping_file_from_list", file=str(path))
            skipped += 1
            continue
        filtered.append(path)

    return filtered, skipped


def _apply_skip_file(
    image_files: list[Path],
    skip_file: Path | None,
) -> list[Path]:
    if not skip_file:
        return image_files

    skip_entries = _load_skip_list(skip_file)
    filtered, skipped = _filter_skipped_files(image_files, skip_entries)
    if skipped:
        logger.info(
            "skip_list_applied",
            skipped=skipped,
            remaining=len(filtered),
            file=str(skip_file),
        )
    elif skip_entries:
        logger.warning("skip_list_matched_no_files", file=str(skip_file))
    return filtered


def _create_agent(
    provider_name: Literal["ollama", "lmstudio"],
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
    retries: int,
) -> Agent:
    resolved_url = api_base_url or PROVIDER_URLS.get(provider_name, DEFAULT_OLLAMA_BASE_URL)
    if api_base_url is None:
        logger.debug("using_default_provider_url", url=resolved_url)

    logger.info(
        "provider_config_resolved",
        provider=provider_name,
        url=resolved_url,
        model=model_name,
    )
    logger.debug("setting_up_llm_agent", provider=provider_name, url=resolved_url, model=model_name)

    if provider_name == "ollama":
        resolved_api_key = api_key or DEFAULT_OLLAMA_API_KEY
        provider = OllamaProvider(base_url=resolved_url, api_key=resolved_api_key)
    else:
        resolved_api_key = api_key or DEFAULT_LMSTUDIO_API_KEY
        _validate_lmstudio_model(resolved_url, model_name, resolved_api_key)
        provider = OpenAIProvider(base_url=resolved_url, api_key=resolved_api_key)

    chat_model = OpenAIChatModel(model_name=model_name, provider=provider)
    return Agent(
        chat_model,
        output_type=GeneratedMetadata,  # type: ignore[arg-type]
        retries=retries,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
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
        Parameter(
            name=("--model", "-m"),
            help="Vision-language model name",
        ),
    ] = DEFAULT_MODEL_NAME,
    provider_name: Annotated[
        Literal["ollama", "lmstudio"],
        Parameter(
            name=("--provider",),
            help="Backend provider: 'ollama' or 'lmstudio'",
        ),
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
        Parameter(
            name=("--temperature",),
            help="Sampling temperature (0.0-1.0)",
        ),
    ] = DEFAULT_TEMPERATURE,
    max_tokens: Annotated[
        int,
        Parameter(
            name=("--max-tokens",),
            help="Maximum tokens to generate",
        ),
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
        Parameter(
            name=("--retries",),
            help="Number of automatic validation retries",
        ),
    ] = DEFAULT_RETRIES,
    file_log_level: Annotated[
        LogLevel,
        Parameter(
            name="--file-log-level",
            help="Log level for file (use 'OFF' to disable)",
        ),
    ] = "DEBUG",
    log_folder: Annotated[
        Path,
        Parameter(
            name=("--log-folder",),
            help="Folder where log files are stored",
        ),
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
    # Setup logging
    setup_logging(
        file_log_level=file_log_level,
        console_log_level=console_log_level,
        log_folder=log_folder,
    )
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
        preserve_keywords=preserve_keywords,
        write_description=write_description,
        write_title=write_title,
        backup_xmp=backup_xmp,
        use_sidecar=use_sidecar,
        temperature=temperature,
        max_tokens=max_tokens,
        jpeg_dimensions=jpeg_dimensions,
        jpeg_quality=jpeg_quality,
        retries=retries,
        log_folder=str(log_folder),
    )

    image_files = _resolve_image_batch(inputs, image_extensions, recursive=recursive)
    image_files = _apply_skip_file(image_files, skip_from)
    if not image_files:
        logger.info("no_files_to_process_after_skipping")
        return
    file_count = len(image_files)

    agent = _create_agent(
        provider_name,
        model_name,
        api_base_url=api_base_url,
        api_key=api_key,
        retries=retries,
    )

    process_kwargs: dict[str, Any] = {
        "agent": agent,
        "preserve_existing_kw": preserve_keywords,
        "write_description": write_description,
        "write_title": write_title,
        "backup_xmp": backup_xmp,
        "use_sidecar": use_sidecar,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "jpeg_dimensions": jpeg_dimensions,
        "jpeg_quality": jpeg_quality,
    }

    # Process each file
    success_count = 0
    pending_failures: list[Path] = []
    for idx, image_file in enumerate(image_files, start=1):
        index = f"{idx}/{file_count}"
        if _execute_process(
            image_file,
            index=index,
            **process_kwargs,
        ):
            success_count += 1
        else:
            logger.warning("file_queued_for_retry", file=image_file.name)
            pending_failures.append(image_file)

    # Retry failed files
    initial_failures = len(pending_failures)
    retry_successes = 0
    if pending_failures:
        logger.info("retrying_failed_files", count=initial_failures)
        remaining_failures: list[Path] = []
        for idx, image_file in enumerate(pending_failures, start=1):
            index = f"{idx}/{initial_failures}"
            if _execute_process(
                image_file,
                index=index,
                retry=True,
                **process_kwargs,
            ):
                success_count += 1
                retry_successes += 1
            else:
                logger.error("file_failed_after_retry", file=image_file.name)
                remaining_failures.append(image_file)
        pending_failures = remaining_failures

    # Summary
    failed_count = len(pending_failures)
    logger.info(
        "processing_summary",
        total_files=file_count,
        successful=success_count,
        failed=failed_count,
        initial_failures=initial_failures,
        retry_successes=retry_successes,
    )
    if pending_failures:
        logger.error(
            "files_failed_after_retry",
            files=[str(path) for path in pending_failures],
        )

    if success_count < file_count:
        raise SystemExit(1)


if __name__ == "__main__":
    app()
