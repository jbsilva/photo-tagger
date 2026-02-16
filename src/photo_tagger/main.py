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
import sys
import time
import urllib.parse
from datetime import UTC, datetime
from http import HTTPStatus
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import httpx
import rawpy
from cyclopts import App, Parameter, validators
from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from exiftool.exceptions import ExifToolExecuteError
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, BinaryContent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider


if TYPE_CHECKING:
    from collections.abc import Iterable


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]
MIN_HIERARCHICAL_DEPTH = 2

# Configuration defaults
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-vl-30b")
DEFAULT_JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
DEFAULT_DIMENSIONS = int(os.getenv("JPEG_DIMENSIONS", "1280"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
DEFAULT_RETRIES = int(os.getenv("RETRIES", "5"))
PROVIDER_URLS = {
    "ollama": DEFAULT_OLLAMA_BASE_URL,
    "lmstudio": DEFAULT_LMSTUDIO_BASE_URL,
}
LOCATION_TAGS = (
    "XMP-photoshop:Country",
    "IPTC:Country-PrimaryLocationName",
    "XMP-photoshop:City",
    "IPTC:City",
)

# Prompt templates
DEFAULT_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "**Persona**: You are a specialist AI photo archivist named 'Metis'. "
    "Your expertise is in analyzing visual information and creating rich, structured metadata.\n"
    "\n"
    "**Mission**: Your mission is to meticulously analyze the provided image and generate a "
    "complete metadata object. The output must strictly conform to the Pydantic schema provided by "
    "the user.\n"
    "\n"
    "**Process**:\n"
    "1.  **Analyze**: Perform a comprehensive visual analysis of the image. Identify the primary "
    "subject, setting, composition, colors, and emotional tone.\n"
    "2.  **Generate Title**: Create a short, descriptive title (under 10 words).\n"
    "3.  **Generate Description**: Write a single, concise sentence (15-25 words) that captures "
    "the essence of the scene.\n"
    "4.  **Generate Keywords**:\n"
    "    *   **Identify Core Concepts**: Brainstorm a list of all identifiable elements: subjects, "
    "objects, environment, actions, mood, and artistic style.\n"
    "    *   **Format and Refine**: Convert these concepts into 10-15 keywords. Each keyword must "
    "be in Title Case.\n"
    "    *   **Build Hierarchies**: For relevant keywords, construct a logical hierarchy from "
    "specific to general using '<' as a separator (e.g., 'Golden Eagle<Bird of Prey<Animal'). "
    "Do not exceed 5 levels.\n"
    "5.  **Final Output**: Assemble the title, description, and keywords into a single, structured "
    "response. Ensure all constraints are met before finalizing.\n"
    "<|im_end|>"
)

DEFAULT_USER_PROMPT = (
    "Execute your mission: analyze this image and generate the structured metadata."
)


# Cyclopts app
__version__ = "0.1.0"
app = App(
    name="photo-tagger",
    version=__version__,
)


def setup_logging(
    file_log_level: LogLevel = "DEBUG",
    console_log_level: LogLevel = "INFO",
    log_folder: Path = Path("logs"),
) -> None:
    """
    Configure Loguru for both console and file logging.

    Args:
        file_log_level: Log level for file (use 'OFF' to disable)
        console_log_level: Log level for console (use 'OFF' to disable)
        log_folder: Directory where log files are stored

    """
    # Remove default handler
    logger.remove()

    # Add file logging
    if file_log_level != "OFF":
        log_folder.mkdir(parents=True, exist_ok=True)
        log_file = log_folder / Path(
            datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S-photo_tagger.log"),
        )
        logger.add(
            log_file,
            level=file_log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name:<8}:{function:<25}:{line:>4} | "
                "{message:<40} | "
                "{extra}"
            ),
            rotation="500 MB",
            retention="10 days",
            compression="zip",
        )

    # Add console logging
    if console_log_level != "OFF":
        logger.add(
            sys.stderr,
            level=console_log_level,
            colorize=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <7}</level> | "
                "<level>{message:<40.50}</level> | "
                "<yellow>{extra}</yellow>"
            ),
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


def parse_hierarchical_keyword(keyword: str) -> tuple[str, list[str]]:
    """
    Parse a hierarchical keyword in format 'Child<Parent<Grandparent' to Lightroom format.

    Args:
        keyword: Keyword string, either flat or hierarchical with '<' separators.
                 Example: "Duck<Bird<Animal" or just "Landscape"

    Returns:
        Tuple of (hierarchical_format, flat_list) where:
        - hierarchical_format: "Grandparent|Parent|Child" (Lightroom format)
        - flat_list: ["Grandparent", "Parent", "Child"] (all levels as separate keywords)

    Examples:
        >>> parse_hierarchical_keyword("Duck<Bird<Animal")
        ("Animal|Bird|Duck", ["Animal", "Bird", "Duck"])
        >>> parse_hierarchical_keyword("Landscape")
        ("Landscape", ["Landscape"])

    """
    keyword = keyword.strip()

    if not keyword:
        return ("", [""])

    # Remove stray '>' characters that occasionally appear in model output.
    sanitized = keyword.replace(">", "")

    if "<" not in sanitized:
        # Flat keyword
        return (sanitized, [sanitized])

    # Parse hierarchical keyword: "Duck<Bird<Animal" -> ["Duck", "Bird", "Animal"]
    parts = [p.strip() for p in sanitized.split("<") if p.strip()]
    if not parts:
        return ("", [])

    # Reverse to get root-to-leaf order: ["Animal", "Bird", "Duck"]
    parts_reversed = list(reversed(parts))

    # Create hierarchical format with pipes: "Animal|Bird|Duck"
    hierarchical = "|".join(parts_reversed)

    return (hierarchical, parts_reversed)


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


class GeneratedMetadata(BaseModel):
    """Schema for structured generation results."""

    title: str
    description: str
    keywords: list[str] = Field(default_factory=list)


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


def read_existing_keywords(image_path: Path) -> dict[str, list[str]]:
    """
    Read existing keywords from either the image or its XMP sidecar using pyexiftool.

    Args:
        image_path: Path to the image file. Reads embedded metadata and any adjacent XMP file.

    Returns:
        Dictionary with three keys:
        - 'subject': Flat keywords aggregated from XMP-dc:Subject and IPTC:Keywords
        - 'hierarchical': List of hierarchical keywords from XMP-lr:HierarchicalSubject
        - 'weighted': List of flat keywords from XMP-lr:WeightedFlatSubject

    Examples:
        >>> read_existing_keywords(Path("/tmp/photo.cr3"))
        {'subject': [], 'hierarchical': [], 'weighted': []}

    Note:
        Returns empty lists if neither the primary file nor its sidecar contain keywords.

    """
    targets = _metadata_targets(image_path)

    # Initialize empty result
    result: dict[str, list[str]] = {"subject": [], "hierarchical": [], "weighted": []}
    if not targets:
        logger.info("no_metadata_targets_found")
        return result

    tags_to_extract = [
        "XMP:Subject",
        "XMP:HierarchicalSubject",
        "XMP:WeightedFlatSubject",
        "IPTC:Keywords",
    ]

    # Map the full tag names to the desired short keys for the result dictionary
    key_map = {
        "XMP:Subject": "subject",
        "XMP:HierarchicalSubject": "hierarchical",
        "XMP:WeightedFlatSubject": "weighted",
        "IPTC:Keywords": "subject",
    }

    iptc_keywords_added = 0

    try:
        with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
            metadata_blocks = et.get_tags(files=targets, tags=tags_to_extract)
    except (ValueError, TypeError, ExifToolExecuteError) as e:
        logger.exception("failed_to_read_existing_keywords", error=str(e))
        return result

    for block in metadata_blocks:
        for exif_key, result_key in key_map.items():
            if exif_key not in block:
                continue
            raw_value = block[exif_key]
            if isinstance(raw_value, (list, tuple, set)):
                values = [str(item) for item in raw_value if str(item).strip()]
            else:
                values = [str(raw_value)] if str(raw_value).strip() else []
            result[result_key].extend(values)
            if exif_key == "IPTC:Keywords":
                iptc_keywords_added += len(values)

    for key, values in result.items():
        if values:
            result[key] = list(dict.fromkeys(values))

    logger.debug(
        "existing_keywords_read",
        subject_count=len(result.get("subject", [])),
        hierarchical_count=len(result.get("hierarchical", [])),
        weighted_count=len(result.get("weighted", [])),
        iptc_keywords_count=iptc_keywords_added,
    )
    return result


def _format_metadata_value(value: Any) -> str:  # noqa: ANN401
    """
    Coerce metadata values (lists, numbers) into a readable string.

    Examples:
        >>> _format_metadata_value(["sky", "", None])
        'sky'
        >>> _format_metadata_value(42)
        '42'

    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value if str(v).strip())
    return str(value)


def _metadata_targets(image_path: Path) -> list[str]:
    """
    Return the file paths that may contain metadata for an image.

    Examples:
        >>> _metadata_targets(Path("/photos/image.cr3"))  # doctest: +SKIP
        ['/photos/image.cr3', '/photos/image.xmp']

    """
    targets: list[str] = []
    xmp_path = image_path.with_suffix(".xmp")
    if image_path.exists():
        targets.append(str(image_path))
    if xmp_path.exists():
        targets.append(str(xmp_path))
    return targets


def read_location_tags(image_path: Path) -> dict[str, str]:
    """
    Read selected IPTC/XMP location tags from the image or its sidecar.

    Examples:
        >>> read_location_tags(Path("/photos/image.cr3"))  # doctest: +SKIP
        {'XMP-photoshop:Country': 'Japan',
         'XMP-photoshop:City': 'Kyoto'}

    """
    targets = _metadata_targets(image_path)
    if not targets:
        return {}

    collected: dict[str, str] = {}
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        try:
            metadata_blocks = et.get_tags(files=targets, tags=list(LOCATION_TAGS))
        except (ValueError, TypeError, ExifToolExecuteError) as e:
            logger.exception("failed_to_read_location_tags", error=str(e))
            return {}

    for block in metadata_blocks:
        for tag in LOCATION_TAGS:
            if tag in block and block[tag] not in (None, ""):
                collected[tag] = _format_metadata_value(block[tag])

    if collected:
        logger.debug("location_tags_read", tags=collected)

    return collected


def read_gps_coordinates(image_path: Path) -> dict[str, str]:
    """
    Read GPS coordinates from either the primary file or its XMP sidecar.

    Examples:
        >>> read_gps_coordinates(Path("/photos/image.cr3"))  # doctest: +SKIP
        {'position': '35 deg N, 135 deg E'}

    """
    targets = _metadata_targets(image_path)
    if not targets:
        return {}

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        try:
            metadata_blocks = et.get_tags(files=targets, tags=["Composite:GPSPosition"])
        except (ValueError, TypeError, ExifToolExecuteError) as e:
            logger.exception("failed_to_read_gps", error=str(e))
            return {}

    for block in metadata_blocks:
        value = block.get("Composite:GPSPosition")
        if value not in (None, ""):
            position = _format_metadata_value(value)
            logger.debug("gps_position_read", position=position)
            return {"position": position}

    return {}


def build_contextual_prompt(
    base_prompt: str,
    flat_keywords: list[str],
    location_tags: dict[str, str],
    gps_info: dict[str, str],
    max_prompt_flat_keywords: int = 4,
) -> str:
    """
    Create a concise user prompt that highlights key existing metadata.

    Examples:
        >>> print(
        ...     build_contextual_prompt(
        ...         "Analyze.",
        ...         ["Beach", "Sunset", "Travel"],
        ...         {"XMP-photoshop:Country": "Japan"},
        ...         {"position": "35 deg N, 135 deg E"},
        ...     )
        ... )  # doctest: +SKIP
        Analyze.

        Existing Metadata:
        - Existing Keywords: Beach, Sunset, Travel
        - Location: Japan
        - GPS: 35 deg N, 135 deg E

    """
    metadata_lines: list[str] = []

    if unique_keywords := list(dict.fromkeys(flat_keywords)):
        selected = unique_keywords[:max_prompt_flat_keywords]
        keyword_line = "- Existing Keywords: " + ", ".join(selected)
        if len(unique_keywords) > max_prompt_flat_keywords:
            keyword_line += ", ..."
        metadata_lines.append(keyword_line)

    if location_value := next(
        (location_tags.get(tag) for tag in LOCATION_TAGS if location_tags.get(tag)),
        None,
    ):
        metadata_lines.append(f"- Location: {location_value}")

    if gps_line := (gps_info or {}).get("position"):
        metadata_lines.append(f"- GPS: {gps_line}")

    sections = [base_prompt.strip()]
    if metadata_lines:
        sections.append("Existing Metadata:\n" + "\n".join(metadata_lines))
    return "\n\n".join(sections)


def _normalize_chain_parts(parts: Iterable[str]) -> list[str]:
    """
    Return Title Case chain segments, skipping blanks.

    Examples:
        >>> _normalize_chain_parts([" duck ", "Bird", ""])
        ['Duck', 'Bird']

    """
    return [segment.strip().title() for segment in parts if segment and segment.strip()]


def _register_chain(
    registry: dict[str, list[str]],
    chain: list[str],
) -> None:
    """
    Keep the longest chain for each leaf.

    Examples:
        >>> reg: dict[str, list[str]] = {}
        >>> _register_chain(reg, ["Animal", "Bird"])
        >>> reg["bird"]
        ['Animal', 'Bird']

    """
    if len(chain) < MIN_HIERARCHICAL_DEPTH:
        return
    leaf_key = chain[-1].casefold()
    current = registry.get(leaf_key)
    if current is None or len(chain) > len(current):
        registry[leaf_key] = chain


def _seed_longest_from_existing(hierarchical_keywords: Iterable[str]) -> dict[str, list[str]]:
    """
    Prime the longest-chain registry with existing hierarchical entries.

    Examples:
        >>> _seed_longest_from_existing(["Animal|Bird", "Plant"])
        {'bird': ['Animal', 'Bird']}

    """
    registry: dict[str, list[str]] = {}
    for entry in hierarchical_keywords:
        normalized = _normalize_chain_parts(entry.split("|"))
        _register_chain(registry, normalized)
    return registry


def _process_new_keywords(
    new_keywords: list[str],
    subject_seen: set[str],
    subject_acc: list[str],
    weighted_acc: list[str],
    chain_registry: dict[str, list[str]],
) -> list[str]:
    """
    Append new flat keywords and update the longest-chain registry.

    Mutates: subject_seen, subject_acc, weighted_acc, chain_registry.

    Args:
        new_keywords: Flat subjects (e.g., "bird") or chains (e.g., "Duck<Bird<Animal").
        subject_seen: Casefolded set for de-duplication.
        subject_acc: Accumulates unique subjects (root→leaf order).
        weighted_acc: Parallel accumulator kept in sync with subject_acc.
        chain_registry: Maps leaf key (casefolded) to longest observed chain (root→leaf list).

    Returns:
        Subjects appended during this call, in append order.

    Examples:
        >>> subjects = []; weighted = []; seen = set(); registry = {}
        >>> _process_new_keywords([
        ...     "Duck<Bird<Animal",
        ...     "bird",
        ... ], seen, subjects, weighted, registry)
        ['Animal', 'Bird', 'Duck']
        >>> subjects
        ['Animal', 'Bird', 'Duck']
        >>> registry['duck']
        ['Animal', 'Bird', 'Duck']

    """
    added_subjects: list[str] = []
    for keyword in new_keywords:
        _, parts = parse_hierarchical_keyword(keyword)
        normalized = _normalize_chain_parts(parts)
        if not normalized:
            continue
        for flat_kw in normalized:
            key = flat_kw.casefold()
            if key not in subject_seen:
                subject_acc.append(flat_kw)
                weighted_acc.append(flat_kw)
                added_subjects.append(flat_kw)
                subject_seen.add(key)
        _register_chain(chain_registry, normalized)
    return added_subjects


def _collect_cumulative_entries(
    chain_registry: dict[str, list[str]],
    hierarchical_seen: set[str],
) -> list[str]:
    """
    Generate Lightroom hierarchy paths from canonical chains.

    Lightroom writes hierarchical keywords as full pipe-separated paths in lr:HierarchicalSubject.
    You cannot add only "Animal|Bird|Duck", you must also add "Animal|Bird".

    Mutates: hierarchical_seen.

    Args:
        chain_registry: Maps each leaf keyword to its full root-to-leaf path.
        hierarchical_seen: Casefolded set for de-duplicating cumulative paths

    Returns:
        New cumulative paths like "A|B", "A|B|C", starting at MIN_HIERARCHICAL_DEPTH.

    Examples:
        >>> chains = {'duck': ['Animal', 'Bird', 'Duck']}
        >>> _collect_cumulative_entries(chains, set())
        ['Animal|Bird', 'Animal|Bird|Duck']

    """
    additions: list[str] = []
    for canonical_chain in chain_registry.values():
        for depth in range(MIN_HIERARCHICAL_DEPTH, len(canonical_chain) + 1):
            cumulative = "|".join(canonical_chain[:depth])
            key = cumulative.casefold()
            if key not in hierarchical_seen:
                hierarchical_seen.add(key)
                additions.append(cumulative)
    return additions


def merge_keywords(
    existing_kw: dict[str, list[str]],
    new_keywords: list[str],
) -> dict[str, list[str]]:
    """
    Merge new AI-generated keywords with existing keywords, preserving hierarchy.

    Args:
        existing_kw: Dictionary of existing keywords from read_existing_keywords()
        new_keywords: List of new keywords from AI (may include hierarchical format)

    Returns:
        Dictionary with merged keywords in three formats:
        - 'subject': All flat keywords (existing + new flattened)
        - 'hierarchical': Hierarchical keywords (existing + new hierarchical)
        - 'weighted': Weighted flat keywords (mirrors subject)

    Note:
        - Duplicate detection is case-insensitive (using casefold)
        - Hierarchical keywords are flattened for Subject/WeightedFlatSubject
        - Original hierarchy is preserved in HierarchicalSubject

    Examples:
        >>> existing = {
        ...     "subject": ["Beach"],
        ...     "hierarchical": ["Animal|Bird"],
        ...     "weighted": ["Beach"],
        ... }
        >>> merge_keywords(existing, ["Seagull<Bird<Animal", "bird"])
        {'subject': ['Beach', 'Animal', 'Bird', 'Seagull'],
         'hierarchical': ['Animal|Bird', 'Animal|Bird|Seagull'],
         'weighted': ['Beach', 'Animal', 'Bird', 'Seagull']}

    """
    # Copy existing lists so we never mutate caller-owned structures.
    existing_subject = list(existing_kw.get("subject", []))
    existing_weighted = list(existing_kw.get("weighted", []))
    existing_hierarchical = [kw for kw in existing_kw.get("hierarchical", []) if "|" in kw]

    subject_seen = {kw.casefold() for kw in existing_subject}
    hierarchical_seen = {kw.casefold() for kw in existing_hierarchical}

    chain_registry = _seed_longest_from_existing(existing_hierarchical)
    new_subjects = _process_new_keywords(
        new_keywords,
        subject_seen,
        existing_subject,
        existing_weighted,
        chain_registry,
    )
    new_hierarchical = _collect_cumulative_entries(chain_registry, hierarchical_seen)

    merged = {
        "subject": existing_subject,
        "hierarchical": existing_hierarchical + new_hierarchical,
        "weighted": existing_weighted,
    }

    logger.debug(
        "keywords_merged",
        new_flat_count=len(new_subjects),
        new_hierarchical_count=len(new_hierarchical),
        total_flat=len(merged["subject"]),
        total_hierarchical=len(merged["hierarchical"]),
    )

    return merged


def write_metadata(
    image_path: Path,
    keywords: dict[str, list[str]],
    *,
    description: str | None = None,
    title: str | None = None,
    backup: bool = True,
    use_sidecar: bool = True,
) -> bool:
    """
    Write tags to either the image file or an XMP sidecar in Lightroom-compatible format.

    Args:
        image_path: Path to the image file. Sidecar will share the name, but with a .xmp extension
        keywords: Dictionary with 'subject' and 'hierarchical' keyword lists (optionally weighted).
        description: Optional short description to write to XMP (and ImageDescription)
        title: Optional short title to write to XMP-dc:Title and IPTC:ObjectName
        backup: If True, let ExifTool create a backup (_original suffix when applicable)
        use_sidecar: If True, write metadata to a sidecar; otherwise embed it in the source file

    Returns:
        True if the metadata write succeeded, False otherwise.

    References:
        - Lightroom XMP format: https://www.adobe.com/products/xmp.html
        - MWG Guidelines: http://www.metadataworkinggroup.org/

    """
    target_path = image_path.with_suffix(".xmp") if use_sidecar else image_path
    tags_to_write: dict[str, str | list[str]] = {}

    # Build a dictionary of tags to write.
    if keywords.get("subject"):
        tags_to_write["XMP-dc:Subject"] = keywords["subject"]
        # Lightroom prioritizes IPTC:Keywords for JPEGs, so mirror the Subject list there.
        tags_to_write["IPTC:Keywords"] = keywords["subject"]
    if keywords.get("hierarchical"):
        tags_to_write["XMP-lr:HierarchicalSubject"] = keywords["hierarchical"]
    if description:
        tags_to_write["XMP-dc:Description"] = description
        tags_to_write["XMP-exif:ImageDescription"] = description
    if title:
        tags_to_write["XMP-dc:Title"] = title
        tags_to_write["IPTC:ObjectName"] = title

    if not tags_to_write:
        logger.warning("no_data_to_write", file=image_path.name)
        return False

    try:
        with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
            if backup:
                et.set_tags(tags=tags_to_write, files=[str(target_path)])
            else:
                et.set_tags(
                    tags=tags_to_write,
                    files=[str(target_path)],
                    params=["-overwrite_original"],
                )
    except (ValueError, TypeError, ExifToolExecuteError) as e:
        logger.exception("xmp_write_failed", error=str(e), target=str(target_path))
        return False
    else:
        logger.info(
            "metadata_written_successfully",
            target=str(target_path),
            mode="sidecar" if use_sidecar else "embedded",
            subject_keywords=len(keywords.get("subject", [])),
            hierarchical_keywords=len(keywords.get("hierarchical", [])),
            backup_created=backup,
        )
        return True


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
    - Ollama server running and reachable (with a vision-language model).

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
        photo-tagger tag -i ./photos/IMG_0001.CR3
        photo-tagger tag -i ./photos --ext cr3,jpg -r
        photo-tagger tag -i ./photos -i ./photos/IMG_0001.CR3 --ext cr3,jpg

    """
    # Setup logging
    setup_logging(file_log_level=file_log_level, console_log_level=console_log_level)
    logger.info(
        "starting_ai_photo_tagger",
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
