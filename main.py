#!/usr/bin/env python3
# ruff: noqa: PLR0913
"""
AI Photo Tagger: CLI app to describe photos and add keywords using AI.

Non-destructive: create/update XMP sidecar files with Lightroom-compatible tags.

Requirements:
 - Exiftool installed and available in PATH.
 - Ollama server running and containing a vision-language model.

Tested with Canon CR3 RAW files using Qwen3-VL:32b (Ollama on MacBook M4 Max with 128 GB RAM), but
should work with different image formats and vision-language models.
"""

import contextlib
import os
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal

import exiftool
import rawpy
from cyclopts import App, Parameter, validators
from exiftool.exceptions import ExifToolExecuteError
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, BinaryContent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]
MIN_HIERARCHICAL_DEPTH = 2


# Configuration defaults
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-vl:32b")
DEFAULT_JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
DEFAULT_DIMENSIONS = int(os.getenv("JPEG_DIMENSIONS", "1280"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
DEFAULT_RETRIES = int(os.getenv("RETRIES", "2"))
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


def setup_logging(file_log_level: LogLevel = "DEBUG", console_log_level: LogLevel = "INFO") -> None:
    """
    Configure Loguru for both console and file logging.

    Args:
        file_log_level: Log level for file (use 'OFF' to disable)
        console_log_level: Log level for console (use 'OFF' to disable)

    """
    # Remove default handler
    logger.remove()

    # Add file logging
    if file_log_level != "OFF":
        log_file = Path(datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S-ai_photo_tagger.log"))
        logger.add(
            log_file,
            level=file_log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message} | "
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
                "<level>{level: <8}</level> | <level>{message}</level> | <yellow>{extra}</yellow>"
            ),
        )


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

    if "<" not in keyword:
        # Flat keyword
        return (keyword, [keyword])

    # Parse hierarchical keyword: "Duck<Bird<Animal" -> ["Duck", "Bird", "Animal"]
    parts = [p.strip() for p in keyword.split("<")]

    # Reverse to get root-to-leaf order: ["Animal", "Bird", "Duck"]
    parts_reversed = list(reversed(parts))

    # Create hierarchical format with pipes: "Animal|Bird|Duck"
    hierarchical = "|".join(parts_reversed)

    return (hierarchical, parts_reversed)


def _pil_from_image_path(image_path: Path) -> Image.Image:
    """Open an image from a path with PIL, handling RAW via rawpy when needed."""
    try:
        # Try RAW first
        with rawpy.imread(str(image_path)) as raw:
            rgb = raw.postprocess()  # 8-bit RGB np.ndarray
        logger.info("image_opened_with_rawpy")
        return Image.fromarray(rgb)
    except Exception as e:  # noqa: BLE001
        # Fallback to PIL's open for non-RAW formats
        logger.warning("rawpy_failed_falling_back_to_pil", error=str(e))
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
    return result.output.title, result.output.description, result.output.keywords


def read_existing_keywords(image_path: Path) -> dict[str, list[str]]:
    """
    Read existing keywords from XMP sidecar using pyexiftool.

    Args:
        image_path: Path to the image file. Will check for adjacent .xmp file

    Returns:
        Dictionary with three keys:
        - 'subject': List of flat keywords from XMP-dc:Subject
        - 'hierarchical': List of hierarchical keywords from XMP-lr:HierarchicalSubject
        - 'weighted': List of flat keywords from XMP-lr:WeightedFlatSubject

    Examples:
        >>> read_existing_keywords(Path("/tmp/photo.cr3"))
        {'subject': [], 'hierarchical': [], 'weighted': []}

    Note:
        Returns empty lists if XMP file doesn't exist or has no keywords.

    """
    # XMP sidecar has same name with .xmp extension
    xmp_path = image_path.with_suffix(".xmp")

    # Initialize empty result
    result: dict[str, list[str]] = {"subject": [], "hierarchical": [], "weighted": []}

    # Check if XMP sidecar exists
    if not xmp_path.exists():
        logger.info("no_existing_xmp")
        return result

    tags_to_extract = ["XMP:Subject", "XMP:HierarchicalSubject", "XMP:WeightedFlatSubject"]

    # Map the full tag names to the desired short keys for the result dictionary
    key_map = {
        "XMP:Subject": "subject",
        "XMP:HierarchicalSubject": "hierarchical",
        "XMP:WeightedFlatSubject": "weighted",
    }

    with exiftool.ExifToolHelper() as et:
        try:
            metadata = et.get_tags(files=str(xmp_path), tags=tags_to_extract)[0]
        except (ValueError, TypeError, ExifToolExecuteError) as e:
            logger.exception(
                "failed_to_read_existing_keywords",
                error=str(e),
            )
            return result

    for exif_key, result_key in key_map.items():
        if exif_key in metadata:
            result[result_key] = metadata[exif_key]

    logger.debug(
        "existing_keywords_read",
        subject_count=len(result.get("subject", [])),
        hierarchical_count=len(result.get("hierarchical", [])),
        weighted_count=len(result.get("weighted", [])),
    )
    return result


def _format_metadata_value(value: Any) -> str:
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
    with exiftool.ExifToolHelper() as et:
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

    with exiftool.ExifToolHelper() as et:
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


def write_xmp(
    image_path: Path,
    keywords: dict[str, list[str]],
    *,
    description: str | None = None,
    title: str | None = None,
    backup: bool = True,
) -> bool:
    """
    Write tags to an XMP sidecar file in Lightroom-compatible format.

    Args:
        image_path: Path to the image file. XMP sidecar will have same name with .xmp extension
        keywords: Dictionary with 'subject', 'hierarchical', and 'weighted' keyword lists
        description: Optional short description to write to XMP (and ImageDescription)
        title: Optional short title to write to XMP-dc:Title and IPTC:ObjectName
        backup: If True, create a backup of existing XMP file before overwriting

    Returns:
        True if XMP write succeeded, False otherwise.

    References:
        - Lightroom XMP format: https://www.adobe.com/products/xmp.html
        - MWG Guidelines: http://www.metadataworkinggroup.org/

    """
    xmp_path = image_path.with_suffix(".xmp")
    tags_to_write: dict[str, str | list[str]] = {}

    # Build a dictionary of tags to write.
    # pyexiftool handles passing lists for tags that support multiple values.
    if keywords.get("subject"):
        tags_to_write["XMP-dc:Subject"] = keywords["subject"]
    if keywords.get("hierarchical"):
        tags_to_write["XMP-lr:HierarchicalSubject"] = keywords["hierarchical"]
    if keywords.get("weighted"):
        tags_to_write["XMP-lr:WeightedFlatSubject"] = keywords["weighted"]
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
        with exiftool.ExifToolHelper() as et:
            if backup:
                # Creates a backup of the original XMP file if it exists (_original suffix)
                et.set_tags(tags=tags_to_write, files=[str(xmp_path)])
            else:
                et.set_tags(
                    tags=tags_to_write,
                    files=[str(xmp_path)],
                    params=["-overwrite_original"],
                )
    except (ValueError, TypeError, ExifToolExecuteError) as e:
        logger.exception(
            "xmp_write_failed",
            error=str(e),
        )
        return False
    else:
        logger.info(
            "xmp_written_successfully",
            xmp_file=xmp_path.name,
            subject_keywords=len(keywords.get("subject", [])),
            hierarchical_keywords=len(keywords.get("hierarchical", [])),
            weighted_keywords=len(keywords.get("weighted", [])),
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
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    jpeg_dimensions: int = DEFAULT_DIMENSIONS,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> bool:
    """
    Convert image to JPEG bytes in memory, analyze with AI, merge and write metadata to XMP.

    Args:
        image_path: Path to the image file to process
        agent: Configured Pydantic AI Agent with Ollama provider
        preserve_existing_kw: If True, merge with existing keywords rather than overwriting
        write_description: If True, also generate and write a short description
        write_title: If True, also generate and write a short title
        backup_xmp: If True, create a backup of existing XMP file before overwriting
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

    # Step 6: Write to XMP sidecar
    return write_xmp(
        image_path,
        merged_keywords,
        description=description if write_description else None,
        title=title if write_title else None,
        backup=backup_xmp,
    )


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
    *,
    image_extensions: Annotated[
        str,
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = "cr3,jpg",
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
            help="Create a backup of existing XMP files before overwriting",
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
            help="Number of automatic validation retries (0-3)",
        ),
    ] = DEFAULT_RETRIES,
    file_log_level: Annotated[
        LogLevel,
        Parameter(
            name="--file-log-level",
            help="Log level for file (use 'OFF' to disable)",
        ),
    ] = "DEBUG",
    console_log_level: Annotated[
        LogLevel,
        Parameter(
            name="--console-log-level",
            help="Log level for console (use 'OFF' to disable)",
        ),
    ] = "DEBUG",
) -> None:
    """
    Tag images with AI and write Lightroom-compatible XMP sidecars.

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
    - Writes a non-destructive XMP sidecar next to the image. Use --no-write-title/
        --no-write-description to skip fields; --no-backup-xmp to avoid backups.

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
        preserve_keywords=preserve_keywords,
        write_description=write_description,
        write_title=write_title,
        backup_xmp=backup_xmp,
        temperature=temperature,
        max_tokens=max_tokens,
        jpeg_dimensions=jpeg_dimensions,
        jpeg_quality=jpeg_quality,
        retries=retries,
    )

    # Parse and normalize extensions
    ext_set = _parse_extensions(image_extensions)
    if not ext_set:
        logger.error("no_valid_extensions_provided", raw_input=image_extensions)
        raise SystemExit(1)
    logger.debug("parsed_extensions", extensions=sorted(ext_set))

    # Resolve inputs into a concrete file list
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
    file_count = len(image_files)
    logger.info("image_files_discovered", count=file_count)

    # Initialize with provider default URL if not overridden
    if api_base_url is None:
        api_base_url = PROVIDER_URLS.get(provider_name, DEFAULT_OLLAMA_BASE_URL)
        logger.debug("using_default_provider_url", url=api_base_url)
    logger.info(
        "provider_config_resolved",
        provider=provider_name,
        url=api_base_url,
        model=model_name,
    )

    # LLM agent setup (Ollama or LM Studio)
    logger.debug("setting_up_llm_agent", provider=provider_name, url=api_base_url, model=model_name)
    if provider_name == "ollama":
        provider = OllamaProvider(
            base_url=api_base_url,
            api_key=api_key or DEFAULT_OLLAMA_API_KEY,
        )
    elif provider_name == "lmstudio":
        provider = OpenAIProvider(
            base_url=api_base_url,
            api_key=api_key or DEFAULT_LMSTUDIO_API_KEY,
        )
    chat_model = OpenAIChatModel(
        model_name=model_name,
        provider=provider,
    )
    agent = Agent(
        chat_model,
        output_type=GeneratedMetadata,
        retries=retries,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    # Process each file
    success_count = 0
    for idx, image_file in enumerate(image_files, start=1):
        with logger.contextualize(file=image_file.name):
            ok = process_photo(
                image_file,
                agent=agent,
                preserve_existing_kw=preserve_keywords,
                write_description=write_description,
                write_title=write_title,
                backup_xmp=backup_xmp,
                temperature=temperature,
                max_tokens=max_tokens,
                jpeg_dimensions=jpeg_dimensions,
                jpeg_quality=jpeg_quality,
            )
            if ok:
                success_count += 1
                logger.info("processing_success", index=f"{idx}/{file_count}")
            else:
                logger.error("processing_failed", index=f"{idx}/{file_count}")

    # Summary
    logger.info(
        "processing_summary",
        total_files=file_count,
        successful=success_count,
        failed=file_count - success_count,
    )

    if success_count < file_count:
        raise SystemExit(1)


if __name__ == "__main__":
    app(["tests/"])
