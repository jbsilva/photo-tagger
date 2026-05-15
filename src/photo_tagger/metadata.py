"""Read and write XMP/IPTC metadata via pyexiftool."""

from typing import TYPE_CHECKING, Any

from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from exiftool.exceptions import ExifToolExecuteError
from loguru import logger


if TYPE_CHECKING:
    from pathlib import Path

from photo_tagger.config import (
    LOCATION_TAGS,
    TAG_IPTC_KEYWORDS,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_SUBJECT,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
)


_GPS_TAG = "Composite:GPSPosition"

# Tags read from a photo when collecting "existing" keywords. The mapping is
# applied in order, so XMP entries arrive before IPTC fall-backs.
_KEYWORD_TAG_TO_BUCKET: tuple[tuple[str, str], ...] = (
    (TAG_XMP_SUBJECT, "subject"),
    (TAG_XMP_HIERARCHICAL_SUBJECT, "hierarchical"),
    (TAG_XMP_WEIGHTED_FLAT_SUBJECT, "weighted"),
    (TAG_IPTC_KEYWORDS, "subject"),
)


def metadata_targets(image_path: Path) -> list[str]:
    """
    Return the file paths that may contain metadata for an image.

    Examples:
        >>> metadata_targets(Path("/photos/image.cr3"))  # doctest: +SKIP
        ['/photos/image.cr3', '/photos/image.xmp']

    """
    targets: list[str] = []
    xmp_path = image_path.with_suffix(".xmp")
    if image_path.exists():
        targets.append(str(image_path))
    if xmp_path.exists():
        targets.append(str(xmp_path))
    return targets


def format_metadata_value(value: Any) -> str:  # noqa: ANN401
    """
    Coerce metadata values (lists, numbers) into a readable string.

    Examples:
        >>> format_metadata_value(["sky", "", None])
        'sky'
        >>> format_metadata_value(42)
        '42'

    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value if str(v).strip())
    return str(value)


def _coerce_to_list(raw_value: Any) -> list[str]:  # noqa: ANN401
    """Return a clean list of strings from a possibly-scalar exiftool field."""
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item) for item in raw_value if str(item).strip()]
    return [str(raw_value)] if str(raw_value).strip() else []


def _empty_keyword_buckets() -> dict[str, list[str]]:
    return {"subject": [], "hierarchical": [], "weighted": []}


def _accumulate_keyword_blocks(
    blocks: list[dict[str, Any]],
    result: dict[str, list[str]],
) -> int:
    """Fill *result* from exiftool *blocks*; returns the count of IPTC entries seen."""
    iptc_count = 0
    for block in blocks:
        for tag, bucket in _KEYWORD_TAG_TO_BUCKET:
            if tag not in block:
                continue
            values = _coerce_to_list(block[tag])
            result[bucket].extend(values)
            if tag == TAG_IPTC_KEYWORDS:
                iptc_count += len(values)
    return iptc_count


def read_existing_keywords(image_path: Path) -> dict[str, list[str]]:
    """
    Read existing keywords from either the image or its XMP sidecar.

    Args:
        image_path: Path to the image file. Reads embedded metadata and any adjacent XMP file.

    Returns:
        Dictionary with three keys:
        - 'subject': Flat keywords aggregated from XMP-dc:Subject and IPTC:Keywords
        - 'hierarchical': List of hierarchical keywords from XMP-lr:HierarchicalSubject
        - 'weighted': List of flat keywords from XMP-lr:WeightedFlatSubject

    Note:
        Returns empty lists if neither the primary file nor its sidecar contain keywords.

    """
    targets = metadata_targets(image_path)
    result = _empty_keyword_buckets()
    if not targets:
        logger.info("no_metadata_targets_found")
        return result

    tags_to_extract = [tag for tag, _ in _KEYWORD_TAG_TO_BUCKET]
    try:
        with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
            blocks = et.get_tags(files=targets, tags=tags_to_extract)
    except (ValueError, TypeError, ExifToolExecuteError) as e:
        logger.exception("failed_to_read_existing_keywords", error=str(e))
        return result

    iptc_count = _accumulate_keyword_blocks(blocks, result)

    for key, values in result.items():
        if values:
            result[key] = list(dict.fromkeys(values))

    logger.debug(
        "existing_keywords_read",
        subject_count=len(result["subject"]),
        hierarchical_count=len(result["hierarchical"]),
        weighted_count=len(result["weighted"]),
        iptc_keywords_count=iptc_count,
    )
    return result


def read_location_tags(image_path: Path) -> dict[str, str]:
    """Read selected IPTC/XMP location tags from the image or its sidecar."""
    targets = metadata_targets(image_path)
    if not targets:
        return {}

    collected: dict[str, str] = {}
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        try:
            blocks = et.get_tags(files=targets, tags=list(LOCATION_TAGS))
        except (ValueError, TypeError, ExifToolExecuteError) as e:
            logger.exception("failed_to_read_location_tags", error=str(e))
            return {}

    for block in blocks:
        for tag in LOCATION_TAGS:
            value = block.get(tag)
            if value not in (None, ""):
                collected[tag] = format_metadata_value(value)

    if collected:
        logger.debug("location_tags_read", tags=collected)
    return collected


def read_gps_coordinates(image_path: Path) -> dict[str, str]:
    """Read GPS coordinates from either the primary file or its XMP sidecar."""
    targets = metadata_targets(image_path)
    if not targets:
        return {}

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        try:
            blocks = et.get_tags(files=targets, tags=[_GPS_TAG])
        except (ValueError, TypeError, ExifToolExecuteError) as e:
            logger.exception("failed_to_read_gps", error=str(e))
            return {}

    for block in blocks:
        value = block.get(_GPS_TAG)
        if value not in (None, ""):
            position = format_metadata_value(value)
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
    """Create a concise user prompt that surfaces existing photo metadata."""
    metadata_lines: list[str] = []

    if unique_keywords := list(dict.fromkeys(flat_keywords)):
        selected = unique_keywords[:max_prompt_flat_keywords]
        line = "- Existing Keywords: " + ", ".join(selected)
        if len(unique_keywords) > max_prompt_flat_keywords:
            line += ", ..."
        metadata_lines.append(line)

    location_value = next(
        (location_tags.get(tag) for tag in LOCATION_TAGS if location_tags.get(tag)),
        None,
    )
    if location_value:
        metadata_lines.append(f"- Location: {location_value}")

    if gps_position := (gps_info or {}).get("position"):
        metadata_lines.append(f"- GPS: {gps_position}")

    sections = [base_prompt.strip()]
    if metadata_lines:
        sections.append("Existing Metadata:\n" + "\n".join(metadata_lines))
    return "\n\n".join(sections)


def _build_write_payload(
    keywords: dict[str, list[str]],
    description: str | None,
    title: str | None,
) -> dict[str, str | list[str]]:
    """Build the exiftool tag map that write_metadata will apply."""
    payload: dict[str, str | list[str]] = {}
    if subjects := keywords.get("subject"):
        payload["XMP-dc:Subject"] = subjects
        # Lightroom prioritises IPTC:Keywords for JPEGs, so mirror the Subject list there.
        payload[TAG_IPTC_KEYWORDS] = subjects
    if hierarchical := keywords.get("hierarchical"):
        payload["XMP-lr:HierarchicalSubject"] = hierarchical
    if description:
        payload["XMP-dc:Description"] = description
        payload["XMP-exif:ImageDescription"] = description
    if title:
        payload["XMP-dc:Title"] = title
        payload["IPTC:ObjectName"] = title
    return payload


def write_metadata(  # noqa: PLR0913 - distinct optional fields are clearer as kwargs.
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
        image_path: Path to the image file. The sidecar shares its name with a `.xmp` extension.
        keywords: Dictionary with 'subject' and 'hierarchical' keyword lists (optionally weighted).
        description: Optional short description to write to XMP (and ImageDescription).
        title: Optional short title to write to XMP-dc:Title and IPTC:ObjectName.
        backup: If True, let ExifTool create a backup (`_original` suffix where applicable).
        use_sidecar: If True, write to a sidecar; otherwise embed in the source file.

    Returns:
        True on success, False on failure.

    """
    target_path = image_path.with_suffix(".xmp") if use_sidecar else image_path
    payload = _build_write_payload(keywords, description, title)
    if not payload:
        logger.warning("no_data_to_write", file=image_path.name)
        return False

    try:
        with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
            if backup:
                et.set_tags(tags=payload, files=[str(target_path)])
            else:
                et.set_tags(
                    tags=payload,
                    files=[str(target_path)],
                    params=["-overwrite_original"],
                )
    except (ValueError, TypeError, ExifToolExecuteError) as e:
        logger.exception("xmp_write_failed", error=str(e), target=str(target_path))
        return False

    logger.info(
        "metadata_written_successfully",
        target=str(target_path),
        mode="sidecar" if use_sidecar else "embedded",
        subject_keywords=len(keywords.get("subject", [])),
        hierarchical_keywords=len(keywords.get("hierarchical", [])),
        backup_created=backup,
    )
    return True
