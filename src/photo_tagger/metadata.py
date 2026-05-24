"""Read and write XMP/IPTC metadata via pyexiftool."""

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from exiftool.exceptions import ExifToolExecuteError
from loguru import logger


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

from photo_tagger.config import (
    CAMERA_TAGS,
    LOCATION_TAGS,
    TAG_IPTC_KEYWORDS,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_SUBJECT,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
)


@contextlib.contextmanager
def _helper(et: ExifToolHelper | None) -> Iterator[ExifToolHelper]:
    """Yield *et* if supplied, otherwise spin up and tear down a one-shot helper."""
    if et is not None:
        yield et
        return
    with ExifToolHelper() as own:  # type: ignore[no-untyped-call]
        yield own


_GPS_TAG = "Composite:GPSPosition"

# Any of these tags being non-empty marks an image as already tagged. Covers the cases
# where another tool (Lightroom, exiftool by hand, a previous photo-tagger run) wrote
# keywords, a description, or a title to either the image or its XMP sidecar.
_TAGGED_INDICATOR_TAGS: tuple[str, ...] = (
    TAG_XMP_SUBJECT,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
    TAG_IPTC_KEYWORDS,
    "XMP:Description",
    "EXIF:ImageDescription",
    "XMP:Title",
    "IPTC:ObjectName",
)

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


def _dedup_preserving_first_case(values: list[str]) -> list[str]:
    """
    Return *values* with case-insensitive duplicates collapsed, first-seen casing kept.

    Photos often carry the same keyword in both XMP-dc:Subject and IPTC:Keywords with
    different casing (``Bird`` vs ``bird``). Exact-match dedup leaves both, and the
    writer faithfully replays the duplicate back to disk. Compare on casefold instead
    so the output keyword list reflects what Lightroom would consider distinct.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


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


def read_existing_keywords(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> dict[str, list[str]]:
    """
    Read existing keywords from either the image or its XMP sidecar.

    Args:
        image_path: Path to the image file. Reads embedded metadata and any adjacent XMP file.
        et: Optional already-open ExifToolHelper to reuse so a batch can avoid spinning up one
            subprocess per call. A one-shot helper is opened when omitted.

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
        with _helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=tags_to_extract)
    except (ValueError, TypeError, ExifToolExecuteError) as e:
        logger.exception("failed_to_read_existing_keywords", error=str(e))
        return result

    iptc_count = _accumulate_keyword_blocks(blocks, result)

    for key, values in result.items():
        if values:
            result[key] = _dedup_preserving_first_case(values)

    logger.debug(
        "existing_keywords_read",
        subject_count=len(result["subject"]),
        hierarchical_count=len(result["hierarchical"]),
        weighted_count=len(result["weighted"]),
        iptc_keywords_count=iptc_count,
    )
    return result


def _value_is_present(value: Any) -> bool:  # noqa: ANN401
    """Return True if an exiftool tag value carries any non-blank content."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(str(item).strip() for item in value)
    return bool(str(value).strip())


def _block_has_indicator(blocks: list[dict[str, Any]]) -> bool:
    """Return True if any indicator tag in the exiftool result blocks is populated."""
    for block in blocks:
        for tag in _TAGGED_INDICATOR_TAGS:
            if _value_is_present(block.get(tag)):
                return True
    return False


def find_tagged_images(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> set[Path]:
    """
    Return the subset of *image_paths* that already carry meaningful metadata.

    An image counts as tagged if either it or its XMP sidecar has any of the indicator
    tags populated (keywords, hierarchical keywords, description, or title). The check
    runs through a single persistent ExifToolHelper context so the pipeline does not pay
    the per-file process startup cost.
    """
    paths = list(image_paths)
    if not paths:
        return set()

    tagged: set[Path] = set()
    try:
        with _helper(et) as helper:
            for image_path in paths:
                targets = metadata_targets(image_path)
                if not targets:
                    continue
                try:
                    blocks = helper.get_tags(files=targets, tags=list(_TAGGED_INDICATOR_TAGS))
                except (ValueError, TypeError, ExifToolExecuteError) as exc:
                    logger.warning(
                        "failed_to_check_tagged_status",
                        file=str(image_path),
                        error=str(exc),
                    )
                    continue
                if _block_has_indicator(blocks):
                    tagged.add(image_path)
    except (ValueError, TypeError, ExifToolExecuteError) as exc:
        logger.exception("failed_to_open_exiftool_for_tagged_check", error=str(exc))
        return set()

    if tagged:
        logger.debug("tagged_images_detected", count=len(tagged))
    return tagged


def read_location_tags(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> dict[str, str]:
    """Read selected IPTC/XMP location tags from the image or its sidecar."""
    targets = metadata_targets(image_path)
    if not targets:
        return {}

    collected: dict[str, str] = {}
    try:
        with _helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=list(LOCATION_TAGS))
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


def read_gps_coordinates(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> dict[str, str]:
    """Read GPS coordinates from either the primary file or its XMP sidecar."""
    targets = metadata_targets(image_path)
    if not targets:
        return {}

    try:
        with _helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=[_GPS_TAG])
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


@dataclass(slots=True, frozen=True)
class ImageContext:
    """
    Bundle of metadata read off a photo before the AI call.

    Replaces the older sequence of separate ``read_existing_keywords`` /
    ``read_location_tags`` / ``read_gps_coordinates`` calls that the pipeline
    used to issue, which cost three exiftool IPC round-trips per image. The
    batched read fetches everything we need in one call.
    """

    existing_keywords: dict[str, list[str]] = field(default_factory=_empty_keyword_buckets)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    camera_info: dict[str, str] = field(default_factory=dict)


def _extract_gps(blocks: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty GPS position string from *blocks*, else None."""
    for block in blocks:
        value = block.get(_GPS_TAG)
        if value not in (None, ""):
            return format_metadata_value(value)
    return None


def _extract_named_tags(
    blocks: list[dict[str, Any]],
    tags: tuple[str, ...],
) -> dict[str, str]:
    """Collect non-blank string values for *tags* across all *blocks*."""
    collected: dict[str, str] = {}
    for block in blocks:
        for tag in tags:
            value = block.get(tag)
            if value not in (None, "") and tag not in collected:
                collected[tag] = format_metadata_value(value)
    return collected


def read_image_context(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> ImageContext:
    """
    Fetch every read-only tag the pipeline needs in a single exiftool call.

    This is the production path used by ``process_photo``. The older single-purpose
    helpers (``read_existing_keywords``, ``read_location_tags``, ``read_gps_coordinates``)
    remain available for callers that need only one slice (Tests, for example).
    """
    targets = metadata_targets(image_path)
    if not targets:
        logger.info("no_metadata_targets_found")
        return ImageContext()

    keyword_tags = [tag for tag, _ in _KEYWORD_TAG_TO_BUCKET]
    all_tags = list(dict.fromkeys([*keyword_tags, *LOCATION_TAGS, *CAMERA_TAGS, _GPS_TAG]))

    try:
        with _helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=all_tags)
    except (ValueError, TypeError, ExifToolExecuteError) as exc:
        logger.exception("failed_to_read_image_context", error=str(exc))
        return ImageContext()

    existing_keywords = _empty_keyword_buckets()
    _accumulate_keyword_blocks(blocks, existing_keywords)
    for key, values in existing_keywords.items():
        if values:
            existing_keywords[key] = _dedup_preserving_first_case(values)

    location_tags = _extract_named_tags(blocks, LOCATION_TAGS)
    camera_info = _extract_named_tags(blocks, CAMERA_TAGS)
    gps_position = _extract_gps(blocks)

    logger.debug(
        "image_context_read",
        subject_count=len(existing_keywords["subject"]),
        hierarchical_count=len(existing_keywords["hierarchical"]),
        location_tag_count=len(location_tags),
        camera_tag_count=len(camera_info),
        gps_present=gps_position is not None,
    )
    return ImageContext(
        existing_keywords=existing_keywords,
        location_tags=location_tags,
        gps_position=gps_position,
        camera_info=camera_info,
    )


def _first_present(values: dict[str, str], *tags: str) -> str | None:
    """Return the first non-empty value in *values* lookup-ordered by *tags*."""
    for tag in tags:
        if (raw := values.get(tag)) and (stripped := raw.strip()):
            return stripped
    return None


def _format_location(location_tags: dict[str, str]) -> str | None:
    """
    Build a "City, Country" hint, falling back to whichever single field is set.

    XMP-photoshop and IPTC each define their own city/country fields, so this
    helper picks the first present in each category and joins them. Returning
    both means the model gets the full place name (e.g. "Barcelona, Spain")
    instead of silently losing the city under the older "first hit wins" logic.
    """
    city = _first_present(location_tags, "XMP-photoshop:City", "IPTC:City")
    country = _first_present(
        location_tags,
        "XMP-photoshop:Country",
        "IPTC:Country-PrimaryLocationName",
    )
    if city and country:
        return f"{city}, {country}"
    return city or country


def _camera_lines(camera_info: dict[str, str]) -> list[str]:
    """Render ``EXIF:Model`` / ``EXIF:LensModel`` / ``EXIF:DateTimeOriginal`` lines."""
    lines: list[str] = []
    if model := camera_info.get("EXIF:Model"):
        lines.append(f"- Camera: {model}")
    if lens := camera_info.get("EXIF:LensModel"):
        lines.append(f"- Lens: {lens}")
    if captured := camera_info.get("EXIF:DateTimeOriginal"):
        lines.append(f"- Captured: {captured}")
    return lines


def build_contextual_prompt(  # noqa: PLR0913 - each kwarg renders an independent section.
    base_prompt: str,
    flat_keywords: list[str],
    location_tags: dict[str, str],
    gps_info: dict[str, str],
    max_prompt_flat_keywords: int = 4,
    *,
    camera_info: dict[str, str] | None = None,
) -> str:
    """
    Create a concise user prompt that surfaces existing photo metadata.

    *camera_info* is a mapping of ExifTool tag names (``EXIF:Model`` etc.) to their
    string values. When present, equipment and capture-date hints are appended so
    the model can take cues from the gear used (macro lens, telephoto, wide angle)
    and the time of year.
    """
    metadata_lines: list[str] = []

    if unique_keywords := list(dict.fromkeys(flat_keywords)):
        selected = unique_keywords[:max_prompt_flat_keywords]
        line = "- Existing Keywords: " + ", ".join(selected)
        if len(unique_keywords) > max_prompt_flat_keywords:
            line += ", ..."
        metadata_lines.append(line)

    if location_value := _format_location(location_tags):
        metadata_lines.append(f"- Location: {location_value}")

    if gps_position := (gps_info or {}).get("position"):
        metadata_lines.append(f"- GPS: {gps_position}")

    metadata_lines.extend(_camera_lines(camera_info or {}))

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
        # Lightroom prioritizes IPTC:Keywords for JPEGs, so mirror the Subject list there.
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
    et: ExifToolHelper | None = None,
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
        et: Optional already-open ExifToolHelper to reuse across a batch.

    Returns:
        True on success, False on failure.

    """
    target_path = image_path.with_suffix(".xmp") if use_sidecar else image_path
    payload = _build_write_payload(keywords, description, title)
    if not payload:
        logger.warning("no_data_to_write", file=image_path.name)
        return False

    set_kwargs: dict[str, Any] = {"tags": payload, "files": [str(target_path)]}
    if not backup:
        set_kwargs["params"] = ["-overwrite_original"]

    try:
        with _helper(et) as helper:
            helper.set_tags(**set_kwargs)
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
