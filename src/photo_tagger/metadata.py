"""Read and write XMP/IPTC metadata via pyexiftool."""

import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from itertools import batched, chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from exiftool.exceptions import ExifToolExecuteError
from loguru import logger


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    from photo_tagger.fields import FieldSpec

from photo_tagger import __version__
from photo_tagger.config import (
    CAMERA_TAGS,
    CITY_TAGS,
    COUNTRY_TAGS,
    LOCATION_TAGS,
    NON_RAW_EXTENSIONS,
    TAG_EXIF_DATE_TIME_ORIGINAL,
    TAG_EXIF_IMAGE_DESCRIPTION,
    TAG_IPTC_KEYWORDS,
    TAG_IPTC_OBJECT_NAME,
    TAG_XMP_DESCRIPTION,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_IMAGE_DESCRIPTION,
    TAG_XMP_SUBJECT,
    TAG_XMP_TITLE,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
    FileTypeTarget,
    SidecarMode,
    SidecarOverrides,
    exiftool_executable,
)
from photo_tagger.fields import COPYRIGHT_READ_TAGS
from photo_tagger.models import KeywordSet


# Exception types that pyexiftool raises on bad inputs or subprocess failures.
# Centralized here to avoid seven copies of the same tuple. FileNotFoundError belongs with them:
# it is what pyexiftool raises when the binary itself is missing, which is the most basic way for
# an exiftool call to fail. Without it every read here raised a bare traceback instead of logging
# and degrading, and every write raised instead of reporting the photo as failed.
_EXIFTOOL_ERRORS = (FileNotFoundError, ValueError, TypeError, ExifToolExecuteError)

# Tag and api option that make exiftool hash the image data only, skipping metadata. The result
# comes back under the "File:" group, hence the separate key constant.
_IMAGE_DATA_HASH_TAG = "ImageDataHash"
_IMAGE_DATA_HASH_KEY = "File:ImageDataHash"
_IMAGE_HASH_API = ["-api", "ImageHashType=SHA256"]


@contextlib.contextmanager
def managed_helper(et: ExifToolHelper | None) -> Generator[ExifToolHelper]:
    """
    Yield *et* if supplied, otherwise spin up and tear down a one-shot helper.

    Honors an explicit ExifTool binary (``PHOTO_TAGGER_EXIFTOOL`` / config ``exiftool_path``); with
    none configured, pyexiftool finds ``exiftool`` on PATH as before.
    """
    if et is not None:
        yield et
        return
    executable = exiftool_executable()
    helper = ExifToolHelper(executable=executable) if executable else ExifToolHelper()
    with helper as own:  # type: ignore[no-untyped-call]
        yield own


_GPS_TAG = "Composite:GPSPosition"

# Existing title/description tags, in read priority order (first non-empty wins). The GUI
# surfaces these so a user can see and edit what is already on the photo before saving.
_TITLE_TAGS: tuple[str, ...] = (TAG_XMP_TITLE, TAG_IPTC_OBJECT_NAME)
# TAG_XMP_IMAGE_DESCRIPTION is the read spelling of XMP-tiff:ImageDescription, the mirror
# write_metadata itself produces, so our own sidecars (and other tools' tiff namespace
# descriptions) count on re-read.
_DESCRIPTION_TAGS: tuple[str, ...] = (
    TAG_XMP_DESCRIPTION,
    TAG_XMP_IMAGE_DESCRIPTION,
    TAG_EXIF_IMAGE_DESCRIPTION,
)

# Any of these tags being non-empty marks an image as already tagged. Covers the cases
# where another tool (Lightroom, exiftool by hand, a previous photo-tagger run) wrote
# keywords, a description, or a title to either the image or its XMP sidecar.
_TAGGED_INDICATOR_TAGS: tuple[str, ...] = (
    TAG_XMP_SUBJECT,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
    TAG_IPTC_KEYWORDS,
    *_DESCRIPTION_TAGS,
    *_TITLE_TAGS,
)

# Names of the metadata fields a caller can ask about individually (vs the coarse "any
# indicator" check above). Stable identifiers the GUI's field-aware deselect uses.
FIELD_TITLE = "title"
FIELD_DESCRIPTION = "description"
FIELD_KEYWORDS = "keywords"
FIELD_COPYRIGHT = "copyright"

# The fields that make a photo "tagged". Copyright is not one of them: many cameras stamp a notice
# on every frame, and that photo still waits for its title, description, and keywords.
TAGGED_FIELDS = frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS})

# The exiftool tags that mark each field as populated. A field counts as present when any of
# its tags carries content, on either the image or its XMP sidecar.
_FIELD_PRESENCE_TAGS: dict[str, tuple[str, ...]] = {
    FIELD_TITLE: _TITLE_TAGS,
    FIELD_DESCRIPTION: _DESCRIPTION_TAGS,
    FIELD_KEYWORDS: (
        TAG_XMP_SUBJECT,
        TAG_XMP_HIERARCHICAL_SUBJECT,
        TAG_XMP_WEIGHTED_FLAT_SUBJECT,
        TAG_IPTC_KEYWORDS,
    ),
    FIELD_COPYRIGHT: COPYRIGHT_READ_TAGS,
}

# How many images one presence read covers. A batch is the smallest unit a caller can report
# progress for or stop at, so a small one keeps the GUI's scan bar moving and its Stop button
# prompt. Splitting costs about a millisecond per extra call (the exiftool process is shared and
# stays open), which is nothing against a network share where a single file costs a round trip and
# reading a few thousand of them runs for minutes.
PRESENCE_BATCH_SIZE = 25

# Tags read from a photo when collecting "existing" keywords, paired with the
# KeywordSet field they feed. Applied in order, so XMP entries arrive before the
# IPTC fall-back.
_KEYWORD_TAG_TO_FIELD: tuple[tuple[str, str], ...] = (
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


def _dedup_preserving_first_case(values: list[str]) -> list[str]:
    """
    Return *values* with case-insensitive duplicates collapsed, first-seen casing kept.

    Photos often carry the same keyword in both XMP-dc:Subject and IPTC:Keywords with different
    casing (``Bird`` vs ``bird``). Exact-match dedup leaves both, and the writer faithfully replays
    the duplicate back to disk. Compare on casefold instead so the output keyword list reflects what
    Lightroom would consider distinct.
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
    result: KeywordSet,
) -> int:
    """Fill *result* from exiftool *blocks*; returns the count of IPTC entries seen."""
    iptc_count = 0
    for block in blocks:
        for tag, field_name in _KEYWORD_TAG_TO_FIELD:
            if tag not in block:
                continue
            values = _coerce_to_list(block[tag])
            getattr(result, field_name).extend(values)
            if tag == TAG_IPTC_KEYWORDS:
                iptc_count += len(values)
    return iptc_count


def _dedup_keyword_set(keywords: KeywordSet) -> None:
    """Collapse case-insensitive duplicates in every view of *keywords* in place."""
    keywords.subject = _dedup_preserving_first_case(keywords.subject)
    keywords.hierarchical = _dedup_preserving_first_case(keywords.hierarchical)
    keywords.weighted = _dedup_preserving_first_case(keywords.weighted)


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


def _build_target_index(paths: list[Path]) -> tuple[list[str], dict[Path, list[Path]]]:
    """
    Flatten every image's metadata targets and map each target back to its image(s).

    The map is keyed by ``Path``, not the string sent to exiftool: exiftool echoes ``SourceFile``
    with forward slashes, so string keys would miss every entry on Windows, where ``str(Path)`` uses
    backslashes. Path equality normalizes the separator.

    A target maps to a *list* because RAW+JPEG pairs sharing a stem (``IMG_0001.CR3`` and
    ``IMG_0001.JPG``) both resolve to the same ``IMG_0001.xmp`` sidecar; a single ``Path`` value
    here would silently drop all but the last image sharing that sidecar.
    """
    all_targets: list[str] = []
    target_to_images: dict[Path, list[Path]] = {}
    for image_path in paths:
        for target in metadata_targets(image_path):
            all_targets.append(target)
            target_to_images.setdefault(Path(target), []).append(image_path)
    return all_targets, target_to_images


def _images_for_block(
    block: dict[str, Any],
    target_to_images: dict[Path, list[Path]],
) -> list[Path]:
    """Map one exiftool result block back to its source image(s) via the target index."""
    source_file = block.get("SourceFile")
    if not source_file:
        return []
    return target_to_images.get(Path(str(source_file)), [])


def _batched_get_tags(
    et: ExifToolHelper | None,
    *,
    files: list[str],
    tags: list[str],
) -> list[dict[str, Any]]:
    """
    Run one ``get_tags`` over *files*, salvaging the payload when a single file is unreadable.

    exiftool exits non-zero when any file in the batch has a format error, and pyexiftool then
    raises even though the JSON for every file (including the healthy ones) is already on stdout.
    Without the salvage, one corrupt or zero-byte photo would wipe out the whole folder's read and
    every image would report "no metadata". Re-raises when there is nothing to salvage.
    """
    with managed_helper(et) as helper:
        try:
            blocks: list[dict[str, Any]] = helper.get_tags(files=files, tags=tags)
        except ExifToolExecuteError as exc:
            try:
                blocks = json.loads(exc.stdout or "")
            except ValueError:
                raise exc from None
            if not isinstance(blocks, list):
                raise
            logger.warning(
                "exiftool_batch_partial_failure",
                files=len(files),
                error=str(exc.stderr).strip(),
            )
        return blocks


def find_tagged_images(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> set[Path]:
    """
    Return the subset of *image_paths* that already carry meaningful metadata.

    An image counts as tagged if either it or its XMP sidecar has any of the indicator tags
    populated (keywords, hierarchical keywords, description, or title). The check batches all target
    files into a single exiftool call and maps the result blocks back to source images, reducing the
    IPC cost from O(N) round-trips to O(1).
    """
    paths = list(image_paths)
    if not paths:
        return set()

    all_targets, target_to_images = _build_target_index(paths)
    if not all_targets:
        return set()

    tagged: set[Path] = set()
    try:
        blocks = _batched_get_tags(et, files=all_targets, tags=list(_TAGGED_INDICATOR_TAGS))
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_open_exiftool_for_tagged_check", error=str(exc))
        return set()

    for block in blocks:
        if _block_has_indicator([block]):
            tagged.update(_images_for_block(block, target_to_images))

    if tagged:
        logger.debug("tagged_images_detected", count=len(tagged))
    return tagged


@dataclass(frozen=True, slots=True)
class PresenceBatch:
    """
    One batch of a chunked field-presence read.

    *scanned* counts the images the batch covered whether or not they could be read, so a caller
    driving a progress bar can advance by it. *presence* is empty when exiftool could not read the
    batch at all: those images stay out of the results rather than being reported as carrying no
    metadata.
    """

    scanned: int
    presence: dict[Path, set[str]]


def _read_field_presence(
    paths: list[Path],
    *,
    tags: list[str],
    et: ExifToolHelper | None,
) -> dict[Path, set[str]]:
    """Read one batch of images; an empty result means exiftool could not be run for it."""
    presence: dict[Path, set[str]] = {path: set() for path in paths}
    all_targets, target_to_images = _build_target_index(paths)
    if not all_targets:
        return presence

    try:
        blocks = _batched_get_tags(et, files=all_targets, tags=tags)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_open_exiftool_for_field_presence", error=str(exc))
        return {}

    for block in blocks:
        images = _images_for_block(block, target_to_images)
        if not images:
            continue
        present_fields = {
            field_name
            for field_name, tags_for_field in _FIELD_PRESENCE_TAGS.items()
            if any(_value_is_present(block.get(tag)) for tag in tags_for_field)
        }
        for image_path in images:
            presence[image_path].update(present_fields)
    return presence


def iter_field_presence(
    image_paths: Iterable[Path],
    *,
    batch_size: int = PRESENCE_BATCH_SIZE,
    et: ExifToolHelper | None = None,
) -> Generator[PresenceBatch]:
    """
    Yield :func:`find_field_presence` results one batch of images at a time.

    The same read, split up: a caller that would otherwise wait minutes for a whole library gets a
    result every *batch_size* images, so it can paint progress and stop early by abandoning the
    iterator. All batches share one exiftool process, which is what keeps the split cheap.

    *image_paths* is pulled one batch at a time, never all at once, so a caller that passes a
    generator can still drop images after the read has started (the GUI does this when a folder
    leaves the list mid-scan).
    """
    batches = batched(image_paths, batch_size, strict=False)  # a short last batch is normal
    first = next(batches, None)
    if first is None:
        return
    tags = sorted({tag for tags in _FIELD_PRESENCE_TAGS.values() for tag in tags})
    try:
        with managed_helper(et) as helper:
            for chunk in chain((first,), batches):
                batch = list(chunk)
                yield PresenceBatch(len(batch), _read_field_presence(batch, tags=tags, et=helper))
    except _EXIFTOOL_ERRORS as exc:
        # Opening the shared helper failed (a missing binary, most likely), so no batch can be
        # read. Yielding nothing leaves every image absent from the results: unknown, not untagged.
        logger.exception("failed_to_open_exiftool_for_field_presence", error=str(exc))


def find_field_presence(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, set[str]]:
    """
    Report which of title/description/keywords/copyright each image already carries.

    Returns a mapping from every input path to the set of present field names (:data:`FIELD_TITLE`,
    :data:`FIELD_DESCRIPTION`, :data:`FIELD_KEYWORDS`, :data:`FIELD_COPYRIGHT`); an empty set means
    the photo was scanned and carries none. An image exiftool could not read at all is left out of
    the mapping entirely, so callers (the GUI's Tagged column) can tell "could not read" from "read
    and found nothing" instead of wrongly reporting it as untagged. A field counts as present when
    any of its tags is populated on the image *or* its XMP sidecar, so presence accumulates across
    both. The read is batched (see :data:`PRESENCE_BATCH_SIZE`) rather than one exiftool call per
    image.

    This is the per-field counterpart the GUI uses to deselect, say, photos that already have a
    title and description while leaving keyword-only photos selected.
    """
    presence: dict[Path, set[str]] = {}
    for batch in iter_field_presence(image_paths, et=et):
        presence.update(batch.presence)
    return presence


def read_keyword_sets(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, KeywordSet]:
    """
    Read the keywords already on each of *image_paths*, image and XMP sidecar merged.

    Every input path maps to a :class:`KeywordSet`, empty when the photo carries no keyword; an
    *empty dict* means exiftool could not be run at all, which callers must tell apart from "read
    and found nothing" (see :func:`find_field_presence`). Like the other census-style reads, this is
    one batched exiftool call rather than one per image, so a whole library can be surveyed without
    paying an IPC round-trip per photo.
    """
    paths = list(image_paths)
    keywords = {path: KeywordSet() for path in paths}
    if not paths:
        return keywords

    all_targets, target_to_images = _build_target_index(paths)
    if not all_targets:
        return keywords

    tags = list(dict.fromkeys(tag for tag, _ in _KEYWORD_TAG_TO_FIELD))
    try:
        blocks = _batched_get_tags(et, files=all_targets, tags=tags)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_open_exiftool_for_keyword_census", error=str(exc))
        return {}

    for block in blocks:
        for image_path in _images_for_block(block, target_to_images):
            _accumulate_keyword_blocks([block], keywords[image_path])
    for keyword_set in keywords.values():
        _dedup_keyword_set(keyword_set)
    return keywords


def _is_sidecar_block(block: dict[str, Any]) -> bool:
    """Report whether an exiftool result block came from an XMP sidecar rather than the photo."""
    return str(block.get("SourceFile", "")).casefold().endswith(".xmp")


def _sidecar_first(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Order exiftool result blocks so an XMP sidecar's answer wins over the image file's.

    The sidecar is the later copy: it is what every XMP-aware catalog reads, and the only thing a
    sidecar-mode run writes. Reading the image first let a stale caption in it shadow the one just
    written beside it. ``sorted`` is stable, so blocks on the same side keep exiftool's order.
    """
    return sorted(blocks, key=lambda block: not _is_sidecar_block(block))


def _first_tag_value(blocks: list[dict[str, Any]], tags: tuple[str, ...]) -> str | None:
    """
    Return the first non-blank value across *blocks* for the first matching *tags* entry.

    Sidecar blocks are consulted before the image's own (see :func:`_sidecar_first`).

    Emptiness is judged by :func:`_value_is_present`, not by comparing against ``""``: an empty
    ``rdf:Bag`` comes back from exiftool as ``[]`` and a blank one as ``[" "]``, and either would
    otherwise count as content and shadow the fall-back tag that holds the real value.
    """
    ordered = _sidecar_first(blocks)
    for tag in tags:
        for block in ordered:
            value = block.get(tag)
            if _value_is_present(value):
                return format_metadata_value(value)
    return None


# Human-readable labels for where existing metadata lives.
SOURCE_IMAGE = "image file"
SOURCE_SIDECAR = "XMP sidecar"


@dataclass(slots=True, frozen=True)
class FieldReading:
    """One value a photo carries for a shown field: what it says, and which tag holds it."""

    label: str
    value: str
    tag: str
    # SOURCE_IMAGE or SOURCE_SIDECAR.
    source: str


def _readings_for_field(spec: FieldSpec, blocks: list[dict[str, Any]]) -> list[FieldReading]:
    """
    Collect one field's distinct values across the photo and its sidecar, best first.

    Distinct is the point: a photo credited the same way in XMP, EXIF, and IPTC is one line, while
    one whose EXIF copyright disagrees with its XMP rights is two, which is exactly the disagreement
    somebody runs ``show`` to find.
    """
    readings: list[FieldReading] = []
    seen: set[str] = set()
    for tag in spec.tags:
        for block in _sidecar_first(blocks):
            value = block.get(tag)
            if not _value_is_present(value):
                continue
            text = format_metadata_value(value)
            if text in seen:
                continue
            seen.add(text)
            readings.append(
                FieldReading(
                    label=spec.label,
                    value=text,
                    tag=tag,
                    source=SOURCE_SIDECAR if _is_sidecar_block(block) else SOURCE_IMAGE,
                ),
            )
    return readings


def read_fields(
    image_path: Path,
    specs: Sequence[FieldSpec],
    *,
    et: ExifToolHelper | None = None,
) -> list[FieldReading]:
    """
    Read the fields in *specs* off a photo and its XMP sidecar, in one exiftool call.

    Returns one entry per distinct value, in the order *specs* lists the fields; a field nothing
    carries is simply absent. This is what ``photo-tagger show`` prints, and it reports the tag
    behind every value because "which copyright does Lightroom actually see" is a question about the
    tag, not about the photo.
    """
    targets = metadata_targets(image_path)
    if not targets:
        return []

    tags = list(dict.fromkeys(tag for spec in specs for tag in spec.tags))
    try:
        blocks = _batched_get_tags(et, files=targets, tags=tags)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_fields", file=str(image_path), error=str(exc))
        return []
    return [reading for spec in specs for reading in _readings_for_field(spec, blocks)]


def read_all_tags(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> dict[str, dict[str, str]]:
    """
    Read every tag exiftool reports, keyed by the file it came from.

    The escape hatch behind ``show --all``: the curated field list answers the usual question, and
    this answers the unusual one without making the user leave the tool. Binary payloads (thumbnail
    data, colour profiles) are left out, since a terminal cannot show them anyway.
    """
    targets = metadata_targets(image_path)
    if not targets:
        return {}

    try:
        with managed_helper(et) as helper:
            blocks: list[dict[str, Any]] = helper.get_metadata(files=targets)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_all_tags", file=str(image_path), error=str(exc))
        return {}

    by_file: dict[str, dict[str, str]] = {}
    for block in blocks:
        source = str(block.get("SourceFile", image_path))
        by_file[source] = _printable_tags(block)
    return by_file


# How exiftool renders a value it will not print: a thumbnail, an ICC profile, a maker note blob.
_BINARY_MARKER = "(Binary data"


def _printable_tags(block: dict[str, Any]) -> dict[str, str]:
    """Flatten one exiftool result block to text, dropping its own key and any binary payload."""
    printable: dict[str, str] = {}
    for tag, value in block.items():
        if tag == "SourceFile":
            continue
        text = format_metadata_value(value)
        if not text.startswith(_BINARY_MARKER):
            printable[tag] = text
    return printable


@dataclass(slots=True, frozen=True)
class CaptionValue:
    """One title or description found on a photo, and which file it came from."""

    value: str
    # SOURCE_IMAGE or SOURCE_SIDECAR.
    source: str


def read_caption_values(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> dict[str, list[CaptionValue]]:
    """
    Read every title and description on a photo, the one photo-tagger uses first.

    Keyed by :data:`FIELD_TITLE` and :data:`FIELD_DESCRIPTION`, each holding one entry per file that
    carries the field, the one in use first. :func:`read_caption` is the "just the values" view.

    The GUI needs the rest of the list to show that the photo itself still holds a caption the
    sidecar shadows: the value every XMP-aware tool ignores but ``exiftool photo.dng`` prints.
    """
    fields: dict[str, list[CaptionValue]] = {FIELD_TITLE: [], FIELD_DESCRIPTION: []}
    targets = metadata_targets(image_path)
    if not targets:
        return fields

    try:
        with managed_helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=[*_TITLE_TAGS, *_DESCRIPTION_TAGS])
    except _EXIFTOOL_ERRORS as e:
        logger.exception("failed_to_read_caption", error=str(e))
        return fields

    ordered = _sidecar_first(blocks)
    for field_name, tags in ((FIELD_TITLE, _TITLE_TAGS), (FIELD_DESCRIPTION, _DESCRIPTION_TAGS)):
        # One block is one file, so what is left inside it is the tag priority.
        per_file = [
            CaptionValue(
                value=value,
                source=SOURCE_SIDECAR if _is_sidecar_block(block) else SOURCE_IMAGE,
            )
            for block in ordered
            if (value := _first_tag_value([block], tags)) is not None
        ]
        fields[field_name] = _winner_first(per_file, _first_tag_value(ordered, tags))
    return fields


def _winner_first(values: list[CaptionValue], winner: str | None) -> list[CaptionValue]:
    """
    Put the value :func:`_first_tag_value` picked at the head of the per-file list.

    Which file answers is decided per tag, not per file: a photo's XMP-dc:Title outranks a sidecar
    carrying only IPTC:ObjectName, and ordering by file alone would invert that.
    """
    if winner is None:
        return values
    return sorted(values, key=lambda entry: entry.value != winner)


def read_caption(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> tuple[str | None, str | None]:
    """
    Read the existing title and description from the image or its XMP sidecar.

    Returns ``(title, description)``, each ``None`` when absent. Title prefers XMP-dc:Title and
    falls back to IPTC:ObjectName; description prefers XMP-dc:Description and falls back to
    EXIF:ImageDescription, and a sidecar's answer wins over the image's. The GUI uses this to show a
    user what is already on a photo before it writes new metadata.
    """
    found = read_caption_values(image_path, et=et)
    return (first_caption(found[FIELD_TITLE]), first_caption(found[FIELD_DESCRIPTION]))


def first_caption(values: list[CaptionValue]) -> str | None:
    """Return the value in use for a caption field, or None when no file carries it."""
    return values[0].value if values else None


def read_metadata_sources(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> list[str]:
    """
    Report where existing metadata lives: the image file, an XMP sidecar, or both.

    Returns a list of source labels (:data:`SOURCE_IMAGE`, :data:`SOURCE_SIDECAR`) for the targets
    that actually carry an indicator tag (keywords, title, or description), so the GUI can tell the
    user whether what it shows came from the photo or its sidecar. Returns an empty list when
    neither target has any such metadata.
    """
    targets = metadata_targets(image_path)
    if not targets:
        return []

    try:
        with managed_helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=list(_TAGGED_INDICATOR_TAGS))
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_metadata_sources", error=str(exc))
        return []

    sources: list[str] = []
    for block in blocks:
        if not _block_has_indicator([block]):
            continue
        label = SOURCE_SIDECAR if _is_sidecar_block(block) else SOURCE_IMAGE
        if label not in sources:
            sources.append(label)
    return sources


@dataclass(slots=True, frozen=True)
class ImageContext:
    """
    Bundle of metadata read off a photo before the AI call.

    Replaces the older sequence of separate single-purpose reads the pipeline used to issue, which
    cost three exiftool IPC round-trips per image. The batched read fetches everything we need in
    one call.
    """

    existing_keywords: KeywordSet = field(default_factory=KeywordSet)
    # The title and description already on the photo, read the same way read_caption reads them.
    # The pipeline needs them to honor --preserve-title / --preserve-description.
    existing_title: str | None = None
    existing_description: str | None = None
    # The copyright notice already on the photo. The GUI shows it next to the one a save writes.
    existing_copyright: str | None = None
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    camera_info: dict[str, str] = field(default_factory=dict)
    # SHA256 of the image data only (exiftool's ImageDataHash), ignoring metadata. None when not
    # requested or when exiftool cannot hash the format. Used as a metadata-independent cache key.
    content_hash: str | None = None


def _extract_gps(blocks: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty GPS position string from *blocks*, else None."""
    for block in blocks:
        value = block.get(_GPS_TAG)
        if _value_is_present(value):
            return format_metadata_value(value)
    return None


def _extract_named_tags(
    blocks: list[dict[str, Any]],
    tags: tuple[str, ...],
) -> dict[str, str]:
    """
    Collect non-blank string values for *tags* across all *blocks*.

    A tag present but blank (``""``, ``[]``, ``[" "]``) is skipped rather than collected as an empty
    string, so the image block and its sidecar can fill in for each other and the prompt never grows
    a dangling ``- Camera:`` line.
    """
    collected: dict[str, str] = {}
    for block in blocks:
        for tag in tags:
            value = block.get(tag)
            if _value_is_present(value) and tag not in collected:
                collected[tag] = format_metadata_value(value)
    return collected


def _extract_content_hash(blocks: list[dict[str, Any]]) -> str | None:
    """
    Return the image-data SHA256 from whichever block carries it, else None.

    Only the image file yields ImageDataHash; XMP sidecars have no image data, so a first-present
    scan across the blocks safely picks the image's hash.
    """
    for block in blocks:
        value = block.get(_IMAGE_DATA_HASH_KEY)
        if value:
            return str(value)
    return None


def read_image_context(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
    include_content_hash: bool = False,
) -> ImageContext:
    """
    Fetch every read-only tag the pipeline needs in a single exiftool call.

    This is the production read path, used by both the CLI's ``process_photo`` and the GUI worker.

    When *include_content_hash* is set, the same call also reads ImageDataHash so callers can key a
    cache on the image content rather than the whole file. It is off by default because hashing the
    image data is wasted work when no cache is configured.
    """
    targets = metadata_targets(image_path)
    if not targets:
        logger.info("no_metadata_targets_found")
        return ImageContext()

    keyword_tags = [tag for tag, _ in _KEYWORD_TAG_TO_FIELD]
    all_tags = list(
        dict.fromkeys(
            [
                *keyword_tags,
                *_TITLE_TAGS,
                *_DESCRIPTION_TAGS,
                *COPYRIGHT_READ_TAGS,
                *LOCATION_TAGS,
                *CAMERA_TAGS,
                _GPS_TAG,
            ],
        ),
    )
    params: list[str] = []
    if include_content_hash:
        all_tags.append(_IMAGE_DATA_HASH_TAG)
        params = _IMAGE_HASH_API

    try:
        with managed_helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=all_tags, params=params)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_image_context", error=str(exc))
        return ImageContext()

    existing_keywords = KeywordSet()
    _accumulate_keyword_blocks(blocks, existing_keywords)
    _dedup_keyword_set(existing_keywords)

    location_tags = _extract_named_tags(blocks, LOCATION_TAGS)
    camera_info = _extract_named_tags(blocks, CAMERA_TAGS)
    gps_position = _extract_gps(blocks)

    logger.debug(
        "image_context_read",
        subject_count=len(existing_keywords.subject),
        hierarchical_count=len(existing_keywords.hierarchical),
        location_tag_count=len(location_tags),
        camera_tag_count=len(camera_info),
        gps_present=gps_position is not None,
    )
    return ImageContext(
        existing_keywords=existing_keywords,
        existing_title=_first_tag_value(blocks, _TITLE_TAGS),
        existing_description=_first_tag_value(blocks, _DESCRIPTION_TAGS),
        existing_copyright=_first_tag_value(blocks, COPYRIGHT_READ_TAGS),
        location_tags=location_tags,
        gps_position=gps_position,
        camera_info=camera_info,
        content_hash=_extract_content_hash(blocks) if include_content_hash else None,
    )


def _parse_exif_datetime(raw: str) -> datetime | None:
    """
    Parse EXIF's ``YYYY:MM:DD HH:MM:SS`` into a naive datetime, or None when it is unusable.

    Sub-second and offset suffixes are cut off rather than parsed: grouping photos into shoots only
    needs whole seconds, and a camera's clock is its own local time either way.
    """
    try:
        return datetime.strptime(raw.strip()[:19], "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007 - camera local time
    except ValueError:
        return None


def read_capture_times(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, datetime]:
    """
    Read ``EXIF:DateTimeOriginal`` for every path in a single exiftool call.

    Returns naive datetimes in the camera's own local time, which is what a capture timestamp means
    and what makes gaps between frames comparable. Paths whose tag is missing, blank, or malformed
    are simply absent from the result, leaving the caller free to fall back to the file mtime.
    """
    paths = [path for path in image_paths if path.is_file()]
    if not paths:
        return {}

    try:
        blocks = _batched_get_tags(
            et,
            files=[str(p) for p in paths],
            tags=[TAG_EXIF_DATE_TIME_ORIGINAL],
        )
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_capture_times", error=str(exc))
        return {}

    by_source = {path: path for path in paths}
    times: dict[Path, datetime] = {}
    for block in blocks:
        source = block.get("SourceFile")
        raw = block.get(TAG_EXIF_DATE_TIME_ORIGINAL)
        if not source or not raw:
            continue
        image_path = by_source.get(Path(str(source)))
        parsed = _parse_exif_datetime(str(raw))
        if image_path is not None and parsed is not None:
            times[image_path] = parsed
    logger.debug("capture_times_read", requested=len(paths), found=len(times))
    return times


def _year_of(image_path: Path, captured: datetime | None) -> int:
    """
    Return *captured*'s year, or the file's own mtime year, or this year.

    A copyright notice needs a year for every photo, so this always answers. The mtime is the honest
    second choice: for a scan or a photo whose EXIF was stripped, it is usually the day the file was
    made. The current year is the last resort, for a file we cannot even stat.
    """
    if captured is not None:
        return captured.year
    try:
        return datetime.fromtimestamp(image_path.stat().st_mtime, tz=UTC).astimezone().year
    except OSError as exc:
        logger.warning("capture_year_fell_back_to_today", file=str(image_path), error=str(exc))
        return datetime.now(tz=UTC).astimezone().year


def capture_year(image_path: Path, captured: str | None) -> int:
    """
    Return the year a photo was taken, from an already-read ``EXIF:DateTimeOriginal`` string.

    For the pipeline, which reads that tag as prompt context anyway (see :func:`read_image_context`)
    and must not pay a second exiftool call per photo to date a copyright notice. Falls back to the
    file's mtime when the tag is missing or malformed.
    """
    parsed = _parse_exif_datetime(captured) if captured else None
    return _year_of(image_path, parsed)


def read_capture_years(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, int]:
    """
    Return the year every photo in *image_paths* was taken, in one batched exiftool call.

    The counterpart to :func:`capture_year` for callers that have read nothing yet, such as the
    ``set`` command. Every input path gets an answer (see :func:`_year_of`).
    """
    paths = list(image_paths)
    captured = read_capture_times(paths, et=et)
    return {path: _year_of(path, captured.get(path)) for path in paths}


def _first_present(values: dict[str, str], *tags: str) -> str | None:
    """Return the first non-empty value in *values* lookup-ordered by *tags*."""
    for tag in tags:
        if (raw := values.get(tag)) and (stripped := raw.strip()):
            return stripped
    return None


def select_camera_fields(
    camera_info: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Return ``(model, lens, capture_date)`` from a :data:`CAMERA_TAGS` mapping."""
    return (
        camera_info.get("EXIF:Model"),
        camera_info.get("EXIF:LensModel"),
        camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL),
    )


def select_location(location_tags: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Return ``(city, country)`` from a :data:`LOCATION_TAGS` mapping.

    XMP-photoshop and IPTC each define their own city/country fields, so this picks the first
    present in each category, the same fallback order :func:`_format_location` uses for the prompt
    hint and the CSV report uses for its columns.
    """
    return (
        _first_present(location_tags, *CITY_TAGS),
        _first_present(location_tags, *COUNTRY_TAGS),
    )


def _format_location(location_tags: dict[str, str]) -> str | None:
    """
    Build a "City, Country" hint, falling back to whichever single field is set.

    Returning both means the model gets the full place name (e.g. "Barcelona, Spain") instead of
    silently losing the city under the older "first hit wins" logic.
    """
    city, country = select_location(location_tags)
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
    if captured := camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL):
        lines.append(f"- Captured: {captured}")
    return lines


def prompt_with_hint(base_prompt: str, hint: str | None) -> str:
    """
    Append a photographer's note to *base_prompt*; return it unchanged when *hint* is blank.

    The note is framed as authoritative, unlike the Existing Metadata block (which the system prompt
    tells the model to distrust on conflict): a hint exists precisely because the model misread the
    image ("that animal is a deer, not a boar"), so on conflict the note must win.
    """
    hint = (hint or "").strip()
    if not hint:
        return base_prompt
    return (
        f"{base_prompt.strip()}\n\n"
        f"Photographer's note about this photo: {hint}\n"
        "The photographer knows what is in the photo. If this note conflicts with your own "
        "identification, trust the note."
    )


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

    *camera_info* is a mapping of ExifTool tag names (``EXIF:Model`` etc.) to their string values.
    When present, equipment and capture-date hints are appended so the model can take cues from the
    gear used (macro lens, telephoto, wide angle) and the time of year.
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


# Tag groups an XMP sidecar cannot hold. ExifTool drops them without complaint, but a payload left
# with nothing in it fails the write outright ("Nothing to write"), which is what a --set of EXIF
# tags under the default sidecar mode would otherwise do to every photo in the run.
_NON_XMP_GROUPS = ("EXIF", "IPTC")


def _sidecar_writable[V](tags: Mapping[str, V]) -> dict[str, V]:
    """Keep the extra tags an XMP sidecar can actually carry (see :data:`_NON_XMP_GROUPS`)."""
    return {
        tag: value
        for tag, value in tags.items()
        if tag.split(":", 1)[0].upper() not in _NON_XMP_GROUPS
    }


def _build_write_payload(
    keywords: KeywordSet,
    description: str | None,
    title: str | None,
    *,
    use_sidecar: bool,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str | list[str]]:
    """
    Build the exiftool tag map that write_metadata will apply.

    *use_sidecar* decides whether EXIF tags belong in the payload: a sidecar holds XMP and nothing
    else, so they are only written when the target is the photo itself.

    *extra* is the resolved copyright and ``--set`` tags (see :mod:`photo_tagger.fields`). It is
    applied last, so a user who names a tag by hand overrides what the generated fields put there.
    """
    payload: dict[str, str | list[str]] = {}
    if subjects := keywords.subject:
        payload["XMP-dc:Subject"] = subjects
        # Lightroom prioritizes IPTC:Keywords for JPEGs, so mirror the Subject list there.
        payload[TAG_IPTC_KEYWORDS] = subjects
    if hierarchical := keywords.hierarchical:
        payload["XMP-lr:HierarchicalSubject"] = hierarchical
    if weighted := keywords.weighted:
        payload[TAG_XMP_WEIGHTED_FLAT_SUBJECT] = weighted
    if description:
        payload["XMP-dc:Description"] = description
        # ImageDescription lives in the XMP *tiff* namespace (exiftool rejects XMP-exif for it,
        # silently: a warning plus exit 0). exiftool maps it back to IFD0 ImageDescription when a
        # sidecar is folded into the image.
        payload["XMP-tiff:ImageDescription"] = description
        if not use_sidecar:
            # The real IFD0 tag: the XMP mirror above does not touch it on a direct write, so a
            # camera-written description (some write a placeholder) would survive every save.
            payload[TAG_EXIF_IMAGE_DESCRIPTION] = description
    if title:
        payload["XMP-dc:Title"] = title
        payload[TAG_IPTC_OBJECT_NAME] = title
    if extra:
        payload.update(_sidecar_writable(extra) if use_sidecar else extra)
    return payload


# Tag groups a file type cannot hold, by suffix. ExifTool drops such a tag without complaint, so it
# never reads back, the unchanged check sees it as missing on every run, and the file is rewritten
# (and its history grows) each time. WebP was measured; the others follow ExifTool's list of the
# groups each format can write (HEIF and AVIF, CR3, and JPEG XL hold EXIF and XMP; GIF holds XMP).
_UNWRITABLE_GROUPS: dict[str, frozenset[str]] = {
    ".webp": frozenset({"IPTC"}),
    ".heic": frozenset({"IPTC"}),
    ".heif": frozenset({"IPTC"}),
    ".avif": frozenset({"IPTC"}),
    ".cr3": frozenset({"IPTC"}),
    ".jxl": frozenset({"IPTC"}),
    ".gif": frozenset({"IPTC", "EXIF"}),
}


# EXIF's family-1 group names, which a --set may use instead of EXIF itself.
_EXIF_FAMILY_ONE = frozenset({"IFD0", "IFD1", "EXIFIFD", "GPS", "INTEROPIFD", "SUBIFD"})


def _tag_family(tag: str) -> str:
    """
    Return the family-0 group a tag name starts with, upper-cased.

    ``XMP-dc:Title`` is XMP, ``EXIF:IFD0:Copyright`` and ``IFD0:Copyright`` are EXIF, and a tag with
    no group has none.
    """
    group, _sep, _name = tag.rpartition(":")
    first = group.split(":", 1)[0].split("-", 1)[0].upper()
    return "EXIF" if first in _EXIF_FAMILY_ONE else first


def _writable_in(target: Path, tags: Mapping[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Keep the tags *target*'s file type can hold (see :data:`_UNWRITABLE_GROUPS`)."""
    unwritable = _UNWRITABLE_GROUPS.get(target.suffix.lower(), frozenset())
    return {tag: value for tag, value in tags.items() if _tag_family(tag) not in unwritable}


# IPTC's byte limit per dataset, for the ones photo-tagger and a --set commonly write. ExifTool cuts
# a longer value itself, but not on a character boundary, so what it writes never reads back as
# what was sent and the file would be rewritten on every save. photo-tagger cuts first instead.
_IPTC_MAX_BYTES = {
    "By-line": 32,
    "By-lineTitle": 32,
    "Caption-Abstract": 2000,
    "City": 32,
    "CopyrightNotice": 128,
    "Country-PrimaryLocationName": 64,
    "Credit": 32,
    "Headline": 256,
    "Keywords": 64,
    "ObjectName": 64,
    "Province-State": 32,
    "Source": 32,
    "Sub-location": 32,
    "SupplementalCategories": 32,
    "Writer-Editor": 32,
}
_IPTC_CHARSET_TAG = "IPTC:CodedCharacterSet"
# The escape sequence that marks IPTC text as UTF-8, as a read under -n returns it, and its name.
_IPTC_UTF8_VALUES = frozenset({"\x1b%G", "UTF8"})
# Datasets that describe the IPTC block itself and are not copied when it is re-encoded.
_IPTC_STRUCTURAL = frozenset(
    {"CodedCharacterSet", "EnvelopeRecordVersion", "ApplicationRecordVersion"},
)


_IPTC_LIMITS_BY_NAME = {name.casefold(): limit for name, limit in _IPTC_MAX_BYTES.items()}


def _fit_bytes(text: str, limit: int) -> str:
    """Cut *text* to at most *limit* bytes of UTF-8, never inside a character."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _fit_iptc(payload: Mapping[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Cut every IPTC value in *payload* to its dataset's limit (see :data:`_IPTC_MAX_BYTES`)."""
    fitted: dict[str, str | list[str]] = {}
    for tag, value in payload.items():
        if not isinstance(value, str | list):
            fitted[tag] = value  # a number carried over from the file, never too long
            continue
        name = tag.rpartition(":")[2].casefold()
        limit = _IPTC_LIMITS_BY_NAME.get(name) if _tag_family(tag) == "IPTC" else None
        if limit is None:
            fitted[tag] = value
        elif isinstance(value, list):
            fitted[tag] = [
                _fit_bytes(item, limit) if isinstance(item, str) else item for item in value
            ]
        else:
            fitted[tag] = _fit_bytes(value, limit)
    return fitted


def _iptc_recoding(
    pending: Mapping[str, str | list[str]],
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Return the tags that switch *current*'s IPTC block to UTF-8, or nothing when none is needed.

    Without the CodedCharacterSet marker ExifTool writes IPTC as Latin-1, where a name like "Łukasz"
    becomes "?ukasz". The marker alone would make the values already in the file read wrong, so each
    of them is written again with it, in the same call: the block is re-encoded whole. Needed only
    when a new IPTC value is not plain ASCII and the block is not UTF-8 yet. Nothing is re-encoded
    when the block could not be read, since the marker alone would garble what is there.
    """
    if current is None or current.get(_IPTC_CHARSET_TAG) in _IPTC_UTF8_VALUES:
        return {}
    iptc_values = [
        item
        for tag, value in pending.items()
        if _tag_family(tag) == "IPTC"
        for item in (value if isinstance(value, list) else [value])
    ]
    if all(str(item).isascii() for item in iptc_values):
        return {}
    written = {tag.casefold() for tag in pending}
    kept = {
        tag: _recovered(value)
        for tag, value in current.items()
        if tag.startswith("IPTC:")
        and tag.partition(":")[2] not in _IPTC_STRUCTURAL
        and tag.casefold() not in written
        and not _is_binary_placeholder(value)
    }
    fitted = _fit_iptc(kept)
    if fitted != kept:
        logger.warning(
            "iptc_values_cut_to_fit_utf8",
            tags=sorted(t for t in kept if fitted[t] != kept[t]),
        )
    return {_IPTC_CHARSET_TAG: "UTF8", **fitted}


def _is_binary_placeholder(value: Any) -> bool:  # noqa: ANN401 - an exiftool value of any JSON type.
    """Say whether *value* is ExifTool's placeholder for binary data, never written back."""
    return isinstance(value, str) and value.startswith("(Binary data ")


def _recovered(value: Any) -> Any:  # noqa: ANN401 - an exiftool value of any JSON type.
    """
    Undo a UTF-8 value read as Latin-1, as some tools write UTF-8 IPTC without the marker.

    ExifTool then reads "José" as "JosÃ©". Re-encoding that as UTF-8 would keep the garbled text for
    good, so a value that decodes cleanly as UTF-8 once taken back to bytes is taken as UTF-8.
    """
    if isinstance(value, list):
        return [_recovered(item) for item in value]
    if not isinstance(value, str) or value.isascii():
        return value
    try:
        return value.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return value


# The name every save stamps into the files it changes, as digiKam and Lightroom do.
SOFTWARE_AGENT = f"photo-tagger {__version__}"

# IFD0 ProcessingSoftware (0x000b) names the program that edited the photo after the camera, which
# is what digiKam writes. IFD0 Software (0x0131) is not touched: it holds the camera's firmware,
# and a metadata edit is no reason to lose it. xmp:CreatorTool names the tool that made the image,
# so it is left alone too.
TAG_EXIF_PROCESSING_SOFTWARE = "EXIF:ProcessingSoftware"


def _software_stamp(
    payload: Mapping[str, str | list[str]],
    *,
    use_sidecar: bool,
) -> dict[str, str]:
    """
    Return the tag that names photo-tagger as the program behind a save.

    Empty for a sidecar, which holds XMP only, and when *payload* already names the tag (a ``--set``
    by hand wins).
    """
    if use_sidecar or any(
        tag.casefold().rpartition(":")[2] == "processingsoftware" for tag in payload
    ):
        return {}
    return {TAG_EXIF_PROCESSING_SOFTWARE: SOFTWARE_AGENT}


def history_event_param(when: datetime) -> str:
    """
    Return the ExifTool argument that appends one save to the file's ``xmpMM:History``.

    The event reads the way Lightroom logs a metadata save. The history is a list, so it goes in
    as a parameter with ``+=``: pyexiftool's tag map can only replace a tag.

    Examples:
        >>> history_event_param(datetime(2026, 9, 24, 10, 0, tzinfo=UTC)).startswith(
        ...     "-XMP-xmpMM:History+={Action=saved,When=2026-09-24T10:00:00+00:00,"
        ... )
        True
    """
    return (
        f"-XMP-xmpMM:History+={{Action=saved,When={when.isoformat(timespec='seconds')},"
        f"SoftwareAgent={SOFTWARE_AGENT},Changed=/metadata}}"
    )


def is_raw_file(image_path: Path) -> bool:
    """
    Report whether *image_path* looks like a camera RAW file, by extension alone.

    Decided by exclusion: a suffix Pillow reads natively is not RAW, anything else is. An unknown
    extension counting as RAW is the cautious answer for the caller that matters here, the "raw"
    sidecar mode, which then leaves the file's own bytes alone.
    """
    return image_path.suffix.lower() not in NON_RAW_EXTENSIONS


def use_sidecar_for(image_path: Path, mode: SidecarMode) -> bool:
    """
    Report whether *image_path* gets an XMP sidecar under *mode*.

    True for "both", which also writes the photo: ask :func:`plan_writes` for the full answer.

    Examples:
        >>> use_sidecar_for(Path("/photos/image.dng"), "raw")
        True
        >>> use_sidecar_for(Path("/photos/image.jpg"), "raw")
        False
    """
    match mode:
        case "none":
            return False
        case "raw":
            return is_raw_file(image_path)
        case _:
            return True


def write_target(image_path: Path, *, use_sidecar: bool) -> Path:
    """
    Return the file :func:`write_metadata` will modify for *image_path*.

    Shared so callers that need to inspect the target before the write (the undo journal, which
    must know whether the sidecar already existed) cannot drift from the writer's own choice.

    Examples:
        >>> write_target(Path("/photos/image.cr3"), use_sidecar=True).name
        'image.xmp'
        >>> write_target(Path("/photos/image.cr3"), use_sidecar=False).name
        'image.cr3'
    """
    return image_path.with_suffix(".xmp") if use_sidecar else image_path


@dataclass(slots=True, frozen=True)
class WriteTarget:
    """One file a save writes for a photo."""

    path: Path
    # Whether this target is the sidecar, which decides the payload: a sidecar holds XMP only.
    use_sidecar: bool
    # Whether the file was there before the write. Undo needs it to tell "restore the backup" from
    # "delete what the run created", and it can only be known beforehand.
    existed: bool


# What each --sidecar-override target writes, as use_sidecar values: photo before sidecar, the
# order metadata_targets reads them in.
_TARGET_SIDECARS: dict[FileTypeTarget, tuple[bool, ...]] = {
    "photo": (False,),
    "sidecar": (True,),
    "both": (False, True),
}
# The same keys by their spelling, so a parsed string comes out typed without a cast.
_TARGET_NAMES: dict[str, FileTypeTarget] = {name: name for name in _TARGET_SIDECARS}


def parse_sidecar_overrides(entries: Iterable[str] | None) -> dict[str, FileTypeTarget]:
    """
    Parse ``EXT=TARGET`` entries into overrides keyed by lowercase suffix.

    The extension takes any case and an optional dot, so ``DNG=photo`` and ``.dng=photo`` agree. A
    later entry for the same extension wins, like a repeated flag. Raises :class:`ValueError` on an
    entry that is not ``EXT=TARGET``, or that names the sidecar extension itself.

    Examples:
        >>> parse_sidecar_overrides(["DNG=photo", "tif=both"])
        {'.dng': 'photo', '.tif': 'both'}
    """
    overrides: dict[str, FileTypeTarget] = {}
    for entry in entries or ():
        extension, _, name = entry.partition("=")
        suffix = "." + extension.strip().lstrip(".").lower()
        target = _TARGET_NAMES.get(name.strip().lower())
        if target is None or suffix == ".":
            msg = (
                f"--sidecar-override {entry!r} must be EXT=TARGET, where TARGET is sidecar, "
                f"photo, or both. For example: dng=photo"
            )
            raise ValueError(msg)
        if suffix == ".xmp":
            msg = f"--sidecar-override {entry!r}: an XMP sidecar cannot get a sidecar of its own"
            raise ValueError(msg)
        overrides[suffix] = target
    return overrides


def _sidecar_choices(
    image_path: Path,
    mode: SidecarMode,
    overrides: SidecarOverrides | None,
) -> tuple[bool, ...]:
    """Return the use_sidecar value of each file a save writes, photo first."""
    if overrides and (target := overrides.get(image_path.suffix.lower())) is not None:
        return _TARGET_SIDECARS[target]
    return (False, True) if mode == "both" else (use_sidecar_for(image_path, mode),)


def plan_writes(
    image_path: Path,
    mode: SidecarMode,
    overrides: SidecarOverrides | None = None,
) -> list[WriteTarget]:
    """
    List the files a save under *mode* writes for *image_path*, as they are before the write.

    One target for every mode but "both", which writes the photo *and* a sidecar beside it: the
    photo travels with its metadata, and the sidecar is what a catalog edits without touching the
    photo again. An entry in *overrides* for the photo's extension wins over *mode*, so a DNG can
    carry its own metadata (as Lightroom Classic expects) while every other file gets a sidecar.

    Examples:
        >>> [t.path.name for t in plan_writes(Path("/photos/image.cr3"), "both")]
        ['image.cr3', 'image.xmp']
        >>> [t.path.name for t in plan_writes(Path("/photos/image.dng"), "all", {".dng": "photo"})]
        ['image.dng']
    """
    targets: list[WriteTarget] = []
    for use_sidecar in _sidecar_choices(image_path, mode, overrides):
        path = write_target(image_path, use_sidecar=use_sidecar)
        targets.append(WriteTarget(path=path, use_sidecar=use_sidecar, existed=path.exists()))
    return targets


def write_metadata_everywhere(  # noqa: PLR0913 - distinct optional fields are clearer as kwargs.
    image_path: Path,
    keywords: KeywordSet,
    *,
    description: str | None = None,
    title: str | None = None,
    extra: Mapping[str, str] | None = None,
    backup: bool = True,
    sidecar_mode: SidecarMode,
    sidecar_overrides: SidecarOverrides | None = None,
    et: ExifToolHelper | None = None,
) -> tuple[bool, list[WriteTarget]]:
    """
    Write *image_path*'s metadata everywhere *sidecar_mode* asks for.

    An entry in *sidecar_overrides* for the photo's extension wins over the mode (see
    :func:`plan_writes`).

    Returns whether every target was saved, and the ones that were written, for the caller's undo
    journal. A target that already carried every value counts as saved but was not touched, so it is
    left out. A partial failure under "both" still reports what landed, so undo can revert it.
    """
    targets = plan_writes(image_path, sidecar_mode, sidecar_overrides)
    saved = 0
    written: list[WriteTarget] = []
    for target in targets:
        result = write_metadata(
            image_path,
            keywords,
            description=description,
            title=title,
            extra=extra,
            backup=backup,
            use_sidecar=target.use_sidecar,
            et=et,
        )
        if not result:
            continue
        saved += 1
        if result != WriteResult.UNCHANGED:
            written.append(target)
    return (saved == len(targets), written)


# What exiftool puts in front of an error it would downgrade to a warning under -m. Matched
# case-insensitively: exiftool spells it "[minor]" in errors and "[Minor]" in some warnings.
_MINOR_ERROR_MARKER = "[minor]"

# exiftool's "ignore minor errors and warnings" flag.
_IGNORE_MINOR_ERRORS = "-m"


def _is_minor_error(exc: ExifToolExecuteError) -> bool:
    """Report whether exiftool refused the write over something ``-m`` would let through."""
    streams = f"{exc.stderr or ''}\n{exc.stdout or ''}"
    return _MINOR_ERROR_MARKER in streams.casefold()


class WriteResult(IntEnum):
    """
    What :func:`write_metadata` did to its target.

    An IntEnum so the old bool contract still holds: FAILED is falsy, and WRITTEN equals True.
    UNCHANGED is the case a bool cannot say: the file already carried every value, so it was not
    touched, and the undo journal must not claim it.
    """

    FAILED = 0
    WRITTEN = 1
    UNCHANGED = 2


# The family-0 group ExifTool reports a written group under (its default -G read spelling).
# Groups not listed here cannot be matched to what a read returns, so their tags always count as
# changed and are written.
_READ_GROUPS = {"exif": "EXIF", "iptc": "IPTC", "xmp": "XMP"}


def _read_key(tag: str) -> str | None:
    """Return the key a read reports *tag* under (``XMP-dc:Title`` is ``XMP:Title``), or None."""
    group, _sep, name = tag.rpartition(":")
    family = _READ_GROUPS.get(group.split("-", 1)[0].casefold())
    return f"{family}:{name}" if family and name else None


def _as_values(value: Any) -> list[str]:  # noqa: ANN401 - an exiftool value of any JSON type.
    """
    Normalize a tag value for comparison: a list of strings, blanks dropped.

    A read returns a one-item list as a plain string and a boolean as ``True``, while the payload
    spells them ``["Duck"]`` and ``"True"``. An empty result also stands for a blank write, which is
    how ExifTool deletes a tag, so "delete" matches a tag that is not there.
    """
    if value is None:
        return []
    return _coerce_to_list(value)


def _changed_payload(
    payload: dict[str, str | list[str]],
    current: Mapping[str, Any],
    *,
    use_sidecar: bool,
) -> dict[str, str | list[str]]:
    """
    Keep the entries of *payload* that differ from *current*, the target's own values.

    Two tags that read back under one key (two XMP namespaces with the same tag name) cannot be told
    apart, so both stay in the payload. A sidecar drops the IPTC mirrors of the keywords and the
    title (it holds XMP only), so they never read back and cannot change it.
    """
    if use_sidecar:
        payload = dict(_sidecar_writable(payload))
    # Tag names are case-insensitive to ExifTool, so ``xmp-dc:rights`` reads back as XMP:Rights.
    keys = {tag: key.casefold() if (key := _read_key(tag)) else None for tag in payload}
    known = {key.casefold(): value for key, value in current.items()}
    counts: dict[str | None, int] = {}
    for key in keys.values():
        counts[key] = counts.get(key, 0) + 1
    return {
        tag: value
        for tag, value in payload.items()
        if (key := keys[tag]) is None
        or counts[key] > 1
        or _as_values(known.get(key)) != _as_values(value)
    }


def _read_list(payload: Mapping[str, str | list[str]]) -> list[str]:
    """
    Name what to read before a write: the payload's tags, and the IPTC block if it writes IPTC.

    A switch to UTF-8 re-encodes every value in that block (see :func:`_iptc_recoding`).
    """
    tags = list(payload)
    if any(_tag_family(tag) == "IPTC" for tag in tags):
        tags.append("IPTC:all")
    return tags


def _read_current(
    helper: ExifToolHelper,
    target: Path,
    tags: Iterable[str],
) -> dict[str, Any] | None:
    """Read *tags* off *target* as it is now, or None when there is nothing to compare with."""
    if not target.exists():
        return None
    try:
        blocks = helper.get_tags(files=[str(target)], tags=list(tags))
    except _EXIFTOOL_ERRORS as exc:
        # A failed read only costs the comparison: the write goes ahead as before.
        logger.warning("metadata_compare_read_failed", target=str(target), error=str(exc))
        return None
    return dict(blocks[0]) if blocks else None


def _set_tags(helper: ExifToolHelper, set_kwargs: dict[str, Any], target: Path) -> None:
    """
    Apply the payload, retrying once with ``-m`` when exiftool refuses over a minor error.

    Some cameras write maker notes exiftool cannot parse, and it refuses the whole write rather than
    move a block whose offsets it cannot fix. The retry is the only way to embed metadata in those
    files. Not passed up front: the check is worth having for every other photo, and each waiver is
    logged.
    """
    try:
        helper.set_tags(**set_kwargs)
    except ExifToolExecuteError as exc:
        if not _is_minor_error(exc):
            raise
        logger.warning(
            "retrying_write_ignoring_minor_errors",
            target=str(target),
            # The parens bind .strip() to the whole or-chain, not to the "" literal.
            reason=(exc.stderr or exc.stdout or "").strip(),  # NOSONAR S1110
        )
        retry_kwargs: dict[str, Any] = dict(set_kwargs)
        retry_kwargs["params"] = [*set_kwargs.get("params", []), _IGNORE_MINOR_ERRORS]
        helper.set_tags(**retry_kwargs)


def write_metadata(  # noqa: PLR0913 - distinct optional fields are clearer as kwargs.
    image_path: Path,
    keywords: KeywordSet,
    *,
    description: str | None = None,
    title: str | None = None,
    extra: Mapping[str, str] | None = None,
    backup: bool = True,
    use_sidecar: bool = True,
    et: ExifToolHelper | None = None,
) -> WriteResult:
    """
    Write tags to either the image file or an XMP sidecar in Lightroom-compatible format.

    Only the tags whose value differs from what the target already carries are written. ExifTool
    rewrites a file even for identical values, which moves its mtime, and a backup tool such as
    rsync then copies an unchanged photo again. A write that does happen also names photo-tagger
    as its software (see :data:`SOFTWARE_AGENT`).

    Args:
        image_path: Path to the image file. The sidecar shares its name with a `.xmp` extension.
        keywords: A :class:`KeywordSet` of subject, hierarchical, and weighted keywords.
        description: Optional short description to write to XMP (and, when embedding, to
            EXIF:ImageDescription, so no camera-written value survives underneath).
        title: Optional short title to write to XMP-dc:Title and IPTC:ObjectName.
        extra: Optional resolved tag map for the fields the model has no part in (copyright,
            creator, raw `--set` assignments). See :mod:`photo_tagger.fields`.
        backup: If True, let ExifTool create a backup (`_original` suffix where applicable).
        use_sidecar: If True, write to a sidecar; otherwise embed in the source file.
        et: Optional already-open ExifToolHelper to reuse across a batch.

    Returns:
        WRITTEN on success, UNCHANGED when the target already carried every value (it is not
        touched), and FAILED on failure.
    """
    target_path = write_target(image_path, use_sidecar=use_sidecar)
    payload = _build_write_payload(
        keywords,
        description,
        title,
        use_sidecar=use_sidecar,
        extra=extra,
    )
    if not use_sidecar:
        payload = _fit_iptc(_writable_in(target_path, payload))
    if not payload:
        logger.warning("no_data_to_write", file=image_path.name)
        return WriteResult.FAILED

    try:
        with managed_helper(et) as helper:
            current = _read_current(helper, target_path, _read_list(payload))
            pending = (
                payload
                if current is None
                else _changed_payload(payload, current, use_sidecar=use_sidecar)
            )
            if not pending:
                logger.info("metadata_unchanged", target=str(target_path))
                return WriteResult.UNCHANGED
            # The stamp joins only a write that happens anyway. Compared like the rest, it would
            # never match (the history grows with every save), and no file would stay untouched.
            params: list[str] = [] if backup else ["-overwrite_original"]
            params.append(history_event_param(datetime.now().astimezone()))
            stamp = _writable_in(target_path, _software_stamp(payload, use_sidecar=use_sidecar))
            recoding = {} if use_sidecar else _iptc_recoding(pending, current)
            set_kwargs: dict[str, Any] = {
                "tags": pending | stamp | recoding,
                "files": [str(target_path)],
                "params": params,
            }
            _set_tags(helper, set_kwargs, target_path)
    except _EXIFTOOL_ERRORS as e:
        logger.exception("xmp_write_failed", error=str(e), target=str(target_path))
        return WriteResult.FAILED

    logger.info(
        "metadata_written_successfully",
        target=str(target_path),
        mode="sidecar" if use_sidecar else "embedded",
        subject_keywords=len(keywords.subject),
        hierarchical_keywords=len(keywords.hierarchical),
        extra_tags=len(extra or {}),
        changed_tags=len(pending),
        backup_created=backup,
    )
    return WriteResult.WRITTEN
