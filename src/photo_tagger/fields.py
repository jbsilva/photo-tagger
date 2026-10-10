"""
The metadata a run writes that the model has no part in: authorship, rights, anything else.

photo-tagger opens a photo to write a title, a description, and keywords. Two things no model can
know belong in the same write: who took the photo, and who owns it. This module holds that schema.
It turns a name and a notice template into the exiftool tags that carry them, and it lists the
fields ``photo-tagger show`` reads back.

Pure and IO-free. :mod:`photo_tagger.metadata` makes the exiftool calls; this decides what goes in
them. What a template reads from the photo (its capture date, the name already on it) arrives as a
mapping, read before the write (see :func:`photo_tagger.metadata.read_field_sources`).
"""

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.config import (
    CITY_TAGS,
    COUNTRY_TAGS,
    TAG_EXIF_DATE_TIME_ORIGINAL,
    TAG_EXIF_IMAGE_DESCRIPTION,
    TAG_IPTC_KEYWORDS,
    TAG_IPTC_OBJECT_NAME,
    TAG_XMP_DESCRIPTION,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_IMAGE_DESCRIPTION,
    TAG_XMP_SUBJECT,
    TAG_XMP_TITLE,
)
from photo_tagger.errors import FieldError
from photo_tagger.templates import (
    SETTING_ARTIST,
    MissingValueError,
    Template,
    parse_template,
    sample_values,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


# Authorship and rights tags, in the write spelling (the specific namespace). Each field is written
# to every namespace that matters, the way the keyword and title writes already mirror XMP into
# IPTC: catalogs read XMP, camera software and older tools read EXIF and IPTC, and a photo that
# answers only one of them looks uncredited everywhere else.
TAG_EXIF_ARTIST = "EXIF:Artist"
TAG_EXIF_COPYRIGHT = "EXIF:Copyright"
TAG_IPTC_BY_LINE = "IPTC:By-line"
TAG_IPTC_COPYRIGHT_NOTICE = "IPTC:CopyrightNotice"
TAG_XMP_CREATOR = "XMP-dc:Creator"
TAG_XMP_RIGHTS = "XMP-dc:Rights"
TAG_XMP_MARKED = "XMP-xmpRights:Marked"

# The same XMP fields in the read spelling. ExifTool reports the family 0 group ("XMP:Creator"), so
# a read asking for "XMP-dc:Creator" gets its answer back under a key nobody looks up.
READ_XMP_CREATOR = "XMP:Creator"
READ_XMP_RIGHTS = "XMP:Rights"
READ_XMP_MARKED = "XMP:Marked"

# Every tag a copyright notice may sit in, in the read spelling, best first.
COPYRIGHT_READ_TAGS: tuple[str, ...] = (
    READ_XMP_RIGHTS,
    TAG_EXIF_COPYRIGHT,
    TAG_IPTC_COPYRIGHT_NOTICE,
)
# The name already on a photo, in the read spelling. With preserve on, any of them keeps the name.
ARTIST_READ_TAGS: tuple[str, ...] = (TAG_EXIF_ARTIST, READ_XMP_CREATOR, TAG_IPTC_BY_LINE)


@dataclass(slots=True, frozen=True)
class FieldSpec:
    """One line of the ``show`` report: a label, and the tags that may carry it, best first."""

    label: str
    tags: tuple[str, ...]


# What ``photo-tagger show`` reports, in reading order. Title and description use the readers'
# tag priority (see metadata.py), so the value shown first is the one a catalog picks up.
SHOWN_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("Title", (TAG_XMP_TITLE, TAG_IPTC_OBJECT_NAME)),
    FieldSpec(
        "Description",
        (TAG_XMP_DESCRIPTION, TAG_XMP_IMAGE_DESCRIPTION, TAG_EXIF_IMAGE_DESCRIPTION),
    ),
    FieldSpec("Keywords", (TAG_XMP_SUBJECT, TAG_IPTC_KEYWORDS)),
    FieldSpec("Hierarchy", (TAG_XMP_HIERARCHICAL_SUBJECT,)),
    FieldSpec("Creator", (READ_XMP_CREATOR, TAG_EXIF_ARTIST, TAG_IPTC_BY_LINE)),
    FieldSpec("Copyright", COPYRIGHT_READ_TAGS),
    FieldSpec("Rights marked", (READ_XMP_MARKED,)),
    FieldSpec("Camera", ("EXIF:Model",)),
    FieldSpec("Lens", ("EXIF:LensModel",)),
    FieldSpec("Captured", (TAG_EXIF_DATE_TIME_ORIGINAL,)),
    FieldSpec("City", CITY_TAGS),
    FieldSpec("Country", COUNTRY_TAGS),
    FieldSpec("GPS", ("Composite:GPSPosition",)),
)


def _parsed(template: str | None, flag: str) -> Template | None:
    """Parse a notice template, None and blank ones staying unwritten."""
    return parse_template(template, flag=flag) if template else None


def _any_present(values: Mapping[str, str], tags: tuple[str, ...]) -> bool:
    """Whether any of *tags* has a non-blank value in *values*."""
    return any(values.get(tag, "").strip() for tag in tags)


@dataclass(slots=True, frozen=True)
class ExtraFields:
    """
    Authorship, rights, and raw tag assignments applied to every photo in a run.

    *notice* and *rights* are templates, not finished text (see :mod:`photo_tagger.templates`): the
    year belongs to the photo, so ``"Copyright (c) {year} {artist}"`` becomes a different string for
    a 2019 frame than for a 2025 one. *rights* is the XMP wording when it has to differ from the
    rest (a real ``©`` where EXIF gets ``(c)``), and falls back to *notice*.

    *marked* is the XMP "this work is protected" flag: True for a copyrighted photo, False to
    declare it public domain, None to leave whatever the file says alone. *tags* holds raw
    ``TAG=VALUE`` assignments and is applied last, so it can override anything above it.

    *preserve* keeps what a photo already says. A photo with a name keeps it, and a photo with a
    notice keeps the notice, the rights, and the flag. So a second run never re-dates a notice, and
    a folder with photos from other people keeps their credit.
    """

    artist: str | None = None
    notice: str | None = None
    rights: str | None = None
    marked: bool | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    preserve: bool = True
    _notice: Template | None = field(init=False, repr=False, compare=False)
    _rights: Template | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Parse the templates once, raising :class:`FieldError` when one is malformed."""
        object.__setattr__(self, "_notice", _parsed(self.notice, "--copyright"))
        object.__setattr__(self, "_rights", _parsed(self.rights, "--rights"))

    @property
    def is_empty(self) -> bool:
        """Report whether this would write nothing, so callers can skip the work entirely."""
        return not (self.artist or self.notice or self.rights or self.tags) and self.marked is None

    @property
    def writes_notice(self) -> bool:
        """Whether a notice, the rights, or the flag is configured: the fields preserve guards."""
        return bool(self.notice or self.rights) or self.marked is not None

    @property
    def source_tags(self) -> tuple[str, ...]:
        """Tags :meth:`resolve` reads: what the templates name, and what it keeps."""
        tags = [
            tag for template in (self._notice, self._rights) if template for tag in template.tags()
        ]
        if self.preserve and self.artist:
            tags += ARTIST_READ_TAGS
        if self.preserve and self.writes_notice:
            tags += COPYRIGHT_READ_TAGS
        return tuple(dict.fromkeys(tags))

    def resolve(
        self,
        values: Mapping[str, str],
        *,
        existing: Mapping[str, str] | None = None,
        photo: str = "",
        skipped: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Render the exiftool tag map for one photo, from its *values* (see :attr:`source_tags`).

        *existing* is what the files this save writes already carry, which is all that preserve
        looks at. It defaults to *values*, for a caller that writes every file it read.

        Returns an empty map when nothing was configured. A blank value is kept rather than dropped:
        ``--set XMP-dc:Rights=`` is how ExifTool deletes a tag, and that is a thing people want from
        a tool that writes rights metadata. A template that finds nothing on this photo leaves its
        field unwritten, with a warning that names *photo*, and adds the option to *skipped*.
        """
        existing = values if existing is None else existing
        payload: dict[str, str] = {}
        keep_creator = self.preserve and _any_present(existing, ARTIST_READ_TAGS)
        if self.artist and not keep_creator:
            payload[TAG_EXIF_ARTIST] = self.artist
            payload[TAG_XMP_CREATOR] = self.artist
            payload[TAG_IPTC_BY_LINE] = self.artist
        if not (self.preserve and _any_present(existing, COPYRIGHT_READ_TAGS)):
            # A kept creator is the photo's author, so {artist} names them, not the --artist.
            artist = "" if keep_creator else (self.artist or "")
            missing = skipped if skipped is not None else []
            payload |= self._notice_tags(values, artist=artist, photo=photo, skipped=missing)
        payload.update(self.tags)
        return payload

    def preview(self, year: int) -> dict[str, str]:
        """Render the tags for a photo taken in *year* with nothing on it yet, for a preview."""
        values = {
            tag: value
            for template in (self._notice, self._rights)
            if template
            for tag, value in sample_values(template, year).items()
        }
        return replace(self, preserve=False).resolve(values)

    def _notice_tags(
        self,
        values: Mapping[str, str],
        *,
        artist: str,
        photo: str,
        skipped: list[str],
    ) -> dict[str, str]:
        """Render the notice, the rights, and the flag."""
        payload: dict[str, str] = {}
        settings = {SETTING_ARTIST: artist}
        notice = self._render(self._notice, values, settings, flag="--copyright", photo=photo)
        if notice is None and self._notice is not None:
            skipped.append("--copyright")
        if notice:
            payload[TAG_EXIF_COPYRIGHT] = notice
            payload[TAG_IPTC_COPYRIGHT_NOTICE] = notice
        rights = self._render(self._rights, values, settings, flag="--rights", photo=photo)
        if rights is None and self._rights is not None and not notice:
            # XMP falls back to the notice, so the field is unwritten only when both find nothing.
            skipped.append("--rights")
        if rights := (rights or notice):
            payload[TAG_XMP_RIGHTS] = rights
        if self.marked is not None:
            # ExifTool spells this boolean out; the XMP value is the literal string.
            payload[TAG_XMP_MARKED] = "True" if self.marked else "False"
        return payload

    def _render(
        self,
        template: Template | None,
        values: Mapping[str, str],
        settings: Mapping[str, str],
        *,
        flag: str,
        photo: str,
    ) -> str | None:
        """Fill one template, or return None (with a warning) when it finds nothing to say."""
        if template is None:
            return None
        try:
            return template.render(values, settings)
        except MissingValueError as exc:
            logger.warning(
                "copyright_field_skipped",
                flag=flag,
                placeholder=exc.placeholder,
                file=photo,
            )
            return None


# The tags a reader is entitled to treat as 7-bit ASCII. EXIF types Artist (0x013B) and Copyright
# (0x8298) as ASCII; UTF-8 only became legal in Exif 3.0 (CIPA DC-008-2023). ExifTool writes the
# UTF-8 bytes anyway, and nothing in the file says what they are, so a '©' there is a bet on the
# reader. XMP is UTF-8 by definition and takes anything.
ASCII_ONLY_TAGS: tuple[str, ...] = (TAG_EXIF_ARTIST, TAG_EXIF_COPYRIGHT)


def non_ascii_tags(payload: Mapping[str, str]) -> list[str]:
    """
    Name the tags in *payload* whose value needs more than ASCII, in :data:`ASCII_ONLY_TAGS` order.

    Not an error: the write goes ahead, because plenty of software does read these fields as UTF-8,
    and a photographer called José should not have to spell their own name differently. It is what
    lets the frontends say which value is a bet rather than leaving it to be discovered later.
    """
    return [tag for tag in ASCII_ONLY_TAGS if not payload.get(tag, "").isascii()]


# One tag, optionally behind groups: ``Artist``, ``XMP-dc:Rights``, ``EXIF:IFD0:Copyright``. No
# wildcard, and no operator glued to the name: ``Keywords+=`` edits a list, which --set does not.
_TAG_NAME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9-]*:)*[A-Za-z][A-Za-z0-9_-]*$")
# Groups whose "tags" are facts about the file, not metadata in it. Writing them renames, moves, or
# re-dates the file, or makes a link to it.
_FILE_GROUPS = frozenset({"file", "system", "composite", "exiftool"})
# The same pseudo-tags, named without their group, and ExifTool's own options spelled as tags.
# cspell:disable - ExifTool tag names in lower case, not words.
_PSEUDO_TAGS = frozenset(
    {
        "all",
        "directory",
        "filename",
        "filecreatedate",
        "filegroupid",
        "filemodifydate",
        "filepermissions",
        "fileuserid",
        "forcewrite",
        "geosync",
        "geotag",
        "geotime",
        "hardlink",
        "symlink",
        "testname",
        # ExifTool options, which it reads as such even when spelled like a tag.
        "addtagsfromfile",
        "alltagsfromfile",
        "csv",
        "j",
        "json",
        "tagsfromfile",
    },
)
# Prefixes of the macOS extended attributes and Spotlight fields ExifTool offers as tags.
_PSEUDO_PREFIXES = ("xattr", "mditem")
# cspell:enable - back to checking.


def _check_tag_name(tag: str, item: str) -> None:
    """Refuse a --set target that is not one plain metadata tag (see :data:`_PSEUDO_TAGS`)."""
    groups, _sep, name = tag.rpartition(":")
    if not _TAG_NAME.match(tag):
        reason = "is not a tag name (letters, digits, '-' and '_', optionally behind GROUP:)"
    elif name.casefold() in _PSEUDO_TAGS or name.casefold().startswith(_PSEUDO_PREFIXES):
        reason = "is not metadata in the photo: writing it would change the file itself"
    elif any(group.casefold() in _FILE_GROUPS for group in groups.split(":") if group):
        reason = "names a group of file facts, not metadata in the photo"
    else:
        return
    logger.error("field_assignment_refused", item=item, tag=tag)
    msg = f"--set {item!r}: {tag!r} {reason}"
    raise FieldError(msg)


def parse_assignment(item: str) -> tuple[str, str]:
    """
    Split one ``TAG=VALUE`` assignment, raising :class:`FieldError` when it is not one.

    A leading dash is dropped so a line copied out of an ExifTool command (``-Artist=Ada``) works
    as it stands. The value is taken verbatim, including an empty one, which deletes the tag.

    Examples:
        >>> parse_assignment("XMP-xmpRights:WebStatement=https://example.com/terms")
        ('XMP-xmpRights:WebStatement', 'https://example.com/terms')
        >>> parse_assignment("-Artist=Ada Lovelace")
        ('Artist', 'Ada Lovelace')
    """
    tag, separator, value = item.partition("=")
    tag = tag.strip().lstrip("-")
    if not separator or not tag:
        logger.error("field_assignment_malformed", item=item)
        msg = f"--set expects TAG=VALUE, got {item!r} (e.g. --set XMP-dc:Rights='All rights reserved')"  # noqa: E501 - breaking this line would hide the example it carries.
        raise FieldError(msg)
    _check_tag_name(tag, item)
    return tag, value


def build_extra_fields(  # noqa: PLR0913 - one keyword per copyright setting.
    *,
    artist: str | None = None,
    notice: str | None = None,
    rights: str | None = None,
    marked: bool | None = None,
    assignments: Iterable[str] | None = None,
    preserve: bool = True,
) -> ExtraFields:
    """
    Validate what the user asked for and bundle it into an :class:`ExtraFields`.

    Raises :class:`FieldError` on a malformed ``--set`` or template, so a typo stops the run before
    the first photo is written rather than stamping five hundred of them with
    ``Copyright (c) {shot_at}``.
    """
    artist = (artist or "").strip() or None
    tags = dict(parse_assignment(item) for item in (assignments or []))
    fields = ExtraFields(
        artist=artist,
        notice=notice,
        rights=rights,
        marked=marked,
        tags=tags,
        preserve=preserve,
    )
    if not fields.is_empty:
        logger.debug(
            "extra_fields_configured",
            artist=bool(artist),
            notice=bool(notice),
            rights=bool(rights),
            marked=marked,
            raw_tags=len(tags),
            preserve=preserve,
        )
    return fields
