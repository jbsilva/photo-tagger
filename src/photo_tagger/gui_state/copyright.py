"""The copyright fields as the window previews them before a save writes them."""

from typing import TYPE_CHECKING

from photo_tagger.errors import FieldError
from photo_tagger.fields import (
    ARTIST_READ_TAGS,
    COPYRIGHT_READ_TAGS,
    TAG_EXIF_COPYRIGHT,
    TAG_XMP_RIGHTS,
    ExtraFields,
    build_extra_fields,
    non_ascii_tags,
)
from photo_tagger.i18n import _, ngettext
from photo_tagger.templates import ARTIST_TAGS, YEAR_TAGS, format_date


if TYPE_CHECKING:
    from collections.abc import Iterable

    from photo_tagger.metadata import (
        FieldSources,
    )
    from photo_tagger.write_plan import WriteTarget


def capture_year(sources: FieldSources) -> int | None:
    """Return the year from the photo's first capture date (see ``{year}``), else None."""
    values = sources.values
    for tag in YEAR_TAGS:
        if (year := format_date(values.get(tag, ""), "%Y")) is not None:
            return int(year)
    return None


def field_tags_to_read(fields: ExtraFields | None) -> tuple[str, ...]:
    """
    Name the tags to read with a photo's existing metadata, for the notice the detail pane shows.

    The default chains and the notice already there always, plus whatever the current templates
    name, so editing the dialog rarely needs a new read.
    """
    tags = [*YEAR_TAGS, *ARTIST_TAGS, *ARTIST_READ_TAGS, *COPYRIGHT_READ_TAGS]
    if fields is not None:
        tags += fields.source_tags
    return tuple(dict.fromkeys(tags))


def build_copyright_fields(  # noqa: PLR0913 - one keyword per dialog field.
    *,
    artist: str,
    notice: str,
    rights: str,
    marked: bool | None,
    assignments: Iterable[str] | None = None,
    preserve: bool = True,
) -> tuple[ExtraFields | None, str]:
    """
    Validate what the Copyright dialog holds: the fields, or the message refusing them.

    Returns ``(fields, "")`` when the dialog is usable and ``(None, message)`` when it is not, so
    the Qt shell shows the reason in the dialog instead of raising out of a slot. *assignments*
    carries raw ``TAG=VALUE`` entries from the config file's ``set_tags``, which the window writes
    and the preview lists but the dialog does not edit.
    """
    try:
        fields = build_extra_fields(
            artist=artist.strip(),
            notice=notice.strip(),
            rights=rights.strip(),
            marked=marked,
            assignments=assignments,
            preserve=preserve,
        )
    except FieldError as exc:
        return None, str(exc)
    return fields, ""


def copyright_summary(fields: ExtraFields | None, year: int) -> str:
    """
    Describe what the copyright settings put on a photo taken in *year*.

    The dialog shows this under the fields, because ``{year}`` only means something once it has a
    photo to stand for: seeing ``Copyright (c) 2019 ...`` is what tells you the template is right.
    """
    if fields is None or fields.is_empty:
        return _("Nothing is written. Fill in a name or a notice above.")
    resolved = fields.preview(year)
    lines = [
        ngettext(
            "Writes {n} tag. On a photo taken in {year}:",
            "Writes {n} tags. On a photo taken in {year}:",
            len(resolved),
        ).format(n=len(resolved), year=year),
    ]
    lines += [f"{tag} = {value}" for tag, value in resolved.items()]
    return "\n".join(lines)


def planned_copyright(
    fields: ExtraFields | None,
    sources: FieldSources,
    targets: Iterable[WriteTarget] = (),
) -> str | None:
    """
    Return the notice a save to *targets* puts on a photo, or None when it writes none.

    The XMP wording comes first, the same order the Existing side reads the file in, so the two
    columns compare like with like.
    """
    if fields is None or fields.is_empty:
        return None
    payload = fields.resolve(sources.values, existing=sources.written(targets))
    return payload.get(TAG_XMP_RIGHTS) or payload.get(TAG_EXIF_COPYRIGHT) or None


def keeps_copyright(
    fields: ExtraFields | None,
    sources: FieldSources,
    targets: Iterable[WriteTarget] = (),
) -> bool:
    """Whether a save to *targets* keeps the notice: those files have one, and preserve is on."""
    existing = sources.written(targets)
    return (
        fields is not None
        and fields.preserve
        and fields.writes_notice
        and any(existing.get(tag, "").strip() for tag in COPYRIGHT_READ_TAGS)
    )


def ascii_warning(fields: ExtraFields | None, year: int) -> str:
    """
    Warn that a value is riding in a field the EXIF spec types as ASCII, or return "".

    EXIF only took UTF-8 from Exif 3.0 on (CIPA DC-008-2023), and nothing in the file says which
    encoding a value uses, so a ``©`` (or an accent in a name) there is a bet on the reader. Said
    once, beside the fields, rather than per photo: the write goes ahead either way, because plenty
    of software does read these as UTF-8 and a photographer called José should not have to spell
    their own name differently.
    """
    if fields is None or fields.is_empty:
        return ""
    if not (tags := non_ascii_tags(fields.preview(year))):
        return ""
    return _(
        "EXIF only allows plain ASCII before Exif 3.0, so some software will show {tags} "
        "wrongly. XMP carries the same text either way.",
    ).format(tags=", ".join(tags))
