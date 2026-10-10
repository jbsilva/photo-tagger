"""Tests for the copyright/authorship schema and its ``--set`` escape hatch."""

import pytest

from photo_tagger.errors import FieldError
from photo_tagger.fields import (
    SHOWN_FIELDS,
    SHOWN_TAGS,
    TAG_EXIF_ARTIST,
    TAG_EXIF_COPYRIGHT,
    TAG_IPTC_BY_LINE,
    TAG_IPTC_COPYRIGHT_NOTICE,
    TAG_XMP_CREATOR,
    TAG_XMP_MARKED,
    TAG_XMP_RIGHTS,
    ExtraFields,
    build_extra_fields,
    non_ascii_tags,
    parse_assignment,
)


def test_empty_fields_resolve_to_nothing() -> None:
    """A run that asked for no field writes no tag, and says so up front."""
    fields = ExtraFields()
    assert fields.is_empty
    assert fields.resolve(2025) == {}


def test_artist_is_written_to_every_namespace() -> None:
    """A credit lands in EXIF, XMP, and IPTC, so no application shows the photo as uncredited."""
    payload = ExtraFields(artist="Ada Lovelace").resolve(2025)
    assert payload == {
        TAG_EXIF_ARTIST: "Ada Lovelace",
        TAG_XMP_CREATOR: "Ada Lovelace",
        TAG_IPTC_BY_LINE: "Ada Lovelace",
    }


def test_notice_fills_the_year_and_the_artist() -> None:
    """Both placeholders are substituted, and the notice reaches all three copyright tags."""
    fields = ExtraFields(artist="Ada Lovelace", notice="Copyright (c) {year} {artist}")
    payload = fields.resolve(2019)
    expected = "Copyright (c) 2019 Ada Lovelace"
    assert payload[TAG_EXIF_COPYRIGHT] == expected
    assert payload[TAG_IPTC_COPYRIGHT_NOTICE] == expected
    assert payload[TAG_XMP_RIGHTS] == expected


def test_rights_overrides_only_the_xmp_spelling() -> None:
    """--rights is the UTF-8 wording; EXIF and IPTC keep the plain one."""
    fields = ExtraFields(
        artist="Ada Lovelace",
        notice="Copyright (c) {year} {artist}",
        rights="© {year} {artist}",
    )
    payload = fields.resolve(2025)
    assert payload[TAG_XMP_RIGHTS] == "© 2025 Ada Lovelace"
    assert payload[TAG_EXIF_COPYRIGHT] == "Copyright (c) 2025 Ada Lovelace"


def test_rights_alone_writes_only_xmp() -> None:
    """Asking for XMP rights and nothing else leaves the EXIF and IPTC notices alone."""
    payload = ExtraFields(rights="© {year} Ada").resolve(2025)
    assert payload == {TAG_XMP_RIGHTS: "© 2025 Ada"}


@pytest.mark.parametrize(("marked", "expected"), [(True, "True"), (False, "False")])
def test_marked_writes_the_literal_xmp_boolean(*, marked: bool, expected: str) -> None:
    """XMP spells this flag out, so the payload carries the word rather than a Python bool."""
    assert ExtraFields(marked=marked).resolve(2025) == {TAG_XMP_MARKED: expected}


def test_marked_unset_leaves_the_tag_alone() -> None:
    """None is the third state: whatever the photo already says stays."""
    assert TAG_XMP_MARKED not in ExtraFields(artist="Ada").resolve(2025)


def test_raw_tags_are_applied_last() -> None:
    """--set names a tag by hand, so it overrides what the copyright fields put there."""
    fields = ExtraFields(artist="Ada", tags={TAG_XMP_CREATOR: "Someone Else"})
    assert fields.resolve(2025)[TAG_XMP_CREATOR] == "Someone Else"
    assert fields.resolve(2025)[TAG_EXIF_ARTIST] == "Ada"


def test_blank_notice_writes_no_copyright() -> None:
    """An empty template is "nothing asked for", not an empty copyright tag."""
    assert ExtraFields(artist="Ada", notice="").resolve(2025) == {
        TAG_EXIF_ARTIST: "Ada",
        TAG_XMP_CREATOR: "Ada",
        TAG_IPTC_BY_LINE: "Ada",
    }


def test_parse_assignment_splits_on_the_first_equals() -> None:
    """A value containing '=' survives: only the first separator counts."""
    assert parse_assignment("XMP-xmpRights:WebStatement=https://x.test/?a=b") == (
        "XMP-xmpRights:WebStatement",
        "https://x.test/?a=b",
    )


def test_parse_assignment_accepts_a_pasted_exiftool_flag() -> None:
    """A line copied out of an ExifTool command works as it stands."""
    assert parse_assignment("-Artist=Ada Lovelace") == ("Artist", "Ada Lovelace")


def test_parse_assignment_keeps_an_empty_value() -> None:
    """An empty value is how ExifTool deletes a tag, so it is not treated as a mistake."""
    assert parse_assignment("XMP-dc:Rights=") == ("XMP-dc:Rights", "")


@pytest.mark.parametrize("item", ["nonsense", "=value", "  =x"])
def test_parse_assignment_rejects_what_is_not_an_assignment(item: str) -> None:
    """A missing separator or a missing tag stops the run instead of writing something odd."""
    with pytest.raises(FieldError, match="TAG=VALUE"):
        parse_assignment(item)


def test_build_rejects_an_unknown_placeholder() -> None:
    """A typo in the template fails before the first photo, not after five hundred."""
    with pytest.raises(FieldError, match=r"\{shot_at\}"):
        build_extra_fields(notice="Copyright (c) {shot_at}")


def test_build_rejects_artist_placeholder_without_an_artist() -> None:
    """{artist} with nobody to name would stamp a notice with a hole in it."""
    with pytest.raises(FieldError, match="--artist"):
        build_extra_fields(notice="Copyright (c) {year} {artist}")


def test_build_accepts_artist_placeholder_with_an_artist() -> None:
    """The same template is fine once --artist says who that is."""
    fields = build_extra_fields(artist="Ada", notice="(c) {year} {artist}")
    assert fields.resolve(2025)[TAG_EXIF_COPYRIGHT] == "(c) 2025 Ada"


def test_build_treats_a_blank_artist_as_none() -> None:
    """Whitespace is not a credit, so it never reaches the file."""
    assert build_extra_fields(artist="   ").is_empty


def test_build_parses_every_assignment() -> None:
    """Repeated --set options accumulate into one tag map."""
    fields = build_extra_fields(assignments=["A=1", "B=2"])
    assert fields.resolve(2025) == {"A": "1", "B": "2"}


def test_shown_tags_covers_every_field_once() -> None:
    """The one read behind ``show`` asks for each tag a shown field can come from, exactly once."""
    assert set(SHOWN_TAGS) == {tag for spec in SHOWN_FIELDS for tag in spec.tags}
    assert len(SHOWN_TAGS) == len(set(SHOWN_TAGS))


def test_shown_fields_report_the_rights_tags() -> None:
    """The point of the command: a copyright a photo carries has a line of its own."""
    labels = {spec.label for spec in SHOWN_FIELDS}
    assert {"Creator", "Copyright", "Rights marked"} <= labels


def test_non_ascii_tags_names_the_exif_fields_that_carry_more_than_ascii() -> None:
    """EXIF types these two as ASCII before Exif 3.0, so a © in them is a bet on the reader."""
    payload = ExtraFields(artist="José", notice="Copyright © {year}").resolve(2025)
    assert non_ascii_tags(payload) == [TAG_EXIF_ARTIST, TAG_EXIF_COPYRIGHT]


def test_non_ascii_tags_is_empty_for_a_plain_notice() -> None:
    """A plain notice raises nothing, and XMP is never flagged: it is UTF-8 by definition."""
    payload = ExtraFields(artist="Ada", notice="(c) {year}", rights="© {year}").resolve(2025)
    assert non_ascii_tags(payload) == []


@pytest.mark.parametrize(
    "item",
    [
        "all=",
        "EXIF:all=",
        "*=x",
        "XMP:*=x",
        "FileName=renamed.jpg",
        "Directory=/tmp",
        "File:FileModifyDate=2020:01:01",
        "System:FileName=x.jpg",
        "HardLink=x",
        "Keywords+=extra",
        "XAttrQuarantine=x",
        "tagsFromFile=/tmp/other.jpg",
        "json=/tmp/tags.json",
    ],
)
def test_parse_assignment_refuses_what_is_not_one_metadata_tag(item: str) -> None:
    """A --set must never wipe a group, rename or move the file, or edit a list by operator."""
    with pytest.raises(FieldError):
        parse_assignment(item)


@pytest.mark.parametrize(
    "item",
    ["Artist=Ada", "XMP-dc:Rights=R", "EXIF:IFD0:Copyright=(c) Ada", "XMP-xmpRights:Marked=True"],
)
def test_parse_assignment_accepts_plain_tags(item: str) -> None:
    """Ordinary tags, with or without their groups, still go through."""
    tag, _value = parse_assignment(item)
    assert tag == item.partition("=")[0]
