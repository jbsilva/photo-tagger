"""Tests for the notice template language: chains, literals, aliases, and date formats."""

from unittest.mock import patch

import pytest

from photo_tagger.errors import FieldError
from photo_tagger.templates import (
    YEAR_TAGS,
    MissingValueError,
    format_date,
    parse_template,
    sample_values,
)


def _render(text: str, values: dict[str, str] | None = None, artist: str = "") -> str:
    """Parse and fill *text* in one step."""
    return parse_template(text, flag="--copyright").render(values or {}, {"artist": artist})


def test_a_chain_takes_the_first_tag_with_a_value() -> None:
    """Blank values count as missing, so the next source answers."""
    values = {"EXIF:Artist": "  ", "XMP-dc:Creator": "Ada"}
    assert _render("{EXIF:Artist|XMP-dc:Creator}", values) == "Ada"


def test_a_literal_wins_when_the_chain_reaches_it() -> None:
    """A quoted fallback is written when no tag answers, and never before."""
    assert _render('{EXIF:Artist|"Unknown"}') == "Unknown"
    assert _render('{EXIF:Artist|"Unknown"}', {"EXIF:Artist": "Ada"}) == "Ada"


def test_a_literal_is_a_template_too() -> None:
    """A fallback can hold placeholders, and one that finds nothing passes to the next source."""
    assert _render('{EXIF:Artist|"(c) {artist}"}', artist="Ada") == "(c) Ada"
    assert _render('{EXIF:Artist|"{artist!}"|"nobody"}') == "nobody"


def test_a_chain_without_a_literal_raises_when_nothing_answers() -> None:
    """The caller skips the field rather than writing a notice with a hole in it."""
    with pytest.raises(MissingValueError) as caught:
        _render("(c) {EXIF:Artist}")
    assert caught.value.placeholder == "{EXIF:Artist}"


def test_the_year_alias_reads_the_capture_date() -> None:
    """{year} is the first capture date, as a four-digit year."""
    assert _render("{year}", {"XMP-photoshop:DateCreated": "2011"}) == "2011"


def test_the_year_alias_drops_out_and_the_bang_form_does_not() -> None:
    """{year} ends in "" and {year!} does not."""
    assert _render("(c) {year} Ada") == "(c) Ada"
    with pytest.raises(MissingValueError):
        _render("(c) {year!} Ada")


def test_an_alias_inside_a_chain_leaves_the_fallback_to_the_chain() -> None:
    """{artist|"Unknown"} reaches the literal, instead of stopping at the alias's own ""."""
    assert _render('{artist|"Unknown"}') == "Unknown"


def test_a_date_format_uses_strftime_codes() -> None:
    """% after a tag formats it as a date."""
    values = {"EXIF:DateTimeOriginal": "2019:05:04 10:00:00"}
    assert _render("{EXIF:DateTimeOriginal%d.%m.%Y}", values) == "04.05.2019"


def test_doubled_braces_are_literal() -> None:
    """{{ and }} write one brace each, also inside a literal."""
    assert _render('{{x}} {EXIF:Model|"{{y}}"}') == "{x} {y}"
    assert _render('{"{{"}') == "{"


def test_a_drop_at_the_start_leaves_no_leading_space() -> None:
    """The text around a dropped placeholder is trimmed."""
    assert _render("{year} Ada") == "Ada"


def test_tags_lists_every_tag_once_and_date_tags_only_the_formatted() -> None:
    """The read asks for each tag once, and the preview dates only the date tags."""
    template = parse_template('{year} {EXIF:Model|"{year}"}', flag="--copyright")
    assert template.tags() == (*YEAR_TAGS, "EXIF:Model")
    assert template.date_tags() == YEAR_TAGS
    assert sample_values(template, 2020)["EXIF:DateTimeOriginal"] == "2020:01:01 00:00:00"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("(c) {shot_at}", r"does not know \{shot_at\}"),
        ("{}", "does not know"),
        ("{year|}", "does not know"),
        ("{EXIF:Artist%}", "does not know"),
        ("{year", "never closed"),
        ('{"open}', "never closed"),
        ("a}", "no '{' before it"),
        ('{"a" "b"}', "does not know"),
    ],
)
def test_a_malformed_template_is_refused_with_the_option_named(text: str, message: str) -> None:
    """A typo stops the run before the first photo, and the message names the option."""
    with pytest.raises(FieldError, match=message) as caught:
        parse_template(text, flag="--rights")
    assert "--rights" in str(caught.value)


def test_an_unclosed_quote_inside_a_chain_is_refused() -> None:
    """A quote left open would swallow the rest of the chain."""
    with pytest.raises(FieldError, match="never closed"):
        parse_template('{EXIF:Artist|"x}', flag="--copyright")


@pytest.mark.parametrize(
    ("raw", "fmt", "expected"),
    [
        ("2019:05:04 10:20:30", "%Y-%m-%d %H:%M:%S", "2019-05-04 10:20:30"),
        ("2019-05-04T10:20:30.5+02:00", "%H", "10"),
        ("2019:05:04", "%Y", "2019"),
        ("2019", "%Y", "2019"),
        ("2019", "%m", None),
        ("2019:05", "%Y", "2019"),
        # IPTC writes an unknown month and day as 00.
        ("2019:00:00", "%Y", "2019"),
        ("2019:00:00", "%d", None),
        ("0000:00:00 00:00:00", "%Y", None),
        ("2019-05-04T10:00+02:00", "%H:%M:%S", "10:00:00"),
        ("2019:05:04 10:00:00-05:00", "%S", "00"),
        ("2019-05-04Z", "%d", "04"),
        ("2019:99999999999999999999:01", "%Y", None),
        ("20190504", "%Y", None),
        ("unknown", "%Y", None),
    ],
)
def test_format_date(raw: str, fmt: str, expected: str | None) -> None:
    """EXIF, XMP, and IPTC dates format; partial ones only answer year codes; bad ones none."""
    assert format_date(raw, fmt) == expected


def test_a_date_format_strftime_rejects_is_refused_when_parsed() -> None:
    """Windows raises on an unknown code, which must not stop a run halfway through."""

    class _StrictDatetime:
        def __init__(self, *_args: object) -> None:
            pass

        @staticmethod
        def strftime(fmt: str) -> str:
            if "%Q" in fmt:
                msg = "Invalid format string"
                raise ValueError(msg)
            return fmt

    with (
        patch("photo_tagger.templates.datetime", _StrictDatetime),
        pytest.raises(FieldError, match="strftime does not know: %Q"),
    ):
        parse_template("(c) {EXIF:DateTimeOriginal%Q}", flag="--copyright")
