"""
The template language of a copyright notice: ``{year}``, ``{artist}``, and fallback chains.

A placeholder is a chain of sources, tried left to right::

    {EXIF:Artist|XMP-dc:Creator|"Unknown"}

A tag name reads that tag from the photo, and a blank value counts as missing. A quoted literal
always wins when the chain reaches it. ``%`` after a tag formats a date with strftime codes
(``{EXIF:DateTimeOriginal%Y}``). A chain that ends in ``""`` drops the placeholder when nothing
answers. A chain with no literal leaves the whole field unwritten on that photo, because a notice
with a hole in it is worse than none. ``{{`` and ``}}`` are literal braces.

``{year}`` and ``{artist}`` are aliases for the chains most notices want. With a ``!`` they lose
their final ``""``, so a photo with no capture date (or no name) gets no notice rather than one with
a hole in it.

Pure and IO-free: the tag values arrive as a mapping, read by :mod:`photo_tagger.metadata`.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from photo_tagger.errors import FieldError


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


# The capture date, best first. No file timestamp: a copy or an edit moves it, and a notice dated by
# the day a file was copied is wrong in a way nobody notices. File:FileModifyDate can be named.
YEAR_TAGS: tuple[str, ...] = (
    "EXIF:DateTimeOriginal",
    "XMP-exif:DateTimeOriginal",
    "XMP-photoshop:DateCreated",
    "IPTC:DateCreated",
    "EXIF:CreateDate",
)
# The name already on a photo, after the --artist setting.
ARTIST_TAGS: tuple[str, ...] = ("EXIF:Artist", "XMP-dc:Creator", "IPTC:By-line")
SETTING_ARTIST = "artist"

# A tag is always named with its group, so a typo such as {shot_at} is an error, not a silent blank.
_TAG = re.compile(r"^(?:[A-Za-z][A-Za-z0-9-]*:)+[A-Za-z][A-Za-z0-9_-]*$")
# EXIF, XMP, and IPTC dates start with the year: "2019:05:04 10:00:00", "2019-05-04T10:00", "2019".
_DATE_START = re.compile(r"\d{4}(?!\d)")
_NUMBER = re.compile(r"\d+")
# Where a time zone starts: a "Z" or a "+", or a "-" past the date (whose own dashes come first).
_OFFSET = re.compile(r"[Z+]|(?<=.{10})-")
# Year, month, day, hour, minute, second: what follows (a fraction, an offset) is not needed.
_DATE_PARTS = 6
# A year, a month, and a day: what any strftime code beyond the year needs.
_FULL_DATE_PARTS = 3
# Any strftime code other than a year's needs the month and the day too.
_BEYOND_YEAR = re.compile(r"%[^YyCG%]")


@dataclass(frozen=True, slots=True)
class TagSource:
    """A tag read from the photo, optionally formatted as a date."""

    tag: str
    date_format: str | None = None


@dataclass(frozen=True, slots=True)
class SettingSource:
    """A value from the configuration, such as the ``--artist`` name."""

    name: str


@dataclass(frozen=True, slots=True)
class LiteralSource:
    """Quoted text, itself a template, so ``"(c) {artist}"`` works as a fallback."""

    template: Template


type Source = TagSource | SettingSource | LiteralSource


@dataclass(frozen=True, slots=True)
class Chain:
    """One placeholder: its sources in order, and its text for the warning when none answers."""

    sources: tuple[Source, ...]
    text: str


class MissingValueError(Exception):
    """A chain with no final literal found nothing on this photo."""

    def __init__(self, placeholder: str) -> None:
        """Keep the placeholder text, so the warning can name it."""
        super().__init__(placeholder)
        self.placeholder = placeholder


@dataclass(frozen=True, slots=True)
class Template:
    """A parsed notice: plain text and chains, in order."""

    parts: tuple[str | Chain, ...]

    def tags(self) -> tuple[str, ...]:
        """Return every tag the template may read, in order, without repeats."""
        return tuple(dict.fromkeys(_walk_tags(self)))

    def date_tags(self) -> tuple[str, ...]:
        """Return the tags the template formats as dates, for a preview with a sample date."""
        return tuple(dict.fromkeys(_walk_tags(self, dates_only=True)))

    def render(self, values: Mapping[str, str], settings: Mapping[str, str]) -> str:
        """
        Fill the template from *values* (tag to text) and *settings* (name to text).

        Raises :class:`MissingValueError` when a chain without a final literal finds nothing. When a
        chain drops out (``""``), the spaces before it go too, so ``(c) {year} {artist}, All rights
        reserved`` without a name reads ``(c) 2019, All rights reserved``.
        """
        out = ""
        dropped = False
        for part in self.parts:
            if isinstance(part, str):
                out += part
                continue
            value = _render_chain(part, values, settings)
            if value:
                out += value
            else:
                out = out.rstrip(" ")
                dropped = True
        return out.strip() if dropped else out


def _walk_tags(template: Template, *, dates_only: bool = False) -> Iterator[str]:
    """Yield the tags of *template* and of the literals inside it."""
    for part in template.parts:
        if isinstance(part, str):
            continue
        for source in part.sources:
            if isinstance(source, LiteralSource):
                yield from _walk_tags(source.template, dates_only=dates_only)
            elif isinstance(source, TagSource) and (source.date_format or not dates_only):
                yield source.tag


def _render_chain(chain: Chain, values: Mapping[str, str], settings: Mapping[str, str]) -> str:
    """Return the first source that answers, or raise :class:`MissingValueError`."""
    for source in chain.sources:
        if (value := _source_value(source, values, settings)) is not None:
            return value
    raise MissingValueError(chain.text)


def _source_value(
    source: Source,
    values: Mapping[str, str],
    settings: Mapping[str, str],
) -> str | None:
    """
    Return what one source says, or None when it says nothing usable.

    A literal answers even when it renders empty: ``""`` is how a chain drops out.
    """
    match source:
        case LiteralSource(template=inner):
            try:
                return inner.render(values, settings)
            except MissingValueError:
                return None
        case TagSource(tag=tag, date_format=fmt):
            value = values.get(tag, "").strip()
            return (format_date(value, fmt) if value and fmt else value) or None
        case SettingSource(name=name):
            return settings.get(name, "").strip() or None


def format_date(raw: str, fmt: str) -> str | None:
    """
    Format an EXIF, XMP, or IPTC date with strftime codes, or return None when it is unusable.

    A date with only a year (XMP allows that, and IPTC writes an unknown month and day as ``00``)
    answers ``%Y`` but not ``%m``. A camera with an unset clock writes ``0000:00:00 00:00:00``,
    which is no date at all.

    Examples:
        >>> format_date("2019:05:04 10:00:00", "%Y")
        '2019'
        >>> format_date("2019", "%d.%m.%Y") is None
        True
    """
    text = raw.strip()
    if not _DATE_START.match(text):
        return None
    # Cut an offset ("+02:00", "-05:00", "Z") after the date, or its digits would read as seconds.
    text = _OFFSET.split(text, maxsplit=1)[0]
    parts = [int(number) for number in _NUMBER.findall(text)[:_DATE_PARTS]]
    if len(parts) < _FULL_DATE_PARTS or 0 in parts[1:_FULL_DATE_PARTS]:
        if _BEYOND_YEAR.search(fmt):
            return None
        parts = [parts[0], 1, 1]
    year, month, day, *clock = parts
    hour, minute, second = [*clock, 0, 0, 0][:3]
    try:
        when = datetime(year, month, day, hour, minute, second)  # noqa: DTZ001 - camera local time.
    except ValueError, OverflowError:
        return None
    return when.strftime(fmt)


def _check_date_format(fmt: str, flag: str, body: str) -> None:
    """
    Refuse a strftime format the platform rejects, before any photo is read.

    Windows raises on an unknown code such as ``%Q``, which would otherwise stop a run halfway.
    """
    try:
        datetime(2000, 1, 2, 3, 4, 5).strftime(fmt)  # noqa: DTZ001 - only the codes matter.
    except ValueError:
        msg = f"{flag} has a date format in {{{body}}} that strftime does not know: {fmt}"
        raise FieldError(msg) from None


def _alias(name: str, *, in_chain: bool) -> tuple[Source, ...] | None:
    """
    Expand ``year`` and ``artist`` (with or without ``!``), or return None for any other name.

    Inside a longer chain the alias never ends in ``""``: the rest of the chain is the fallback.
    """
    required = name.endswith("!")
    match name.removesuffix("!"):
        case "year":
            sources: tuple[Source, ...] = tuple(TagSource(tag, "%Y") for tag in YEAR_TAGS)
        case "artist":
            sources = (SettingSource(SETTING_ARTIST), *(TagSource(tag) for tag in ARTIST_TAGS))
        case _:
            return None
    if required or in_chain:
        return sources
    return (*sources, LiteralSource(Template(())))


def _split_chain(body: str, flag: str) -> list[str]:
    """Split a placeholder's body on ``|``, leaving what is inside quotes whole."""
    pieces: list[str] = []
    current = ""
    quoted = False
    for char in body:
        if char == '"':
            quoted = not quoted
        if char == "|" and not quoted:
            pieces.append(current)
            current = ""
        else:
            current += char
    if quoted:
        msg = f"{flag}: the quote in {{{body}}} is never closed"
        raise FieldError(msg)
    pieces.append(current)
    return pieces


def _parse_source(piece: str, body: str, flag: str, *, in_chain: bool) -> tuple[Source, ...]:
    """Turn one piece of a chain into its sources: a literal, an alias, or a tag."""
    text = piece.strip()
    inner = text[1:-1]
    if len(text) > 1 and text[0] == text[-1] == '"' and '"' not in inner:
        return (LiteralSource(parse_template(inner, flag=flag)),)
    if (expanded := _alias(text, in_chain=in_chain)) is not None:
        return expanded
    tag, percent, fmt = text.partition("%")
    if _TAG.match(tag) and (not percent or fmt):
        if percent:
            _check_date_format(f"%{fmt}", flag, body)
        return (TagSource(tag, f"%{fmt}" if percent else None),)
    msg = (
        f"{flag} does not know {{{body}}}. Use {{year}}, {{artist}}, a tag with its group such as "
        f'{{EXIF:Artist}}, or a chain such as {{EXIF:Artist|XMP-dc:Creator|"Unknown"}}'
    )
    raise FieldError(msg)


def _parse_chain(body: str, flag: str) -> Chain:
    """Parse the text between a placeholder's braces."""
    pieces = _split_chain(body, flag)
    in_chain = len(pieces) > 1
    sources = [
        source for piece in pieces for source in _parse_source(piece, body, flag, in_chain=in_chain)
    ]
    return Chain(tuple(sources), f"{{{body}}}")


def _placeholder_end(text: str, start: int, flag: str) -> int:
    """
    Return the index of the ``}`` that closes the placeholder opened at *start*.

    A literal holds no quote, so a quote always ends it, and any brace inside it is its own.
    """
    quoted = False
    for index in range(start, len(text)):
        char = text[index]
        if char == '"':
            quoted = not quoted
        elif char == "}" and not quoted:
            return index
    msg = f"{flag}: a placeholder in {text!r} is never closed"
    raise FieldError(msg)


def parse_template(text: str, *, flag: str) -> Template:
    """
    Parse a notice template, raising :class:`FieldError` when it is malformed.

    *flag* names the option in the error, so the message says which field to fix.

    Examples:
        >>> parse_template("(c) {year}", flag="--copyright").tags()[0]
        'EXIF:DateTimeOriginal'
    """
    parts: list[str | Chain] = []
    literal = ""
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair in {"{{", "}}"}:
            literal += pair[0]
            index += 2
        elif text[index] == "{":
            end = _placeholder_end(text, index + 1, flag)
            if literal:
                parts.append(literal)
                literal = ""
            parts.append(_parse_chain(text[index + 1 : end], flag))
            index = end + 1
        elif text[index] == "}":
            msg = f"{flag}: {text!r} has a '}}' with no '{{' before it (write '}}}}' for a brace)"
            raise FieldError(msg)
        else:
            literal += text[index]
            index += 1
    if literal:
        parts.append(literal)
    return Template(tuple(parts))


def sample_values(template: Template, year: int) -> dict[str, str]:
    """
    Give every tag of *template* a stand-in, for a preview with no photo at hand.

    A date tag gets a date in *year*, and any other tag its own name in angle brackets, so the
    preview shows where each value goes rather than a fallback no photo may need.
    """
    samples = {tag: f"<{tag}>" for tag in template.tags()}
    return samples | dict.fromkeys(template.date_tags(), f"{year}:01:01 00:00:00")
