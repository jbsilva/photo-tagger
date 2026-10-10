"""The Tagged and Status columns, the sort orders, and the badges on a thumbnail."""

from typing import TYPE_CHECKING

from photo_tagger.gui_state.files import (
    file_type_label,
)
from photo_tagger.gui_state.items import (
    FAILED,
    PENDING,
    READY,
    SAVED,
    WORKING,
    PhotoItem,
)
from photo_tagger.gui_state.review import (
    mark_mixed_script,
)
from photo_tagger.i18n import _, gettext_noop, pgettext
from photo_tagger.metadata import (
    FIELD_COPYRIGHT,
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    TAGGED_FIELDS,
)


if TYPE_CHECKING:
    from collections.abc import Iterable


# Order photos take when the tree is sorted by Status (ascending): lifecycle, failures last.
STATUS_SORT_ORDER = (PENDING, WORKING, READY, SAVED, FAILED)


def status_sort_rank(status: str) -> int:
    """
    Rank a photo status for sorting the tree's Status column.

    Lower ranks sort first when ascending, following the lifecycle order in
    :data:`STATUS_SORT_ORDER` (pending, working, ready, saved, failed). Sorting on the raw status
    label would order it alphabetically ("failed" before "ready"), which is not useful; this gives
    the column a meaningful order that the user can flip by clicking the header.
    """
    try:
        return STATUS_SORT_ORDER.index(status)
    except ValueError:
        return len(STATUS_SORT_ORDER)


# Short status word shown in the tree's Status column. A photo not generated yet shows nothing.
_STATUS_LABELS = {
    PENDING: "",
    WORKING: gettext_noop("working..."),
    READY: gettext_noop("ready"),
    SAVED: gettext_noop("saved ✓"),
    FAILED: gettext_noop("failed ✗"),
}


def status_label(item: PhotoItem) -> str:
    """
    Word the tree's Status cell for *item*: its lifecycle state, and the mixed-script mark.

    A proposal replayed from the cache says so, since no model call made it this time.
    """
    word = _STATUS_LABELS.get(item.status, "")
    label = _(word) if word else ""
    if item.status == READY and item.from_cache:
        label = _("ready (cached)")
    return mark_mixed_script(label, item.foreign_terms)


# The Tagged indicator's letter and display label per field, in display order. The letters go
# through the catalog under a "Tagged column letter" msgctxt (see _tagged_letter), so a language
# whose field names start with other letters can remap them; the tooltips below spell out whatever
# letters are active, so they stay clear either way.
_FIELD_LETTERS = (
    (FIELD_TITLE, "T", gettext_noop("title")),
    (FIELD_DESCRIPTION, "D", gettext_noop("description")),
    (FIELD_KEYWORDS, "K", gettext_noop("keywords")),
    (FIELD_COPYRIGHT, "C", gettext_noop("copyright")),
)


def _tagged_letter(letter: str) -> str:
    """
    Translate one Tagged column letter under the "Tagged column letter" msgctxt.

    pybabel only extracts pgettext calls whose arguments are string literals, so each letter needs
    its own literal call here instead of looping pgettext(context, letter) over _FIELD_LETTERS.
    """
    match letter:
        case "T":
            # The repeated context is the point (see the docstring), so S1192 does not apply.
            return pgettext("Tagged column letter", "T")  # NOSONAR S1192
        case "D":
            return pgettext("Tagged column letter", "D")
        case "K":
            return pgettext("Tagged column letter", "K")
        case "C":
            return pgettext("Tagged column letter", "C")
        case _:  # pragma: no cover - _FIELD_LETTERS only ever supplies T, D, K, C
            message = f"no Tagged column letter translation for {letter!r}"
            raise ValueError(message)


def tagged_summary(fields: set[str]) -> str:
    """
    Compress a file's present metadata fields into the Tagged column label.

    "TK" means a title and keywords exist, and a "C" adds a copyright notice; "-" means the scan ran
    and found none. tagged_legend (the header tooltip) and tagged_tooltip (the cell tooltip) spell
    the letters out.
    """
    letters = "".join(
        _tagged_letter(letter)
        for field_name, letter, _label in _FIELD_LETTERS
        if field_name in fields
    )
    return letters or "-"


def tagged_legend() -> str:
    """Spell out every Tagged letter ("T = title, D = description, ..."), localized."""
    return ", ".join(
        f"{_tagged_letter(letter)} = {_(label)}" for _field_name, letter, label in _FIELD_LETTERS
    )


def tagged_tooltip(fields: set[str]) -> str:
    """
    Explain a row's Tagged cell on hover, naming each present field next to its letter.

    Built from the same table as tagged_summary so the tooltip cannot drift from the letters, and
    from the display labels rather than the raw field constants so the names are translated.
    """
    present = ", ".join(
        f"{_tagged_letter(letter)} = {_(label)}"
        for field_name, letter, label in _FIELD_LETTERS
        if field_name in fields
    )
    return _("Already on the file: {fields}").format(fields=present or _("nothing"))


# How the folder thumbnail grid orders its photos. These mirror the tree's sortable columns so the
# two views agree on what "by type" or "by status" means, and SORT_NAME is the default.
SORT_NAME = "name"


SORT_TYPE = "type"


SORT_STATUS = "status"


SORT_TAGGED = "tagged"


# The grid's Sort by menu, in the order it lists them: each sort with its label, marked here and
# translated where it is shown.
SORT_CHOICES: tuple[tuple[str, str], ...] = (
    (gettext_noop("Name"), SORT_NAME),
    (gettext_noop("Type"), SORT_TYPE),
    (gettext_noop("Status"), SORT_STATUS),
    (gettext_noop("Tagged"), SORT_TAGGED),
)


def photo_sort_key(item: PhotoItem, criterion: str) -> tuple[str, str, str]:
    """
    Build the sort key for one photo under a grid sort *criterion*.

    Mirrors the tree's column sorts: Type sorts by the extension label, Status by lifecycle rank
    (not the raw word, which would put "failed" before "ready"), and Tagged by the compressed
    letters. Every key carries the filename then the full path as tiebreaks, so photos that tie on
    the primary field keep a stable, readable order (two files can share a name in a recursive
    folder). All branches return same-shape string tuples so any one criterion sorts a whole list.
    """
    name = item.path.name.casefold()
    full = str(item.path).casefold()
    if criterion == SORT_TYPE:
        primary = file_type_label(item.path)
    elif criterion == SORT_STATUS:
        # Zero-padded so the numeric rank sorts as a string alongside the other criteria's keys.
        primary = f"{status_sort_rank(item.status):03d}"
    elif criterion == SORT_TAGGED:
        primary = tagged_summary(item.known_fields or set()).casefold()
    else:
        primary = name
    return (primary, name, full)


def sort_photos(items: Iterable[PhotoItem], criterion: str, *, descending: bool) -> list[PhotoItem]:
    """
    Order *items* by *criterion*, reversed end to end when *descending*.

    Descending flips the tiebreaks too (names run Z to A within a group), matching how the tree
    reverses its whole comparison for a descending header click.
    """
    return sorted(items, key=lambda item: photo_sort_key(item, criterion), reverse=descending)


# Badge names for the folder grid's thumbnail overlays, in the order they are drawn.
BADGE_FAILED = "failed"


BADGE_SAVED = "saved"


BADGE_UNSAVED = "unsaved"


BADGE_METADATA = "metadata"


BADGE_COPYRIGHT = "copyright"


BADGE_SIDECAR = "sidecar"


BADGE_MIXED_SCRIPT = "mixed_script"


def thumb_badges(item: PhotoItem, *, has_sidecar: bool) -> list[str]:
    """
    Decide which overlay badges a photo's grid thumbnail shows.

    At most one lifecycle badge (failed beats saved beats unsaved-proposal), plus the already-has-
    metadata, copyright, sidecar, and mixed-script markers. The GUI maps each name to a color and
    glyph.

    Mixed script is informational, not a lifecycle state: the proposal is perfectly usable, it just
    wants reading first, so it marks a thumbnail that already says "generated" rather than replacing
    that.
    """
    badges: list[str] = []
    if item.status == FAILED:
        badges.append(BADGE_FAILED)
    elif item.status == SAVED:
        badges.append(BADGE_SAVED)
    elif item.has_proposal:
        badges.append(BADGE_UNSAVED)
    found = item.known_fields or set()
    if found & TAGGED_FIELDS:
        badges.append(BADGE_METADATA)
    if FIELD_COPYRIGHT in found:
        badges.append(BADGE_COPYRIGHT)
    if has_sidecar:
        badges.append(BADGE_SIDECAR)
    if item.foreign_terms:
        badges.append(BADGE_MIXED_SCRIPT)
    return badges


# What each badge means, for the thumbnail's tooltip: the glyphs alone are too small to explain.
_BADGE_NOTES = {
    BADGE_FAILED: gettext_noop("generation failed"),
    BADGE_SAVED: gettext_noop("saved"),
    BADGE_UNSAVED: gettext_noop("generated, not saved yet"),
    BADGE_METADATA: gettext_noop("already has metadata"),
    BADGE_COPYRIGHT: gettext_noop("has a copyright notice"),
    BADGE_SIDECAR: gettext_noop("has an XMP sidecar"),
    BADGE_MIXED_SCRIPT: gettext_noop("mixes writing systems"),
}


def thumb_tooltip(item: PhotoItem, badges: Iterable[str]) -> str:
    """
    Build a thumbnail's tooltip: the file name, then one line per badge from :func:`thumb_badges`.

    A failed photo adds its error last, so the reason can be read without opening the photo.
    """
    lines = [item.path.name, *(_(_BADGE_NOTES[name]) for name in badges)]
    if item.status == FAILED and item.error:
        lines.append(item.error)
    return "\n".join(lines)
