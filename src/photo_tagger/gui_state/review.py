"""The detail pane's notes: where a caption came from, and words in a second writing system."""

from typing import TYPE_CHECKING

from photo_tagger.gui_state.text import (
    MIXED_SCRIPT_MARK,
)
from photo_tagger.i18n import _, gettext_noop
from photo_tagger.metadata import (
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    CaptionValue,
)


if TYPE_CHECKING:
    from collections.abc import Iterable


# metadata.py keeps the sources as stable identifiers; only the window turns them into words.
_SOURCE_LABELS = {
    SOURCE_IMAGE: gettext_noop("image file"),
    SOURCE_SIDECAR: gettext_noop("XMP sidecar"),
}


def source_label(source: str) -> str:
    """Translate one metadata source name, leaving an unknown one as it came."""
    return _(_SOURCE_LABELS.get(source, source))


def caption_source_note(values: list[CaptionValue]) -> str:
    """
    Describe where one existing title or description came from, and what it shadows.

    The dimmed line under the field: which file the shown value came from, plus a line for any other
    file holding a different value. That second case is the point: a caption inside the photo is
    still there and still what ``exiftool photo.dng`` prints, but the sidecar wins on read, so it
    was invisible. Empty for a field no file carries.
    """
    if not values:
        return ""
    shown, *rest = values
    lines = [_("from {source}").format(source=source_label(shown.source))]
    lines += [
        _('shadows "{value}" in the {source}').format(
            value=other.value,
            source=source_label(other.source),
        )
        for other in rest
        if other.value != shown.value
    ]
    return "\n".join(lines)


def mark_mixed_script(label: str, terms: Iterable[str]) -> str:
    """
    Append the mixed-script mark to a Status cell *label*, or return it unchanged.

    The mark is what lets you find a flagged photo by scrolling, without first knowing the filter
    exists. A photo flagged before it was ever generated has no label to append to, so the mark
    stands alone.
    """
    if not list(terms):
        return label
    return f"{label} {MIXED_SCRIPT_MARK}" if label else MIXED_SCRIPT_MARK


def foreign_terms_note(terms: Iterable[str]) -> str:
    """
    Build the detail pane's line about mixed-script text, empty when the text is not mixed.

    Worded as a prompt to look, not as an error: a foreign term is right when the photo shows that
    text, and only the person looking at the photo can tell.
    """
    listed = list(terms)
    if not listed:
        return ""
    return _("Mixed script, worth a look: {terms}").format(terms=", ".join(listed))
