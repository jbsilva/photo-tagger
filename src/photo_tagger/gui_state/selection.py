"""The Select menu, the search box, and the status filters of the list and the grid."""

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from photo_tagger.gui_state.files import (
    _extension,
)
from photo_tagger.gui_state.items import (
    FAILED,
    PENDING,
    SAVED,
    PhotoItem,
)
from photo_tagger.i18n import gettext_noop
from photo_tagger.metadata import (
    FIELD_COPYRIGHT,
    TAGGED_FIELDS,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaggedCriterion:
    """
    One "already tagged" rule: which fields count as done, and how to say so.

    *required* is the set of fields to look for, *match_all* picks AND over OR (see
    :func:`paths_matching_fields`), and *phrase* names the rule in the status bar ("a title and a
    description"). Kept as one value so the menu preset, the action waiting on a metadata scan, and
    the deselection that finally runs all pass the same thing around.
    """

    required: frozenset[str]
    match_all: bool
    phrase: str

    def matches(self, presence: dict[Path, set[str]]) -> set[Path]:
        """Pick the paths in *presence* that satisfy this rule."""
        return paths_matching_fields(presence, set(self.required), match_all=self.match_all)


def paths_matching_fields(
    presence: dict[Path, set[str]],
    required: set[str],
    *,
    match_all: bool,
) -> set[Path]:
    """
    Pick the paths whose present fields satisfy *required*.

    With *match_all* (AND), a path matches only when it has every required field, so ``{title,
    description}`` selects photos that already carry both. Without it (OR), having any one required
    field is enough, which is how the broad "any metadata" criterion works. An empty *required* set
    matches nothing, so a caller cannot accidentally select every photo by passing no fields.
    """
    if not required:
        return set()
    matched: set[Path] = set()
    for path, present in presence.items():
        overlap = present & required
        if (overlap == required) if match_all else bool(overlap):
            matched.add(path)
    return matched


def deselect_paths(items: dict[str, PhotoItem], paths: Iterable[Path]) -> int:
    """
    Uncheck the items whose path is in *paths*; return how many actually changed.

    This is how the GUI "skips" photos: rather than dropping them from the list like the CLI does,
    it just deselects them, so the user still sees what was skipped and can re-check any of it.
    Items already unchecked (or not in the list) are left alone and not counted, so the returned
    tally is the number of newly-skipped photos.
    """
    changed = 0
    for path in paths:
        item = items.get(str(path))
        if item is not None and item.selected:
            item.selected = False
            changed += 1
    return changed


def keys_by_check(items: Iterable[PhotoItem], *, checked: bool) -> list[str]:
    """Return the list keys of the photos whose checkbox matches *checked*, for a bulk remove."""
    return [str(item.path) for item in items if item.selected is checked]


# What a bulk selection action does to the photos it matched. SELECT_ONLY also unchecks everything
# it did not match, which is the "show me just these" move the other two cannot express.
SELECT_CHECK = "check"


SELECT_UNCHECK = "uncheck"


SELECT_ONLY = "only"


@dataclass(frozen=True, slots=True)
class SelectionChange:
    """How many photos a bulk selection action matched, and how many checkboxes it moved."""

    matched: int
    changed: int


def apply_selection(
    items: Iterable[PhotoItem],
    matches: Callable[[PhotoItem], bool],
    verb: str,
) -> SelectionChange:
    """
    Check or uncheck the photos *matches* picks out, and report what happened.

    The two counts answer different questions the status bar needs to tell apart: nothing matched
    (the criterion found no photos) reads differently from nothing changed (they were all already
    checked). Photos the criterion missed are left alone, except under :data:`SELECT_ONLY`, which
    unchecks them.
    """
    matched = 0
    changed = 0
    for item in items:
        hit = matches(item)
        matched += int(hit)
        if hit:
            wanted = verb != SELECT_UNCHECK
        elif verb == SELECT_ONLY:
            wanted = False
        else:
            continue
        if item.selected != wanted:
            item.selected = wanted
            changed += 1
    return SelectionChange(matched=matched, changed=changed)


def extension_counts(items: Iterable[PhotoItem]) -> list[tuple[str, int]]:
    """
    Tally the photos per file extension, the most common type first.

    Feeds the Select menu's by-file-type entries, which is why the count rides along: a mixed
    folder's menu reads "dng (120)" and "jpg (120)" rather than making the user guess.
    """
    counts: dict[str, int] = {}
    for item in items:
        extension = _extension(item.path)
        counts[extension] = counts.get(extension, 0) + 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))


def matches_extension(item: PhotoItem, extension: str) -> bool:
    """Whether *item* is of type *extension* (given without the dot, case-insensitive)."""
    return _extension(item.path) == extension.lstrip(".").lower()


# The characters that make a name pattern a glob rather than a plain piece of text.
_GLOB_CHARS = "*?["


def _forward(text: str) -> str:
    """Rewrite path separators as forward slashes, so both spellings compare equal."""
    return text.replace("\\", "/")


def matches_name_pattern(item: PhotoItem, pattern: str) -> bool:
    """
    Whether *item*'s filename matches the glob *pattern*, ignoring case.

    A pattern with no wildcard in it is read as "contains", so typing ``IMG`` finds ``IMG_0001.dng``
    rather than nothing at all. One that holds a path separator is matched against the whole path
    instead of the bare filename, so a pattern can name a subfolder.
    """
    if not pattern:
        return False
    text = str(item.path) if ("/" in pattern or "\\" in pattern) else item.path.name
    glob = pattern if any(char in pattern for char in _GLOB_CHARS) else f"*{pattern}*"
    # Compare on forward slashes: a Windows path stringifies with backslashes, but the box invites
    # either separator, and fnmatch would otherwise never match the one the user did not type.
    return fnmatch.fnmatchcase(_forward(text).casefold(), _forward(glob).casefold())


# Folder grid filters: which photos the grid shows. FILTER_ALL (the default) shows everything.
FILTER_ALL = "all"


FILTER_SELECTED = "selected"


FILTER_PENDING = "pending"


FILTER_GENERATED = "generated"


FILTER_SAVED = "saved"


FILTER_FAILED = "failed"


FILTER_UNTAGGED = "untagged"


FILTER_NO_COPYRIGHT = "no_copyright"


FILTER_MIXED_SCRIPT = "mixed_script"


# The grid's Show menu, in the order it lists them: each filter with its label, marked here and
# translated where it is shown.
FILTER_CHOICES: tuple[tuple[str, str], ...] = (
    (gettext_noop("All"), FILTER_ALL),
    (gettext_noop("Selected"), FILTER_SELECTED),
    (gettext_noop("Not generated"), FILTER_PENDING),
    (gettext_noop("Generated"), FILTER_GENERATED),
    (gettext_noop("Saved"), FILTER_SAVED),
    (gettext_noop("Failed"), FILTER_FAILED),
    (gettext_noop("Untagged"), FILTER_UNTAGGED),
    (gettext_noop("No copyright"), FILTER_NO_COPYRIGHT),
    (gettext_noop("Mixed writing systems"), FILTER_MIXED_SCRIPT),
)


# The filters that read the background metadata scan, so a view showing one must re-filter when
# the scan reports.
SCAN_FILTERS = frozenset({FILTER_UNTAGGED, FILTER_NO_COPYRIGHT})


def photo_matches_filter(item: PhotoItem, criterion: str) -> bool:
    """
    Report whether *item* passes the grid filter *criterion*.

    The status filters key off the lifecycle state the badges show, and FILTER_SELECTED follows the
    checkbox. FILTER_UNTAGGED and FILTER_NO_COPYRIGHT match only once the metadata scan has run and
    found nothing (a None/unknown scan state does not match, so no photo is hidden while the scan is
    still pending). FILTER_MIXED_SCRIPT is what turns the run's "3 photos mix writing systems" line
    into three rows you can actually open. FILTER_ALL and any unrecognized value match everything.
    """
    scanned = item.known_fields is not None
    found = item.known_fields or set()
    # A dict of cheap boolean checks keeps this to one return (no PLR0911). FILTER_ALL is not a
    # key, so it falls through to the default True, as does any value the grid never sends.
    checks = {
        FILTER_SELECTED: item.selected,
        FILTER_PENDING: item.status == PENDING,
        FILTER_GENERATED: item.has_proposal,
        FILTER_SAVED: item.status == SAVED,
        FILTER_FAILED: item.status == FAILED,
        FILTER_UNTAGGED: scanned and not (found & TAGGED_FIELDS),
        FILTER_NO_COPYRIGHT: scanned and FIELD_COPYRIGHT not in found,
        FILTER_MIXED_SCRIPT: bool(item.foreign_terms),
    }
    return checks.get(criterion, True)


def filter_photos(items: Iterable[PhotoItem], criterion: str) -> list[PhotoItem]:
    """Keep only the photos that pass the grid filter *criterion*, order preserved."""
    return [item for item in items if photo_matches_filter(item, criterion)]
