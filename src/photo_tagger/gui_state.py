"""
Qt-free helpers and per-photo state for the desktop GUI.

Kept separate from :mod:`photo_tagger.gui` so the GUI's logic (expanding dropped paths, grouping
files for the tree, parsing the editable keyword field, building the keyword set to write) is plain
Python the test suite covers normally, with no display server and no dependency on the optional
PySide6 extra. ``gui.py`` is then just the widget and event-loop shell that wires these helpers to
Qt.
"""

import fnmatch
import os
import shutil
import subprocess  # nosec B404 - only used to read the user's own login-shell PATH (see below)
import textwrap

# Imported at runtime, not under TYPE_CHECKING: _config_table isinstance-checks against it.
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import tomlkit
from tomlkit.exceptions import TOMLKitError

from photo_tagger import telemetry
from photo_tagger.config import (
    DEFAULT_BATTERY_FLOOR,
    DEFAULT_BATTERY_RESUME,
    DEFAULT_DUTY_PERCENT,
    DEFAULT_MAX_PAUSE_MINUTES,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_SIDECAR_MODE,
    DEFAULT_WORK_MINUTES,
    SidecarMode,
)
from photo_tagger.csv_report import ReportRow
from photo_tagger.discovery import parse_extensions, resolve_image_files
from photo_tagger.errors import ConfigFileError, FieldError
from photo_tagger.fields import ExtraFields, build_extra_fields, non_ascii_tags
from photo_tagger.i18n import AUTO, _, gettext_noop, ngettext, pgettext
from photo_tagger.keywords import dedupe_keywords, merge_keywords
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    CaptionValue,
    select_camera_fields,
    select_location,
)
from photo_tagger.models import KeywordSet
from photo_tagger.pacing import (
    CAUTION_BATTERY_FLOOR,
    FULL_DUTY,
    SECONDS_PER_MINUTE,
    PacingPlan,
)
from photo_tagger.pipeline import MAX_TRACKED_DROPPED_TERMS
from photo_tagger.sessions import build_session_vocabulary, plan_sessions
from photo_tagger.undo import (
    CHANGED,
    DELETED,
    FAILED as UNDO_FAILED,
    MISSING,
    NO_BACKUP,
    RESTORED,
)
from photo_tagger.vocabulary import VocabularyError, load_vocabulary
from photo_tagger.watch import DEFAULT_INTERVAL_SECONDS, DEFAULT_SETTLE_SECONDS


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from tomlkit import TOMLDocument

    from photo_tagger.undo import UndoResult
    from photo_tagger.vocabulary import Vocabulary


# Per-photo status values, shown as an icon/word in the tree.
PENDING = "pending"  # added, not yet generated
WORKING = "working"  # generation in flight
READY = "ready"  # a proposal is available to review
SAVED = "saved"  # written to the file
FAILED = "failed"  # generation or save failed

# A broad, common default for the GUI's folder-scan extensions. Each distinct extension is listed
# because matching is case-insensitive but not variant-aware (jpg does not cover jpeg).
DEFAULT_GUI_EXTENSIONS = "jpg,jpeg,png,dng,cr3,nef,arw,heic,heif,tif,tiff,webp"

# Pre-filled choices for the Metadata Language menu. The values are English names on purpose:
# they are spliced into the (English) system prompt as-is; the GUI translates them for display
# only. The menu's Other... entry accepts any language the model understands, so this list does
# not limit the choice. The first entry must stay DEFAULT_OUTPUT_LANGUAGE (a test asserts it).
OUTPUT_LANGUAGE_SUGGESTIONS = (
    gettext_noop("English"),
    gettext_noop("Brazilian Portuguese"),
    gettext_noop("Dutch"),
    gettext_noop("French"),
    gettext_noop("German"),
    gettext_noop("Italian"),
    gettext_noop("Japanese"),
    gettext_noop("Korean"),
    gettext_noop("Portuguese"),
    gettext_noop("Russian"),
    gettext_noop("Simplified Chinese"),
    gettext_noop("Spanish"),
)

# Substrings that hint a model is vision-capable, used to surface likely picks first.
_VISION_HINTS = (
    "vl",
    "vision",
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "cogvlm",
    "internvl",
    "pixtral",
    "gemma3",
    "smolvlm",
    "-v-",
)


# Where tooltips are broken into lines. Qt only word-wraps a tooltip it takes for rich text, so a
# long plain-text one is drawn as a single line, often wider than the window. Hard-wrapping keeps
# them readable blocks. 72 columns is narrow enough to scan and wide enough to avoid ragged text.
TOOLTIP_WIDTH = 72


def wrap_tooltip(text: str, width: int = TOOLTIP_WIDTH) -> str:
    """
    Break *text* into lines of at most *width* characters for display in a tooltip.

    Line breaks already in the text are kept (each line is wrapped on its own), so a tooltip written
    as two short paragraphs stays two paragraphs, and re-wrapping an already-wrapped tooltip changes
    nothing.
    """
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else line for line in text.splitlines()
    )


def tooltip(message: str, /, **values: object) -> str:
    """
    Translate a tooltip *message*, fill in its ``{placeholders}``, and wrap it for display.

    The wrapping happens after translation on purpose: a translated tooltip is often longer than the
    English one, so hard-coding the breaks in the source strings would leave the catalogs ragged.
    Babel extracts from this function too (see ``scripts/extract_translations.py``), so tooltips
    stay translatable without a separate ``_()`` call.
    """
    text = _(message)
    return wrap_tooltip(text.format(**values) if values else text)


@dataclass(slots=True)
class PhotoItem:
    """
    Mutable per-photo state shared between the file tree and the detail pane.

    ``existing_*`` holds what was read off the file; ``title``/``description``/ ``keywords`` are the
    editable working copy that :func:`keywords_to_save` and the Save action write. The working copy
    is seeded from a :class:`Proposal` (or from the existing values when the user opens a file
    without generating).

    The trailing read-context fields (``camera_info``/``location_tags``/``gps_position``) and the
    token/timing counters are captured when a proposal is generated and feed the CSV export; they
    stay empty for photos that were never generated.

    ``hint`` is the photographer's note for the model, typed in the detail pane. It is a generation
    input, not metadata: it rides along in the prompt whenever this photo is generated and is never
    written to the file.
    """

    path: Path
    selected: bool = True
    status: str = PENDING
    error: str = ""
    hint: str = ""
    loaded: bool = False
    # Which indicator fields (title/description/keywords) the file already carries. None means
    # the background metadata scan has not reported yet; a set (possibly empty) means it has.
    known_fields: set[str] | None = None
    existing_title: str | None = None
    existing_description: str | None = None
    existing_keywords: KeywordSet = field(default_factory=KeywordSet)
    existing_sources: list[str] = field(default_factory=list)
    # Every title and description found, per field, the one in use first. What it adds over the two
    # fields above is the value a sidecar shadows, which the detail pane names rather than hide.
    existing_caption_sources: dict[str, list[CaptionValue]] = field(default_factory=dict)
    sources_read: bool = False
    has_proposal: bool = False
    from_cache: bool = False
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    camera_info: dict[str, str] = field(default_factory=dict)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
    # Mixed-script bits of the generated text, carried over from the proposal so the detail pane
    # can point at them. Cleared when the photo is generated again.
    foreign_terms: list[str] = field(default_factory=list)
    # The proposal as the window first showed it, so telemetry can count what was changed by hand
    # before a save (the counts leave the machine, never the text). proposed_at is a monotonic
    # clock reading, None until the photo is generated.
    proposed_title: str = ""
    proposed_description: str = ""
    proposed_keywords: list[str] = field(default_factory=list)
    proposed_at: float | None = None
    hinted: bool = False


@dataclass(frozen=True, slots=True)
class Proposal:
    """
    One file's AI proposal plus the existing metadata and read context alongside it.

    ``keywords`` is what a save would write, so a controlled vocabulary has already had its say by
    the time a proposal reaches the window; the two ``vocabulary_*`` fields report what it did, for
    the run's summary line.

    ``foreign_terms`` is non-empty when the generated text mixes writing systems (see
    :mod:`photo_tagger.script_guard`), which the detail pane says out loud before a save.
    """

    path: Path
    existing_title: str | None
    existing_description: str | None
    existing_keywords: KeywordSet
    title: str
    description: str
    keywords: list[str]
    camera_info: dict[str, str] = field(default_factory=dict)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    from_cache: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
    vocabulary_mapped: int = 0
    vocabulary_dropped: list[str] = field(default_factory=list)
    foreign_terms: list[str] = field(default_factory=list)
    # Whether a hint went into the prompt, and whether a second call rewrote mixed-script text.
    hinted: bool = False
    repaired: bool = False


def expand_inputs(
    paths: Iterable[Path],
    image_extensions: str,
    *,
    recursive: bool,
) -> list[Path]:
    """
    Expand dropped files and folders into a de-duplicated list of image files.

    Reuses the CLI's discovery: directories are walked and extension-filtered, explicit files are
    kept as is, and order is preserved. An empty extension string yields no files rather than
    raising.
    """
    ext_set = parse_extensions(image_extensions)
    if not ext_set:
        return []
    return resolve_image_files(list(paths), ext_set, recursive=recursive)


def new_paths(existing: Iterable[Path], found: Iterable[Path]) -> list[Path]:
    """Return entries of *found* not already present in *existing* (order preserved)."""
    seen = set(existing)
    out: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def group_by_parent(paths: Iterable[Path]) -> list[tuple[Path, list[Path]]]:
    """Group file paths under their parent directory, first-seen order preserved."""
    groups: dict[Path, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent, []).append(path)
    return list(groups.items())


def parse_keyword_lines(text: str) -> list[str]:
    """
    Parse the editable keyword field (one keyword per line) into a clean list.

    Repeats are collapsed case-insensitively so the change counts stay honest; the model (and a
    pasting user) sometimes repeats a keyword.
    """
    return dedupe_keywords(text.splitlines())


def keywords_to_text(keywords: list[str]) -> str:
    """Render keywords one per line for the editable text field."""
    return "\n".join(keywords)


def keywords_to_save(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> KeywordSet:
    """
    Build the :class:`KeywordSet` to write from the edited keywords.

    Merges with the existing keywords unless *overwrite* is set, in which case the existing keywords
    are dropped first. Hierarchical entries (``Duck<Bird<Animal``) are parsed by
    :func:`merge_keywords` exactly as the CLI does.

    *verbatim* carries the controlled vocabulary's own spelling of each term (its ``exact`` index),
    which the merge step must not touch: a catalog that keeps its keywords in lower case means it,
    and title-casing "gegenlicht" on the way out would write the near-duplicate the vocabulary
    exists to prevent. The CLI's own write path passes the same index.
    """
    base = KeywordSet() if overwrite else existing
    return merge_keywords(base, edited_keywords, verbatim=verbatim)


@dataclass(frozen=True, slots=True)
class SaveOptions:
    """
    The state of the save toggles as one value object.

    Bundling them keeps the write rules (which fields, merge or overwrite, sidecar or embedded) out
    of the Qt shell, so :func:`build_save_job` is testable without a window.
    """

    write_title: bool = True
    write_description: bool = True
    write_keywords: bool = True
    # One "replace or keep" answer per field. Keywords merge, so keeping them is the default; a
    # title and a description hold one value each, so the generated one replaces it.
    overwrite_title: bool = True
    overwrite_description: bool = True
    overwrite_keywords: bool = False
    backup: bool = True
    # Sidecar, embedded, or one per file type: resolved per photo by metadata.use_sidecar_for.
    sidecar_mode: SidecarMode = DEFAULT_SIDECAR_MODE
    # The active vocabulary's own spelling of each term, keyed by casefolded term. None when no
    # vocabulary is in force, which is the only time the merge step may capitalize freely.
    verbatim: Mapping[str, str] | None = None
    # Copyright and credit (Settings > Copyright). Written on every save, like the CLI's own
    # [copyright] table, and dated per photo, so it is resolved at write time rather than here.
    extra_fields: ExtraFields | None = None

    @property
    def any_field(self) -> bool:
        """Whether at least one field is picked; a save with none of them would write nothing."""
        return self.write_title or self.write_description or self.write_keywords

    @property
    def writes_copyright(self) -> bool:
        """Whether a save also stamps the copyright block."""
        return self.extra_fields is not None and not self.extra_fields.is_empty

    @property
    def writes_anything(self) -> bool:
        """
        Whether a save would put anything on the file at all.

        Wider than :attr:`any_field`: with a copyright configured, a save with every generated field
        switched off still writes, and stopping it would be wrong.
        """
        return self.any_field or self.writes_copyright


@dataclass(frozen=True, slots=True)
class SaveJob:
    """
    One photo's resolved write: exactly the values ExifTool should put on the file.

    Resolved on the UI thread (it reads the toggles and the edited fields) so the background writer
    only needs these plain values, never a widget.
    """

    path: Path
    keywords: KeywordSet
    title: str | None
    description: str | None

    @property
    def fields(self) -> set[str]:
        """Which indicator fields this job puts on the file, for the Tagged column."""
        return fields_written(self.title, self.description, self.keywords)


def _caption_to_save(
    edited: str,
    existing: str | None,
    *,
    write: bool,
    overwrite: bool,
) -> str | None:
    """
    Resolve one text field's saved value, None meaning "leave what the photo has".

    A field switched off is left alone, and so is one the photo already fills when its Overwrite
    toggle is off; only then does the edited text go on. Mirrors the CLI's own rule.
    """
    if not write:
        return None
    if not overwrite and existing:
        return None
    return edited or None


def build_save_job(item: PhotoItem, options: SaveOptions) -> SaveJob:
    """
    Resolve what saving *item* writes: only the checked fields, the unchecked ones left untouched.

    A field that is switched off becomes None (or an empty keyword set), which write_metadata leaves
    out of its payload. So does a field whose Overwrite toggle is off on a photo that already
    carries one.
    """
    keywords = (
        keywords_to_save(
            item.existing_keywords,
            item.keywords,
            overwrite=options.overwrite_keywords,
            verbatim=options.verbatim,
        )
        if options.write_keywords
        else KeywordSet()
    )
    return SaveJob(
        path=item.path,
        keywords=keywords,
        title=_caption_to_save(
            item.title,
            item.existing_title,
            write=options.write_title,
            overwrite=options.overwrite_title,
        ),
        description=_caption_to_save(
            item.description,
            item.existing_description,
            write=options.write_description,
            overwrite=options.overwrite_description,
        ),
    )


def resolve_extra_tags(
    options: SaveOptions,
    path: Path,
    years: Mapping[Path, int],
) -> dict[str, str]:
    """
    Render one photo's copyright tags, empty when the window writes none.

    *years* is a batch read of capture years (see :func:`~photo_tagger.metadata.read_capture_years`)
    so a save pays one ExifTool call for the whole batch. A photo missing from it falls back to this
    year: a notice dated slightly wrong beats one with a hole in it.
    """
    fields = options.extra_fields
    if fields is None or fields.is_empty:
        return {}
    return fields.resolve(years.get(path) or datetime.now(tz=UTC).astimezone().year)


def build_copyright_fields(
    *,
    artist: str,
    notice: str,
    rights: str,
    marked: bool | None,
    assignments: Iterable[str] | None = None,
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
    resolved = fields.resolve(year)
    lines = [
        ngettext(
            "Writes {n} tag. On a photo taken in {year}:",
            "Writes {n} tags. On a photo taken in {year}:",
            len(resolved),
        ).format(n=len(resolved), year=year),
    ]
    lines += [f"{tag} = {value}" for tag, value in resolved.items()]
    return "\n".join(lines)


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
    if not (tags := non_ascii_tags(fields.resolve(year))):
        return ""
    return _(
        "EXIF only allows plain ASCII before Exif 3.0, so some software will show {tags} "
        "wrongly. XMP carries the same text either way.",
    ).format(tags=", ".join(tags))


def apply_proposal(
    item: PhotoItem,
    proposal: Proposal,
    *,
    verbatim: Mapping[str, str] | None = None,
    now: float | None = None,
) -> None:
    """
    Fill *item*'s existing metadata and seed its editable copy from *proposal*.

    The keywords are normalized on the way in (see :func:`normalize_keyword_lines`), so the field
    shows the set a save would write rather than the model's two raw lists. *verbatim* is the
    vocabulary's own spelling of each term, which that merge must not touch. *now* stamps when the
    proposal arrived, for the time a review takes.
    """
    item.existing_title = proposal.existing_title
    item.existing_description = proposal.existing_description
    item.existing_keywords = proposal.existing_keywords
    item.loaded = True
    item.title = proposal.title
    item.description = proposal.description
    item.keywords = normalize_keyword_lines(list(proposal.keywords), verbatim=verbatim)
    item.camera_info = proposal.camera_info
    item.location_tags = proposal.location_tags
    item.gps_position = proposal.gps_position
    item.input_tokens = proposal.input_tokens
    item.output_tokens = proposal.output_tokens
    item.total_tokens = proposal.total_tokens
    item.seconds = proposal.seconds
    item.has_proposal = True
    item.from_cache = proposal.from_cache
    item.foreign_terms = list(proposal.foreign_terms)
    item.status = READY
    item.error = ""
    item.proposed_title = item.title
    item.proposed_description = item.description
    item.proposed_keywords = list(item.keywords)
    item.proposed_at = now
    item.hinted = proposal.hinted


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

    The dimmed line under the field: which file the shown value came from, plus a line for any
    other file holding a different value. That second case is the point: a caption inside the photo
    is still there and still what ``exiftool photo.dng`` prints, but the sidecar wins on read, so it
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


# Appended to a flagged photo's Status cell. A glyph rather than a word: the column is narrow, it
# has to survive translation, and it reads at a glance down a list of hundreds of rows.
MIXED_SCRIPT_MARK = "⚠"  # warning sign


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


def photo_item_to_report_row(
    item: PhotoItem,
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> ReportRow:
    """
    Flatten a GUI :class:`PhotoItem` into a CSV :class:`ReportRow`.

    The keyword columns reflect what a Save would write: the working keywords merged with (or, when
    *overwrite*, replacing) the existing ones, exactly as :func:`keywords_to_save` computes for the
    Save action. The EXIF and token columns are whatever was captured at generation time, and stay
    blank for a photo that was added but never generated.
    """
    to_write = keywords_to_save(
        item.existing_keywords,
        item.keywords,
        overwrite=overwrite,
        verbatim=verbatim,
    )
    model, lens, captured = select_camera_fields(item.camera_info)
    city, country = select_location(item.location_tags)
    return ReportRow(
        file=str(item.path),
        filename=item.path.name,
        status=item.status,
        error=item.error,
        title=item.title,
        description=item.description,
        keywords=list(to_write.subject),
        hierarchical_keywords=list(to_write.hierarchical),
        existing_keywords=list(item.existing_keywords.subject),
        existing_title=item.existing_title or "",
        existing_description=item.existing_description or "",
        camera_model=model or "",
        lens_model=lens or "",
        capture_date=captured or "",
        gps_position=item.gps_position or "",
        city=city or "",
        country=country or "",
        input_tokens=item.input_tokens,
        output_tokens=item.output_tokens,
        total_tokens=item.total_tokens,
        seconds=item.seconds,
        from_cache=item.from_cache if item.has_proposal else None,
        foreign_terms=list(item.foreign_terms),
    )


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


def _extension(path: Path) -> str:
    """Return a path's extension, lowercased and without the dot ("" when it has none)."""
    return path.suffix.lstrip(".").lower()


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


@dataclass(slots=True)
class FolderNode:
    """A folder in the file tree, with its display label, subfolders, and files."""

    path: Path
    label: str
    folders: list[FolderNode]
    files: list[Path]


def _raw_dirs(paths: list[Path]) -> dict[Path, FolderNode]:
    """Build a directory -> node map linking every file's ancestor chain."""
    nodes: dict[Path, FolderNode] = {}
    suborder: dict[Path, list[Path]] = {}

    def ensure(directory: Path) -> None:
        if directory in nodes:
            return
        nodes[directory] = FolderNode(path=directory, label=directory.name, folders=[], files=[])
        suborder[directory] = []
        parent = directory.parent
        if parent != directory:  # stop at the filesystem root, whose parent is itself
            ensure(parent)
            suborder[parent].append(directory)

    for file in paths:
        ensure(file.parent)
        nodes[file.parent].files.append(file)
    for parent, subs in suborder.items():
        nodes[parent].folders = [nodes[s] for s in subs]
    return nodes


def _first_significant(node: FolderNode) -> FolderNode:
    """Descend through single-child, file-less folders to the first meaningful node."""
    while not node.files and len(node.folders) == 1:
        node = node.folders[0]
    return node


def _display_node(node: FolderNode, parent_path: Path | None) -> FolderNode:
    """Re-label *node* relative to its display parent and collapse its child chains."""
    label = str(node.path) if parent_path is None else str(node.path.relative_to(parent_path))
    folders = [_display_node(_first_significant(child), node.path) for child in node.folders]
    return FolderNode(path=node.path, label=label, folders=folders, files=list(node.files))


def build_tree(paths: Iterable[Path]) -> list[FolderNode]:
    """
    Group file paths into a nested folder tree for the GUI.

    Leading single-child directory chains are collapsed (``a/b/c`` shows as one node when only ``c``
    holds files), subfolders nest under their parent, and disjoint roots become separate top-level
    nodes. Each node's ``label`` is its path relative to its display parent (the absolute path for a
    top-level node).
    """
    paths = list(paths)
    if not paths:
        return []
    nodes = _raw_dirs(paths)
    tops: list[FolderNode] = []
    for root in (d for d in nodes if d.parent == d):
        node = _first_significant(nodes[root])
        if node.path.parent == node.path and not node.files and len(node.folders) > 1:
            # The filesystem root is just a container for disjoint trees; promote each branch.
            tops.extend(_first_significant(child) for child in node.folders)
        else:
            tops.append(node)
    return [_display_node(top, None) for top in tops]


def flat_labels(paths: Iterable[Path]) -> dict[Path, str]:
    """
    Label every path for the flat list: its filename, or its path below the folder they share.

    Flattening drops the folder each photo came from, which matters the moment two of them are
    called DSC_0042.NEF. One folder's worth of photos therefore keeps plain filenames, and a list
    spanning several shows just the part of the path that tells them apart. Paths with nothing in
    common (two drives on Windows) fall back to the full path, which always does.
    """
    paths = list(paths)
    if len({path.parent for path in paths}) <= 1:
        return {path: path.name for path in paths}
    try:
        root = os.path.commonpath([str(path) for path in paths])
    except ValueError:
        return {path: str(path) for path in paths}
    return {path: str(path.relative_to(root)) for path in paths}


def paths_under(paths: Iterable[Path], folder: Path) -> list[Path]:
    """Return the paths that live under *folder* (at any depth), order preserved."""
    return [path for path in paths if path.is_relative_to(folder)]


@dataclass(frozen=True, slots=True)
class Location:
    """One place the right-hand pane can show: a folder's thumbnail grid, or a photo's detail."""

    path: Path
    is_dir: bool


def location_crumb(location: Location) -> str:
    """
    Name a place for the Back/Forward hints and the pane's header.

    A photo carries its folder ("Shoot 1 / DSC_0042.NEF") because the file name alone is rarely
    enough to tell two frames of the same shoot apart.
    """
    if location.is_dir:
        return location.path.name or str(location.path)
    folder = location.path.parent.name
    return f"{folder} / {location.path.name}" if folder else location.path.name


# How many places back the history remembers. Deep enough that reviewing a shoot never runs out of
# trail, shallow enough that the whole thing stays a handful of strings.
HISTORY_LIMIT = 100


class NavigationHistory:
    """
    Where the user has been in the right-hand pane, so Back and Forward can retrace it.

    Browser semantics: visiting a place pushes the one being left onto the back trail and drops the
    forward trail, so what Back offers is always the route actually taken. Re-visiting the place
    already shown is ignored, which is what makes it safe to record every navigation, including the
    ones Back and Forward cause themselves.
    """

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        """Start an empty trail, remembering at most *limit* places to go back to."""
        self._limit = limit
        self._back: list[Location] = []
        self._forward: list[Location] = []
        self._current: Location | None = None

    @property
    def current(self) -> Location | None:
        """The place shown now, or None when the pane is showing nothing."""
        return self._current

    def visit(self, location: Location) -> None:
        """Record *location* as the place now shown, ending the forward trail."""
        if location == self._current:
            return
        if self._current is not None:
            self._back.append(self._current)
            if len(self._back) > self._limit:
                del self._back[0]  # one place at a time, so dropping the oldest is enough
        self._current = location
        self._forward.clear()

    def leave(self) -> None:
        """Note that the pane went empty, so Back returns to the last place rather than past it."""
        if self._current is not None:
            self._back.append(self._current)
            self._current = None

    def peek_back(self) -> Location | None:
        """Return where Back would go (for naming it), or None when nothing is behind."""
        return self._back[-1] if self._back else None

    def peek_forward(self) -> Location | None:
        """Return where Forward would go, or None until a Back has been taken."""
        return self._forward[-1] if self._forward else None

    def back(self) -> Location | None:
        """Step one place back and return it, or None when there is nothing behind."""
        return self._step(self._back, self._forward)

    def forward(self) -> Location | None:
        """Step one place forward and return it, or None when there is nothing ahead."""
        return self._step(self._forward, self._back)

    def _step(self, source: list[Location], sink: list[Location]) -> Location | None:
        if not source:
            return None
        if self._current is not None:
            sink.append(self._current)
        self._current = source.pop()
        return self._current

    def prune(self, keep: Callable[[Location], bool]) -> None:
        """
        Forget every place *keep* rejects, e.g. a photo just removed from the list.

        Dropping the place currently shown leaves the pane on nothing, so Back then lands on the
        last surviving place before it instead of skipping over it.
        """
        self._back = [location for location in self._back if keep(location)]
        self._forward = [location for location in self._forward if keep(location)]
        if self._current is not None and not keep(self._current):
            self._current = None


def rank_vision_models(model_ids: Iterable[str]) -> list[str]:
    """
    Order model ids with likely vision-capable ones first, keeping all of them.

    The provider listings do not reliably flag modality, so this is a name heuristic (``vl``,
    ``vision``, ``llava``, ...) used only to surface probable picks; nothing is hidden, so a model
    the heuristic misses is still selectable.
    """
    model_ids = list(model_ids)
    likely = [m for m in model_ids if any(hint in m.lower() for hint in _VISION_HINTS)]
    likely_set = set(likely)
    others = [m for m in model_ids if m not in likely_set]
    return likely + others


# A nested name -> children mapping used to fold cumulative hierarchy paths into one tree.
type _Tree = dict[str, "_Tree"]


def chain_to_display(path: str) -> str:
    """
    Convert a Lightroom root-first ``A|B|C`` path to the GUI's leaf-first ``C<B<A`` notation.

    The editable keyword field already speaks the '<' form (it is what the model emits and what the
    CLI documents), so every read-only view uses it too instead of leaking the on-disk '|'.
    """
    return "<".join(reversed(path.split("|")))


def keyword_lines(keywords: KeywordSet) -> list[str]:
    """
    Render a keyword set as the shortest list of lines that reproduces it, sorted.

    Each hierarchy shows once, as its deepest chain (``Duck<Bird<Animal``); the intermediate flat
    copies Lightroom also stores (Animal, Bird) are folded into it. Flat keywords that belong to no
    hierarchy join them. This mirrors what a user would type to reproduce the same metadata.

    Sorting is what lets the Existing and New columns be read side by side: a keyword lands on the
    same line in both. The '<' notation is leaf-first, so sorting the rendered line sorts by leaf.
    """
    chains = [entry for entry in keywords.hierarchical if "|" in entry]
    deepest = [
        entry
        for entry in chains
        if not any(other.startswith(entry + "|") for other in chains if other != entry)
    ]
    covered = {segment.casefold() for entry in chains for segment in entry.split("|")}
    lines = [chain_to_display(entry) for entry in deepest]
    lines += [kw for kw in keywords.subject if kw.casefold() not in covered]
    return sorted(lines, key=str.casefold)


def format_existing_keywords(keywords: KeywordSet) -> str:
    """Render existing keywords one per line, in the same ``<`` notation as the editable field."""
    return "\n".join(keyword_lines(keywords))


def normalize_keyword_lines(
    keywords: list[str],
    *,
    verbatim: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Fold an edited keyword list into the shortest list that writes the same metadata, sorted.

    A model returns its flat keywords and its hierarchies as two lists, so every leaf reached the
    field twice (``Beach`` and ``Beach<Sandy Area<Outdoor Area``). Merging expands every chain into
    its levels, so the bare copies were noise. Run through the save's own merge, so the field shows
    what the file would get. Applied when a proposal arrives, never while the user types.
    """
    return keyword_lines(merge_keywords(KeywordSet(), keywords, verbatim=verbatim))


def _walk_tree(node: _Tree, prefix: str, lines: list[str]) -> None:
    """Append *node*'s children to *lines* with ``tree``-style guide characters."""
    entries = list(node.items())
    for index, (name, child) in enumerate(entries):
        last = index == len(entries) - 1
        lines.append(prefix + ("└─ " if last else "├─ ") + name)
        _walk_tree(child, prefix + ("   " if last else "│  "), lines)


def hierarchy_tree_text(paths: Iterable[str]) -> str:
    """
    Render Lightroom ``A|B|C`` paths as a ``tree``-style diagram with branch guides.

    Cumulative paths ("A|B", "A|B|C") collapse into one branch, so the view shows the taxonomy shape
    rather than repeating every prefix line. Roots sit at column zero and children hang off
    ``├─``/``└─`` connectors like the CLI ``tree`` command, which reads far better than plain two-
    space indentation.
    """
    root: _Tree = {}
    for path in paths:
        node = root
        for segment in path.split("|"):
            node = node.setdefault(segment, {})

    lines: list[str] = []
    for name, child in root.items():
        lines.append(name)
        _walk_tree(child, "", lines)
    return "\n".join(lines)


def hierarchy_preview(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> str:
    """Render the keyword tree that saving the edited keywords would produce."""
    return hierarchy_tree_text(
        keywords_to_save(
            existing,
            edited_keywords,
            overwrite=overwrite,
            verbatim=verbatim,
        ).hierarchical,
    )


# Diff states for a keyword when comparing the existing flat subjects to what a save writes.
ADDED = "added"
REMOVED = "removed"
UNCHANGED = "unchanged"


def keyword_diff(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """
    Compare existing flat keywords to the result of saving the edited keywords.

    Returns ``(keyword, state)`` pairs where state is :data:`ADDED`, :data:`REMOVED`, or
    :data:`UNCHANGED`. The keywords that will be written come first in write order, then any that
    would be dropped (only possible with *overwrite*). Comparison is case-insensitive.
    """
    result = keywords_to_save(
        existing,
        edited_keywords,
        overwrite=overwrite,
        verbatim=verbatim,
    ).subject
    existing_folds = {kw.casefold() for kw in existing.subject}
    result_folds = {kw.casefold() for kw in result}
    diff = [(kw, UNCHANGED if kw.casefold() in existing_folds else ADDED) for kw in result]
    diff += [(kw, REMOVED) for kw in existing.subject if kw.casefold() not in result_folds]
    return diff


# How many rejected terms the vocabulary status line names before it stops. The point is to show
# what kind of keyword is being dropped, not to list every one; the log has them all.
_MAX_LISTED_DROPPED = 5


@dataclass(frozen=True, slots=True)
class VocabularyOutcome:
    """
    What a controlled vocabulary made of one photo's generated keywords.

    ``mapped`` counts the keywords rewritten to the catalog's own spelling and hierarchy, and
    ``dropped`` names the ones strict mode refused, so a run can report both instead of silently
    changing what the model said.
    """

    keywords: list[str] = field(default_factory=list)
    mapped: int = 0
    dropped: list[str] = field(default_factory=list)


def apply_vocabulary(
    keywords: Iterable[str],
    vocabulary: Vocabulary | None,
    *,
    strict: bool = False,
) -> VocabularyOutcome:
    """
    Snap generated keywords onto *vocabulary*, or pass them through when there is none.

    The GUI applies this to a proposal before it reaches the review pane, so what you edit is what a
    save writes, exactly as the CLI's ``--vocabulary`` rewrites keywords before merging them.
    """
    if not vocabulary:
        return VocabularyOutcome(keywords=list(keywords))
    result = vocabulary.snap(keywords, strict=strict)
    return VocabularyOutcome(
        keywords=result.keywords,
        mapped=len(result.mapped),
        dropped=list(result.dropped),
    )


def record_dropped_terms(tally: dict[str, int], dropped: Iterable[str]) -> None:
    """
    Fold one photo's rejected keywords into the run's tally, counting how often each came up.

    New terms stop being recorded past :data:`~photo_tagger.pipeline.MAX_TRACKED_DROPPED_TERMS`,
    the same bound the CLI keeps: a run against the wrong vocabulary rejects thousands of distinct
    keywords, and the summary only ever names the busiest few of them anyway.
    """
    for term in dropped:
        if term in tally:
            tally[term] += 1
        elif len(tally) < MAX_TRACKED_DROPPED_TERMS:
            tally[term] = 1


def load_vocabulary_file(
    path: Path,
    *,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> tuple[Vocabulary | None, str]:
    """
    Load a vocabulary file, returning it or the message explaining why it could not be used.

    The window shows the message rather than raising: choosing the wrong file is a normal mistake,
    and the run it would have affected has not started yet.
    """
    try:
        return load_vocabulary(path, output_language=output_language), ""
    except VocabularyError as exc:
        return None, str(exc)


def vocabulary_status(path: Path | None, vocabulary: Vocabulary | None, error: str = "") -> str:
    """Describe the vocabulary in force, for the label under the file picker."""
    if error:
        return error
    if path is None or vocabulary is None:
        return _("No vocabulary: keywords are written as the model wrote them.")
    terms = ngettext("{n} keyword", "{n} keywords", len(vocabulary.terms)).format(
        n=len(vocabulary.terms),
    )
    return _("{terms} from {name}.").format(terms=terms, name=path.name)


def review_summary(flagged: int) -> str:
    """
    One line for the status bar naming how many photos in the run mix writing systems.

    Empty when none do. The line names the menu path that isolates them: a bare count sent people
    scrolling a list of hundreds for a row it had never said how to find. :func:`mark_mixed_script`
    marks those rows and :func:`foreign_terms_note` says which words, once you are on one.
    """
    if not flagged:
        return ""
    return ngettext(
        "{n} photo mixes writing systems (Select > Check Only > Status).",
        "{n} photos mix writing systems (Select > Check Only > Status).",
        flagged,
    ).format(n=flagged)


def failure_summary(failed: int) -> str:
    """
    One line for the status bar naming how many photos of a run did not generate.

    Empty when none did. A run that writes by itself ends on its save's tally, which counts only the
    photos it wrote, so without this line a batch left alone comes back reading as a clean success
    it was not. The line names the button that tries again.
    """
    if not failed:
        return ""
    return ngettext(
        "{n} photo failed (Retry Failed to try again).",
        "{n} photos failed (Retry Failed to try again).",
        failed,
    ).format(n=failed)


def vocabulary_summary(mapped: int, dropped: Mapping[str, int]) -> str:
    """
    One line for the status bar after a run: what the vocabulary changed across the batch.

    Empty when it changed nothing, so a run with a vocabulary that already fits says nothing rather
    than reporting two zeros. The named terms are the most frequent rejections, which is what tells
    you whether the vocabulary needs a new keyword or the photos need a different one.
    """
    parts: list[str] = []
    if mapped:
        parts.append(
            ngettext("{n} keyword rewritten", "{n} keywords rewritten", mapped).format(n=mapped),
        )
    if dropped:
        ranked = sorted(dropped.items(), key=lambda item: (-item[1], item[0].casefold()))
        listed = ", ".join(term for term, _count in ranked[:_MAX_LISTED_DROPPED])
        if len(ranked) > _MAX_LISTED_DROPPED:
            listed += ", ..."
        parts.append(
            ngettext(
                "{n} keyword dropped ({terms})",
                "{n} keywords dropped ({terms})",
                len(ranked),
            ).format(n=len(ranked), terms=listed),
        )
    if not parts:
        return ""
    return _("Vocabulary: {summary}.").format(summary=", ".join(parts))


@dataclass(frozen=True, slots=True)
class HarmonizeResult:
    """
    What harmonizing the generated proposals changed.

    ``keywords`` maps a photo's key (its path as a string, as the window holds it) to its new
    keyword list, and only carries the photos that actually changed.
    """

    keywords: dict[str, list[str]] = field(default_factory=dict)
    sessions: int = 0


def harmonize_sessions(
    keywords_by_path: Mapping[Path, list[str]],
    *,
    gap_minutes: float,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> HarmonizeResult:
    """
    Group the generated photos into shoots and make each shoot's keywords agree with itself.

    The CLI holds a session's writes until every photo in it has been analyzed; the GUI writes
    nothing until you press Save, so the same harmonization runs over the finished proposals
    instead. Either way the vocabulary comes from the session's own output, which is what makes it
    deterministic.

    Returns an empty result when the feature is off (*gap_minutes* of zero) or there is nothing to
    group.
    """
    paths = list(keywords_by_path)
    plan = plan_sessions(paths, gap_minutes=gap_minutes)
    if plan is None:
        return HarmonizeResult()
    changed: dict[str, list[str]] = {}
    for session in plan.sessions:
        vocabulary = build_session_vocabulary(
            (keywords_by_path[path] for path in session),
            output_language=output_language,
        )
        if not vocabulary:
            continue
        for path in session:
            snapped = vocabulary.snap(keywords_by_path[path]).keywords
            if snapped != keywords_by_path[path]:
                changed[str(path)] = snapped
    return HarmonizeResult(keywords=changed, sessions=len(plan.sessions))


def harmonize_summary(result: HarmonizeResult) -> str:
    """Status line after harmonizing: how many shoots there were, and how many photos changed."""
    if not result.sessions:
        return _("Nothing to harmonize yet: generate some photos first.")
    shoots = ngettext("{n} shoot", "{n} shoots", result.sessions).format(n=result.sessions)
    changed = len(result.keywords)
    if not changed:
        return _("{shoots}: the keywords already agreed.").format(shoots=shoots)
    return ngettext(
        "{shoots}: harmonized the keywords on {n} photo.",
        "{shoots}: harmonized the keywords on {n} photos.",
        changed,
    ).format(shoots=shoots, n=changed)


# How to read the timestamp a journal's filename starts with, by its width. The shorter one is a
# journal written before the stamp carried microseconds. Width decides, because strptime is happy
# to read the second-only stamp under the longer format and land on the wrong time.
_JOURNAL_STAMP_FORMATS = {20: "%Y%m%d%H%M%S%f", 14: "%Y%m%d%H%M%S"}

# What each undo outcome is called in the window. The module's own constants are log-facing
# identifiers; these are the phrases a user reads next to a file name.
_UNDO_ACTION_LABELS = {
    RESTORED: gettext_noop("restored"),
    DELETED: gettext_noop("deleted"),
    MISSING: gettext_noop("no longer there"),
    CHANGED: gettext_noop("changed since the run"),
    NO_BACKUP: gettext_noop("no backup to restore from"),
    UNDO_FAILED: gettext_noop("failed"),
}

# Outcomes that mean the file is back as it was; everything else was left alone.
UNDO_OK_ACTIONS = frozenset({RESTORED, DELETED})


def journal_time(path: Path) -> datetime | None:
    """Parse a journal's start time out of its filename, or None when the name is not ours."""
    stamp = path.name.split("-", 1)[0]
    time_format = _JOURNAL_STAMP_FORMATS.get(len(stamp))
    if time_format is None:
        return None
    try:
        return datetime.strptime(stamp, time_format).replace(tzinfo=UTC)
    except ValueError:
        return None


def journal_label(path: Path, entries: int) -> str:
    """Describe one recorded run for the undo list: when it ran, and how much it wrote."""
    started = journal_time(path)
    when = started.astimezone().strftime("%Y-%m-%d %H:%M") if started else path.stem
    files = ngettext("{n} file", "{n} files", entries).format(n=entries)
    return f"{when} · {files}"


def undo_action_label(action: str) -> str:
    """Translate one undo outcome for display, falling back to the raw name."""
    label = _UNDO_ACTION_LABELS.get(action)
    return _(label) if label else action


def undo_summary(results: Iterable[UndoResult]) -> str:
    """One line for the status bar: how many writes were put back, and how many were left alone."""
    outcomes = list(results)
    if not outcomes:
        return _("That run recorded no writes.")
    ok = sum(1 for result in outcomes if result.action in UNDO_OK_ACTIONS)
    left = len(outcomes) - ok
    restored = ngettext("Put back {n} file", "Put back {n} files", ok).format(n=ok)
    if not left:
        return f"{restored}."
    return _("{restored}; left {left} alone.").format(restored=restored, left=left)


@dataclass(frozen=True, slots=True)
class WatchSettings:
    """
    What a folder watch was started with.

    ``generate`` runs the model on each photo as it lands (the point of watching at all) while
    ``save`` writes the result without review. Saving is opt-in on purpose: the window's whole
    workflow is review-before-write, and a watch that writes unattended is the CLI's job.
    """

    folders: tuple[Path, ...] = ()
    extensions: str = DEFAULT_GUI_EXTENSIONS
    recursive: bool = True
    interval: float = DEFAULT_INTERVAL_SECONDS
    settle: float = DEFAULT_SETTLE_SECONDS
    generate: bool = True
    save: bool = False


def watch_status_text(settings: WatchSettings, *, added: int) -> str:
    """Status line while a watch runs: where it is looking, and what it has picked up so far."""
    folders = ", ".join(folder.name or str(folder) for folder in settings.folders)
    if not added:
        return _("Watching {folders} for new photos...").format(folders=folders)
    return ngettext(
        "Watching {folders}: {n} photo added so far.",
        "Watching {folders}: {n} photos added so far.",
        added,
    ).format(folders=folders, n=added)


# The extra passes a run that writes by itself makes before it saves, in the order it makes them.
# Both are the window doing by itself what a person at the keyboard would do next: press Retry
# Failed, and generate the odd photo whose text came back in two writing systems.
RERUN_FAILED = "failed"
RERUN_MIXED_SCRIPT = "mixed-script"
EXTRA_PASSES = (RERUN_FAILED, RERUN_MIXED_SCRIPT)


def photos_for_pass(name: str, items: Iterable[PhotoItem]) -> list[PhotoItem]:
    """
    Pick the photos an extra pass re-runs.

    The failed ones, or the ones whose generated text mixes writing systems: a second answer often
    comes back in one script, the way a second attempt often succeeds.
    """
    if name == RERUN_FAILED:
        return [item for item in items if item.status == FAILED]
    return [item for item in items if item.foreign_terms]


def extra_pass_status(name: str, count: int) -> str:
    """Say what a run about to ask the model again for part of its batch is doing."""
    if name == RERUN_FAILED:
        return ngettext(
            "Retrying {n} photo that failed...",
            "Retrying {n} photos that failed...",
            count,
        ).format(n=count)
    return ngettext(
        "Retrying {n} photo that mixed writing systems...",
        "Retrying {n} photos that mixed writing systems...",
        count,
    ).format(n=count)


def photos_to_write(items: Iterable[PhotoItem]) -> list[PhotoItem]:
    """
    Pick the photos a save nobody reviews is to write: generated, and not written yet.

    The caller decides the scope. A run that chains a save passes its own batch, so a proposal left
    over from an earlier run is not swept up; a watch passes the whole list, because it has no
    batch, only whatever has landed.
    """
    return [item for item in items if item.has_proposal and item.status == READY]


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


def status_summary(items: Iterable[PhotoItem]) -> str:
    """One-line counts for the status bar: selected, generated, saved, failed."""
    items = list(items)
    selected = sum(1 for i in items if i.selected)
    generated = sum(1 for i in items if i.has_proposal)
    saved = sum(1 for i in items if i.status == SAVED)
    failed = sum(1 for i in items if i.status == FAILED)
    files = ngettext("{n} file", "{n} files", len(items)).format(n=len(items))
    return _(
        "{files} · {selected} selected · {generated} generated · {saved} saved · {failed} failed",
    ).format(
        files=files,
        selected=selected,
        generated=generated,
        saved=saved,
        failed=failed,
    )


def search_summary(shown: int, total: int) -> str:
    """
    Lead the status line with what an active search is hiding.

    Without this a filtered list looks like a list that lost photos, and the counts beside it (which
    are of the whole list, since a hidden photo is still processed) would not add up.
    """
    return ngettext(
        "Showing {shown} of {total} photo",
        "Showing {shown} of {total} photos",
        total,
    ).format(shown=shown, total=total)


def format_duration(seconds: float) -> str:
    """
    Render a duration as ``m:ss``, growing to ``h:mm:ss`` once it passes an hour.

    Anything below zero clamps to ``0:00``: a long batch is timed against a monotonic clock, but a
    caller doing its own arithmetic should never be able to print a negative countdown.
    """
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def estimate_remaining(done: int, total: int, elapsed: float) -> float | None:
    """
    Estimate the seconds left, extrapolating from how long the finished items took.

    Returns None when there is nothing honest to say: before the first item finishes there is no
    rate yet, and once the batch is complete there is nothing left to wait for.
    """
    if done <= 0 or done >= total:
        return None
    return elapsed / done * (total - done)


def progress_timing_text(done: int, total: int, elapsed: float) -> str:
    """
    Build the readout shown next to the progress bar: time spent, plus the estimate of time left.

    The estimate joins in only once the first item has finished, so a run starts out showing bare
    elapsed time. That still tells the user the program is working, which is the point.
    """
    spent = format_duration(elapsed)
    remaining = estimate_remaining(done, total, elapsed)
    if remaining is None:
        return _("{elapsed} elapsed").format(elapsed=spent)
    return _("{elapsed} elapsed · {remaining} left").format(
        elapsed=spent,
        remaining=format_duration(remaining),
    )


def pacing_status_text(
    reason: str,
    seconds: float,
    *,
    duty_percent: int,
    resume_percent: int,
) -> str:
    """
    Status line while a paced run rests: why it stopped, and for how long.

    A battery rest ends when the charge comes back, which may be sooner than *seconds*, so that
    number is worded as the upper bound it is.
    """
    duration = format_duration(seconds)
    if reason == "battery":
        return _("Resting up to {duration} so the battery charges back to {percent}%.").format(
            duration=duration,
            percent=resume_percent,
        )
    return _("Resting {duration}: this run works {percent}% of the time.").format(
        duration=duration,
        percent=duty_percent,
    )


def _spoken_duration(seconds: float) -> str:
    """
    Render *seconds* as whole minutes, or as seconds while it is under a minute.

    Spelled out rather than the ``m:ss`` of :func:`format_duration`, because this one lands mid
    sentence, where "about 4 minutes of rest" reads and "about 4:00 of rest" does not. Rounding is
    honest in the direction that matters: a rest of a few seconds should say so, since that is the
    sign of a duty cycle set too close to 100 to do anything.
    """
    if seconds < SECONDS_PER_MINUTE:
        count = max(1, round(seconds))
        return ngettext("{n} second", "{n} seconds", count).format(n=count)
    minutes = max(1, round(seconds / SECONDS_PER_MINUTE))
    return ngettext("{n} minute", "{n} minutes", minutes).format(n=minutes)


def pacing_summary(plan: PacingPlan) -> str:
    """
    Describe what a run will actually do under *plan*, for the Pacing dialog to show live.

    The dialog needs this because its own fields do not say it: a duty cycle is a percentage, and
    what a person wants to know is how long the run stops for and how often.
    """
    if not plan.rests:
        return _("Left as it is, a run works flat out.")
    lines: list[str] = []
    if plan.duty_percent < FULL_DUTY:
        lines.append(
            _(
                "Works {percent}% of the time: about {rest} of rest after every {work} of work.",
            ).format(
                percent=plan.duty_percent,
                rest=_spoken_duration(plan.rest_for(plan.work_seconds)),
                work=_spoken_duration(plan.work_seconds),
            ),
        )
    if plan.battery_floor > 0:
        lines.append(
            _(
                "Whenever the battery falls to {floor}%, rests until it is back to {resume}%, "
                "for at most {cap}.",
            ).format(
                floor=plan.battery_floor,
                resume=plan.resume_percent,
                cap=_spoken_duration(plan.max_pause_seconds),
            ),
        )
    if plan.battery_floor > CAUTION_BATTERY_FLOOR:
        lines.append(
            _(
                "{mark} A plugged-in laptop is rarely that full, so the run would rest after "
                "nearly every window and wait out the cap each time.",
            ).format(mark=MIXED_SCRIPT_MARK),
        )
    return "\n".join(lines)


def navigation_shortcuts(platform_name: str) -> tuple[str, str, str]:
    """
    Return the Back, Forward, and enclosing-folder keys to use on *platform_name*.

    Each platform gets its own idiom: the browser keys on macOS (Qt reads a portable "Ctrl" as Cmd
    there) and the file manager keys elsewhere. Qt's own ``StandardKey.Back`` is Alt+Left on every
    platform, which on macOS is Option+Left, the word-left motion every text field needs, so the
    keys are chosen here rather than taken from Qt.
    """
    if platform_name == "darwin":
        return ("Ctrl+[", "Ctrl+]", "Ctrl+Up")
    return ("Alt+Left", "Alt+Right", "Alt+Up")


def reveal_label(platform_name: str) -> str:
    """Name the OS file browser for the context-menu action ("Reveal in Finder" on macOS)."""
    if platform_name == "darwin":
        return _("Reveal in Finder")
    if platform_name.startswith("win"):
        return _("Show in Explorer")
    return _("Show in File Manager")


def reveal_command(path: Path, platform_name: str) -> list[str] | None:
    """
    Return the argv that reveals *path* selected in the OS file browser, or None.

    None means the caller should fall back to opening the containing folder instead: the platform
    has no standard "reveal" command (Linux file managers vary), or the one it does have is not on
    PATH. The program is resolved here rather than left to the OS, which on Windows searches the
    current working directory (whichever folder the user last opened photos from) before PATH.
    """
    if platform_name == "darwin":
        argv = ["open", "-R", str(path)]
    elif platform_name.startswith("win"):
        # Explorer's /select switch takes the path in the same argument, comma-separated.
        argv = ["explorer", f"/select,{path}"]
    else:
        return None
    program = shutil.which(argv[0])
    return [program, *argv[1:]] if program is not None else None


def file_type_label(path: Path) -> str:
    """
    Label a photo's Type column: its extension, plus "+xmp" when an XMP sidecar sits beside it.

    The sidecar check is a plain filesystem stat, so this is cheap enough to run for every file at
    add time (no exiftool involved).
    """
    suffix = _extension(path)
    return f"{suffix}+xmp" if path.with_suffix(".xmp").exists() else suffix


# Image formats the picker always offers as hints, regardless of the user's configured file types.
# Format names are proper nouns, so they are not translated.
KNOWN_IMAGE_FORMATS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("JPEG", ("jpg", "jpeg")),
    ("PNG", ("png",)),
    ("Camera Raw", ("arw", "cr2", "cr3", "dng", "nef", "orf", "raf", "rw2")),
    ("HEIF", ("heic", "heif")),
    ("TIFF", ("tif", "tiff")),
    ("WebP", ("webp",)),
)


def _glob_patterns(extensions_without_dot: Iterable[str]) -> str:
    return " ".join(f"*.{ext}" for ext in extensions_without_dot)


def file_dialog_name_filters(extensions: str) -> list[str]:
    """
    Build Qt open-dialog name filters for the Add Photos picker.

    The first entry (the dialog's default) covers the user's configured file types, so the picker
    highlights what the pipeline will actually accept. The known image formats follow as hints,
    Photoshop-style, and an "All files" escape hatch keeps unusual formats openable.
    """
    filters: list[str] = []
    configured = sorted({ext.lstrip(".").lower() for ext in parse_extensions(extensions)})
    if configured:
        label = _("Your file types ({patterns})")
        filters.append(label.format(patterns=_glob_patterns(configured)))
    known = sorted({ext for _name, exts in KNOWN_IMAGE_FORMATS for ext in exts})
    filters.append(_("All known image formats ({patterns})").format(patterns=_glob_patterns(known)))
    filters.extend(f"{name} ({_glob_patterns(exts)})" for name, exts in KNOWN_IMAGE_FORMATS)
    filters.append(_("All files (*)"))
    return filters


# The Tagged indicator's letter and display label per field, in display order. The letters go
# through the catalog under a "Tagged column letter" msgctxt (see _tagged_letter), so a language
# whose field names start with other letters can remap them; the tooltips below spell out whatever
# letters are active, so they stay clear either way.
_FIELD_LETTERS = (
    (FIELD_TITLE, "T", gettext_noop("title")),
    (FIELD_DESCRIPTION, "D", gettext_noop("description")),
    (FIELD_KEYWORDS, "K", gettext_noop("keywords")),
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
        case _:  # pragma: no cover - _FIELD_LETTERS only ever supplies T, D, K
            message = f"no Tagged column letter translation for {letter!r}"
            raise ValueError(message)


def tagged_summary(fields: set[str]) -> str:
    """
    Compress a file's present metadata fields into the Tagged column label.

    "TDK" means title, description, and keywords all exist; "-" means the scan ran and found none.
    tagged_legend (the header tooltip) and tagged_tooltip (the cell tooltip) spell the letters out.
    """
    letters = "".join(
        _tagged_letter(letter)
        for field_name, letter, _label in _FIELD_LETTERS
        if field_name in fields
    )
    return letters or "-"


def tagged_legend() -> str:
    """Spell out every Tagged letter ("T = title, D = description, K = keywords"), localized."""
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


def fields_written(title: str | None, description: str | None, keywords: KeywordSet) -> set[str]:
    """
    Name the indicator fields a metadata write put on the file.

    Mirrors write_metadata's payload rules (empty values write nothing), so a successful save can
    update the Tagged column without re-running the exiftool presence scan.
    """
    written: set[str] = set()
    if title:
        written.add(FIELD_TITLE)
    if description:
        written.add(FIELD_DESCRIPTION)
    if not keywords.is_empty():
        written.add(FIELD_KEYWORDS)
    return written


# How the folder thumbnail grid orders its photos. These mirror the tree's sortable columns so the
# two views agree on what "by type" or "by status" means, and SORT_NAME is the default.
SORT_NAME = "name"
SORT_TYPE = "type"
SORT_STATUS = "status"
SORT_TAGGED = "tagged"


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


# Folder grid filters: which photos the grid shows. FILTER_ALL (the default) shows everything.
FILTER_ALL = "all"
FILTER_SELECTED = "selected"
FILTER_PENDING = "pending"
FILTER_GENERATED = "generated"
FILTER_SAVED = "saved"
FILTER_FAILED = "failed"
FILTER_UNTAGGED = "untagged"
FILTER_MIXED_SCRIPT = "mixed_script"


def photo_matches_filter(item: PhotoItem, criterion: str) -> bool:
    """
    Report whether *item* passes the grid filter *criterion*.

    The status filters key off the lifecycle state the badges show, FILTER_SELECTED follows the
    checkbox, and FILTER_UNTAGGED matches only once the metadata scan has run and found nothing (a
    None/unknown scan state does not match, so no photo is hidden while the scan is still pending).
    FILTER_MIXED_SCRIPT is what turns the run's "3 photos mix writing systems" line into three rows
    you can actually open. FILTER_ALL and any unrecognized value match everything.
    """
    # A dict of cheap boolean checks keeps this to one return (no PLR0911). FILTER_ALL is not a
    # key, so it falls through to the default True, as does any value the grid never sends.
    checks = {
        FILTER_SELECTED: item.selected,
        FILTER_PENDING: item.status == PENDING,
        FILTER_GENERATED: item.has_proposal,
        FILTER_SAVED: item.status == SAVED,
        FILTER_FAILED: item.status == FAILED,
        FILTER_UNTAGGED: item.known_fields is not None and not item.known_fields,
        FILTER_MIXED_SCRIPT: bool(item.foreign_terms),
    }
    return checks.get(criterion, True)


def filter_photos(items: Iterable[PhotoItem], criterion: str) -> list[PhotoItem]:
    """Keep only the photos that pass the grid filter *criterion*, order preserved."""
    return [item for item in items if photo_matches_filter(item, criterion)]


# Badge names for the folder grid's thumbnail overlays, in the order they are drawn.
BADGE_FAILED = "failed"
BADGE_SAVED = "saved"
BADGE_UNSAVED = "unsaved"
BADGE_METADATA = "metadata"
BADGE_SIDECAR = "sidecar"
BADGE_MIXED_SCRIPT = "mixed_script"


def thumb_badges(item: PhotoItem, *, has_sidecar: bool) -> list[str]:
    """
    Decide which overlay badges a photo's grid thumbnail shows.

    At most one lifecycle badge (failed beats saved beats unsaved-proposal), plus the already-has-
    metadata, sidecar, and mixed-script markers. The GUI maps each name to a color and glyph.

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
    if item.known_fields:
        badges.append(BADGE_METADATA)
    if has_sidecar:
        badges.append(BADGE_SIDECAR)
    if item.foreign_terms:
        badges.append(BADGE_MIXED_SCRIPT)
    return badges


# Zoom limits for the full-size image viewer, as a multiple of the image's own pixel size. The
# floor keeps a huge photo reachable in one "fit" step; the ceiling stops a click-and-hold from
# asking Qt to scale a 40-megapixel image into gigabytes of pixmap.
MIN_ZOOM = 0.02
MAX_ZOOM = 8.0
# One notch of zoom in or out. 1.25 is small enough to aim with and large enough to feel like
# progress: eight notches cover the whole 1:10 range a fit-to-window view usually spans.
ZOOM_STEP = 1.25


def clamp_zoom(zoom: float) -> float:
    """Hold a zoom factor inside the viewer's usable range."""
    return min(MAX_ZOOM, max(MIN_ZOOM, zoom))


def fit_zoom(image: tuple[int, int], viewport: tuple[int, int]) -> float:
    """
    Zoom factor that makes *image* fit entirely inside *viewport*.

    Never goes above 1.0: fitting a small photo to a big window would upscale it into a blurry mess,
    so a photo smaller than the window is shown at its own size instead. A degenerate size (a failed
    decode, or a window not laid out yet) falls back to 1.0.
    """
    width, height = image
    view_width, view_height = viewport
    if width <= 0 or height <= 0 or view_width <= 0 or view_height <= 0:
        return 1.0
    return clamp_zoom(min(1.0, view_width / width, view_height / height))


def step_zoom(zoom: float, notches: int) -> float:
    """Move *zoom* by *notches* steps in or out (negative zooms out), clamped to the range."""
    return clamp_zoom(zoom * ZOOM_STEP**notches)


def zoom_label(zoom: float) -> str:
    """Render a zoom factor as the percentage shown in the viewer's toolbar."""
    return f"{round(zoom * 100)}%"


def anchored_scroll(value: int, page_step: int, factor: float) -> int:
    """
    Where a scrollbar must land so the viewport keeps showing the same part of the image.

    After the image is rescaled by *factor*, the pixel that was at the center of the viewport has
    moved to ``factor`` times its old offset. Scrolling to that offset, minus half a viewport, puts
    it back under the user's eyes; without this, zooming walks the view toward the top-left corner.
    Negative results are clamped, since a scrollbar cannot go below its minimum.
    """
    return max(0, round(factor * value + (factor - 1) * page_step / 2))


def _toml_str(value: str) -> str:
    """Quote *value* as a TOML basic string."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_bool(value: bool) -> str:  # noqa: FBT001  # the bool is the value being rendered
    """Render a TOML boolean."""
    return "true" if value else "false"


def telemetry_toggle_note(reason: str, *, do_not_track: bool) -> str:
    """Explain the telemetry toggle when an environment variable decides it, else return ""."""
    if reason == telemetry.CONSENT_ENV_OPT_OUT:
        return _(
            "Turned off by PHOTO_TAGGER_TELEMETRY or PHOTO_TAGGER_NO_TELEMETRY in your "
            "environment. Remove it to choose here.",
        )
    if reason == telemetry.CONSENT_DO_NOT_TRACK:
        return _(
            "DO_NOT_TRACK is set in your environment, so nothing is sent. Check this to send "
            "anyway, for Photo Tagger only.",
        )
    if not do_not_track:
        return ""
    if reason == telemetry.CONSENT_GUI_OPT_IN:
        return _("Sending even though DO_NOT_TRACK is set, because you turned this on here.")
    if reason == telemetry.CONSENT_ENV_OPT_IN:
        return _(
            "Sending even though DO_NOT_TRACK is set, because PHOTO_TAGGER_TELEMETRY=1 is set.",
        )
    return ""


@dataclass(frozen=True, slots=True)
class GuiConfigValues:
    """
    The GUI settings that persist to the config file.

    The API key is deliberately not a field: it must never be written to disk.

    The trailing fields have defaults because they were added later and every caller that only
    cares about the provider and the save options should stay readable.
    """

    provider_name: str
    model_name: str
    api_base_url: str | None
    extensions: str
    recursive: bool
    write_title: bool
    write_description: bool
    write_keywords: bool
    preserve_keywords: bool
    sidecar_mode: SidecarMode
    backup_xmp: bool
    telemetry_enabled: bool
    vocabulary: Path | None = None
    vocabulary_strict: bool = False
    session_gap_minutes: float = 0.0
    undo_log: bool = True
    preserve_title: bool = False
    preserve_description: bool = False
    notify: bool = False
    notify_sound: bool = False
    repair_language: bool = False
    # Settings > Copyright. Blank strings and a None flag mean "write nothing", which removes the
    # keys from the file rather than pinning empty ones.
    write_copyright: bool = True
    artist: str = ""
    copyright_notice: str = ""
    rights: str = ""
    marked: bool | None = None
    # Window-only, under [gui]: the CLI writes every photo it generates, so it has no flag for it.
    save_after_generating: bool = False
    # Settings > Pacing, under [pacing]. The built-ins pace nothing, so a config the GUI writes
    # without touching the dialog leaves every run working flat out.
    duty: int = DEFAULT_DUTY_PERCENT
    battery_floor: int = DEFAULT_BATTERY_FLOOR
    battery_resume: int = DEFAULT_BATTERY_RESUME
    work_minutes: float = DEFAULT_WORK_MINUTES
    max_pause_minutes: float = DEFAULT_MAX_PAUSE_MINUTES


def _copyright_lines(values: GuiConfigValues) -> list[str]:
    """Render the ``[copyright]`` table, or nothing at all when no field is filled in."""
    entries = [
        (key, value)
        for key, value in (
            ("artist", values.artist),
            ("copyright_notice", values.copyright_notice),
            ("rights", values.rights),
        )
        if value
    ]
    if not entries and values.marked is None:
        return []
    lines = ["", "[copyright]"]
    lines += [f"{key} = {_toml_str(value)}" for key, value in entries]
    if values.marked is not None:
        lines.append(f"marked = {_toml_bool(values.marked)}")
    return lines


def config_toml_text(values: GuiConfigValues) -> str:
    """
    Render *values* as a fresh TOML config file the CLI and GUI both load.

    Key names mirror the config tables ``load_defaults`` reads ([provider], [output], [artifacts],
    [telemetry], plus the top-level extensions/recursive). Used only when no config file exists yet;
    an existing file goes through :func:`merged_config_text` instead so nothing is lost.
    """
    lines = [
        "# Written by the Photo Tagger GUI (Settings > Save Settings as Defaults).",
        f"extensions = {_toml_str(values.extensions)}",
        f"recursive = {_toml_bool(values.recursive)}",
        "",
        "[provider]",
        f"provider_name = {_toml_str(values.provider_name)}",
        f"model_name = {_toml_str(values.model_name)}",
    ]
    if values.api_base_url:
        lines.append(f"api_base_url = {_toml_str(values.api_base_url)}")
    lines += [
        "",
        "[output]",
        f"write_title = {_toml_bool(values.write_title)}",
        f"write_description = {_toml_bool(values.write_description)}",
        f"write_keywords = {_toml_bool(values.write_keywords)}",
        f"write_copyright = {_toml_bool(values.write_copyright)}",
        f"preserve_keywords = {_toml_bool(values.preserve_keywords)}",
        f"preserve_title = {_toml_bool(values.preserve_title)}",
        f"preserve_description = {_toml_bool(values.preserve_description)}",
        f"sidecar_mode = {_toml_str(values.sidecar_mode)}",
        f"backup_xmp = {_toml_bool(values.backup_xmp)}",
    ]
    if values.vocabulary is not None:
        lines.append(f"vocabulary = {_toml_str(str(values.vocabulary))}")
    lines += [
        f"vocabulary_strict = {_toml_bool(values.vocabulary_strict)}",
        f"session_gap_minutes = {values.session_gap_minutes}",
    ]
    lines += _copyright_lines(values)
    lines += [
        "",
        "[inference]",
        f"repair_language = {_toml_bool(values.repair_language)}",
        "",
        "[display]",
        f"notify = {_toml_bool(values.notify)}",
        f"notify_sound = {_toml_bool(values.notify_sound)}",
        "",
        "[pacing]",
        f"duty = {values.duty}",
        f"battery_floor = {values.battery_floor}",
        f"battery_resume = {values.battery_resume}",
        f"work_minutes = {values.work_minutes}",
        f"max_pause_minutes = {values.max_pause_minutes}",
        "",
        "[artifacts]",
        f"undo_log = {_toml_bool(values.undo_log)}",
        "",
        "[gui]",
        f"save_after_generating = {_toml_bool(values.save_after_generating)}",
        "",
        "[telemetry]",
        f"enabled = {_toml_bool(values.telemetry_enabled)}",
        "",
    ]
    return "\n".join(lines)


def _parse_config(existing_text: str) -> TOMLDocument:
    """
    Parse a config file the GUI is about to update, or refuse the update.

    Loading a config for *reading* tolerates a broken file (it is logged and ignored), but rewriting
    one cannot: preserving the keys, comments, and ordering the GUI does not manage means starting
    from a document, and a file with a syntax error yields none. Without this the tomlkit ParseError
    escaped a Qt slot and crashed the window on a save meant to be forgiving.
    """
    try:
        return tomlkit.parse(existing_text)
    except TOMLKitError as exc:
        msg = f"The config file is not valid TOML, so it was left unchanged: {exc}"
        raise ConfigFileError(msg) from exc


def _config_table(document: TOMLDocument, name: str) -> MutableMapping[str, object]:
    """
    Return *document*'s ``[name]`` table, creating it when absent.

    A file where the key exists but is not a table (``provider = "lmstudio"``) is refused rather
    than overwritten: assigning into it raised a bare TypeError out of the save slot, and silently
    replacing it would throw away whatever the user meant by it.
    """
    existing = document.get(name)
    if existing is None:
        table = tomlkit.table()
        document[name] = table
        return table
    if not isinstance(existing, MutableMapping):
        msg = f"The config file's '{name}' entry is not a [{name}] table, so it was left unchanged."
        raise ConfigFileError(msg)
    return existing


def _merge_copyright(document: TOMLDocument, values: GuiConfigValues) -> None:
    """
    Write the ``[copyright]`` table, dropping the keys the dialog left empty.

    An emptied field has to remove its key, not write ``""``: the CLI reads the same table, and a
    blank notice there would look like a setting rather than the absence of one. A table left with
    nothing in it goes too.
    """
    if "copyright" not in document and not any(
        (values.artist, values.copyright_notice, values.rights, values.marked is not None),
    ):
        return
    table = _config_table(document, "copyright")
    for key, value in (
        ("artist", values.artist),
        ("copyright_notice", values.copyright_notice),
        ("rights", values.rights),
    ):
        if value:
            table[key] = value
        else:
            table.pop(key, None)
    if values.marked is None:
        table.pop("marked", None)
    else:
        table["marked"] = values.marked
    if not table:
        document.pop("copyright", None)


def merged_config_text(existing_text: str, values: GuiConfigValues) -> str:
    """
    Update *existing_text* (a TOML config file) with *values*, preserving everything else.

    tomlkit keeps comments, ordering, and keys the GUI does not manage, so saving from the GUI never
    destroys a hand-written config. Only the GUI-managed keys are set; a blank base URL removes the
    key so the provider default applies again.

    Raises :class:`~photo_tagger.errors.ConfigFileError` when the file cannot be updated in place.
    """
    document = _parse_config(existing_text)
    document["extensions"] = values.extensions
    document["recursive"] = values.recursive

    provider = _config_table(document, "provider")
    provider["provider_name"] = values.provider_name
    provider["model_name"] = values.model_name
    if values.api_base_url:
        provider["api_base_url"] = values.api_base_url
    else:
        provider.pop("api_base_url", None)

    output = _config_table(document, "output")
    output["write_title"] = values.write_title
    output["write_description"] = values.write_description
    output["write_keywords"] = values.write_keywords
    output["write_copyright"] = values.write_copyright
    output["preserve_keywords"] = values.preserve_keywords
    output["preserve_title"] = values.preserve_title
    output["preserve_description"] = values.preserve_description
    output["sidecar_mode"] = values.sidecar_mode
    # The boolean this replaced only knew sidecar-or-not, and the mode wins over it anyway. Drop
    # it so a config the GUI wrote never carries two answers to the same question.
    output.pop("use_sidecar", None)
    output["backup_xmp"] = values.backup_xmp
    output["vocabulary_strict"] = values.vocabulary_strict
    output["session_gap_minutes"] = values.session_gap_minutes
    if values.vocabulary is not None:
        output["vocabulary"] = str(values.vocabulary)
    else:
        # No vocabulary chosen: drop the key rather than write an empty path the CLI would then
        # fail to open on its next run.
        output.pop("vocabulary", None)

    _merge_copyright(document, values)

    inference = _config_table(document, "inference")
    inference["repair_language"] = values.repair_language

    display = _config_table(document, "display")
    display["notify"] = values.notify
    display["notify_sound"] = values.notify_sound

    pacing = _config_table(document, "pacing")
    pacing["duty"] = values.duty
    pacing["battery_floor"] = values.battery_floor
    pacing["battery_resume"] = values.battery_resume
    pacing["work_minutes"] = values.work_minutes
    pacing["max_pause_minutes"] = values.max_pause_minutes

    artifacts = _config_table(document, "artifacts")
    artifacts["undo_log"] = values.undo_log

    window = _config_table(document, "gui")
    window["save_after_generating"] = values.save_after_generating

    telemetry = _config_table(document, "telemetry")
    telemetry["enabled"] = values.telemetry_enabled
    return tomlkit.dumps(document)


def config_text_with_language(existing_text: str, language: str) -> str:
    """
    Set the top-level ``language`` key in a TOML config, preserving everything else.

    Choosing :data:`~photo_tagger.i18n.AUTO` (follow the OS locale, the built-in default) removes
    the key instead of writing it, so a config never pins a language the user did not pick. tomlkit
    keeps comments and ordering intact, like :func:`merged_config_text`.

    Raises :class:`~photo_tagger.errors.ConfigFileError` when the file cannot be updated in place.
    """
    document = _parse_config(existing_text)
    if language == AUTO:
        document.pop("language", None)
    else:
        document["language"] = language
    return tomlkit.dumps(document)


def config_text_with_output_language(existing_text: str, language: str) -> str:
    """
    Set ``[inference] output_language`` in a TOML config, preserving everything else.

    The key the CLI's ``--output-language`` flag reads its default from, so the GUI choice carries
    over to CLI runs too. Choosing the built-in default (English) removes the key instead of pinning
    it, mirroring :func:`config_text_with_language`; an ``[inference]`` table left empty by that
    removal is dropped as well.

    Raises :class:`~photo_tagger.errors.ConfigFileError` when the file cannot be updated in place.
    """
    document = _parse_config(existing_text)
    if language.strip().casefold() == DEFAULT_OUTPUT_LANGUAGE.casefold():
        if "inference" in document:
            inference = _config_table(document, "inference")
            inference.pop("output_language", None)
            if not inference:
                document.pop("inference", None)
    else:
        _config_table(document, "inference")["output_language"] = language.strip()
    return tomlkit.dumps(document)


# How long to wait for the login shell to report its PATH before giving up.
_SHELL_PATH_TIMEOUT = 5.0


def login_shell_path() -> list[str]:
    """
    Return the PATH entries a login shell would set, or ``[]`` if they can't be determined.

    A Finder/Dock launch inherits a minimal PATH, so exiftool installed by any package manager
    (Homebrew, Nix, MacPorts, ...) is missing - those locations are only added by shell startup
    files. Asking the user's own login shell for its PATH recovers exiftool however it was
    installed, instead of hard-coding each manager's bin dir. Best-effort: a missing ``$SHELL``, a
    non-zero exit, or a timeout all yield ``[]``.
    """
    shell = os.environ.get("SHELL")
    if not shell:
        return []
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, trusted $SHELL
            [shell, "-l", "-c", 'printf %s "$PATH"'],
            capture_output=True,
            text=True,
            timeout=_SHELL_PATH_TIMEOUT,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return []
    return [entry for entry in result.stdout.strip().split(os.pathsep) if entry]


def ensure_path_dirs(current_path: str, dirs: Iterable[str]) -> str:
    """
    Prepend any of *dirs* not already in *current_path* to it, order preserved.

    A GUI launched from Finder or the Dock inherits a minimal ``PATH``, so Homebrew's bin
    directories (where exiftool usually lives) are missing and exiftool discovery fails even though
    it works fine from a shell. Prepending them lets a double-clicked app find exiftool the same
    way. Directories that do not exist are harmless on ``PATH``, so no filesystem check is needed.
    """
    entries = current_path.split(os.pathsep) if current_path else []
    additions = [d for d in dirs if d not in entries]
    if not additions:
        return current_path
    return os.pathsep.join([*additions, *entries])
