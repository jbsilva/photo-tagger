"""The state of one photo in the window, and the proposal a generation hands it."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from photo_tagger.gui_state.keyword_view import (
    normalize_keyword_lines,
)
from photo_tagger.metadata import (
    CaptionValue,
    FieldSources,
)
from photo_tagger.models import KeywordSet
from photo_tagger.script_guard import find_foreign_script


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


# Per-photo status values, shown as an icon/word in the tree.
PENDING = "pending"  # added, not yet generated


WORKING = "working"  # generation in flight


READY = "ready"  # a proposal is available to review


SAVED = "saved"  # written to the file


FAILED = "failed"  # generation or save failed


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
    existing_copyright: str | None = None
    # The tags a copyright template reads (the capture date, the name and notice already there),
    # read with the existing metadata. They fill the notice the detail pane says a save writes,
    # before the photo is ever generated.
    field_sources: FieldSources = field(default_factory=FieldSources)
    # Apart from ``loaded``: a proposal fills the existing metadata but not these tags, so a photo
    # generated before it was opened still needs them read.
    field_sources_read: bool = False
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
    existing_copyright: str | None = None
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
    # The first answer came from the cache, even when a repair then made a call of its own.
    replayed: bool = False


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
    item.existing_copyright = proposal.existing_copyright
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


def recheck_foreign_terms(item: PhotoItem) -> bool:
    """
    Run the script check again on a flagged photo's text as it stands now, and say if that changed.

    The flag comes with the proposal, but a hand edit or harmonization can fix the words it named.
    Left on, it would keep the photo in the Mixed writing systems filter and in the pass that asks
    the model again, which would throw the fix away. Only a flag is checked again: text a person
    typed is theirs, so the check never raises a new flag on it.
    """
    if not item.foreign_terms:
        return False
    finding = find_foreign_script([item.title, item.description, *item.keywords])
    terms = list(finding.terms) if finding is not None else []
    if terms == item.foreign_terms:
        return False
    item.foreign_terms = terms
    return True
