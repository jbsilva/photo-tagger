"""What a save writes for each photo, and how its result folds back into the list."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from photo_tagger.config import (
    DEFAULT_SIDECAR_MODE,
    FileTypeTarget,
    SidecarMode,
    SidecarOverrides,
)
from photo_tagger.csv_report import ReportRow
from photo_tagger.fields import (
    TAG_EXIF_COPYRIGHT,
    TAG_IPTC_COPYRIGHT_NOTICE,
    TAG_XMP_RIGHTS,
    ExtraFields,
)
from photo_tagger.gui_state.items import (
    FAILED,
    SAVED,
    PhotoItem,
)
from photo_tagger.gui_state.keyword_view import (
    keywords_to_save,
)
from photo_tagger.i18n import _, gettext_noop
from photo_tagger.metadata import (
    FIELD_COPYRIGHT,
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    FieldSources,
)
from photo_tagger.models import KeywordSet
from photo_tagger.prompt_context import select_camera_fields, select_location
from photo_tagger.write_plan import WriteTarget, parse_sidecar_overrides, plan_writes


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


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
    # Sidecar, embedded, or one per file type: resolved per photo by metadata.plan_writes.
    sidecar_mode: SidecarMode = DEFAULT_SIDECAR_MODE
    # The config file's sidecar_overrides (dng = photo), which win over the mode per extension.
    # The window has no control for them: they are set once, for a catalog's rules.
    sidecar_overrides: SidecarOverrides = field(default_factory=dict)
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
    def writes_notice(self) -> bool:
        """Whether a save puts a copyright notice on the file, not only a name or a flag."""
        fields = self.extra_fields
        return fields is not None and bool(fields.notice or fields.rights)

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
    # The notice itself is resolved at write time (it needs the capture year). This only says
    # that one goes on, so the No copyright filter lets go of the photo without a rescan.
    copyright: bool = False

    @property
    def fields(self) -> set[str]:
        """Which scanned fields this job puts on the file, for the Tagged column and its filters."""
        written = fields_written(self.title, self.description, self.keywords)
        if self.copyright:
            written.add(FIELD_COPYRIGHT)
        return written


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
        copyright=options.writes_notice,
    )


def reports_status(item: PhotoItem, job: SaveJob) -> bool:
    """
    Say whether saving *item* with *job* is news for its status word.

    It is for a photo with a proposal, and for text written by hand (Save This Photo on a photo that
    was never generated). A copyright stamp alone is not: see :func:`apply_write_result`.
    """
    return item.has_proposal or bool(fields_written(job.title, job.description, job.keywords))


def with_notice_written(job: SaveJob, extra: Mapping[str, str]) -> SaveJob:
    """
    Return *job* saying whether its photo got a notice, now that its tags are resolved.

    The run decides that a notice goes on, but a template chain with no literal finds nothing on
    some photos and writes none there. Such a photo keeps its place in the No copyright filter. A
    notice the photo kept was already known, and the Tagged state only ever grows.
    """
    notices = (TAG_EXIF_COPYRIGHT, TAG_XMP_RIGHTS, TAG_IPTC_COPYRIGHT_NOTICE)
    return replace(job, copyright=job.copyright and any(extra.get(tag) for tag in notices))


def apply_write_result(item: PhotoItem, job: SaveJob, *, ok: bool) -> None:
    """
    Fold one finished write into *item*: its status, and what the Tagged column knows.

    A write that put only a copyright on the file says nothing about the photo's generation, so its
    status stays: a stamped photo that was never generated is not FAILED (Retry Failed would then
    run the model on it), and a failed generation is not hidden behind SAVED. Its outcome still
    counts in the save's tally. A photo the scan has not read yet keeps its unknown fields, so the
    scan still reads it rather than trusting the few fields this write happened to touch.
    """
    if reports_status(item, job):
        item.status = SAVED if ok else FAILED
    if ok and item.known_fields is not None:
        item.known_fields = item.known_fields | job.fields


def photos_to_save(items: Iterable[PhotoItem], options: SaveOptions) -> list[PhotoItem]:
    """
    Pick the photos Save Selected writes: the checked ones that have a proposal.

    With a copyright set, the checked photos that were never generated come too, so the window can
    stamp a notice on photos that are already tagged without generating them again.
    """
    return [
        item for item in items if item.selected and (item.has_proposal or options.writes_copyright)
    ]


def save_options_for(item: PhotoItem, options: SaveOptions) -> SaveOptions:
    """
    Narrow *options* to the copyright block for a photo that was never generated.

    Its text fields hold at most what the file already carries, so writing them back would only
    rewrite the file's own values. Save This Photo still writes a hand edit.
    """
    if item.has_proposal:
        return options
    return replace(  # NOSONAR S5886 - replace() returns the type of options
        options,
        write_title=False,
        write_description=False,
        write_keywords=False,
    )


def resolve_extra_tags(
    options: SaveOptions,
    path: Path,
    sources: Mapping[Path, FieldSources],
) -> dict[str, str]:
    """
    Render one photo's copyright tags, empty when the window writes none.

    *sources* is a batch read of the tags the templates need (see
    :func:`~photo_tagger.metadata.read_field_sources`), so a save pays one ExifTool call for the
    whole batch. A photo missing from it reads as one with nothing on it. Preserve looks only at
    the files this save writes.
    """
    fields = options.extra_fields
    if fields is None or fields.is_empty:
        return {}
    found = sources.get(path, FieldSources())
    return fields.resolve(
        found.values,
        existing=found.written(save_targets(options, path)),
        photo=str(path),
    )


def save_targets(options: SaveOptions, path: Path) -> list[WriteTarget]:
    """List the files a save with *options* writes for *path*."""
    return plan_writes(path, options.sidecar_mode, options.sidecar_overrides)


def writes_nothing(job: SaveJob, extra: Mapping[str, str]) -> bool:
    """
    Whether a save has nothing at all to put on the photo.

    That happens when the copyright is the only thing to write and the photo already has it. The
    writer refuses an empty write, but here it is not a failure: the photo is as asked.
    """
    return job.keywords.is_empty() and job.title is None and job.description is None and not extra


def load_sidecar_overrides(
    entries: Iterable[str] | None,
) -> tuple[dict[str, FileTypeTarget], str]:
    """
    Parse the config file's ``sidecar_overrides``: the overrides, or the message refusing them.

    The CLI refuses a malformed entry at parse time, but the window reads the config file without
    that step. It must still open, so it gets ``({}, message)`` to log, and every save follows the
    Write To mode alone.
    """
    try:
        return parse_sidecar_overrides(entries), ""
    except ValueError as exc:
        return {}, str(exc)


# How the Save tooltip names each override target. The same words as the Write To summaries in the
# Qt shell, so they share one translation.
_OVERRIDE_SUMMARIES: dict[FileTypeTarget, str] = {
    "photo": gettext_noop("into the image file"),
    "sidecar": gettext_noop("to an XMP sidecar"),
    "both": gettext_noop("into the image file and an XMP sidecar"),
}


def sidecar_overrides_summary(overrides: SidecarOverrides) -> str:
    """
    Describe the per-type overrides, or "" when there are none.

    A save writes somewhere else for these types than the Write To menu says, so the Save tooltip
    and that menu both name them.
    """
    if not overrides:
        return ""
    parts = [
        _("{type} {target}").format(
            type=suffix.lstrip(".").upper(),
            target=_(_OVERRIDE_SUMMARIES[target]),
        )
        for suffix, target in sorted(overrides.items())
    ]
    return ", ".join(parts)


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
