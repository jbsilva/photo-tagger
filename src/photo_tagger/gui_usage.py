"""
The desktop window's usage tracking for telemetry, kept free of Qt so it is tested normally.

The window reports each run as it ends, its launch, and the whole session when it closes (see
docs/telemetry.md). This module keeps the tallies behind those reports: how long the window was busy
or in use, what each run did, which features were used, and how much of each proposal was changed
before it was saved. Nothing here keeps text or a path for the beacon: an edit is counted, never
sent.
"""

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from photo_tagger import telemetry
from photo_tagger.gui_state import DEFAULT_GUI_EXTENSIONS, FAILED
from photo_tagger.telemetry_facts import endpoint_kind, extension_list, mean, quantile


if TYPE_CHECKING:
    from collections.abc import Iterable

    from photo_tagger.fields import ExtraFields
    from photo_tagger.gui_state import GuiConfigValues, PhotoItem, Proposal


# Each click or key press counts the next minute as time in use.
ENGAGED_GRACE_SECONDS = 60.0

# What starts a GUI run.
TRIGGER_BATCH = "batch"  # Generate Selected, Save Selected
TRIGGER_SINGLE = "single"  # Generate This Photo
TRIGGER_HINT = "hint"  # Enter in the hint box
TRIGGER_RETRY = "retry"  # Retry Failed
TRIGGER_WATCH = "watch"  # a photo the folder watch picked up

JOB_GENERATE = "generate"
JOB_SAVE = "save"

# Retries per photo in a window run: fewer than the CLI's, since a person is watching and can retry
# a failure by hand.
GUI_GENERATE_RETRIES = 2

# The window's own defaults, which differ from the CLI's: a broader list of file types, subfolders
# on, the result cache on, and its own retries. A setting counts as changed against these.
GUI_BASELINE = replace(
    telemetry.Settings(),
    extensions=extension_list(DEFAULT_GUI_EXTENSIONS.split(",")),
    recursive=True,
    cache=True,
    retries=GUI_GENERATE_RETRIES,
)

type _Span = tuple[float, float]


def _merged(spans: Iterable[_Span]) -> list[_Span]:
    """Merge overlapping spans into a sorted list of disjoint ones."""
    merged: list[_Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _length(spans: Iterable[_Span]) -> float:
    return sum(end - start for start, end in spans)


def _overlap(first: list[_Span], second: list[_Span]) -> float:
    """Return how long two sorted lists of disjoint spans share."""
    total = 0.0
    i = j = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        total += max(0.0, end - start)
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return total


def _clipped(spans: Iterable[_Span], now: float) -> list[_Span]:
    """Cut spans off at *now*: nothing has happened past it yet."""
    return [(start, min(end, now)) for start, end in spans if start < now]


@dataclass(slots=True, frozen=True)
class TimeSplit:
    """How a session's time divides: busy, engaged, idle, and busy with nobody there."""

    session: float
    busy: float
    engaged: float
    idle: float
    unattended: float


class ActivityClock:
    """
    Split a window's open time into busy (a run in flight), engaged (input), and idle.

    Each touch counts the next :data:`ENGAGED_GRACE_SECONDS` as engaged, so half a minute spent
    reading a photo between two clicks still counts as use, and a window left alone overnight does
    not. Times are monotonic clock readings.
    """

    def __init__(self, started: float) -> None:
        """Start the clock at *started*, the moment the window opened."""
        self.started = started
        self._engaged: list[_Span] = []
        self._busy: list[_Span] = []
        self._busy_since: float | None = None

    def touch(self, now: float) -> None:
        """Record input at *now*."""
        end = now + ENGAGED_GRACE_SECONDS
        if self._engaged and now <= self._engaged[-1][1]:
            self._engaged[-1] = (self._engaged[-1][0], max(self._engaged[-1][1], end))
        else:
            self._engaged.append((now, end))

    def set_busy(self, now: float, *, busy: bool) -> None:
        """Mark a run starting (*busy*) or ending at *now*."""
        if busy and self._busy_since is None:
            self._busy_since = now
        elif not busy and self._busy_since is not None:
            self._busy.append((self._busy_since, now))
            self._busy_since = None

    def engaged_between(self, start: float, end: float) -> float:
        """Return the engaged time inside ``[start, end]``: how much of a run someone watched."""
        return _overlap(_merged(_clipped(self._engaged, end)), [(start, end)])

    def split(self, now: float) -> TimeSplit:
        """Divide the time from opening to *now*."""
        busy_spans = [*self._busy]
        if self._busy_since is not None:
            busy_spans.append((self._busy_since, now))
        busy = _merged(_clipped(busy_spans, now))
        engaged = _merged(_clipped(self._engaged, now))
        session = max(0.0, now - self.started)
        busy_total = _length(busy)
        active = _length(_merged([*busy, *engaged]))
        return TimeSplit(
            session=session,
            busy=busy_total,
            engaged=_length(engaged),
            idle=max(0.0, session - active),
            unattended=max(0.0, busy_total - _overlap(busy, engaged)),
        )


@dataclass(slots=True, frozen=True)
class EditCount:
    """How much of one proposal was changed by hand before a save."""

    title: bool
    description: bool
    keywords_added: int
    keywords_removed: int

    @property
    def keywords(self) -> bool:
        """Report whether any keyword was added or removed."""
        return bool(self.keywords_added or self.keywords_removed)

    @property
    def any(self) -> bool:
        """Report whether anything at all was changed."""
        return self.title or self.description or self.keywords


def count_edits(item: PhotoItem) -> EditCount:
    """
    Compare what *item* is about to be saved with against what the model proposed.

    Keywords are compared ignoring case, the way the keyword field itself removes repeats. A photo
    without a proposal has nothing to compare against, so it counts as unedited.
    """
    proposed = {keyword.casefold() for keyword in item.proposed_keywords}
    current = {keyword.casefold() for keyword in item.keywords}
    return EditCount(
        title=item.title.strip() != item.proposed_title.strip(),
        description=item.description.strip() != item.proposed_description.strip(),
        keywords_added=len(current - proposed),
        keywords_removed=len(proposed - current),
    )


@dataclass(slots=True)
class ReviewTally:
    """What happened between generating photos and saving them, summed over the saves."""

    saved: int = 0
    edited: int = 0
    edited_title: int = 0
    edited_description: int = 0
    edited_keywords: int = 0
    keywords_added: int = 0
    keywords_removed: int = 0
    hinted: int = 0
    flagged: int = 0
    from_cache: int = 0
    manual_only: int = 0
    copyright_only: int = 0
    review_seconds: list[float] = field(default_factory=list)

    def add(self, item: PhotoItem, *, now: float, stamp_only: bool = False) -> None:
        """Count one saved photo; *stamp_only* is a save that wrote nothing but the copyright."""
        self.saved += 1
        if stamp_only:
            # A notice stamped on a photo that was never generated: a feature of its own, not the
            # window used as a plain metadata editor.
            self.copyright_only += 1
            return
        if not item.has_proposal:
            # Saved without ever being generated: the window used as a plain metadata editor.
            self.manual_only += 1
            return
        edits = count_edits(item)
        self.edited += int(edits.any)
        self.edited_title += int(edits.title)
        self.edited_description += int(edits.description)
        self.edited_keywords += int(edits.keywords)
        self.keywords_added += edits.keywords_added
        self.keywords_removed += edits.keywords_removed
        self.hinted += int(item.hinted)
        self.flagged += int(bool(item.foreign_terms))
        self.from_cache += int(item.from_cache)
        if item.proposed_at is not None:
            self.review_seconds.append(max(0.0, now - item.proposed_at))

    def stats(self) -> telemetry.ReviewStats:
        """Return the figures a beacon carries."""
        return telemetry.ReviewStats(
            saved=self.saved,
            edited=self.edited,
            edited_title=self.edited_title,
            edited_description=self.edited_description,
            edited_keywords=self.edited_keywords,
            keywords_added=self.keywords_added,
            keywords_removed=self.keywords_removed,
            hinted=self.hinted,
            flagged=self.flagged,
            from_cache=self.from_cache,
            manual_only=self.manual_only,
            copyright_only=self.copyright_only,
            review_seconds_median=quantile(self.review_seconds, 0.5),
            review_seconds_max=max(self.review_seconds, default=0.0),
        )


@dataclass(slots=True, frozen=True)
class RunContext:
    """What a run was started with: the model, the languages, the settings, how they were set."""

    provider: str
    model: str
    output_language: str
    ui_language: str
    settings: telemetry.Settings
    setup: telemetry.Setup


@dataclass(slots=True)
class RunTally:
    """What one GUI run did, gathered as it goes and turned into a beacon when it ends."""

    job: str
    trigger: str
    started: float
    batch_size: int
    context: RunContext
    use_cache: bool = True
    autosave: bool = False
    hinted: int = 0
    regenerated_flagged: int = 0
    initial_failures: int = 0
    cache_hits: int = 0
    language_repairs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_calls: int = 0
    inference_seconds: float = 0.0
    vocabulary_mapped: int = 0
    vocabulary_dropped: set[str] = field(default_factory=set)
    rested_seconds: float = 0.0
    rests_taken: int = 0
    passing: str | None = None  # the extra pass running now, if any
    retry_pass: int = 0
    retry_pass_recovered: int = 0
    mixed_pass: int = 0
    mixed_pass_fixed: int = 0
    harmonized: int = 0
    generate_seconds: float = 0.0
    save_started: float | None = None
    save_seconds: float = 0.0
    saved: int = 0
    save_failures: int = 0
    review: ReviewTally = field(default_factory=ReviewTally)
    failures: dict[str, str] = field(default_factory=dict)  # photo key -> failure kind
    failure_types: dict[str, str] = field(default_factory=dict)  # photo key -> exception type
    photo_seconds: list[float] = field(default_factory=list)
    keywords: list[float] = field(default_factory=list)
    titles: list[float] = field(default_factory=list)
    descriptions: list[float] = field(default_factory=list)

    def pass_started(self, name: str, count: int, *, retry: bool) -> None:
        """Record an extra pass over *count* photos: the retry pass, or the mixed-script one."""
        self.passing = name
        if retry:
            self.retry_pass += count
        else:
            self.mixed_pass += count

    def generated(self, key: str, proposal: Proposal) -> None:
        """Count one photo the model answered, and what the answer was like."""
        if self.passing is not None:
            if key in self.failures:
                self.retry_pass_recovered += 1
            elif not proposal.foreign_terms:
                self.mixed_pass_fixed += 1
        self.failures.pop(key, None)
        self.failure_types.pop(key, None)
        self.cache_hits += int(proposal.from_cache)
        self.language_repairs += int(proposal.repaired)
        self.vocabulary_mapped += proposal.vocabulary_mapped
        self.vocabulary_dropped.update(proposal.vocabulary_dropped)
        self.input_tokens += proposal.input_tokens
        self.output_tokens += proposal.output_tokens
        self.total_tokens += proposal.total_tokens
        if not proposal.from_cache:
            self.inference_calls += 1 + int(proposal.repaired)
            self.inference_seconds += proposal.seconds
            if proposal.seconds > 0:
                self.photo_seconds.append(proposal.seconds)
        self.keywords.append(len(proposal.keywords))
        self.titles.append(len(proposal.title))
        self.descriptions.append(len(proposal.description))

    def failed(self, key: str, kind: str, error_type: str = "") -> None:
        """Count one photo that failed; a later success in a pass takes it back."""
        if self.passing is None:
            self.initial_failures += 1
        self.failures[key] = kind
        if error_type:
            self.failure_types[key] = error_type

    def generation_ended(self, now: float, *, rested: float, rests: int) -> None:
        """Close a generation pass, adding the time and the pacing rests it took."""
        self.generate_seconds = max(0.0, now - self.started - self.save_seconds)
        self.rested_seconds += rested
        self.rests_taken += rests

    def save_begun(self, now: float) -> None:
        """Note when the save of this run started."""
        self.save_started = now

    def photo_saved(self, item: PhotoItem, *, ok: bool, now: float) -> None:
        """Count one photo this run's save wrote, or failed to."""
        if not ok:
            self.save_failures += 1
            return
        self.saved += 1
        self.review.add(item, now=now)

    def save_ended(self, now: float) -> None:
        """Close the save, adding its time."""
        if self.save_started is not None:
            self.save_seconds += max(0.0, now - self.save_started)
            self.save_started = None


@dataclass(slots=True, frozen=True)
class RunEnd:
    """The facts only the window knows when a run ends: how it ended, and what it left behind."""

    now: float
    outcome: str | None  # None: work it out from the counts
    attended: float
    succeeded: int  # photos of the batch with a proposal (generate) or written (save)
    file_types: str
    flagged: int  # photos of the batch whose text still mixes writing systems
    # Photos of the batch that ended failed, a failed chained save included (generate only).
    failed: int = 0


def run_info(tally: RunTally, end: RunEnd) -> telemetry.RunInfo:
    """Turn a finished run into its beacon."""
    failed = tally.save_failures if tally.job == JOB_SAVE else max(len(tally.failures), end.failed)
    context = tally.context
    return telemetry.RunInfo(
        interface="gui",
        command="gui",
        job=tally.job,
        trigger=tally.trigger,
        provider=context.provider,
        model=context.model,
        batch_size=tally.batch_size,
        duration_seconds=max(0.0, end.now - tally.started),
        output_language=context.output_language,
        ui_language=context.ui_language,
        file_types=end.file_types,
        outcome=end.outcome or telemetry.outcome_for(tally.batch_size, failed),
        success_count=end.succeeded,
        failure_count=failed,
        initial_failures=tally.initial_failures,
        cache_hits=tally.cache_hits,
        retry_successes=tally.retry_pass_recovered,
        workers=1,
        input_tokens=tally.input_tokens,
        output_tokens=tally.output_tokens,
        total_tokens=tally.total_tokens,
        inference_calls=tally.inference_calls,
        inference_seconds=tally.inference_seconds,
        failure_kinds=telemetry.failure_kinds_summary(Counter(tally.failures.values())),
        failure_types=telemetry.failure_kinds_summary(Counter(tally.failure_types.values())),
        language_repairs=tally.language_repairs,
        flagged=end.flagged,
        vocabulary_mapped=tally.vocabulary_mapped,
        vocabulary_dropped=len(tally.vocabulary_dropped),
        rested_seconds=tally.rested_seconds,
        rests_taken=tally.rests_taken,
        photo_seconds_median=quantile(tally.photo_seconds, 0.5),
        photo_seconds_p90=quantile(tally.photo_seconds, 0.9),
        keywords_mean=mean(tally.keywords),
        title_chars_mean=mean(tally.titles),
        description_chars_mean=mean(tally.descriptions),
        settings=context.settings,
        setup=context.setup,
        gui=telemetry.GuiJob(
            use_cache=tally.use_cache,
            autosave=tally.autosave,
            hinted=tally.hinted,
            regenerated_flagged=tally.regenerated_flagged,
            retry_pass=tally.retry_pass,
            retry_pass_recovered=tally.retry_pass_recovered,
            mixed_pass=tally.mixed_pass,
            mixed_pass_fixed=tally.mixed_pass_fixed,
            harmonized=tally.harmonized,
            saved=tally.saved,
            save_failures=tally.save_failures,
            generate_seconds=tally.generate_seconds,
            save_seconds=tally.save_seconds,
            attended_seconds=end.attended,
        ),
        review=tally.review.stats() if tally.review.saved else None,
    )


def batch_outcome(items: Iterable[PhotoItem]) -> tuple[int, int]:
    """Count a batch's photos that ended with a proposal, and the ones that ended failed."""
    succeeded = failed = 0
    for item in items:
        if item.status == FAILED:
            failed += 1
        elif item.has_proposal:
            succeeded += 1
    return succeeded, failed


class GuiUsage:
    """
    One window's usage, from opening to closing: its runs, its saves, and its features.

    The Qt shell calls into this at each step and reads the beacons back out; this class never sends
    anything itself.
    """

    def __init__(self, now: float) -> None:
        """Start tracking a window that opened at *now*."""
        self.clock = ActivityClock(now)
        self.features: Counter[str] = Counter()
        self.review = ReviewTally()
        self.run: RunTally | None = None
        self.photos_added = 0
        self.runs = 0
        self.save_runs = 0
        self.single_saves = 0
        self._first_run: float | None = None
        self._generated: set[str] = set()
        self._saved: set[str] = set()

    def count(self, feature: str, now: float) -> None:
        """Count one use of *feature*; using a feature is input, so it touches the clock too."""
        self.features[feature] += 1
        self.clock.touch(now)

    def start_run(self, tally: RunTally) -> RunTally:
        """Open a run; the window is busy until :meth:`finish_run`."""
        self.run = tally
        self.clock.set_busy(tally.started, busy=True)
        if tally.job == JOB_GENERATE:
            self.runs += 1
        else:
            self.save_runs += 1
        if self._first_run is None:
            self._first_run = tally.started
        return tally

    def generated(self, key: str, proposal: Proposal) -> None:
        """Count one proposal, in the run in flight if there is one."""
        self._generated.add(key)
        if self.run is not None:
            self.run.generated(key, proposal)

    def saved(self, item: PhotoItem, *, ok: bool, now: float, stamp_only: bool = False) -> None:
        """Count one photo a save wrote (or failed to), in the run and in the session."""
        if self.run is not None:
            self.run.photo_saved(item, ok=ok, now=now)
        if ok:
            self._saved.add(str(item.path))
            self.review.add(item, now=now, stamp_only=stamp_only)

    def saved_single(self, item: PhotoItem, *, ok: bool, now: float) -> None:
        """Count a photo saved on its own from the detail pane: part of the session, not a run."""
        self.single_saves += 1
        self.saved(item, ok=ok, now=now)

    def finish_run(self, end: RunEnd) -> telemetry.RunInfo | None:
        """Close the run in flight and return its beacon, or None when no run was open."""
        if self.run is None:
            return None
        tally, self.run = self.run, None
        self.clock.set_busy(end.now, busy=False)
        return run_info(tally, end)

    def session_info(  # noqa: PLR0913 - the session plus what only the window knows about it.
        self,
        now: float,
        *,
        context: RunContext,
        unsaved: int,
        closed_with_unsaved: bool,
        view_mode: str,
        display: telemetry.Display,
    ) -> telemetry.SessionInfo:
        """Return the session beacon for a window closing at *now*."""
        split = self.clock.split(now)
        first = -1.0 if self._first_run is None else self._first_run - self.clock.started
        return telemetry.SessionInfo(
            interface="gui",
            command="gui",
            session_seconds=split.session,
            provider=context.provider,
            model=context.model,
            output_language=context.output_language,
            ui_language=context.ui_language,
            busy_seconds=split.busy,
            engaged_seconds=split.engaged,
            idle_seconds=split.idle,
            unattended_seconds=split.unattended,
            runs=self.runs,
            save_runs=self.save_runs,
            single_saves=self.single_saves,
            first_run_seconds=first,
            photos_added=self.photos_added,
            photos_generated=len(self._generated),
            photos_saved=len(self._saved),
            photos_unsaved=unsaved,
            closed_with_unsaved=closed_with_unsaved,
            view_mode=view_mode,
            settings=context.settings,
            setup=context.setup,
            review=self.review.stats() if self.review.saved else None,
            display=display,
            features=dict(self.features),
        )


def _marked_label(*, marked: bool | None) -> str:
    """Name the XMP Marked choice the way telemetry reports it."""
    if marked is None:
        return "unset"
    return "copyrighted" if marked else "public-domain"


@dataclass(slots=True, frozen=True)
class WindowFacts:
    """The settings the window holds outside what it saves to the config file."""

    url: str  # the URL in effect, typed or the provider default; only its category is sent
    custom_url: bool
    api_key: bool
    cache: bool
    jpeg_dimensions: int
    vocabulary_terms: int
    copyright: ExtraFields | None
    watch_interval: float
    watch_settle: float
    sidecar_overrides: int = 0  # from the config file: the window has no control for them


def window_settings(values: GuiConfigValues, facts: WindowFacts) -> telemetry.Settings:
    """Reduce the window's settings to the bounded values telemetry reports."""
    copyright_fields = facts.copyright
    return replace(
        GUI_BASELINE,
        endpoint=endpoint_kind(facts.url),
        secure=facts.url.lower().startswith("https://"),
        custom_url=facts.custom_url,
        api_key=facts.api_key,
        jpeg_dimensions=facts.jpeg_dimensions,
        repair_language=values.repair_language,
        cache=facts.cache,
        vocabulary=values.vocabulary is not None,
        vocabulary_terms=facts.vocabulary_terms,
        vocabulary_strict=values.vocabulary_strict,
        session_gap_minutes=values.session_gap_minutes,
        sidecar_mode=values.sidecar_mode,
        sidecar_overrides=facts.sidecar_overrides,
        write_title=values.write_title,
        write_description=values.write_description,
        write_keywords=values.write_keywords,
        write_copyright=values.write_copyright,
        overwrite_title=not values.preserve_title,
        overwrite_description=not values.preserve_description,
        overwrite_keywords=not values.preserve_keywords,
        backup=values.backup_xmp,
        undo_log=values.undo_log,
        artist=bool(copyright_fields and copyright_fields.artist),
        copyright_notice=bool(copyright_fields and copyright_fields.notice),
        rights=bool(copyright_fields and copyright_fields.rights),
        marked=_marked_label(marked=copyright_fields.marked if copyright_fields else None),
        set_tags=len(copyright_fields.tags) if copyright_fields else 0,
        paced=values.duty < telemetry.Settings().duty or values.battery_floor > 0,
        duty=values.duty,
        battery_floor=values.battery_floor,
        battery_resume=values.battery_resume,
        work_minutes=values.work_minutes,
        max_pause_minutes=values.max_pause_minutes,
        notify=values.notify,
        notify_sound=values.notify_sound,
        extensions=extension_list(values.extensions.split(",")),
        recursive=values.recursive,
        watch_interval=facts.watch_interval,
        watch_settle=facts.watch_settle,
        autosave=values.save_after_generating,
    )
