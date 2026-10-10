# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""
The background jobs the window runs: one QObject per job, each moved onto its own QThread.

A worker talks to the window only through Qt signals and never touches a widget, and the pipeline
code it calls is not Qt-aware. Generation, saving, grid thumbnails, the metadata scan, shoot
harmonization, the vocabulary build, and the folder watch each get their own, so one never waits on
another.
"""

from dataclasses import replace
from itertools import islice
from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import QObject, Signal

from photo_tagger.ai import analyze_image_with_ai, create_agent
from photo_tagger.cache import (
    InferenceCache,
    content_cache_key,
    open_cache,
    safe_cache_get,
    safe_cache_put,
)
from photo_tagger.config import DEFAULT_USER_PROMPT, build_repair_prompt
from photo_tagger.errors import PhotoTaggerError
from photo_tagger.gui_state import (
    GenerateOptions,
    HarmonizeResult,
    Proposal,
    SaveJob,
    SaveOptions,
    WatchSettings,
    apply_vocabulary,
    gui_cache_namespace,
    harmonize_sessions,
    pacing_status_text,
    resolve_extra_tags,
    writes_nothing,
)
from photo_tagger.gui_usage import GUI_GENERATE_RETRIES
from photo_tagger.i18n import _, ngettext
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.metadata import (
    build_contextual_prompt,
    iter_field_presence,
    managed_helper,
    prompt_with_hint,
    read_caption,
    read_field_sources,
    read_image_context,
    write_metadata_everywhere,
)
from photo_tagger.pacing import Pacer, PacingPlan, build_pacer
from photo_tagger.pipeline import classify_failure, failure_type
from photo_tagger.script_guard import find_in_result
from photo_tagger.vocabulary import prompt_with_vocabulary
from photo_tagger.vocabulary_build import (
    KeywordCensus,
    TrimRules,
    census_from_export,
    census_from_photos,
    render_drop_report,
    render_vocabulary,
    trim,
    vocabulary_header,
)
from photo_tagger.vocabulary_organize import organize
from photo_tagger.watch import watch_batches


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from exiftool import ExifToolHelper
    from pydantic_ai import BinaryContent

    from photo_tagger.metadata import FieldSources
    from photo_tagger.models import InferenceResult
    from photo_tagger.providers import ProviderName
    from photo_tagger.undo import UndoJournal
    from photo_tagger.vocabulary_build import TrimResult
    from photo_tagger.vocabulary_organize import OrganizeStats


_THUMB_MAX = 200  # pixels for the grid thumbnails the model never sees


class GenerateWorker(QObject):
    """
    Generates AI proposals for a list of photos off the UI thread.

    For each photo it reads the existing metadata, builds the same contextual prompt as the CLI,
    runs the model, and emits a :class:`~photo_tagger.gui_state.Proposal`. All communication with
    the window is via Qt signals; widgets are never touched here.
    """

    started = Signal(int)
    file_done = Signal(object)  # Proposal
    file_failed = Signal(str, str, str, str)  # path, error message, failure kind, exception type
    resting = Signal(str)  # status line for a pacing rest
    finished = Signal()

    def __init__(  # noqa: PLR0913 - the connection inputs plus one options value.
        self,
        provider: ProviderName,
        model: str,
        api_base_url: str | None,
        paths: list[Path],
        *,
        api_key: str | None = None,
        cache_file: Path | None = None,
        options: GenerateOptions | None = None,
    ) -> None:
        """
        Store the run parameters; nothing happens until :meth:`run`.

        *options* says how the run behaves (see :class:`~photo_tagger.gui_state.GenerateOptions`).
        Its *refresh* reads past the cache and stores what comes back, for a run whose whole point
        is that the cached answer is the one it is unhappy with. Its *pacing* is followed per run: a
        window starts with this run and is forgotten when it ends, so a quick Generate on a handful
        of photos never waits.
        """
        super().__init__()
        options = options or GenerateOptions()
        self._provider = provider
        self._model = model
        self._api_base_url = api_base_url
        self._paths = paths
        self._api_key = api_key
        self._cache_file = cache_file
        self._output_language = options.output_language
        self._hints = dict(options.hints)
        self._vocabulary = options.vocabulary
        self._vocabulary_strict = options.vocabulary_strict
        self._repair_language = options.repair_language
        self._refresh = options.refresh
        # The longest edge of the JPEG the model is shown. It keys the cache too, so a window that
        # sends a different size than a CLI run never replays the other one's answers.
        self._jpeg_dimensions = options.jpeg_dimensions
        # The vocabulary listing rides in the user prompt (so the model prefers those terms in the
        # first place), which also puts it in the cache namespace below: swapping vocabularies
        # starts a fresh slice instead of replaying keywords chosen under the old one.
        self._prompt = prompt_with_vocabulary(DEFAULT_USER_PROMPT, options.vocabulary)
        self._pacing = options.pacing
        self._pacer: Pacer | None = None
        self._rest_announced = False
        self._stop = False

    def rest_totals(self) -> tuple[float, int]:
        """Return the seconds this run rested and how many rests it took."""
        if self._pacer is None:
            return 0.0, 0
        return self._pacer.rested_seconds, self._pacer.rests_taken

    def stop(self) -> None:
        """
        Ask the loop to stop before starting the next photo.

        The model call for the photo already in flight runs to completion (there is no way to
        interrupt a blocking request), so cancellation takes effect at the next photo boundary. A
        run that is resting wakes up instead of waiting the rest out.
        """
        self._stop = True
        if self._pacer is not None:
            self._pacer.cancel()

    def _announce_rest(self, reason: str, seconds: float) -> None:
        """Tell the window a rest has started, so it can say so instead of looking wedged."""
        plan = self._pacing or PacingPlan()
        self._rest_announced = True
        self.resting.emit(
            pacing_status_text(
                reason,
                seconds,
                duty_percent=plan.duty_percent,
                resume_percent=plan.resume_percent,
            ),
        )

    def _may_start_next(self) -> bool:
        """
        Let a paced run rest before the next photo; False means the run is stopping.

        The blank line at the end is how the window learns the rest is over and puts its own status
        text back.
        """
        if self._pacer is None:
            return True
        self._rest_announced = False
        proceed = self._pacer.wait_before_next()
        if self._rest_announced:  # NOSONAR - wait_before_next sets this via the on_rest callback.
            self.resting.emit("")
        return proceed

    def run(self) -> None:
        """Build the agent once, then generate a proposal per photo."""
        try:
            agent = create_agent(
                self._provider,
                self._model,
                api_base_url=self._api_base_url,
                api_key=self._api_key,
                retries=GUI_GENERATE_RETRIES,
                output_language=self._output_language,
            )
        except Exception as exc:  # noqa: BLE001
            # Not just PhotoTaggerError: an exception escaping this slot on the worker thread
            # means `finished` never fires, so the window stays in the "running" state (buttons
            # disabled, photos stuck at "working...") for the rest of the session.
            logger.exception("gui_agent_construction_failed", error=str(exc))
            for path in self._paths:
                self.file_failed.emit(str(path), str(exc), classify_failure(exc), failure_type(exc))
            self.finished.emit()
            return

        cache: InferenceCache | None = None
        if self._pacing is not None:
            self._pacer = build_pacer(self._pacing, on_rest=self._announce_rest)
        try:
            cache = self._open_cache()
            self.started.emit(len(self._paths))
            for path in self._paths:
                if self._stop:
                    break
                if not self._may_start_next():
                    break
                try:
                    proposal = self._generate_one(agent, path, cache)
                except Exception as exc:  # noqa: BLE001
                    # One photo's failure must not stop the rest of the batch.
                    logger.exception("gui_generate_failed", file=path.name, error=str(exc))
                    self.file_failed.emit(
                        str(path),
                        str(exc),
                        classify_failure(exc),
                        failure_type(exc),
                    )
                else:
                    self.file_done.emit(proposal)
        finally:
            if cache is not None:
                cache.close()
            # Unconditional, for the reason given above: whatever went wrong, the window has to
            # hear that the run is over, or it stays in the "running" state (buttons disabled,
            # photos stuck at "working...") for the rest of the session. Opening the cache and
            # emitting `started` used to sit outside this, so a failure in either left it stuck.
            self.finished.emit()

    def _open_cache(self) -> InferenceCache | None:
        """Open the result cache for this run, degrading to no cache on any failure."""
        return open_cache(
            self._cache_file,
            namespace=gui_cache_namespace(
                self._model,
                self._output_language,
                user_prompt=self._prompt,
                jpeg_dimensions=self._jpeg_dimensions,
            ),
        )

    def _repair_one(
        self,
        inference: InferenceResult,
        agent: object,
        jpeg: BinaryContent,
        *,
        file_name: str,
    ) -> tuple[InferenceResult, bool]:
        """
        Ask the model once more when its answer mixed writing systems, if the setting is on.

        The GUI's counterpart to the pipeline's ``--repair-language``, kept here rather than shared
        because the two call the agent differently (no sampling knobs here, and no batch usage to
        fold into). A failed repair keeps the original answer: the note over the fields still marks
        it, and a proposal you can edit beats no proposal. So does a repair that comes back mixed
        again. The flag says whether a repair ran.
        """
        if not self._repair_language:
            return inference, False
        finding = find_in_result(inference)
        if finding is None:
            return inference, False
        logger.info("gui_foreign_script_repair_started", file=file_name, terms=finding.terms)
        prompt = build_repair_prompt(
            title=inference.title,
            description=inference.description,
            keywords=inference.keywords,
            terms=finding.terms,
            output_language=self._output_language,
        )
        try:
            repaired = analyze_image_with_ai(image_bytes=jpeg, agent=agent, user_prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - a failed repair must not cost the photo its proposal.
            logger.warning("gui_foreign_script_repair_failed", file=file_name, error=str(exc))
            return inference, False
        # Mixed again: the original stays, as in the CLI, and the proposal counts both calls.
        if find_in_result(repaired) is not None:
            return inference.with_cost_of(repaired), True
        return repaired.with_cost_of(inference), True

    def _repair_cached(
        self,
        cached: InferenceResult,
        agent: object,
        path: Path,
        cache: InferenceCache | None,
        content_key: str | None,
    ) -> tuple[InferenceResult, bool]:
        """
        Repair a replayed answer that mixes scripts, when the setting is on, and store the fix.

        An answer cached before the setting was turned on would otherwise come back mixed on every
        run, as in the CLI pipeline. An earlier run paid for the cached call, so the proposal counts
        only the repair. A repair that does not help leaves the cache as it was.
        """
        if not self._repair_language or find_in_result(cached) is None:
            return cached, False
        jpeg = prepare_image_for_agent(path, max_size=self._jpeg_dimensions)
        unpaid = replace(cached, input_tokens=0, output_tokens=0, total_tokens=0, seconds=0.0)
        result, repaired = self._repair_one(unpaid, agent, jpeg, file_name=path.name)
        if not repaired:
            return cached, False  # the call failed: a plain replay, with what it cost back then
        text = (result.title, result.description, result.keywords)
        helped = text != (cached.title, cached.description, cached.keywords)
        if helped and cache is not None and content_key is not None:
            safe_cache_put(cache, content_key, result, file_name=path.name)
        return result, repaired

    def _generate_one(self, agent: object, path: Path, cache: InferenceCache | None) -> Proposal:
        """Read existing metadata, run the model (or hit the cache), and assemble a proposal."""
        # The content hash rides along in the context read (one exiftool call), so the cache
        # key covers the image stream only: embedding metadata does not invalidate the entry.
        context = read_image_context(path, include_content_hash=cache is not None)
        existing_title, existing_description = read_caption(path)
        gps_info = {"position": context.gps_position} if context.gps_position else {}
        hint = self._hints.get(str(path), "").strip()
        prompt = build_contextual_prompt(
            # The hint goes in before the vocabulary listing, as it does on the CLI, so the two
            # never reorder the prompt between runs and split the cache.
            prompt_with_vocabulary(
                prompt_with_hint(DEFAULT_USER_PROMPT, hint),
                self._vocabulary,
            ),
            context.existing_keywords.subject,
            context.location_tags,
            gps_info,
            camera_info=context.camera_info,
        )
        # Keying and I/O go through the same swallow-and-degrade helpers as the CLI pipeline,
        # so a broken cache entry (or unhashable format) costs a model call, never the photo.
        content_key = content_cache_key(path, context.content_hash) if cache is not None else None
        # The cache key is keyed on image content only, not on the hint text, and the CLI's
        # equivalent --hint folds into the run's cache *namespace* instead (one hint for the whole
        # run). Neither applies here: a GUI hint is per-photo and one-off. Skip both the lookup
        # and the write when hinted, or a hint-biased answer would be stored under the same key a
        # later hint-less regeneration of this photo reads from, silently replaying someone else's
        # correction (or last week's) as if it were the model's generic read.
        skip_cache = cache is None or content_key is None or hint
        # A refresh run reads past the entry but still writes: the mixed-script pass asks again
        # precisely because the stored answer is the unusable one, and replaying it would not be
        # asking at all. What comes back is a plain generic answer for this photo, so it belongs
        # under the same key, and a later run of this folder stops resurrecting the old one.
        cached = (
            None
            if skip_cache or self._refresh
            else safe_cache_get(cache, content_key, file_name=path.name)
        )
        inference = cached
        repaired = False
        if inference is None:
            jpeg = prepare_image_for_agent(path, max_size=self._jpeg_dimensions)
            inference = analyze_image_with_ai(image_bytes=jpeg, agent=agent, user_prompt=prompt)
            # Before the cache write, as in the CLI pipeline, so what a rerun replays is the
            # repaired answer and the extra call is paid once.
            inference, repaired = self._repair_one(inference, agent, jpeg, file_name=path.name)
            if not skip_cache:
                safe_cache_put(cache, content_key, inference, file_name=path.name)
        else:
            logger.info("gui_cache_hit", file=path.name)
            inference, repaired = self._repair_cached(inference, agent, path, cache, content_key)
        # After the cache, not before it: a cached answer is the model's raw output, so a
        # vocabulary chosen (or made stricter) since it was stored still applies to it.
        snapped = apply_vocabulary(
            inference.keywords,
            self._vocabulary,
            strict=self._vocabulary_strict,
        )
        if snapped.mapped or snapped.dropped:
            logger.info(
                "gui_vocabulary_applied",
                file=path.name,
                mapped=snapped.mapped,
                dropped=snapped.dropped,
            )
        # Re-checked on a cache hit too: the check is pure, so the flag never has to be stored.
        finding = find_in_result(inference)
        if finding is not None:
            logger.warning(
                "gui_foreign_script_in_output",
                file=path.name,
                dominant=finding.dominant,
                terms=finding.terms,
            )
        return Proposal(
            path=path,
            existing_title=existing_title,
            existing_description=existing_description,
            existing_keywords=context.existing_keywords,
            title=inference.title,
            description=inference.description,
            keywords=snapped.keywords,
            camera_info=dict(context.camera_info),
            location_tags=dict(context.location_tags),
            gps_position=context.gps_position,
            existing_copyright=context.existing_copyright,
            # A replay that needed a repair paid for a model call, so it is no cache hit.
            from_cache=cached is not None and not repaired,
            replayed=cached is not None,
            input_tokens=inference.input_tokens,
            output_tokens=inference.output_tokens,
            total_tokens=inference.total_tokens,
            seconds=inference.seconds,
            vocabulary_mapped=snapped.mapped,
            vocabulary_dropped=snapped.dropped,
            foreign_terms=list(finding.terms) if finding is not None else [],
            hinted=bool(hint),
            repaired=repaired,
        )


class SaveWorker(QObject):
    """
    Writes the resolved metadata for a list of photos off the UI thread.

    Saving a large batch is minutes of ExifTool work: doing it in the click handler froze the window
    (the OS "busy" cursor and nothing else), which reads as a crash. Here each write reports back by
    signal so the window can tick its progress bar and stay responsive.

    One ExifTool process serves the whole batch, the same way the CLI's batch reads do, instead of
    spawning one per photo.
    """

    file_done = Signal(str, bool)  # path, write succeeded
    finished = Signal()

    def __init__(
        self,
        jobs: list[SaveJob],
        *,
        options: SaveOptions,
        journal: UndoJournal | None = None,
    ) -> None:
        """Store the resolved write jobs and the save options; nothing runs until run."""
        super().__init__()
        self._jobs = jobs
        self._options = options
        self._journal = journal
        self._emitted = 0
        self._stop = False
        # The photos with nothing to write, because they keep the notice they have. Filled before
        # file_done is sent, so the window's slot can tell such a save from a stamp.
        self.kept: set[str] = set()

    def stop(self) -> None:
        """Ask the loop to stop before the next photo; the write in flight finishes."""
        self._stop = True

    def run(self) -> None:
        """Write every job through one shared ExifTool, then report the batch as finished."""
        try:
            with managed_helper(None) as helper:
                self._write_all(helper)
        except Exception as exc:  # noqa: BLE001
            # Per-photo failures are handled in the loop, so this is ExifTool itself failing to
            # start or shut down. Report whatever never got a result as failed, or the window would
            # wait forever for photos that can no longer be written.
            logger.exception("gui_save_batch_failed", error=str(exc))
            for job in self._jobs[self._emitted :]:
                self.file_done.emit(str(job.path), False)  # noqa: FBT003 - Qt signal argument
        self.finished.emit()

    def _field_sources(self, helper: ExifToolHelper) -> dict[Path, FieldSources]:
        """
        Read the tags the copyright templates need for the whole batch, in one call.

        Skipped entirely when no copyright is configured, so the usual save still costs exactly one
        ExifTool write per photo and nothing else.
        """
        fields = self._options.extra_fields
        if fields is None or fields.is_empty:
            return {}
        return read_field_sources([job.path for job in self._jobs], fields.source_tags, et=helper)

    def _write_all(self, helper: ExifToolHelper) -> None:
        """Write each job in turn, one photo's failure never stopping the rest."""
        sources = self._field_sources(helper)
        for job in self._jobs:
            if self._stop:
                return
            try:
                extra = resolve_extra_tags(self._options, job.path, sources)
                if writes_nothing(job, extra):
                    self.kept.add(str(job.path))
                ok, targets = (
                    (True, [])
                    if writes_nothing(job, extra)
                    else write_metadata_everywhere(
                        job.path,
                        job.keywords,
                        description=job.description,
                        title=job.title,
                        extra=extra,
                        backup=self._options.backup,
                        sidecar_mode=self._options.sidecar_mode,
                        sidecar_overrides=self._options.sidecar_overrides,
                        et=helper,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("gui_save_failed", file=job.path.name, error=str(exc))
                ok, targets = False, []
            if self._journal is not None:
                self._journal.record_writes(job.path, targets, backup=self._options.backup)
            self._emitted += 1
            self.file_done.emit(str(job.path), ok)


class ThumbnailWorker(QObject):
    """
    Decodes grid thumbnails off the UI thread.

    Each thumbnail is a small JPEG (the same RAW-aware loader the model uses, at a tiny size). The
    bytes are emitted back to the main thread, which builds the QPixmap there (QPixmap must not be
    created off the GUI thread). :meth:`stop` lets the window abandon a folder's load when the user
    navigates away.
    """

    ready = Signal(str, bytes)  # path, JPEG bytes
    finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        """Store the paths to decode; nothing runs until :meth:`run`."""
        super().__init__()
        self._paths = paths
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to stop before the next thumbnail."""
        self._stop = True

    def run(self) -> None:
        """Decode each thumbnail and emit its bytes, until done or stopped."""
        for path in self._paths:
            if self._stop:
                break
            try:
                content = prepare_image_for_agent(path, max_size=_THUMB_MAX)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gui_thumbnail_failed", file=path.name, error=str(exc))
                continue
            self.ready.emit(str(path), bytes(content.data))
        self.finished.emit()


class MetadataScanWorker(QObject):
    """
    Reads which metadata fields each photo already carries, off the UI thread.

    The read is chunked, so each small batch of photos lands in the tree's Tagged column as it is
    read rather than all at once at the end, and the scan can be stopped between batches. That is
    what makes a few thousand photos on a network share bearable: the read runs for minutes there,
    and it is also what "Uncheck Already Tagged" waits on. Failures degrade to a partial report.
    """

    done = Signal(object)  # dict[str, set[str]]: path -> present fields, one batch at a time
    progress = Signal(int, int)  # photos read so far, photos in this scan
    finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        """Store the paths to scan; nothing runs until :meth:`run`."""
        super().__init__()
        self._paths = paths
        self._dropped: frozenset[Path] = frozenset()
        self._position = 0  # how far _live_paths has walked the list
        self._skipped = 0  # of those, how many had been removed from the list
        self._stop = False

    def stop(self) -> None:
        """
        Ask the scan to stop after the batch it is on.

        One batched exiftool call cannot be interrupted mid-flight, so stopping takes effect at the
        next batch boundary rather than immediately.
        """
        self._stop = True

    def drop(self, paths: Iterable[Path]) -> None:
        """
        Leave these photos unread: they left the list after the scan started.

        The scan works from the list it was handed, so removing a folder of 19000 photos used to
        read all of them anyway. Rebinding the set instead of mutating it is what makes this safe to
        call from the UI thread while the scan thread reads it.
        """
        self._dropped = self._dropped | frozenset(paths)

    def _live_paths(self) -> Iterator[Path]:
        """Yield the paths still in the list, re-reading :meth:`drop` between photos."""
        for position, path in enumerate(self._paths, start=1):
            self._position = position
            if path in self._dropped:
                self._skipped += 1
            else:
                yield path

    def _total(self) -> int:
        """
        Count the photos this scan will read in all, removals taken out.

        Ahead of the cursor the removed photos have not been walked past yet, so they have to be
        counted here; otherwise a folder removed early would keep padding the progress bar for the
        rest of the scan.
        """
        if not self._dropped:
            return len(self._paths)
        ahead = islice(self._paths, self._position, None)
        return len(self._paths) - self._skipped - sum(1 for path in ahead if path in self._dropped)

    def run(self) -> None:
        """Emit each batch of per-path field sets as it is read, until done or stopped."""
        scanned = 0
        self.progress.emit(scanned, len(self._paths))
        try:
            for batch in iter_field_presence(self._live_paths()):
                scanned += batch.scanned
                if batch.presence:
                    self.done.emit({str(p): set(f) for p, f in batch.presence.items()})
                self.progress.emit(scanned, self._total())
                if self._stop:
                    break
        except Exception as exc:  # noqa: BLE001
            # The scan is a convenience; a broken exiftool must not take the window down.
            logger.warning("gui_metadata_scan_failed", error=str(exc))
        self.finished.emit()


class HarmonizeWorker(QObject):
    """
    Groups the finished proposals into shoots and harmonizes each one, off the UI thread.

    Grouping reads every photo's capture time through exiftool, which is a batched read but not an
    instant one, so it does not belong in a click handler. The window applies the returned keywords
    to its items when the result arrives.
    """

    done = Signal(object)  # HarmonizeResult
    finished = Signal()

    def __init__(
        self,
        keywords_by_path: dict[Path, list[str]],
        *,
        gap_minutes: float,
        output_language: str,
    ) -> None:
        """Store the generated keywords to harmonize; nothing runs until :meth:`run`."""
        super().__init__()
        self._keywords = keywords_by_path
        self._gap_minutes = gap_minutes
        self._output_language = output_language

    def run(self) -> None:
        """Harmonize every session and emit the result, degrading to no change on failure."""
        try:
            result = harmonize_sessions(
                self._keywords,
                gap_minutes=self._gap_minutes,
                output_language=self._output_language,
            )
        except Exception as exc:  # noqa: BLE001
            # Harmonization is a refinement of proposals that are already usable; a broken
            # exiftool must not cost the user the run they just waited for.
            logger.exception("gui_harmonize_failed", error=str(exc))
            result = HarmonizeResult()
        self.done.emit(result)
        self.finished.emit()


class VocabularyBuildWorker(QObject):
    """
    Builds a controlled vocabulary out of a library's own keywords, off the UI thread.

    The same three steps as ``photo-tagger vocabulary``: count what the photos (or a Lightroom
    export) carry, trim the count with deterministic rules, and optionally ask the model to fold
    synonyms and give the list a hierarchy. Only the result file is written; no photo is touched.
    """

    progress = Signal(str)
    done = Signal(str, int)  # status message, keywords kept
    failed = Signal(str)
    finished = Signal()

    def __init__(  # noqa: PLR0913  # the source, the rules, the output, and the model are distinct
        self,
        paths: list[Path],
        export_file: Path | None,
        output: Path,
        *,
        rules: TrimRules,
        flat: bool = False,
        report_file: Path | None = None,
        organize_workers: int = 1,
        provider: ProviderName | None = None,
        model: str = "",
        api_base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Store what to read, how to trim it, and where to write it; nothing runs until run."""
        super().__init__()
        self._paths = paths
        self._export_file = export_file
        self._output = output
        self._rules = rules
        self._flat = flat
        self._report_file = report_file
        self._organize_workers = organize_workers
        self._provider = provider
        self._model = model
        self._api_base_url = api_base_url
        self._api_key = api_key

    def run(self) -> None:
        """Count, trim, optionally organize, and write the file, reporting either way."""
        try:
            census, source = self._census()
            if not census.uses:
                self.failed.emit(
                    _("No keywords found: there is nothing to build a vocabulary from."),
                )
                self.finished.emit()
                return
            result = trim(census, self._rules)
            stats: OrganizeStats | None = None
            if self._provider is not None:
                result, stats = self._organize(result)
            header = vocabulary_header(
                source,
                len(result.kept),
                len(result.dropped),
                self._rules,
                stats,
            )
            self._output.write_text(
                render_vocabulary(result, header=header, flat=self._flat),
                encoding="utf-8",
            )
            if self._report_file is not None:
                self._report_file.write_text(render_drop_report(result), encoding="utf-8")
        except PhotoTaggerError as exc:
            logger.error("gui_vocabulary_build_failed", error=str(exc))
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - a broken exiftool or an unwritable path
            logger.exception("gui_vocabulary_build_crashed", error=str(exc))
            self.failed.emit(str(exc))
        else:
            self.done.emit(
                _("Wrote {kept} keyword(s) to {file}, dropped {dropped}.").format(
                    kept=len(result.kept),
                    file=self._output.name,
                    dropped=len(result.dropped),
                ),
                len(result.kept),
            )
        self.finished.emit()

    def _census(self) -> tuple[KeywordCensus, str]:
        """Count the keywords on the photos, in the export, or in both; also name the source."""
        census = KeywordCensus()
        sources: list[str] = []
        if self._export_file is not None:
            self.progress.emit(_("Reading {file}...").format(file=self._export_file.name))
            census.merge(
                # utf-8-sig for the same reason load_vocabulary uses it: a BOM would be read as
                # part of the first keyword.
                census_from_export(self._export_file.read_text(encoding="utf-8-sig")),
            )
            sources.append(f"keyword export {self._export_file.name} (counts are tree occurrences)")
        if self._paths:
            self.progress.emit(
                ngettext(
                    "Reading the keywords on {n} photo...",
                    "Reading the keywords on {n} photos...",
                    len(self._paths),
                ).format(n=len(self._paths)),
            )
            photos = census_from_photos(self._paths)
            census.merge(photos)
            sources.append(f"{photos.photos} photo(s)")
        return census, " and ".join(sources)

    def _organize(self, result: TrimResult) -> tuple[TrimResult, OrganizeStats]:
        """Ask the model to fold synonyms and give the kept keywords a hierarchy."""
        self.progress.emit(
            _("Organizing {n} keyword(s) with {model}...").format(
                n=len(result.kept),
                model=self._model,
            ),
        )
        return organize(
            result,
            provider_name=self._provider,
            model_name=self._model,
            api_base_url=self._api_base_url,
            api_key=self._api_key,
            workers=self._organize_workers,
        )


class WatchWorker(QObject):
    """
    Polls the watched folders for new photos, off the UI thread.

    The same loop the CLI's ``watch`` command runs: a directory listing every few seconds, and a
    file only counts once it has stopped changing. :meth:`stop` ends it at the next poll, and cuts
    the wait short, so a Stop button does not have to sit out an interval.
    """

    batch = Signal(object)  # list[Path]
    finished = Signal()

    def __init__(self, settings: WatchSettings) -> None:
        """Store what to watch; nothing runs until :meth:`run`."""
        super().__init__()
        self._settings = settings
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to end at the next poll."""
        self._stop = True

    def run(self) -> None:
        """Emit each settled batch until the watch is stopped."""
        settings = self._settings
        try:
            for batch in watch_batches(
                list(settings.folders),
                settings.extensions,
                recursive=settings.recursive,
                interval_seconds=settings.interval,
                settle_seconds=settings.settle,
                should_stop=lambda: self._stop,
            ):
                self.batch.emit(list(batch))
        except Exception as exc:  # noqa: BLE001
            # A watch that dies must not take the window with it; the user can start another.
            logger.exception("gui_watch_failed", error=str(exc))
        self.finished.emit()
