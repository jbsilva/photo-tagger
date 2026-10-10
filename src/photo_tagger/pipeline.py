"""High-level photo processing pipeline shared by the CLI and the retry loop."""

import contextlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypedDict

from loguru import logger
from pydantic_ai.exceptions import ModelHTTPError

from photo_tagger.ai import analyze_image_with_ai, partial_usage_from, requests_per_photo
from photo_tagger.cache import InferenceCache, content_cache_key, safe_cache_get, safe_cache_put
from photo_tagger.config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_RETRIES,
    DEFAULT_SIDECAR_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_PROMPT,
    TAG_EXIF_DATE_TIME_ORIGINAL,
    SidecarMode,
    build_repair_prompt,
)
from photo_tagger.errors import BatchError
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.keywords import merge_keywords
from photo_tagger.metadata import (
    build_contextual_prompt,
    capture_year,
    managed_helper,
    read_image_context,
    write_metadata_everywhere,
)
from photo_tagger.models import InferenceResult, KeywordSet
from photo_tagger.script_guard import find_in_result
from photo_tagger.sessions import build_session_vocabulary


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from pathlib import Path

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]
    from pydantic_ai import Agent, BinaryContent

    from photo_tagger.fields import ExtraFields
    from photo_tagger.metadata import ImageContext
    from photo_tagger.models import GeneratedMetadata
    from photo_tagger.pacing import Pacer
    from photo_tagger.sessions import SessionPlan
    from photo_tagger.undo import UndoJournal
    from photo_tagger.vocabulary import Vocabulary

    OnSuccess = Callable[[Path], None]
    ProgressCallback = Callable[[Path, bool], None]
    OnComplete = Callable[["BatchTotals"], None]
    OnImageResult = Callable[["ImageOutcome"], None]

    # Only ever named in annotations, which PEP 649 leaves unevaluated, so it costs nothing at
    # runtime to keep it here. It also has to stay here: as a runtime class its member types
    # (InferenceResult, ImageContext) resolve to nothing, and pycroscope errors out trying to
    # evaluate them, which silently degrades its analysis of the whole module.
    class _InferenceScratch(TypedDict, total=False):
        """Typed scratch pad passed between process_photo and _emit_outcome."""

        inference: InferenceResult
        from_cache: bool
        context: ImageContext
        merged_keywords: KeywordSet
        foreign_terms: list[str]


# Pause between the first pass and the retry pass. First-pass failures often mean the model
# server is overloaded or mid-restart; re-hitting it immediately retries into the same outage.
# The test suite zeroes this via a conftest fixture.
_RETRY_PASS_DELAY_SECONDS = 5.0

# How many distinct keywords a strict vocabulary run reports as dropped before it stops collecting
# new ones. Generous for the intended use (spotting gaps in a catalog) and bounded for the one that
# is not (pointing --vocabulary at an unrelated file). Public because the GUI keeps the same tally
# for its own runs and there is no reason for the two to disagree.
MAX_TRACKED_DROPPED_TERMS = 200


@dataclass(slots=True, frozen=True)
class ImageOutcome:
    """
    Per-image result the pipeline streams to consumers via ``on_image_result``.

    Carries the AI fields (or what the cache replayed) plus the success bit and a ``from_cache``
    flag. The CLI uses this to emit one NDJSON line per photo when ``--json`` is set, so downstream
    tools can act on each result as soon as it lands instead of waiting for the BatchTotals summary
    at the end.

    The trailing fields carry the rest of what ``--csv-file`` reports: the final keywords actually
    written (``written_keywords`` / ``hierarchical_keywords``), the keywords already on the file,
    and the camera/location EXIF read as context. They default to empty so a photo that fails before
    inference still yields a row with whatever was read. ``keywords`` stays the raw AI list (what
    NDJSON emits); ``written_keywords`` is the merged set that landed on the file.

    A non-empty ``foreign_terms`` marks the photo for review: the generated text mixes writing
    systems (see :mod:`photo_tagger.script_guard`), and these are the bits in the odd one out.
    """

    file: Path
    success: bool
    from_cache: bool
    retry: bool
    title: str | None
    description: str | None
    keywords: list[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    seconds: float
    written_keywords: list[str] = field(default_factory=list)
    hierarchical_keywords: list[str] = field(default_factory=list)
    existing_keywords: list[str] = field(default_factory=list)
    camera_info: dict[str, str] = field(default_factory=dict)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    foreign_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessingOptions:
    """Bundle of per-photo settings that the CLI hands to the pipeline."""

    preserve_existing_kw: bool = True
    # Keywords merge, so preserving them is the default. A title and a description hold one value
    # each, so preserving one drops the generated text; replacing stays the default there.
    preserve_existing_title: bool = False
    preserve_existing_description: bool = False
    write_description: bool = True
    write_title: bool = True
    write_keywords: bool = True
    backup_xmp: bool = True
    # Sidecar, embedded, or both: resolved per photo by photo_tagger.metadata.plan_writes.
    sidecar_mode: SidecarMode = DEFAULT_SIDECAR_MODE
    dry_run: bool = False
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    # The agent's own retry budget, for bounding how long a photo can take (see requests_per_photo).
    retries: int = DEFAULT_RETRIES
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    jpeg_dimensions: int = DEFAULT_DIMENSIONS
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    max_new_keywords: int | None = None
    # Terms the generated keywords are snapped onto (see photo_tagger.vocabulary). None leaves the
    # model's own wording alone; vocabulary_strict additionally drops what the vocabulary lacks.
    vocabulary: Vocabulary | None = None
    vocabulary_strict: bool = False
    # The language the keywords are in. The pipeline itself does not generate text, but session
    # harmonization has to know which morphology rules may be applied when matching terms, and the
    # repair prompt has to name the language to rewrite into.
    output_language: str = DEFAULT_OUTPUT_LANGUAGE
    # Spend a second model call rewriting an answer that mixed writing systems. Off by default: the
    # flag alone is free and always right, while a repair costs a call and can strip a foreign term
    # the photo genuinely showed.
    repair_language: bool = False
    # Copyright, creator, and any raw tag assignment, written alongside the generated fields. None
    # (and an empty one) means the run touches only what the model produced.
    extra_fields: ExtraFields | None = None


@contextlib.contextmanager
def _no_helper() -> Generator[None]:
    """Yield None as the shared ExifToolHelper for the concurrent path."""
    yield None


@dataclass(slots=True)
class _UsageAccumulator:
    """Thread-safe running totals for token usage and failure kinds across a batch."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_seconds: float = 0.0
    inference_calls: int = 0
    cache_hits: int = 0
    failure_kinds: dict[str, int] = field(default_factory=dict)
    vocabulary_mapped: int = 0
    vocabulary_dropped: dict[str, int] = field(default_factory=dict)
    flagged_for_review: set[Path] = field(default_factory=set)
    language_repairs: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, result: InferenceResult) -> None:
        """Fold *result*'s usage into the running totals under the lock."""
        with self._lock:
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.total_tokens += result.total_tokens
            self.inference_seconds += result.seconds
            self.inference_calls += 1

    def add_cache_hit(self) -> None:
        """Count one photo that skipped the model call thanks to a cache hit."""
        with self._lock:
            self.cache_hits += 1

    def add_failed_usage(self, usage: tuple[int, int, int]) -> None:
        """
        Fold tokens burned by a call that ultimately failed into the running totals.

        pydantic-ai retries invalid structured output internally before giving up; each attempt
        is a real, billed request even though the run as a whole raises. inference_calls and
        inference_seconds are deliberately left untouched: they count completed, usable calls,
        and folding a failure into them would blur that meaning.
        """
        input_tokens, output_tokens, total_tokens = usage
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens

    def add_failure(self, kind: str) -> None:
        """Count one photo that failed for good, bucketed by coarse *kind*."""
        with self._lock:
            self.failure_kinds[kind] = self.failure_kinds.get(kind, 0) + 1

    def add_review(self, path: Path) -> None:
        """
        Mark one photo as worth a second look before its metadata is trusted.

        A set, not a counter: a photo that fails its write and comes back through the retry pass is
        analyzed twice, and the same flag raised twice is still one photo to review.
        """
        with self._lock:
            self.flagged_for_review.add(path)

    def add_repair(self) -> None:
        """Count one extra model call spent rewriting an answer into the output language."""
        with self._lock:
            self.language_repairs += 1

    def add_vocabulary(self, *, mapped: int, dropped: Iterable[str]) -> None:
        """
        Fold one photo's vocabulary rewrites and rejections into the running totals.

        The dropped-term tally is what a user acts on after a strict run (these are the concepts the
        catalog has no name for yet), so it is kept per term rather than as a bare count. New terms
        stop being recorded past :data:`MAX_TRACKED_DROPPED_TERMS`: a run against the wrong
        vocabulary can reject thousands of distinct keywords, and an unbounded dict would grow with
        them and then be dumped into the summary file.
        """
        with self._lock:
            self.vocabulary_mapped += mapped
            for term in dropped:
                if term in self.vocabulary_dropped:
                    self.vocabulary_dropped[term] += 1
                elif len(self.vocabulary_dropped) < MAX_TRACKED_DROPPED_TERMS:
                    self.vocabulary_dropped[term] = 1


# The coarse failure buckets. Classification is heuristic by exception class name/module so no
# provider SDK needs importing here; the buckets answer "why do photos fail" without carrying
# any message text.
FAILURE_TIMEOUT = "timeout"
FAILURE_CONNECTION = "connection"
FAILURE_MODEL_VALIDATION = "model-validation"
FAILURE_MODEL_API = "model-api"
FAILURE_IMAGE_READ = "image-read"
FAILURE_METADATA_WRITE = "metadata-write"
FAILURE_OTHER = "other"


def classify_failure(exc: BaseException) -> str:
    """Map an exception from one photo's processing to a coarse failure bucket."""
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return FAILURE_TIMEOUT
    if "Connect" in name or "Network" in name or "Pool" in name:
        return FAILURE_CONNECTION
    if "UnexpectedModelBehavior" in name or "Validation" in name:
        return FAILURE_MODEL_VALIDATION
    if module.startswith(("rawpy", "PIL")) or "Image" in name:
        return FAILURE_IMAGE_READ
    if "Status" in name or "HTTP" in name or "API" in name:
        return FAILURE_MODEL_API
    return FAILURE_OTHER


# HTTP status codes that retrying cannot fix: the request itself is rejected, not the server
# having a bad moment. Deliberately narrow (contrast 429/500/502/503, which a 5s pause and a
# second attempt can plausibly clear): a wrong or revoked credential does not become right by
# waiting, but almost every other status code pydantic-ai's ModelHTTPError can carry might.
_NON_RETRYABLE_STATUS_CODES = frozenset({401, 403})


def _is_permanent_failure(exc: BaseException) -> bool:
    """Report whether *exc* is one no retry pass could ever recover from."""
    return isinstance(exc, ModelHTTPError) and exc.status_code in _NON_RETRYABLE_STATUS_CODES


@dataclass(slots=True)
class _PendingWrite:
    """
    One analyzed photo waiting for its session to finish before it is written.

    Holds everything the write still needs: the keywords (which harmonization may rewrite), the two
    text fields, what the photo already carries in all three (to merge keywords into, and to decide
    whether a preserved title or description is written at all), and the scratch pad the deferred
    :class:`ImageOutcome` is built from.

    ``extra_tags`` is resolved here rather than at write time because its copyright year comes from
    this photo's own capture date, which was read with the rest of its context.
    """

    keywords: list[str]
    title: str
    description: str
    existing: KeywordSet
    scratch: _InferenceScratch
    existing_title: str | None = None
    existing_description: str | None = None
    extra_tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _BatchContext:
    """
    Shared state threaded through every function in a single run_batch call.

    Bundles the agent, options, callbacks, and accumulator so individual functions don't need 10+
    keyword arguments each.
    """

    agent: Agent[None, GeneratedMetadata]
    options: ProcessingOptions
    user_prompt: str
    usage: _UsageAccumulator
    cache: InferenceCache | None = None
    on_success: OnSuccess | None = None
    on_image_result: OnImageResult | None = None
    progress: ProgressCallback | None = None
    # In-flight inference coordination for the concurrent path: content key -> the Event the
    # first worker (the leader) will set once its result is cached. Followers with the same
    # pixels wait on it and replay the cache instead of paying a duplicate model call.
    inflight: dict[str, threading.Event] = field(default_factory=dict)
    inflight_lock: threading.Lock = field(default_factory=threading.Lock)
    # First-pass failures classified as permanent (see _is_permanent_failure), keyed by path so
    # run_batch can route them straight to the final tally instead of a doomed retry pass.
    permanently_failed: dict[Path, str] = field(default_factory=dict)
    permanently_failed_lock: threading.Lock = field(default_factory=threading.Lock)
    # Set when --session-gap groups the batch into shoots. Photos are then analyzed a session at a
    # time and written only once the whole session agrees on its keywords.
    session_plan: SessionPlan | None = None
    # Non-None while a session is being analyzed: finished analyses land here instead of on disk.
    # Everything that must wait for the write (success counting, the skip list, the per-photo
    # outcome) checks this flag rather than firing early.
    pending: dict[Path, _PendingWrite] | None = None
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    # Records every file this run writes, so `photo-tagger undo` can put it back.
    journal: UndoJournal | None = None
    # Set when the run is paced (see photo_tagger.pacing): it decides, at each photo boundary,
    # whether the machine has earned a rest. None means the run works flat out.
    pacer: Pacer | None = None


def _call_model(ctx: _BatchContext, jpeg_bytes: BinaryContent, prompt: str) -> InferenceResult:
    """Run one inference on prepared bytes and fold its usage into the batch totals."""
    try:
        inference = analyze_image_with_ai(
            image_bytes=jpeg_bytes,
            agent=ctx.agent,
            user_prompt=prompt,
            temperature=ctx.options.temperature,
            max_tokens=ctx.options.max_tokens,
            timeout_seconds=ctx.options.timeout_seconds,
            frequency_penalty=ctx.options.frequency_penalty,
        )
    except Exception as exc:
        # Tokens burned by attempts pydantic-ai retried internally before giving up would
        # otherwise vanish from the batch summary: there is no successful InferenceResult to
        # fold into ctx.usage below.
        partial = partial_usage_from(exc)
        if partial is not None:
            ctx.usage.add_failed_usage(partial)
        raise
    ctx.usage.add(inference)
    return inference


def _repair_foreign_script(
    inference: InferenceResult,
    ctx: _BatchContext,
    jpeg_bytes: BinaryContent,
    *,
    file_name: str,
) -> InferenceResult:
    """
    Ask the model once more, in place, when its answer mixed writing systems.

    Only the same agent and the same image: the question that decides the right fix is whether the
    photo shows that text, which no text-only pass can answer. A failed repair is logged and the
    original answer kept, because a photo that was tagged imperfectly still beats one not tagged. A
    repair that comes back mixed again keeps the original too, and gets no third try.

    Runs before the cache write in :func:`_resolve_inference`, so what gets stored (and replayed) is
    the repaired answer and a rerun never pays for the second call again.
    """
    if not ctx.options.repair_language:
        return inference
    finding = find_in_result(inference)
    if finding is None:
        return inference
    logger.info("foreign_script_repair_started", file=file_name, terms=finding.terms)
    prompt = build_repair_prompt(
        title=inference.title,
        description=inference.description,
        keywords=inference.keywords,
        terms=finding.terms,
        output_language=ctx.options.output_language,
    )
    try:
        repaired = _call_model(ctx, jpeg_bytes, prompt)
    except Exception as exc:  # noqa: BLE001 - a failed repair must not cost the photo its tags.
        logger.warning("foreign_script_repair_failed", file=file_name, error=str(exc))
        return inference
    ctx.usage.add_repair()
    still_mixed = find_in_result(repaired) is not None
    logger.info(
        "foreign_script_repair_finished",
        file=file_name,
        title=repaired.title,
        still_mixed=still_mixed,
    )
    # A repair that mixes scripts again is no better, and the original is the answer the photo
    # was flagged on: keep it, so what the person reviews is what the check found.
    return inference.with_cost_of(repaired) if still_mixed else repaired.with_cost_of(inference)


def _repair_cached(
    image_path: Path,
    cached: InferenceResult,
    ctx: _BatchContext,
    *,
    cache: InferenceCache,
    content_key: str,
) -> tuple[InferenceResult, bool]:
    """
    Return a replayed answer, repaired first when it mixes scripts and --repair-language is on.

    An answer cached by a run without the setting would otherwise be replayed mixed forever. A
    repair that helps replaces the cached answer, so the next run replays the fixed one. The copy
    handed to the repair carries no cost, since the cached call was paid for by an earlier run.
    """
    if not ctx.options.repair_language or find_in_result(cached) is None:
        return cached, True
    jpeg_bytes = prepare_image_for_agent(
        image_path,
        jpg_quality=ctx.options.jpeg_quality,
        max_size=ctx.options.jpeg_dimensions,
    )
    unpaid = replace(cached, input_tokens=0, output_tokens=0, total_tokens=0, seconds=0.0)
    result = _repair_foreign_script(
        unpaid,  # NOSONAR S5655 - replace() keeps the type of cached
        ctx,
        jpeg_bytes,
        file_name=image_path.name,
    )
    if (
        result.title == cached.title
        and result.keywords == cached.keywords
        and (result.description == cached.description)
    ):
        # The repair failed or came back mixed again: the cached answer stands, as it was.
        return result, result.total_tokens == 0
    safe_cache_put(cache, content_key, result, file_name=image_path.name)
    return result, False


def _run_model(image_path: Path, ctx: _BatchContext, *, contextual_prompt: str) -> InferenceResult:
    """Prepare the JPEG bytes, call the model, and repair an answer that mixed writing systems."""
    jpeg_bytes = prepare_image_for_agent(
        image_path,
        jpg_quality=ctx.options.jpeg_quality,
        max_size=ctx.options.jpeg_dimensions,
    )
    inference = _call_model(ctx, jpeg_bytes, contextual_prompt)
    return _repair_foreign_script(inference, ctx, jpeg_bytes, file_name=image_path.name)


def _cache_replay(
    ctx: _BatchContext,
    content_key: str,
    *,
    file_name: str,
) -> InferenceResult | None:
    """Return the cached result for *content_key* (counting the hit), or None on miss."""
    if ctx.cache is None:
        return None
    cached = safe_cache_get(ctx.cache, content_key, file_name=file_name)
    if cached is not None:
        logger.info("cache_hit", file=file_name)
        ctx.usage.add_cache_hit()
    return cached


def _resolve_inference(
    image_path: Path,
    ctx: _BatchContext,
    *,
    contextual_prompt: str,
    content_key: str | None,
) -> tuple[InferenceResult, bool]:
    """
    Return ``(inference, from_cache)`` for *image_path*.

    Hits the on-disk cache when one is provided and *content_key* matches a prior entry recorded
    under the same namespace. On miss, prepares the JPEG bytes, calls the model, and writes the
    result back to the cache.

    Concurrent misses on the *same* content key (burst duplicates, one image in two folders) are
    coordinated: the first worker becomes the leader and runs the model; the others wait for it and
    replay the cache, so identical pixels never pay twice. If the leader fails (or cannot cache its
    result), the waiters fall back to their own model call.

    Cache I/O failures are logged at warning level but never raised: a broken SQLite file or full
    disk degrades the run to "no cache" without aborting photos that the model would otherwise
    process successfully.
    """
    if ctx.cache is None or content_key is None:
        return _run_model(image_path, ctx, contextual_prompt=contextual_prompt), False

    cached = _cache_replay(ctx, content_key, file_name=image_path.name)
    if cached is not None:
        return _repair_cached(image_path, cached, ctx, cache=ctx.cache, content_key=content_key)

    with ctx.inflight_lock:
        leader_event = ctx.inflight.get(content_key)
        if leader_event is None:
            ctx.inflight[content_key] = threading.Event()

    if leader_event is not None:
        # Another worker is already inferring these pixels; wait (bounded by every request it may
        # make, a repair included, plus slack for image preparation) and replay its cached result.
        repairs = 2 if ctx.options.repair_language else 1
        requests = requests_per_photo(ctx.options.retries) * repairs
        leader_event.wait(timeout=ctx.options.timeout_seconds * requests + 30.0)
        cached = _cache_replay(ctx, content_key, file_name=image_path.name)
        if cached is not None:
            return cached, True
        # The leader failed or could not cache; pay our own call rather than give up.
        return _run_model(image_path, ctx, contextual_prompt=contextual_prompt), False

    try:
        inference = _run_model(image_path, ctx, contextual_prompt=contextual_prompt)
        safe_cache_put(ctx.cache, content_key, inference, file_name=image_path.name)
        return inference, False
    finally:
        # Wake the waiters whether we succeeded or raised; they re-check the cache either way.
        with ctx.inflight_lock:
            our_event = ctx.inflight.pop(content_key, None)
        if our_event is not None:
            our_event.set()


def _record_scratch(  # noqa: PLR0913 - one kwarg per piece the sink collects; a bundle adds nothing.
    sink: _InferenceScratch | None,
    *,
    context: ImageContext | None = None,
    inference: InferenceResult | None = None,
    from_cache: bool = False,
    merged_keywords: KeywordSet | None = None,
    foreign_terms: list[str] | None = None,
) -> None:
    """
    Write the supplied pieces into the per-call *sink*, a no-op when no sink was given.

    Centralizes the ``outcome_sink is not None`` guard so process_photo records context, inference,
    and merged keywords as plain one-liners as each becomes available.
    """
    if sink is None:
        return
    if context is not None:
        sink["context"] = context
    if inference is not None:
        sink["inference"] = inference
        sink["from_cache"] = from_cache
    if merged_keywords is not None:
        sink["merged_keywords"] = merged_keywords
    if foreign_terms is not None:
        sink["foreign_terms"] = foreign_terms


def _flag_foreign_script(
    inference: InferenceResult,
    ctx: _BatchContext,
    image_path: Path,
    *,
    sink: _InferenceScratch | None,
) -> None:
    """
    Mark the photo for review when its generated text mixes writing systems.

    Runs on every photo, cache replays included: the check is pure and costs nothing, so recomputing
    it beats storing a verdict next to the cached inference and having to version that.

    The photo is still written. Mixed script is sometimes right (the photo shows the text), so this
    points a human at the result rather than deciding for them; ``--repair-language`` is what tries
    to fix it.
    """
    finding = find_in_result(inference)
    if finding is None:
        return
    logger.warning(
        "foreign_script_in_output",
        file=image_path.name,
        dominant=finding.dominant,
        terms=finding.terms,
    )
    ctx.usage.add_review(image_path)
    _record_scratch(sink, foreign_terms=finding.terms)


def _apply_vocabulary(keywords: list[str], ctx: _BatchContext, *, file_name: str) -> list[str]:
    """
    Snap the model's keywords onto the configured vocabulary, recording what it changed.

    Returns the keywords untouched when no vocabulary is configured. Rewrites and (in strict mode)
    rejections are logged per photo and tallied on the batch so the summary file can report them.
    """
    vocabulary = ctx.options.vocabulary
    if not vocabulary:
        return keywords
    result = vocabulary.snap(keywords, strict=ctx.options.vocabulary_strict)
    if result.mapped or result.dropped:
        logger.info(
            "vocabulary_applied",
            file=file_name,
            mapped=result.mapped,
            dropped=result.dropped,
        )
        ctx.usage.add_vocabulary(mapped=len(result.mapped), dropped=result.dropped)
    return result.keywords


def _apply_session_vocabulary(
    keywords: list[str],
    ctx: _BatchContext,
    image_path: Path,
) -> list[str]:
    """
    Snap keywords onto the vocabulary this photo's session settled on, if it has one yet.

    Only the retry pass finds one: while a session is being analyzed there is nothing to agree with
    yet, and the flush that follows harmonizes the whole session at once. A photo that comes back
    through the retry pass afterwards still lands on the same terms as the rest of its shoot.
    """
    if ctx.session_plan is None:
        return keywords
    vocabulary = ctx.session_plan.vocabulary_for(image_path)
    if not vocabulary:
        return keywords
    return vocabulary.snap(keywords).keywords


def _resolve_extra_tags(
    image_path: Path,
    context: ImageContext,
    options: ProcessingOptions,
) -> dict[str, str]:
    """
    Render this photo's copyright and ``--set`` tags, dating the notice by its capture year.

    The capture date is already in *context* (the prompt uses it as a hint), so a per-photo year
    costs no extra exiftool call.
    """
    if options.extra_fields is None or options.extra_fields.is_empty:
        return {}
    captured = context.camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL)
    return options.extra_fields.resolve(capture_year(image_path, captured))


def process_photo(
    image_path: Path,
    ctx: _BatchContext,
    *,
    et: ExifToolHelper | None = None,
    outcome_sink: _InferenceScratch | None = None,
) -> bool:
    """
    Convert an image to JPEG bytes in memory, query the model, and persist metadata.

    Args:
        image_path: Image to process.
        ctx: Shared batch context (agent, options, user_prompt, cache, usage).
        et: Optional pre-opened ExifToolHelper. Reused for every read/write in this call
            when supplied, otherwise one helper is opened for the duration of this photo.
            Concurrent callers must NOT share a helper across threads (the underlying
            -stay_open subprocess uses a single stdin/stdout pipe); pass et=None instead
            and let each task get its own.
        outcome_sink: Optional per-call scratch dict. When provided, this function
            populates ``inference`` (InferenceResult) and ``from_cache`` (bool) so the
            caller can wrap them into an ImageOutcome without retracing the work.

    Returns:
        True if every step succeeded, False if metadata writing failed. In session mode the write
        is deferred (see :class:`_PendingWrite`), so True there means "analyzed, queued to write".
    """
    logger.info("processing_photo")
    options = ctx.options

    with managed_helper(et) as helper:
        # Read the content hash in the same call only when a cache can use it.
        context = read_image_context(
            image_path,
            et=helper,
            include_content_hash=ctx.cache is not None,
        )
        existing_keywords_full = context.existing_keywords
        if not existing_keywords_full.is_empty():
            logger.info(
                "existing_keywords_found",
                count=len(existing_keywords_full.subject),
            )
        # Stash the read context early so even a photo that fails during inference still
        # carries its EXIF/existing-keyword columns into the CSV report.
        _record_scratch(outcome_sink, context=context)

        gps_info = {"position": context.gps_position} if context.gps_position else {}
        contextual_prompt = build_contextual_prompt(
            ctx.user_prompt,
            # build_contextual_prompt de-duplicates its flat keywords itself.
            existing_keywords_full.subject,
            context.location_tags,
            gps_info,
            camera_info=context.camera_info,
        )

        content_key = (
            content_cache_key(image_path, context.content_hash) if ctx.cache is not None else None
        )
        inference, from_cache = _resolve_inference(
            image_path,
            ctx,
            contextual_prompt=contextual_prompt,
            content_key=content_key,
        )
        _record_scratch(outcome_sink, inference=inference, from_cache=from_cache)
        _flag_foreign_script(inference, ctx, image_path, sink=outcome_sink)

        title = inference.title
        description = inference.description
        # The vocabulary runs before the cap so the cap counts keywords that will actually be
        # written: in strict mode it would otherwise be spent on terms about to be dropped.
        keywords = _apply_vocabulary(inference.keywords, ctx, file_name=image_path.name)

        if options.max_new_keywords is not None and len(keywords) > options.max_new_keywords:
            logger.info(
                "trimming_ai_keywords",
                returned=len(keywords),
                cap=options.max_new_keywords,
            )
            keywords = keywords[: options.max_new_keywords]

        keywords = _apply_session_vocabulary(keywords, ctx, image_path)

        pending = _PendingWrite(
            keywords=keywords,
            title=title,
            description=description,
            existing=existing_keywords_full if options.preserve_existing_kw else KeywordSet(),
            scratch=outcome_sink if outcome_sink is not None else {},
            existing_title=context.existing_title,
            existing_description=context.existing_description,
            extra_tags=_resolve_extra_tags(image_path, context, options),
        )
        if ctx.pending is not None:
            # Session mode: hold the analysis until the whole shoot has one shared vocabulary.
            with ctx.pending_lock:
                ctx.pending[image_path] = pending
            return True

        return _write_pending(image_path, pending, ctx, et=helper)


def _caption_to_write(
    generated: str,
    existing: str | None,
    *,
    write: bool,
    preserve: bool,
) -> str | None:
    """
    Decide what a title or a description field puts on the file, None meaning "leave it alone".

    One place, so the dry-run preview and the real write cannot disagree.
    """
    if not write:
        return None
    if preserve and existing:
        return None
    return generated


def _write_pending(
    image_path: Path,
    pending: _PendingWrite,
    ctx: _BatchContext,
    *,
    et: ExifToolHelper | None = None,
) -> bool:
    """
    Merge one analyzed photo's keywords with what is already on it and write the result.

    Split out of :func:`process_photo` so session mode can run it later, after harmonization has had
    its say over ``pending.keywords``. Returns True on a successful write or a dry run.
    """
    options = ctx.options
    # An empty set when keywords are disabled, so write_metadata emits no keyword tags and
    # leaves whatever is already on the file untouched (e.g. refresh only title/description).
    # The user's vocabulary spelled its own terms; the session vocabulary only mirrors what the
    # model said, so only the former overrides the merge step's capitalization.
    verbatim = options.vocabulary.exact if options.vocabulary else None
    merged_keywords = (
        merge_keywords(pending.existing, pending.keywords, verbatim=verbatim)
        if options.write_keywords
        else KeywordSet()
    )
    _record_scratch(pending.scratch, merged_keywords=merged_keywords)
    title = _caption_to_write(
        pending.title,
        pending.existing_title,
        write=options.write_title,
        preserve=options.preserve_existing_title,
    )
    description = _caption_to_write(
        pending.description,
        pending.existing_description,
        write=options.write_description,
        preserve=options.preserve_existing_description,
    )

    if options.dry_run:
        logger.info(
            "dry_run_preview",
            file=image_path.name,
            title=title,
            description=description,
            subject_keywords=merged_keywords.subject,
            hierarchical_keywords=merged_keywords.hierarchical,
            extra_tags=pending.extra_tags,
        )
        return True

    written, targets = write_metadata_everywhere(
        image_path,
        merged_keywords,
        description=description,
        title=title,
        extra=pending.extra_tags,
        backup=options.backup_xmp,
        sidecar_mode=options.sidecar_mode,
        et=et,
    )
    if ctx.journal is not None:
        for target in targets:
            ctx.journal.record(
                image_path,
                target.path,
                created=not target.existed,
                backed_up=options.backup_xmp,
            )
    return written


def _emit_outcome(
    on_image_result: OnImageResult | None,
    image_file: Path,
    scratch: _InferenceScratch,
    *,
    success: bool,
    retry: bool,
) -> None:
    """Build an ImageOutcome from *scratch* and *success* and call *on_image_result*."""
    if on_image_result is None:
        return
    inference: InferenceResult | None = scratch.get("inference")
    context: ImageContext | None = scratch.get("context")
    merged: KeywordSet | None = scratch.get("merged_keywords")
    outcome = ImageOutcome(
        file=image_file,
        success=success,
        from_cache=bool(scratch.get("from_cache", False)),
        retry=retry,
        title=inference.title if inference is not None else None,
        description=inference.description if inference is not None else None,
        keywords=list(inference.keywords) if inference is not None else [],
        input_tokens=inference.input_tokens if inference is not None else 0,
        output_tokens=inference.output_tokens if inference is not None else 0,
        total_tokens=inference.total_tokens if inference is not None else 0,
        seconds=inference.seconds if inference is not None else 0.0,
        written_keywords=list(merged.subject) if merged is not None else [],
        hierarchical_keywords=list(merged.hierarchical) if merged is not None else [],
        existing_keywords=list(context.existing_keywords.subject) if context is not None else [],
        camera_info=dict(context.camera_info) if context is not None else {},
        location_tags=dict(context.location_tags) if context is not None else {},
        gps_position=context.gps_position if context is not None else None,
        foreign_terms=list(scratch.get("foreign_terms", [])),
    )
    try:
        on_image_result(outcome)
    except Exception as exc:  # noqa: BLE001 - callback errors must not break the batch.
        logger.exception("on_image_result_callback_failed", file=image_file.name, error=str(exc))


def _rest_before_photo(ctx: _BatchContext, image_file: Path) -> bool:
    """
    Let a paced run rest before *image_file* starts; False means the run is stopping.

    Every worker thread calls this, and they all rest together: the pacer lets one of them do the
    sleeping while the rest queue behind it.
    """
    if ctx.pacer is None or ctx.pacer.wait_before_next():
        return True
    logger.info("photo_not_started", file=image_file.name)
    return False


def execute_process(
    image_file: Path,
    ctx: _BatchContext,
    *,
    index: str,
    retry: bool = False,
    et: ExifToolHelper | None = None,
) -> bool:
    """
    Run process_photo once with consistent logging and error handling.

    A paced run may rest here first, before the photo starts: this is the photo boundary, the one
    place where stopping costs nothing. A rest that gets cancelled means the run is ending, so the
    photo is left untouched and reported as still pending.
    """
    if not _rest_before_photo(ctx, image_file):
        return False

    context_kwargs: dict[str, Any] = {"file": image_file.name}
    if retry:
        context_kwargs["retry"] = True

    scratch: _InferenceScratch = {}
    with logger.contextualize(**context_kwargs):
        try:
            ok = process_photo(
                image_file,
                ctx,
                et=et,
                outcome_sink=scratch,
            )
        except Exception as exc:  # noqa: BLE001 - process_photo wraps several SDKs
            event = "processing_retry_exception" if retry else "processing_exception"
            logger.exception(event, error=str(exc))
            if retry:
                # Only the retry pass records a kind: it is the photo's final failure.
                ctx.usage.add_failure(classify_failure(exc))
            elif _is_permanent_failure(exc):
                # No retry pass could recover this one (e.g. a rejected credential). This is
                # its final failure too, so classify it now, and flag it so run_batch routes
                # it straight to the tally instead of into a pass doomed to repeat it.
                kind = classify_failure(exc)
                ctx.usage.add_failure(kind)
                with ctx.permanently_failed_lock:
                    ctx.permanently_failed[image_file] = kind
            _emit_outcome(ctx.on_image_result, image_file, scratch, success=False, retry=retry)
            return False

        if ok:
            event = "retry_success" if retry else "processing_success"
            logger.info(event, index=index)
            if ctx.pending is None:
                # In session mode the photo is only analyzed at this point; the flush emits its
                # outcome once the write has actually happened and can be reported truthfully.
                _emit_outcome(ctx.on_image_result, image_file, scratch, success=True, retry=retry)
            return True

        if retry:
            logger.error("retry_failed", index=index)
            # A clean False from process_photo means every step up to the write succeeded.
            ctx.usage.add_failure(FAILURE_METADATA_WRITE)
        else:
            logger.error("processing_failed", index=index, queued_for_retry=True)
        _emit_outcome(ctx.on_image_result, image_file, scratch, success=False, retry=retry)
        return False


@dataclass(slots=True)
class BatchTotals:
    """
    Public summary of one ``run_batch`` invocation.

    The CLI surfaces this in logs and uses it to write the optional JSON summary file.
    """

    total_files: int = 0
    success: int = 0
    initial_failures: int = 0
    retry_successes: int = 0
    failed_files: list[str] = field(default_factory=list)
    successful_files: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_seconds: float = 0.0
    inference_calls: int = 0
    cache_hits: int = 0
    workers: int = 1
    dry_run: bool = False
    # Final failures bucketed by coarse kind (timeout, connection, model-validation, ...), so
    # the summary file and telemetry can say WHY photos failed, not just how many.
    failure_kinds: dict[str, int] = field(default_factory=dict)
    # What --vocabulary did: how many keywords it rewrote to the catalog's spelling, and which
    # terms strict mode rejected (with how often), so the summary file names the gaps to fill.
    vocabulary_mapped: int = 0
    vocabulary_dropped: dict[str, int] = field(default_factory=dict)
    # Photos whose generated text mixes writing systems (see photo_tagger.script_guard). They were
    # written like any other; the list is what to go back and read. Any that --repair-language
    # rewrote are already gone from it; language_repairs counts the extra calls that took.
    flagged_files: list[str] = field(default_factory=list)
    language_repairs: int = 0
    # Wall-clock seconds the run spent resting on purpose (see photo_tagger.pacing), and how many
    # rests that took. Both stay zero unless the run was paced.
    rested_seconds: float = 0.0
    rests_taken: int = 0


def _notify_success(ctx: _BatchContext, image_file: Path) -> None:
    """
    Invoke ``ctx.on_success`` defensively; a callback failure must not abort the batch.

    Silent while a session is being analyzed: nothing has been written yet, and the callback is
    what appends to the resume skip list. The flush calls this once the write lands.
    """
    if ctx.on_success is None or ctx.pending is not None:
        return
    try:
        ctx.on_success(image_file)
    except Exception as exc:  # noqa: BLE001 - any callback error must not break the batch.
        logger.exception("on_success_callback_failed", file=image_file.name, error=str(exc))


def _run_pass_serial(
    image_files: list[Path],
    ctx: _BatchContext,
    *,
    retry: bool,
    et: ExifToolHelper | None,
) -> tuple[int, list[Path], bool]:
    """
    Run *image_files* one after another in the calling thread.

    A ``KeyboardInterrupt`` raised between photos stops the pass and returns ``(successes, still-
    pending, interrupted=True)`` so the caller can still emit a BatchTotals + summary file. Photos
    that were not yet attempted land in the failed list so they show up in the summary too.
    """
    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    for idx, image_file in enumerate(image_files, start=1):
        try:
            ok = execute_process(
                image_file,
                ctx,
                index=f"{idx}/{total}",
                retry=retry,
                et=et,
            )
        except KeyboardInterrupt:
            logger.warning("batch_interrupted_by_user", remaining=total - idx + 1)
            failed.extend(image_files[idx - 1 :])
            return successes, failed, True
        if ok:
            successes += 1
            _notify_success(ctx, image_file)
        elif retry:
            logger.error("file_failed_after_retry", file=image_file.name)
            failed.append(image_file)
        else:
            logger.warning("file_queued_for_retry", file=image_file.name)
            failed.append(image_file)
        # Advance the bar only when we're truly done with this file: a success
        # at any point, or any outcome in the retry pass. A first-pass failure
        # is still pending retry, so it must not tick yet (otherwise the bar
        # would overshoot once the retry pass also ticks).
        # A session's photos are not finished when they analyze: the flush writes them and
        # reports each one with the result of its write.
        if ctx.progress is not None and (ok or retry) and ctx.pending is None:
            ctx.progress(image_file, ok)
    return successes, failed, False


def _run_pass_concurrent(
    image_files: list[Path],
    ctx: _BatchContext,
    *,
    retry: bool,
    workers: int,
) -> tuple[int, list[Path], bool]:
    """
    Run *image_files* across a thread pool, giving each task its own ExifToolHelper.

    pyexiftool's -stay_open subprocess uses one stdin/stdout pipe per helper, so a helper cannot be
    shared across threads safely. Each task receives et=None and process_photo opens a short-lived
    helper for the duration of one photo.

    A ``KeyboardInterrupt`` during the as_completed loop cancels pending futures and returns
    ``(successes, still-pending, interrupted=True)``. Already in-flight workers cannot be
    interrupted from the outside, so this is a best-effort quick stop rather than an instant abort.
    """
    total = len(image_files)
    successes = 0
    failed: list[Path] = []
    interrupted = False
    indexed = list(enumerate(image_files, start=1))

    pool = ThreadPoolExecutor(max_workers=workers)
    # Submission happens inside the KeyboardInterrupt-handling try below: a Ctrl-C while futures
    # are still being queued must take the same cancel-and-drain path as one during the loop,
    # not fall through to the blocking shutdown(wait=True).
    future_to_image: dict[Future[bool], Path] = {}
    consumed: set[Future[bool]] = set()
    try:
        try:
            for idx, image_file in indexed:
                future = pool.submit(
                    execute_process,
                    image_file,
                    ctx,
                    index=f"{idx}/{total}",
                    retry=retry,
                    et=None,
                )
                future_to_image[future] = image_file
            for future in as_completed(future_to_image):
                consumed.add(future)
                image_file = future_to_image[future]
                try:
                    ok = future.result()
                except Exception as exc:  # noqa: BLE001 - per-file failure stays per-file.
                    logger.exception(
                        "concurrent_worker_exception",
                        file=image_file.name,
                        error=str(exc),
                    )
                    ok = False

                if ok:
                    successes += 1
                    _notify_success(ctx, image_file)
                elif retry:
                    logger.error("file_failed_after_retry", file=image_file.name)
                    failed.append(image_file)
                else:
                    logger.warning("file_queued_for_retry", file=image_file.name)
                    failed.append(image_file)
                # See _run_pass_serial: tick only when this file is finally done.
                # Silent during a session's analysis; the flush reports those photos.
                if ctx.progress is not None and (ok or retry) and ctx.pending is None:
                    ctx.progress(image_file, ok)
        except KeyboardInterrupt:
            interrupted = True
            # Workers asleep in a pacing rest would otherwise hold the drain below for minutes.
            if ctx.pacer is not None:
                ctx.pacer.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            remaining = {f: img for f, img in future_to_image.items() if f not in consumed}
            drained_ok, drained_failed, cancelled = _drain_after_interrupt(remaining, ctx)
            # Images the interrupt caught before submission are pending too.
            submitted = set(future_to_image.values())
            never_submitted = [img for _, img in indexed if img not in submitted]
            successes += drained_ok
            failed.extend(drained_failed)
            failed.extend(cancelled)
            failed.extend(never_submitted)
            logger.warning(
                "batch_interrupted_by_user",
                pending=len(cancelled) + len(never_submitted),
                drained=drained_ok + len(drained_failed),
            )
    finally:
        pool.shutdown(wait=True)
    return successes, failed, interrupted


def _drain_after_interrupt(
    remaining: dict[Future[bool], Path],
    ctx: _BatchContext,
) -> tuple[int, list[Path], list[Path]]:
    """
    Settle the futures a KeyboardInterrupt left behind; return (successes, failures, cancelled).

    Queued futures were cancelled and stay pending, but a task already in flight cannot be
    interrupted and runs to completion anyway. A photo whose metadata was written during that drain
    is a real success (it is on disk); reporting it as pending would miscount the summary and make a
    resume redo finished work.
    """
    drained_successes = 0
    failures: list[Path] = []
    cancelled: list[Path] = []
    for future, image_file in remaining.items():
        if future.cancelled():
            cancelled.append(image_file)
            continue
        # exception() blocks until the in-flight task settles and hands the error back without
        # re-raising, so a worker that also hit the KeyboardInterrupt cannot abort the drain.
        error = future.exception()
        if error is not None:
            logger.error("concurrent_worker_exception", file=image_file.name, error=str(error))
        ok = future.result() if error is None else False
        if ok:
            drained_successes += 1
            _notify_success(ctx, image_file)
        else:
            failures.append(image_file)
    return drained_successes, failures, cancelled


def _run_pass(
    image_files: list[Path],
    ctx: _BatchContext,
    *,
    retry: bool,
    et: ExifToolHelper | None,
    workers: int,
) -> tuple[int, list[Path], bool]:
    """
    Run a single pass (initial or retry) over the batch.

    Returns ``(success_count, still_failing, interrupted)``. The first pass logs failures with
    ``file_queued_for_retry``; the retry pass logs them with ``file_failed_after_retry``. The
    interrupted flag is True if a Ctrl-C caused the pass to abort early; the caller uses it to skip
    the retry pass and to mark the run as a partial completion.
    """
    if not image_files:
        return 0, [], False

    if retry:
        logger.info("retrying_failed_files", count=len(image_files))

    if workers <= 1:
        return _run_pass_serial(
            image_files,
            ctx,
            retry=retry,
            et=et,
        )
    return _run_pass_concurrent(
        image_files,
        ctx,
        retry=retry,
        workers=workers,
    )


@dataclass(slots=True, frozen=True)
class _SessionOutcome:
    """
    What one pass over the session plan produced.

    ``retryable`` are photos that failed during analysis and are worth another model call.
    ``final_failures`` are photos whose analysis succeeded but whose write did not: the model work
    is already done and harmonized, and an exiftool write that failed (unwritable folder, full disk)
    is not the kind of failure a second attempt clears, so they skip the retry pass.
    """

    successes: int = 0
    retryable: list[Path] = field(default_factory=list)
    final_failures: list[Path] = field(default_factory=list)
    interrupted: bool = False


def _flush_session(
    files: list[Path],
    ctx: _BatchContext,
    collected: dict[Path, _PendingWrite],
    *,
    index: int,
    et: ExifToolHelper | None,
) -> tuple[list[Path], list[Path]]:
    """
    Harmonize one session's keywords, then write every photo in it.

    Returns ``(written, write_failures)``. The written list is what the caller settles the pass's
    bookkeeping against: a photo the pass reported as unfinished (a Ctrl-C caught between its
    analysis and the end of the pass) is on disk all the same and must not also be queued for a
    retry. The session's vocabulary is kept on the plan so the retry pass can apply it to any photo
    that reaches the write later.
    """
    if not collected:
        return [], []

    vocabulary = build_session_vocabulary(
        (pending.keywords for pending in collected.values()),
        output_language=ctx.options.output_language,
    )
    if ctx.session_plan is not None:
        ctx.session_plan.remember(index, vocabulary)
    logger.info(
        "session_harmonized",
        session=index + 1,
        photos=len(collected),
        terms=len(vocabulary.terms),
    )

    written: list[Path] = []
    failures: list[Path] = []
    for image_file in files:
        pending = collected.get(image_file)
        if pending is None:
            continue  # Analysis failed; already counted and queued for the retry pass.
        pending.keywords = vocabulary.snap(pending.keywords).keywords
        ok = _write_pending(image_file, pending, ctx, et=et)
        _emit_outcome(ctx.on_image_result, image_file, pending.scratch, success=ok, retry=False)
        if ctx.progress is not None:
            # The photo is finished only now, and a write that failed is final: there is no retry
            # pass for it, so the bar must show it as the failure it is.
            ctx.progress(image_file, ok)
        if ok:
            written.append(image_file)
            _notify_success(ctx, image_file)
        else:
            logger.error("session_write_failed", file=image_file.name)
            ctx.usage.add_failure(FAILURE_METADATA_WRITE)
            failures.append(image_file)
    return written, failures


def _run_session_passes(
    image_files: list[Path],
    ctx: _BatchContext,
    *,
    et: ExifToolHelper | None,
    workers: int,
) -> _SessionOutcome:
    """
    Analyze, harmonize, and write the batch one session at a time.

    Sessions run in sequence so a shoot is complete (and can agree with itself) before the next one
    starts; the photos inside a session still run across the thread pool. A Ctrl-C stops after the
    session in flight has been written, so no analyzed photo is lost.
    """
    plan = ctx.session_plan
    if plan is None:  # pragma: no cover - callers check before getting here
        return _SessionOutcome()

    wanted = set(image_files)
    successes = 0
    retryable: list[Path] = []
    final_failures: list[Path] = []
    for index, session in enumerate(plan.sessions):
        files = [path for path in session if path in wanted]
        if not files:
            continue
        ctx.pending = {}
        _, failed, interrupted = _run_pass(
            files,
            ctx,
            retry=False,
            et=et,
            # A one-photo session has nothing to parallelize, and a pool per photo would cost
            # more than it saves on a batch of singletons.
            workers=min(workers, len(files)),
        )
        collected = ctx.pending
        ctx.pending = None
        written, write_failures = _flush_session(files, ctx, collected, index=index, et=et)
        successes += len(written)
        final_failures.extend(write_failures)
        # A photo the pass called unfinished but the flush wrote anyway (Ctrl-C between the two)
        # is done; only what never reached disk is worth another model call.
        settled = set(written) | set(write_failures)
        retryable.extend(path for path in failed if path not in settled)
        if interrupted:
            done = {path for session_files in plan.sessions[: index + 1] for path in session_files}
            retryable.extend(path for path in image_files if path not in done)
            return _SessionOutcome(successes, retryable, final_failures, interrupted=True)
    return _SessionOutcome(successes, retryable, final_failures)


def _run_retry_phase(
    pending: list[Path],
    ctx: _BatchContext,
    *,
    et: ExifToolHelper | None,
    workers: int,
) -> tuple[int, list[Path]]:
    """
    Pause, then retry *pending*, skipping failures already known unrecoverable.

    Returns ``(retry_successes, still_failing)``. Failures classify_failure/execute_process
    already flagged as permanent (see _is_permanent_failure) skip the pass entirely, and the
    delay before it: nothing about waiting and asking again fixes a rejected credential, and
    doing so anyway would double the batch's wall-clock time and outbound requests for a photo
    that cannot succeed. A Ctrl-C during the pre-retry pause is treated the same as one mid-pass:
    nothing in the retryable set was attempted, so it all stays pending for the summary.
    """
    permanent = [path for path in pending if path in ctx.permanently_failed]
    retryable = [path for path in pending if path not in ctx.permanently_failed]
    for path in permanent:
        # Already classified and counted in execute_process; this only settles the progress
        # bar, which otherwise waits for a retry pass these will never enter.
        if ctx.progress is not None:
            ctx.progress(path, False)  # noqa: FBT003 - progress callback's own signature

    if not retryable:
        return 0, permanent

    if _RETRY_PASS_DELAY_SECONDS > 0:
        logger.info("pausing_before_retry_pass", seconds=_RETRY_PASS_DELAY_SECONDS)
        try:
            time.sleep(_RETRY_PASS_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.warning("batch_interrupted_by_user", remaining=len(retryable))
            return 0, retryable + permanent

    retry_successes, still_failing_from_retry, _ = _run_pass(
        retryable,
        ctx,
        retry=True,
        et=et,
        workers=workers,
    )
    return retry_successes, still_failing_from_retry + permanent


def _log_totals(totals: BatchTotals, *, workers: int) -> None:
    """
    Report one finished batch: the tally always, then whatever else the run has to say.

    The three conditional lines exist because a summary file is optional and a console is not: a
    strict vocabulary that discarded half the keywords, a photo whose text needs a second look, and
    a photo that never got written are all things a user has to learn without opening a report.
    """
    logger.info(
        "processing_summary",
        total_files=totals.total_files,
        successful=totals.success,
        failed=len(totals.failed_files),
        initial_failures=totals.initial_failures,
        retry_successes=totals.retry_successes,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        inference_calls=totals.inference_calls,
        inference_seconds=totals.inference_seconds,
        cache_hits=totals.cache_hits,
        dry_run=totals.dry_run,
        workers=workers,
    )
    if totals.vocabulary_mapped or totals.vocabulary_dropped:
        logger.info(
            "vocabulary_summary",
            mapped=totals.vocabulary_mapped,
            dropped_terms=len(totals.vocabulary_dropped),
            dropped=totals.vocabulary_dropped,
        )
    if totals.language_repairs:
        logger.info("language_repairs", calls=totals.language_repairs)
    if totals.rests_taken:
        logger.info(
            "pacing_summary",
            rests=totals.rests_taken,
            seconds=totals.rested_seconds,
        )
    if totals.flagged_files:
        logger.warning(
            "files_flagged_for_review",
            count=len(totals.flagged_files),
            files=totals.flagged_files,
        )
    if totals.failed_files:
        logger.error("files_failed_after_retry", files=totals.failed_files)


def run_batch(  # noqa: PLR0913 - public entry point; each kwarg is a distinct caller knob.
    image_files: list[Path],
    agent: Agent[None, GeneratedMetadata],
    options: ProcessingOptions,
    *,
    on_success: OnSuccess | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
    workers: int = 1,
    progress: ProgressCallback | None = None,
    on_complete: OnComplete | None = None,
    cache: InferenceCache | None = None,
    on_image_result: OnImageResult | None = None,
    session_plan: SessionPlan | None = None,
    journal: UndoJournal | None = None,
    pacer: Pacer | None = None,
) -> BatchTotals:
    """
    Run the initial pass plus a single retry pass and return summary totals.

    When workers == 1 a single ExifToolHelper is opened for the whole batch so every metadata read
    and write reuses one long-running exiftool subprocess. When workers > 1 the tasks run on a
    ThreadPoolExecutor and each task opens its own short-lived helper (pyexiftool's -stay_open pipe
    is not safe to share across threads).

    The optional *on_success* callback fires once per image that completes successfully (whether on
    the first pass or after a retry). It receives the image path. The CLI uses this to append
    filenames to a skip list as work progresses, so a killed run can be resumed without redoing
    finished photos. Dry runs never fire it: nothing was written, so nothing may be recorded as
    done.

    The optional *progress* callback fires exactly once per image when the pipeline is finally done
    with it: either on first-pass success, or on retry-pass success or failure. First-pass failures
    are silent because they're still pending retry; this guarantees the bar reaches 100% without
    overshooting on flaky files.

    With a *pacer*, the run rests between work windows instead of tagging flat out, so it stays
    inside the machine's power and heat budget (see :mod:`photo_tagger.pacing`). Rests happen at
    photo boundaries only, and ``rested_seconds`` in the totals says how much time they cost.

    With a *session_plan*, the batch is processed one shoot at a time: every photo in a session is
    analyzed first, the session's keywords are harmonized (see :mod:`photo_tagger.sessions`), and
    only then is anything written. Progress and the *on_success* callback still fire once per photo,
    the latter after its write rather than after its analysis.
    """
    usage = _UsageAccumulator()
    successful_files: list[Path] = []
    # A watch shares its pacer across batches, so this batch reports only the rests it took.
    rested_before = pacer.rested_seconds if pacer is not None else 0.0
    rests_before = pacer.rests_taken if pacer is not None else 0

    def _record_success(path: Path) -> None:
        successful_files.append(path)
        # A dry run writes no metadata, so the caller's callback must not fire: the CLI wires it
        # to the --append-to-skip-file appender, and recording previewed photos there would make
        # a later real run silently skip them.
        if on_success is not None and not options.dry_run:
            on_success(path)

    ctx = _BatchContext(
        agent=agent,
        options=options,
        user_prompt=user_prompt,
        usage=usage,
        cache=cache,
        on_success=_record_success,
        on_image_result=on_image_result,
        progress=progress,
        session_plan=session_plan,
        journal=journal,
        pacer=pacer,
    )

    try:
        with managed_helper(None) if workers <= 1 else _no_helper() as et:
            if session_plan is not None:
                outcome = _run_session_passes(image_files, ctx, et=et, workers=workers)
                success, pending, interrupted = (
                    outcome.successes,
                    outcome.retryable,
                    outcome.interrupted,
                )
                unrecoverable = outcome.final_failures
            else:
                success, pending, interrupted = _run_pass(
                    image_files,
                    ctx,
                    retry=False,
                    et=et,
                    workers=workers,
                )
                unrecoverable = []
            if interrupted:
                # Don't retry after the user asked us to stop. Mark everything that
                # was still pending as failed so the summary file reflects reality.
                retry_successes = 0
                still_failing = pending
            else:
                retry_successes, still_failing = _run_retry_phase(
                    pending,
                    ctx,
                    et=et,
                    workers=workers,
                )
    finally:
        # Waiting for the next batch of a watch is not work.
        if pacer is not None:
            pacer.pause()

    # Session-mode write failures never entered the retry pass, so fold them in here.
    still_failing = still_failing + unrecoverable
    totals = BatchTotals(
        total_files=len(image_files),
        success=success + retry_successes,
        initial_failures=len(pending) + len(unrecoverable),
        retry_successes=retry_successes,
        failed_files=[str(path) for path in still_failing],
        successful_files=[str(path) for path in successful_files],
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        inference_seconds=round(usage.inference_seconds, 3),
        inference_calls=usage.inference_calls,
        cache_hits=usage.cache_hits,
        workers=workers,
        dry_run=options.dry_run,
        failure_kinds=dict(usage.failure_kinds),
        vocabulary_mapped=usage.vocabulary_mapped,
        vocabulary_dropped=dict(usage.vocabulary_dropped),
        flagged_files=sorted(str(path) for path in usage.flagged_for_review),
        language_repairs=usage.language_repairs,
        rested_seconds=round(pacer.rested_seconds - rested_before, 1) if pacer is not None else 0.0,
        rests_taken=pacer.rests_taken - rests_before if pacer is not None else 0,
    )

    _log_totals(totals, workers=workers)

    if on_complete is not None:
        try:
            on_complete(totals)
        except Exception as exc:  # noqa: BLE001 - summary writers must not abort the run.
            logger.exception("on_complete_callback_failed", error=str(exc))

    if totals.success < len(image_files):
        raise BatchError(totals)
    return totals
