"""
The telemetry events: the closed list of what a beacon can carry, and nothing that sends it.

Each dataclass is one event the CLI or the GUI reports. Their fields, with the envelope and the
payload builders in :mod:`photo_tagger.telemetry`, are everything that leaves the machine, so a new
field needs a line in docs/telemetry.md.
"""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar

from photo_tagger.config import (
    DEFAULT_BATTERY_FLOOR,
    DEFAULT_BATTERY_RESUME,
    DEFAULT_DIMENSIONS,
    DEFAULT_DUTY_PERCENT,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_PAUSE_MINUTES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRIES,
    DEFAULT_SETTLE_SECONDS,
    DEFAULT_SIDECAR_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORK_MINUTES,
)
from photo_tagger.telemetry_facts import (
    ENDPOINT_LOCALHOST,
    extension_list,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


# Event names.
EVENT_RUN = "run"


EVENT_CRASH = "crash"


EVENT_SESSION = "session"


EVENT_COMMAND = "command"


EVENT_LAUNCH = "launch"


# How a run or command ended.
OUTCOME_OK = "ok"


OUTCOME_PARTIAL = "partial"  # some photos failed


OUTCOME_FAILED = "failed"  # every photo failed, or the command exited non-zero


OUTCOME_CANCELLED = "cancelled"  # stopped from the GUI


OUTCOME_INTERRUPTED = "interrupted"  # Ctrl-C


OUTCOME_EMPTY = "empty"  # nothing left to do after the filters


OUTCOME_SETUP_FAILED = "setup-failed"  # stopped before the first photo


OUTCOME_USAGE_ERROR = "usage-error"  # a bad flag or value on the command line


# Longest string a command detail may carry.
_MAX_DETAIL_CHARS = 200


# The GUI features a session counts (see gui_usage). Only these names are ever sent.
GUI_FEATURES = frozenset(
    {
        "about",
        "add_folder",
        "add_photos",
        "autosave",
        "build_vocabulary",
        "cache_toggle",
        "cancel",
        "choose_vocabulary",
        "clear_list",
        "connection",
        "copyright",
        "docs",
        "drop",
        "edit_config",
        "export_csv",
        "generate",
        "generate_one",
        "go_back",
        "go_forward",
        "go_up",
        "grid",
        "grid_filter",
        "grid_sort",
        "harmonize_now",
        "hint_enter",
        "keyword_changes",
        "keyword_rules",
        "notify_toggle",
        "open_logs",
        "output_language",
        "pacing",
        "refresh_models",
        "remove",
        "repair_toggle",
        "retry_failed",
        "reveal",
        "save",
        "save_defaults",
        "save_one",
        "save_options",
        "search",
        "select",
        "skip_cache",
        "skip_file",
        "test_connection",
        "ui_language",
        "uncheck_tagged",
        "undo",
        "undo_log_toggle",
        "undo_preview",
        "view_mode",
        "viewer",
        "vocabulary_strict",
        "session_gap",
        "watch_start",
        "watch_stop",
    },
)


# The facts a command may add with note() (see CommandInfo). Only these names are ever sent.
COMMAND_DETAILS = frozenset(
    {
        "checks_failed",
        "entries",
        "failed",
        "keywords_dropped",
        "keywords_kept",
        "keywords_read",
        "left_alone",
        "photos",
        "problems",
        "put_back",
        "source",
        "written",
    },
)


@dataclass(slots=True, frozen=True)
class Settings:
    """
    What a run was set up to do, as bounded values: a switch, a number, or a mode name.

    Paths and free text never appear: a vocabulary file becomes "one is set" and its term count, a
    hint becomes its length. The defaults are the built-in ones, which is what lets a run report
    which settings a user changed.
    """

    # The model server: where it runs (see telemetry_facts.endpoint_kind), never its URL.
    endpoint: str = ENDPOINT_LOCALHOST
    secure: bool = False  # https
    custom_url: bool = False
    api_key: bool = False
    retries: int = DEFAULT_RETRIES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    # What the model is asked.
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    jpeg_dimensions: int = DEFAULT_DIMENSIONS
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    repair_language: bool = False
    hint: bool = False
    hint_chars: int = 0
    custom_prompt: bool = False
    workers: int = 1
    cache: bool = False
    # Keywords.
    max_keywords: int = 0  # 0: no cap
    vocabulary: bool = False
    vocabulary_terms: int = 0
    vocabulary_strict: bool = False
    session_gap_minutes: float = 0.0
    # What a save writes, and where.
    sidecar_mode: str = DEFAULT_SIDECAR_MODE
    sidecar_overrides: int = 0  # how many file types --sidecar-override sends elsewhere
    write_title: bool = True
    write_description: bool = True
    write_keywords: bool = True
    write_copyright: bool = True
    overwrite_title: bool = True
    overwrite_description: bool = True
    overwrite_keywords: bool = False
    backup: bool = True
    undo_log: bool = True
    dry_run: bool = False
    # Copyright: which fields are filled in, never what they say.
    artist: bool = False
    copyright_notice: bool = False
    rights: bool = False
    marked: str = "unset"  # unset | copyrighted | public-domain
    overwrite_copyright: bool = False  # replace a name or notice a photo already has
    set_tags: int = 0
    # Pacing.
    paced: bool = False
    duty: int = DEFAULT_DUTY_PERCENT
    battery_floor: int = DEFAULT_BATTERY_FLOOR
    battery_resume: int = DEFAULT_BATTERY_RESUME
    work_minutes: float = DEFAULT_WORK_MINUTES
    max_pause_minutes: float = DEFAULT_MAX_PAUSE_MINUTES
    # Alerts.
    notify: bool = False
    notify_sound: bool = False
    # Which photos a run takes. Extensions outside the known image formats are counted as "other".
    extensions: str = "cr3,jpg"
    recursive: bool = False
    skip_tagged: bool = False
    date_filter: bool = False
    skip_list: bool = False
    resume_list: bool = False
    # Reports (CLI).
    summary_file: bool = False
    csv_report: bool = False
    json_output: bool = False
    progress_bar: bool = True
    lock_file: bool = False
    otel: bool = False
    otel_content: bool = False
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    # Folder watching.
    watch_interval: float = DEFAULT_INTERVAL_SECONDS
    watch_settle: float = DEFAULT_SETTLE_SECONDS
    # GUI only.
    autosave: bool = False


@dataclass(slots=True, frozen=True)
class Setup:
    """How the settings were given: the config file, the command line, the environment."""

    config_file: str = "none"  # none | env | cwd | user: which config file, never its path
    config_keys: str = ""  # the known "table.key" names the file sets, comma-joined
    unknown_config_keys: int = 0  # keys photo-tagger ignores: counted, never named
    cli_flags: str = ""  # the options typed on the command line, never their values
    env_vars: str = ""  # the photo-tagger variables set, never their values
    changed: str = ""  # the Settings fields that differ from the (env-moved) defaults


@dataclass(slots=True, frozen=True)
class GuiJob:
    """What one GUI generation or save involved beyond the counts every run reports."""

    use_cache: bool = True
    autosave: bool = False  # Save After Generating wrote the photos without review
    hinted: int = 0  # photos generated with a hint
    regenerated_flagged: int = 0  # photos asked again because their text mixed writing systems
    retry_pass: int = 0  # photos the automatic retry pass ran again
    retry_pass_recovered: int = 0
    mixed_pass: int = 0  # photos the automatic mixed-script pass ran again
    mixed_pass_fixed: int = 0
    harmonized: int = 0  # photos whose keywords shoot harmonization changed
    saved: int = 0  # photos the chained save wrote
    save_failures: int = 0
    generate_seconds: float = 0.0
    save_seconds: float = 0.0
    attended_seconds: float = 0.0  # time with keyboard or mouse activity while it ran


@dataclass(slots=True, frozen=True)
class ReviewStats:
    """What happened between generating photos and saving them: edits, hints, and waiting time."""

    saved: int = 0
    edited: int = 0  # photos with any manual edit to the proposal
    edited_title: int = 0
    edited_description: int = 0
    edited_keywords: int = 0
    keywords_added: int = 0
    keywords_removed: int = 0
    hinted: int = 0
    flagged: int = 0  # saved while the text still mixed writing systems
    from_cache: int = 0
    manual_only: int = 0  # saved without ever being generated: the GUI as a plain editor
    copyright_only: int = 0  # copyright fields stamped on a photo that was never generated
    review_seconds_median: float = 0.0  # from proposal to save
    review_seconds_max: float = 0.0


@dataclass(slots=True, frozen=True)
class Display:
    """The GUI's toolkit and screen: versions, the primary screen's size, the color scheme."""

    qt_version: str = ""
    pyside_version: str = ""
    screen: str = ""  # the primary screen in logical pixels and its scale, e.g. "1512x982@2"
    screens: int = 0
    color_scheme: str = ""  # dark | light
    window: str = ""  # the window's size, or maximized / full-screen


@dataclass(slots=True, frozen=True)
class RunInfo:
    """
    One run: a CLI batch (``tag``, or one ``watch`` batch), or one GUI generation or save.

    Everything else in the beacon (platform, hardware, version, install id) is filled in by
    :func:`build_payload`, so callers only describe what they alone know.
    """

    interface: str  # "cli" or "gui"
    provider: str
    model: str
    batch_size: int
    duration_seconds: float  # the work itself, from the first photo to the last, rests included
    output_language: str  # the language the model writes metadata in, e.g. "English"
    ui_language: str  # the resolved app UI language code, e.g. "en" or "pt_BR"
    file_types: str  # distinct extensions in the batch, sorted and comma-joined, e.g. "cr3,jpg"
    command: str = "tag"  # tag | watch | gui
    job: str = "tag"  # tag (a CLI batch) | generate | save (a GUI run)
    trigger: str = ""  # GUI: batch | single | hint | retry | watch
    outcome: str = OUTCOME_OK
    error_kind: str = ""  # the exception class that stopped a run before its first photo
    success_count: int = 0  # photos that ended up written (or previewed, on a dry run)
    failure_count: int = 0  # photos still failing after the retry pass
    initial_failures: int = 0  # photos that failed their first attempt
    cache_hits: int = 0  # photos answered from the inference cache instead of the model
    retry_successes: int = 0  # photos recovered by the retry pass
    workers: int = 0  # thread-pool width the batch ran with (CLI only)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_calls: int = 0
    inference_seconds: float = 0.0  # time spent inside model calls (vs wall-clock duration)
    dry_run: bool = False
    failure_kinds: str = ""  # compact final-failure buckets, e.g. "timeout:3,metadata-write:1"
    failure_types: str = ""  # the exception behind each final failure, e.g. "ModelHTTPError(404):3"
    language_repairs: int = 0  # extra calls that rewrote mixed-script text
    flagged: int = 0  # photos whose text still mixes writing systems
    vocabulary_mapped: int = 0
    vocabulary_dropped: int = 0  # distinct terms strict mode dropped
    rested_seconds: float = 0.0  # pacing rests
    rests_taken: int = 0
    shoots: int = 0  # sessions the batch was split into (--session-gap)
    photo_seconds_median: float = 0.0  # model time per photo
    photo_seconds_p90: float = 0.0
    keywords_mean: float = 0.0  # generated keywords per photo
    title_chars_mean: float = 0.0
    description_chars_mean: float = 0.0
    settings: Settings | None = None
    setup: Setup | None = None
    gui: GuiJob | None = None
    review: ReviewStats | None = None

    EVENT: ClassVar[str] = EVENT_RUN

    def body(self) -> dict[str, object]:
        """Return the run's own payload fields (build_payload adds the power source)."""
        fields = _body(self, drop=_ENVELOPE_FIELDS)
        # The Worker's AE column reads a number: a v2 Worker still deployed must see 0 or 1.
        fields["dry_run"] = int(self.dry_run)
        return fields


@dataclass(slots=True, frozen=True)
class SessionInfo:
    """
    One GUI window from open to close, or one ``watch`` from start to stop.

    The time splits into busy (a run in flight), engaged (keyboard or mouse activity), and idle
    (neither), so a window left open overnight reads as the idle hours it was.
    """

    interface: str
    command: str
    session_seconds: float
    provider: str = ""
    model: str = ""
    output_language: str = ""
    ui_language: str = ""
    busy_seconds: float = 0.0
    engaged_seconds: float = 0.0
    idle_seconds: float = 0.0
    unattended_seconds: float = 0.0  # busy with nobody at the keyboard
    runs: int = 0
    save_runs: int = 0
    single_saves: int = 0
    first_run_seconds: float = -1.0  # from opening to the first run; -1 when there was none
    photos_added: int = 0
    photos_generated: int = 0
    photos_saved: int = 0
    photos_unsaved: int = 0  # proposals still unsaved at the end
    closed_with_unsaved: bool = False
    view_mode: str = ""
    settings: Settings | None = None
    setup: Setup | None = None
    review: ReviewStats | None = None
    display: Display | None = None
    features: Mapping[str, int] = field(default_factory=dict)  # GUI_FEATURES -> uses

    EVENT: ClassVar[str] = EVENT_SESSION

    def body(self) -> dict[str, object]:
        """Return the session's own payload fields, with only allowlisted feature names."""
        fields = _body(self, drop=_ENVELOPE_FIELDS | {"features"})
        fields["features"] = _features(self.features)
        return fields


@dataclass(slots=True, frozen=True)
class CommandInfo:
    """One CLI command that is not a run: show, set, vocabulary, undo, doctor, or a failed start."""

    command: str
    outcome: str
    exit_code: int
    duration_seconds: float
    error_kind: str = ""
    setup: Setup | None = None
    details: Mapping[str, str | float] = field(default_factory=dict)
    interface: str = "cli"

    EVENT: ClassVar[str] = EVENT_COMMAND

    def body(self) -> dict[str, object]:
        """Return the command's own payload fields, with only allowlisted details."""
        fields = _body(self, drop=_ENVELOPE_FIELDS | {"details"})
        fields["details"] = _details(self.details)
        return fields


@dataclass(slots=True, frozen=True)
class LaunchInfo:
    """The GUI window opening: how long it took, and the setup it opened with."""

    startup_seconds: float
    settings: Settings | None = None
    setup: Setup | None = None
    display: Display | None = None
    interface: str = "gui"
    command: str = "gui"

    EVENT: ClassVar[str] = EVENT_LAUNCH

    def body(self) -> dict[str, object]:
        """Return the launch's own payload fields."""
        return _body(self, drop=_ENVELOPE_FIELDS)


type Event = RunInfo | SessionInfo | CommandInfo | LaunchInfo


def outcome_for(total: int, failed: int, *, interrupted: bool = False) -> str:
    """Name how a batch of *total* photos ended with *failed* of them still failing."""
    if interrupted:
        return OUTCOME_INTERRUPTED
    if total == 0:
        return OUTCOME_EMPTY
    if failed == 0:
        return OUTCOME_OK
    return OUTCOME_FAILED if failed >= total else OUTCOME_PARTIAL


def failure_kinds_summary(kinds: Mapping[str, int]) -> str:
    """
    Encode per-run failure counts as a compact, sorted string: ``"timeout:3,other:1"``.

    The keys are the pipeline's fixed coarse buckets (timeout, connection, model-validation, model-
    api, image-read, metadata-write, other) or exception class names with an HTTP status, never
    message text, so nothing user-derived can ride along.
    """
    return ",".join(f"{kind}:{count}" for kind, count in sorted(kinds.items()) if count)


def file_types_summary(paths: Iterable[Path]) -> str:
    """
    Summarize a batch as its distinct file extensions: lowercased, dot-stripped, comma-joined.

    ``{"a.CR3", "b.jpg", "c.cr3"}`` becomes ``"cr3,jpg"``. Only known image formats are named; any
    other extension becomes "other", so not even an unusual suffix leaves the machine.
    """
    return extension_list(path.suffix for path in paths)


def _compact(value: object) -> object:
    """Round floats to three decimals, all the way down, so beacons stay small."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _compact(inner) for key, inner in value.items() if inner is not None}
    return value


def _body(event: Event, *, drop: frozenset[str]) -> dict[str, object]:
    """Turn *event* into payload fields, leaving out the envelope's and the empty groups."""
    return {
        key: _compact(value)
        for key, value in asdict(event).items()
        if key not in drop and value is not None
    }


def _features(counts: Mapping[str, int]) -> dict[str, int]:
    """Keep only the allowlisted feature names, so nothing else can ride along."""
    return {name: int(uses) for name, uses in counts.items() if name in GUI_FEATURES and uses}


def _details(facts: Mapping[str, object]) -> dict[str, object]:
    """Keep only allowlisted command facts with scalar values; clamp strings."""
    kept: dict[str, object] = {}
    for name, value in facts.items():
        if name not in COMMAND_DETAILS:
            continue
        if isinstance(value, str):
            kept[name] = value[:_MAX_DETAIL_CHARS]
        elif isinstance(value, bool | int | float):
            kept[name] = value
    return kept


_ENVELOPE_FIELDS = frozenset({"interface", "command"})
