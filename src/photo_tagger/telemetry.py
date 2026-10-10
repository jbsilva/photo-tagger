"""
Anonymous, opt-out usage telemetry.

Small JSON beacons tell the project how photo-tagger is used: which models, settings, and features
people pick, how runs go, and on what machines. One is sent at the end of each run, for each other
command, when a GUI window opens and closes, and when the app crashes. It is deliberately
privacy-preserving:

- **No personal or content data.** File paths, filenames, photo pixels, generated tags, titles,
  descriptions, prompts, hints, API keys, and server URLs are never collected. A setting is sent as
  a bounded value (a switch, a count, a mode name), never as the text or path it holds. Crash
  beacons carry the exception *type* and its stack trace as code locations (module, function, and
  line), never the message. The dataclasses below and :func:`build_payload` are the closed set of
  what leaves the machine.
- **Opt-out, three ways.** Set ``PHOTO_TAGGER_TELEMETRY=0`` (or the cross-tool ``DO_NOT_TRACK=1``)
  in the environment, pass ``--no-telemetry``, or put ``enabled = false`` under ``[telemetry]`` in
  the config file. Any opt-out beats any opt-in, except that ``PHOTO_TAGGER_TELEMETRY=1`` overrules
  the cross-tool ``DO_NOT_TRACK`` (see :func:`resolve_consent`).
- **Never in the way.** Sending happens on a daemon thread with a short timeout, and every failure
  is swallowed. Telemetry must never crash, slow, or block a tagging run.

The collector is our own Cloudflare Worker at ``telemetry.tagger.photo`` (its source lives in the
``telemetry/`` directory of this repo). No third-party analytics service is involved.

The per-machine ``install_id`` is a random UUID generated once and stored in the user state
directory. It is **not** derived from any hardware identifier, so it cannot be used to fingerprint a
machine; it only lets repeated runs from the same install be counted as one active user over time.
The ``session_id`` is a random UUID per process. It is never written to disk, so it only ties
together the beacons one launch of the app sends.
"""

import os
import platform
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar

import httpx2
from loguru import logger

from photo_tagger import __version__
from photo_tagger.config import (
    DEFAULT_BATTERY_FLOOR,
    DEFAULT_BATTERY_RESUME,
    DEFAULT_DIMENSIONS,
    DEFAULT_DUTY_PERCENT,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_PAUSE_MINUTES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRIES,
    DEFAULT_SIDECAR_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WORK_MINUTES,
    state_dir,
)
from photo_tagger.hardware import hardware_info
from photo_tagger.pacing import read_battery
from photo_tagger.telemetry_facts import (
    ENDPOINT_LOCALHOST,
    MAX_STACK_FRAMES,
    attached_terminal,
    exception_chain,
    extension_list,
    install_method,
    installer,
    language_label,
    library_versions,
    local_clock,
    model_label,
    running_in_ci,
    stack_frames,
    system_locale,
)
from photo_tagger.watch import DEFAULT_INTERVAL_SECONDS, DEFAULT_SETTLE_SECONDS


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path


# The collector endpoint (our own Cloudflare Worker). Kept as a module constant so tests can patch
# it and so there is a single place to repoint it.
ENDPOINT = "https://telemetry.tagger.photo"

# Bumped when the payload shape changes. v2 added crash beacons, hardware, and outcome counters; v3
# added the session, command, and launch events, the settings and setup groups, and the GUI's
# per-job runs. The Worker still accepts v1 and v2 from older installs.
SCHEMA_VERSION = 3

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

# Environment switches: our own, which works both ways and speaks for this tool alone, and the
# cross-tool standard from consoledonottrack.com, which our own "on" can overrule.
_ENV_SWITCH = "PHOTO_TAGGER_TELEMETRY"
_ENV_DO_NOT_TRACK = "DO_NOT_TRACK"
# The opt-out of 0.9 and earlier. It still turns telemetry off, so nobody who opted out with it
# starts sending after an upgrade.
_ENV_LEGACY_OPT_OUT = "PHOTO_TAGGER_NO_TELEMETRY"

# The "yes" spellings. An empty value counts as not set. Any other value of our own switch turns
# telemetry off: someone who typed "disabled" or "none" meant no, and an opt-out must fail closed.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Why a beacon is or is not sent.
CONSENT_DEFAULT = "default"  # nobody said anything either way: on, as disclosed
CONSENT_ENV_OPT_IN = "env-opt-in"  # the PHOTO_TAGGER_TELEMETRY switch set to on
CONSENT_GUI_OPT_IN = "gui-opt-in"  # the window's toggle, turned on by hand
CONSENT_ENV_OPT_OUT = "env-opt-out"  # the switch set to off, or 0.9's NO_TELEMETRY switch
CONSENT_SETTING_OFF = "setting-off"  # --no-telemetry, [telemetry] enabled = false, or the toggle
CONSENT_DO_NOT_TRACK = "do-not-track"  # DO_NOT_TRACK=1 and no opt-in for this tool

# Network call budget. A healthy beacon is a few tens of milliseconds; this only bounds the bad case
# (no network, DNS hang) so the call cannot wedge process exit.
_SEND_TIMEOUT_SECONDS = 2.0
# How long a blocking caller (the CLI, which exits right after) waits for the beacon to flush before
# giving up and letting the daemon thread be abandoned at interpreter exit.
_FLUSH_TIMEOUT_SECONDS = 2.5

# State files. The install id, the "first-run notice already shown" marker, the GUI's persisted
# on/off choice, and the hardware-probe cache live under the user state directory, mirroring
# config_file.py's XDG-style location.
_INSTALL_ID_FILE = "install-id"
_NOTICE_MARKER_FILE = "telemetry-notice-shown"
# The GUI's own marker: its dialog lists more (idle time, screen size) and offers Turn It Off, so a
# CLI user who later opens the window still sees it once.
_GUI_NOTICE_MARKER_FILE = "telemetry-notice-shown-gui"
_GUI_PREF_FILE = "gui-telemetry"
# What that file holds: a hand-checked toggle, a cleared one, and the "on" that 0.9 wrote.
_GUI_PREF_OPT_IN = "opt-in"
_GUI_PREF_OFF = "0"
_GUI_PREF_LEGACY_ON = "1"
_HARDWARE_CACHE_FILE = "hardware.json"
_INSTALL_ID_LOCK = threading.Lock()

# Bumped whenever the disclosure below materially changes (new data categories), so existing
# installs see the updated notice once. v2: hardware and crash reports. v3: settings and features,
# where the model server runs, power and battery, locale and time zone, and how long the GUI sits
# idle.
NOTICE_VERSION = 3

# Shown once, to stderr, on the first run telemetry is active. Opt-out tools are expected to
# disclose up front; this is that disclosure.
FIRST_RUN_NOTICE = (
    "photo-tagger sends anonymous usage stats to guide development: the settings and features you "
    "use (as switches, counts, and mode names, never the text or paths in them), the model and "
    "where its server runs (local, your network, or which public service), batch size, file "
    "types, timing, outcomes, OS, CPU/GPU, RAM, power source and battery level, locale, and time "
    "zone, plus crash reports (error "
    "type and stack trace, code locations only).\n"
    "No photos, paths, filenames, tags, titles, descriptions, prompts, URLs, error messages, or "
    "personal data are ever sent. Details: https://jbsilva.github.io/photo-tagger/telemetry/\n"
    "Turn it off with --no-telemetry, PHOTO_TAGGER_TELEMETRY=0, or [telemetry] enabled = false "
    "in your config."
)

# Crash beacons are capped per process so a pathological crash loop (a repeatedly-failing Qt slot)
# cannot spam the collector.
_MAX_CRASH_BEACONS = 3
# How many in-package stack frames a crash beacon carries (deepest last).
_MAX_CRASH_FRAMES = 5
# Longest string a command detail may carry.
_MAX_DETAIL_CHARS = 200

# Random per process, never stored: it ties together the beacons of one launch.
SESSION_ID = str(uuid.uuid4())
_PROCESS_STARTED = time.monotonic()

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

# Filled by the GUI from Qt's own locale, which knows the OS language when a Finder launch leaves
# the environment without LANG.
_locale_hint: str | None = None
# Set once any beacon is on its way, so the CLI sends its fallback command beacon only when nothing
# else reported the invocation.
_sent_any = False
# Facts the running command noted for its beacon (see note()).
_notes: dict[str, str | float] = {}


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
        """Return the run's own payload fields, plus the power source it ended on."""
        fields = _body(self, drop=_ENVELOPE_FIELDS)
        # The Worker's AE column reads a number: a v2 Worker still deployed must see 0 or 1.
        fields["dry_run"] = int(self.dry_run)
        return {**_power(), **fields}


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


def set_locale_hint(name: str | None) -> None:
    """Report the OS locale from another source (the GUI passes Qt's) for later beacons."""
    global _locale_hint  # noqa: PLW0603 - one process-wide hint set once at startup.
    _locale_hint = name


def note(**facts: str | float) -> None:
    """Add facts about the running command to its beacon (only COMMAND_DETAILS names are sent)."""
    _notes.update(facts)


def sent_any() -> bool:
    """Return True once this process has sent (or is sending) any beacon."""
    return _sent_any


def _mark_sent() -> None:
    global _sent_any  # noqa: PLW0603 - deliberate per-process flag.
    _sent_any = True


def mark_reported() -> None:
    """Count this process as reported without sending, so no catch-all beacon follows it."""
    _mark_sent()


def _is_truthy(value: str | None) -> bool:
    """Return True when *value* is one of the accepted "yes" spellings."""
    return value is not None and value.strip().lower() in _TRUTHY


def _is_opt_out(value: str | None) -> bool:
    """Return True when *value* is set, not blank, and not one of the "yes" spellings."""
    return value is not None and bool(value.strip()) and not _is_truthy(value)


@dataclass(slots=True, frozen=True)
class Consent:
    """Whether telemetry may send, and which switch decided it (one of the ``CONSENT_*`` names)."""

    send: bool
    reason: str


def resolve_consent(*, enabled: bool, opted_in: bool = False) -> Consent:
    """
    Decide whether to send, and say why.

    *enabled* is the flag/config (CLI) or toggle (GUI) choice; *opted_in* means the GUI toggle was
    turned on by hand. Any opt-out beats any opt-in, and an opt-in for this tool beats the
    cross-tool ``DO_NOT_TRACK``. The config file's ``enabled = true`` is not an opt-in: it is the
    default, and the GUI writes it on every save.
    """
    switch = os.getenv(_ENV_SWITCH)
    if _is_opt_out(switch) or _is_truthy(os.getenv(_ENV_LEGACY_OPT_OUT)):
        return Consent(send=False, reason=CONSENT_ENV_OPT_OUT)
    if not enabled:
        return Consent(send=False, reason=CONSENT_SETTING_OFF)
    if _is_truthy(switch):
        return Consent(send=True, reason=CONSENT_ENV_OPT_IN)
    if opted_in:
        return Consent(send=True, reason=CONSENT_GUI_OPT_IN)
    if do_not_track():
        return Consent(send=False, reason=CONSENT_DO_NOT_TRACK)
    return Consent(send=True, reason=CONSENT_DEFAULT)


def do_not_track() -> bool:
    """Return True when the cross-tool ``DO_NOT_TRACK`` switch is set."""
    return _is_truthy(os.getenv(_ENV_DO_NOT_TRACK))


def should_send(*, config_enabled: bool, opted_in: bool = False) -> bool:
    """Resolve whether to send, given the already-merged flag/config value (see resolve_consent)."""
    return resolve_consent(enabled=config_enabled, opted_in=opted_in).send


def _state_dir() -> Path:
    """Return the shared state directory telemetry keeps its install id and markers in."""
    return state_dir()


def install_id() -> str:
    """
    Return this install's stable anonymous id, creating it on first use.

    The id is a random UUID4 persisted under the state directory. If it cannot be read or written
    (read-only home, permission denied), a fresh ephemeral id is returned instead so the run is
    still counted; only the across-runs "same install" linkage is lost in that rare case.

    Two beacons of one first run are sent from two threads, and a CLI and a window can start at the
    same moment, so the id is minted under a lock and created exclusively: whoever loses the race
    reads the winner's id instead of writing its own.
    """
    with _INSTALL_ID_LOCK:
        path = _state_dir() / _INSTALL_ID_FILE
        if (existing := _read_install_id(path)) is not None:
            return existing
        new_id = str(uuid.uuid4())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _create_whole(path, new_id + "\n")
        except FileExistsError:
            return _read_install_id(path) or _replace_install_id(path, new_id)
        except OSError as exc:
            logger.debug("telemetry_install_id_persist_failed", error=str(exc))
        return new_id


def _create_whole(path: Path, text: str) -> None:
    """
    Create *path* holding *text*, or raise FileExistsError; never a moment with the file empty.

    Opening exclusively and then writing leaves an instant where another process finds the file
    empty, reads it as corrupt, and writes its own id. Writing a private file first and linking it
    into place makes the file appear whole. A filesystem without hard links (exFAT, some SMB shares)
    falls back to the exclusive create: a rare race beats a new install id on every run.
    """
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    try:
        os.link(temp, path)
    except FileExistsError:
        raise
    except OSError:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    finally:
        temp.unlink(missing_ok=True)


def _read_install_id(path: Path) -> str | None:
    """Return the id stored at *path*, or None when it is missing, empty, or corrupt."""
    try:
        return str(uuid.UUID(path.read_text(encoding="utf-8").strip()))
    except OSError, ValueError:
        return None


def _replace_install_id(path: Path, new_id: str) -> str:
    """Overwrite a corrupt id file with *new_id*, and return it."""
    try:
        path.write_text(new_id + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_install_id_persist_failed", error=str(exc))
    return new_id


def _notice_marker(interface: str) -> Path:
    """Return the marker file that records the notice *interface* ("cli" or "gui") has shown."""
    return _state_dir() / (_GUI_NOTICE_MARKER_FILE if interface == "gui" else _NOTICE_MARKER_FILE)


def notice_due(*, interface: str = "cli") -> bool:
    """Say whether *interface* has yet to show the current notice, without marking it shown."""
    try:
        return _notice_marker(interface).read_text(encoding="utf-8").strip() != str(NOTICE_VERSION)
    except OSError:
        return True


def first_run_notice(*, interface: str = "cli") -> str | None:
    """
    Return :data:`FIRST_RUN_NOTICE` the first time telemetry is active, else ``None``.

    A marker file records the :data:`NOTICE_VERSION` that has been shown, so the notice appears once
    per disclosure version: when new data categories are added, existing installs see the updated
    text once too. The marker is written before returning, so two calls in one process do not both
    show it. If the marker cannot be written the notice may reappear on a later run, which is
    harmless.
    """
    if not notice_due(interface=interface):
        return None
    marker = _notice_marker(interface)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{NOTICE_VERSION}\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_notice_marker_failed", error=str(exc))
    return FIRST_RUN_NOTICE


def read_gui_pref() -> bool | None:
    """
    Return the GUI's persisted telemetry choice, or ``None`` if the user never made one.

    The GUI's Settings toggle writes this so the choice survives restarts. ``None`` means "no
    explicit choice", so the GUI falls back to the config-file default. Anything unreadable or
    unrecognized is treated as no choice.
    """
    text = _read_gui_pref_text()
    if text in {_GUI_PREF_OPT_IN, _GUI_PREF_LEGACY_ON}:
        return True
    if text == _GUI_PREF_OFF:
        return False
    return None


def _read_gui_pref_text() -> str | None:
    """Return the raw content of the GUI's preference file, or None when there is none."""
    try:
        return (_state_dir() / _GUI_PREF_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def write_gui_pref(*, enabled: bool) -> None:
    """Persist the GUI's telemetry choice so the Settings toggle sticks across restarts."""
    path = _state_dir() / _GUI_PREF_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_GUI_PREF_OPT_IN if enabled else _GUI_PREF_OFF, encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_gui_pref_persist_failed", error=str(exc))


def gui_choice(*, config_enabled: bool) -> tuple[bool, bool]:
    """
    Return the GUI's ``(enabled, opted_in)`` pair for :func:`resolve_consent`.

    A cleared toggle turns telemetry off, and so does ``enabled = false`` in the config file until
    the toggle is checked by hand: the window writes that key itself on every settings save, so
    the later click is the newer choice. Only a toggle checked by hand since 0.10 is an opt-in that
    overrules ``DO_NOT_TRACK``: 0.9 wrote "1" for any check, even one made while ``DO_NOT_TRACK``
    kept the beacons off.
    """
    pref = _read_gui_pref_text()
    if pref == _GUI_PREF_OPT_IN:
        return True, True
    return config_enabled and pref != _GUI_PREF_OFF, False


def _envelope(event: str, *, interface: str, command: str, consent: str) -> dict[str, object]:
    """Collect the facts every beacon carries: identity, version, platform, and environment."""
    hardware = hardware_info(_state_dir() / _HARDWARE_CACHE_FILE)
    utc_offset, hour, weekday = local_clock()
    return {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "install_id": install_id(),
        "session_id": SESSION_ID,
        "app_version": __version__,
        "interface": interface,
        "command": command,
        "consent": consent,
        "uptime_seconds": round(time.monotonic() - _PROCESS_STARTED, 1),
        "arch": platform.machine(),  # e.g. "arm64", "x86_64", "aarch64"
        "os": platform.system(),  # "Darwin", "Linux", "Windows"
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu": hardware.cpu,  # model string, e.g. "Apple M3 Pro"; "" when unknown
        "gpu": hardware.gpu,  # model string, e.g. "NVIDIA GeForce RTX 4070"; "" when unknown
        "cpu_count": hardware.cpu_count,
        "memory_gb": hardware.memory_gb,
        "install_method": install_method(),
        "installer": installer(),
        "system_locale": system_locale(_locale_hint),
        "utc_offset_minutes": utc_offset,
        "local_hour": hour,
        "local_weekday": weekday,
        "ci": running_in_ci(),
        "terminal": attached_terminal(),
    }


def _power() -> dict[str, object]:
    """Say whether the machine runs on a charger or its battery, for the pacing feature."""
    battery = read_battery()
    if battery is None:
        return {"power": "none", "battery_percent": -1}
    return {"power": "ac" if battery.on_ac else "battery", "battery_percent": battery.percent}


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


def build_payload(event: Event, *, consent: str = CONSENT_DEFAULT) -> dict[str, object]:
    """
    Build the exact beacon body for *event*, plus the locally-derived envelope.

    This (together with :func:`build_crash_payload`) is the single source of truth for what leaves
    the machine. Every value is a bounded name, a count, a duration, a switch, a coarse hardware
    model string, or the random ids. Nothing derived from the user's photos, paths, or identity
    appears, by construction.
    """
    envelope = _envelope(
        event.EVENT,
        interface=event.interface,
        command=event.command,
        consent=consent,
    )
    payload = {**envelope, **event.body()}
    # Both are typed by the user or chosen by a server, so they are reduced here, where every
    # beacon passes, rather than at each place that fills them in.
    if isinstance(model := payload.get("model"), str):
        payload["model"] = model_label(model)
    if isinstance(language := payload.get("output_language"), str) and language:
        payload["output_language"] = language_label(language)
    return payload


def crash_summary(exc: BaseException) -> tuple[str, str, str]:
    """
    Reduce *exc* to ``(exception_type, location, frames)`` for photo-tagger's own frames.

    Each frame renders as ``module:function:line`` (e.g. ``pipeline:run_batch:714``), *location* is
    the deepest one, and *frames* joins the last few with ``>``. The whole stack, library frames
    included, goes in ``crash_stack`` (see :func:`build_crash_payload`). The exception **message is
    deliberately dropped**: messages routinely embed file paths and other user data.
    """
    frames = [
        f"{module.removeprefix('photo_tagger.')}:{function}:{line}"
        for module, function, line in stack_frames(exc.__traceback__)
        if module.startswith("photo_tagger.")
    ]
    kept = frames[-_MAX_CRASH_FRAMES:]
    return (type(exc).__name__, kept[-1] if kept else "", ">".join(kept))


def build_crash_payload(
    exc: BaseException,
    *,
    interface: str,
    command: str = "",
    consent: str = CONSENT_DEFAULT,
) -> dict[str, object]:
    """
    Build the crash-beacon body: the envelope plus the exception and where it was raised.

    The stack names every frame by module, function, and line (never a file path), the deepest
    :data:`~photo_tagger.telemetry_facts.MAX_STACK_FRAMES` of them. The chain names the exceptions
    behind this one, and the library versions tell which release of a dependency was running.
    """
    exception_type, location, frames = crash_summary(exc)
    stack = [
        f"{module}:{function}:{line}" for module, function, line in stack_frames(exc.__traceback__)
    ]
    return {
        **_envelope(EVENT_CRASH, interface=interface, command=command, consent=consent),
        "exception_type": exception_type,
        "exception_module": type(exc).__module__,
        "exception_chain": exception_chain(exc),
        "crash_location": location,
        "crash_frames": frames,
        "crash_stack": ">".join(stack[-MAX_STACK_FRAMES:]),
        "libraries": library_versions(),
    }


def _safe_post(payload: dict[str, object]) -> None:
    """POST *payload* to the collector, swallowing every error (telemetry is never load-bearing)."""
    try:
        httpx2.post(ENDPOINT, json=payload, timeout=_SEND_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - a beacon must never surface a failure to the user.
        logger.debug("telemetry_send_failed", error=str(exc))


def _build_and_send(builder: Callable[[], dict[str, object]], *, failure_event: str) -> None:
    """Build the payload and POST it, entirely on the calling (background) thread."""
    try:
        payload = builder()
    except Exception as exc:  # noqa: BLE001 - building the payload must not crash the caller.
        logger.debug(failure_event, error=str(exc))
        return
    _safe_post(payload)


def _dispatch(
    builder: Callable[[], dict[str, object]],
    *,
    block: bool,
    failure_event: str,
) -> threading.Thread:
    """
    Start the daemon thread that builds *and* sends the payload, optionally waiting for it.

    The builder (which probes hardware on a cache miss, see :mod:`photo_tagger.hardware`) runs
    inside this thread rather than before it starts: a slow or hung probe must delay the beacon,
    never the tagging run that is about to exit.
    """
    _mark_sent()
    thread = threading.Thread(
        target=_build_and_send,
        args=(builder,),
        kwargs={"failure_event": failure_event},
        name="photo-tagger-telemetry",
        daemon=True,
    )
    thread.start()
    if block:
        thread.join(timeout=_FLUSH_TIMEOUT_SECONDS)
    return thread


def emit(
    event: Event,
    *,
    enabled: bool,
    opted_in: bool = False,
    block: bool = False,
) -> threading.Thread | None:
    """
    Fire one telemetry beacon for *event* on a background daemon thread.

    Returns the thread (or ``None`` when telemetry is disabled), regardless of whether building the
    payload later succeeds inside that thread. Pass ``block=True`` to wait up to
    :data:`_FLUSH_TIMEOUT_SECONDS` for delivery. Both the CLI (which exits right after the batch)
    and the GUI (whose process exits right after ``closeEvent``) use it: a daemon thread abandoned
    at interpreter exit is killed mid-send and the beacon is lost.
    """
    consent = resolve_consent(enabled=enabled, opted_in=opted_in)
    if not consent.send:
        return None
    return _dispatch(
        lambda: build_payload(event, consent=consent.reason),
        block=block,
        failure_event="telemetry_payload_failed",
    )


def flush(threads: Iterable[threading.Thread | None]) -> None:
    """Wait for beacons already on their way, within one flush budget for all of them."""
    deadline = time.monotonic() + _FLUSH_TIMEOUT_SECONDS
    for thread in threads:
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))


def command_notes() -> dict[str, str | float]:
    """Return what the running command noted for its beacon (see :func:`note`)."""
    return dict(_notes)


# Crash beacons sent by this process, capped at _MAX_CRASH_BEACONS (see emit_crash).
_crash_beacons_sent = 0


def emit_crash(  # noqa: PLR0913 - the crash plus where it happened plus the consent inputs.
    exc: BaseException,
    *,
    interface: str,
    command: str = "",
    enabled: bool,
    opted_in: bool = False,
    block: bool = True,
) -> threading.Thread | None:
    """
    Fire one anonymous crash beacon for *exc*; never raises.

    Honors the same opt-outs as :func:`emit` and defaults to ``block=True`` because the process is
    usually about to die when this is called. At most :data:`_MAX_CRASH_BEACONS` are sent per
    process so a crash loop cannot spam the collector; the cap is charged per attempt rather than
    per successful build, so it also bounds a crash loop that fails inside the payload builder.
    """
    global _crash_beacons_sent  # noqa: PLW0603 - deliberate per-process counter.
    consent = resolve_consent(enabled=enabled, opted_in=opted_in)
    if _crash_beacons_sent >= _MAX_CRASH_BEACONS or not consent.send:
        return None
    _crash_beacons_sent += 1
    return _dispatch(
        lambda: build_crash_payload(
            exc,
            interface=interface,
            command=command,
            consent=consent.reason,
        ),
        block=block,
        failure_event="telemetry_crash_payload_failed",
    )
