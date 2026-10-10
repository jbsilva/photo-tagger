"""
Anonymous, opt-out usage telemetry.

One small JSON beacon is sent at the end of a run so the project can see which models, platforms,
and hardware people actually use, and one is sent when the app crashes so breakage is visible
without waiting for bug reports. It is deliberately minimal and privacy-preserving:

- **No personal or content data.** File paths, filenames, photo pixels, generated tags, titles,
  descriptions, prompts, and API keys are never collected. Crash beacons carry only the exception
  *type* and code locations inside this package, never the exception message (messages can embed
  paths). See :func:`build_payload` and :func:`crash_summary` for the exact, closed set of fields
  that leave the machine.
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
"""

import os
import platform
import threading
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
from loguru import logger

from photo_tagger import __version__
from photo_tagger.config import state_dir
from photo_tagger.hardware import hardware_info


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# The collector endpoint (our own Cloudflare Worker). Kept as a module constant so tests can patch
# it and so there is a single place to repoint it.
ENDPOINT = "https://telemetry.tagger.photo"

# Bumped only when the payload shape changes, so the Worker can branch on it if it ever needs to.
# v2 added the event field ("run" | "crash"), hardware facts, and per-run outcome counters; the
# Worker still accepts v1 from older installs.
SCHEMA_VERSION = 2

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
_GUI_PREF_FILE = "gui-telemetry"
# What that file holds: a hand-checked toggle, a cleared one, and the "on" that 0.9 wrote.
_GUI_PREF_OPT_IN = "opt-in"
_GUI_PREF_OFF = "0"
_GUI_PREF_LEGACY_ON = "1"
_HARDWARE_CACHE_FILE = "hardware.json"

# Bumped whenever the disclosure below materially changes (new data categories), so existing
# installs see the updated notice once. v2: hardware facts and crash reports were added.
NOTICE_VERSION = 2

# Shown once, to stderr, on the first run telemetry is active. Opt-out tools are expected to
# disclose up front; this is that disclosure.
FIRST_RUN_NOTICE = (
    "photo-tagger collects anonymous usage stats (model name, batch size, file types, output and "
    "UI language, OS, CPU/GPU model, RAM size, timing, success/cache counts) plus anonymous crash "
    "reports (exception type and in-app code location only) to guide development.\n"
    "No photos, file paths, filenames, tags, error messages, or personal data are ever sent. See "
    "the Telemetry section of the README.\n"
    "Disable it any time with --no-telemetry, PHOTO_TAGGER_TELEMETRY=0, or [telemetry] "
    "enabled = false in your config."
)

# Crash beacons are capped per process so a pathological crash loop (a repeatedly-failing Qt slot)
# cannot spam the collector.
_MAX_CRASH_BEACONS = 3
# How many in-package stack frames a crash beacon carries (deepest last).
_MAX_CRASH_FRAMES = 5


@dataclass(slots=True, frozen=True)
class RunInfo:
    """
    The caller-supplied facts about one run.

    Everything else in the beacon (platform, hardware, version, install id) is filled in by
    :func:`build_payload`, so callers only describe what they alone know. The outcome fields default
    to zero because the GUI cannot always attribute them per session; the CLI fills them from its
    BatchTotals.
    """

    interface: str  # "cli" or "gui"
    provider: str
    model: str
    batch_size: int
    duration_seconds: float
    output_language: str  # the language the model writes metadata in, e.g. "English"
    ui_language: str  # the resolved app UI language code, e.g. "en" or "pt_BR"
    file_types: str  # distinct extensions in the batch, sorted and comma-joined, e.g. "cr3,jpg"
    success_count: int = 0  # photos that ended up written (or previewed, on a dry run)
    failure_count: int = 0  # photos still failing after the retry pass
    cache_hits: int = 0  # photos answered from the inference cache instead of the model
    retry_successes: int = 0  # photos recovered by the retry pass
    workers: int = 0  # thread-pool width the batch ran with (CLI only)
    total_tokens: int = 0  # tokens across every model call in the run
    inference_seconds: float = 0.0  # time spent inside model calls (vs wall-clock duration)
    dry_run: bool = False  # the run previewed metadata without writing it
    failure_kinds: str = ""  # compact final-failure buckets, e.g. "timeout:3,metadata-write:1"


def failure_kinds_summary(kinds: dict[str, int]) -> str:
    """
    Encode per-run failure buckets as a compact, sorted string: ``"timeout:3,other:1"``.

    The keys are the pipeline's fixed coarse buckets (timeout, connection, model-validation, model-
    api, image-read, metadata-write, other), never message text, so nothing user-derived can ride
    along.
    """
    return ",".join(f"{kind}:{count}" for kind, count in sorted(kinds.items()) if count)


def file_types_summary(paths: Iterable[Path]) -> str:
    """
    Summarize a batch as its distinct file extensions: lowercased, dot-stripped, comma-joined.

    ``{"a.CR3", "b.jpg", "c.cr3"}`` becomes ``"cr3,jpg"``. This answers "which formats do people
    run?" without revealing a single filename: only the set of extensions leaves the machine.
    """
    extensions = {path.suffix.lower().lstrip(".") for path in paths if path.suffix}
    return ",".join(sorted(extensions))


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
    """
    path = _state_dir() / _INSTALL_ID_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        return str(uuid.UUID(existing))
    except OSError, ValueError:
        # Missing, empty, or corrupt: mint a new one below.
        pass

    new_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_install_id_persist_failed", error=str(exc))
    return new_id


def first_run_notice() -> str | None:
    """
    Return :data:`FIRST_RUN_NOTICE` the first time telemetry is active, else ``None``.

    A marker file records the :data:`NOTICE_VERSION` that has been shown, so the notice appears once
    per disclosure version: when new data categories are added (hardware, crash reports), existing
    installs see the updated text once too. The marker is written before returning, so two calls in
    one process do not both show it. If the marker cannot be written the notice may reappear on a
    later run, which is harmless.
    """
    marker = _state_dir() / _NOTICE_MARKER_FILE
    try:
        if marker.read_text(encoding="utf-8").strip() == str(NOTICE_VERSION):
            return None
    except OSError:
        pass  # Missing marker: first run, show the notice below.
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


def _platform_facts() -> dict[str, object]:
    """Collect the locally-derived facts every beacon (run or crash) carries."""
    hardware = hardware_info(_state_dir() / _HARDWARE_CACHE_FILE)
    return {
        "schema_version": SCHEMA_VERSION,
        "install_id": install_id(),
        "app_version": __version__,
        "arch": platform.machine(),  # e.g. "arm64", "x86_64", "aarch64"
        "os": platform.system(),  # "Darwin", "Linux", "Windows"
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu": hardware.cpu,  # model string, e.g. "Apple M3 Pro"; "" when unknown
        "gpu": hardware.gpu,  # model string, e.g. "NVIDIA GeForce RTX 4070"; "" when unknown
        "cpu_count": hardware.cpu_count,
        "memory_gb": hardware.memory_gb,
    }


def build_payload(run: RunInfo) -> dict[str, object]:
    """
    Build the exact run-beacon body from *run* plus locally-derived platform facts.

    This (together with :func:`build_crash_payload`) is the single source of truth for what leaves
    the machine. Every value here is either a bounded enum (interface, provider), a coarse platform
    or hardware model string, a count, a duration, or the random install id. Nothing derived from
    the user's photos, paths, or identity appears, by construction.
    """
    return {
        **_platform_facts(),
        "event": "run",
        "interface": run.interface,
        "provider": run.provider,
        "model": run.model,
        "batch_size": run.batch_size,
        "duration_seconds": round(run.duration_seconds, 3),
        "output_language": run.output_language,
        "ui_language": run.ui_language,
        "file_types": run.file_types,
        "success_count": run.success_count,
        "failure_count": run.failure_count,
        "cache_hits": run.cache_hits,
        "retry_successes": run.retry_successes,
        "workers": run.workers,
        "total_tokens": run.total_tokens,
        "inference_seconds": round(run.inference_seconds, 3),
        "dry_run": int(run.dry_run),
        "failure_kinds": run.failure_kinds,
    }


def crash_summary(exc: BaseException) -> tuple[str, str, str]:
    """
    Reduce *exc* to ``(exception_type, location, frames)`` safe to send.

    Only code coordinates inside the ``photo_tagger`` package leave the machine: each frame renders
    as ``module:function:line`` (e.g. ``pipeline:run_batch:714``), *location* is the deepest such
    frame, and *frames* is the chain of the last few, joined by ``>``. The exception **message is
    deliberately dropped**: messages routinely embed file paths and other user data. Frames from the
    stdlib or third-party packages are skipped; they carry no photo-tagger fix anyway.
    """
    frames: list[str] = []
    for frame in traceback.extract_tb(exc.__traceback__):
        module = _frame_module(frame.filename)
        if module is None:
            continue
        short = module.removeprefix("photo_tagger.")
        frames.append(f"{short}:{frame.name}:{frame.lineno}")
    kept = frames[-_MAX_CRASH_FRAMES:]
    return (type(exc).__name__, kept[-1] if kept else "", ">".join(kept))


def _frame_module(filename: str) -> str | None:
    """Map a traceback frame's file path to its dotted module inside this package, else None."""
    parts = Path(filename).with_suffix("").parts
    try:
        anchor = len(parts) - 1 - parts[::-1].index("photo_tagger")
    except ValueError:
        return None
    return ".".join(parts[anchor:])


def build_crash_payload(exc: BaseException, *, interface: str) -> dict[str, object]:
    """Build the crash-beacon body: platform facts plus the sanitized exception coordinates."""
    exception_type, location, frames = crash_summary(exc)
    return {
        **_platform_facts(),
        "event": "crash",
        "interface": interface,
        "exception_type": exception_type,
        "crash_location": location,
        "crash_frames": frames,
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
    run: RunInfo,
    *,
    enabled: bool,
    opted_in: bool = False,
    block: bool = False,
) -> threading.Thread | None:
    """
    Fire one telemetry beacon for *run* on a background daemon thread.

    Returns the thread (or ``None`` when telemetry is disabled), regardless of whether building the
    payload later succeeds inside that thread. Pass ``block=True`` to wait up to
    :data:`_FLUSH_TIMEOUT_SECONDS` for delivery. Both the CLI (which exits right after the batch)
    and the GUI (whose process exits right after ``closeEvent``) use it: a daemon thread abandoned
    at interpreter exit is killed mid-send and the beacon is lost.
    """
    if not should_send(config_enabled=enabled, opted_in=opted_in):
        return None
    return _dispatch(
        lambda: build_payload(run),
        block=block,
        failure_event="telemetry_payload_failed",
    )


# Crash beacons sent by this process, capped at _MAX_CRASH_BEACONS (see emit_crash).
_crash_beacons_sent = 0


def emit_crash(
    exc: BaseException,
    *,
    interface: str,
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
    if _crash_beacons_sent >= _MAX_CRASH_BEACONS or not should_send(
        config_enabled=enabled,
        opted_in=opted_in,
    ):
        return None
    _crash_beacons_sent += 1
    return _dispatch(
        lambda: build_crash_payload(exc, interface=interface),
        block=block,
        failure_event="telemetry_crash_payload_failed",
    )
