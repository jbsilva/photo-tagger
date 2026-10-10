"""
Anonymous, opt-out usage telemetry.

Small JSON beacons tell the project how photo-tagger is used: which models, settings, and features
people pick, how runs go, and on what machines. One is sent at the end of each run, for each other
command, when a GUI window opens and closes, when a ``watch`` stops, and when the app crashes. It
is deliberately privacy-preserving:

- **No personal or content data.** File paths, filenames, photo pixels, generated tags, titles,
  descriptions, prompts, hints, API keys, and server URLs are never collected. A setting is sent as
  a bounded value (a switch, a count, a mode name), never as the text or path it holds. Crash
  beacons carry the exception *type* and its stack trace as code locations (module, function, and
  line), never the message. The dataclasses below and :func:`build_payload` are the closed set of
  what leaves the machine.
- **Opt-out, four ways.** Set ``PHOTO_TAGGER_TELEMETRY=0`` (or the cross-tool ``DO_NOT_TRACK=1``)
  in the environment, pass ``--no-telemetry`` to ``tag`` or ``watch``, put ``enabled = false``
  under ``[telemetry]`` in the config file, or clear the GUI's telemetry toggle. Any opt-out beats
  any opt-in, with two exceptions: ``PHOTO_TAGGER_TELEMETRY=1`` overrules the cross-tool
  ``DO_NOT_TRACK``, and a GUI toggle checked by hand overrules ``enabled = false`` (see
  :func:`resolve_consent`).
- **Never in the way.** Sending happens on a daemon thread with a short timeout, and every failure
  is swallowed. Telemetry must never crash or block a tagging run. At exit, the CLI and the GUI
  wait at most :data:`_FLUSH_TIMEOUT_SECONDS` for a beacon still on its way.

The collector is our own Cloudflare Worker at ``telemetry.tagger.photo`` (its source lives in the
``telemetry/`` directory of this repo). No third-party analytics service is involved.

The per-machine ``install_id`` is a random UUID generated once and stored in the user state
directory. It is **not** derived from any hardware identifier, so it cannot be used to fingerprint a
machine; it only lets repeated runs from the same install be counted as one active user over time.
The ``session_id`` is a random UUID per process. It is never written to disk, so it only ties
together the beacons one launch of the app sends.
"""

import platform
import threading
import time
import uuid
from typing import TYPE_CHECKING

import httpx2
from loguru import logger

from photo_tagger import __version__
from photo_tagger.config import (
    state_dir,
)
from photo_tagger.hardware import hardware_info
from photo_tagger.pacing import read_battery
from photo_tagger.telemetry_consent import (
    CONSENT_DEFAULT,
    CONSENT_DO_NOT_TRACK,
    CONSENT_ENV_OPT_IN,
    CONSENT_ENV_OPT_OUT,
    CONSENT_GUI_OPT_IN,
    CONSENT_SETTING_OFF,
    Consent,
    do_not_track,
    gui_choice,
    read_gui_pref,
    resolve_consent,
    should_send,
    write_gui_pref,
)
from photo_tagger.telemetry_events import (
    COMMAND_DETAILS,
    EVENT_COMMAND,
    EVENT_CRASH,
    EVENT_LAUNCH,
    EVENT_RUN,
    EVENT_SESSION,
    GUI_FEATURES,
    OUTCOME_CANCELLED,
    OUTCOME_EMPTY,
    OUTCOME_FAILED,
    OUTCOME_INTERRUPTED,
    OUTCOME_OK,
    OUTCOME_PARTIAL,
    OUTCOME_SETUP_FAILED,
    OUTCOME_USAGE_ERROR,
    CommandInfo,
    Display,
    Event,
    GuiJob,
    LaunchInfo,
    ReviewStats,
    RunInfo,
    SessionInfo,
    Settings,
    Setup,
    failure_kinds_summary,
    file_types_summary,
    outcome_for,
)
from photo_tagger.telemetry_facts import (
    MAX_STACK_FRAMES,
    attached_terminal,
    exception_chain,
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
from photo_tagger.telemetry_state import (
    FIRST_RUN_NOTICE,
    NOTICE_VERSION,
    first_run_notice,
    install_id,
    notice_due,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


# The public surface: callers use telemetry.X, whichever module defines X.
__all__ = [
    "COMMAND_DETAILS",
    "CONSENT_DEFAULT",
    "CONSENT_DO_NOT_TRACK",
    "CONSENT_ENV_OPT_IN",
    "CONSENT_ENV_OPT_OUT",
    "CONSENT_GUI_OPT_IN",
    "CONSENT_SETTING_OFF",
    "ENDPOINT",
    "EVENT_COMMAND",
    "EVENT_CRASH",
    "EVENT_LAUNCH",
    "EVENT_RUN",
    "EVENT_SESSION",
    "FIRST_RUN_NOTICE",
    "GUI_FEATURES",
    "NOTICE_VERSION",
    "OUTCOME_CANCELLED",
    "OUTCOME_EMPTY",
    "OUTCOME_FAILED",
    "OUTCOME_INTERRUPTED",
    "OUTCOME_OK",
    "OUTCOME_PARTIAL",
    "OUTCOME_SETUP_FAILED",
    "OUTCOME_USAGE_ERROR",
    "SCHEMA_VERSION",
    "SESSION_ID",
    "CommandInfo",
    "Consent",
    "Display",
    "Event",
    "GuiJob",
    "LaunchInfo",
    "ReviewStats",
    "RunInfo",
    "SessionInfo",
    "Settings",
    "Setup",
    "build_crash_payload",
    "build_payload",
    "command_notes",
    "crash_summary",
    "do_not_track",
    "emit",
    "emit_crash",
    "failure_kinds_summary",
    "file_types_summary",
    "first_run_notice",
    "flush",
    "gui_choice",
    "install_id",
    "mark_reported",
    "note",
    "notice_due",
    "outcome_for",
    "read_gui_pref",
    "resolve_consent",
    "sent_any",
    "set_locale_hint",
    "should_send",
    "write_gui_pref",
]


# The collector endpoint (our own Cloudflare Worker). Kept as a module constant so tests can patch
# it and so there is a single place to repoint it.
ENDPOINT = "https://telemetry.tagger.photo"


# Bumped when the payload shape changes. v2 added crash beacons, hardware, and outcome counters; v3
# added the session, command, and launch events, the settings and setup groups, and the GUI's
# per-job runs. The Worker still accepts v1 and v2 from older installs.
SCHEMA_VERSION = 3


# Network call budget. A healthy beacon is a few tens of milliseconds; this only bounds the bad case
# (no network, DNS hang) so the call cannot wedge process exit.
_SEND_TIMEOUT_SECONDS = 2.0


# How long a blocking caller (the CLI, which exits right after) waits for the beacon to flush before
# giving up and letting the daemon thread be abandoned at interpreter exit.
_FLUSH_TIMEOUT_SECONDS = 2.5


_HARDWARE_CACHE_FILE = "hardware.json"


# Crash beacons are capped per process so a pathological crash loop (a repeatedly-failing Qt slot)
# cannot spam the collector.
_MAX_CRASH_BEACONS = 3


# How many in-package stack frames a crash beacon carries (deepest last).
_MAX_CRASH_FRAMES = 5


# Random per process, never stored: it ties together the beacons of one launch.
SESSION_ID = str(uuid.uuid4())


_PROCESS_STARTED = time.monotonic()


# Filled by the GUI from Qt's own locale, which knows the OS language when a Finder launch leaves
# the environment without LANG.
_locale_hint: str | None = None


# Set once any beacon is on its way, so the CLI sends its fallback command beacon only when nothing
# else reported the invocation.
_sent_any = False


# Facts the running command noted for its beacon (see note()).
_notes: dict[str, str | float] = {}


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


def _state_dir() -> Path:
    """Return the shared state directory telemetry keeps its install id and markers in."""
    return state_dir()


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
    # A run also says whether the machine ended it on a charger or its battery (the events are
    # plain data, so the battery is read here, at the one place every beacon passes).
    power = _power() if event.EVENT == EVENT_RUN else {}
    payload = {**envelope, **power, **event.body()}
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
