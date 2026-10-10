"""What telemetry keeps between runs: the install id and the notice markers."""

import os
import threading
import uuid
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.config import (
    state_dir,
)


if TYPE_CHECKING:
    from pathlib import Path


# State files: the install id and the "first-run notice already shown" markers, under
# config.state_dir() (XDG_STATE_HOME). The GUI's toggle lives beside them (see telemetry_consent).
_INSTALL_ID_FILE = "install-id"


_NOTICE_MARKER_FILE = "telemetry-notice-shown"


# The GUI's own marker: its dialog lists more (idle time, screen size) and offers Turn It Off, so a
# CLI user who later opens the window still sees it once.
_GUI_NOTICE_MARKER_FILE = "telemetry-notice-shown-gui"


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
        path = state_dir() / _INSTALL_ID_FILE
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
    return state_dir() / (_GUI_NOTICE_MARKER_FILE if interface == "gui" else _NOTICE_MARKER_FILE)


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
