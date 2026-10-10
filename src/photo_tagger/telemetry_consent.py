"""Whether to send telemetry, and why: the environment, the flags, and the GUI toggle."""

import os
from dataclasses import dataclass

from loguru import logger

from photo_tagger.config import (
    state_dir,
)


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


_GUI_PREF_FILE = "gui-telemetry"


# What that file holds: a hand-checked toggle, a cleared one, and the "on" that 0.9 wrote.
_GUI_PREF_OPT_IN = "opt-in"


_GUI_PREF_OFF = "0"


_GUI_PREF_LEGACY_ON = "1"


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
        return (state_dir() / _GUI_PREF_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def write_gui_pref(*, enabled: bool) -> None:
    """Persist the GUI's telemetry choice so the Settings toggle sticks across restarts."""
    path = state_dir() / _GUI_PREF_FILE
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
