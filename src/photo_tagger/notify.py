"""
Desktop alert when a batch finishes.

A long run is meant to be walked away from, so the one thing worth interrupting the user for is the
run being over. :func:`notify` posts a native desktop notification, optionally with a sound, through
whatever the platform already ships: ``osascript`` on macOS, ``notify-send`` on Linux and the BSDs,
a PowerShell toast on Windows. Nothing new to install, and the CLI stays dependency-light.

Everything here is best-effort. No notifier on PATH, a headless session, a denied notification
permission: each of those means the alert does not appear, and none of them is a reason to fail a
batch whose metadata is already on disk. Failures are logged at debug level and swallowed.

The work runs on a daemon thread because posting a notification spawns a helper process, which can
take a second or two. The CLI passes ``block=True`` (it exits right after the batch, and an
abandoned thread would lose the alert); the GUI does not, so its window never freezes on one.
"""

import os
import subprocess  # nosec B404 - only used to run the OS's own notifier and sound player
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from photo_tagger.programs import find_program


# What the notification calls itself. Not translated: it is the application's name.
APP_NAME = "Photo Tagger"

# How long one helper process gets before it is killed. Generous, because the batch is already over
# and the only cost of waiting is a notification that arrives late.
_COMMAND_TIMEOUT_SECONDS = 10.0

# How long a blocking caller waits for the alert before giving up. The helper process survives being
# abandoned (it is not a thread), so a slow notifier still lands after the CLI has exited.
_FLUSH_TIMEOUT_SECONDS = 5.0

# The macOS alert sound, by the name `display notification` knows it under, and the same sound as a
# file for when there is no notification to hang it on.
_MACOS_SOUND = "Glass"
_MACOS_SOUND_FILE = Path(f"/System/Library/Sounds/{_MACOS_SOUND}.aiff")

# The freedesktop sound theme's "task complete" sample, played directly when the small helper that
# knows the theme by name (canberra-gtk-play) is not installed.
_FREEDESKTOP_SOUND_FILE = Path("/usr/share/sounds/freedesktop/stereo/complete.oga")

# Windows shows a toast under an application id, and only one that is registered with the shell gets
# through. Windows PowerShell registers its own, and it is the host running the script, so the toast
# is attributed honestly rather than to an id photo-tagger would have to install first.
_WINDOWS_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"

# The toast text travels in the environment, not in the script: PowerShell quoting rules are their
# own subject, and an apostrophe in a message must not be able to change what runs.
_TITLE_VAR = "PHOTO_TAGGER_ALERT_TITLE"
_BODY_VAR = "PHOTO_TAGGER_ALERT_BODY"
_APP_ID_VAR = "PHOTO_TAGGER_ALERT_APP_ID"


@dataclass(frozen=True, slots=True)
class Alert:
    """
    One helper process that raises the alert.

    ``plays_sound`` records whether running it is already audible, so a platform whose notifier
    carries its own sound (macOS, Windows) is not chased with a second, redundant chime.
    """

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    plays_sound: bool = False


# The WinRT class that owns toasts, spelled out once because PowerShell wants it in full.
_TOAST_MANAGER = "[Windows.UI.Notifications.ToastNotificationManager]"


def _windows_toast_script(*, sound: bool) -> str:
    """
    Build the PowerShell that shows one toast, reading its text from the environment.

    ``ToastText02`` is the bold-heading-plus-body template, which is the shape of this alert. The
    text goes in through ``CreateTextNode`` rather than by formatting XML, so no message can break
    the document. A toast is audible by default, hence the explicit ``silent`` attribute.
    """
    load_winrt = (
        "$null = [Windows.UI.Notifications.ToastNotificationManager,"
        " Windows.UI.Notifications, ContentType = WindowsRuntime]"
    )
    load_template = (
        f"$xml = {_TOAST_MANAGER}::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02)"
    )
    show = f"{_TOAST_MANAGER}::CreateToastNotifier($env:{_APP_ID_VAR}).Show($toast)"
    return "; ".join(
        (
            "$ErrorActionPreference = 'Stop'",
            load_winrt,
            load_template,
            "$text = $xml.GetElementsByTagName('text')",
            f"$null = $text.Item(0).AppendChild($xml.CreateTextNode($env:{_TITLE_VAR}))",
            f"$null = $text.Item(1).AppendChild($xml.CreateTextNode($env:{_BODY_VAR}))",
            "$audio = $xml.CreateElement('audio')",
            f"$audio.SetAttribute('silent', '{str(not sound).lower()}')",
            "$null = $xml.DocumentElement.AppendChild($audio)",
            "$toast = New-Object Windows.UI.Notifications.ToastNotification $xml",
            show,
        ),
    )


# Plays the system's "something finished" sound and waits for it, since PowerShell exiting would cut
# an asynchronous Play() short.
_WINDOWS_CHIME_SCRIPT = (
    "[System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep -Milliseconds 700"
)

# Keeps the toast and the chime consistent: no profile to slow the launch down or change behavior
# per machine, and no prompt for a script that has nobody to answer it.
_POWERSHELL_FLAGS = ("-NoProfile", "-NonInteractive", "-Command")


def _powershell_alert(script: str, env: dict[str, str], *, plays_sound: bool) -> Alert | None:
    """Wrap *script* as a Windows PowerShell invocation, or None when the host is not installed."""
    # Windows PowerShell, not pwsh: the WinRT type resolution the toast relies on is a 5.1 feature,
    # and 5.1 is on every supported Windows while pwsh is an optional install.
    powershell = find_program("powershell")
    if powershell is None:
        return None
    return Alert(argv=[powershell, *_POWERSHELL_FLAGS, script], env=env, plays_sound=plays_sound)


def _macos_notification(title: str, message: str, *, sound: bool) -> Alert | None:
    """Build the ``osascript`` alert for macOS, or None when osascript is missing."""
    osascript = find_program("osascript")
    if osascript is None:
        return None
    # The text is passed as arguments to an `on run argv` handler instead of being pasted into the
    # script, so a quote or a backslash in a message stays text rather than becoming AppleScript.
    statement = "display notification (item 1 of argv) with title (item 2 of argv)"
    if sound:
        statement = f'{statement} sound name "{_MACOS_SOUND}"'
    return Alert(
        argv=[
            osascript,
            "-e",
            "on run argv",
            "-e",
            statement,
            "-e",
            "end run",
            "--",
            message,
            title,
        ],
        plays_sound=sound,
    )


def _libnotify_notification(title: str, message: str) -> Alert | None:
    """Build the ``notify-send`` alert used on Linux and the BSDs, or None when it is missing."""
    notify_send = find_program("notify-send")
    if notify_send is None:
        return None
    # notify-send has no sound of its own, so plays_sound stays False and the caller adds a chime.
    return Alert(argv=[notify_send, "--app-name", APP_NAME, "--", title, message])


def notification_command(
    title: str,
    message: str,
    platform_name: str,
    *,
    sound: bool = False,
) -> Alert | None:
    """
    Return the helper process that shows *title* / *message* on *platform_name*, or None.

    None means this machine has no notifier photo-tagger knows how to drive, which is not an error:
    the caller falls back to a sound when one was asked for, and otherwise stays quiet.
    """
    if platform_name == "darwin":
        return _macos_notification(title, message, sound=sound)
    if platform_name.startswith("win"):
        return _powershell_alert(
            _windows_toast_script(sound=sound),
            {_TITLE_VAR: title, _BODY_VAR: message, _APP_ID_VAR: _WINDOWS_APP_ID},
            plays_sound=sound,
        )
    return _libnotify_notification(title, message)


def _linux_chime() -> Alert | None:
    """Play the sound theme's completion sample, by name if possible and by path if not."""
    canberra = find_program("canberra-gtk-play")
    if canberra is not None:
        # Asking for the event by name lets the user's chosen sound theme answer.
        return Alert(argv=[canberra, "-i", "complete"], plays_sound=True)
    paplay = find_program("paplay")
    if paplay is not None and _FREEDESKTOP_SOUND_FILE.is_file():
        return Alert(argv=[paplay, str(_FREEDESKTOP_SOUND_FILE)], plays_sound=True)
    return None


def sound_command(platform_name: str) -> Alert | None:
    """
    Return the helper process that plays a short alert sound on *platform_name*, or None.

    Only needed where the notification could not carry its own sound, or where there was no
    notification at all. None leaves the terminal bell as the last resort.
    """
    if platform_name == "darwin":
        afplay = find_program("afplay")
        if afplay is None or not _MACOS_SOUND_FILE.is_file():
            return None
        return Alert(argv=[afplay, str(_MACOS_SOUND_FILE)], plays_sound=True)
    if platform_name.startswith("win"):
        return _powershell_alert(_WINDOWS_CHIME_SCRIPT, {}, plays_sound=True)
    return _linux_chime()


def _run(alert: Alert) -> bool:
    """
    Run *alert*, reporting whether it succeeded.

    Never raises.
    """
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv; only the text varies
            alert.argv,
            env={**os.environ, **alert.env} if alert.env else None,
            capture_output=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("notify_command_failed", program=alert.argv[0], error=str(exc))
        return False
    if result.returncode != 0:
        logger.debug("notify_command_rejected", program=alert.argv[0], code=result.returncode)
        return False
    return True


def _terminal_bell() -> bool:
    """
    Ring the terminal bell, reporting whether it was rung.

    The last resort when the platform offers nothing better. Skipped when stderr is redirected: a
    BEL byte in a log file is litter nobody hears.
    """
    try:
        if not sys.stderr.isatty():
            return False
        sys.stderr.write("\a")
        sys.stderr.flush()
    except OSError, ValueError:
        # A closed or detached stderr. Nothing to do about it, and nothing worth reporting.
        return False
    return True


def deliver(title: str, message: str, *, sound: bool, platform_name: str) -> None:
    """
    Show one alert on the calling thread; never raises.

    The sound is only chased separately when the notification did not already make one, so macOS and
    Windows spend a single helper process on an alert that Linux needs two for.
    """
    alert = notification_command(title, message, platform_name, sound=sound)
    played = alert.plays_sound if alert is not None and _run(alert) else False
    if not sound or played:
        return
    chime = sound_command(platform_name)
    if chime is None or not _run(chime):
        _terminal_bell()


def notify(
    title: str,
    message: str,
    *,
    enabled: bool,
    sound: bool = False,
    block: bool = False,
) -> threading.Thread | None:
    """
    Raise a desktop alert on a background daemon thread.

    Returns the thread, or None when alerts are switched off. Pass ``block=True`` to wait up to
    :data:`_FLUSH_TIMEOUT_SECONDS` for it, which is what a process about to exit wants.
    """
    if not enabled:
        return None
    thread = threading.Thread(
        target=deliver,
        args=(title, message),
        kwargs={"sound": sound, "platform_name": sys.platform},
        name="photo-tagger-notify",
        daemon=True,
    )
    thread.start()
    if block:
        thread.join(timeout=_FLUSH_TIMEOUT_SECONDS)
    return thread
