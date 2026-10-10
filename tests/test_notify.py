"""
Unit tests for the end-of-batch desktop alert.

Nothing here posts a real notification: a test cannot see one, and a CI runner has no desktop to
post it to. What is checked instead is what the module actually decides - which helper process it
picks per platform, how the message reaches that process without being able to change what runs,
when the sound is chased separately, and that every failure mode stays silent.
"""

import pathlib
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable

import pytest

from photo_tagger import notify as notify_module
from photo_tagger.notify import (
    Alert,
    deliver,
    notification_command,
    notify,
    sound_command,
)


def _fake_which(available: dict[str, str]) -> Callable[[str], str | None]:
    """Build a shutil.which stand-in that only knows the programs in *available*."""

    def which(program: str, **_kwargs: object) -> str | None:
        return available.get(program)

    return which


@pytest.fixture
def no_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the machine has none of the helper programs on PATH."""
    monkeypatch.setattr(shutil, "which", _fake_which({}))


def test_macos_notification_passes_text_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message travels as an osascript argument, so it can never become AppleScript."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"osascript": "/usr/bin/osascript"}),
    )

    alert = notification_command('Photo "Tagger"', "3 of 3 photos tagged.", "darwin")

    assert alert is not None
    assert alert.argv[0] == "/usr/bin/osascript"
    assert alert.argv[-2:] == ["3 of 3 photos tagged.", 'Photo "Tagger"']
    # The script names the arguments rather than embedding either string.
    assert "item 1 of argv" in alert.argv[4]
    assert "3 of 3" not in alert.argv[4]
    assert not alert.plays_sound


def test_macos_notification_carries_its_own_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A macOS notification plays the sound itself, so no second helper is needed."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"osascript": "/usr/bin/osascript"}),
    )

    alert = notification_command("Photo Tagger", "done", "darwin", sound=True)

    assert alert is not None
    assert "sound name" in alert.argv[4]
    assert alert.plays_sound


def test_linux_notification_uses_notify_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """The summary and the body reach notify-send positionally, after a -- guard."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"notify-send": "/usr/bin/notify-send"}),
    )

    alert = notification_command("Photo Tagger", "3 of 3 photos tagged.", "linux")

    assert alert is not None
    assert alert.argv == [
        "/usr/bin/notify-send",
        "--app-name",
        "Photo Tagger",
        "--",
        "Photo Tagger",
        "3 of 3 photos tagged.",
    ]
    # notify-send has no sound of its own, whatever the caller asked for.
    assert not alert.plays_sound


def test_linux_notification_never_claims_to_play_a_sound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for a sound on Linux still leaves it to the separate chime."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"notify-send": "/usr/bin/notify-send"}),
    )

    alert = notification_command("Photo Tagger", "done", "linux", sound=True)

    assert alert is not None
    assert not alert.plays_sound


def test_windows_toast_keeps_the_text_out_of_the_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The toast reads title and body from the environment, never from the PowerShell source."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"powershell": r"C:\powershell.exe"}),
    )

    alert = notification_command("Photo Tagger", "3 of 3; rm -rf", "win32")

    assert alert is not None
    script = alert.argv[-1]
    assert "rm -rf" not in script
    assert alert.env["PHOTO_TAGGER_ALERT_BODY"] == "3 of 3; rm -rf"
    assert alert.env["PHOTO_TAGGER_ALERT_TITLE"] == "Photo Tagger"
    assert "PHOTO_TAGGER_ALERT_APP_ID" in alert.env


@pytest.mark.parametrize(("sound", "silent"), [(True, "'false'"), (False, "'true'")])
def test_windows_toast_silences_itself_unless_a_sound_was_asked_for(
    monkeypatch: pytest.MonkeyPatch,
    sound: bool,  # noqa: FBT001 - the parametrized input.
    silent: str,
) -> None:
    """A Windows toast is audible by default, so the quiet case must say so explicitly."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"powershell": r"C:\powershell.exe"}),
    )

    alert = notification_command("Photo Tagger", "done", "win32", sound=sound)

    assert alert is not None
    assert f"SetAttribute('silent', {silent})" in alert.argv[-1]
    assert alert.plays_sound is sound


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "win32", "freebsd14"])
def test_no_notifier_installed_is_not_an_error(
    no_programs: None,
    platform_name: str,
) -> None:
    """A machine without the helper program yields no alert rather than a failure."""
    assert notification_command("Photo Tagger", "done", platform_name) is None


def test_macos_sound_plays_a_system_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone macOS chime is afplay on a stock system sound."""
    monkeypatch.setattr(shutil, "which", _fake_which({"afplay": "/usr/bin/afplay"}))
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: True)

    alert = sound_command("darwin")

    assert alert is not None
    assert alert.argv[0] == "/usr/bin/afplay"
    assert alert.argv[1].endswith(".aiff")
    assert alert.plays_sound


def test_macos_sound_skipped_when_the_system_file_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A macOS without the stock sound file falls through rather than running a doomed afplay."""
    monkeypatch.setattr(shutil, "which", _fake_which({"afplay": "/usr/bin/afplay"}))
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: False)

    assert sound_command("darwin") is None


def test_linux_sound_prefers_the_theme_over_a_hardcoded_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking canberra-gtk-play for the event by name lets the user's sound theme answer."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"canberra-gtk-play": "/usr/bin/canberra-gtk-play", "paplay": "/usr/bin/pap"}),
    )

    alert = sound_command("linux")

    assert alert is not None
    assert alert.argv == ["/usr/bin/canberra-gtk-play", "-i", "complete"]


def test_linux_sound_falls_back_to_paplay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without canberra, the freedesktop sample is played directly."""
    monkeypatch.setattr(shutil, "which", _fake_which({"paplay": "/usr/bin/paplay"}))
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: True)

    alert = sound_command("linux")

    assert alert is not None
    assert alert.argv[0] == "/usr/bin/paplay"
    assert alert.argv[1].endswith(".oga")


def test_linux_sound_skipped_when_the_sample_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running paplay with nothing to play is not worth the process."""
    monkeypatch.setattr(shutil, "which", _fake_which({"paplay": "/usr/bin/paplay"}))
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: False)

    assert sound_command("linux") is None


def test_windows_sound_is_a_powershell_chime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows chime waits for the sound, since PowerShell exiting would cut it short."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"powershell": r"C:\powershell.exe"}),
    )

    alert = sound_command("win32")

    assert alert is not None
    assert "SystemSounds" in alert.argv[-1]
    assert "Start-Sleep" in alert.argv[-1]


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "win32"])
def test_no_sound_program_installed_is_not_an_error(
    no_programs: None,
    platform_name: str,
) -> None:
    """Every platform tolerates having nothing to play the sound with."""
    assert sound_command(platform_name) is None


class _Recorder:
    """Stands in for subprocess.run, recording each call and replaying a fixed outcome."""

    def __init__(self, returncode: int = 0, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.returncode = returncode
        self.error = error

    def __call__(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(list(argv))
        self.envs.append(None if env is None else dict(env))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(argv, self.returncode)


def test_deliver_spends_one_process_when_the_notifier_carries_the_sound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MacOS gets its sound from the notification itself, so no chime follows."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"osascript": "/usr/bin/osascript", "afplay": "/usr/bin/afplay"}),
    )
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    deliver("Photo Tagger", "done", sound=True, platform_name="darwin")

    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] == "/usr/bin/osascript"


def test_deliver_chases_the_sound_where_the_notifier_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux needs a second process, because notify-send cannot make a sound."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which(
            {"notify-send": "/usr/bin/notify-send", "canberra-gtk-play": "/usr/bin/canberra"},
        ),
    )
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    deliver("Photo Tagger", "done", sound=True, platform_name="linux")

    assert [call[0] for call in recorder.calls] == ["/usr/bin/notify-send", "/usr/bin/canberra"]


def test_deliver_stays_quiet_when_no_sound_was_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --notify-sound the chime is never even looked for."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which(
            {"notify-send": "/usr/bin/notify-send", "canberra-gtk-play": "/usr/bin/canberra"},
        ),
    )
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    deliver("Photo Tagger", "done", sound=False, platform_name="linux")

    assert [call[0] for call in recorder.calls] == ["/usr/bin/notify-send"]


def test_deliver_rings_the_bell_when_nothing_can_make_a_sound(
    no_programs: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The terminal bell is the last resort on a machine with no audio helper."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)

    deliver("Photo Tagger", "done", sound=True, platform_name="linux")

    assert "\a" in capsys.readouterr().err


def test_deliver_does_not_litter_a_redirected_stderr_with_a_bell(
    no_programs: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A BEL byte in a log file is noise nobody hears, so a non-tty stderr gets nothing."""
    # capsys replaces stderr with a non-tty capture buffer, which is exactly the case under test.
    deliver("Photo Tagger", "done", sound=True, platform_name="linux")

    assert capsys.readouterr().err == ""


def test_deliver_survives_a_notifier_that_cannot_be_launched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OSError from the helper process is swallowed; the batch is already written."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"notify-send": "/usr/bin/notify-send"}),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        _Recorder(error=OSError("no such file")),
    )

    deliver("Photo Tagger", "done", sound=False, platform_name="linux")


def test_deliver_survives_a_notifier_that_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung notifier is killed by the timeout and reported no further."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"notify-send": "/usr/bin/notify-send"}),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        _Recorder(error=subprocess.TimeoutExpired("notify-send", 10.0)),
    )

    deliver("Photo Tagger", "done", sound=False, platform_name="linux")


def test_a_rejected_notification_still_falls_through_to_the_sound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A notifier that exits non-zero showed nothing, so its claimed sound cannot be trusted."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"osascript": "/usr/bin/osascript", "afplay": "/usr/bin/afplay"}),
    )
    monkeypatch.setattr(pathlib.Path, "is_file", lambda _self: True)
    recorder = _Recorder(returncode=1)
    monkeypatch.setattr(subprocess, "run", recorder)

    deliver("Photo Tagger", "done", sound=True, platform_name="darwin")

    assert [call[0] for call in recorder.calls] == ["/usr/bin/osascript", "/usr/bin/afplay"]


def test_alert_env_is_layered_over_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A toast must inherit PATH and friends, not run with only its three own variables."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"powershell": r"C:\powershell.exe"}),
    )
    monkeypatch.setenv("PHOTO_TAGGER_TEST_MARKER", "kept")
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    deliver("Photo Tagger", "done", sound=False, platform_name="win32")

    env = recorder.envs[0]
    assert env is not None
    assert env["PHOTO_TAGGER_TEST_MARKER"] == "kept"
    assert env["PHOTO_TAGGER_ALERT_BODY"] == "done"


def test_an_alert_without_extra_env_inherits_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing env=None is what makes a child inherit os.environ untouched."""
    monkeypatch.setattr(
        shutil,
        "which",
        _fake_which({"notify-send": "/usr/bin/notify-send"}),
    )
    recorder = _Recorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    deliver("Photo Tagger", "done", sound=False, platform_name="linux")

    assert recorder.envs == [None]


def test_notify_disabled_starts_no_thread() -> None:
    """The off switch is checked before anything is spawned."""
    assert notify("Photo Tagger", "done", enabled=False) is None


def test_notify_runs_on_a_daemon_thread_and_can_be_waited_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing block=True is what lets a process about to exit still get its alert out."""
    seen: list[tuple[str, str, bool]] = []

    def fake_deliver(title: str, message: str, *, sound: bool, platform_name: str) -> None:
        seen.append((title, message, sound))
        assert platform_name  # the caller resolves the platform, not deliver's default.

    monkeypatch.setattr(notify_module, "deliver", fake_deliver)

    thread = notify("Photo Tagger", "done", enabled=True, sound=True, block=True)

    assert thread is not None
    assert thread.daemon
    assert not thread.is_alive()
    assert seen == [("Photo Tagger", "done", True)]


def test_deliver_survives_a_stderr_that_has_gone_away(
    no_programs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detached stderr (a windowed launch) must not turn the last-resort bell into a crash."""

    class _ClosedStderr:
        def isatty(self) -> bool:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    monkeypatch.setattr(sys, "stderr", _ClosedStderr())

    deliver("Photo Tagger", "done", sound=True, platform_name="linux")


def test_notify_without_block_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GUI never blocks its own thread on an alert, so block defaults to off."""
    joins: list[float | None] = []
    monkeypatch.setattr(
        threading.Thread,
        "join",
        lambda _self, timeout=None: joins.append(timeout),
    )
    monkeypatch.setattr(notify_module, "deliver", lambda *_args, **_kwargs: None)

    thread = notify("Photo Tagger", "done", enabled=True)

    assert thread is not None
    assert joins == []


def test_alert_defaults_keep_a_plain_command_silent() -> None:
    """An Alert built with only an argv carries no environment and claims no sound."""
    alert = Alert(argv=["/bin/true"])

    assert alert.env == {}
    assert not alert.plays_sound
