"""
Tests for the anonymous, opt-out usage telemetry.

These prove the three things that actually matter: telemetry is off when the user asks for it off
(by any of the three mechanisms), the beacon never carries anything personal, and a send failure can
never escape to the caller.
"""

import errno
import json
import platform
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import httpx2
import pytest

from photo_tagger import __version__, telemetry, telemetry_state
from photo_tagger.hardware import HardwareInfo
from photo_tagger.pacing import BatteryState
from photo_tagger.telemetry import RunInfo


_FAKE_HARDWARE = HardwareInfo(cpu="Apple M3 Pro", gpu="Apple M3 Pro", cpu_count=12, memory_gb=36)


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Point the state directory at a tmp location, stub hardware, and clear the opt-out env vars.

    Without this a developer's real ``~/.local/state/photo-tagger`` (or a ``DO_NOT_TRACK`` in their
    shell) would leak into the tests and make the opt-out assertions flaky; the hardware stub keeps
    payload tests from shelling out to real probes.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PHOTO_TAGGER_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setattr(telemetry, "hardware_info", lambda _path: _FAKE_HARDWARE)
    monkeypatch.setattr(telemetry, "read_battery", lambda: None)
    monkeypatch.setattr(telemetry, "_crash_beacons_sent", 0)


def _sample_run() -> RunInfo:
    """Build a representative run for payload/emit tests."""
    return RunInfo(
        interface="cli",
        provider="lmstudio",
        model="qwen/qwen3-vl-30b",
        batch_size=12,
        duration_seconds=4.5,
        output_language="English",
        ui_language="en",
        file_types="cr3,jpg",
    )


# --- opt-out precedence ------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "NO", "Off"])
def test_env_switch_off_opts_out(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Any accepted "no" spelling of the switch turns telemetry off, even over a GUI opt-in."""
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", value)
    assert telemetry.resolve_consent(enabled=True, opted_in=True) == telemetry.Consent(
        send=False,
        reason=telemetry.CONSENT_ENV_OPT_OUT,
    )


@pytest.mark.parametrize("value", ["", "   "])
def test_env_switch_empty_counts_as_not_set(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """An empty value changes nothing, so a stray ``PHOTO_TAGGER_TELEMETRY=`` is harmless."""
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", value)
    assert telemetry.resolve_consent(enabled=True).reason == telemetry.CONSENT_DEFAULT


@pytest.mark.parametrize("value", ["maybe", "2", "disabled", "none"])
def test_env_switch_unknown_spelling_opts_out(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Someone who set the switch to a word it does not know meant no: an opt-out fails closed."""
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", value)
    assert telemetry.resolve_consent(enabled=True, opted_in=True) == telemetry.Consent(
        send=False,
        reason=telemetry.CONSENT_ENV_OPT_OUT,
    )


def test_legacy_env_opt_out_still_opts_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.9's ``PHOTO_TAGGER_NO_TELEMETRY=1`` keeps working, so no opt-out is lost in an upgrade."""
    monkeypatch.setenv("PHOTO_TAGGER_NO_TELEMETRY", "1")
    assert telemetry.resolve_consent(enabled=True, opted_in=True) == telemetry.Consent(
        send=False,
        reason=telemetry.CONSENT_ENV_OPT_OUT,
    )


def test_do_not_track_opts_out_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cross-tool DO_NOT_TRACK standard turns telemetry off when nothing overrules it."""
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert telemetry.do_not_track() is True
    assert telemetry.resolve_consent(enabled=True) == telemetry.Consent(
        send=False,
        reason=telemetry.CONSENT_DO_NOT_TRACK,
    )


@pytest.mark.parametrize("value", ["1", "true", "yes", "ON"])
def test_tool_switch_overrules_do_not_track(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """PHOTO_TAGGER_TELEMETRY=1 is a choice for this tool alone, so it beats DO_NOT_TRACK."""
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", value)
    assert telemetry.resolve_consent(enabled=True) == telemetry.Consent(
        send=True,
        reason=telemetry.CONSENT_ENV_OPT_IN,
    )


def test_gui_opt_in_overrules_do_not_track(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GUI toggle turned on by hand is an opt-in for this tool, so it beats DO_NOT_TRACK too."""
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert telemetry.resolve_consent(enabled=True, opted_in=True) == telemetry.Consent(
        send=True,
        reason=telemetry.CONSENT_GUI_OPT_IN,
    )


def test_any_opt_out_beats_any_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit off anywhere wins: no opt-in may overrule a user who turned telemetry off."""
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", "1")
    assert telemetry.resolve_consent(enabled=False).reason == telemetry.CONSENT_SETTING_OFF


def test_should_send_true_only_when_enabled_and_no_env_optout() -> None:
    """With nothing opting out, the merged flag/config bool decides."""
    assert telemetry.should_send(config_enabled=True) is True
    assert telemetry.should_send(config_enabled=False) is False
    assert telemetry.resolve_consent(enabled=True).reason == telemetry.CONSENT_DEFAULT


def test_env_optout_overrides_enabled_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment wins even when flag/config asked to keep telemetry on."""
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", "0")
    assert telemetry.should_send(config_enabled=True) is False


def test_emit_sends_under_do_not_track_only_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emit honors DO_NOT_TRACK, and an explicit opt-in is what lets a beacon through anyway."""
    posted: list[object] = []
    monkeypatch.setattr(httpx2, "post", lambda *_a, **k: posted.append(k))
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert telemetry.emit(_sample_run(), enabled=True, block=True) is None
    assert posted == []
    assert telemetry.emit(_sample_run(), enabled=True, opted_in=True, block=True) is not None
    assert len(posted) == 1


# --- install id --------------------------------------------------------------------------------


def test_install_id_is_stable_and_persisted(tmp_path: Path) -> None:
    """The id is a valid UUID, written to disk, and identical across calls."""
    first = telemetry.install_id()
    uuid.UUID(first)  # raises if not a valid UUID
    id_file = tmp_path / "state" / "photo-tagger" / "install-id"
    assert id_file.is_file()
    assert telemetry.install_id() == first


def test_install_id_regenerates_when_corrupt(tmp_path: Path) -> None:
    """A garbage id file is replaced with a fresh valid UUID."""
    id_file = tmp_path / "state" / "photo-tagger" / "install-id"
    id_file.parent.mkdir(parents=True)
    id_file.write_text("not-a-uuid")
    fresh = telemetry.install_id()
    uuid.UUID(fresh)
    assert fresh != "not-a-uuid"


def test_install_id_degrades_when_state_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the state dir cannot be created, a valid ephemeral id is still returned."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    fresh = telemetry.install_id()
    uuid.UUID(fresh)  # still usable, just not persisted


# --- first-run notice --------------------------------------------------------------------------


def test_install_id_is_one_id_when_first_beacons_race() -> None:
    """Two threads minting the first id at once end up with the same id, not two installs."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = set(pool.map(lambda _: telemetry.install_id(), range(16)))
    assert len(ids) == 1


def test_install_id_reads_the_winner_when_another_process_wrote_first(tmp_path: Path) -> None:
    """Losing the exclusive create to another process means using that process's id."""
    path = tmp_path / "state" / "photo-tagger" / "install-id"
    winner = "12345678-1234-4321-8765-123456789abc"

    def other_process_wins(_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(winner, encoding="utf-8")

    with patch.object(telemetry_state, "_read_install_id", side_effect=[None, winner]) as read:
        other_process_wins(path)
        assert telemetry.install_id() == winner
    assert read.call_count == 2  # noqa: PLR2004 - once before the create, once after losing it.


def test_first_run_notice_shows_once() -> None:
    """The notice text is returned on the first call and suppressed thereafter."""
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE
    assert telemetry.first_run_notice() is None


def test_gui_notice_has_its_own_marker() -> None:
    """A notice shown by the CLI does not stop the window from showing its own dialog once."""
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE
    assert telemetry.notice_due(interface="gui") is True
    assert telemetry.first_run_notice(interface="gui") == telemetry.FIRST_RUN_NOTICE
    assert telemetry.notice_due(interface="gui") is False


def test_notice_due_does_not_mark_the_notice_shown() -> None:
    """Asking is not showing: the notice is still due after the question."""
    assert telemetry.notice_due() is True
    assert telemetry.notice_due() is True
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE


def test_first_run_notice_returns_when_disclosure_version_changes(tmp_path: Path) -> None:
    """
    A marker from an older disclosure version does not suppress the updated notice.

    v1 installs wrote an empty marker; v2 added hardware and crash reports to the disclosure, and
    those users must see the expanded text once.
    """
    marker = tmp_path / "state" / "photo-tagger" / "telemetry-notice-shown"
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")  # what a v1 install left behind
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE
    assert telemetry.first_run_notice() is None
    assert marker.read_text(encoding="utf-8").strip() == str(telemetry.NOTICE_VERSION)


def test_first_run_notice_still_shows_when_marker_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the marker file cannot be written, the notice is still returned (just not suppressed)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE


# --- GUI preference ----------------------------------------------------------------------------


def test_gui_pref_round_trips() -> None:
    """An unset pref reads as None; a written choice reads back unchanged."""
    assert telemetry.read_gui_pref() is None
    telemetry.write_gui_pref(enabled=False)
    assert telemetry.read_gui_pref() is False
    telemetry.write_gui_pref(enabled=True)
    assert telemetry.read_gui_pref() is True


def test_gui_pref_unrecognized_reads_as_none(tmp_path: Path) -> None:
    """A garbled pref file is treated as 'no choice', not a crash."""
    pref = tmp_path / "state" / "photo-tagger" / "gui-telemetry"
    pref.parent.mkdir(parents=True)
    pref.write_text("maybe")
    assert telemetry.read_gui_pref() is None


def test_gui_choice_follows_the_config_and_marks_a_hand_check_as_opt_in() -> None:
    """No saved toggle defers to the config; a hand-check is an opt-in, a cleared toggle off."""
    assert telemetry.gui_choice(config_enabled=True) == (True, False)
    assert telemetry.gui_choice(config_enabled=False) == (False, False)
    telemetry.write_gui_pref(enabled=True)
    assert telemetry.gui_choice(config_enabled=True) == (True, True)
    telemetry.write_gui_pref(enabled=False)
    assert telemetry.gui_choice(config_enabled=True) == (False, False)


def test_gui_choice_lets_a_hand_check_beat_the_config_opt_out() -> None:
    """
    A toggle checked by hand beats ``enabled = false`` in the config file.

    The window writes that key on every settings save, so a later click is the newer choice. Without
    a hand check, the config's opt-out still wins.
    """
    assert telemetry.gui_choice(config_enabled=False) == (False, False)
    telemetry.write_gui_pref(enabled=True)
    assert telemetry.gui_choice(config_enabled=False) == (True, True)


def test_gui_choice_reads_a_legacy_on_as_no_opt_in(tmp_path: Path) -> None:
    """
    0.9 wrote "1" for any check, even one made while DO_NOT_TRACK kept the beacons off.

    Reading it as an opt-in would start sending for an upgraded DO_NOT_TRACK user.
    """
    pref = tmp_path / "state" / "photo-tagger" / "gui-telemetry"
    pref.parent.mkdir(parents=True)
    pref.write_text("1")
    assert telemetry.read_gui_pref() is True
    assert telemetry.gui_choice(config_enabled=True) == (True, False)


def test_gui_pref_write_degrades_when_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the state dir cannot be created, persisting the pref is swallowed, never raised."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    telemetry.write_gui_pref(enabled=False)  # must not raise


# --- payload contents (the privacy guarantee) --------------------------------------------------


_ENVELOPE_KEYS = {
    "schema_version",
    "event",
    "install_id",
    "session_id",
    "app_version",
    "interface",
    "command",
    "consent",
    "uptime_seconds",
    "arch",
    "os",
    "os_release",
    "python_version",
    "cpu",
    "gpu",
    "cpu_count",
    "memory_gb",
    "install_method",
    "installer",
    "system_locale",
    "utc_offset_minutes",
    "local_hour",
    "local_weekday",
    "ci",
    "terminal",
}

_RUN_KEYS = {
    "job",
    "trigger",
    "outcome",
    "error_kind",
    "provider",
    "model",
    "batch_size",
    "duration_seconds",
    "output_language",
    "ui_language",
    "file_types",
    "success_count",
    "failure_count",
    "initial_failures",
    "cache_hits",
    "retry_successes",
    "workers",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "inference_calls",
    "inference_seconds",
    "dry_run",
    "failure_kinds",
    "failure_types",
    "language_repairs",
    "flagged",
    "vocabulary_mapped",
    "vocabulary_dropped",
    "rested_seconds",
    "rests_taken",
    "shoots",
    "photo_seconds_median",
    "photo_seconds_p90",
    "keywords_mean",
    "title_chars_mean",
    "description_chars_mean",
    "power",
    "battery_percent",
}

_EXPECTED_CRASH_KEYS = _ENVELOPE_KEYS | {
    "exception_type",
    "exception_module",
    "exception_chain",
    "crash_location",
    "crash_frames",
    "crash_stack",
    "libraries",
}


def test_build_payload_has_exactly_the_allowlisted_keys() -> None:
    """The run beacon is a closed set: no field sneaks in unnoticed, and empty groups stay out."""
    payload = telemetry.build_payload(_sample_run())
    assert set(payload) == _ENVELOPE_KEYS | _RUN_KEYS
    assert payload["event"] == "run"
    assert payload["consent"] == telemetry.CONSENT_DEFAULT


def test_build_payload_reduces_the_model_and_the_language() -> None:
    """A model path keeps its file name, and a typed language name that is not known is "other"."""
    run = replace(
        _sample_run(),
        model="/Users/someone/models/qwen3-vl-30b.gguf",
        output_language="Klingon, but formal",
    )
    payload = telemetry.build_payload(run)
    assert payload["model"] == "qwen3-vl-30b.gguf"
    assert payload["output_language"] == "other"


def test_build_payload_nests_the_groups_a_run_carries() -> None:
    """Settings, setup, and the GUI groups travel as nested objects of their own fields only."""
    run = replace(
        _sample_run(),
        settings=telemetry.Settings(sidecar_mode="raw"),
        setup=telemetry.Setup(cli_flags="--dry-run"),
        gui=telemetry.GuiJob(hinted=2),
        review=telemetry.ReviewStats(edited=1),
    )
    payload = telemetry.build_payload(run, consent=telemetry.CONSENT_ENV_OPT_IN)
    assert set(payload) == _ENVELOPE_KEYS | _RUN_KEYS | {"settings", "setup", "gui", "review"}
    settings = payload["settings"]
    assert isinstance(settings, dict)
    assert set(settings) == {f.name for f in fields(telemetry.Settings)}
    assert settings["sidecar_mode"] == "raw"
    assert payload["setup"]["cli_flags"] == "--dry-run"  # type: ignore[index]
    assert payload["gui"]["hinted"] == 2  # type: ignore[index]  # noqa: PLR2004 - the sample count
    assert payload["consent"] == telemetry.CONSENT_ENV_OPT_IN


def test_build_payload_carries_run_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller-supplied run facts and the resolved platform values land in the payload."""
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    payload = telemetry.build_payload(_sample_run())
    assert payload["interface"] == "cli"
    assert payload["command"] == "tag"
    assert payload["model"] == "qwen/qwen3-vl-30b"
    assert payload["batch_size"] == 12  # noqa: PLR2004 - the sample run's count
    assert payload["arch"] == "arm64"
    assert payload["os"] == "Darwin"
    assert payload["app_version"] == __version__
    assert payload["schema_version"] == telemetry.SCHEMA_VERSION
    assert payload["session_id"] == telemetry.SESSION_ID


def test_build_payload_reports_the_power_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run says whether the machine was on its charger and how full the battery was."""
    monkeypatch.setattr(telemetry, "read_battery", lambda: BatteryState(percent=64, on_ac=False))
    payload = telemetry.build_payload(_sample_run())
    assert (payload["power"], payload["battery_percent"]) == ("battery", 64)
    monkeypatch.setattr(telemetry, "read_battery", lambda: None)
    assert telemetry.build_payload(_sample_run())["power"] == "none"


def test_build_payload_leaks_no_paths_or_content() -> None:
    """No payload value resembles a filesystem path; the ids are the only opaque tokens."""
    payload = telemetry.build_payload(_sample_run())
    for key, value in payload.items():
        if key in {"install_id", "session_id", "model"}:  # uuids and the (slashed) model id
            continue
        assert "/" not in str(value)
        assert "\\" not in str(value)
    uuid.UUID(str(payload["install_id"]))
    uuid.UUID(str(payload["session_id"]))


def test_build_payload_carries_language_and_file_types() -> None:
    """The language and file-type run facts reach the beacon body verbatim."""
    payload = telemetry.build_payload(_sample_run())
    assert payload["output_language"] == "English"
    assert payload["ui_language"] == "en"
    assert payload["file_types"] == "cr3,jpg"


def test_build_payload_carries_hardware_and_outcome_fields() -> None:
    """Hardware facts and per-run outcome counters land in the beacon body."""
    run = replace(
        _sample_run(),
        batch_size=10,
        success_count=9,
        failure_count=1,
        cache_hits=3,
        retry_successes=2,
        workers=4,
        total_tokens=12345,
        inference_seconds=41.5,
        dry_run=True,
        outcome=telemetry.OUTCOME_PARTIAL,
    )
    payload = telemetry.build_payload(run)
    assert payload["cpu"] == "Apple M3 Pro"
    assert payload["gpu"] == "Apple M3 Pro"
    assert payload["cpu_count"] == 12  # noqa: PLR2004 - the stubbed hardware fixture
    assert payload["memory_gb"] == 36  # noqa: PLR2004 - the stubbed hardware fixture
    assert payload["success_count"] == 9  # noqa: PLR2004 - the sample run's count
    assert payload["failure_count"] == 1
    assert payload["cache_hits"] == 3  # noqa: PLR2004 - the sample run's count
    assert payload["retry_successes"] == 2  # noqa: PLR2004 - the sample run's count
    assert payload["workers"] == 4  # noqa: PLR2004 - the sample run's count
    assert payload["total_tokens"] == 12345  # noqa: PLR2004 - the sample run's count
    assert payload["inference_seconds"] == 41.5  # noqa: PLR2004 - the sample run's value
    assert payload["dry_run"] == 1, "a v2 Worker reads dry_run as a number"
    assert payload["outcome"] == "partial"


def test_session_payload_keeps_only_allowlisted_features() -> None:
    """A feature name outside GUI_FEATURES never leaves the machine, whatever the caller passes."""
    session = telemetry.SessionInfo(
        interface="gui",
        command="gui",
        session_seconds=3600.0,
        busy_seconds=600.0,
        features={"viewer": 3, "grid": 0, "/Users/someone/secret": 1},
        display=telemetry.Display(screen="1512x982@2"),
    )
    payload = telemetry.build_payload(session)
    assert payload["event"] == "session"
    assert payload["features"] == {"viewer": 3}
    assert payload["display"]["screen"] == "1512x982@2"  # type: ignore[index]
    assert "settings" not in payload


def test_command_payload_keeps_only_allowlisted_scalar_details() -> None:
    """Command details are a closed set too, with strings clamped."""
    command = telemetry.CommandInfo(
        command="set",
        outcome=telemetry.OUTCOME_OK,
        exit_code=0,
        duration_seconds=1.25,
        details={"photos": 12, "written": 12, "path": "/secret", "source": "x" * 500},
    )
    payload = telemetry.build_payload(command)
    assert payload["event"] == "command"
    assert payload["interface"] == "cli"
    assert payload["command"] == "set"
    assert payload["details"] == {"photos": 12, "written": 12, "source": "x" * 200}


def test_launch_payload_is_a_gui_event() -> None:
    """The window opening reports itself as a GUI launch with its startup time."""
    payload = telemetry.build_payload(telemetry.LaunchInfo(startup_seconds=1.23456))
    assert (payload["event"], payload["interface"], payload["command"]) == ("launch", "gui", "gui")
    assert payload["startup_seconds"] == 1.235  # noqa: PLR2004 - rounded to three places


def test_outcome_for_names_how_a_batch_ended() -> None:
    """Interrupted beats everything; then empty, ok, failed, and partial by the failure count."""
    assert telemetry.outcome_for(5, 0, interrupted=True) == telemetry.OUTCOME_INTERRUPTED
    assert telemetry.outcome_for(0, 0) == telemetry.OUTCOME_EMPTY
    assert telemetry.outcome_for(5, 0) == telemetry.OUTCOME_OK
    assert telemetry.outcome_for(5, 5) == telemetry.OUTCOME_FAILED
    assert telemetry.outcome_for(5, 2) == telemetry.OUTCOME_PARTIAL


def test_notes_are_collected_for_the_command_beacon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Note() adds to the running command's facts; command_notes() hands out a copy."""
    monkeypatch.setattr(telemetry, "_notes", {})
    telemetry.note(photos=3)
    telemetry.note(written=2)
    notes = telemetry.command_notes()
    assert notes == {"photos": 3, "written": 2}
    notes["photos"] = 99
    assert telemetry.command_notes()["photos"] == 3  # noqa: PLR2004 - the noted count


def test_sent_any_turns_on_with_the_first_beacon(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's fallback beacon relies on this flag; a disabled emit must not set it."""
    monkeypatch.setattr(telemetry, "_sent_any", False)
    monkeypatch.setattr(httpx2, "post", lambda *_a, **_k: object())
    telemetry.emit(_sample_run(), enabled=False)
    assert telemetry.sent_any() is False
    telemetry.emit(_sample_run(), enabled=True, block=True)
    assert telemetry.sent_any() is True


def test_locale_hint_is_used_for_the_system_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GUI hands over Qt's locale, which knows the OS language when LANG is unset."""
    monkeypatch.setattr(telemetry, "_locale_hint", None)
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)
    telemetry.set_locale_hint("pt_BR")
    assert telemetry.build_payload(_sample_run())["system_locale"] == "pt_BR"


def test_failure_kinds_summary_is_compact_sorted_and_skips_zeros() -> None:
    """The per-run failure buckets encode as a stable "kind:count" list."""
    assert telemetry.failure_kinds_summary({}) == ""
    assert (
        telemetry.failure_kinds_summary({"timeout": 3, "other": 1, "connection": 0})
        == "other:1,timeout:3"
    )


def test_file_types_summary_is_sorted_lowercased_and_deduplicated() -> None:
    """Extensions collapse to a sorted, lowercased, dot-stripped, comma-joined set."""
    paths = [Path("/photos/a.CR3"), Path("/photos/b.jpg"), Path("/photos/c.cr3")]
    assert telemetry.file_types_summary(paths) == "cr3,jpg"


def test_file_types_summary_reveals_no_filenames() -> None:
    """Only extensions leave the machine: the summary carries no path or filename."""
    summary = telemetry.file_types_summary([Path("/private/vacation-2026/IMG_0001.DNG")])
    assert summary == "dng"


def test_file_types_summary_names_unknown_extensions_other() -> None:
    """An unusual suffix is counted as "other" rather than sent as typed."""
    summary = telemetry.file_types_summary([Path("/p/a.jpg"), Path("/p/b.secret")])
    assert summary == "jpg,other"


def test_file_types_summary_handles_empty_and_no_suffix() -> None:
    """No paths, or paths without a suffix, yield an empty string rather than a stray comma."""
    assert telemetry.file_types_summary([]) == ""
    assert telemetry.file_types_summary([Path("/photos/README")]) == ""


# --- crash reporting ---------------------------------------------------------------------------


def _package_exception() -> BaseException:
    """Raise (and catch) an error whose traceback passes through a real photo_tagger module."""
    from photo_tagger.keywords import parse_hierarchical_keyword  # noqa: PLC0415 - test-local

    try:
        # Deliberate misuse so a real traceback through a photo_tagger module exists.
        parse_hierarchical_keyword(None)  # type: ignore[arg-type]
    except AttributeError as exc:
        return exc
    msg = "expected parse_hierarchical_keyword(None) to raise"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def test_crash_summary_reports_type_and_in_package_location() -> None:
    """The summary names the exception class and the deepest photo_tagger frame."""
    name, location, frames = telemetry.crash_summary(_package_exception())
    assert name == "AttributeError"
    assert location.startswith("keywords:parse_hierarchical_keyword:")
    assert location in frames


def test_crash_summary_skips_frames_outside_the_package() -> None:
    """Test-file (and stdlib) frames never appear; only photo_tagger code coordinates do."""
    _, _, frames = telemetry.crash_summary(_package_exception())
    assert "test_telemetry" not in frames


def test_crash_summary_never_carries_the_message() -> None:
    """
    The exception message is dropped by construction.

    Messages routinely embed file paths ("could not open /Users/x/IMG.CR3"), so no part of the
    summary, and nothing in the whole crash beacon, may contain it.
    """
    try:
        msg = "/Users/someone/secret/IMG_0001.CR3 could not be read"
        raise RuntimeError(msg)  # noqa: TRY301 - the test needs a real traceback.
    except RuntimeError as exc:
        summary = telemetry.crash_summary(exc)
        payload = json.dumps(telemetry.build_crash_payload(exc, interface="cli"))
    assert all("/Users/" not in part and "IMG_0001" not in part for part in summary)
    # Raised in this test file, outside the package: no code coordinates at all.
    assert summary == ("RuntimeError", "", "")
    assert "/Users/" not in payload
    assert "IMG_0001" not in payload


def test_build_crash_payload_has_exactly_the_allowlisted_keys() -> None:
    """The crash beacon is a closed set too, with the crash event marker and the command."""
    payload = telemetry.build_crash_payload(_package_exception(), interface="cli", command="show")
    assert set(payload) == _EXPECTED_CRASH_KEYS
    assert payload["event"] == "crash"
    assert payload["command"] == "show"
    assert payload["exception_type"] == "AttributeError"


def test_crash_payload_carries_the_whole_stack_by_module_never_by_path() -> None:
    """Library and test frames appear by module name, the package's by its own, and no path."""
    payload = telemetry.build_crash_payload(_package_exception(), interface="gui")
    stack = str(payload["crash_stack"])
    frames = stack.split(">")
    assert frames[-1].startswith("photo_tagger.keywords:parse_hierarchical_keyword:")
    assert any(":_package_exception:" in frame for frame in frames)
    assert "/" not in stack
    assert "\\" not in stack
    assert (payload["exception_module"], payload["exception_chain"]) == (
        "builtins",
        "AttributeError",
    )
    assert "pydantic=" in str(payload["libraries"])


def test_emit_crash_posts_when_enabled_and_respects_optout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enabled crash beacon POSTs; the config opt-out short-circuits before any network."""
    posted: list[dict[str, object]] = []
    monkeypatch.setattr(httpx2, "post", lambda _url, *, json, timeout: posted.append(json))  # noqa: ARG005 - httpx2.post signature

    assert telemetry.emit_crash(_package_exception(), interface="cli", enabled=False) is None
    assert posted == []

    thread = telemetry.emit_crash(_package_exception(), interface="cli", enabled=True)
    assert thread is not None
    assert posted[0]["event"] == "crash"


def test_emit_crash_is_capped_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash loop cannot spam the collector: beacons stop after the per-process cap."""
    posted: list[object] = []
    monkeypatch.setattr(httpx2, "post", lambda *_a, **k: posted.append(k))
    exc = _package_exception()
    sent = [telemetry.emit_crash(exc, interface="gui", enabled=True) for _ in range(5)]
    assert sum(thread is not None for thread in sent) == 3  # noqa: PLR2004 - the cap
    assert len(posted) == 3  # noqa: PLR2004 - the cap


# --- emit (sending) ----------------------------------------------------------------------------


def test_emit_returns_none_and_sends_nothing_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled config short-circuits before any network attempt."""
    calls: list[object] = []
    monkeypatch.setattr(httpx2, "post", lambda *a, **k: calls.append((a, k)))
    assert telemetry.emit(_sample_run(), enabled=False) is None
    assert calls == []


def test_emit_posts_the_payload_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """An enabled run POSTs the built payload to the collector endpoint."""
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> object:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return object()

    monkeypatch.setattr(httpx2, "post", fake_post)
    thread = telemetry.emit(_sample_run(), enabled=True, block=True)
    assert thread is not None
    assert captured["url"] == telemetry.ENDPOINT
    assert captured["json"]["interface"] == "cli"  # type: ignore[index]


def test_emit_non_blocking_returns_started_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default non-blocking path returns the worker thread without waiting on it."""
    monkeypatch.setattr(httpx2, "post", lambda *_a, **_k: object())
    thread = telemetry.emit(_sample_run(), enabled=True)
    assert thread is not None
    thread.join(timeout=2.0)  # tidy up so the post lands before the test ends


def test_emit_swallows_payload_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A failure while assembling the payload sends nothing and never escapes to the caller.

    The builder now runs inside the background thread (see
    test_emit_does_not_block_the_caller_while_building), so emit() itself cannot know in advance
    whether the build will fail; it still returns the started thread, not None.
    """

    def boom(_run: RunInfo, **_kwargs: object) -> dict[str, object]:
        msg = "platform probe blew up"
        raise RuntimeError(msg)

    posted: list[object] = []
    monkeypatch.setattr(telemetry, "build_payload", boom)
    monkeypatch.setattr(httpx2, "post", lambda *a, **k: posted.append((a, k)))
    thread = telemetry.emit(_sample_run(), enabled=True, block=True)
    assert thread is not None
    assert posted == []


def test_emit_does_not_block_the_caller_while_building(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Building the payload never happens on the caller's thread.

    build_payload can shell out to probe hardware on a cache miss; the non-blocking path must return
    immediately regardless of how long that takes, or a cold hardware-probe cache would stall the
    tagging run right as it is about to exit.
    """
    finished = threading.Event()

    def slow_build(_run: RunInfo, **_kwargs: object) -> dict[str, object]:
        time.sleep(0.2)
        finished.set()
        return {}

    monkeypatch.setattr(telemetry, "build_payload", slow_build)
    monkeypatch.setattr(httpx2, "post", lambda *_a, **_k: object())

    start = time.monotonic()
    thread = telemetry.emit(_sample_run(), enabled=True, block=False)
    elapsed = time.monotonic() - start

    assert thread is not None
    assert elapsed < 0.1  # noqa: PLR2004 - well under slow_build's 0.2s sleep
    assert not finished.is_set()  # the builder is still running on the background thread
    thread.join(timeout=2.0)
    assert finished.is_set()


def test_emit_swallows_send_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network failure inside the beacon never propagates to the caller."""

    def boom(*_a: object, **_k: object) -> object:
        msg = "network down"
        raise httpx2.ConnectError(msg)

    monkeypatch.setattr(httpx2, "post", boom)
    # block=True joins the worker; the test passes simply by not raising.
    telemetry.emit(_sample_run(), enabled=True, block=True)


def test_install_id_persists_where_hard_links_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExFAT and some SMB shares refuse os.link; the id must still be the same on the next run."""

    def refuse(*_args: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr("os.link", refuse)
    first = telemetry.install_id()

    assert telemetry.install_id() == first
    folder = tmp_path / "state" / "photo-tagger"
    assert (folder / "install-id").read_text(encoding="utf-8").strip() == first
    assert not list(folder.glob("install-id.*.tmp"))


def test_install_id_file_never_appears_empty(tmp_path: Path) -> None:
    """The id file is linked into place whole, so a reader never finds it empty."""
    telemetry.install_id()
    folder = tmp_path / "state" / "photo-tagger"
    assert (folder / "install-id").read_text(encoding="utf-8").strip()
    assert not list(folder.glob("install-id.*.tmp"))
