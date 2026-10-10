"""Tests for the rest-between-work-windows pacer and its battery probes."""

import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from photo_tagger.pacing import (
    BatteryState,
    Pacer,
    PacingPlan,
    _read_battery_windows,
    _run_probe,
    build_pacer,
    parse_pmset,
    read_battery,
    read_battery_sysfs,
)


if TYPE_CHECKING:
    from pathlib import Path


_MAC_ON_CHARGER = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=23265379)\t75%; charging; 1:58 remaining present: true\n"
)
_MAC_ON_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=23265379)\t41%; discharging; 2:10 remaining present: true\n"
)
_MAC_DESKTOP = "Now drawing from 'AC Power'\n"

# A ten-minute work window, and the rest a 50 percent duty cycle earns from it.
_WINDOW_SECONDS = 600.0
_EARNED_REST_SECONDS = 600.0
_FLOOR = 30
_RESUME_FROM_FLOOR = 45
_EXPLICIT_RESUME = 80
_FULL_CHARGE = 100
_LONG_IDLE_SECONDS = 10_000.0
_NAPS_BEFORE_CANCEL = 3


class _FakeClock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        """Return the current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake time instead of waiting."""
        self.now += seconds

    def work(self, seconds: float) -> None:
        """Advance the fake time as if the run had been tagging photos."""
        self.now += seconds


def _pacer(plan: PacingPlan, clock: _FakeClock, battery: BatteryState | None = None) -> Pacer:
    """Build a pacer driven by *clock*, reporting *battery* on every probe."""
    return Pacer(
        plan,
        battery_reader=lambda: battery,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_parse_pmset_reads_charge_and_power_source() -> None:
    """A laptop line yields the percentage, and the header yields the power source."""
    assert parse_pmset(_MAC_ON_CHARGER) == BatteryState(percent=75, on_ac=True)
    assert parse_pmset(_MAC_ON_BATTERY) == BatteryState(percent=41, on_ac=False)


def test_parse_pmset_without_battery_is_none() -> None:
    """A desktop Mac prints no battery line, which reads as "no battery"."""
    assert parse_pmset(_MAC_DESKTOP) is None
    assert parse_pmset("") is None


def test_read_battery_sysfs_reads_first_battery_and_mains(tmp_path: Path) -> None:
    """The first Battery gives the charge; an online Mains means a charger is plugged in."""
    battery = tmp_path / "BAT0"
    battery.mkdir()
    (battery / "type").write_text("Battery\n", encoding="utf-8")
    (battery / "capacity").write_text("62\n", encoding="utf-8")
    mains = tmp_path / "AC"
    mains.mkdir()
    (mains / "type").write_text("Mains\n", encoding="utf-8")
    (mains / "online").write_text("1\n", encoding="utf-8")

    assert read_battery_sysfs(tmp_path) == BatteryState(percent=62, on_ac=True)


def _supply(root: Path, name: str, **files: str) -> None:
    """Create one sysfs power supply directory under *root* with the given attribute files."""
    supply = root / name
    supply.mkdir()
    for attribute, value in files.items():
        (supply / attribute).write_text(f"{value}\n", encoding="utf-8")


def test_read_battery_sysfs_skips_a_peripheral_battery(tmp_path: Path) -> None:
    """A wireless mouse sorts first but is not the laptop's battery."""
    _supply(tmp_path, "0-mouse", type="Battery", scope="Device", capacity="5")
    _supply(tmp_path, "BAT0", type="Battery", capacity="62", status="Discharging")

    assert read_battery_sysfs(tmp_path) == BatteryState(percent=62, on_ac=False)


def test_read_battery_sysfs_counts_a_usb_c_charger(tmp_path: Path) -> None:
    """A laptop charged over USB-C lists its charger as a USB supply, not as Mains."""
    _supply(tmp_path, "BAT0", type="Battery", capacity="40", status="Unknown")
    _supply(tmp_path, "ucsi-source-psy", type="USB", online="1")

    assert read_battery_sysfs(tmp_path) == BatteryState(percent=40, on_ac=True)


def test_read_battery_sysfs_reads_a_charging_status_as_a_charger(tmp_path: Path) -> None:
    """A laptop that lists no charger at all still says so through its battery's status."""
    _supply(tmp_path, "BAT0", type="Battery", capacity="80", status="Not charging")

    assert read_battery_sysfs(tmp_path) == BatteryState(percent=80, on_ac=True)


def test_read_battery_sysfs_without_battery_is_none(tmp_path: Path) -> None:
    """A machine with only a mains supply, or no sysfs at all, reports no battery."""
    mains = tmp_path / "AC"
    mains.mkdir()
    (mains / "type").write_text("Mains\n", encoding="utf-8")
    (mains / "online").write_text("0\n", encoding="utf-8")
    # A supply that declares nothing at all, as some virtual machines expose.
    (tmp_path / "mystery").mkdir()

    assert read_battery_sysfs(tmp_path) is None
    assert read_battery_sysfs(tmp_path / "missing") is None


def test_read_battery_sysfs_survives_unreadable_capacity(tmp_path: Path) -> None:
    """A capacity that is not a number leaves the battery unknown instead of raising."""
    battery = tmp_path / "BAT0"
    battery.mkdir()
    (battery / "type").write_text("Battery\n", encoding="utf-8")
    (battery / "capacity").write_text("unknown\n", encoding="utf-8")

    assert read_battery_sysfs(tmp_path) is None


def test_read_battery_on_unknown_platform_is_none() -> None:
    """An OS with no probe of its own never blocks a run."""
    with patch("photo_tagger.pacing.platform.system", return_value="Haiku"):
        assert read_battery() is None


def test_read_battery_on_macos_runs_pmset() -> None:
    """Reading the charge on macOS shells out to pmset."""
    with (
        patch("photo_tagger.pacing.platform.system", return_value="Darwin"),
        patch("photo_tagger.pacing._run_probe", return_value=_MAC_ON_CHARGER) as probe,
    ):
        assert read_battery() == BatteryState(percent=75, on_ac=True)
    assert probe.call_args.args[0][0] == "pmset"


def test_read_battery_dispatches_per_platform() -> None:
    """Linux and Windows reach their own probe instead of the macOS one."""
    with (
        patch("photo_tagger.pacing.platform.system", return_value="Linux"),
        patch("photo_tagger.pacing.read_battery_sysfs", return_value=None) as sysfs,
    ):
        assert read_battery() is None
    assert sysfs.called

    with (
        patch("photo_tagger.pacing.platform.system", return_value="Windows"),
        patch("photo_tagger.pacing._read_battery_windows", return_value=None) as windows,
    ):
        assert read_battery() is None
    assert windows.called


def test_windows_probe_is_none_off_windows() -> None:
    """There is no kernel32 to ask here, and a probe that cannot run returns nothing."""
    assert _read_battery_windows() is None


def test_run_probe_returns_empty_without_the_program() -> None:
    """A tool that is not installed yields "", never an exception."""
    with patch("photo_tagger.programs.shutil.which", return_value=None):
        assert _run_probe(["pmset", "-g", "batt"]) == ""


def test_run_probe_returns_empty_on_failure() -> None:
    """A non-zero exit, a timeout, or an OS error all read as "nothing to report"."""
    with (
        patch("photo_tagger.programs.shutil.which", return_value="/usr/bin/pmset"),
        patch("photo_tagger.pacing.subprocess.run") as run,
    ):
        run.return_value = SimpleNamespace(returncode=1, stdout="whatever")
        assert _run_probe(["pmset"]) == ""
        run.side_effect = OSError("no such tool")
        assert _run_probe(["pmset"]) == ""
        run.side_effect = subprocess.TimeoutExpired(cmd="pmset", timeout=3.0)
        assert _run_probe(["pmset"]) == ""


def test_run_probe_returns_stdout() -> None:
    """A clean exit hands back the trimmed output."""
    with (
        patch("photo_tagger.programs.shutil.which", return_value="/usr/bin/pmset"),
        patch("photo_tagger.pacing.subprocess.run") as run,
    ):
        run.return_value = SimpleNamespace(returncode=0, stdout=" 75%\n")
        assert _run_probe(["pmset"]) == "75%"


def test_plan_off_by_default() -> None:
    """The built-in plan rests nothing, so an unconfigured run is unaffected."""
    assert PacingPlan().rests is False
    assert build_pacer(PacingPlan()) is None


def test_plan_earns_rest_in_proportion_to_work() -> None:
    """At 50 percent duty, ten minutes of work earns ten minutes of rest."""
    plan = PacingPlan(duty_percent=50, work_seconds=600)
    assert plan.rest_for(600) == pytest.approx(600)
    assert PacingPlan(duty_percent=75, work_seconds=600).rest_for(600) == pytest.approx(200)


def test_plan_caps_the_rest_it_earns() -> None:
    """No single rest may exceed the cap, however long the window ran over."""
    plan = PacingPlan(duty_percent=10, work_seconds=600, max_pause_seconds=900)
    assert plan.rest_for(600) == pytest.approx(900)


def test_plan_resume_level_derived_from_floor() -> None:
    """A floor with no resume level of its own resumes a margin above it, never past full."""
    assert PacingPlan(battery_floor=_FLOOR).resume_percent == _RESUME_FROM_FLOOR
    explicit = PacingPlan(battery_floor=_FLOOR, battery_resume=_EXPLICIT_RESUME)
    assert explicit.resume_percent == _EXPLICIT_RESUME
    assert PacingPlan(battery_floor=95).resume_percent == _FULL_CHARGE
    # A resume level at or below the floor would rest again one photo later.
    below = PacingPlan(battery_floor=_FLOOR, battery_resume=20)
    assert below.resume_percent == _RESUME_FROM_FLOOR


def test_pacer_without_a_plan_never_sleeps() -> None:
    """A plan that rests nothing returns immediately and leaves the clock alone."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(), clock)
    clock.work(_LONG_IDLE_SECONDS)

    assert pacer.wait_before_next() is True
    assert clock.now == _LONG_IDLE_SECONDS
    assert pacer.rested_seconds == 0


def test_pacer_rests_once_the_work_window_is_full() -> None:
    """Photos run untouched until the window is full, then the run rests its earned share."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(duty_percent=50, work_seconds=600), clock)

    assert pacer.wait_before_next() is True  # starts the clock
    clock.work(300)
    assert pacer.wait_before_next() is True
    assert pacer.rested_seconds == 0

    clock.work(300)
    assert pacer.wait_before_next() is True
    assert pacer.rested_seconds == pytest.approx(600, abs=1)
    assert pacer.rests_taken == 1


def test_pacer_starts_a_fresh_window_after_a_rest() -> None:
    """The next window is measured from the end of the rest, not from the start of the run."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(duty_percent=50, work_seconds=600), clock)
    pacer.wait_before_next()
    clock.work(600)
    pacer.wait_before_next()

    clock.work(100)
    pacer.wait_before_next()
    assert pacer.rests_taken == 1


def test_pacer_waits_for_the_battery_to_recover() -> None:
    """A charge at the floor holds the run until a later reading has climbed back."""
    clock = _FakeClock()
    # Three readings at the floor (one per poll interval), then a charged one.
    charge = iter([BatteryState(30, on_ac=True)] * 3 + [BatteryState(45, on_ac=True)] * 10)
    pacer = Pacer(
        PacingPlan(duty_percent=100, battery_floor=30, work_seconds=600, max_pause_seconds=900),
        battery_reader=lambda: next(charge),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    pacer.wait_before_next()
    clock.work(600)
    pacer.wait_before_next()

    assert pacer.rests_taken == 1
    # Two further polls, thirty seconds apart, before the charge came back. Well short of the cap.
    assert pacer.rested_seconds == pytest.approx(90, abs=1)


def test_pacer_battery_rest_waits_for_the_resume_level() -> None:
    """
    A rest that began at the floor lasts until the resume level, not one percent above the floor.

    Otherwise the pair is no band, and the run rests again after the very next window.
    """
    clock = _FakeClock()
    # One percent per poll, from the floor up.
    charge = iter(BatteryState(percent, on_ac=True) for percent in range(_FLOOR, _FULL_CHARGE))
    pacer = Pacer(
        PacingPlan(battery_floor=_FLOOR, battery_resume=_FLOOR + 5, max_pause_seconds=900),
        battery_reader=lambda: next(charge),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    pacer.wait_before_next()
    clock.work(_WINDOW_SECONDS)
    pacer.wait_before_next()

    # Five more polls, thirty seconds apart, from 30 to 35.
    assert pacer.rested_seconds == pytest.approx(150, abs=1)


def test_pacer_duty_rest_that_finds_the_battery_low_waits_for_it() -> None:
    """A duty-cycle rest that ends with the charge at the floor keeps resting until it is back."""
    clock = _FakeClock()
    readings = iter(
        [BatteryState(_FLOOR + 1, on_ac=True)]
        + [BatteryState(_FLOOR, on_ac=True)] * 25
        + [BatteryState(_RESUME_FROM_FLOOR, on_ac=True)] * 10,
    )
    pacer = Pacer(
        PacingPlan(duty_percent=50, battery_floor=_FLOOR, max_pause_seconds=3600),
        battery_reader=lambda: next(readings),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    pacer.wait_before_next()
    clock.work(_WINDOW_SECONDS)
    pacer.wait_before_next()

    assert pacer.rested_seconds > _EARNED_REST_SECONDS


def test_pacer_pause_does_not_count_idle_time_as_work() -> None:
    """Between two batches of a watch the clock is stopped, so an import after hours never rests."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(duty_percent=50), clock)
    pacer.wait_before_next()
    clock.work(_WINDOW_SECONDS / 5)
    pacer.pause()
    clock.work(_LONG_IDLE_SECONDS)

    pacer.wait_before_next()
    pacer.wait_before_next()

    assert pacer.rests_taken == 0


def test_pacer_pause_carries_the_window_over_a_short_gap() -> None:
    """A gap shorter than the rest the window earned does not reset it: the work adds up."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(duty_percent=50), clock)
    pacer.wait_before_next()
    clock.work(_WINDOW_SECONDS / 2)
    pacer.pause()
    clock.work(1)

    pacer.wait_before_next()
    clock.work(_WINDOW_SECONDS / 2)
    pacer.wait_before_next()

    assert pacer.rests_taken == 1


def test_pacer_pause_without_a_window_is_harmless() -> None:
    """A batch that never reached its first photo has no window to stop."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(duty_percent=50), clock)
    pacer.pause()

    assert pacer.wait_before_next() is True


def test_pacer_gives_up_on_a_battery_that_never_recovers() -> None:
    """A rest ends at the cap even with the charge still low, so a run is never stuck."""
    clock = _FakeClock()
    pacer = _pacer(
        PacingPlan(battery_floor=30, work_seconds=600, max_pause_seconds=120),
        clock,
        BatteryState(12, on_ac=True),
    )
    pacer.wait_before_next()
    clock.work(600)
    pacer.wait_before_next()

    assert pacer.rested_seconds == pytest.approx(120, abs=1)


def test_pacer_ignores_a_low_battery_with_no_charger() -> None:
    """On battery power a rest recharges nothing, so the run keeps going."""
    clock = _FakeClock()
    pacer = _pacer(
        PacingPlan(battery_floor=30, work_seconds=600),
        clock,
        BatteryState(12, on_ac=False),
    )
    pacer.wait_before_next()
    clock.work(600)

    assert pacer.wait_before_next() is True
    assert pacer.rested_seconds == 0


def test_pacer_ignores_a_machine_with_no_battery() -> None:
    """A failed or absent probe means duty pacing only."""
    clock = _FakeClock()
    pacer = _pacer(PacingPlan(battery_floor=30, work_seconds=600), clock, None)
    pacer.wait_before_next()
    clock.work(600)

    assert pacer.wait_before_next() is True
    assert pacer.rested_seconds == 0


def test_pacer_cancel_ends_the_rest_and_every_later_wait() -> None:
    """Cancelling wakes the sleeper and turns every later call into a "stop now" answer."""
    clock = _FakeClock()
    plan = PacingPlan(duty_percent=50, work_seconds=600)
    pacer = Pacer(plan, monotonic=clock.monotonic, sleep=clock.sleep)
    pacer.wait_before_next()
    clock.work(600)
    pacer.cancel()

    assert pacer.wait_before_next() is False
    assert pacer.rested_seconds == 0


def test_pacer_cancel_cuts_a_rest_short() -> None:
    """A cancel raised while the run is resting ends the sleep at the next nap."""
    clock = _FakeClock()
    plan = PacingPlan(duty_percent=50, work_seconds=600, max_pause_seconds=600)
    pacer = Pacer(plan, monotonic=clock.monotonic, sleep=clock.sleep)

    naps = 0
    running: list[Pacer] = []

    def cancel_after_three_naps(seconds: float) -> None:
        nonlocal naps
        naps += 1
        clock.sleep(seconds)
        if naps == _NAPS_BEFORE_CANCEL:
            running[0].cancel()

    pacer = Pacer(plan, monotonic=clock.monotonic, sleep=cancel_after_three_naps)
    running.append(pacer)
    pacer.wait_before_next()
    clock.work(600)

    assert pacer.wait_before_next() is False
    assert pacer.rested_seconds < _EARNED_REST_SECONDS


def test_pacer_reports_each_rest_to_its_callback() -> None:
    """The UI hook hears the reason and the expected length of every rest."""
    clock = _FakeClock()
    seen: list[tuple[str, float]] = []
    pacer = Pacer(
        PacingPlan(duty_percent=50, work_seconds=600),
        battery_reader=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        on_rest=lambda reason, seconds: seen.append((reason, seconds)),
    )
    pacer.wait_before_next()
    clock.work(600)
    pacer.wait_before_next()

    assert len(seen) == 1
    assert seen[0][0] == "duty"
    assert seen[0][1] == pytest.approx(_EARNED_REST_SECONDS)


def test_pacer_survives_a_broken_callback() -> None:
    """A callback that raises is logged, not fatal: the run must outlive its own UI."""
    clock = _FakeClock()

    def explode(reason: str, seconds: float) -> None:
        msg = f"no {reason} for {seconds}"
        raise RuntimeError(msg)

    pacer = Pacer(
        PacingPlan(duty_percent=50, work_seconds=600),
        battery_reader=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        on_rest=explode,
    )
    pacer.wait_before_next()
    clock.work(600)

    assert pacer.wait_before_next() is True
    assert pacer.rests_taken == 1


def test_pacer_reuses_a_recent_battery_reading() -> None:
    """The probe is polled at most every half minute, not once per photo."""
    clock = _FakeClock()
    reads = 0

    def count_reads() -> BatteryState:
        nonlocal reads
        reads += 1
        return BatteryState(80, on_ac=True)

    pacer = Pacer(
        PacingPlan(battery_floor=30, work_seconds=5),
        battery_reader=count_reads,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(4):
        pacer.wait_before_next()
        clock.work(5)

    assert reads == 1


def test_read_battery_sysfs_trusts_an_offline_charger_over_the_battery_status(
    tmp_path: Path,
) -> None:
    """Some firmware says "Not charging" on battery; a listed charger that is offline decides."""
    _supply(tmp_path, "AC", type="Mains", online="0")
    _supply(tmp_path, "BAT0", type="Battery", capacity="50", status="Not charging")

    assert read_battery_sysfs(tmp_path) == BatteryState(percent=50, on_ac=False)


def test_pacer_checks_the_battery_when_a_window_starts_after_a_pause() -> None:
    """Short watch batches never fill a window, so a low charge is caught as the next one starts."""
    clock = _FakeClock()
    readings = iter(
        [BatteryState(_FLOOR - 10, on_ac=True)] * 3 + [BatteryState(90, on_ac=True)] * 50,
    )
    pacer = Pacer(
        PacingPlan(battery_floor=_FLOOR, max_pause_seconds=3600),
        battery_reader=lambda: next(readings),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    pacer.wait_before_next()
    clock.work(60)
    pacer.pause()
    clock.work(1)

    pacer.wait_before_next()

    assert pacer.rests_taken == 1
