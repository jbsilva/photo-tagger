"""
Rest a run now and then, so it stays inside the machine's power and heat budget.

A vision model at full tilt can ask for more power than the charger supplies. The laptop then
discharges while plugged in, and a long run can flatten the battery before it finishes. The same
load runs hot, spins the fans up, and leaves little of the machine for anything else.

Resting fixes all three, because what counts is average power, not peak power. Work for a while,
then stop for a while: the battery charges back up, the fans wind down, and the average draw lands
under what the supply provides.

Two inputs decide when to rest, and both are read at the same moment, the end of a work window:

- The duty cycle. After working for ``work_seconds``, rest long enough to keep working time at the
  configured share of the total. No battery needed, so a desktop gets this too.
- The battery charge. At or below ``battery_floor``, rest until it climbs back to the resume level.
  That needs no wattage numbers from the user, and it follows screen brightness and whatever else
  the machine is doing without being told.

Deciding once per window rather than once per photo is deliberate. A rest after every photo would
be too short to charge anything, and it would leave the model server idle in small doses all day,
which is how a server that unloads idle models ends up reloading one. Every rest is capped for the
same reason: LM Studio, for one, drops a JIT-loaded model after an hour.

Reading the battery is best-effort on every platform. A machine with no battery, or one this module
cannot read, simply never rests for battery reasons; the duty cycle still applies.
"""

import contextlib
import platform
import re
import shutil
import subprocess  # nosec B404 - only used to ask macOS for the battery charge
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from photo_tagger.config import (
    DEFAULT_BATTERY_FLOOR,
    DEFAULT_BATTERY_RESUME,
    DEFAULT_DUTY_PERCENT,
    DEFAULT_MAX_PAUSE_MINUTES,
    DEFAULT_WORK_MINUTES,
)


if TYPE_CHECKING:
    from collections.abc import Callable


SECONDS_PER_MINUTE = 60.0

# A duty cycle of 100 is "work all the time", which is the same as no duty cycle at all. Public
# because the GUI's live summary asks the same question of a plan.
FULL_DUTY = 100
_FULL_CHARGE = 100

# Resume this far above the floor when the user gave a floor but no resume level. Small enough to
# be reached in minutes on a charger that is winning, large enough that the run does not resume
# straight back into another rest.
_RESUME_MARGIN = 15

# The highest floor that still leaves room for a resume level above it, which every battery rest
# needs to end on something other than the cap.
MAX_BATTERY_FLOOR = 99

# Above this, a floor is worth warning about rather than refusing. A plugged-in laptop spends much
# of its life short of full (on macOS, optimized charging parks it around 80), so a floor up here
# sits under the charge nearly all the time: the run rests after almost every window and waits out
# the cap, which is halting rather than pacing. Someone may still mean it, so both frontends say so
# and carry on.
CAUTION_BATTERY_FLOOR = 90

# How long one battery reading is reused. Under load the charge moves by about a percent a minute,
# so a reading this old still answers the question, and on macOS the probe costs a subprocess.
_BATTERY_POLL_SECONDS = 30.0

# How long a rest sleeps before looking at the cancel flag again. Short enough that Ctrl-C and the
# GUI's Stop button feel immediate, long enough to cost nothing.
_NAP_SECONDS = 0.5

_PROBE_TIMEOUT_SECONDS = 3.0

# The charge in "75%; charging; 1:58 remaining present: true".
_PERCENT_PATTERN = re.compile(r"(\d{1,3})%")

_POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")


@dataclass(slots=True, frozen=True)
class BatteryState:
    """What the battery is doing: its charge in percent, and whether a charger is plugged in."""

    percent: int
    on_ac: bool


@dataclass(slots=True, frozen=True)
class PacingPlan:
    """
    How much of the time a run may work, and how low the battery may get before it rests.

    The defaults pace nothing. A run rests only once the user asks for a duty cycle below 100 or
    sets a battery floor.
    """

    duty_percent: int = DEFAULT_DUTY_PERCENT
    battery_floor: int = DEFAULT_BATTERY_FLOOR
    battery_resume: int = DEFAULT_BATTERY_RESUME
    work_seconds: float = DEFAULT_WORK_MINUTES * SECONDS_PER_MINUTE
    max_pause_seconds: float = DEFAULT_MAX_PAUSE_MINUTES * SECONDS_PER_MINUTE

    @property
    def rests(self) -> bool:
        """True when this plan can ever stop a run."""
        return self.duty_percent < FULL_DUTY or self.battery_floor > 0

    @property
    def resume_percent(self) -> int:
        """
        The charge a battery rest waits for.

        Derived from the floor when the user gave no resume level of its own, and never below it:
        resuming at the floor would rest again a photo later.
        """
        if self.battery_resume > self.battery_floor:
            return min(self.battery_resume, _FULL_CHARGE)
        return min(self.battery_floor + _RESUME_MARGIN, _FULL_CHARGE)

    def rest_for(self, worked_seconds: float) -> float:
        """
        Seconds of rest that *worked_seconds* of work has earned under the duty cycle.

        Measured against the time actually worked rather than the configured window, so a batch of
        slow photos that overruns the window still earns its proportional rest.
        """
        if self.duty_percent >= FULL_DUTY:
            return 0.0
        duty = max(self.duty_percent, 1)
        earned = worked_seconds * (FULL_DUTY - duty) / duty
        return min(earned, self.max_pause_seconds)


def _run_probe(args: list[str]) -> str:
    """
    Run *args* and return stripped stdout, or "" on any failure.

    The program is resolved with ``shutil.which`` so the absolute path is what gets executed, rather
    than letting the OS search a path that (on Windows) starts with the current directory.
    """
    program = shutil.which(args[0])
    if program is None:
        return ""
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv from constants above
            [program, *args[1:]],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def parse_pmset(text: str) -> BatteryState | None:
    """
    Read a charge out of ``pmset -g batt`` output; None when the machine reports no battery.

    The output is a power source line followed by one line per battery::

        Now drawing from 'AC Power'
         -InternalBattery-0 (id=23265379)       75%; charging; 1:58 remaining present: true

    A desktop Mac prints the first line only, which is how "no battery" reads here.
    """
    match = _PERCENT_PATTERN.search(text)
    if match is None:
        return None
    return BatteryState(percent=int(match.group(1)), on_ac="'AC Power'" in text)


def _first_line(path: Path) -> str:
    """Return the first line of *path* stripped, or "" when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# A battery's own status that means power is coming in: charging, or held at a charge limit.
_ON_CHARGER_STATUSES = frozenset({"Charging", "Full", "Not charging"})


def read_battery_sysfs(root: Path = _POWER_SUPPLY_ROOT) -> BatteryState | None:
    """
    Read the charge from Linux sysfs; None when nothing under *root* is the machine's battery.

    Every power supply the kernel knows sits under one directory, each declaring its ``type``. The
    first ``Battery`` that is not a peripheral's (a mouse reports ``scope`` ``Device``) gives the
    charge. A charger is an online ``Mains`` or USB supply (a USB-C charger reports ``USB``). Only
    on a laptop that lists no charger at all does a battery that says it is charging, or held full,
    count as one.
    """
    percent: int | None = None
    charger_listed = charger_online = charging = False
    try:
        supplies = sorted(root.iterdir())
    except OSError:
        return None
    for supply in supplies:
        kind = _first_line(supply / "type")
        if kind == "Battery" and _first_line(supply / "scope") != "Device":
            if percent is None:
                with contextlib.suppress(ValueError):
                    percent = int(_first_line(supply / "capacity"))
            charging = charging or _first_line(supply / "status") in _ON_CHARGER_STATUSES
        elif kind == "Mains" or kind.startswith("USB"):
            charger_listed = True
            charger_online = charger_online or _first_line(supply / "online") == "1"
    if percent is None:
        return None
    # A listed charger is the authority: some firmware says "Not charging" while on battery.
    return BatteryState(percent=percent, on_ac=charger_online if charger_listed else charging)


def _read_battery_windows() -> BatteryState | None:
    """Read the charge through GetSystemPowerStatus; None off Windows or on any failure."""
    try:
        import ctypes  # noqa: PLC0415 - a Windows-only import kept out of the module path.

        class _PowerStatus(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32),
                ("BatteryFullLifeTime", ctypes.c_uint32),
            ]

        # Typed as Any: ctypes generates the field attributes from _fields_ at runtime, which
        # static analyzers cannot see.
        status: Any = _PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        percent = int(status.BatteryLifePercent)
        if percent > _FULL_CHARGE:
            # 255 means unknown, which is what a desktop with no battery reports.
            return None
        return BatteryState(percent=percent, on_ac=int(status.ACLineStatus) == 1)
    except Exception as exc:  # noqa: BLE001 - a hardware probe must never raise.
        logger.debug("battery_probe_failed", error=str(exc))
    return None


def read_battery() -> BatteryState | None:
    """Read the machine's battery, or None when there is none or it cannot be read."""
    match platform.system():
        case "Darwin":
            return parse_pmset(_run_probe(["pmset", "-g", "batt"]))
        case "Linux":
            return read_battery_sysfs()
        case "Windows":
            return _read_battery_windows()
        case _:
            return None


class Pacer:
    """
    Decides when a run rests, and does the resting.

    One instance is shared by every worker thread. The first thread to reach the end of a work
    window rests while the others queue behind it on the lock; they then find the window reset and
    go straight back to work, so the whole pool rests together and nothing is left half done.

    Everything the class needs from the outside world (the clock, the sleep, the battery) is a
    constructor argument, so tests drive a run of any length in no time at all.
    """

    def __init__(
        self,
        plan: PacingPlan,
        *,
        battery_reader: Callable[[], BatteryState | None] = read_battery,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_rest: Callable[[str, float], None] | None = None,
    ) -> None:
        """Build a pacer for *plan*; *on_rest* is told the reason and the expected seconds."""
        self._plan = plan
        self._read_battery = battery_reader
        self._monotonic = monotonic
        self._sleep = sleep
        self._on_rest = on_rest
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._window_started: float | None = None
        self._battery: BatteryState | None = None
        self._battery_read_at: float | None = None
        self._rested_seconds = 0.0
        self._rests_taken = 0
        # Between two batches of a watch the clock is stopped: waiting for photos is not work.
        self._worked_before_pause = 0.0
        self._paused_at: float | None = None
        self._waiting_for_battery = False

    @property
    def rested_seconds(self) -> float:
        """Seconds this run has spent resting, for the run summary."""
        return self._rested_seconds

    @property
    def rests_taken(self) -> int:
        """How many times this run has rested."""
        return self._rests_taken

    def pause(self) -> None:
        """
        Stop the work clock at the end of a batch, keeping the work done so far.

        A watch shares one pacer across its batches, so a window carries over from one handful of
        photos to the next. The time spent waiting for the next import must not count as work.
        """
        with self._lock:
            if self._window_started is None:
                return
            now = self._monotonic()
            self._worked_before_pause = now - self._window_started
            self._window_started = None
            self._paused_at = now

    def cancel(self) -> None:
        """End the rest in progress, and let every later call return immediately."""
        self._cancelled.set()

    def wait_before_next(self) -> bool:
        """
        Block until the caller may start its next photo; False means the run is stopping.

        Call this at a photo boundary, from any thread. The work window is measured from the
        previous rest, so the first call only starts the clock.
        """
        if not self._plan.rests:
            return True
        if self._cancelled.is_set():
            return False
        with self._lock:
            if self._cancelled.is_set():
                return False
            now = self._monotonic()
            if self._window_started is None:
                resumed = self._paused_at is not None
                self._window_started = now - self._carried_over(now)
                if resumed and self._battery_is_low():
                    # A watch's batches can each be shorter than a window, so the battery is also
                    # checked when one starts again; otherwise a low charge would go unnoticed.
                    self._rest(0.0)
                    self._window_started = self._monotonic()
                return not self._cancelled.is_set()
            if now - self._window_started >= self._plan.work_seconds:
                self._rest(now - self._window_started)
                self._window_started = self._monotonic()
            return not self._cancelled.is_set()

    def _carried_over(self, now: float) -> float:
        """
        Return the work a paused window carries into the next batch.

        An idle gap at least as long as the rest that work earned already was that rest, so the next
        batch starts a fresh window instead of resting again.
        """
        worked, paused_at = self._worked_before_pause, self._paused_at
        self._worked_before_pause, self._paused_at = 0.0, None
        if paused_at is None or now - paused_at >= self._plan.rest_for(worked):
            return 0.0
        return worked

    def _battery_is_low(self) -> bool:
        """
        Say whether the charge is at or below the floor with a charger plugged in.

        Resting on battery power alone is pointless: nothing is charging, so the rest would only
        spend the same energy over a longer day. The run keeps going and the machine decides.
        """
        if self._plan.battery_floor <= 0:
            return False
        state = self._battery_state()
        return state is not None and state.on_ac and state.percent <= self._plan.battery_floor

    def _battery_state(self) -> BatteryState | None:
        """Return a recent battery reading, probing at most every _BATTERY_POLL_SECONDS."""
        now = self._monotonic()
        read_at = self._battery_read_at
        if read_at is not None and now - read_at < _BATTERY_POLL_SECONDS:
            return self._battery
        self._battery_read_at = now
        self._battery = self._read_battery()
        return self._battery

    def _battery_has_recovered(self) -> bool:
        """
        Say whether a battery rest may end: the charge is back at the resume level.

        Waiting for the resume level rather than for the floor is what makes the pair a band. A rest
        that ended one percent above the floor would come back after the very next window. A charger
        pulled mid-rest ends the wait too, since nothing is charging any more.
        """
        state = self._battery_state()
        return state is None or not state.on_ac or state.percent >= self._plan.resume_percent

    def _rest_is_over(self, now: float, *, until: float, deadline: float) -> bool:
        """
        Report whether the current rest has run its course.

        Once the charge is found at or below the floor, the rest waits for the resume level, even
        when it began as a duty-cycle rest.
        """
        if now >= deadline:
            return True
        if now < until:
            return False
        self._waiting_for_battery = self._waiting_for_battery or self._battery_is_low()
        return not self._waiting_for_battery or self._battery_has_recovered()

    def _rest(self, worked: float) -> None:
        """Sleep off what *worked* seconds of work earned, plus any wait for the battery."""
        earned = self._plan.rest_for(worked)
        low = self._battery_is_low()
        if earned <= 0 and not low:
            return
        reason = "battery" if low else "duty"
        expected = self._plan.max_pause_seconds if low else earned
        started = self._monotonic()
        until = started + earned
        deadline = started + self._plan.max_pause_seconds
        logger.info(
            "pacing_rest_started",
            reason=reason,
            seconds=round(expected, 1),
            worked=round(worked, 1),
            battery=self._battery.percent if self._battery is not None else None,
        )
        self._announce(reason, expected)
        self._waiting_for_battery = low
        try:
            while not self._cancelled.is_set():
                if self._rest_is_over(self._monotonic(), until=until, deadline=deadline):
                    break
                self._sleep(_NAP_SECONDS)
        finally:
            # Counted even when a Ctrl-C cuts the rest short, so the summary still shows it.
            rested = self._monotonic() - started
            self._rested_seconds += rested
            self._rests_taken += 1
            logger.info("pacing_rest_ended", reason=reason, seconds=round(rested, 1))

    def _announce(self, reason: str, seconds: float) -> None:
        """Tell the caller's UI about a rest; a broken callback must not stop the run."""
        if self._on_rest is None:
            return
        try:
            self._on_rest(reason, seconds)
        except Exception as exc:  # noqa: BLE001 - any callback error must not break the batch.
            logger.exception("pacing_callback_failed", error=str(exc))


def build_pacer(
    plan: PacingPlan,
    *,
    on_rest: Callable[[str, float], None] | None = None,
) -> Pacer | None:
    """Return a pacer for *plan*, or None when the plan would never rest anything."""
    return Pacer(plan, on_rest=on_rest) if plan.rests else None
