"""The status-bar lines about pacing rests."""

from photo_tagger.gui_state.text import (
    MIXED_SCRIPT_MARK,
    format_duration,
)
from photo_tagger.i18n import _, ngettext
from photo_tagger.pacing import (
    CAUTION_BATTERY_FLOOR,
    FULL_DUTY,
    SECONDS_PER_MINUTE,
    PacingPlan,
)


def pacing_status_text(
    reason: str,
    seconds: float,
    *,
    duty_percent: int,
    resume_percent: int,
) -> str:
    """
    Status line while a paced run rests: why it stopped, and for how long.

    A battery rest ends when the charge comes back, which may be sooner than *seconds*, so that
    number is worded as the upper bound it is.
    """
    duration = format_duration(seconds)
    if reason == "battery":
        return _("Resting up to {duration} so the battery charges back to {percent}%.").format(
            duration=duration,
            percent=resume_percent,
        )
    return _("Resting {duration}: this run works {percent}% of the time.").format(
        duration=duration,
        percent=duty_percent,
    )


def _spoken_duration(seconds: float) -> str:
    """
    Render *seconds* as whole minutes, or as seconds while it is under a minute.

    Spelled out rather than the ``m:ss`` of :func:`format_duration`, because this one lands mid
    sentence, where "about 4 minutes of rest" reads and "about 4:00 of rest" does not. Rounding is
    honest in the direction that matters: a rest of a few seconds should say so, since that is the
    sign of a duty cycle set too close to 100 to do anything.
    """
    if seconds < SECONDS_PER_MINUTE:
        count = max(1, round(seconds))
        return ngettext("{n} second", "{n} seconds", count).format(n=count)
    minutes = max(1, round(seconds / SECONDS_PER_MINUTE))
    return ngettext("{n} minute", "{n} minutes", minutes).format(n=minutes)


def pacing_summary(plan: PacingPlan) -> str:
    """
    Describe what a run will actually do under *plan*, for the Pacing dialog to show live.

    The dialog needs this because its own fields do not say it: a duty cycle is a percentage, and
    what a person wants to know is how long the run stops for and how often.
    """
    if not plan.rests:
        return _("Left as it is, a run works flat out.")
    lines: list[str] = []
    if plan.duty_percent < FULL_DUTY:
        lines.append(
            _(
                "Works {percent}% of the time: about {rest} of rest after every {work} of work.",
            ).format(
                percent=plan.duty_percent,
                rest=_spoken_duration(plan.rest_for(plan.work_seconds)),
                work=_spoken_duration(plan.work_seconds),
            ),
        )
    if plan.battery_floor > 0:
        lines.append(
            _(
                "Whenever the battery falls to {floor}%, rests until it is back to {resume}%, "
                "for at most {cap}.",
            ).format(
                floor=plan.battery_floor,
                resume=plan.resume_percent,
                cap=_spoken_duration(plan.max_pause_seconds),
            ),
        )
    if plan.battery_floor > CAUTION_BATTERY_FLOOR:
        lines.append(
            _(
                "{mark} A plugged-in laptop is rarely that full, so the run would rest after "
                "nearly every window and wait out the cap each time.",
            ).format(mark=MIXED_SCRIPT_MARK),
        )
    return "\n".join(lines)
