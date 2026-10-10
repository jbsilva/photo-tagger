"""
Tests for the desktop window's Qt-free usage tracking.

The headline case is the one the telemetry docs walk through: a window left open overnight, one long
unattended run, a review in the morning, a save, and a second folder. It must read as one session,
three runs, a few busy hours, and many idle ones.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from photo_tagger import telemetry
from photo_tagger.fields import ExtraFields
from photo_tagger.gui_state import FAILED, READY, GuiConfigValues, PhotoItem, Proposal
from photo_tagger.gui_usage import (
    ENGAGED_GRACE_SECONDS,
    GUI_BASELINE,
    JOB_GENERATE,
    JOB_SAVE,
    TRIGGER_BATCH,
    ActivityClock,
    GuiUsage,
    ReviewTally,
    RunContext,
    RunEnd,
    RunTally,
    WindowFacts,
    batch_outcome,
    count_edits,
    run_info,
    window_settings,
)
from photo_tagger.models import KeywordSet


HOUR = 3600.0

_CONTEXT = RunContext(
    provider="lmstudio",
    model="qwen/qwen3-vl-30b",
    output_language="English",
    ui_language="en",
    settings=GUI_BASELINE,
    setup=telemetry.Setup(),
)


def _proposal(path: Path, **overrides: object) -> Proposal:
    fields: dict[str, object] = {
        "path": path,
        "existing_title": None,
        "existing_description": None,
        "existing_keywords": KeywordSet(),
        "title": "Heron at Dawn",
        "description": "A grey heron.",
        "keywords": ["Heron", "Bird"],
        "seconds": 3.0,
        "total_tokens": 100,
    }
    fields.update(overrides)
    return Proposal(**fields)  # type: ignore[arg-type]


def _generated(path: Path, *, at: float = 0.0, **overrides: object) -> PhotoItem:
    """Return a photo that received a proposal at *at*, as apply_proposal leaves it."""
    item = PhotoItem(path=path, has_proposal=True, status=READY)
    item.title = item.proposed_title = "Heron at Dawn"
    item.description = item.proposed_description = "A grey heron."
    item.keywords = ["Heron", "Bird"]
    item.proposed_keywords = ["Heron", "Bird"]
    item.proposed_at = at
    for name, value in overrides.items():
        setattr(item, name, value)
    return item


def _tally(job: str = JOB_GENERATE, *, started: float = 0.0, batch: int = 3) -> RunTally:
    return RunTally(
        job=job,
        trigger=TRIGGER_BATCH,
        started=started,
        batch_size=batch,
        context=_CONTEXT,
    )


# --- activity clock ------------------------------------------------------------------------------


def test_the_overnight_window_reads_as_mostly_idle() -> None:
    """Open, start a 2 hour run, walk away for 8 hours, review, save, and start another folder."""
    clock = ActivityClock(started=0.0)
    for moment in (0.0, 20.0, 40.0):  # adding the folder and clicking Generate
        clock.touch(moment)
    clock.set_busy(45.0, busy=True)
    clock.set_busy(45.0 + 2 * HOUR, busy=False)  # the run ends while everyone sleeps
    morning = 8 * HOUR
    for minute in range(5):  # five minutes of review, a click every half minute
        clock.touch(morning + minute * 60.0)
        clock.touch(morning + minute * 60.0 + 30.0)
    clock.set_busy(morning + 300.0, busy=True)  # Save Selected
    clock.set_busy(morning + 360.0, busy=False)
    clock.touch(morning + 370.0)
    clock.set_busy(morning + 380.0, busy=True)  # the second folder, still running at the end
    split = clock.split(morning + 480.0)

    assert split.session == morning + 480.0
    # The first run, the save, and the 100 seconds the second run has had so far.
    assert split.busy == pytest.approx(2 * HOUR + 60.0 + 100.0)
    # 100 s around the first clicks, 330 s of review (the last click plus its grace minute), and
    # the grace minute after the click between the save and the second run.
    assert split.engaged == pytest.approx(100.0 + 330.0 + ENGAGED_GRACE_SECONDS)
    # Busy or engaged: from opening to the end of the long run, then the morning's 470 s.
    assert split.idle == pytest.approx(split.session - (2 * HOUR + 45.0) - 360.0 - 110.0)
    assert split.idle > 5.9 * HOUR
    # Watched: 55 s of the long run, 30 s of the save, and 50 s of the second run.
    assert split.unattended == pytest.approx(2 * HOUR + 160.0 - 135.0)


def test_engaged_time_during_a_run_is_how_much_of_it_someone_watched() -> None:
    """A click near the start of a long run covers one grace minute of it, no more."""
    clock = ActivityClock(started=0.0)
    clock.touch(10.0)
    assert clock.engaged_between(0.0, 1000.0) == pytest.approx(ENGAGED_GRACE_SECONDS)
    assert clock.engaged_between(500.0, 1000.0) == 0.0


def test_touches_close_together_merge_into_one_span() -> None:
    """Input every few seconds is one engaged stretch, not a pile of overlapping minutes."""
    clock = ActivityClock(started=0.0)
    for moment in range(0, 100, 5):
        clock.touch(float(moment))
    assert clock.split(1000.0).engaged == pytest.approx(95.0 + ENGAGED_GRACE_SECONDS)


# --- edits and review ---------------------------------------------------------------------------


def test_count_edits_compares_the_save_with_the_proposal(tmp_path: Path) -> None:
    """A changed title and one swapped keyword are counted; a case change in a keyword is not."""
    item = _generated(tmp_path / "a.jpg")
    item.title = "Grey Heron at Dawn"
    item.keywords = ["heron", "Wading Bird"]
    edits = count_edits(item)
    assert (edits.title, edits.description) == (True, False)
    assert (edits.keywords_added, edits.keywords_removed) == (1, 1)
    assert edits.any is True
    assert count_edits(_generated(tmp_path / "b.jpg")).any is False


def test_review_tally_counts_edits_hints_flags_and_waiting(tmp_path: Path) -> None:
    """Each saved photo adds to the tally; one never generated counts as a plain edit."""
    tally = ReviewTally()
    edited = _generated(tmp_path / "a.jpg", at=100.0, hinted=True)
    edited.description = "A grey heron fishing."
    flagged = _generated(tmp_path / "b.jpg", at=100.0, foreign_terms=["陶笛"], from_cache=True)
    manual = PhotoItem(path=tmp_path / "c.jpg")
    tally.add(edited, now=400.0)
    tally.add(flagged, now=700.0)
    tally.add(manual, now=700.0)

    stats = tally.stats()
    assert (stats.saved, stats.edited, stats.edited_description) == (3, 1, 1)
    assert (stats.hinted, stats.flagged, stats.from_cache, stats.manual_only) == (1, 1, 1, 1)
    assert stats.review_seconds_median == pytest.approx(450.0)
    assert stats.review_seconds_max == pytest.approx(600.0)


# --- runs ------------------------------------------------------------------------------------


def test_a_generation_run_counts_passes_repairs_and_failures(tmp_path: Path) -> None:
    """The retry pass saves a photo, the mixed-script pass fixes another, one stays failed."""
    a, b, c = (tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg"))
    tally = _tally()
    tally.generated(str(a), _proposal(a, foreign_terms=["陶笛"], repaired=True))
    tally.failed(str(b), "timeout", "ReadTimeout")
    tally.failed(str(c), "connection")
    tally.generation_ended(100.0, rested=5.0, rests=1)
    tally.pass_started("failed", 2, retry=True)
    tally.generated(str(b), _proposal(b, from_cache=True))
    tally.failed(str(c), "connection", "ConnectError")
    tally.pass_started("mixed-script", 1, retry=False)
    tally.generated(str(a), _proposal(a))
    tally.generation_ended(150.0, rested=0.0, rests=0)

    assert (tally.initial_failures, tally.retry_pass, tally.retry_pass_recovered) == (2, 2, 1)
    assert (tally.mixed_pass, tally.mixed_pass_fixed) == (1, 1)
    assert (tally.language_repairs, tally.cache_hits, tally.inference_calls) == (1, 1, 3)
    assert (tally.rested_seconds, tally.rests_taken) == (5.0, 1)

    info = run_info(
        tally,
        RunEnd(now=160.0, outcome=None, attended=12.0, succeeded=2, file_types="jpg", flagged=0),
    )
    assert (info.job, info.outcome, info.success_count, info.failure_count) == (
        "generate",
        "partial",
        2,
        1,
    )
    assert info.failure_kinds == "connection:1"
    assert info.failure_types == "ConnectError:1", "the recovered photo takes its type back"
    assert info.gui is not None
    assert (info.gui.retry_pass_recovered, info.gui.attended_seconds) == (1, 12.0)
    assert info.review is None, "nothing was saved"


def test_a_save_run_reports_its_review(tmp_path: Path) -> None:
    """A batch save carries what the saved photos went through since they were generated."""
    tally = _tally(JOB_SAVE, started=1000.0, batch=2)
    tally.save_begun(1000.0)
    edited = _generated(tmp_path / "a.jpg", at=100.0)
    edited.keywords = ["Heron", "Bird", "Marsh"]
    tally.photo_saved(edited, ok=True, now=1010.0)
    tally.photo_saved(_generated(tmp_path / "b.jpg"), ok=False, now=1020.0)
    tally.save_ended(1030.0)

    info = run_info(
        tally,
        RunEnd(now=1030.0, outcome=None, attended=0.0, succeeded=1, file_types="jpg", flagged=0),
    )
    assert (info.job, info.outcome, info.failure_count) == ("save", "partial", 1)
    assert info.gui is not None
    assert (info.gui.saved, info.gui.save_failures, info.gui.save_seconds) == (1, 1, 30.0)
    assert info.review is not None
    assert (info.review.edited_keywords, info.review.keywords_added) == (1, 1)


def test_a_cancelled_run_says_so(tmp_path: Path) -> None:
    """An explicit outcome from the window wins over the one the counts would give."""
    info = run_info(
        _tally(),
        RunEnd(
            now=5.0,
            outcome=telemetry.OUTCOME_CANCELLED,
            attended=0.0,
            succeeded=0,
            file_types="",
            flagged=0,
        ),
    )
    assert info.outcome == "cancelled"


def test_batch_outcome_counts_proposals_and_failures(tmp_path: Path) -> None:
    """A photo with a proposal succeeded; one marked failed did not; a pending one is neither."""
    items = [
        _generated(tmp_path / "a.jpg"),
        PhotoItem(path=tmp_path / "b.jpg", status=FAILED),
        PhotoItem(path=tmp_path / "c.jpg"),
    ]
    assert batch_outcome(items) == (1, 1)


# --- the session ---------------------------------------------------------------------------------


def test_the_session_adds_up_runs_saves_and_features(tmp_path: Path) -> None:
    """A session counts its runs, its photos, and each feature, and times the first run."""
    usage = GuiUsage(now=0.0)
    usage.count("viewer", 5.0)
    usage.count("viewer", 6.0)
    usage.photos_added = 3
    usage.start_run(_tally(started=30.0))
    a = tmp_path / "a.jpg"
    usage.generated(str(a), _proposal(a))
    usage.finish_run(
        RunEnd(now=90.0, outcome=None, attended=0.0, succeeded=1, file_types="jpg", flagged=0),
    )
    usage.saved_single(_generated(a, at=90.0), ok=True, now=120.0)

    session = usage.session_info(
        600.0,
        context=_CONTEXT,
        unsaved=0,
        closed_with_unsaved=False,
        view_mode="tree",
        display=telemetry.Display(color_scheme="dark"),
    )
    assert (session.runs, session.save_runs, session.single_saves) == (1, 0, 1)
    assert (session.photos_added, session.photos_generated, session.photos_saved) == (3, 1, 1)
    assert session.first_run_seconds == 30.0  # noqa: PLR2004 - from opening to the first click
    assert session.busy_seconds == pytest.approx(60.0)
    assert session.features == {"viewer": 2}
    assert session.review is not None
    assert session.review.saved == 1
    assert (
        usage.finish_run(
            RunEnd(now=700.0, outcome=None, attended=0.0, succeeded=0, file_types="", flagged=0),
        )
        is None
    ), "with no run open there is nothing to finish"


def test_a_session_without_runs_reports_no_first_run() -> None:
    """A window opened and closed without generating anything says so with -1."""
    session = GuiUsage(now=0.0).session_info(
        10.0,
        context=_CONTEXT,
        unsaved=0,
        closed_with_unsaved=False,
        view_mode="list",
        display=telemetry.Display(),
    )
    assert session.first_run_seconds == -1.0
    assert session.review is None


# --- settings --------------------------------------------------------------------------------


def _config_values(**overrides: object) -> GuiConfigValues:
    fields: dict[str, object] = {
        "provider_name": "lmstudio",
        "model_name": "qwen/qwen3-vl-30b",
        "api_base_url": None,
        "extensions": "jpg,jpeg,png,dng,cr3,nef,arw,heic,heif,tif,tiff,webp",
        "recursive": True,
        "write_title": True,
        "write_description": True,
        "write_keywords": True,
        "preserve_keywords": True,
        "sidecar_mode": "all",
        "backup_xmp": True,
        "telemetry_enabled": True,
    }
    fields.update(overrides)
    return GuiConfigValues(**fields)  # type: ignore[arg-type]


_FACTS = WindowFacts(
    url="http://localhost:1234/v1",
    custom_url=False,
    api_key=False,
    cache=True,
    jpeg_dimensions=GUI_BASELINE.jpeg_dimensions,
    vocabulary_terms=0,
    copyright=None,
    watch_interval=GUI_BASELINE.watch_interval,
    watch_settle=GUI_BASELINE.watch_settle,
)


def test_window_settings_of_an_untouched_window_are_the_gui_defaults() -> None:
    """Nothing reads as changed in a window nobody has set up."""
    assert window_settings(_config_values(), _FACTS) == GUI_BASELINE


def test_window_settings_keep_no_path_or_text() -> None:
    """A vocabulary path, a remote URL, and a copyright notice come out as switches and names."""
    settings = window_settings(
        _config_values(
            vocabulary=Path("/Users/someone/keywords.txt"),
            duty=50,
            save_after_generating=True,
        ),
        replace(
            _FACTS,
            url="https://gpu.example.net/v1",
            custom_url=True,
            copyright=ExtraFields(artist="Ada", notice="(c) {year} Ada", marked=False),
        ),
    )
    assert (settings.endpoint, settings.secure, settings.custom_url) == ("public", True, True)
    assert (settings.vocabulary, settings.paced, settings.autosave) == (True, True, True)
    assert (settings.artist, settings.copyright_notice, settings.marked) == (
        True,
        True,
        "public-domain",
    )
    assert "Ada" not in repr(settings)
    assert "someone" not in repr(settings)


def test_a_generate_run_counts_a_failed_chained_save_as_failed() -> None:
    """A photo generated fine but not written ends failed, and the run's counts must say so."""
    tally = _tally(JOB_GENERATE)
    end = RunEnd(now=10.0, outcome=None, attended=0.0, succeeded=0, file_types="", flagged=0)
    assert run_info(tally, replace(end, failed=1)).failure_count == 1


def test_window_settings_report_the_retries_a_window_run_uses() -> None:
    """The window retries a photo fewer times than the CLI, and its beacon says how many."""
    from photo_tagger.gui_usage import GUI_BASELINE, GUI_GENERATE_RETRIES  # noqa: PLC0415

    assert GUI_BASELINE.retries == GUI_GENERATE_RETRIES
