"""
End-to-end wiring tests for the cyclopts CLI: main and the command modules it registers.

The agent (network calls) and pipeline (long-running work) are mocked; the goal is to prove that
flag values reach the right collaborators with the right shape, and that the short-circuit / skip
code paths run end to end without raising. Real work is exercised elsewhere by the per-module unit
tests.
"""

import contextlib
import csv
import importlib
import json
import os
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from photo_tagger import (
    cli_fields,
    cli_options,
    cli_reports,
    cli_run,
    cli_telemetry,
    cli_vocabulary,
    cli_watch,
    config_file,
    diagnostics,
    main as main_module,
    telemetry,
)
from photo_tagger.cli_options import (
    ConfigFileSource,
    Defaults,
    RunOptions,
    WatchConfig,
    known_config_keys,
    load_defaults,
    resolve_notify,
    to_pacing_plan,
    to_telemetry_settings,
)
from photo_tagger.errors import BatchError, DiscoveryError, ProviderError
from photo_tagger.metadata import SOURCE_SIDECAR, FieldReading, FieldSources
from photo_tagger.pacing import SECONDS_PER_MINUTE
from photo_tagger.pipeline import BatchTotals, ImageOutcome
from photo_tagger.undo import UndoError, UndoJournal, read_journal
from photo_tagger.vocabulary_build import KeywordCensus, TrimResult
from photo_tagger.vocabulary_organize import OrganizeStats
from photo_tagger.write_plan import WriteTarget


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _no_real_exiftool_in_show_and_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ``show`` and ``set`` a fake ExifTool: their tests fake each read and write anyway."""
    monkeypatch.setattr(
        cli_fields,
        "managed_helper",
        lambda _et: contextlib.nullcontext(MagicMock()),
    )


def _make_jpeg(path: Path) -> Path:
    """Write a small placeholder file so cyclopts' --input validator accepts the path."""
    path.write_bytes(b"\xff\xd8stub")
    return path


def _run_app(args: list[str]) -> None:
    """Invoke the cyclopts app while absorbing the SystemExit it always raises."""
    with contextlib.suppress(SystemExit):
        main_module.app(args)


def _patches(captured: dict[str, Any]) -> Any:  # noqa: ANN401 - context manager juggling.
    """Patch every IO collaborator main.tag invokes; capture the values for assertions."""

    def fake_run_batch(
        image_files: list[Path],
        agent: object,
        options: object,
        **kwargs: object,
    ) -> object:
        captured["image_files"] = list(image_files)
        captured["options"] = options
        captured["on_success"] = kwargs.get("on_success")
        captured["on_complete"] = kwargs.get("on_complete")
        captured["user_prompt"] = kwargs.get("user_prompt")
        captured["workers"] = kwargs.get("workers")
        captured["on_image_result"] = kwargs.get("on_image_result")
        captured["cache"] = kwargs.get("cache")
        captured["progress"] = kwargs.get("progress")
        captured["session_plan"] = kwargs.get("session_plan")
        captured["journal"] = kwargs.get("journal")
        captured["pacer"] = kwargs.get("pacer")
        return None

    return (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=fake_run_batch),
    )


def test_cli_passes_inputs_and_dry_run_through_to_pipeline(tmp_path: Path) -> None:
    """A minimal invocation feeds the resolved file list and dry_run flag to run_batch."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--dry-run"])

    assert captured["image_files"] == [image.resolve()]
    assert captured["options"].dry_run is True
    # Default skip_tagged path means no on_success appender created.
    assert captured["on_success"] is None


def test_cli_creates_appender_when_append_to_skip_file_provided(tmp_path: Path) -> None:
    """Passing --append-to-skip-file installs an on_success callback on run_batch."""
    image = _make_jpeg(tmp_path / "img.cr3")
    skip_file = tmp_path / "processed.txt"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--append-to-skip-file", str(skip_file)])

    assert callable(captured["on_success"])

    # The callback writes a single line per call. Exercise it once to be sure.
    captured["on_success"](image)
    assert skip_file.read_text(encoding="utf-8").splitlines() == [str(image)]


def test_cli_skip_tagged_filters_before_pipeline(tmp_path: Path) -> None:
    """--skip-tagged removes already-tagged paths before run_batch is even called."""
    keep = _make_jpeg(tmp_path / "keep.cr3")
    drop = _make_jpeg(tmp_path / "drop.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.discovery.find_tagged_images", return_value={drop.resolve()}),
    ):
        _run_app(["--input", str(keep), "--input", str(drop), "--skip-tagged"])

    assert captured["image_files"] == [keep.resolve()]


def test_cli_short_circuits_when_skip_filters_remove_everything(tmp_path: Path) -> None:
    """When all inputs are skipped, run_batch is not called and the CLI exits cleanly."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.discovery.find_tagged_images", return_value={image.resolve()}),
    ):
        _run_app(["--input", str(image), "--skip-tagged"])

    assert "image_files" not in captured  # run_batch never invoked


def test_cli_exits_when_no_inputs_passed() -> None:
    """No --input is a hard error so accidental empty runs surface immediately."""
    setup, create_agent, run_batch = _patches({})
    with setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app([])


_EXPECTED_WORKERS = 3


def test_cli_workers_and_prompt_file_reach_run_batch(tmp_path: Path) -> None:
    """--workers and --prompt-file both flow into the run_batch call."""
    image = _make_jpeg(tmp_path / "img.cr3")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Describe like a wildlife photographer.\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            [
                "--input",
                str(image),
                "--workers",
                str(_EXPECTED_WORKERS),
                "--prompt-file",
                str(prompt),
            ],
        )

    assert captured["workers"] == _EXPECTED_WORKERS
    assert captured["user_prompt"] == "Describe like a wildlife photographer."


def test_cli_hint_lands_in_the_user_prompt(tmp_path: Path) -> None:
    """--hint rides inside the user prompt, so the cache namespace changes with it too."""
    from photo_tagger.config import DEFAULT_USER_PROMPT  # noqa: PLC0415

    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--hint", "The animal is a deer"])

    prompt = captured["user_prompt"]
    assert "Photographer's note about this photo: The animal is a deer" in prompt
    assert prompt.startswith(DEFAULT_USER_PROMPT)


def test_cli_vocabulary_reaches_the_pipeline_and_the_prompt(tmp_path: Path) -> None:
    """--vocabulary loads the file, hands it to the pipeline, and lists it in the prompt."""
    image = _make_jpeg(tmp_path / "img.cr3")
    vocabulary = tmp_path / "keywords.txt"
    vocabulary.write_text("Animal\n\tBird\n\t\tOsprey\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            ["--input", str(image), "--vocabulary", str(vocabulary), "--vocabulary-strict"],
        )

    options = captured["options"]
    assert options.vocabulary.match("ospreys") == "Osprey"
    assert options.vocabulary_strict is True
    assert "- Animal > Bird > Osprey" in captured["user_prompt"]


def test_cli_session_gap_builds_a_plan_for_the_batch(tmp_path: Path) -> None:
    """--session-gap groups the resolved batch and hands the plan to run_batch."""
    first = _make_jpeg(tmp_path / "a.cr3")
    second = _make_jpeg(tmp_path / "b.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(first), "--input", str(second), "--session-gap", "30"])

    plan = captured["session_plan"]
    assert plan is not None
    assert plan.sessions == [[first.resolve(), second.resolve()]]


def test_cli_without_session_gap_passes_no_plan(tmp_path: Path) -> None:
    """The default keeps the per-photo behavior, with no grouping pass at all."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["session_plan"] is None


def test_cli_exits_when_the_vocabulary_file_has_no_terms(tmp_path: Path) -> None:
    """An unusable vocabulary stops the run before any model call, with exit 1."""
    image = _make_jpeg(tmp_path / "img.cr3")
    vocabulary = tmp_path / "keywords.txt"
    vocabulary.write_text("# nothing here\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch, pytest.raises(SystemExit) as exit_info:
        main_module.app(["--input", str(image), "--vocabulary", str(vocabulary)])

    assert exit_info.value.code == 1
    assert "image_files" not in captured  # run_batch never invoked


def _outcome(file: Path, *, success: bool = True, from_cache: bool = False) -> ImageOutcome:
    """Build a representative ImageOutcome for NDJSON-emitter tests."""
    return ImageOutcome(
        file=file,
        success=success,
        from_cache=from_cache,
        retry=False,
        title="A Title",
        description="A description.",
        keywords=["Beach", "Sunset"],
        input_tokens=42,
        output_tokens=7,
        total_tokens=49,
        seconds=1.5,
    )


def _rich_outcome(file: Path) -> ImageOutcome:
    """Build an ImageOutcome with the EXIF/merged-keyword fields the CSV report reads."""
    return ImageOutcome(
        file=file,
        success=True,
        from_cache=False,
        retry=False,
        title="A Title",
        description="A description.",
        keywords=["Beach", "Sunset"],
        input_tokens=42,
        output_tokens=7,
        total_tokens=49,
        seconds=1.5,
        written_keywords=["Beach", "Sunset"],
        hierarchical_keywords=["Nature|Beach"],
        existing_keywords=["Old"],
        camera_info={"EXIF:Model": "Canon EOS R5", "EXIF:LensModel": "RF 100mm"},
        location_tags={"XMP-photoshop:City": "Hamburg", "XMP-photoshop:Country": "Germany"},
        gps_position="53 N, 9 E",
    )


def test_cli_json_flag_installs_ndjson_emitter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--json makes run_batch's on_image_result callback print one NDJSON line per photo."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--json"])
    capsys.readouterr()

    captured["on_image_result"](_rich_outcome(image))
    assert json.loads(capsys.readouterr().out)["file"] == str(image)


def test_cli_default_does_not_install_ndjson_emitter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --json the callback only keeps telemetry figures: nothing reaches stdout."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])
    capsys.readouterr()

    captured["on_image_result"](_rich_outcome(image))
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# --csv-file report
# ---------------------------------------------------------------------------


def test_cli_csv_file_writes_streamed_report(tmp_path: Path) -> None:
    """--csv-file produces a header plus one row per photo the pipeline reports."""
    import csv as csv_module  # noqa: PLC0415 - test-local parser.

    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"

    def fake_run_batch(
        image_files: list[Path],
        agent: object,
        options: object,
        **kwargs: object,
    ) -> object:
        callback = kwargs.get("on_image_result")
        assert callback is not None
        callback(_rich_outcome(image))  # type: ignore[operator]
        return None

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=fake_run_batch),
    ):
        _run_app(["--input", str(image), "--csv-file", str(csv_path)])

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv_module.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["filename"] == "img.cr3"
    assert rows[0]["title"] == "A Title"
    assert rows[0]["keywords"] == "Beach; Sunset"
    assert rows[0]["camera_model"] == "Canon EOS R5"
    assert rows[0]["city"] == "Hamburg"
    assert rows[0]["status"] == "ok"


def test_cli_csv_file_installs_csv_sink_and_writes_header(tmp_path: Path) -> None:
    """--csv-file writes the header up front and a row for each photo run_batch reports."""
    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"

    def fake_run_batch(*_args: object, **kwargs: Any) -> None:  # noqa: ANN401 - run_batch's kwargs.
        kwargs["on_image_result"](_rich_outcome(image))

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=fake_run_batch),
    ):
        _run_app(["--input", str(image), "--csv-file", str(csv_path)])

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    assert [row["filename"] for row in rows] == ["img.cr3"]


def test_cli_csv_and_json_install_fan_out(tmp_path: Path) -> None:
    """--json with --csv-file fans each outcome out to both sinks."""
    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--json", "--csv-file", str(csv_path)])

    callback = captured["on_image_result"]
    assert callable(callback)
    # Neither single sink: the combiner wrapped both behind one callable.
    assert not isinstance(callback, cli_reports.NDJSONEmitter)
    assert not isinstance(callback, cli_reports.CsvImageResultSink)


def test_cli_newer_than_filters_input_batch(tmp_path: Path) -> None:
    """--newer-than parses the timestamp and drops files older than the bound."""
    import os  # noqa: PLC0415 - test-local import.
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}
    boundary = datetime(2024, 1, 1, tzinfo=UTC)
    old_ts = (boundary - timedelta(days=10)).timestamp()
    os.utime(image, (old_ts, old_ts))

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--newer-than", "2024-01-01"])

    # The lone file is older than the bound so run_batch should never be invoked.
    assert "image_files" not in captured


def test_cli_older_than_filters_input_batch(tmp_path: Path) -> None:
    """
    --older-than parses the timestamp and drops files newer than the bound.

    The mirror of the --newer-than test: without it, swapping the two keyword arguments handed to
    apply_date_filter (or dropping older_than entirely) passed the whole suite.
    """
    import os  # noqa: PLC0415 - test-local import.
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}
    boundary = datetime(2024, 1, 1, tzinfo=UTC)
    new_ts = (boundary + timedelta(days=10)).timestamp()
    os.utime(image, (new_ts, new_ts))

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--older-than", "2024-01-01"])

    # The lone file is newer than the bound so run_batch should never be invoked.
    assert "image_files" not in captured


def test_cli_date_window_applies_both_bounds(tmp_path: Path) -> None:
    """--newer-than and --older-than combine into a window; only the inside file survives."""
    import os  # noqa: PLC0415 - test-local import.
    from datetime import UTC, datetime  # noqa: PLC0415 - test-local import.

    before = _make_jpeg(tmp_path / "before.cr3")
    inside = _make_jpeg(tmp_path / "inside.cr3")
    after = _make_jpeg(tmp_path / "after.cr3")
    for path, when in (
        (before, datetime(2023, 6, 1, tzinfo=UTC)),
        (inside, datetime(2024, 6, 1, tzinfo=UTC)),
        (after, datetime(2025, 6, 1, tzinfo=UTC)),
    ):
        os.utime(path, (when.timestamp(), when.timestamp()))

    captured: dict[str, Any] = {}
    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            [
                "--input",
                str(tmp_path),
                "--newer-than",
                "2024-01-01T00:00:00+00:00",
                "--older-than",
                "2025-01-01T00:00:00+00:00",
            ],
        )

    assert [p.name for p in captured["image_files"]] == [inside.name]


def test_cli_rejects_malformed_newer_than(tmp_path: Path) -> None:
    """--newer-than with a non-ISO string exits before scheduling work."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app(["--input", str(image), "--newer-than", "not-a-date"])


def test_parse_filter_date_treats_naive_as_local_time() -> None:
    """A naive ISO date attaches the system local zone, not UTC."""
    from datetime import datetime  # noqa: PLC0415 - test-local import.

    parsed = cli_run.parse_filter_date("2024-01-01T00:00:00", flag="--newer-than")
    assert parsed is not None
    assert parsed.tzinfo is not None
    # Wall-clock fields are preserved verbatim: the user wrote midnight local.
    naive_expected = datetime(2024, 1, 1, 0, 0, 0)  # noqa: DTZ001 - naive on purpose.
    assert parsed.replace(tzinfo=None) == naive_expected
    # Attached offset matches the system's local offset for *that* wall-clock
    # time (which may differ from "now" across DST boundaries). Comparing
    # against datetime.astimezone() of the same naive moment guards against
    # the parser silently falling back to UTC.
    assert parsed.utcoffset() == naive_expected.astimezone().utcoffset()


def test_parse_filter_date_preserves_explicit_timezone() -> None:
    """An ISO string that already carries a timezone is passed through unchanged."""
    parsed = cli_run.parse_filter_date("2024-01-01T00:00:00+00:00", flag="--newer-than")
    assert parsed is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_parse_filter_date_returns_none_for_none() -> None:
    """Passing None short-circuits without raising."""
    assert cli_run.parse_filter_date(None, flag="--newer-than") is None


def test_cli_skips_cache_when_open_fails(tmp_path: Path) -> None:
    """--cache-file with an unusable target still lets the batch run (no cache)."""
    image = _make_jpeg(tmp_path / "img.cr3")
    cache_path = tmp_path / "cache.sqlite3"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.cache.InferenceCache", side_effect=OSError("denied")),
    ):
        _run_app(["--input", str(image), "--cache-file", str(cache_path)])

    # run_batch was invoked, and the cache kwarg fell through as None.
    assert captured.get("image_files") == [image.resolve()]
    assert captured.get("cache") is None


def test_cli_cache_file_opened_is_passed_and_closed(tmp_path: Path) -> None:
    """A usable --cache-file opens a cache, hands it to run_batch, and closes it in the finally."""
    image = _make_jpeg(tmp_path / "img.cr3")
    cache_path = tmp_path / "cache.sqlite3"
    captured: dict[str, Any] = {}

    # Spy on the opened cache so the close is actually asserted: cache_path.exists() alone
    # holds even if the finally block stops closing (a real handle leak on Windows).
    from photo_tagger.cache import open_cache  # noqa: PLC0415 - test-local import.

    spies: list[MagicMock] = []

    def spying_open_cache(path: Path, *, namespace: str) -> MagicMock:
        spy = MagicMock(wraps=open_cache(path, namespace=namespace))
        spies.append(spy)
        return spy

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch.object(cli_run, "open_cache", spying_open_cache),
    ):
        _run_app(["--input", str(image), "--cache-file", str(cache_path)])

    assert cache_path.exists()
    assert captured["cache"] is spies[0]  # the opened cache reached run_batch
    spies[0].close.assert_called_once()  # and the finally closed it


def test_cli_csv_writer_is_closed_after_the_batch(tmp_path: Path) -> None:
    """The finally block closes the CSV writer, so the report handle never leaks."""
    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"
    captured: dict[str, Any] = {}

    from photo_tagger.csv_report import CsvReportWriter  # noqa: PLC0415 - test-local import.

    spies: list[MagicMock] = []

    def spying_writer(path: Path) -> MagicMock:
        spy = MagicMock(wraps=CsvReportWriter(path))
        spies.append(spy)
        return spy

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch.object(cli_reports, "CsvReportWriter", spying_writer),
    ):
        _run_app(["--input", str(image), "--csv-file", str(csv_path)])

    assert csv_path.exists()
    spies[0].close.assert_called_once()


def test_cli_closes_the_cache_when_building_the_run_fails(tmp_path: Path) -> None:
    """
    A failure after the cache is opened still closes it, so its SQLite handle cannot leak.

    Only the CSV writer used to be closed here. The cache is opened in the same step and the run
    that would have closed it never starts, so its handle stayed open for the life of the process.
    """
    image = _make_jpeg(tmp_path / "img.cr3")
    cache = MagicMock()

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "open_cache", return_value=cache),
        patch.object(cli_run, "open_journal", side_effect=OSError("no state dir")),
        pytest.raises(OSError, match="no state dir"),
    ):
        main_module.app(["--input", str(image), "--cache-file", str(tmp_path / "c.sqlite")])

    cache.close.assert_called_once()


def test_an_empty_batch_leaves_the_previous_report_and_the_provider_alone(tmp_path: Path) -> None:
    """
    Nothing to tag must cost nothing: no truncated report, no model validation.

    Building the run opens the CSV report with mode "w", so doing it before discovery replaced the
    last run's report with a bare header for a run that then processed no photos.
    """
    image = _make_jpeg(tmp_path / "img.cr3")
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text("img.cr3\n", encoding="utf-8")
    csv_path = tmp_path / "report.csv"
    csv_path.write_text("previous run's report\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent as agent_mock, run_batch as batch_mock:
        _run_app(
            [
                "--input",
                str(image),
                "--csv-file",
                str(csv_path),
                "--skip-from",
                str(skip_file),
            ],
        )

    assert csv_path.read_text(encoding="utf-8") == "previous run's report\n"
    agent_mock.assert_not_called()
    batch_mock.assert_not_called()


def test_cli_no_progress_disables_the_bar(tmp_path: Path) -> None:
    """--no-progress reaches run_batch as progress=None (batch_progress yields no callback)."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--no-progress"])

    assert "progress" in captured
    assert captured["progress"] is None


def test_cli_lock_file_blocks_second_run(tmp_path: Path) -> None:
    """A second --lock-file invocation while another holds the lock exits with code 1."""
    from photo_tagger.locking import FileLock  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    lock_path = tmp_path / "run.lock"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    # Hold the lock in this thread, then invoke the CLI which should fail to acquire.
    with FileLock(lock_path), setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app(["--input", str(image), "--lock-file", str(lock_path)])

    # run_batch never ran because the CLI bailed out at lock acquisition.
    assert "image_files" not in captured


def test_watch_takes_the_lock_file_too(tmp_path: Path) -> None:
    """Two watches on one folder must not race: --lock-file holds for watch as it does for tag."""
    from photo_tagger.locking import FileLock  # noqa: PLC0415 - test-local import.

    lock_path = tmp_path / "run.lock"
    with (
        FileLock(lock_path),
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_watch, "_watch_forever", side_effect=AssertionError("no watch")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.app(["watch", str(tmp_path), "--lock-file", str(lock_path)])

    assert exc_info.value.code == 1


def test_cli_log_setup_failure_exits_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    An OSError from setup_logging (e.g. an unwritable --log-folder) exits 1 with a clean message.

    Regression test: this used to propagate as a raw OSError traceback instead of the same clean
    "log and exit 1" pattern every other pre-flight failure in this command already follows. Cannot
    rely on the logger for the message: setup_logging removes the default sink before it can fail,
    so nothing is guaranteed to be listening.
    """
    image = _make_jpeg(tmp_path / "img.cr3")

    with (
        patch.object(cli_run, "setup_logging", side_effect=OSError("disk full")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.app(["--input", str(image)])

    assert exc_info.value.code == 1
    assert "disk full" in capsys.readouterr().err


def test_cli_otel_without_an_endpoint_exits_before_tagging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``--otel`` with nowhere to send traces stops the run instead of tagging untraced.

    Somebody who passed the flag wants the traces. Tagging the batch anyway would burn the model
    calls they were trying to observe and leave them to notice the empty collector afterwards.
    """
    image = _make_jpeg(tmp_path / "img.cr3")
    for name in ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)

    with (
        patch.object(cli_run, "create_agent") as agent_mock,
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.app(["--input", str(image), "--otel", "--log-folder", str(tmp_path / "logs")])

    assert exc_info.value.code == 1
    agent_mock.assert_not_called()


def test_cli_lock_file_open_failure_exits(tmp_path: Path) -> None:
    """An OSError while opening the lock file (e.g. unwritable dir) exits with code 1."""
    image = _make_jpeg(tmp_path / "img.cr3")
    lock_path = tmp_path / "run.lock"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch.object(cli_run, "FileLock", side_effect=OSError("disk full")),
        pytest.raises(SystemExit),
    ):
        main_module.app(["--input", str(image), "--lock-file", str(lock_path)])

    assert "image_files" not in captured


_EXPECTED_TOTAL_TOKENS = 49


def test_cli_summary_file_written_on_completion(tmp_path: Path) -> None:
    """--summary-file receives a JSON payload with run totals after the batch finishes."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    summary = tmp_path / "summary.json"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    fake_totals = BatchTotals(
        total_files=1,
        success=1,
        successful_files=[image.name],
        input_tokens=42,
        output_tokens=7,
        total_tokens=_EXPECTED_TOTAL_TOKENS,
        inference_calls=1,
    )

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--summary-file", str(summary)])
        # Simulate run_batch's on_complete callback firing with realistic totals.
        on_complete = captured["on_complete"]
        assert on_complete is not None
        on_complete(fake_totals)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["total_tokens"] == _EXPECTED_TOTAL_TOKENS
    assert payload["successful_files"] == [image.name]
    assert payload["model"]  # provider/model fields are populated.
    assert "started_at" in payload
    assert "finished_at" in payload


def test_cli_summary_file_creates_missing_parent_dir(tmp_path: Path) -> None:
    """--summary-file pointing into a not-yet-existing folder is created transparently."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    nested = tmp_path / "reports" / "today" / "summary.json"
    assert not nested.parent.exists()
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    fake_totals = BatchTotals(total_files=1, success=1, successful_files=[image.name])

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--summary-file", str(nested)])
        captured["on_complete"](fake_totals)

    assert nested.exists()
    payload = json.loads(nested.read_text(encoding="utf-8"))
    assert payload["total_files"] == 1


# ---------------------------------------------------------------------------
# read_prompt_file edge cases
# ---------------------------------------------------------------------------


def test_read_prompt_file_exits_on_empty_file(tmp_path: Path) -> None:
    """A prompt file that contains only whitespace is rejected with SystemExit."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli_run.read_prompt_file(prompt)


def test_read_prompt_file_exits_on_read_error(tmp_path: Path) -> None:
    """An unreadable prompt file triggers SystemExit."""
    prompt = tmp_path / "missing.txt"
    with pytest.raises(SystemExit):
        cli_run.read_prompt_file(prompt)


def test_read_prompt_file_returns_default_when_none() -> None:
    """Passing None returns the built-in default prompt."""
    from photo_tagger.config import DEFAULT_USER_PROMPT  # noqa: PLC0415

    assert cli_run.read_prompt_file(None) == DEFAULT_USER_PROMPT


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------


def test_doctor_exits_zero_when_all_checks_pass() -> None:
    """A clean checklist returns normally (cyclopts treats no SystemExit as exit 0)."""
    from photo_tagger.diagnostics import CheckResult  # noqa: PLC0415 - test-local import.

    ok = [CheckResult("ExifTool", ok=True, detail="/usr/bin/exiftool")]
    with patch.object(main_module, "run_checks", return_value=ok):
        # No SystemExit means success; _run_app would swallow one if raised.
        main_module.doctor(provider="lmstudio", model="m", url=None, api_key=None)


def test_doctor_exits_one_when_a_check_fails() -> None:
    """A failing check makes the command raise SystemExit(1)."""
    from photo_tagger.diagnostics import CheckResult  # noqa: PLC0415 - test-local import.

    bad = [CheckResult("ExifTool", ok=False, detail="missing")]
    with (
        patch.object(main_module, "run_checks", return_value=bad),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.doctor(provider="lmstudio", model="m", url=None, api_key=None)
    assert exc_info.value.code == 1


def test_doctor_honors_the_config_file_through_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Invoking the real subcommand picks up config-file defaults, as its docstring promises.

    The other doctor tests call the function directly with kwargs, which can never catch a
    ConfigFileSource regression specific to subcommands (say, config keys doctor has no flag for
    suddenly erroring, or the [provider] table not reaching --model).
    """
    from photo_tagger.diagnostics import CheckResult  # noqa: PLC0415 - test-local import.

    _point_config_at(
        tmp_path,
        monkeypatch,
        '[provider]\nmodel_name = "configured-model"\n\n[inference]\nmax_tokens = 500\n',
    )
    received: dict[str, Any] = {}

    def fake_run_checks(provider: str, model: str, **kwargs: Any) -> list[Any]:  # noqa: ANN401
        received["provider"] = provider
        received["model"] = model
        received.update(kwargs)
        return [CheckResult("ExifTool", ok=True, detail="ok")]

    with (
        patch.object(main_module, "run_checks", side_effect=fake_run_checks),
        contextlib.suppress(SystemExit),
    ):
        main_module.app(["doctor"])

    assert received["model"] == "configured-model"
    assert received["provider"] == "lmstudio"  # untouched fields keep their built-in defaults


# ---------------------------------------------------------------------------
# gui command (lazy PySide6 import)
# ---------------------------------------------------------------------------


def test_gui_reports_missing_pyside6(capsys: pytest.CaptureFixture[str]) -> None:
    """Without the optional extra, the command exits 1 with a pip install hint."""

    def raise_import_error(_name: str) -> object:
        msg = "No module named 'PySide6'"
        raise ImportError(msg, name="PySide6")

    with (
        patch("photo_tagger.main.importlib.import_module", raise_import_error),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.gui()
    assert exc_info.value.code == 1
    assert "photo-tagger[gui]" in capsys.readouterr().err


def test_gui_propagates_non_qt_import_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """
    An ImportError from inside the gui module chain is not "PySide6 missing".

    Regression test: a broken transitive dependency used to be reported with the pip install hint,
    sending the user to reinstall an extra that was never the problem.
    """

    def raise_import_error(_name: str) -> object:
        msg = "cannot import name 'Broken' from 'somewhere.else'"
        raise ImportError(msg, name="somewhere.else")

    with (
        patch("photo_tagger.main.importlib.import_module", raise_import_error),
        pytest.raises(ImportError, match=r"somewhere\.else"),
    ):
        main_module.gui()
    assert "photo-tagger[gui]" not in capsys.readouterr().err


def test_gui_launches_when_available() -> None:
    """When PySide6 is present, the command delegates to gui.launch() and exits with its code."""
    fake_gui = type("FakeGui", (), {"launch": staticmethod(lambda: 0)})
    with (
        patch("photo_tagger.main.importlib.import_module", return_value=fake_gui),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.gui()
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# telemetry wiring
# ---------------------------------------------------------------------------


def _run_batch_firing_complete(
    image_files: list[Path],
    _agent: object,
    _options: object,
    **kwargs: object,
) -> object:
    """Stand-in for run_batch that drives the on_complete callback so the beacon fires."""
    on_complete = kwargs.get("on_complete")
    if on_complete is not None:
        on_complete(BatchTotals(total_files=len(image_files)))  # type: ignore[operator]
    return None


def test_cli_emits_telemetry_on_completion(tmp_path: Path) -> None:
    """A successful run reports interface, provider, model, and batch size to telemetry."""
    image = _make_jpeg(tmp_path / "img.cr3")
    emitted: dict[str, Any] = {}

    def fake_emit(run: Any, *, enabled: bool, block: bool = False) -> None:  # noqa: ANN401
        emitted["run"] = run
        emitted["enabled"] = enabled
        emitted["block"] = block

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=_run_batch_firing_complete),
        patch.object(telemetry, "emit", side_effect=fake_emit),
    ):
        _run_app(["--input", str(image), "--provider", "ollama", "--model", "my-vlm"])

    assert emitted["enabled"] is True
    assert emitted["block"] is True
    assert emitted["run"].interface == "cli"
    assert emitted["run"].provider == "ollama"
    assert emitted["run"].model == "my-vlm"
    assert emitted["run"].batch_size == 1


def test_cli_no_telemetry_flag_disables_and_silences_notice(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--no-telemetry passes enabled=False to emit and suppresses the first-run notice."""
    image = _make_jpeg(tmp_path / "img.cr3")
    emitted: dict[str, Any] = {}

    def fake_emit(_run: Any, *, enabled: bool, block: bool = False) -> None:  # noqa: ANN401
        emitted["enabled"] = enabled

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=_run_batch_firing_complete),
        patch.object(telemetry, "emit", side_effect=fake_emit),
    ):
        _run_app(["--input", str(image), "--no-telemetry"])

    assert emitted["enabled"] is False
    assert "anonymous usage stats" not in capsys.readouterr().err


def test_cli_telemetry_notice_shown_only_on_first_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With telemetry on (the default), the disclosure prints once and stays quiet after."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])
        first_run_err = capsys.readouterr().err
        _run_app(["--input", str(image)])
        second_run_err = capsys.readouterr().err

    assert "anonymous usage stats" in first_run_err
    assert "anonymous usage stats" not in second_run_err


def _run_with_notify(tmp_path: Path, args: list[str], totals: BatchTotals) -> dict[str, Any]:
    """Run the CLI over one photo with *totals*, capturing the desktop alert it raises."""
    image = _make_jpeg(tmp_path / "img.cr3")
    raised: dict[str, Any] = {}

    def fake_run_batch(image_files: list[Path], *_args: object, **kwargs: object) -> None:
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            on_complete(totals)  # type: ignore[operator]
        assert image_files

    def fake_notify(title: str, message: str, **kwargs: object) -> None:
        raised["title"] = title
        raised["message"] = message
        raised.update(kwargs)

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=fake_run_batch),
        patch.object(cli_run, "notify", side_effect=fake_notify),
    ):
        _run_app(["--input", str(image), *args])
    return raised


def test_cli_raises_no_alert_by_default(tmp_path: Path) -> None:
    """A run nobody asked to be told about passes enabled=False, so nothing is spawned."""
    raised = _run_with_notify(tmp_path, [], BatchTotals(total_files=1, success=1))

    assert raised["enabled"] is False


def test_cli_notify_flag_announces_the_finished_batch(tmp_path: Path) -> None:
    """--notify alerts with the run's tally, waiting for it because the CLI exits right after."""
    raised = _run_with_notify(
        tmp_path,
        ["--notify"],
        BatchTotals(total_files=4, success=4),
    )

    assert raised["enabled"] is True
    assert raised["sound"] is False
    assert raised["block"] is True
    assert raised["title"] == "Photo Tagger"
    assert raised["message"] == "4 of 4 photos tagged."


def test_cli_notify_sound_implies_the_notification(tmp_path: Path) -> None:
    """--notify-sound on its own asks for the notification that carries the sound."""
    raised = _run_with_notify(
        tmp_path,
        ["--notify-sound"],
        BatchTotals(total_files=1, success=1),
    )

    assert raised["enabled"] is True
    assert raised["sound"] is True


def test_cli_notify_reports_a_partly_failed_batch(tmp_path: Path) -> None:
    """The alert fires for a run that half failed, which is the one worth being called back for."""
    raised = _run_with_notify(
        tmp_path,
        ["--notify"],
        BatchTotals(total_files=4, success=3, failed_files=["bad.cr3"]),
    )

    assert raised["message"] == "3 of 4 photos tagged, 1 failed."


def test_cli_notify_can_be_switched_on_from_the_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[display] notify = true spares the user from typing the flag on every run."""
    _point_config_at(tmp_path, monkeypatch, "[display]\nnotify = true\nnotify_sound = true\n")

    raised = _run_with_notify(tmp_path, [], BatchTotals(total_files=1, success=1))

    assert raised["enabled"] is True
    assert raised["sound"] is True


def test_cli_no_notify_flag_overrides_the_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag still beats the config, so one quiet run does not mean editing the file."""
    _point_config_at(tmp_path, monkeypatch, "[display]\nnotify = true\n")

    raised = _run_with_notify(tmp_path, ["--no-notify"], BatchTotals(total_files=1, success=1))

    assert raised["enabled"] is False


def _point_config_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Write *text* to a TOML file and point PHOTO_TAGGER_CONFIG at it for this test."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(text, encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(cfg))


def test_config_file_fills_flags_the_user_did_not_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config values reach the pipeline when the corresponding flags are absent."""
    _point_config_at(
        tmp_path,
        monkeypatch,
        "workers = 2\n\n[inference]\nmax_tokens = 500\n\n[output]\nbackup_xmp = false\n",
    )
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["options"].max_tokens == 500  # noqa: PLR2004
    assert captured["options"].backup_xmp is False
    assert captured["workers"] == 2  # noqa: PLR2004


def test_config_file_survives_sibling_cli_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Passing one flag from a group must not reset the group's other fields to built-ins.

    Regression test: config defaults used to be baked into the default group instances, and cyclopts
    rebuilds a group from class defaults whenever any of its flags is passed, silently dropping the
    config values of every sibling field.
    """
    _point_config_at(tmp_path, monkeypatch, "[inference]\nmax_tokens = 500\n")
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--temperature", "0.9"])

    assert captured["options"].temperature == 0.9  # noqa: PLR2004
    assert captured["options"].max_tokens == 500  # noqa: PLR2004


def test_config_file_sets_a_flag_a_command_shares_its_name_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``[output] vocabulary`` must reach the run even though a ``vocabulary`` command exists.

    Regression test: cyclopts drops every flat-config key that names a subcommand before matching it
    against anything, so this key was silently ignored the moment the command was added, while the
    keys next to it in the same table kept working.
    """
    listing = tmp_path / "keywords.txt"
    listing.write_text("Osprey\n", encoding="utf-8")
    _point_config_at(
        tmp_path,
        monkeypatch,
        f'[output]\nvocabulary = "{listing.as_posix()}"\nmax_keywords = 7\n',
    )
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    vocabulary = captured["options"].vocabulary
    assert vocabulary is not None
    assert vocabulary.match("ospreys") == "Osprey"
    assert captured["options"].max_new_keywords == 7  # noqa: PLR2004 - the key next to it


def test_cli_flag_beats_a_config_key_a_command_shares_its_name_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command line still wins for a key applied outside cyclopts' own config layer."""
    from_config = tmp_path / "config-keywords.txt"
    from_config.write_text("Osprey\n", encoding="utf-8")
    from_flag = tmp_path / "flag-keywords.txt"
    from_flag.write_text("Tractor\n", encoding="utf-8")
    _point_config_at(
        tmp_path,
        monkeypatch,
        f'[output]\nvocabulary = "{from_config.as_posix()}"\n',
    )
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--vocabulary", str(from_flag)])

    assert captured["options"].vocabulary.terms == ("Tractor",)


def test_cli_flag_overrides_config_file_for_the_same_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit flag beats the config file for that field."""
    _point_config_at(tmp_path, monkeypatch, "[inference]\nmax_tokens = 500\n")
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--max-tokens", "800"])

    assert captured["options"].max_tokens == 800  # noqa: PLR2004


def test_cli_config_overrides_translates_field_names_to_option_names() -> None:
    """Renamed fields map to their CLI spelling; unknown keys and tables are dropped."""
    from photo_tagger.cli_options import cli_config_overrides  # noqa: PLC0415

    flat = cli_config_overrides(
        {
            "provider": {"model_name": "my-vlm", "api_base_url": "http://h:1/v1", "nope": 1},
            "output": {"use_sidecar": False},
            "telemetry": {"enabled": False},
            "display": {"progress_bar": False, "json_output": True},
            "workers": 4,
            "recursive": True,
            "exiftool_path": "/x/exiftool",
            "future_table": {"key": "value"},
        },
    )

    assert flat == {
        "model": "my-vlm",
        "url": "http://h:1/v1",
        "write-sidecar": False,
        "telemetry": False,
        "progress": False,
        "json": True,
        "workers": 4,
        "recursive": True,
    }


@pytest.mark.parametrize(
    ("sidecar_mode", "use_sidecar", "expected"),
    [
        # Nothing asked: the non-destructive default.
        (None, None, "all"),
        # The old two-state flags still mean what they always did.
        (None, True, "all"),
        (None, False, "none"),
        # The mode is the specific answer, so it wins over the shorthand either way.
        ("raw", None, "raw"),
        ("raw", False, "raw"),
        ("none", True, "none"),
    ],
)
def test_resolve_sidecar_mode_precedence(
    sidecar_mode: str | None,
    expected: str,
    *,
    use_sidecar: bool | None,
) -> None:
    """--sidecar-mode wins; --write-sidecar/--embed-in-photo stay the two-state shorthand."""
    from photo_tagger.cli_options import OutputConfig, resolve_sidecar_mode  # noqa: PLC0415

    output = OutputConfig(sidecar_mode=sidecar_mode, use_sidecar=use_sidecar)  # type: ignore[arg-type]

    assert resolve_sidecar_mode(output) == expected


@pytest.mark.parametrize(
    ("notify_flag", "sound_flag", "expected"),
    [
        (None, False, (False, False)),
        (False, False, (False, False)),
        (True, False, (True, False)),
        (True, True, (True, True)),
        # --notify-sound alone: the sound is how the notification announces itself.
        (None, True, (True, True)),
        # A typed --no-notify (or notify = false in the config file) beats the sound.
        (False, True, (False, False)),
    ],
)
def test_resolve_notify_precedence(
    expected: tuple[bool, bool],
    *,
    notify_flag: bool | None,
    sound_flag: bool,
) -> None:
    """The two alert flags fold into (raise an alert, make it audible)."""
    from photo_tagger.cli_options import DisplayConfig, resolve_notify  # noqa: PLC0415

    display = DisplayConfig(notify=notify_flag, notify_sound=sound_flag)

    assert resolve_notify(display) == expected


def test_sidecar_mode_reaches_the_pipeline_options() -> None:
    """A config file's [output] sidecar_mode lands on the options the pipeline writes with."""
    from photo_tagger.cli_options import load_defaults, to_processing_options  # noqa: PLC0415

    defaults = load_defaults({"output": {"sidecar_mode": "raw"}})

    options = to_processing_options(defaults.output, defaults.inference)

    assert options.sidecar_mode == "raw"


def test_sidecar_overrides_reach_the_pipeline_options() -> None:
    """A config file's [output] sidecar_overrides lands on the options, parsed per extension."""
    from photo_tagger.cli_options import load_defaults, to_processing_options  # noqa: PLC0415

    defaults = load_defaults({"output": {"sidecar_overrides": ["DNG=photo"]}})

    options = to_processing_options(defaults.output, defaults.inference)

    assert options.sidecar_overrides == {".dng": "photo"}


def test_sidecar_override_flag_reaches_the_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag wins over the config file's list, and it leaves the mode it sits beside alone."""
    _point_config_at(tmp_path, monkeypatch, '[output]\nsidecar_overrides = ["tif=both"]\n')
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--sidecar-override", "dng=photo"])

    # A list flag replaces the config's list, as every repeatable flag does.
    assert captured["options"].sidecar_overrides == {".dng": "photo"}
    assert captured["options"].sidecar_mode == "all"


def test_sidecar_override_typo_fails_before_any_photo(tmp_path: Path) -> None:
    """A wrong target is a parse error, not a run that quietly writes sidecars instead."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch, pytest.raises(SystemExit) as exit_info:
        main_module.app(["--input", str(image), "--sidecar-override", "dng=embed"])

    assert exit_info.value.code != 0
    assert "options" not in captured


def test_preserve_flags_reach_the_pipeline_per_field() -> None:
    """Each field's keep-or-replace answer travels on its own, keywords included."""
    from photo_tagger.cli_options import load_defaults, to_processing_options  # noqa: PLC0415

    defaults = load_defaults(
        {"output": {"preserve_description": True, "preserve_keywords": False}},
    )

    options = to_processing_options(defaults.output, defaults.inference)

    assert options.preserve_existing_description is True
    assert options.preserve_existing_title is False
    assert options.preserve_existing_kw is False


def test_main_reports_unhandled_crashes_and_re_raises(tmp_path: Path) -> None:
    """
    A crash escaping the CLI fires one anonymous crash beacon and still surfaces the traceback.

    SystemExit (clean error handling) must never be reported as a crash.
    """
    image = _make_jpeg(tmp_path / "img.cr3")
    crashes: list[dict[str, Any]] = []

    def fake_emit_crash(exc: BaseException, **kwargs: Any) -> None:  # noqa: ANN401
        crashes.append({"exc": exc, **kwargs})

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=RuntimeError("boom")),
        patch.object(telemetry, "emit_crash", side_effect=fake_emit_crash),
        pytest.raises(RuntimeError, match="boom"),
    ):
        main_module.main(["--input", str(image)])

    assert len(crashes) == 1
    assert isinstance(crashes[0]["exc"], RuntimeError)
    assert crashes[0]["interface"] == "cli"


def test_a_first_run_crash_shows_the_notice_before_its_beacon(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A crash on the very first run is the first beacon, so the disclosure must come before it."""
    image = _make_jpeg(tmp_path / "img.cr3")
    notice_before_send: list[bool] = []

    def fake_emit_crash(_exc: BaseException, **_kwargs: Any) -> None:  # noqa: ANN401
        notice_before_send.append(not telemetry.notice_due())

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", side_effect=RuntimeError("boom")),
        patch.object(telemetry, "emit_crash", side_effect=fake_emit_crash),
        pytest.raises(RuntimeError, match="boom"),
    ):
        main_module.main(["--input", str(image)])

    assert notice_before_send == [True]
    assert "anonymous usage stats" in capsys.readouterr().err


def test_main_does_not_report_clean_exits_as_crashes() -> None:
    """SystemExit from normal error handling passes through without a crash beacon."""
    crashes: list[object] = []
    with (
        patch.object(telemetry, "emit_crash", side_effect=lambda *a, **_k: crashes.append(a)),
        patch.object(main_module, "app", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        main_module.main([])
    assert crashes == []


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["undo", "--list"], "undo"),
        (["undo", "--help"], "help"),
        (["--version"], "version"),
        (["-i", "photo.jpg"], "tag"),
        ([], "tag"),
    ],
)
def test_command_reporter_names_the_command(tokens: list[str], expected: str) -> None:
    """Help and the version win over a subcommand, and anything else is the default tag."""
    reporter = cli_telemetry.CommandReporter(
        tokens,
        commands=frozenset({"undo", "watch"}),
        source=ConfigFileSource(),
    )

    assert reporter.command == expected


def test_telemetry_enabled_for_honors_argv_flag() -> None:
    """A --no-telemetry anywhere on the command line disables the crash beacon too."""
    assert cli_telemetry.telemetry_enabled_for(["-i", "x", "--no-telemetry"]) is False
    assert cli_telemetry.telemetry_enabled_for(["-i", "x"]) is True


def test_telemetry_enabled_for_honors_the_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[telemetry] enabled = false in the config disables crash beacons; garbage tables do not."""
    _point_config_at(tmp_path, monkeypatch, "[telemetry]\nenabled = false\n")
    assert cli_telemetry.telemetry_enabled_for([]) is False

    _point_config_at(tmp_path, monkeypatch, 'telemetry = "not-a-table"\n')
    assert cli_telemetry.telemetry_enabled_for([]) is True


def test_suite_is_isolated_from_developer_config() -> None:
    """
    Conftest points PHOTO_TAGGER_CONFIG at an empty file, so defaults are the built-ins.

    Guards that isolation: a real ~/.config/photo-tagger/config.toml must not leak into the suite.
    (A developer config setting a relative ``summary_file`` once wrote a stray file into the repo
    when the on_complete tests fired.)
    """
    defaults = load_defaults()
    assert defaults.artifacts.summary_file is None
    assert defaults.telemetry.enabled is True


def test_load_defaults_reads_exiftool_path() -> None:
    """A config-file exiftool_path surfaces on Defaults; absent means None."""
    assert load_defaults({"exiftool_path": "/x/exiftool"}).exiftool_path == "/x/exiftool"
    assert load_defaults({}).exiftool_path is None


def test_export_exiftool_path_bridges_to_env_without_overriding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill an unset env var from config; never override an exported one; no-op when unset."""
    import os  # noqa: PLC0415 - test-local

    monkeypatch.setattr(config_file, "configured_exiftool_path", lambda: "/cfg/exiftool")
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    config_file.export_exiftool_path()
    assert os.environ["PHOTO_TAGGER_EXIFTOOL"] == "/cfg/exiftool"

    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "/env/exiftool")
    config_file.export_exiftool_path()
    assert os.environ["PHOTO_TAGGER_EXIFTOOL"] == "/env/exiftool"

    monkeypatch.setattr(config_file, "configured_exiftool_path", lambda: None)
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    config_file.export_exiftool_path()
    assert "PHOTO_TAGGER_EXIFTOOL" not in os.environ


def test_cli_vocabulary_follows_the_output_language(tmp_path: Path) -> None:
    """--lang decides whether the vocabulary index folds English plurals."""
    image = _make_jpeg(tmp_path / "img.cr3")
    vocabulary = tmp_path / "keywords.txt"
    vocabulary.write_text("Alle\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            ["--input", str(image), "--vocabulary", str(vocabulary), "--lang", "German"],
        )

    options = captured["options"]
    assert options.output_language == "German"
    assert options.vocabulary.fold_plurals is False
    assert options.vocabulary.match("Alles") is None


def test_vocabulary_command_builds_a_file_from_the_photos(tmp_path: Path) -> None:
    """The census reads the photos' own keywords and writes a file the tagger can load."""
    image = _make_jpeg(tmp_path / "img.cr3")
    output = tmp_path / "vocabulary.txt"
    report = tmp_path / "dropped.csv"
    census = KeywordCensus()
    census.add(["Animal", "Bird"], weight=5)
    census.add(["One Off"], weight=1)

    with patch.object(cli_vocabulary, "census_from_photos", return_value=census) as counted:
        _run_app(
            [
                "vocabulary",
                "--input",
                str(image),
                "--output",
                str(output),
                "--report",
                str(report),
            ],
        )

    assert counted.call_args.args[0] == [image]
    written = output.read_text(encoding="utf-8")
    assert "Animal\nAnimal|Bird\n" in written
    assert "One Off" not in written
    assert "One Off,1,rare" in report.read_text(encoding="utf-8")


def test_vocabulary_command_reads_a_keyword_export(tmp_path: Path) -> None:
    """--from-export works without any photos, for a catalog that is not on this machine."""
    export = tmp_path / "keywords.txt"
    export.write_text("Animal\n\tBird\nAnimal\n\tBird\n", encoding="utf-8")
    output = tmp_path / "vocabulary.txt"

    _run_app(["vocabulary", "--from-export", str(export), "--output", str(output)])

    written = output.read_text(encoding="utf-8")
    assert "Bird" in written
    assert "keyword export keywords.txt" in written


def test_vocabulary_command_reads_an_export_saved_with_a_bom(tmp_path: Path) -> None:
    """A keyword export from a Windows editor starts with a BOM; the first keyword survives it."""
    export = tmp_path / "keywords.txt"
    export.write_bytes(b"\xef\xbb\xbfAnimal\n\tBird\nAnimal\n\tBird\n")
    output = tmp_path / "vocabulary.txt"

    _run_app(["vocabulary", "--from-export", str(export), "--output", str(output)])

    assert output.read_text(encoding="utf-8").splitlines()[-2:] == ["Animal", "Animal|Bird"]


def test_vocabulary_command_exits_1_on_an_export_it_cannot_decode(tmp_path: Path) -> None:
    """A UTF-16 export (or a binary file picked by mistake) is a bad input, not a traceback."""
    export = tmp_path / "keywords.txt"
    export.write_bytes("Osprey\nBird\n".encode("utf-16"))
    output = tmp_path / "vocabulary.txt"

    with pytest.raises(SystemExit) as exit_info:
        main_module.app(
            ["vocabulary", "--from-export", str(export), "--output", str(output)],
        )

    assert exit_info.value.code == 1
    assert not output.exists()


def test_vocabulary_command_exits_1_without_a_source(tmp_path: Path) -> None:
    """Neither photos nor an export means there is nothing to count."""
    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["vocabulary", "--output", str(tmp_path / "out.txt")])

    assert exit_info.value.code == 1


def test_vocabulary_command_exits_1_when_no_keywords_are_found(tmp_path: Path) -> None:
    """An untagged library cannot seed a vocabulary, so it says so instead of writing a file."""
    image = _make_jpeg(tmp_path / "img.cr3")
    output = tmp_path / "vocabulary.txt"

    with (
        patch.object(cli_vocabulary, "census_from_photos", return_value=KeywordCensus()),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["vocabulary", "--input", str(image), "--output", str(output)])

    assert exit_info.value.code == 1
    assert not output.exists()


def test_vocabulary_command_organizes_when_asked(tmp_path: Path) -> None:
    """--organize hands the trimmed list and the provider flags to the model pass."""
    export = tmp_path / "keywords.txt"
    export.write_text("Golden Hour\nGolden Hour\nGolden Light\nGolden Light\n", encoding="utf-8")
    output = tmp_path / "vocabulary.txt"
    census = KeywordCensus()
    census.add(["Golden Hour"], weight=2)
    organized = TrimResult(kept=["Golden Hour"], census=census, synonyms={})
    stats = OrganizeStats(model_name="test-model", categories=["Lighting"], grouped=1)

    with patch.object(cli_vocabulary, "organize", return_value=(organized, stats)) as organizer:
        _run_app(
            [
                "vocabulary",
                "--from-export",
                str(export),
                "--output",
                str(output),
                "--organize",
                "--model",
                "test-model",
                "--provider",
                "ollama",
            ],
        )

    kwargs = organizer.call_args.kwargs
    assert kwargs["provider_name"] == "ollama"
    assert kwargs["model_name"] == "test-model"
    written = output.read_text(encoding="utf-8")
    assert "Organized by test-model" in written
    assert "Categories (written to your photos as parents): Lighting" in written


def test_vocabulary_command_exits_1_when_the_provider_is_unreachable(tmp_path: Path) -> None:
    """Asking for --organize and silently not organizing would be worse than stopping."""
    export = tmp_path / "keywords.txt"
    export.write_text("Bird\nBird\n", encoding="utf-8")
    output = tmp_path / "vocabulary.txt"

    with (
        patch.object(cli_vocabulary, "organize", side_effect=ProviderError("no provider")),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(
            ["vocabulary", "--from-export", str(export), "--output", str(output), "--organize"],
        )

    assert exit_info.value.code == 1


# ---------------------------------------------------------------------------
# undo command
# ---------------------------------------------------------------------------


def _journal_with(tmp_path: Path, target: Path, *, created: bool) -> Path:
    """Write a one-entry journal describing *target* as it currently is on disk."""
    stat = target.stat()
    journal = tmp_path / "runs" / "20260501090000-1.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps(
            {
                "image": str(target.with_suffix(".cr3")),
                "target": str(target),
                "created": created,
                "backup": None,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    return journal


def test_undo_exits_when_there_is_nothing_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine that never tagged anything gets a message and exit 1, not a traceback."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["undo"])
    assert exit_info.value.code == 1


def test_undo_puts_back_the_last_run(tmp_path: Path) -> None:
    """The default target is the newest journal, and a created sidecar is deleted."""
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("generated", encoding="utf-8")
    journal = _journal_with(tmp_path, sidecar, created=True)

    with contextlib.suppress(SystemExit):
        main_module.app(["undo", "--run", str(journal)])

    assert not sidecar.exists()


def test_undo_dry_run_leaves_the_files_alone(tmp_path: Path) -> None:
    """--dry-run reports the same plan without carrying it out."""
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("generated", encoding="utf-8")
    journal = _journal_with(tmp_path, sidecar, created=True)

    with contextlib.suppress(SystemExit):
        main_module.app(["undo", "--run", str(journal), "--dry-run"])

    assert sidecar.exists()


def test_undo_exits_1_when_an_entry_is_left_alone(tmp_path: Path) -> None:
    """A file changed since the run blocks a clean exit, so scripts notice."""
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("generated", encoding="utf-8")
    journal = _journal_with(tmp_path, sidecar, created=True)
    sidecar.write_text("edited since the run", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["undo", "--run", str(journal)])

    assert exit_info.value.code == 1
    assert sidecar.exists()


def test_undo_list_reports_recorded_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--list names each journal and how many files it covers."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("generated", encoding="utf-8")
    journal = _journal_with(tmp_path / "state" / "photo-tagger", sidecar, created=True)

    with contextlib.suppress(SystemExit):
        main_module.app(["undo", "--list"])

    out = capsys.readouterr().out
    assert journal.name in out
    assert "1 file(s)" in out


def test_undo_rejects_an_empty_journal(tmp_path: Path) -> None:
    """A journal with no entries is nothing to undo, and says so with exit 1."""
    journal = tmp_path / "empty.jsonl"
    journal.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["undo", "--run", str(journal)])

    assert exit_info.value.code == 1


def test_cli_opens_an_undo_journal_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal run records what it writes; --no-undo-log turns that off."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])
    assert captured["journal"] is not None

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--no-undo-log"])
    assert captured["journal"] is None


def test_cli_records_no_undo_journal_for_a_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry run writes nothing, so there is nothing to put back."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--dry-run"])

    assert captured["journal"] is None


def test_undo_list_says_when_there_is_nothing_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--list on a machine that never tagged anything exits 0 with a plain message."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with contextlib.suppress(SystemExit):
        main_module.app(["undo", "--list"])
    assert "No recorded runs to undo." in capsys.readouterr().out


def test_undo_reports_an_unreadable_journal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A journal that cannot be read is a message and exit 1, not a traceback."""
    journal = tmp_path / "run.jsonl"
    journal.write_text("{}", encoding="utf-8")

    with (
        patch("photo_tagger.cli_undo.read_journal", side_effect=UndoError("boom")),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["undo", "--run", str(journal)])

    assert exit_info.value.code == 1
    assert "boom" in capsys.readouterr().out


def test_cli_reports_the_journal_it_wrote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that recorded writes names its journal in the log, so undo is discoverable."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    image = _make_jpeg(tmp_path / "img.cr3")
    sidecar = tmp_path / "img.xmp"
    sidecar.write_text("written", encoding="utf-8")
    captured: dict[str, Any] = {}

    def recording_run_batch(*_args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        captured["journal"] = kwargs["journal"]
        kwargs["journal"].record(image, sidecar, created=True)

    setup, create_agent, _ = _patches(captured)
    with (
        setup,
        create_agent,
        patch.object(cli_run, "run_batch", side_effect=recording_run_batch),
    ):
        _run_app(["--input", str(image)])

    assert captured["journal"].entries == 1
    assert captured["journal"].path.exists()


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------


def _settled_photo(path: Path) -> Path:
    """Create a photo old enough for the watcher to consider it finished."""
    _make_jpeg(path)
    stamp = time.time() - 60
    os.utime(path, (stamp, stamp))
    return path


def test_watch_tags_each_batch_with_one_shared_setup(tmp_path: Path) -> None:
    """Every batch reuses the same agent, cache, and journal instead of rebuilding them."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    first = _settled_photo(inbox / "a.cr3")
    calls: list[list[Path]] = []
    setups: list[object] = []

    def fake_process(image_files: list[Path], setup: object) -> None:
        calls.append(list(image_files))
        setups.append(setup)
        if len(calls) == 1:
            _settled_photo(inbox / "b.cr3")

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "process_batch", side_effect=fake_process),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])

    assert calls == [[first], [inbox / "b.cr3"]]
    assert setups[0] is setups[1]


def test_watch_records_one_undo_journal_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A week-long watch must not fold every import into one journal to undo."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    journals: list[UndoJournal | None] = []

    def fake_process(image_files: list[Path], setup: Any) -> None:  # noqa: ANN401 - the run setup
        journals.append(setup.journal)
        if len(journals) == 1:
            _settled_photo(inbox / "b.cr3")

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "process_batch", side_effect=fake_process),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])

    first_journal, second_journal = journals
    assert first_journal is not None
    assert second_journal is not None
    assert first_journal.path != second_journal.path


def test_watch_records_nothing_when_the_undo_log_is_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-undo-log holds for every batch, not just the first."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    journals: list[UndoJournal | None] = []

    def fake_process(image_files: list[Path], setup: Any) -> None:  # noqa: ANN401 - the run setup
        journals.append(setup.journal)

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "process_batch", side_effect=fake_process),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox), "--no-undo-log"])

    assert journals == [None]


def _bounded_watch(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - passthrough shim
    """
    Run the real watcher, bounded so the test terminates.

    Four polls: every file needs two (one to record it, one to confirm it has not changed), and a
    photo that lands while the first batch is being tagged only starts that clock on poll three.
    """
    from photo_tagger.watch import watch_batches as real_watch  # noqa: PLC0415

    kwargs["interval_seconds"] = 0
    return real_watch(*args, max_polls=4, **kwargs)


def test_watch_keeps_going_after_a_batch_with_failures(tmp_path: Path) -> None:
    """One bad batch is logged; the watch does not end because a photo failed."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    calls: list[list[Path]] = []

    def failing_process(image_files: list[Path], _setup: object) -> None:
        calls.append(list(image_files))
        if len(calls) == 1:
            _settled_photo(inbox / "b.cr3")
            raise BatchError(BatchTotals(total_files=1))

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "process_batch", side_effect=failing_process),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])

    expected_batches = 2
    assert len(calls) == expected_batches


def test_watch_stops_after_an_interrupted_batch(tmp_path: Path) -> None:
    """
    A Ctrl-C mid-batch ends the watch instead of waiting for the next import.

    run_batch absorbs the interrupt to write its summary, so the watch reads it off the totals.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    calls: list[list[Path]] = []

    def interrupted_process(image_files: list[Path], _setup: object) -> None:
        calls.append(list(image_files))
        _settled_photo(inbox / "b.cr3")
        raise BatchError(BatchTotals(total_files=1, interrupted=True))

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "process_batch", side_effect=interrupted_process),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])

    assert len(calls) == 1


def test_watch_applies_the_batch_filters(tmp_path: Path) -> None:
    """--skip-tagged and friends filter each batch, and an empty one is not processed."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    photo = _settled_photo(inbox / "a.cr3")
    calls: list[list[Path]] = []

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(
            cli_run,
            "process_batch",
            side_effect=lambda files, _s: calls.append(
                list(files),
            ),
        ),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
        patch("photo_tagger.discovery.find_tagged_images", return_value={photo}),
    ):
        _run_app(["watch", "--input", str(inbox), "--skip-tagged"])

    assert calls == []


def test_watch_workers_flag_reaches_the_run(tmp_path: Path) -> None:
    """--workers is read off the watch option group, so the flag must still land on the run."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_watch, "_watch_forever") as forever,
    ):
        _run_app(["watch", "--input", str(inbox), "--workers", "3"])

    assert forever.call_args.args[0].workers == 3  # noqa: PLR2004 - the value typed above.


def test_watch_alerts_once_per_batch(tmp_path: Path) -> None:
    """A watch is the unattended case, so every batch it finishes gets its own notification."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    raised: list[tuple[str, bool]] = []

    def run_one_batch(image_files: list[Path], *_args: object, **kwargs: object) -> None:
        # A second photo lands while the first batch runs, so the watch has another to pick up.
        if not raised:
            _settled_photo(inbox / "b.cr3")
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            on_complete(BatchTotals(total_files=len(image_files), success=len(image_files)))  # type: ignore[operator]

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=run_one_batch),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
        patch.object(
            cli_run,
            "notify",
            side_effect=lambda _t, message, **kwargs: raised.append((message, kwargs["enabled"])),
        ),
    ):
        _run_app(["watch", "--input", str(inbox), "--notify"])

    assert raised == [("1 of 1 photo tagged.", True), ("1 of 1 photo tagged.", True)]


def test_watch_stops_cleanly_on_ctrl_c(tmp_path: Path) -> None:
    """Ctrl-C ends the watch without a traceback and closes what the run opened."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")

    def interrupt(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        raise KeyboardInterrupt

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "process_batch", side_effect=interrupt),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])


def test_watch_requires_an_input(tmp_path: Path) -> None:
    """Watching nothing is a clean exit 1, not an infinite loop over an empty list."""
    with (
        patch.object(cli_run, "setup_logging"),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["watch"])

    assert exit_info.value.code == 1
    assert not (tmp_path / "unused").exists()


def test_watch_exits_when_logging_cannot_be_set_up(tmp_path: Path) -> None:
    """The same clean failure as `tag` when the log folder is unusable."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    with (
        patch.object(cli_run, "setup_logging", side_effect=OSError("read-only")),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["watch", "--input", str(inbox)])

    assert exit_info.value.code == 1


def test_watch_exits_1_when_the_vocabulary_is_unusable(tmp_path: Path) -> None:
    """A bad --vocabulary stops the watch before it starts, like it stops a tagging run."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    vocabulary = tmp_path / "keywords.txt"
    vocabulary.write_text("# nothing\n", encoding="utf-8")

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["watch", "--input", str(inbox), "--vocabulary", str(vocabulary)])

    assert exit_info.value.code == 1


def test_watch_reads_the_interval_and_settle_flags(tmp_path: Path) -> None:
    """The two watch knobs moved into an option group; the flags themselves did not change."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    seen: dict[str, float] = {}

    def fake_watch(*_args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - passthrough shim
        seen["interval"] = kwargs["interval_seconds"]
        seen["settle"] = kwargs["settle_seconds"]
        return iter(())

    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_watch, "watch_batches", side_effect=fake_watch),
    ):
        _run_app(["watch", "--input", str(inbox), "--interval", "11", "--settle", "3"])

    assert seen == {"interval": 11.0, "settle": 3.0}


def test_show_prints_the_curated_fields_with_their_tags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The table names the value and the ExifTool tag behind it."""
    image = _make_jpeg(tmp_path / "img.cr3")
    readings = [
        FieldReading(label="Creator", value="Ada Lovelace", tag="XMP:Creator", source="image file"),
    ]

    with patch.object(cli_fields, "read_fields", return_value=readings):
        _run_app(["show", "--input", str(image)])

    out = capsys.readouterr().out
    assert "Ada Lovelace" in out
    assert "XMP:Creator" in out


def test_show_marks_a_value_that_lives_in_the_sidecar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Which file answers is part of the report, because a catalog reads the sidecar."""
    image = _make_jpeg(tmp_path / "img.cr3")
    readings = [
        FieldReading(
            label="Title",
            value="From the sidecar",
            tag="XMP:Title",
            source=SOURCE_SIDECAR,
        ),
    ]

    with patch.object(cli_fields, "read_fields", return_value=readings):
        _run_app(["show", "--input", str(image)])

    assert "sidecar" in capsys.readouterr().out


def test_show_json_emits_one_parseable_line_per_photo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--json is a stream a pipe can read: one object per photo, never wrapped by the console."""
    image = _make_jpeg(tmp_path / "img.cr3")
    readings = [
        FieldReading(
            label="Copyright",
            value="© 2019 Ada Lovelace, All rights reserved, and then some more text to be sure",
            tag="XMP:Rights",
            source="image file",
        ),
    ]

    with patch.object(cli_fields, "read_fields", return_value=readings):
        _run_app(["show", "--input", str(image), "--json"])

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["file"] == str(image.resolve())
    assert payload["fields"][0]["tag"] == "XMP:Rights"


def test_show_reports_a_photo_with_no_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nothing found is said plainly, rather than printing an empty table."""
    image = _make_jpeg(tmp_path / "img.cr3")

    with patch.object(cli_fields, "read_fields", return_value=[]):
        _run_app(["show", "--input", str(image)])

    assert "no metadata" in capsys.readouterr().out


def test_show_all_dumps_every_tag_per_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--all groups by the file each tag came from, photo and sidecar alike."""
    image = _make_jpeg(tmp_path / "img.cr3")
    dump = {str(image): {"EXIF:Artist": "Ada"}, str(tmp_path / "img.xmp"): {"XMP:Creator": "Ada"}}

    with patch.object(cli_fields, "read_all_tags", return_value=dump):
        _run_app(["show", "--input", str(image), "--all"])

    out = capsys.readouterr().out
    assert "EXIF:Artist" in out
    assert "XMP:Creator" in out


def test_show_exits_1_when_nothing_matches(tmp_path: Path) -> None:
    """An empty folder is exit 1 with discovery's own message, so a script notices."""
    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["show", "--input", str(tmp_path)])
    assert exit_info.value.code == 1


def _taken(year: int, **more: str) -> FieldSources:
    """
    Return what a photo taken in *year* carries in its sidecar, plus *more* tags there.

    The sidecar, because `set` writes a RAW's sidecar by default, and preserve reads that file.
    """
    extra = {tag.replace("_", ":", 1): value for tag, value in more.items()}
    return FieldSources(sidecar={"EXIF:DateTimeOriginal": f"{year}:05:04 10:00:00", **extra})


def test_set_writes_the_copyright_fields_dated_per_photo(tmp_path: Path) -> None:
    """The year comes from each photo, so one command stamps a decade of shoots correctly."""
    old = _make_jpeg(tmp_path / "old.cr3")
    new = _make_jpeg(tmp_path / "new.cr3")
    written: dict[Path, dict[str, str]] = {}

    def fake_write(image: Path, _keywords: object, **kwargs: Any) -> tuple[bool, list[Any]]:  # noqa: ANN401 - mirrors the writer it stands in for.
        written[image] = kwargs["extra"]
        return (True, [])

    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={old: _taken(2011), new: _taken(2025)},
        ),
        patch.object(cli_fields, "write_metadata_everywhere", side_effect=fake_write),
    ):
        _run_app(
            [
                "set",
                "--input",
                str(old),
                "--input",
                str(new),
                "--artist",
                "Ada Lovelace",
                "--copyright",
                "Copyright (c) {year} {artist}, All rights reserved",
                "--rights",
                "© {year} {artist}, All rights reserved",
                "--marked",
            ],
        )

    assert written[old]["EXIF:Copyright"] == "Copyright (c) 2011 Ada Lovelace, All rights reserved"
    assert written[new]["XMP-dc:Rights"] == "© 2025 Ada Lovelace, All rights reserved"
    assert written[new]["XMP-dc:Creator"] == "Ada Lovelace"
    assert written[new]["XMP-xmpRights:Marked"] == "True"


def test_set_passes_the_sidecar_overrides_to_the_writer(tmp_path: Path) -> None:
    """`set` stamps a DNG the same place `tag` would, or Lightroom sees half the metadata."""
    image = _make_jpeg(tmp_path / "img.dng")

    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={image.resolve(): _taken(2025)},
        ),
        patch.object(cli_fields, "write_metadata_everywhere", return_value=(True, [])) as write,
    ):
        _run_app(
            ["set", "--input", str(image), "--artist", "Ada", "--sidecar-override", "dng=photo"],
        )

    assert write.call_args.kwargs["sidecar_overrides"] == {".dng": "photo"}


def test_set_records_its_writes_for_undo(tmp_path: Path) -> None:
    """A run that went to the wrong folder is one `photo-tagger undo` away."""
    image = _make_jpeg(tmp_path / "img.cr3")
    target = WriteTarget(path=tmp_path / "img.xmp", use_sidecar=True, existed=False)
    target.path.write_text("xmp")  # what the stubbed write would have created
    journal = UndoJournal(tmp_path / "run.jsonl")

    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={image.resolve(): _taken(2025)},
        ),
        patch.object(cli_fields, "write_metadata_everywhere", return_value=(True, [target])),
        patch.object(cli_fields, "open_journal", return_value=journal),
    ):
        _run_app(["set", "--input", str(image), "--artist", "Ada"])

    assert [record.created for record in read_journal(journal.path)] == [True]


def test_set_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run prints the tags each photo would get and opens nothing for writing."""
    image = _make_jpeg(tmp_path / "img.cr3")

    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={image.resolve(): _taken(2025)},
        ),
        patch.object(cli_fields, "write_metadata_everywhere") as write,
    ):
        _run_app(["set", "--input", str(image), "--dry-run", "--copyright", "(c) {year} Ada"])

    assert "(c) 2025 Ada" in capsys.readouterr().out
    write.assert_not_called()


def test_set_exits_1_when_no_field_was_asked_for(tmp_path: Path) -> None:
    """A `set` with nothing to set is a mistake worth reporting, not a silent no-op."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["set", "--input", str(image)])
    assert exit_info.value.code == 1


def test_set_refuses_exif_tags_that_only_sidecars_would_get(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """EXIF in a sidecar-only run would fail on every photo, so it stops first and says why."""
    image = _make_jpeg(tmp_path / "img.jpg")
    with (
        patch.object(cli_fields, "write_metadata_everywhere") as write,
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["set", "--input", str(image), "--set", "EXIF:Copyright=(c) Ada"])

    assert exit_info.value.code == 1
    assert "--embed-in-photo" in capsys.readouterr().out
    write.assert_not_called()


def test_set_lets_exif_tags_through_when_the_photo_is_written(tmp_path: Path) -> None:
    """With the photo among the targets, the same EXIF tag has somewhere to go."""
    image = _make_jpeg(tmp_path / "img.jpg")
    with patch.object(cli_fields, "write_metadata_everywhere", return_value=(True, [])) as write:
        _run_app(
            ["set", "--input", str(image), "--embed-in-photo", "--set", "EXIF:Copyright=(c) Ada"],
        )
    write.assert_called_once()


def test_set_exits_1_on_a_template_it_cannot_fill(tmp_path: Path) -> None:
    """A typo in the notice stops the run before the first photo is touched."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with (
        patch.object(cli_fields, "write_metadata_everywhere") as write,
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["set", "--input", str(image), "--copyright", "(c) {shot_at}"])
    assert exit_info.value.code == 1
    write.assert_not_called()


def test_set_exits_1_when_a_write_fails(tmp_path: Path) -> None:
    """A failed write is exit 1, so a scripted stamp does not report success it did not have."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={image.resolve(): _taken(2025)},
        ),
        patch.object(cli_fields, "write_metadata_everywhere", return_value=(False, [])),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(["set", "--input", str(image), "--artist", "Ada"])
    assert exit_info.value.code == 1


def test_set_says_how_many_photos_got_no_notice(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """{year!} on an undated photo writes no notice, and the summary says so."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={image.resolve(): FieldSources()},
        ),
        patch.object(cli_fields, "write_metadata_everywhere", return_value=(True, [])) as write,
    ):
        _run_app(["set", "--input", str(image), "--artist", "Ada", "--copyright", "(c) {year!}"])

    assert "EXIF:Copyright" not in write.call_args.kwargs["extra"]
    assert "1 photo(s) got no notice" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flag", "expected"),
    [([], None), (["--no-preserve-copyright"], "(c) 2025 Ada")],
)
def test_set_keeps_a_notice_unless_told_to_replace_it(
    tmp_path: Path,
    flag: list[str],
    expected: str | None,
) -> None:
    """A photo with a notice keeps it by default, and --no-preserve-copyright replaces it."""
    image = _make_jpeg(tmp_path / "img.cr3")
    values = _taken(2025, EXIF_Copyright="(c) 2011 Ada")
    with (
        patch.object(cli_fields, "read_field_sources", return_value={image.resolve(): values}),
        patch.object(cli_fields, "write_metadata_everywhere", return_value=(True, [])) as write,
    ):
        _run_app(["set", "--input", str(image), "--copyright", "(c) {year} Ada", *flag])

    extra: dict[str, str] = write.call_args.kwargs["extra"] if write.call_args else {}
    assert extra.get("EXIF:Copyright") == expected


def test_set_counts_a_photo_that_keeps_everything_as_kept_not_failed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second `set` over a stamped folder writes nothing and still exits 0."""
    image = _make_jpeg(tmp_path / "img.cr3")
    values = FieldSources(sidecar={"XMP:Rights": "(c) 2011 Ada", "XMP:Creator": "Ada"})
    with (
        patch.object(cli_fields, "read_field_sources", return_value={image.resolve(): values}),
        patch.object(cli_fields, "write_metadata_everywhere") as write,
    ):
        _run_app(["set", "--input", str(image), "--artist", "Ada", "--copyright", "(c) {year}"])

    write.assert_not_called()
    out = capsys.readouterr().out
    assert "Wrote 0 of 1" in out
    assert "1 photo(s) already carried everything asked for" in out


def test_set_does_not_count_a_skipped_photo_as_kept(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """{year!} on an undated photo with nothing else to write is skipped, not kept."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with (
        patch.object(
            cli_fields,
            "read_field_sources",
            return_value={image.resolve(): FieldSources()},
        ),
        patch.object(cli_fields, "write_metadata_everywhere") as write,
    ):
        _run_app(["set", "--input", str(image), "--copyright", "(c) {year!}"])

    write.assert_not_called()
    out = capsys.readouterr().out
    assert "already carried" not in out
    assert "1 photo(s) got no notice" in out


def test_tag_passes_the_copyright_fields_to_the_pipeline(tmp_path: Path) -> None:
    """The same flags work on a tagging run, so one pass credits the photos it touches."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--artist", "Ada", "--copyright", "(c) {year} {artist}"])

    fields = captured["options"].extra_fields
    assert fields.resolve(_taken(2025).values)["EXIF:Copyright"] == "(c) 2025 Ada"


def test_tag_reads_the_copyright_table_from_the_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[copyright] in the config file is how "always credit my photos" is spelled."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[copyright]\nartist = "Ada Lovelace"\ncopyright_notice = "(c) {year} {artist}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["options"].extra_fields.resolve({})["EXIF:Artist"] == "Ada Lovelace"


def test_show_all_json_emits_the_raw_tag_dump(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--all --json hands over everything ExifTool reported, keyed by file."""
    image = _make_jpeg(tmp_path / "img.cr3")
    dump = {str(image): {"EXIF:Artist": "Ada"}}

    with patch.object(cli_fields, "read_all_tags", return_value=dump):
        _run_app(["show", "--input", str(image), "--all", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["tags"][str(image)]["EXIF:Artist"] == "Ada"


@pytest.mark.parametrize("command", ["show", "set"])
def test_read_and_write_commands_report_an_unusable_input(tmp_path: Path, command: str) -> None:
    """Discovery failing is exit 1 with the reason, not a traceback."""
    image = _make_jpeg(tmp_path / "img.cr3")
    args = [command, "--input", str(image)]
    if command == "set":
        args += ["--artist", "Ada"]

    with (
        patch.object(cli_fields, "resolve_image_batch", side_effect=DiscoveryError("unreadable")),
        pytest.raises(SystemExit) as exit_info,
    ):
        main_module.app(args)

    assert exit_info.value.code == 1


def test_set_exits_1_when_no_photo_matches(tmp_path: Path) -> None:
    """Asking to credit an empty folder is exit 1, not a silent success."""
    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["set", "--input", str(tmp_path), "--artist", "Ada"])
    assert exit_info.value.code == 1


def test_no_write_copyright_skips_a_configured_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag skips a copyright the config file sets, for one run, without editing the file."""
    config = tmp_path / "config.toml"
    config.write_text('[copyright]\nartist = "Ada Lovelace"\n', encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--no-write-copyright"])

    assert captured["options"].extra_fields.is_empty


def test_the_copyright_table_wins_over_a_stale_output_key() -> None:
    """The GUI saves [copyright]; an older [output] artist must not undo what it saved."""
    defaults = load_defaults({"output": {"artist": "Old"}, "copyright": {"artist": "Me"}})
    assert defaults.output.artist == "Me"


_CONFIG_DUTY = 60
_CONFIG_FLOOR = 25
_CLI_DUTY = 80


def test_cli_without_pacing_flags_builds_no_pacer(tmp_path: Path) -> None:
    """An ordinary run is never paced, so nothing new can slow it down."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["pacer"] is None


def test_cli_duty_flag_paces_the_run(tmp_path: Path) -> None:
    """--duty reaches run_batch as a pacer carrying the plan the flags describe."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            [
                "--input",
                str(image),
                "--duty",
                str(_CLI_DUTY),
                "--work-minutes",
                "5",
                "--max-pause-minutes",
                "3",
            ],
        )

    pacer = captured["pacer"]
    assert pacer is not None
    assert pacer.plan.duty_percent == _CLI_DUTY
    assert pacer.plan.work_seconds == 5 * SECONDS_PER_MINUTE
    assert pacer.plan.max_pause_seconds == 3 * SECONDS_PER_MINUTE


def test_cli_battery_floor_alone_paces_the_run(tmp_path: Path) -> None:
    """Watching the battery is reason enough to build a pacer, with no duty cycle at all."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--battery-floor", str(_CONFIG_FLOOR)])

    pacer = captured["pacer"]
    assert pacer is not None
    assert pacer.plan.battery_floor == _CONFIG_FLOOR


def test_pacing_reads_the_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A [pacing] table paces every run without the flags being typed each time."""
    _point_config_at(
        tmp_path,
        monkeypatch,
        f"[pacing]\nduty = {_CONFIG_DUTY}\nbattery_floor = {_CONFIG_FLOOR}\n",
    )
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    pacer = captured["pacer"]
    assert pacer is not None
    assert pacer.plan.duty_percent == _CONFIG_DUTY
    assert pacer.plan.battery_floor == _CONFIG_FLOOR


def test_pacing_cli_flag_wins_over_the_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed --duty overrides the configured one, and leaves its siblings alone."""
    _point_config_at(
        tmp_path,
        monkeypatch,
        f"[pacing]\nduty = {_CONFIG_DUTY}\nbattery_floor = {_CONFIG_FLOOR}\n",
    )
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--duty", str(_CLI_DUTY)])

    pacer = captured["pacer"]
    assert pacer is not None
    assert pacer.plan.duty_percent == _CLI_DUTY
    assert pacer.plan.battery_floor == _CONFIG_FLOOR


def test_pacing_rejects_an_impossible_duty(tmp_path: Path) -> None:
    """A duty cycle outside 1-100 fails at parse time instead of resting forever."""
    image = _make_jpeg(tmp_path / "img.cr3")

    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["--input", str(image), "--duty", "0"])

    assert exit_info.value.code != 0


def test_load_defaults_exposes_the_pacing_group() -> None:
    """The GUI path resolves [pacing] too, since it never parses a command line."""
    defaults = load_defaults({"pacing": {"duty": _CONFIG_DUTY}})
    assert to_pacing_plan(defaults.pacing).duty_percent == _CONFIG_DUTY


def test_pacing_rejects_a_resume_level_below_the_floor(tmp_path: Path) -> None:
    """The pair only means something as a band, so a contradictory one fails at parse time."""
    image = _make_jpeg(tmp_path / "img.cr3")

    with pytest.raises(SystemExit) as exit_info:
        main_module.app(
            ["--input", str(image), "--battery-floor", "30", "--battery-resume", "20"],
        )

    assert exit_info.value.code != 0


def test_pacing_accepts_zero_as_a_derived_resume_level(tmp_path: Path) -> None:
    """Zero is not a contradiction: it asks for a level a margin above the floor."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--battery-floor", "30", "--battery-resume", "0"])

    assert captured["pacer"].plan.resume_percent == 45  # noqa: PLR2004 - floor plus the margin.


def test_pacing_keeps_a_floor_near_full_and_says_it_is_a_poor_idea(tmp_path: Path) -> None:
    """The run is paced as asked; the warning is in the log, not in a refusal."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--battery-floor", "99"])

    assert captured["pacer"].plan.battery_floor == 99  # noqa: PLR2004 - exactly what was asked.


def test_pacing_rejects_a_floor_with_no_room_for_a_resume_level(tmp_path: Path) -> None:
    """A floor at full leaves the resume level nowhere to sit, so it is refused."""
    image = _make_jpeg(tmp_path / "img.cr3")

    with pytest.raises(SystemExit) as exit_info:
        main_module.app(["--input", str(image), "--battery-floor", "100"])

    assert exit_info.value.code != 0


# ---------------------------------------------------------------------------
# telemetry beacons
# ---------------------------------------------------------------------------


@pytest.fixture
def beacons(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Capture every event the CLI hands to telemetry.emit, marking the process as having sent."""
    sent: list[object] = []

    def fake_emit(event: object, **_kwargs: object) -> None:
        sent.append(event)
        telemetry._mark_sent()  # noqa: SLF001 - what a real dispatch does.

    monkeypatch.setattr(telemetry, "_sent_any", False)
    monkeypatch.setattr(telemetry, "_notes", {})
    monkeypatch.setattr(telemetry, "emit", fake_emit)
    return sent


def _finishing_run_batch(totals: BatchTotals) -> Any:  # noqa: ANN401 - a patch side effect.
    """Return a run_batch stand-in that reports *totals* the way the real one does."""

    def fake(image_files: list[Path], *_args: object, **kwargs: Any) -> BatchTotals:  # noqa: ANN401
        on_result = kwargs["on_image_result"]
        for path in image_files:
            on_result(_rich_outcome(path))
        kwargs["on_complete"](totals)
        return totals

    return fake


def test_run_beacon_describes_the_batch_and_how_it_was_set_up(
    tmp_path: Path,
    beacons: list[object],
) -> None:
    """One finished batch sends one run beacon with its outcome, settings, and setup."""
    image = _make_jpeg(tmp_path / "img.cr3")
    totals = BatchTotals(total_files=1, success=1, workers=1, inference_calls=1)
    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=_finishing_run_batch(totals)),
        pytest.raises(SystemExit),
    ):
        main_module.main(["--input", str(image), "--temperature", "0.3", "--dry-run"])

    assert len(beacons) == 1, "a run beacon covers the invocation: no command beacon follows it"
    run = beacons[0]
    assert isinstance(run, telemetry.RunInfo)
    assert (run.command, run.job, run.outcome, run.batch_size) == ("tag", "tag", "ok", 1)
    assert run.settings is not None
    assert run.settings.temperature == 0.3  # noqa: PLR2004 - the flag's value
    assert run.settings.dry_run is True
    assert run.setup is not None
    assert run.setup.cli_flags == "--dry-run,--input,--temperature"
    assert {"dry_run", "temperature"} <= set(run.setup.changed.split(","))
    assert run.photo_seconds_median == 1.5  # noqa: PLR2004 - the outcome's model time
    assert run.keywords_mean == 2.0  # noqa: PLR2004 - the outcome's two keywords


def test_run_beacon_reports_a_batch_with_nothing_to_do(
    tmp_path: Path,
    beacons: list[object],
) -> None:
    """A rerun whose filters leave nothing to tag still says so, as an empty run."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "filter_batch", return_value=[]),
        pytest.raises(SystemExit),
    ):
        main_module.main(["--input", str(image)])

    assert [type(event) for event in beacons] == [telemetry.RunInfo]
    assert beacons[0].outcome == telemetry.OUTCOME_EMPTY  # type: ignore[attr-defined]


def test_run_beacon_reports_a_run_that_never_started(
    tmp_path: Path,
    beacons: list[object],
) -> None:
    """A model that cannot be reached stops the run before its first photo, and that is reported."""
    image = _make_jpeg(tmp_path / "img.cr3")
    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", side_effect=ProviderError("down")),
        pytest.raises(SystemExit),
    ):
        main_module.main(["--input", str(image)])

    assert len(beacons) == 1
    run = beacons[0]
    assert isinstance(run, telemetry.RunInfo)
    assert (run.outcome, run.error_kind) == (telemetry.OUTCOME_SETUP_FAILED, "ProviderError")


def test_a_telemetry_snapshot_that_raises_does_not_stop_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bug in building the beacon turns telemetry off for the run instead of a traceback."""
    image = _make_jpeg(tmp_path / "img.cr3")
    totals = BatchTotals(total_files=1, success=1, workers=1, inference_calls=1)
    enabled: list[object] = []
    monkeypatch.setattr(telemetry, "_sent_any", False)
    monkeypatch.setattr(
        telemetry,
        "emit",
        lambda _event, **kwargs: enabled.append(kwargs["enabled"]),
    )
    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=_finishing_run_batch(totals)),
        patch.object(cli_telemetry, "to_telemetry_settings", side_effect=ValueError("boom")),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.main(["--input", str(image)])

    assert exc_info.value.code == 0
    # The run beacon goes out disabled, and no command beacon stands in for it on the way out.
    assert enabled == [False]


def test_other_commands_send_a_command_beacon(
    tmp_path: Path,
    beacons: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command with no run of its own reports its name, outcome, and the flags it was given."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    with pytest.raises(SystemExit):
        main_module.main(["undo", "--list"])

    assert len(beacons) == 1
    command = beacons[0]
    assert isinstance(command, telemetry.CommandInfo)
    assert (command.command, command.outcome, command.exit_code) == ("undo", "ok", 0)
    assert command.setup is not None
    assert command.setup.cli_flags == "--list"


def test_command_beacon_names_a_usage_error(beacons: list[object]) -> None:
    """A mistyped flag is reported as a usage error, without the flag's text."""
    with pytest.raises(SystemExit):
        main_module.main(["--no-such-flag"])

    assert len(beacons) == 1
    command = beacons[0]
    assert isinstance(command, telemetry.CommandInfo)
    assert (command.command, command.outcome) == ("tag", telemetry.OUTCOME_USAGE_ERROR)


def test_command_beacon_names_help_and_version(beacons: list[object]) -> None:
    """Reading the help is its own kind of invocation, not a tag run."""
    with pytest.raises(SystemExit):
        main_module.main(["--help"])
    assert beacons[0].command == "help"  # type: ignore[attr-defined]


def test_command_beacon_respects_the_opt_out(
    tmp_path: Path,
    beacons: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PHOTO_TAGGER_TELEMETRY=0 silences the fallback beacon too."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PHOTO_TAGGER_TELEMETRY", "0")
    with pytest.raises(SystemExit):
        main_module.main(["undo", "--list"])
    assert beacons == []


def test_gui_command_leaves_reporting_to_the_window(
    beacons: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window reports itself under its own toggle, so the CLI sends nothing for `gui`."""

    def no_qt(_name: str) -> object:
        raise ImportError(name="PySide6")

    monkeypatch.setattr(importlib, "import_module", no_qt)
    with pytest.raises(SystemExit):
        main_module.main(["gui"])
    assert beacons == []


def test_doctor_notes_which_checks_failed(beacons: list[object]) -> None:
    """The doctor beacon carries fixed problem codes, never the details that name paths or URLs."""
    results = [
        diagnostics.CheckResult("ExifTool", ok=False, detail="/secret", problem="not-found"),
        diagnostics.CheckResult("Model", ok=True, detail="http://host:1234"),
    ]
    with (
        patch.object(main_module, "run_checks", return_value=results),
        patch.object(main_module, "render_report", return_value=False),
        pytest.raises(SystemExit),
    ):
        main_module.main(["doctor"])

    command = beacons[0]
    assert isinstance(command, telemetry.CommandInfo)
    assert command.details == {"checks_failed": 1, "problems": "not-found"}


def test_watch_reports_each_batch_and_then_the_session(
    tmp_path: Path,
    beacons: list[object],
) -> None:
    """A watch sends a run per batch, then one session covering the whole watch."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _settled_photo(inbox / "a.cr3")
    totals = BatchTotals(total_files=1, success=1)
    with (
        patch.object(cli_run, "setup_logging"),
        patch.object(cli_run, "create_agent", return_value=object()),
        patch.object(cli_run, "run_batch", side_effect=_finishing_run_batch(totals)),
        patch.object(cli_watch, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])

    assert [type(event) for event in beacons] == [telemetry.RunInfo, telemetry.SessionInfo]
    run, session = beacons
    assert (run.command, run.trigger) == ("watch", "watch")  # type: ignore[attr-defined]
    assert isinstance(session, telemetry.SessionInfo)
    assert (session.command, session.runs, session.photos_generated) == ("watch", 1, 1)
    assert session.settings is not None
    assert session.settings.watch_interval == WatchConfig().interval


def test_watch_session_counts_only_the_photos_that_came_out_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch of three with two failures adds one photo to the session, as the GUI counts it."""
    monkeypatch.setattr(telemetry, "emit", lambda *_args, **_kwargs: None)
    beacons = cli_telemetry.Beacons(
        enabled=False,
        command="watch",
        provider="lmstudio",
        model="m",
        output_language="English",
        settings=telemetry.Settings(),
        setup=telemetry.Setup(),
    )
    beacons.record(replace(beacons.base_run(), batch_size=3, success_count=1))

    assert beacons.photos == 1


def test_config_hook_records_the_command_and_the_typed_flags(tmp_path: Path) -> None:
    """The hook names what was typed, by option name, and never keeps a value."""
    main_module.app.parse_args(["watch", str(tmp_path), "--interval", "5", "-w", "2"])
    invocation = cli_options.CONFIG_SOURCE.invocation
    assert invocation.command == "watch"
    assert invocation.cli_flags == ("--input", "--interval", "--workers")


def test_a_typed_embed_in_photo_beats_a_saved_sidecar_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GUI saves a mode on every save; a shorthand typed on the command line still wins."""
    config = tmp_path / "config.toml"
    config.write_text('[output]\nsidecar_mode = "all"\n', encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))
    image = _make_jpeg(tmp_path / "img.jpg")
    seen: list[str] = []

    def fake_write(*_args: object, **kwargs: Any) -> tuple[bool, list[object]]:  # noqa: ANN401
        seen.append(kwargs["sidecar_mode"])
        return True, []

    with patch.object(cli_fields, "write_metadata_everywhere", side_effect=fake_write):
        _run_app(["set", "--input", str(image), "--embed-in-photo", "--artist", "Ada"])

    assert seen == ["none"]


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        # The sound implies the alert, even over a notify = false the GUI saved.
        (["--notify-sound"], (True, True)),
        # A typed --no-notify beats the saved sound.
        (["--no-notify"], (False, False)),
    ],
)
def test_a_typed_notify_flag_beats_the_saved_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    typed: list[str],
    expected: tuple[bool, bool],
) -> None:
    """What is typed wins over what the GUI saved, in both directions."""
    config = tmp_path / "config.toml"
    saved_off = "notify = false" if "--notify-sound" in typed else "notify_sound = true"
    config.write_text(f"[display]\n{saved_off}\n", encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(config))

    _command, bound, _ignored = main_module.app.parse_args(["--input", str(tmp_path), *typed])

    assert resolve_notify(bound.arguments["display"]) == expected


def test_config_hook_reports_an_alias_by_its_option_name(tmp_path: Path) -> None:
    """``--lang`` and ``--output-language`` are one option, and a negative keeps its "no"."""
    main_module.app.parse_args(["--input", str(tmp_path), "--lang", "German", "--no-backup-xmp"])
    invocation = cli_options.CONFIG_SOURCE.invocation
    assert invocation.cli_flags == ("--input", "--no-backup-xmp", "--output-language")


def _run_options(defaults: Defaults, **changes: Any) -> RunOptions:  # noqa: ANN401 - any field.
    """Build the RunOptions a run with *defaults* would parse, with *changes* on top."""
    options = RunOptions(
        provider=defaults.provider,
        inference=defaults.inference,
        output=defaults.output,
        artifacts=defaults.artifacts,
        display=defaults.display,
        filter_=defaults.filter,
        log=defaults.log,
        pacing=defaults.pacing,
        telemetry_enabled=defaults.telemetry.enabled,
        workers=defaults.workers,
        extensions=defaults.extensions,
        recursive=defaults.recursive,
    )
    return replace(options, **changes)


def test_telemetry_settings_of_the_defaults_are_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Built-in option groups map onto the built-in Settings, so nothing reads as changed."""
    for name in ("LM_STUDIO_API_KEY", "OPENAI_API_KEY", "LM_STUDIO_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    defaults = load_defaults({})
    settings = to_telemetry_settings(_run_options(defaults))
    assert settings == telemetry.Settings()


def test_telemetry_settings_count_the_sidecar_overrides() -> None:
    """
    --sidecar-override is reported as how many file types it sends elsewhere, never which.

    File types, not entries: a later entry for the same type replaces the earlier one, as the GUI
    counts them.
    """
    overrides = ["dng=photo", "tif=both", "DNG=sidecar"]
    defaults = load_defaults({"output": {"sidecar_overrides": overrides}})
    settings = to_telemetry_settings(_run_options(defaults))
    assert settings.sidecar_overrides == 2  # noqa: PLR2004 - the two entries above.


def test_telemetry_settings_keep_no_path_or_text(tmp_path: Path) -> None:
    """A hint, a copyright notice, and a remote URL come out as switches and categories."""
    defaults = load_defaults(
        {
            "provider": {"api_base_url": "https://gpu.example.com/v1", "provider_name": "openai"},
            "inference": {"hint": "The animal is a deer"},
            "copyright": {"artist": "Ada", "copyright_notice": "(c) {year} Ada", "marked": True},
        },
    )
    options = _run_options(defaults, workers=1, extensions="CR3,jpg,weird", recursive=True)
    settings = to_telemetry_settings(options)
    assert (settings.endpoint, settings.secure, settings.custom_url) == ("public", True, True)
    assert (settings.hint, settings.hint_chars) == (True, len("The animal is a deer"))
    assert (settings.artist, settings.copyright_notice, settings.marked) == (
        True,
        True,
        "copyrighted",
    )
    assert settings.extensions == "cr3,jpg,other"
    assert "Ada" not in repr(settings)
    assert "example.com" not in repr(settings)


def test_known_config_keys_cover_every_table_the_config_reads() -> None:
    """Every table and top-level key photo-tagger reads can be named in a beacon."""
    known = known_config_keys()
    assert "model_name" in known["provider"]
    assert "save_after_generating" in known["gui"]
    assert {"extensions", "workers", "recursive", "language", "exiftool_path"} <= known[""]


def test_a_gui_crash_is_left_to_the_window_to_report(beacons: list[object]) -> None:
    """The window reports its own crashes under its own consent, so the CLI does not add one."""
    crashes: list[object] = []
    app = MagicMock(side_effect=RuntimeError("boom"))
    app.__iter__.return_value = iter(["gui", "set", "show"])
    with (
        patch.object(telemetry, "emit_crash", side_effect=lambda exc, **_k: crashes.append(exc)),
        patch.object(main_module, "app", app),
        pytest.raises(RuntimeError),
    ):
        main_module.main(["gui"])

    assert crashes == []
