"""
End-to-end wiring tests for the cyclopts CLI in photo_tagger.main.

The agent (network calls) and pipeline (long-running work) are mocked; the goal is to prove that
flag values reach the right collaborators with the right shape, and that the short-circuit / skip
code paths run end to end without raising. Real work is exercised elsewhere by the per-module unit
tests.
"""

import contextlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from photo_tagger import (
    main as main_module,
    telemetry,
)
from photo_tagger.cli_options import load_defaults
from photo_tagger.errors import BatchError, ProviderError
from photo_tagger.pipeline import BatchTotals, ImageOutcome
from photo_tagger.undo import UndoError, UndoJournal
from photo_tagger.vocabulary_build import KeywordCensus, TrimResult
from photo_tagger.vocabulary_organize import OrganizeStats


if TYPE_CHECKING:
    from pathlib import Path


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
        return None

    return (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=fake_run_batch),
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


def test_ndjson_emitter_writes_one_line_per_outcome(tmp_path: Path) -> None:
    """Each call to the emitter writes exactly one JSON line that round-trips through json.loads."""
    buf = io.StringIO()
    emitter = main_module._NDJSONEmitter(buf)  # noqa: SLF001
    emitter(_outcome(tmp_path / "a.cr3", success=True, from_cache=False))
    emitter(_outcome(tmp_path / "b.cr3", success=False, from_cache=True))

    lines = buf.getvalue().splitlines()
    expected_lines = 2
    assert len(lines) == expected_lines
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["status"] == "ok"
    assert first["from_cache"] is False
    assert first["title"] == "A Title"
    assert first["keywords"] == ["Beach", "Sunset"]
    assert second["status"] == "failed"
    assert second["from_cache"] is True


class _ChunkedSink:
    """
    A stream whose write lands character by character, yielding the GIL between characters.

    io.StringIO.write of a whole line is atomic under the GIL, so a test against it can never
    interleave and would keep passing even with the emitter's lock deleted. This sink models a
    buffered real stdout, where an unlocked concurrent write genuinely shreds lines.
    """

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, s: str) -> int:
        for char in s:
            self.chunks.append(char)
            time.sleep(0)  # invite the scheduler to interleave another writer
        return len(s)

    def flush(self) -> None:
        """Match the _TextSink protocol; nothing to do."""


def test_ndjson_emitter_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent emitters never interleave a partial line, even on a non-atomic sink."""
    sink = _ChunkedSink()
    emitter = main_module._NDJSONEmitter(sink)  # noqa: SLF001
    paths = [tmp_path / f"img{i:03d}.cr3" for i in range(60)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: emitter(_outcome(p)), paths))

    lines = "".join(sink.chunks).splitlines()
    # Each line parses cleanly: proof that nothing interleaved.
    decoded = [json.loads(line) for line in lines]
    assert sorted(d["file"] for d in decoded) == sorted(str(p) for p in paths)


def test_cli_json_flag_installs_ndjson_emitter(tmp_path: Path) -> None:
    """--json wires an _NDJSONEmitter onto run_batch's on_image_result callback."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--json"])

    emitter = captured["on_image_result"]
    assert isinstance(emitter, main_module._NDJSONEmitter)  # noqa: SLF001


def test_cli_default_does_not_install_ndjson_emitter(tmp_path: Path) -> None:
    """Without --json the on_image_result callback stays None."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["on_image_result"] is None


# ---------------------------------------------------------------------------
# --csv-file report
# ---------------------------------------------------------------------------


def test_outcome_to_report_row_maps_fields(tmp_path: Path) -> None:
    """An ImageOutcome maps onto the report row, EXIF dicts resolved to scalar columns."""
    rendered = main_module._outcome_to_report_row(_rich_outcome(tmp_path / "img.cr3")).as_dict()  # noqa: SLF001
    assert rendered["filename"] == "img.cr3"
    assert rendered["status"] == "ok"
    assert rendered["title"] == "A Title"
    assert rendered["keywords"] == "Beach; Sunset"
    assert rendered["hierarchical_keywords"] == "Nature|Beach"
    assert rendered["existing_keywords"] == "Old"
    assert rendered["camera_model"] == "Canon EOS R5"
    assert rendered["lens_model"] == "RF 100mm"
    assert rendered["gps_position"] == "53 N, 9 E"
    assert rendered["city"] == "Hamburg"
    assert rendered["country"] == "Germany"
    assert rendered["from_cache"] == "false"


def test_combine_image_result_callbacks_handles_zero_one_and_many(tmp_path: Path) -> None:
    """Combining returns None for no sinks, the lone sink for one, and a fan-out for many."""
    seen: list[tuple[str, ImageOutcome]] = []

    def first(outcome: ImageOutcome) -> None:
        seen.append(("first", outcome))

    def second(outcome: ImageOutcome) -> None:
        seen.append(("second", outcome))

    assert main_module._combine_image_result_callbacks(None, None) is None  # noqa: SLF001
    assert main_module._combine_image_result_callbacks(None, first) is first  # noqa: SLF001

    combined = main_module._combine_image_result_callbacks(first, second)  # noqa: SLF001
    assert combined is not None
    outcome = _rich_outcome(tmp_path / "img.cr3")
    combined(outcome)
    assert seen == [("first", outcome), ("second", outcome)]


def test_open_csv_report_returns_none_for_none_path() -> None:
    """No --csv-file means no writer, so nothing to open or close."""
    assert main_module._open_csv_report(None) is None  # noqa: SLF001


def test_open_csv_report_degrades_on_open_error(tmp_path: Path) -> None:
    """A writer that cannot open is logged and skipped, never raised."""
    with patch.object(main_module, "CsvReportWriter", side_effect=OSError("denied")):
        assert main_module._open_csv_report(tmp_path / "report.csv") is None  # noqa: SLF001


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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=fake_run_batch),
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
    """--csv-file alone wires a _CsvImageResultSink and emits the header even with no rows."""
    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--csv-file", str(csv_path)])

    assert isinstance(captured["on_image_result"], main_module._CsvImageResultSink)  # noqa: SLF001
    assert csv_path.exists()


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
    assert not isinstance(callback, main_module._NDJSONEmitter)  # noqa: SLF001
    assert not isinstance(callback, main_module._CsvImageResultSink)  # noqa: SLF001


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

    parsed = main_module._parse_filter_date("2024-01-01T00:00:00", flag="--newer-than")  # noqa: SLF001
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
    parsed = main_module._parse_filter_date("2024-01-01T00:00:00+00:00", flag="--newer-than")  # noqa: SLF001
    assert parsed is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_parse_filter_date_returns_none_for_none() -> None:
    """Passing None short-circuits without raising."""
    assert main_module._parse_filter_date(None, flag="--newer-than") is None  # noqa: SLF001


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
        patch.object(main_module, "open_cache", spying_open_cache),
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
        patch.object(main_module, "CsvReportWriter", spying_writer),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "open_cache", return_value=cache),
        patch.object(main_module, "open_journal", side_effect=OSError("no state dir")),
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


def test_cli_log_setup_failure_exits_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    An OSError from setup_logging (e.g. an unwritable --log-folder) exits 1 with a clean message.

    Regression test: this used to propagate as a raw OSError traceback instead of the same clean
    "log and exit 1" pattern every other pre-flight failure in this command already follows.
    Cannot rely on the logger for the message: setup_logging removes the default sink before it
    can fail, so nothing is guaranteed to be listening.
    """
    image = _make_jpeg(tmp_path / "img.cr3")

    with (
        patch.object(main_module, "setup_logging", side_effect=OSError("disk full")),
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
        patch.object(main_module, "create_agent") as agent_mock,
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
        patch.object(main_module, "FileLock", side_effect=OSError("disk full")),
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
# _read_prompt_file edge cases
# ---------------------------------------------------------------------------


def test_read_prompt_file_exits_on_empty_file(tmp_path: Path) -> None:
    """A prompt file that contains only whitespace is rejected with SystemExit."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(SystemExit):
        main_module._read_prompt_file(prompt)  # noqa: SLF001


def test_read_prompt_file_exits_on_read_error(tmp_path: Path) -> None:
    """An unreadable prompt file triggers SystemExit."""
    prompt = tmp_path / "missing.txt"
    with pytest.raises(SystemExit):
        main_module._read_prompt_file(prompt)  # noqa: SLF001


def test_read_prompt_file_returns_default_when_none() -> None:
    """Passing None returns the built-in default prompt."""
    from photo_tagger.config import DEFAULT_USER_PROMPT  # noqa: PLC0415

    assert main_module._read_prompt_file(None) == DEFAULT_USER_PROMPT  # noqa: SLF001


# ---------------------------------------------------------------------------
# _write_summary_file edge cases
# ---------------------------------------------------------------------------


def test_write_summary_file_noop_when_path_is_none(tmp_path: Path) -> None:
    """summary_file=None is the normal no-write path."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Should not raise or create any file.
    main_module._write_summary_file(  # noqa: SLF001
        None,
        BatchTotals(),
        started_at=datetime.now(tz=UTC),
        model_name="m",
        provider_name="p",
        user_prompt_chars=0,
    )


def test_write_summary_file_noop_when_totals_is_none(tmp_path: Path) -> None:
    """``totals=None`` is the early-exit path when the batch never ran."""
    from datetime import UTC, datetime  # noqa: PLC0415

    dest = tmp_path / "out.json"
    main_module._write_summary_file(  # noqa: SLF001
        dest,
        None,
        started_at=datetime.now(tz=UTC),
        model_name="m",
        provider_name="p",
        user_prompt_chars=0,
    )
    assert not dest.exists()


def test_write_summary_file_swallows_write_error(tmp_path: Path) -> None:
    """An OSError during write is logged, not raised."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Point at a directory path so write fails.
    dest = tmp_path / "dir_not_file"
    dest.mkdir()
    dest = dest / "nested" / "summary.json"
    with patch.object(main_module, "_atomic_write_text", side_effect=OSError("boom")):
        # Must not raise.
        main_module._write_summary_file(  # noqa: SLF001
            dest,
            BatchTotals(),
            started_at=datetime.now(tz=UTC),
            model_name="m",
            provider_name="p",
            user_prompt_chars=0,
        )


# ---------------------------------------------------------------------------
# _atomic_write_text edge cases
# ---------------------------------------------------------------------------


def test_atomic_write_text_cleans_up_on_write_failure(tmp_path: Path) -> None:
    """If the write to the temp file fails, the temp file is removed and the error re-raised."""
    target = tmp_path / "output.json"

    # Patch os.fdopen to raise after mkstemp creates the temp file.
    with (
        patch("photo_tagger.main.os.fdopen", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        main_module._atomic_write_text(target, "content")  # noqa: SLF001

    # Target must not exist, and no temp files should be left behind.
    assert not target.exists()
    leftovers = list(tmp_path.glob(f".{target.name}.*"))
    assert leftovers == []


def test_atomic_write_text_cleans_up_when_rename_fails(tmp_path: Path) -> None:
    """
    A failure after the temp file was fully written (the rename) also removes it.

    Unlike the fdopen failure above, the descriptor is already closed here, so this exercises the
    branch that skips the extra close before unlinking.
    """
    target = tmp_path / "output.json"

    with (
        patch("photo_tagger.main.Path.replace", side_effect=OSError("read-only filesystem")),
        pytest.raises(OSError, match="read-only"),
    ):
        main_module._atomic_write_text(target, "content")  # noqa: SLF001

    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.*")) == []


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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=_run_batch_firing_complete),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=_run_batch_firing_complete),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=fake_run_batch),
        patch.object(main_module, "notify", side_effect=fake_notify),
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


@pytest.mark.parametrize(
    ("totals", "expected"),
    [
        (BatchTotals(total_files=1, success=1), "1 of 1 photo tagged."),
        (BatchTotals(total_files=9, success=9), "9 of 9 photos tagged."),
        (
            BatchTotals(total_files=2, success=0, failed_files=["a.cr3", "b.cr3"]),
            "0 of 2 photos tagged, 2 failed.",
        ),
        (
            BatchTotals(total_files=3, success=3, dry_run=True),
            "Dry run: 3 of 3 photos checked.",
        ),
    ],
)
def test_alert_message_wording(totals: BatchTotals, expected: str) -> None:
    """The alert says what happened in one glanceable line, singular included."""
    assert main_module._alert_message(totals) == expected  # noqa: SLF001 - module-private helper.


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

    Regression test: cyclopts drops every flat-config key that names a subcommand before matching
    it against anything, so this key was silently ignored the moment the command was added, while
    the keys next to it in the same table kept working.
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
        (False, False, (False, False)),
        (True, False, (True, False)),
        (True, True, (True, True)),
        # --notify-sound alone: the sound is how the notification announces itself.
        (False, True, (True, True)),
    ],
)
def test_resolve_notify_precedence(
    expected: tuple[bool, bool],
    *,
    notify_flag: bool,
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=RuntimeError("boom")),
        patch.object(telemetry, "emit_crash", side_effect=fake_emit_crash),
        pytest.raises(RuntimeError, match="boom"),
    ):
        main_module.main(["--input", str(image)])

    assert len(crashes) == 1
    assert isinstance(crashes[0]["exc"], RuntimeError)
    assert crashes[0]["interface"] == "cli"


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


def test_crash_telemetry_enabled_honors_argv_flag() -> None:
    """A --no-telemetry anywhere on the command line disables the crash beacon too."""
    assert main_module._crash_telemetry_enabled(["-i", "x", "--no-telemetry"]) is False  # noqa: SLF001
    assert main_module._crash_telemetry_enabled(["-i", "x"]) is True  # noqa: SLF001


def test_crash_telemetry_enabled_honors_the_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[telemetry] enabled = false in the config disables crash beacons; garbage tables do not."""
    _point_config_at(tmp_path, monkeypatch, "[telemetry]\nenabled = false\n")
    assert main_module._crash_telemetry_enabled([]) is False  # noqa: SLF001

    _point_config_at(tmp_path, monkeypatch, 'telemetry = "not-a-table"\n')
    assert main_module._crash_telemetry_enabled([]) is True  # noqa: SLF001


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


def test_apply_exiftool_path_bridges_to_env_without_overriding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill an unset env var from config; never override an exported one; no-op on None."""
    import os  # noqa: PLC0415 - test-local

    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    main_module._apply_exiftool_path("/cfg/exiftool")  # noqa: SLF001
    assert os.environ["PHOTO_TAGGER_EXIFTOOL"] == "/cfg/exiftool"

    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "/env/exiftool")
    main_module._apply_exiftool_path("/cfg/exiftool")  # noqa: SLF001
    assert os.environ["PHOTO_TAGGER_EXIFTOOL"] == "/env/exiftool"

    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    main_module._apply_exiftool_path(None)  # noqa: SLF001
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

    with patch.object(main_module, "census_from_photos", return_value=census) as counted:
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
        patch.object(main_module, "census_from_photos", return_value=KeywordCensus()),
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

    with patch.object(main_module, "organize", return_value=(organized, stats)) as organizer:
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
        patch.object(main_module, "organize", side_effect=ProviderError("no provider")),
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
        patch("photo_tagger.main.read_journal", side_effect=UndoError("boom")),
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
        patch.object(main_module, "run_batch", side_effect=recording_run_batch),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "_process_batch", side_effect=fake_process),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "_process_batch", side_effect=fake_process),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "_process_batch", side_effect=fake_process),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "_process_batch", side_effect=failing_process),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])

    expected_batches = 2
    assert len(calls) == expected_batches


def test_watch_applies_the_batch_filters(tmp_path: Path) -> None:
    """--skip-tagged and friends filter each batch, and an empty one is not processed."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    photo = _settled_photo(inbox / "a.cr3")
    calls: list[list[Path]] = []

    with (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(
            main_module,
            "_process_batch",
            side_effect=lambda files, _s: calls.append(
                list(files),
            ),
        ),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
        patch("photo_tagger.discovery.find_tagged_images", return_value={photo}),
    ):
        _run_app(["watch", "--input", str(inbox), "--skip-tagged"])

    assert calls == []


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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=run_one_batch),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
        patch.object(
            main_module,
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "_process_batch", side_effect=interrupt),
        patch.object(main_module, "watch_batches", side_effect=_bounded_watch),
    ):
        _run_app(["watch", "--input", str(inbox)])


def test_watch_requires_an_input(tmp_path: Path) -> None:
    """Watching nothing is a clean exit 1, not an infinite loop over an empty list."""
    with (
        patch.object(main_module, "setup_logging"),
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
        patch.object(main_module, "setup_logging", side_effect=OSError("read-only")),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
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
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "watch_batches", side_effect=fake_watch),
    ):
        _run_app(["watch", "--input", str(inbox), "--interval", "11", "--settle", "3"])

    assert seen == {"interval": 11.0, "settle": 3.0}
