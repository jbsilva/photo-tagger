"""Tests for the report sinks of a CLI run: NDJSON, the CSV report, the summary, the alert."""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from photo_tagger import cli_reports
from photo_tagger.pipeline import BatchTotals, ImageOutcome


if TYPE_CHECKING:
    from pathlib import Path


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
    emitter = cli_reports.NDJSONEmitter(buf)
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
        """Match the TextSink protocol; nothing to do."""


def test_ndjson_emitter_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent emitters never interleave a partial line, even on a non-atomic sink."""
    sink = _ChunkedSink()
    emitter = cli_reports.NDJSONEmitter(sink)
    paths = [tmp_path / f"img{i:03d}.cr3" for i in range(60)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: emitter(_outcome(p)), paths))

    lines = "".join(sink.chunks).splitlines()
    # Each line parses cleanly: proof that nothing interleaved.
    decoded = [json.loads(line) for line in lines]
    assert sorted(d["file"] for d in decoded) == sorted(str(p) for p in paths)


def test_outcome_to_report_row_maps_fields(tmp_path: Path) -> None:
    """An ImageOutcome maps onto the report row, EXIF dicts resolved to scalar columns."""
    rendered = cli_reports.outcome_to_report_row(_rich_outcome(tmp_path / "img.cr3")).as_dict()
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

    assert cli_reports.combine_image_result_callbacks(None, None) is None
    assert cli_reports.combine_image_result_callbacks(None, first) is first

    combined = cli_reports.combine_image_result_callbacks(first, second)
    assert combined is not None
    outcome = _rich_outcome(tmp_path / "img.cr3")
    combined(outcome)
    assert seen == [("first", outcome), ("second", outcome)]


def test_open_csv_report_returns_none_for_none_path() -> None:
    """No --csv-file means no writer, so nothing to open or close."""
    assert cli_reports.open_csv_report(None) is None


def test_open_csv_report_degrades_on_open_error(tmp_path: Path) -> None:
    """A writer that cannot open is logged and skipped, never raised."""
    with patch.object(cli_reports, "CsvReportWriter", side_effect=OSError("denied")):
        assert cli_reports.open_csv_report(tmp_path / "report.csv") is None


def test_write_summary_file_noop_when_path_is_none(tmp_path: Path) -> None:
    """summary_file=None is the normal no-write path."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Should not raise or create any file.
    cli_reports.write_summary_file(
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
    cli_reports.write_summary_file(
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
    with patch.object(cli_reports, "atomic_write_text", side_effect=OSError("boom")):
        # Must not raise.
        cli_reports.write_summary_file(
            dest,
            BatchTotals(),
            started_at=datetime.now(tz=UTC),
            model_name="m",
            provider_name="p",
            user_prompt_chars=0,
        )


def test_atomic_write_text_cleans_up_on_write_failure(tmp_path: Path) -> None:
    """If the write to the temp file fails, the temp file is removed and the error re-raised."""
    target = tmp_path / "output.json"

    # Patch os.fdopen to raise after mkstemp creates the temp file.
    with (
        patch("photo_tagger.cli_reports.os.fdopen", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        cli_reports.atomic_write_text(target, "content")

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
        patch("photo_tagger.cli_reports.Path.replace", side_effect=OSError("read-only filesystem")),
        pytest.raises(OSError, match="read-only"),
    ):
        cli_reports.atomic_write_text(target, "content")

    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.*")) == []


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
        (
            BatchTotals(total_files=5, success=2, failed_files=["c", "d", "e"], interrupted=True),
            "Stopped: 2 of 5 photos tagged.",
        ),
    ],
)
def test_alert_message_wording(totals: BatchTotals, expected: str) -> None:
    """The alert says what happened in one glanceable line, singular included."""
    assert cli_reports.alert_message(totals) == expected
