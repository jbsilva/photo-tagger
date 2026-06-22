"""Tests for the per-photo CSV report rows and writers."""

import csv
from pathlib import Path

from photo_tagger.csv_report import (
    CSV_FIELDNAMES,
    CsvReportWriter,
    ReportRow,
    write_report,
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(header, rows)`` parsed back from a CSV file."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def test_fieldnames_match_as_dict_keys() -> None:
    """The header columns are exactly the keys ReportRow.as_dict emits, in order."""
    assert list(ReportRow().as_dict().keys()) == CSV_FIELDNAMES


def test_as_dict_joins_lists_with_semicolons() -> None:
    """Multi-value cells (keyword lists) join with '; ' so commas inside stay unambiguous."""
    row = ReportRow(
        keywords=["Beach", "Sunset"],
        hierarchical_keywords=["Nature|Beach", "Sky|Sunset"],
        existing_keywords=["Old"],
    )
    rendered = row.as_dict()
    assert rendered["keywords"] == "Beach; Sunset"
    assert rendered["hierarchical_keywords"] == "Nature|Beach; Sky|Sunset"
    assert rendered["existing_keywords"] == "Old"


def test_as_dict_renders_tristate_bool_and_numbers() -> None:
    """from_cache/retry are true/false when set and blank when None; numbers stringify."""
    known = ReportRow(from_cache=True, retry=False, input_tokens=5, seconds=1.5).as_dict()
    assert known["from_cache"] == "true"
    assert known["retry"] == "false"
    assert known["input_tokens"] == "5"
    assert known["seconds"] == "1.500"

    blank = ReportRow().as_dict()
    assert blank["from_cache"] == ""
    assert blank["retry"] == ""
    assert blank["seconds"] == "0.000"


def test_csv_report_writer_streams_header_and_rows(tmp_path: Path) -> None:
    """The streaming writer emits a header once, then one parseable row per write."""
    target = tmp_path / "report.csv"
    writer = CsvReportWriter(target)
    writer.write(ReportRow(filename="a.cr3", title="First", keywords=["X"], from_cache=True))
    writer.write(ReportRow(filename="b.cr3", title="Second", status="failed"))
    writer.close()

    header, rows = _read_csv(target)
    assert header == CSV_FIELDNAMES
    assert [r["filename"] for r in rows] == ["a.cr3", "b.cr3"]
    assert rows[0]["title"] == "First"
    assert rows[0]["keywords"] == "X"
    assert rows[0]["from_cache"] == "true"
    assert rows[1]["status"] == "failed"


def test_csv_report_writer_flushes_each_row(tmp_path: Path) -> None:
    """A row is on disk before close, so an interrupted run still leaves a valid file."""
    target = tmp_path / "report.csv"
    writer = CsvReportWriter(target)
    try:
        writer.write(ReportRow(filename="a.cr3"))
        _, rows = _read_csv(target)
        assert [r["filename"] for r in rows] == ["a.cr3"]
    finally:
        writer.close()


def test_csv_report_writer_creates_missing_parent(tmp_path: Path) -> None:
    """A path into a not-yet-existing folder is created transparently."""
    target = tmp_path / "nested" / "deep" / "report.csv"
    writer = CsvReportWriter(target)
    writer.close()
    assert target.exists()


def test_write_report_writes_all_rows_at_once(tmp_path: Path) -> None:
    """The batch writer (GUI path) writes a header plus every supplied row."""
    target = tmp_path / "out" / "report.csv"
    rows = [
        ReportRow(filename="a.cr3", title="A", city="Hamburg", country="Germany"),
        ReportRow(filename="b.cr3", title="B"),
    ]
    write_report(target, rows)

    header, parsed = _read_csv(target)
    assert header == CSV_FIELDNAMES
    assert [r["filename"] for r in parsed] == ["a.cr3", "b.cr3"]
    assert parsed[0]["city"] == "Hamburg"
    assert parsed[0]["country"] == "Germany"


def test_write_report_header_only_for_empty_rows(tmp_path: Path) -> None:
    """No photos still yields a valid, header-only CSV rather than an empty file."""
    target = tmp_path / "report.csv"
    write_report(target, [])
    header, parsed = _read_csv(target)
    assert header == CSV_FIELDNAMES
    assert parsed == []
