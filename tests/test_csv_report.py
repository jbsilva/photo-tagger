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
    # utf-8-sig strips the Excel-compatibility BOM the writers emit (and is a no-op without one).
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def test_report_starts_with_excel_bom(tmp_path: Path) -> None:
    """
    Both writers emit a UTF-8 BOM so Excel decodes non-ASCII content correctly.

    Without it, double-clicking the export in Excel reads legacy ANSI and garbles any accented title
    or keyword.
    """
    streamed = tmp_path / "streamed.csv"
    writer = CsvReportWriter(streamed)
    writer.close()
    assert streamed.read_bytes().startswith(b"\xef\xbb\xbf")

    exported = tmp_path / "exported.csv"
    write_report(exported, [ReportRow(filename="café.jpg")])
    assert exported.read_bytes().startswith(b"\xef\xbb\xbf")
    _, rows = _read_csv(exported)
    assert rows[0]["filename"] == "café.jpg"


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


def test_as_dict_reports_the_mixed_script_terms() -> None:
    """The review column names what to look at, and is blank for the photos with nothing wrong."""
    flagged = ReportRow(foreign_terms=["陶笛", "居酒屋"]).as_dict()
    assert flagged["foreign_terms"] == "陶笛; 居酒屋"
    assert ReportRow().as_dict()["foreign_terms"] == ""


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


def test_as_dict_neutralizes_formula_trigger_characters() -> None:
    """
    A cell starting with '=', '+', '-', or '@' gets a leading quote so it renders as text.

    Otherwise a spreadsheet reads it as a formula (CSV/formula injection, CWE-1236). filename is
    attacker-controlled (it is whatever the photo on disk is named); title, description, and
    keywords are model-generated text the tool does not fully control either.
    """
    row = ReportRow(
        filename="=1+1+cmd|'/bin/calc'!A0.jpg",
        title="+SUM(1,1)",
        description="-2+3",
        keywords=["@mention", "Sunset"],
        error='=HYPERLINK("http://evil")',
    )
    rendered = row.as_dict()
    assert rendered["filename"] == "'=1+1+cmd|'/bin/calc'!A0.jpg"
    assert rendered["title"] == "'+SUM(1,1)"
    assert rendered["description"] == "'-2+3"
    assert rendered["keywords"] == "'@mention; Sunset"
    assert rendered["error"] == '\'=HYPERLINK("http://evil")'


def test_as_dict_leaves_ordinary_text_unchanged() -> None:
    """Cells that do not start with a formula trigger character are rendered as-is."""
    row = ReportRow(filename="IMG_0001.CR3", title="Sunset over the bay")
    rendered = row.as_dict()
    assert rendered["filename"] == "IMG_0001.CR3"
    assert rendered["title"] == "Sunset over the bay"


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
