"""
Build and write a per-photo CSV report of generated and extracted metadata.

The report is the tabular sibling of the JSON run summary (``--summary-file``) and the per-photo
NDJSON stream (``--json``): one row per photo carrying the generated title, description, and
keywords, the metadata already on the file, the camera/location EXIF read as context, and the token
usage and timing of each call.

Both frontends funnel through the same :class:`ReportRow` and the single :data:`CSV_FIELDNAMES`
column order, so the CLI and the GUI emit an identical schema. They differ only in *when* they
write: the CLI streams a row as each photo finishes (:class:`CsvReportWriter`), while the GUI has
every row in hand and writes them at once on an Export action (:func:`write_report`).
"""

import csv
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


# Joins the values of a multi-value cell (the keyword lists). A semicolon, not a comma, so the
# cell stays readable on its own without colliding with the CSV field separator.
_LIST_SEP = "; "

# Leading characters that Excel, LibreOffice, and Google Sheets interpret as a formula
# introducer. A filename or a model-generated keyword can start with any of these; without
# neutralizing them, opening the report spreadsheet executes attacker- or model-supplied
# content instead of displaying it as text (CSV/formula injection, CWE-1236).
_FORMULA_TRIGGER_CHARS = ("=", "+", "-", "@", "\t", "\r")


def neutralize_formula(value: str) -> str:
    """
    Prefix *value* with a single quote if a spreadsheet would read it as a formula.

    Public because every CSV this project writes owes the reader the same protection, and the
    keyword cells all carry content read off photos or typed by a model. The vocabulary builder's
    drop report uses it too.

    Examples:
        >>> neutralize_formula("=1+1")
        "'=1+1"
        >>> neutralize_formula("Osprey")
        'Osprey'
    """
    if value.startswith(_FORMULA_TRIGGER_CHARS):
        return f"'{value}"
    return value


def _format_bool(*, value: bool | None) -> str:
    """Render a tri-state flag: ``true``/``false`` when known, blank when not applicable."""
    if value is None:
        return ""
    return "true" if value else "false"


@dataclass(slots=True, frozen=True)
class ReportRow:
    """
    One photo's row in the CSV report, with every field rendering to a single cell.

    Defaults are empty/zero so each frontend can fill only the columns it knows: the CLI leaves
    ``existing_title``/``existing_description`` blank, the GUI leaves ``from_cache``/``retry`` as
    ``None`` (it has no cache or retry pass), and a photo that failed before inference still
    produces a valid row with whatever was read off the file.
    """

    file: str = ""
    filename: str = ""
    status: str = ""
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    hierarchical_keywords: list[str] = field(default_factory=list)
    existing_keywords: list[str] = field(default_factory=list)
    existing_title: str = ""
    existing_description: str = ""
    camera_model: str = ""
    lens_model: str = ""
    capture_date: str = ""
    gps_position: str = ""
    city: str = ""
    country: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
    from_cache: bool | None = None
    retry: bool | None = None
    # Non-empty when the generated text mixes writing systems: the bits in the odd script out. A
    # column rather than a bare flag, so the cell says what to look at without opening the photo.
    foreign_terms: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict[str, str]:
        """Render this row as a ``column -> cell`` mapping ready for :class:`csv.DictWriter`."""
        return {
            "filename": neutralize_formula(self.filename),
            "file": neutralize_formula(self.file),
            "status": self.status,
            "title": neutralize_formula(self.title),
            "description": neutralize_formula(self.description),
            "keywords": neutralize_formula(_LIST_SEP.join(self.keywords)),
            "hierarchical_keywords": neutralize_formula(
                _LIST_SEP.join(self.hierarchical_keywords),
            ),
            "existing_keywords": neutralize_formula(_LIST_SEP.join(self.existing_keywords)),
            "existing_title": neutralize_formula(self.existing_title),
            "existing_description": neutralize_formula(self.existing_description),
            "camera_model": neutralize_formula(self.camera_model),
            "lens_model": neutralize_formula(self.lens_model),
            "capture_date": neutralize_formula(self.capture_date),
            "gps_position": neutralize_formula(self.gps_position),
            "city": neutralize_formula(self.city),
            "country": neutralize_formula(self.country),
            "input_tokens": str(self.input_tokens),
            "output_tokens": str(self.output_tokens),
            "total_tokens": str(self.total_tokens),
            "seconds": f"{self.seconds:.3f}",
            "from_cache": _format_bool(value=self.from_cache),
            "retry": _format_bool(value=self.retry),
            "foreign_terms": neutralize_formula(_LIST_SEP.join(self.foreign_terms)),
            "error": neutralize_formula(self.error),
        }


# The CSV header, derived from a default row so the columns and their order can never drift from
# what :meth:`ReportRow.as_dict` emits: a new field missing from ``as_dict`` would surface as a
# DictWriter mismatch, and a renamed key changes both at once.
CSV_FIELDNAMES: list[str] = list(ReportRow().as_dict().keys())


class CsvReportWriter:
    """
    Thread-safe ``on_image_result`` sink that streams report rows to a CSV file.

    Opens the file and writes the header on construction, appends one row per :meth:`write` call
    under a lock (workers > 1 call this concurrently), and flushes each row so a run that is killed
    or interrupted still leaves a complete, readable CSV. Call :meth:`close` when done.
    """

    __slots__ = ("_fh", "_lock", "_writer")

    def __init__(self, path: Path) -> None:
        """Open *path* for writing (creating parent dirs) and emit the header row."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" per the csv module docs, so the writer controls line endings itself.
        # utf-8-sig: without the BOM, double-clicking the file in Excel decodes it as legacy
        # ANSI and mangles any non-ASCII title or keyword. Python's csv readers strip the BOM.
        self._fh = path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDNAMES)
        self._writer.writeheader()
        self._fh.flush()
        self._lock = Lock()

    def write(self, row: ReportRow) -> None:
        """Append *row* to the report and flush so the file is valid mid-run."""
        rendered = row.as_dict()
        with self._lock:
            self._writer.writerow(rendered)
            self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        with self._lock:
            self._fh.close()


def write_report(path: Path, rows: Iterable[ReportRow]) -> None:
    """
    Write *rows* to *path* as a CSV with a header, creating parent dirs as needed.

    Used by the GUI's Export action, which holds every row at once. Raises ``OSError`` on a write
    failure so the caller can surface it; the CLI streams instead via :class:`CsvReportWriter`.
    """
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig for Excel; see CsvReportWriter.
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.as_dict())
    logger.info("csv_report_written", file=str(path), rows=len(materialized))
