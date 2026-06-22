"""
Build and write a per-photo CSV report of generated and extracted metadata.

The report is the tabular sibling of the JSON run summary (``--summary-file``) and the per-photo
NDJSON stream (``--json``): one row per photo carrying the generated title, description, and
keywords, the metadata already on the file, the camera/location EXIF read as context, and the per-
call token usage and timing.

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
    error: str = ""

    def as_dict(self) -> dict[str, str]:
        """Render this row as a ``column -> cell`` mapping ready for :class:`csv.DictWriter`."""
        return {
            "filename": self.filename,
            "file": self.file,
            "status": self.status,
            "title": self.title,
            "description": self.description,
            "keywords": _LIST_SEP.join(self.keywords),
            "hierarchical_keywords": _LIST_SEP.join(self.hierarchical_keywords),
            "existing_keywords": _LIST_SEP.join(self.existing_keywords),
            "existing_title": self.existing_title,
            "existing_description": self.existing_description,
            "camera_model": self.camera_model,
            "lens_model": self.lens_model,
            "capture_date": self.capture_date,
            "gps_position": self.gps_position,
            "city": self.city,
            "country": self.country,
            "input_tokens": str(self.input_tokens),
            "output_tokens": str(self.output_tokens),
            "total_tokens": str(self.total_tokens),
            "seconds": f"{self.seconds:.3f}",
            "from_cache": _format_bool(value=self.from_cache),
            "retry": _format_bool(value=self.retry),
            "error": self.error,
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
        self._fh = path.open("w", encoding="utf-8", newline="")
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
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.as_dict())
    logger.info("csv_report_written", file=str(path), rows=len(materialized))
