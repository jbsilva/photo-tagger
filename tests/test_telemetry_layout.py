"""
Keep the Analytics Engine column layout the same in the three places that spell it out.

The Worker writes each beacon field to a numbered blob or double, queries.sql documents the map, and
the dashboard reads old beacons back by slot. A field moved in one place and not the others would
put one field's values under another's name, with no error anywhere.
"""

import re
from pathlib import Path


_TELEMETRY = Path(__file__).resolve().parent.parent / "telemetry"
# The last blob holds the whole beacon as JSON, so it has no single field to compare.
_JSON_BLOB = 19


def _worker_layout() -> dict[str, str]:
    """Map each slot ("blob3") to the field the Worker writes there."""
    source = (_TELEMETRY / "worker.js").read_text(encoding="utf-8")
    found = re.findall(r"(?:str|num)\(body\.(\w+)\)[^,]*, //\s*((?:blob|double)\d+)", source)
    return {slot: name for name, slot in found}


def _queries_layout() -> dict[str, str]:
    """Map each slot to the field the column map at the top of queries.sql names."""
    source = (_TELEMETRY / "queries.sql").read_text(encoding="utf-8")
    # Only the comment lines: a query's own WHERE clause (double13 = 0) is not part of the map.
    comments = "\n".join(line for line in source.splitlines() if line.startswith("--"))
    found = re.findall(r"((?:blob|double)\d+)\s*= (\w+)", comments)
    return {slot: name for slot, name in found if slot != f"blob{_JSON_BLOB}"}


def _dashboard_layout() -> dict[str, str]:
    """Map each slot to the field the dashboard reads from it, from its slot-ordered tuples."""
    source = (_TELEMETRY / "dashboard.py").read_text(encoding="utf-8")
    layout: dict[str, str] = {}
    for kind in ("blob", "double"):
        match = re.search(rf"_{kind.upper()}S = \(([^)]*)\)", source)
        assert match is not None, f"dashboard.py has no _{kind.upper()}S tuple"
        names = re.findall(r'"(\w+)"', match.group(1))
        layout |= {f"{kind}{index}": name for index, name in enumerate(names, start=1)}
    return layout


def test_the_worker_writes_every_slot_once() -> None:
    """Slots run from 1 with no gap, so a comment typo in worker.js cannot hide a field."""
    layout = _worker_layout()
    for kind in ("blob", "double"):
        numbers = sorted(int(slot.removeprefix(kind)) for slot in layout if slot.startswith(kind))
        assert numbers == list(range(1, len(numbers) + 1)), kind


def test_queries_sql_documents_the_worker_layout() -> None:
    """The column map in queries.sql is what people read when they write a new query."""
    assert _queries_layout() == _worker_layout()


def test_the_dashboard_reads_the_slots_the_worker_writes() -> None:
    """The dashboard rebuilds a v1 or v2 beacon from these slots, so they must line up."""
    assert _dashboard_layout() == _worker_layout()
