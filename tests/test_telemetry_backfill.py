"""
Tests for telemetry/backfill.py, the script that copies Analytics Engine rows into D1.

The statements it prints are run by wrangler against the real table, and the values in them come
from beacons anyone can send, so these load them into SQLite through the real migration.
"""

import importlib.util
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest


_TELEMETRY = Path(__file__).resolve().parent.parent / "telemetry"


@pytest.fixture(scope="module")
def backfill() -> ModuleType:
    """Load the script as a module; it lives outside the package on purpose."""
    spec = importlib.util.spec_from_file_location("backfill", _TELEMETRY / "backfill.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def events_db() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory database with the D1 schema the Worker writes to."""
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(
            (_TELEMETRY / "migrations" / "0001_events.sql").read_text(encoding="utf-8"),
        )
        yield db
    finally:
        db.close()


def test_a_v2_row_is_rebuilt_from_its_columns(
    backfill: ModuleType,
    events_db: sqlite3.Connection,
) -> None:
    """A row from before blob19 comes back as a beacon with its fields named again."""
    row = {
        "timestamp": "2026-07-01 10:00:00",
        "index1": "install-a",
        "_sample_interval": 1,
        "blob1": "0.9.0",
        "blob2": "cli",
        "blob4": "qwen/qwen3-vl-30b",
        "blob12": "",
        "double1": 2,
        "double2": 12,
        "double3": 45.5,
    }
    events_db.execute(backfill.insert_statement(row))
    stored = events_db.execute(
        "SELECT received_at, install_id, event, interface, schema_version, payload FROM events",
    ).fetchone()
    assert stored[:5] == ("2026-07-01T10:00:00.000Z", "install-a", "run", "cli", 2)
    payload = json.loads(stored[5])
    assert payload["model"] == "qwen/qwen3-vl-30b"
    assert payload["batch_size"] == 12  # noqa: PLR2004 - the row's own count
    assert payload["duration_seconds"] == 45.5  # noqa: PLR2004 - the row's own duration


def test_a_v3_row_keeps_its_json_and_its_sample_weight(
    backfill: ModuleType,
    events_db: sqlite3.Connection,
) -> None:
    """Blob19 holds the whole beacon, so it is copied as is, plus the weight of a sampled row."""
    beacon = {"schema_version": 3, "event": "session", "install_id": "b", "settings": {"x": 1}}
    row = {"timestamp": "2026-09-01 08:30:00", "_sample_interval": 4, "blob19": json.dumps(beacon)}
    events_db.execute(backfill.insert_statement(row))
    payload = json.loads(events_db.execute("SELECT payload FROM events").fetchone()[0])
    assert payload == {**beacon, "sample_weight": 4}


def test_hostile_text_stays_inside_its_literal(
    backfill: ModuleType,
    events_db: sqlite3.Connection,
) -> None:
    """A quote or a control character in a beacon cannot end the literal or break the file."""
    row = {
        "timestamp": "2026-07-02 00:00:00",
        "index1": "x'); DROP TABLE events; --",
        "blob2": "gui\x00\x07",
        "blob4": "it's",
    }
    events_db.execute(backfill.insert_statement(row))
    install_id, interface, payload = events_db.execute(
        "SELECT install_id, interface, payload FROM events",
    ).fetchone()
    assert install_id == "x'); DROP TABLE events; --"
    assert interface == "gui"
    assert json.loads(payload)["model"] == "it's"
