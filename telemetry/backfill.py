# /// script
# requires-python = ">=3.14"
# dependencies = ["httpx2>=2.10"]
# ///
"""
Copy the events Analytics Engine still holds into the D1 table, as SQL for wrangler to run.

AE keeps three months; D1 keeps everything. Run it once, after the Worker starts writing to D1:
"Copy the Analytics Engine history into D1" in README.md has the two commands.

Only rows older than the first D1 row are copied, so no event is stored twice and a second run
copies nothing. Needs CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (Account Analytics: Read and D1:
Read), and CLOUDFLARE_D1_DATABASE_ID. Pass --before ISO_TIMESTAMP to pick the cutoff yourself.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import cast

import httpx2


_API = "https://api.cloudflare.com/client/v4/accounts/{account}"
_DATASET = "photo_tagger_telemetry"
_TIMEOUT_SECONDS = 60.0

# The AE column layout (see worker.js and queries.sql), mapped back to beacon field names.
_BLOBS = (
    "app_version",
    "interface",
    "provider",
    "model",
    "arch",
    "os",
    "os_release",
    "python_version",
    "output_language",
    "ui_language",
    "file_types",
    "event",
    "cpu",
    "gpu",
    "exception_type",
    "crash_location",
    "crash_frames",
    "failure_kinds",
)
_DOUBLES = (
    "schema_version",
    "batch_size",
    "duration_seconds",
    "success_count",
    "failure_count",
    "cache_hits",
    "retry_successes",
    "workers",
    "total_tokens",
    "inference_seconds",
    "cpu_count",
    "memory_gb",
    "dry_run",
)
_INT_DOUBLES = frozenset(_DOUBLES) - {"duration_seconds", "inference_seconds"}


def _env(name: str) -> str:
    """Return a required environment variable, or stop with a message naming it."""
    if value := os.environ.get(name, ""):
        return value
    sys.exit(f"Set {name} first (see telemetry/README.md).")


def _post(url: str, token: str, *, sql: str, as_json: bool) -> dict[str, object]:
    """POST one statement to a Cloudflare SQL endpoint and return the decoded reply."""
    headers = {"Authorization": f"Bearer {token}"}
    if as_json:
        response = httpx2.post(url, headers=headers, json={"sql": sql}, timeout=_TIMEOUT_SECONDS)
    else:
        response = httpx2.post(url, headers=headers, content=sql, timeout=_TIMEOUT_SECONDS)
    if response.status_code != httpx2.codes.OK:
        sys.exit(f"{url} returned {response.status_code}: {response.text[:500]}")
    reply: dict[str, object] = response.json()
    return reply


def first_d1_event(account: str, token: str, database: str) -> datetime | None:
    """Return when the oldest D1 row arrived, or None while the table is empty."""
    url = f"{_API.format(account=account)}/d1/database/{database}/query"
    reply = _post(url, token, sql="SELECT MIN(received_at) AS first FROM events", as_json=True)
    result = cast("list[dict[str, list[dict[str, str | None]]]]", reply["result"])
    first = result[0]["results"][0]["first"]
    return parse_time(first) if first else None


def ae_rows(account: str, token: str, before: datetime) -> list[dict[str, object]]:
    """Fetch every AE data point older than *before*, oldest first."""
    columns = ", ".join(
        [
            "timestamp",
            "index1",
            "_sample_interval",
            *(f"blob{i}" for i in range(1, 20)),
            *(f"double{i}" for i in range(1, 14)),
        ],
    )
    sql = (
        f"SELECT {columns} FROM {_DATASET} "  # noqa: S608 - no user input; the cutoff is an int.
        f"WHERE toUnixTimestamp(timestamp) < {int(before.timestamp())} "
        "ORDER BY timestamp FORMAT JSON"
    )
    url = f"{_API.format(account=account)}/analytics_engine/sql"
    return cast("list[dict[str, object]]", _post(url, token, sql=sql, as_json=False)["data"])


def parse_time(text: str) -> datetime:
    """Parse an AE ("2026-09-20 12:34:56") or D1 (ISO 8601) timestamp as UTC."""
    parsed = datetime.fromisoformat(text.replace(" ", "T"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _json_copy(row: dict[str, object]) -> dict[str, object] | None:
    """Return the beacon AE kept as JSON in blob19 (v3 on), or None when there is none."""
    try:
        copy = json.loads(str(row.get("blob19") or ""))
    except ValueError:
        return None
    return cast("dict[str, object]", copy) if isinstance(copy, dict) else None


def _from_columns(row: dict[str, object]) -> dict[str, object]:
    """Rebuild a v1/v2 beacon from the AE columns, which is all AE kept of it."""
    payload: dict[str, object] = {"install_id": str(row.get("index1") or "")}
    for number, name in enumerate(_BLOBS, start=1):
        if value := str(row.get(f"blob{number}") or ""):
            payload[name] = value
    for number, name in enumerate(_DOUBLES, start=1):
        value = float(str(row.get(f"double{number}") or 0))
        payload[name] = int(value) if name in _INT_DOUBLES else value
    payload.setdefault("event", "run")
    return payload


def to_payload(row: dict[str, object]) -> dict[str, object]:
    """Rebuild the beacon a row came from: its JSON copy when AE kept one, else the columns."""
    payload = _json_copy(row) or _from_columns(row)
    weight = int(float(str(row.get("_sample_interval") or 1)))
    if weight > 1:
        # A sampled AE row stands for several events; the dashboard weighs the copy the same way.
        payload["sample_weight"] = weight
    return payload


def _literal(value: object) -> str:
    """
    Quote *value* as an SQLite string literal.

    Control characters go; quotes are doubled.
    """
    text = "".join(char for char in str(value) if char >= " ")
    return "'" + text.replace("'", "''") + "'"


def insert_statement(row: dict[str, object]) -> str:
    """Render one AE row as an INSERT into the D1 events table."""
    payload = to_payload(row)
    received = parse_time(str(row["timestamp"])).isoformat(timespec="milliseconds")
    values = [
        _literal(received.replace("+00:00", "Z")),
        _literal(payload.get("install_id", "")),
        _literal(payload.get("session_id", "")),
        _literal(payload.get("event") or "run"),
        _literal(payload.get("interface", "")),
        _literal(payload.get("app_version", "")),
        str(int(float(str(payload.get("schema_version") or 0)))),
        _literal(json.dumps(payload, separators=(",", ":"))),
    ]
    columns = "received_at, install_id, session_id, event, interface, app_version, schema_version"
    # Every value went through _literal, which quotes and escapes it.
    return f"INSERT INTO events ({columns}, payload) VALUES ({', '.join(values)});"  # noqa: S608


def main() -> None:
    """Write the INSERT statements for every AE row older than the cutoff to stdout."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", help="copy AE rows older than this ISO 8601 time")
    args = parser.parse_args()
    account = _env("CLOUDFLARE_ACCOUNT_ID")
    token = _env("CLOUDFLARE_API_TOKEN")
    if args.before:
        before = parse_time(args.before)
    else:
        first = first_d1_event(account, token, _env("CLOUDFLARE_D1_DATABASE_ID"))
        before = first or datetime.now(tz=UTC)
    rows = ae_rows(account, token, before)
    sys.stdout.writelines(insert_statement(row) + "\n" for row in rows)
    sys.stderr.write(f"{len(rows)} event(s) older than {before.isoformat()}.\n")


if __name__ == "__main__":
    main()
