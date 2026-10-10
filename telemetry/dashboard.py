# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair>=6.2.2",
#     "httpx2>=2.10",
#     "marimo>=0.23",
#     "pandas>=2.3.3",
# ]
# ///
# mypy: ignore-errors
# The dependencies above live only in marimo's --sandbox venv, never in the project venv, so zuban
# cannot resolve them here (same escape hatch as the gui package). Altair 6 is required: 5.x fails
# to import on Python 3.14 (it declares `closed=True` TypedDicts that 3.14's stdlib typing does not
# accept).
"""
Marimo dashboard over the photo-tagger telemetry events.

It reads the D1 table when CLOUDFLARE_D1_DATABASE_ID is set (every event since the collector
started), else Analytics Engine (the last three months). The raw events are fetched once per
refresh and every view is computed from them in pandas, so the filters (window, interface, which
installs) apply instantly and to everything at once. Nothing here can write.

Run it (read-only app view, dependencies resolved into an isolated venv):

    uv run poe dashboard

or open it as an editable notebook with `uv run poe dashboard-edit`. It needs CLOUDFLARE_ACCOUNT_ID
and CLOUDFLARE_API_TOKEN (Account Analytics: Read, plus D1: Read for the D1 table) in the
environment; see telemetry/README.md. `uv run poe dashboard-demo` shows made-up events instead.
"""

import marimo


__generated_with = "0.23.13"
app = marimo.App(width="full", app_title="photo-tagger telemetry")


@app.cell
def _():
    import json
    import os
    import random
    import re
    import uuid
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    import altair as alt
    import httpx2
    import marimo as mo
    import pandas as pd

    return UTC, Path, alt, datetime, httpx2, json, mo, os, pd, random, re, timedelta, uuid


@app.cell
def _(mo):
    # Validated categorical/ink palette (the project's reference palette), stepped per theme.
    # Marks wear the hue; text and chrome wear the ink tokens, never the series color. Crash
    # views wear the reserved status hues, which never impersonate a data series. The first three
    # categorical slots (validated in both modes) are the only ones in use; aqua sits below 3:1 on
    # the light surface, so every chart using it carries direct labels and a table view.
    _dark = mo.app_meta().theme == "dark"
    COLORS = {
        "hue": "#3987e5" if _dark else "#2a78d6",
        "slot2": "#d95926" if _dark else "#eb6834",
        "slot3": "#199e70" if _dark else "#1baf7a",
        "critical": "#d03b3b",
        "good": "#0ca30c",
        "rest": "#383835" if _dark else "#e1e0d9",
        "surface": "#1a1a19" if _dark else "#fcfcfb",
        "grid": "#2c2c2a" if _dark else "#e1e0d9",
        "baseline": "#383835" if _dark else "#c3c2b7",
        "ink": "#ffffff" if _dark else "#0b0b0b",
        "ink2": "#c3c2b7" if _dark else "#52514e",
        "muted": "#898781",
        # Sequential blue, anchored at the surface: light near zero on a light surface, dark near
        # zero on a dark one.
        "seq_low": "#184f95" if _dark else "#cde2fb",
        "seq_high": "#cde2fb" if _dark else "#104281",
    }
    return (COLORS,)


@app.cell
def _(mo, os):
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    database_id = os.environ.get("CLOUDFLARE_D1_DATABASE_ID", "")
    _demo = os.environ.get("PHOTO_TAGGER_DASHBOARD_DEMO", "").strip().lower()
    if _demo in {"1", "true", "yes", "on"}:
        source = "demo"
    elif account_id and api_token:
        source = "d1" if database_id else "ae"
    else:
        source = ""
    mo.stop(
        not source,
        mo.md(
            """
            **Credentials missing.** The dashboard reads the telemetry through the Cloudflare API,
            which needs:

            1. `CLOUDFLARE_ACCOUNT_ID`: the account id on the Cloudflare dashboard overview.
            2. `CLOUDFLARE_API_TOKEN`: an [API token](https://dash.cloudflare.com/profile/api-tokens)
               with **Account Analytics: Read**, plus **D1: Read** for the long-term table.
            3. `CLOUDFLARE_D1_DATABASE_ID` (optional): the D1 database id. With it, the dashboard
               reads every event. Without it, it reads Analytics Engine, which keeps three months.

            Export them and restart, or try it on made-up events first:

            ```bash
            export CLOUDFLARE_ACCOUNT_ID=...
            export CLOUDFLARE_API_TOKEN=...
            export CLOUDFLARE_D1_DATABASE_ID=...
            uv run poe dashboard
            uv run poe dashboard-demo
            ```
            """,
        ).callout(kind="warn"),
    )
    return account_id, api_token, database_id, source


@app.cell
def _(mo, source):
    _where = {
        "d1": "the **D1** table, which keeps every event since the collector started",
        "ae": (
            "**Analytics Engine**, which keeps three months: set `CLOUDFLARE_D1_DATABASE_ID` to "
            "look further back"
        ),
        "demo": "**made-up demo events** (`PHOTO_TAGGER_DASHBOARD_DEMO`), not real telemetry",
    }[source]
    mo.md(
        f"""
        # photo-tagger telemetry
        Anonymous usage and crash beacons from the [collector](README.md), read from {_where}.
        Counts weigh each event by its sample weight, so they stay honest if Analytics Engine
        samples. Fields added in schema v3 (settings, sessions, the GUI's runs) are empty on events
        from older installs, so those views count v3 events only.
        """,
    )


@app.cell
def _(mo):
    window = mo.ui.dropdown(
        options={
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 365 days": 365,
            "All time": 0,
        },
        value="Last 30 days",
        label="Window",
    )
    refresh = mo.ui.button(label="Refresh", value=0, on_click=lambda count: count + 1)
    return refresh, window


@app.cell
def _():
    # The demo's made-up spread of each setting: a switch's odds of being on (CLI, GUI), and a
    # multi-valued setting's pool to pick from. None of it is a measurement.
    _SWITCH_ODDS = {
        "vocabulary": (0.25, 0.25),
        "vocabulary_strict": (0.1, 0.1),
        "repair_language": (0.2, 0.3),
        "artist": (0.3, 0.3),
        "copyright_notice": (0.25, 0.25),
        "paced": (0.15, 0.15),
        "recursive": (0.6, 1.0),
        "skip_tagged": (0.4, 0.0),
        "cache": (0.3, 1.0),
        "autosave": (0.0, 0.35),
        "dry_run": (0.1, 0.0),
        "hint": (0.08, 0.0),
    }
    _POOLS = {
        "endpoint": ["localhost"] * 6 + ["lan", "lan", "api.openai.com", "public"],
        "sidecar_mode": ["all", "all", "raw", "none", "both"],
        "jpeg_dimensions": [640, 640, 640, 896, 1024],
        "temperature": [0.2, 0.2, 0.2, 0.4],
        "duty": [100, 100, 100, 50],
        "workers": [1, 1, 2, 4],
    }
    _DEFAULTS = {
        "endpoint": "localhost",
        "sidecar_mode": "all",
        "jpeg_dimensions": 640,
        "temperature": 0.2,
        "duty": 100,
        "workers": 1,
    }
    _HARDWARE = (  # arch, cpu, gpu, cpu_count, memory_gb
        ("arm64", "Apple M3 Pro", "Apple M3 Pro", 12, 36),
        ("arm64", "Apple M1", "Apple M1", 8, 16),
        ("x86_64", "AMD Ryzen 9 7950X", "NVIDIA GeForce RTX 4070", 32, 64),
        ("x86_64", "Intel Core i7-12700H", "Intel Iris Xe", 20, 32),
    )
    _MODELS = ("qwen/qwen3-vl-30b", "gemma-3-27b", "llava:13b", "gpt-4o-mini")

    def demo_envelope(rng, install, ts, interface, event):
        """Make up the fields every beacon carries, like the client's envelope."""
        arch, cpu, gpu, cpu_count, memory_gb = _HARDWARE[install["hardware"]]
        return {
            "ts": ts.isoformat(),
            "schema_version": 3,
            "event": event,
            "install_id": install["install_id"],
            "session_id": f"s-{rng.getrandbits(32):08x}",
            "app_version": rng.choice(["0.9.0", "0.10.0", "0.10.0"]),
            "interface": interface,
            "command": "gui" if interface == "gui" else "tag",
            "consent": rng.choice(["default"] * 12 + ["env-opt-in", "gui-opt-in"]),
            "arch": arch,
            "os": install["os"],
            "python_version": "3.14.0",
            "cpu": cpu,
            "gpu": gpu,
            "cpu_count": cpu_count,
            "memory_gb": memory_gb,
            "install_method": install["method"],
            "installer": "uv" if install["method"] in {"uv-tool", "editable"} else "pip",
            "system_locale": install["locale"],
            "utc_offset_minutes": install["utc_offset"],
            "local_hour": rng.choice([9, 10, 14, 20, 21, 22, 23, 1, 2]),
            "local_weekday": rng.randrange(7),
            "ci": False,
            "terminal": interface == "cli",
        }

    def demo_settings(rng, interface):
        """Make up the settings of one run or window, under the names the client sends."""
        gui = interface == "gui"
        settings = {name: rng.choice(pool) for name, pool in _POOLS.items()}
        return settings | {name: rng.random() < odds[gui] for name, odds in _SWITCH_ODDS.items()}

    def demo_run(rng, base, settings, job, trigger, size):
        """Make up one run beacon: a CLI batch, or a GUI generation or save."""
        failed = min(size - 1, rng.choice([0, 0, 0, 0, 1, 2]))
        seconds = rng.uniform(2.0, 9.0)
        changed = [name for name, value in settings.items() if value != _DEFAULTS.get(name, False)]
        return base | {
            "event": "run",
            "job": job,
            "trigger": trigger,
            "outcome": "partial" if failed else "ok",
            "provider": "openai" if settings["endpoint"] == "api.openai.com" else "lmstudio",
            "model": rng.choice(_MODELS),
            "batch_size": size,
            "duration_seconds": size * seconds * rng.uniform(1.0, 1.4),
            "output_language": rng.choice(["English"] * 5 + ["German", "Brazilian Portuguese"]),
            "ui_language": rng.choice(["en", "en", "pt_BR"]),
            "file_types": rng.choice(["cr3,jpg", "jpg", "nef", "arw,jpg", "dng"]),
            "success_count": size - failed,
            "failure_count": failed,
            "failure_kinds": f"timeout:{failed}" if failed else "",
            "failure_types": f"ReadTimeout:{failed}" if failed else "",
            "cache_hits": rng.randrange(max(1, size // 3)),
            "total_tokens": size * rng.randrange(900, 1500),
            "inference_seconds": size * seconds,
            "dry_run": int(settings["dry_run"]),
            "flagged": rng.choice([0, 0, 0, 1]),
            "language_repairs": rng.choice([0, 1]) * settings["repair_language"],
            "photo_seconds_median": seconds,
            "keywords_mean": rng.uniform(12, 24),
            "power": rng.choice(["ac", "ac", "battery", "none"]),
            "settings": settings,
            "setup": {
                "config_file": rng.choice(["none", "user", "user", "cwd"]),
                "config_keys": rng.choice(["", "provider.model", "output.sidecar,provider.model"]),
                "cli_flags": rng.choice(["--input", "--input,--recursive", "--dry-run,--input"]),
                "changed": ",".join(sorted(changed)),
            },
        }

    return demo_envelope, demo_run, demo_settings


@app.cell
def _(UTC, datetime, demo_envelope, demo_run, demo_settings, random, timedelta, uuid):
    _FEATURES = (
        "viewer",
        "grid",
        "search",
        "select",
        "copyright",
        "pacing",
        "export_csv",
        "undo",
        "watch_start",
        "keyword_changes",
    )
    _COMMANDS = ("doctor", "show", "set", "undo", "vocabulary")
    _CRASHES = (  # exception type, frames from the outermost to the deepest
        ("KeyError", "main:tag:1204>pipeline:run_batch:714>metadata:read_existing:88"),
        ("OSError", "main:tag:1204>pipeline:run_batch:714>image_io:load_preview:57"),
    )
    _NO_START = {"outcome": "setup-failed", "error_kind": "ProviderError", "success_count": 0}
    _GUI_INSTALLS = 0.55  # installs that ever open the window
    _GUI_SHARE = 0.7  # of those installs' invocations, the share that open the window
    _SAVE_LATER = 0.6  # generations reviewed first, then written with Save Selected
    _NOT_A_MODEL_CALL = {"photo_seconds_median": 0.0, "inference_seconds": 0.0, "total_tokens": 0}

    def _cli_events(rng, install, ts):
        base = demo_envelope(rng, install, ts, "cli", "run")
        kind = rng.choices(("tag", "command", "crash", "no-start"), weights=(76, 18, 3, 3))[0]
        if kind == "command":
            return [
                base
                | {
                    "event": "command",
                    "command": rng.choice(_COMMANDS),
                    "outcome": "ok",
                    "exit_code": 0,
                    "duration_seconds": rng.uniform(0.5, 4.0),
                },
            ]
        if kind == "crash":
            exception, frames = rng.choice(_CRASHES)
            location = frames.rsplit(">", 1)[-1]
            return [
                base
                | {
                    "event": "crash",
                    "exception_type": exception,
                    "exception_module": "builtins",
                    "exception_chain": exception,
                    "crash_location": location,
                    "crash_frames": frames,
                    "crash_stack": "cyclopts.core:__call__:1320>"
                    + ">".join(f"photo_tagger.{frame}" for frame in frames.split(">")),
                    "libraries": "openai=2.1.0,pydantic=2.13.5,pydantic-ai-slim=2.46.0",
                },
            ]
        run = demo_run(rng, base, demo_settings(rng, "cli"), "tag", "", rng.randrange(1, 400))
        return [run | _NO_START if kind == "no-start" else run]

    def _gui_events(rng, install, ts):
        base = demo_envelope(rng, install, ts, "gui", "session")
        settings = demo_settings(rng, "gui")
        runs = []
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            size = rng.randrange(1, 300)
            trigger = rng.choice(["batch", "batch", "batch", "single", "hint", "retry", "watch"])
            chained = settings["autosave"] or trigger == "watch"
            run = demo_run(rng, base, settings, "generate", trigger, size)
            run["gui"] = {
                "autosave": chained,
                "saved": size if chained else 0,
                "hinted": size if trigger == "hint" else rng.choice([0, 0, 1]),
                "mixed_pass": run["flagged"],
                "regenerated_flagged": rng.choice([0, 0, 1]),
            }
            runs.append(run)
        saves = [
            demo_run(rng, base, settings, "save", "batch", run["batch_size"]) | _NOT_A_MODEL_CALL
            for run in runs
            if not run["gui"]["autosave"] and rng.random() < _SAVE_LATER
        ]
        busy = sum(run["duration_seconds"] for run in [*runs, *saves])
        length = busy + rng.uniform(60, 12 * 3600)
        unattended = busy * rng.uniform(0.5, 1.0)
        in_use = min(length - busy, rng.uniform(30, 1800))  # input while no run was in flight
        generated = sum(run["batch_size"] for run in runs)
        edited = rng.randrange(20)
        session = base | {
            "session_seconds": length,
            "busy_seconds": busy,
            "engaged_seconds": busy - unattended + in_use,
            "idle_seconds": length - busy - in_use,
            "unattended_seconds": unattended,
            "runs": len(runs),
            "save_runs": len(saves),
            "single_saves": rng.choice([0, 0, 3, 12]),
            "first_run_seconds": rng.uniform(20, 600) if runs else -1.0,
            "photos_generated": generated,
            "photos_saved": generated // 2,
            "view_mode": rng.choice(["tree", "tree", "list"]),
            "features": {name: rng.choice([0, 0, 1, 2, 5]) for name in _FEATURES},
            "review": {
                "saved": edited + 20,
                "edited": edited,
                "edited_title": edited // 2,
                "edited_description": edited // 3,
                "edited_keywords": edited,
                "keywords_added": edited * 2,
                "keywords_removed": edited * 3,
                "hinted": rng.choice([0, 1]),
                "flagged": rng.choice([0, 0, 1]),
                "review_seconds_median": rng.uniform(20, 36000),
            },
            "display": {
                "screen": rng.choice(["1512x982@2", "1920x1080@1", "2560x1440@1", "1440x900@2"]),
                "color_scheme": rng.choice(["dark", "light"]),
            },
            "settings": settings,
        }
        return [*runs, *saves, session]

    def demo_events(days: int) -> list[dict]:
        """Make up five months of plausible events, so every view can be seen without data."""
        rng = random.Random(7)  # the same events on every refresh; the window only filters them
        now = datetime.now(tz=UTC)
        installs = [
            {
                "install_id": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
                "os": rng.choice(["Darwin", "Darwin", "Linux", "Windows"]),
                "hardware": rng.randrange(4),
                "locale": rng.choice(["en_US", "de_DE", "pt_BR", "en_GB", "fr_FR"]),
                "utc_offset": rng.choice([-300, 0, 60, 120, -180]),
                "gui": rng.random() < _GUI_INSTALLS,
                "method": rng.choice(["uv-tool", "uv-tool", "pipx", "conda", "app", "venv"]),
                "age": rng.uniform(0, 150),
                "activity": rng.choice([1, 2, 3, 8, 20]),
            }
            for _ in range(45)
        ]
        installs[0] |= {"method": "editable", "activity": 40}  # a developer, hidden by default
        events = []
        for install in installs:
            for _ in range(install["activity"]):
                ts = now - timedelta(days=rng.uniform(0, install["age"]), hours=rng.uniform(0, 24))
                gui = install["gui"] and rng.random() < _GUI_SHARE
                events += _gui_events(rng, install, ts) if gui else _cli_events(rng, install, ts)
        cutoff = (now - timedelta(days=days)).isoformat()
        return [event for event in events if not days or event["ts"] >= cutoff]

    return (demo_events,)


@app.cell
def _(account_id, api_token, database_id, httpx2):
    def _post(path: str, **kwargs: object) -> dict:
        response = httpx2.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/{path}",
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=60.0,
            **kwargs,
        )
        if response.status_code != httpx2.codes.OK:
            msg = f"Cloudflare API returned {response.status_code}: {response.text[:500]}"
            raise RuntimeError(msg)
        return response.json()

    def query_d1(sql: str, params: list[str] | None = None) -> list[dict]:
        """Run one statement against the D1 table and return its rows."""
        body = _post(f"d1/database/{database_id}/query", json={"sql": sql, "params": params or []})
        return body["result"][0]["results"]

    def query_ae(sql: str) -> list[dict]:
        """Run one statement against Analytics Engine and return its rows."""
        return _post("analytics_engine/sql", content=sql.strip().removesuffix(";")).get("data", [])

    return query_ae, query_d1


@app.cell
def _(UTC, datetime, json, query_ae, query_d1, timedelta):
    # The AE columns of a v1/v2 beacon, in slot order. From v3 on, blob19 holds the whole beacon.
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
    _AE_COLUMNS = ", ".join(
        ["timestamp", "index1", "_sample_interval"]
        + [f"blob{i}" for i in range(1, 20)]
        + [f"double{i}" for i in range(1, len(_DOUBLES) + 1)],
    )
    # Analytics Engine drops data after this long, whatever the window asks for.
    AE_RETENTION_DAYS = 90
    # Rows per D1 request, so a long history never has to fit in one response.
    _D1_PAGE = 5000

    def _parse(text: object) -> dict | None:
        """Return the beacon a stored JSON string holds, or None when it holds none."""
        try:
            value = json.loads(str(text or ""))
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def _from_columns(row: dict) -> dict:
        """Rebuild a v1/v2 beacon from the AE columns, which is all AE kept of it."""
        event = {
            name: row[f"blob{number}"]
            for number, name in enumerate(_BLOBS, start=1)
            if row.get(f"blob{number}")
        }
        event |= {
            name: float(row.get(f"double{number}") or 0)
            for number, name in enumerate(_DOUBLES, start=1)
        }
        return event | {"install_id": row.get("index1") or ""}

    def fetch_d1(days: int) -> list[dict]:
        """Return every D1 event in the window as the payload the client sent, oldest first."""
        since = ""  # every ISO timestamp sorts after the empty string
        if days:
            since = (datetime.now(tz=UTC) - timedelta(days=days)).isoformat(timespec="milliseconds")
        # The Worker stamps received_at as ISO 8601 with a Z, so the text comparison is by time.
        after = [since.replace("+00:00", "Z"), "0"]
        events = []
        # Pages follow (received_at, id), which the received_at index serves: a short window reads
        # only its own rows, and D1 bills rows read, however long the history grows.
        while rows := query_d1(
            "SELECT id, received_at, payload FROM events WHERE (received_at, id) > (?, ?) "
            f"ORDER BY received_at, id LIMIT {_D1_PAGE}",
            after,
        ):
            after = [rows[-1]["received_at"], str(rows[-1]["id"])]
            events += [
                event | {"ts": row["received_at"]}
                for row in rows
                if (event := _parse(row["payload"])) is not None
            ]
        return events

    def fetch_ae(days: int) -> list[dict]:
        """Return every AE data point in the window, rebuilt into the payload the client sent."""
        span = min(days or AE_RETENTION_DAYS, AE_RETENTION_DAYS)
        rows = query_ae(
            f"SELECT {_AE_COLUMNS} FROM photo_tagger_telemetry "
            f"WHERE timestamp > NOW() - INTERVAL '{int(span)}' DAY FORMAT JSON",
        )
        return [
            (_parse(row.get("blob19")) or _from_columns(row))
            | {"ts": row["timestamp"], "sample_weight": float(row.get("_sample_interval") or 1)}
            for row in rows
        ]

    return AE_RETENTION_DAYS, fetch_ae, fetch_d1


@app.cell
def _(demo_events, fetch_ae, fetch_d1, refresh, source, window):
    _ = refresh.value  # the button makes this cell (and so every view) fetch again
    days = int(window.value)
    if source == "demo":
        payloads = demo_events(days)
    elif source == "d1":
        payloads = fetch_d1(days)
    else:
        payloads = fetch_ae(days)
    return days, payloads


@app.cell
def _(pd):
    # The schema that added settings, sessions, and the GUI's own runs. Older events lack them.
    SCHEMA_V3 = 3
    # Columns every view may read. An event kind that lacks one gets it empty, so no view has to
    # guard against a KeyError on a window that happens to hold no such event.
    _TEXT = (
        "event",
        "install_id",
        "session_id",
        "interface",
        "command",
        "job",
        "trigger",
        "outcome",
        "error_kind",
        "consent",
        "app_version",
        "python_version",
        "provider",
        "model",
        "arch",
        "os",
        "cpu",
        "gpu",
        "install_method",
        "installer",
        "system_locale",
        "output_language",
        "ui_language",
        "file_types",
        "failure_kinds",
        "failure_types",
        "exception_type",
        "exception_module",
        "exception_chain",
        "crash_location",
        "crash_frames",
        "crash_stack",
        "libraries",
        "power",
        "view_mode",
        "setup.config_file",
        "setup.config_keys",
        "setup.cli_flags",
        "setup.env_vars",
        "setup.changed",
        "display.screen",
        "display.color_scheme",
    )
    _NUMBERS = (
        "sample_weight",
        "schema_version",
        "batch_size",
        "duration_seconds",
        "success_count",
        "failure_count",
        "cache_hits",
        "total_tokens",
        "inference_seconds",
        "dry_run",
        "cpu_count",
        "memory_gb",
        "flagged",
        "language_repairs",
        "photo_seconds_median",
        "keywords_mean",
        "local_hour",
        "local_weekday",
        "session_seconds",
        "busy_seconds",
        "engaged_seconds",
        "idle_seconds",
        "unattended_seconds",
        "runs",
        "single_saves",
        "first_run_seconds",
        "photos_generated",
        "photos_saved",
        "photos_unsaved",
        "startup_seconds",
        "gui.autosave",
        "gui.saved",
        "gui.hinted",
        "gui.mixed_pass",
        "gui.regenerated_flagged",
        "review.saved",
        "review.edited",
        "review.edited_title",
        "review.edited_description",
        "review.edited_keywords",
        "review.keywords_added",
        "review.keywords_removed",
        "review.hinted",
        "review.flagged",
        "review.review_seconds_median",
    )

    def to_frame(payloads: list[dict]) -> pd.DataFrame:
        """Flatten the payloads into one row per event, nested groups as dotted columns."""
        frame = pd.json_normalize(payloads, sep=".") if payloads else pd.DataFrame()
        for column in _TEXT:
            if column not in frame:
                frame[column] = ""
            frame[column] = frame[column].fillna("").astype(str)
        for column in _NUMBERS:
            if column not in frame:
                frame[column] = float("nan")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "ts" not in frame:
            frame["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, format="mixed")
        # An event AE did not sample stands for itself alone.
        frame["weight"] = frame["sample_weight"].fillna(1.0)
        # v1 rows may have no event name: they were all runs.
        frame.loc[frame["event"] == "", "event"] = "run"
        # A GUI "run" from before schema v3 is a whole window, idle time included: a session.
        legacy_gui = (
            (frame["schema_version"].fillna(0) < SCHEMA_V3)
            & (frame["interface"] == "gui")
            & (frame["event"] == "run")
        )
        frame.loc[legacy_gui, "event"] = "session"
        frame.loc[legacy_gui, "session_seconds"] = frame.loc[legacy_gui, "duration_seconds"]
        frame.loc[legacy_gui, "photos_generated"] = frame.loc[legacy_gui, "batch_size"]
        frame.loc[legacy_gui, "command"] = "gui"
        # A CLI run from before v3 was always the default command.
        frame.loc[(frame["event"] == "run") & (frame["command"] == ""), "command"] = "tag"
        frame["day"] = frame["ts"].dt.floor("D")
        return frame

    return SCHEMA_V3, to_frame


@app.cell
def _(payloads, to_frame):
    events = to_frame(payloads)
    return (events,)


@app.cell
def _(Path, events, mo, os, pd):
    def _local_install_ids() -> set[str]:
        """Return this machine's install id: the developer's own test runs come from it."""
        base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        try:
            return {(Path(base) / "photo-tagger" / "install-id").read_text().strip()}
        except OSError:
            return set()

    _env_ids = {
        part.strip()
        for part in os.environ.get("PHOTO_TAGGER_DASHBOARD_EXCLUDE", "").split(",")
        if part.strip()
    }
    # The collector's own deploy checks send this id, never a real install.
    _synthetic = {"00000000-0000-0000-0000-000000000000"}
    _mine = _env_ids | _local_install_ids() | _synthetic
    if events.empty:
        _installs = pd.DataFrame(columns=["install_id", "events", "last", "os", "arch"])
    else:
        _installs = (
            events.groupby("install_id")
            .agg(
                events=("weight", "sum"),
                last=("ts", "max"),
                os=("os", "last"),
                arch=("arch", "last"),
            )
            .reset_index()
            .sort_values("events", ascending=False)
        )
    install_labels = {
        f"{row.install_id[:8]} · {row.os or '?'} {row.arch or ''} · {row.events:,.0f} events · "
        f"last {row.last:%Y-%m-%d}"
        + (" · this machine" if row.install_id in _mine else ""): row.install_id
        for row in _installs.itertuples()
        if row.install_id
    }
    include = mo.ui.multiselect(options=install_labels, label="Only these installs")
    exclude = mo.ui.multiselect(
        options=install_labels,
        value=[label for label, install in install_labels.items() if install in _mine],
        label="Exclude installs",
    )
    interface = mo.ui.dropdown(
        options={"All interfaces": "", "CLI only": "cli", "GUI only": "gui"},
        value="All interfaces",
        label="Interface",
    )
    hide_dev = mo.ui.checkbox(value=True, label="Hide development checkouts")
    return exclude, hide_dev, include, install_labels, interface


@app.cell
def _(AE_RETENTION_DAYS, days, exclude, hide_dev, include, interface, mo, refresh, source, window):
    _controls = mo.hstack(
        [window, interface, include, exclude, hide_dev, refresh],
        justify="start",
        align="end",
        gap=1,
        wrap=True,
    )
    _capped = source == "ae" and (days == 0 or days > AE_RETENTION_DAYS)
    mo.vstack(
        [
            _controls,
            mo.md(
                f"Analytics Engine keeps {AE_RETENTION_DAYS} days, so this window shows those "
                "days only. Set `CLOUDFLARE_D1_DATABASE_ID` to read every event.",
            ).callout(kind="info")
            if _capped
            else mo.md(""),
        ],
        gap=0.5,
    )


@app.cell
def _(SCHEMA_V3, events, exclude, hide_dev, include, interface, mo):
    ev = events
    if interface.value:
        ev = ev[ev["interface"] == interface.value]
    if include.value:
        ev = ev[ev["install_id"].isin(include.value)]
    if exclude.value:
        ev = ev[~ev["install_id"].isin(exclude.value)]
    if hide_dev.value:
        # editable: a development checkout; source: running from a clone without installing.
        ev = ev[~ev["install_method"].isin(["editable", "source"])]
    _all_runs = ev[ev["event"] == "run"]
    # A GUI save writes what an earlier run generated, so it is no tagging work of its own: "runs"
    # are CLI batches and GUI generations, and the views that need the saves read them apart.
    runs = _all_runs[_all_runs["job"] != "save"]
    save_runs = _all_runs[_all_runs["job"] == "save"]
    sessions = ev[ev["event"] == "session"]
    commands = ev[ev["event"] == "command"]
    launches = ev[ev["event"] == "launch"]
    crashes = ev[ev["event"] == "crash"]
    v3_runs = runs[runs["schema_version"] >= SCHEMA_V3]
    gui_runs = v3_runs[v3_runs["interface"] == "gui"]
    gui_sessions = sessions[
        (sessions["interface"] == "gui") & (sessions["schema_version"] >= SCHEMA_V3)
    ]
    _hidden = len(events) - len(ev)
    mo.md(
        f"_{len(ev):,} events from {ev['install_id'].nunique():,} installs in view"
        + (f"; {_hidden:,} hidden by the filters._" if _hidden else "._"),
    )
    return (
        commands,
        crashes,
        ev,
        gui_runs,
        gui_sessions,
        launches,
        runs,
        save_runs,
        sessions,
        v3_runs,
    )


@app.cell
def _(pd, re):
    def fmt_int(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.0f}"

    def fmt_num(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.1f}"

    def fmt_pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:.1%}"

    def fmt_duration(seconds: float) -> str:
        """Write seconds in the unit a person reads them in: 45 s, 12m 05s, 3h 04m."""
        if pd.isna(seconds):
            return "n/a"
        seconds = float(seconds)
        if seconds < 60:  # noqa: PLR2004 - a minute
            return f"{seconds:,.1f} s"
        if seconds < 3600:  # noqa: PLR2004 - an hour
            return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"

    def version_key(text: str) -> tuple[int, ...]:
        """Sort key so '0.10.2' orders above '0.9.0' (string sort would not)."""
        return tuple(int(part) for part in re.findall(r"\d+", str(text))[:4]) or (0,)

    def ratio(numerator: float, denominator: float) -> float:
        """Divide safely, yielding NaN (rendered "n/a") instead of exploding on zero."""
        return numerator / denominator if denominator else float("nan")

    return fmt_duration, fmt_int, fmt_num, fmt_pct, ratio, version_key


@app.cell
def _(pd):
    def total(frame: pd.DataFrame, column: str | None = None) -> float:
        """Weighted count of *frame*'s events, or weighted sum of *column*."""
        if frame.empty:
            return 0.0
        values = frame["weight"] if column is None else frame[column].fillna(0) * frame["weight"]
        return float(values.sum())

    def quantile(frame: pd.DataFrame, column: str, q: float = 0.5) -> float:
        """Weighted quantile of *column*, ignoring missing values."""
        data = frame[[column, "weight"]].dropna()
        if data.empty:
            return float("nan")
        data = data.sort_values(column)
        cumulative = data["weight"].cumsum()
        return float(data.loc[cumulative >= q * data["weight"].sum(), column].iloc[0])

    def counts(frame: pd.DataFrame, column: str, *, top: int | None = None) -> pd.DataFrame:
        """Weighted events per value of *column*, blanks left out, largest first."""
        data = frame[frame[column].astype(str) != ""]
        if data.empty:
            return pd.DataFrame(columns=[column, "events", "share", "label"])
        result = data.groupby(column, as_index=False)["weight"].sum()
        result = result.rename(columns={"weight": "events"}).sort_values("events", ascending=False)
        result["share"] = result["events"] / result["events"].sum()
        result["label"] = result.apply(lambda r: f"{r['events']:,.0f} ({r['share']:.0%})", axis=1)
        return result.head(top) if top else result

    def exploded(frame: pd.DataFrame, column: str, *, of: float | None = None) -> pd.DataFrame:
        """Split a comma-joined *column* into one row per name: how many events carry each."""
        data = frame[frame[column].astype(str) != ""][[column, "weight"]]
        whole = of if of is not None else total(frame)
        if data.empty or not whole:
            return pd.DataFrame(columns=["name", "events", "share", "label"])
        data = data.assign(name=data[column].str.split(",")).explode("name")
        result = data.groupby("name", as_index=False)["weight"].sum()
        result = result.rename(columns={"weight": "events"}).sort_values("events", ascending=False)
        result["share"] = result["events"] / whole
        result["label"] = result["share"].map(lambda share: f"{share:.0%}")
        return result

    def filled(
        series: pd.Series,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str = "D",
        axis: str = "day",
    ) -> pd.DataFrame:
        """
        Spread *series* over every step from *start* to *end*, with zero where it has no value.

        A line drawn over only the days that had events would slope across the empty days
        between them, as if something happened there.
        """
        steps = pd.date_range(start, end, freq=freq)
        return series.reindex(steps, fill_value=0).rename_axis(axis).reset_index()

    return counts, exploded, filled, quantile, total


@app.cell
def _(COLORS, alt, mo, pd):
    def _field(name: str) -> str:
        """Escape a column name for Vega-Lite, which reads "settings.endpoint" as a nested path."""
        return name.replace(".", "\\.")

    def _themed(chart, title: str, subtitle: str):
        return (
            chart.properties(
                width="container",
                title=alt.TitleParams(
                    title,
                    # None fails Altair's schema validation; Undefined omits the subtitle instead.
                    subtitle=subtitle or alt.Undefined,
                    anchor="start",
                    color=COLORS["ink"],
                    subtitleColor=COLORS["muted"],
                    fontSize=13,
                    fontWeight=600,
                    subtitleFontSize=11,
                    offset=14,
                ),
                padding={"left": 4, "right": 56, "top": 4, "bottom": 4},
            )
            .configure(background="transparent", font="system-ui, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(
                gridColor=COLORS["grid"],
                gridWidth=1,
                domainColor=COLORS["baseline"],
                tickColor=COLORS["baseline"],
                labelColor=COLORS["muted"],
                titleColor=COLORS["muted"],
                labelFontSize=11,
            )
            .configure_legend(labelColor=COLORS["ink2"], titleColor=COLORS["muted"], orient="top")
        )

    def hbar(
        frame: pd.DataFrame,
        cat: str,
        val: str,
        *,
        title: str,
        subtitle: str = "",
        label: str | None = None,
        tooltips: list | None = None,
        color: str | None = None,
        fmt: str = ",.0f",
    ):
        """
        Ranked horizontal bars in a single hue, magnitude read by length.

        Values are labeled at the tips, so the value axis and its grid can go; each bar has a
        4px rounded data-end and stays square at the baseline.
        """
        cat, val = _field(cat), _field(val)
        y = alt.Y(
            f"{cat}:N",
            sort=None,  # Keep the frame's order; the caller already ranked it.
            title=None,
            axis=alt.Axis(ticks=False, domain=False, labelLimit=240, labelPadding=8),
        )
        x = alt.X(f"{val}:Q", axis=None)
        bars = (
            alt.Chart(frame)
            .mark_bar(size=18, cornerRadiusEnd=4, color=color or COLORS["hue"])
            .encode(
                y=y,
                x=x,
                tooltip=tooltips or [alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=fmt)],
            )
        )
        # A pre-formatted label column rides as-is; otherwise format the raw value at the tip.
        text = alt.Text(f"{_field(label)}:N") if label else alt.Text(f"{val}:Q", format=fmt)
        tips = (
            alt.Chart(frame)
            .mark_text(align="left", dx=6, color=COLORS["ink2"], fontSize=11)
            .encode(y=y, x=x, text=text)
        )
        return _themed((bars + tips).properties(height=alt.Step(28)), title, subtitle)

    def trend_line(
        frame: pd.DataFrame,
        val: str,
        *,
        title: str,
        subtitle: str = "",
        val_title: str | None = None,
        color: str | None = None,
        x_field: str = "day",
    ):
        """
        Single-series trend as a 2px line over a 10% area wash.

        A hover tooltip and end-dot mark the reading, and the last value is labeled (no legend,
        since the title already names the one series).
        """
        hue = color or COLORS["hue"]
        x = alt.X(f"{x_field}:T", title=None, axis=alt.Axis(grid=False, format="%b %d"))
        y = alt.Y(
            f"{val}:Q",
            title=None,
            axis=alt.Axis(domain=False, ticks=False, tickCount=4, tickMinStep=1, format=",d"),
        )
        base = alt.Chart(frame).encode(x=x, y=y)
        tooltip = [
            alt.Tooltip(f"{x_field}:T", title="Date", format="%b %d, %Y"),
            alt.Tooltip(f"{val}:Q", title=val_title or val, format=",.0f"),
        ]
        hover = alt.selection_point(fields=[x_field], nearest=True, on="pointerover", empty=False)
        line = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round", color=hue)
        wash = base.mark_area(opacity=0.1, color=hue)
        # Invisible wide targets carry the hover selection and the tooltip, so the reader never
        # has to land on the 2px line itself.
        targets = base.mark_point(size=400, opacity=0).encode(tooltip=tooltip).add_params(hover)
        dot = base.mark_point(
            filled=True,
            size=80,
            color=hue,
            stroke=COLORS["surface"],
            strokeWidth=2,
        ).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        last = frame.tail(1)
        end_dot = (
            alt.Chart(last)
            .mark_point(filled=True, size=70, color=hue, stroke=COLORS["surface"], strokeWidth=2)
            .encode(x=x, y=y)
        )
        end_label = (
            alt.Chart(last)
            .mark_text(align="left", dx=8, fontWeight=600, color=COLORS["ink2"], fontSize=11)
            .encode(x=x, y=y, text=alt.Text(f"{val}:Q", format=",.0f"))
        )
        chart = (wash + line + targets + dot + end_dot + end_label).properties(height=240)
        return _themed(chart, title, subtitle)

    def heatmap(
        frame: pd.DataFrame,
        x: str,
        y: str,
        val: str,
        *,
        title: str,
        subtitle: str = "",
        x_title: str = "",
        x_domain: list | None = None,
        y_sort: list | None = None,
    ):
        """Draw a grid of cells in one sequential hue, each cell with its own tooltip."""
        chart = (
            alt.Chart(frame)
            .mark_rect(cornerRadius=2, stroke=COLORS["surface"], strokeWidth=2)
            .encode(
                x=alt.X(
                    f"{x}:O",
                    title=x_title or None,
                    # A fixed domain keeps the empty columns, so a gap reads as a gap.
                    scale=alt.Scale(domain=x_domain) if x_domain else alt.Undefined,
                    axis=alt.Axis(labelAngle=0, ticks=False),
                ),
                y=alt.Y(f"{y}:O", title=None, sort=y_sort, axis=alt.Axis(ticks=False)),
                color=alt.Color(
                    f"{val}:Q",
                    scale=alt.Scale(range=[COLORS["seq_low"], COLORS["seq_high"]]),
                    legend=alt.Legend(title=None, orient="right", gradientLength=120),
                ),
                tooltip=[
                    alt.Tooltip(f"{y}:O"),
                    alt.Tooltip(f"{x}:O"),
                    alt.Tooltip(f"{val}:Q", format=",.0f"),
                ],
            )
            .properties(height=alt.Step(26))
        )
        return _themed(chart, title, subtitle)

    def stacked_share(frame: pd.DataFrame, *, title: str, subtitle: str = ""):
        """
        One horizontal bar split into named parts, for a part-to-whole reading.

        *frame* has part, share, and color columns in reading order. A legend names the parts,
        a 2px surface gap separates them, and a part wide enough for its label carries it.
        """
        data = frame.assign(order=range(len(frame)), row="all")
        data["end"] = data["share"].cumsum()
        data["start"] = data["end"] - data["share"]
        data["mid"] = (data["start"] + data["end"]) / 2
        data["label"] = data["share"].map(lambda share: f"{share:.0%}")
        # White reads on the saturated hues; the pale rest color needs the ink instead.
        on_rest = data["color"] == COLORS["rest"]
        data["ink"] = on_rest.map({True: COLORS["ink"], False: "#fff"})
        color = alt.Color(
            "part:N",
            scale=alt.Scale(domain=list(data["part"]), range=list(data["color"])),
            legend=alt.Legend(title=None, orient="top", symbolType="square"),
        )
        bars = (
            alt.Chart(data)
            .mark_bar(size=28, stroke=COLORS["surface"], strokeWidth=2)
            .encode(
                x=alt.X("start:Q", axis=alt.Axis(format="%", tickCount=5, grid=False), title=None),
                x2="end:Q",
                y=alt.Y("row:N", axis=None),
                color=color,
                tooltip=[alt.Tooltip("part:N"), alt.Tooltip("share:Q", format=".1%")],
            )
        )
        # Label only the parts wide enough to hold it; the tooltip and the table keep the rest.
        labels = (
            alt.Chart(data[data["share"] >= 0.08])  # noqa: PLR2004 - room for "99%"
            .mark_text(fontSize=11, fontWeight=600)
            .encode(
                x="mid:Q",
                y=alt.Y("row:N", axis=None),
                text="label:N",
                color=alt.Color("ink:N", scale=None),
            )
        )
        return _themed((bars + labels).properties(height=60), title, subtitle)

    def chart_card(chart, frame: pd.DataFrame):
        """Pair every chart with its table twin, so no value is gated behind color or hover."""
        return mo.ui.tabs({"Chart": chart, "Table": mo.ui.table(frame, page_size=10)})

    def empty(message: str):
        """Show a placeholder when the filters leave a view nothing to draw."""
        return mo.md(f"_{message}_")

    def pair(left, right):
        """Two views side by side, wrapping on a narrow screen."""
        return mo.hstack([left, right], widths="equal", gap=1, wrap=True)

    return chart_card, empty, hbar, heatmap, pair, stacked_share, trend_line


@app.cell
def _(crashes, ev, fmt_duration, fmt_int, mo, runs, sessions, quantile, total):
    # A GUI "run" before v3 was the whole window and counts as a session now, so every run left is
    # one batch of work, timed from its start to its end: a CLI batch or a GUI generation.
    _cli = runs[runs["interface"] == "cli"]
    _gui = runs[runs["interface"] == "gui"]
    _gui_sessions = sessions[sessions["interface"] == "gui"]
    # A run answered from the cache makes no model call, so it has no model time to report.
    _inferred = runs[runs["photo_seconds_median"] > 0]
    mo.hstack(
        [
            mo.stat(
                fmt_int(ev["install_id"].nunique()),
                label="Active installs",
                caption="distinct install ids in view",
                bordered=True,
            ),
            mo.stat(
                fmt_int(total(runs)),
                label="Runs",
                caption=f"{fmt_int(total(runs, 'batch_size'))} photos",
                bordered=True,
            ),
            mo.stat(
                fmt_duration(quantile(runs, "duration_seconds")),
                label="Median run duration",
                caption=(
                    f"CLI {fmt_duration(quantile(_cli, 'duration_seconds'))} · "
                    f"GUI {fmt_duration(quantile(_gui, 'duration_seconds'))}"
                ),
                bordered=True,
            ),
            mo.stat(
                fmt_duration(quantile(_inferred, "photo_seconds_median")),
                label="Model time per photo",
                caption="median across runs (v3 on)",
                bordered=True,
            ),
            mo.stat(
                fmt_duration(quantile(_gui_sessions, "session_seconds")),
                label="Median GUI session",
                caption=f"{fmt_int(total(_gui_sessions))} sessions",
                bordered=True,
            ),
            mo.stat(
                fmt_int(total(crashes)),
                label="Crashes",
                caption=f"{fmt_int(crashes['install_id'].nunique())} installs affected",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
        wrap=True,
    )


@app.cell
def _(fmt_pct, mo, ratio, runs, total):
    _real = runs[runs["dry_run"].fillna(0) == 0]
    _ok = total(_real, "success_count")
    _failed = total(_real, "failure_count")
    _cli = runs[runs["interface"] == "cli"]
    mo.hstack(
        [
            mo.stat(
                fmt_pct(ratio(_ok, _ok + _failed)),
                label="Photo success rate",
                caption="photos with no final failure, dry runs left out",
                bordered=True,
            ),
            mo.stat(
                fmt_pct(ratio(total(_real, "cache_hits"), total(_real, "batch_size"))),
                label="Cache hit rate",
                caption="photos answered without a model call",
                bordered=True,
            ),
            mo.stat(
                fmt_pct(ratio(total(_cli[_cli["dry_run"] == 1]), total(_cli))),
                label="Dry runs",
                caption="CLI runs that wrote nothing on purpose",
                bordered=True,
            ),
            mo.stat(
                fmt_pct(ratio(total(runs[runs["outcome"] == "setup-failed"]), total(runs))),
                label="Runs that never started",
                caption="stopped before the first photo",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
        wrap=True,
    )


@app.cell
def _(mo):
    mo.md("## Adoption")


@app.cell
def _(chart_card, empty, ev, filled, mo, trend_line):
    mo.stop(ev.empty, empty("No events in this window yet."))
    _daily = filled(
        ev.groupby("day")["install_id"].nunique().rename("active_installs"),
        start=ev["day"].min(),
        end=ev["day"].max(),
    )
    _first = (
        ev.groupby("install_id")["ts"].min().dt.tz_convert(None).dt.to_period("W").dt.start_time
    )
    _new = _first.value_counts().rename("new_installs")
    _weekly = filled(_new, start=_new.index.min(), end=_new.index.max(), freq="7D", axis="week")
    mo.hstack(
        [
            chart_card(
                trend_line(
                    _daily,
                    "active_installs",
                    title="Active installs per day",
                    subtitle="distinct install ids seen each day",
                    val_title="Active installs",
                ),
                _daily,
            ),
            chart_card(
                trend_line(
                    _weekly,
                    "new_installs",
                    title="New installs per week",
                    subtitle="first seen in the window that week",
                    val_title="New installs",
                    x_field="week",
                ),
                _weekly,
            ),
        ],
        widths="equal",
        gap=1,
        wrap=True,
    )


@app.cell
def _(chart_card, counts, empty, ev, hbar, mo, pair, pd, version_key):
    mo.stop(ev.empty, empty("No events in this window yet."))
    _versions = counts(ev, "app_version", top=12)
    _versions = _versions.sort_values(
        "app_version",
        key=lambda s: s.map(version_key),
        ascending=False,
    )
    _days = ev.groupby("install_id")["day"].nunique()
    _buckets = pd.cut(
        _days,
        [0, 1, 3, 7, 30, float("inf")],
        labels=["1 day", "2-3 days", "4-7 days", "8-30 days", "31+ days"],
    )
    _active = _buckets.value_counts(sort=False).reset_index()
    _active.columns = ["active", "installs"]
    _active["active"] = _active["active"].astype(str)
    pair(
        chart_card(
            hbar(
                _versions,
                "app_version",
                "events",
                title="photo-tagger versions",
                subtitle="newest first: is the latest release being picked up?",
                label="label",
            ),
            _versions,
        ),
        chart_card(
            hbar(
                _active,
                "active",
                "installs",
                title="How many days each install was used",
                subtitle="installs by active days in the window: one-off tries vs regulars",
            ),
            _active,
        ),
    )


@app.cell
def _(chart_card, counts, empty, ev, hbar, mo, pair):
    mo.stop(ev.empty, empty("No events in this window yet."))
    _latest = ev.sort_values("ts").groupby("install_id").tail(1)
    _methods = counts(_latest[_latest["install_method"] != ""], "install_method")
    _consent = counts(ev[ev["consent"] != ""], "consent")
    pair(
        chart_card(
            hbar(
                _methods,
                "install_method",
                "events",
                title="How photo-tagger was installed",
                subtitle="installs, by their latest event (v3 only)",
                label="label",
            ),
            _methods,
        )
        if not _methods.empty
        else empty("No install methods yet (schema v3 only)."),
        chart_card(
            hbar(
                _consent,
                "consent",
                "events",
                title="Why beacons are allowed",
                subtitle="default, or an explicit opt-in that overrules DO_NOT_TRACK",
                label="label",
            ),
            _consent,
        )
        if not _consent.empty
        else empty("No consent reasons yet (schema v3 only)."),
    )


@app.cell
def _(mo):
    mo.md("## Usage")


@app.cell
def _(chart_card, commands, counts, empty, ev, hbar, mo, pair, pd, runs, sessions):
    mo.stop(ev.empty, empty("No events in this window yet."))
    # One entry per invocation: a tag run, a watch session, a GUI window, or another command.
    _cli_runs = runs[(runs["interface"] == "cli") & (runs["command"] == "tag")]
    _invocations = counts(pd.concat([_cli_runs, sessions, commands]), "command")
    _interfaces = counts(runs, "interface")
    _interfaces["interface"] = _interfaces["interface"].str.upper()
    pair(
        chart_card(
            hbar(
                _invocations,
                "command",
                "events",
                title="Commands",
                subtitle="invocations: tag runs, watch sessions, GUI windows, other commands",
                label="label",
            ),
            _invocations,
        ),
        chart_card(
            hbar(
                _interfaces,
                "interface",
                "events",
                title="CLI vs GUI",
                subtitle="runs per interface",
                label="label",
            ),
            _interfaces,
        ),
    )


@app.cell
def _(chart_card, counts, empty, hbar, mo, pair, runs):
    mo.stop(runs.empty, empty("No runs in this window."))
    _models = counts(runs, "model", top=12)
    _providers = counts(runs, "provider")
    pair(
        chart_card(
            hbar(
                _models,
                "model",
                "events",
                title="Most-used models",
                subtitle="top 12 by runs",
                label="label",
            ),
            _models,
        ),
        chart_card(
            hbar(_providers, "provider", "events", title="Providers", label="label"),
            _providers,
        ),
    )


@app.cell
def _(chart_card, counts, empty, exploded, hbar, mo, pair, pd, runs, v3_runs):
    mo.stop(runs.empty, empty("No runs in this window."))
    _endpoints = counts(v3_runs, "settings.endpoint") if "settings.endpoint" in v3_runs else None
    _sizes = runs[runs["batch_size"] > 0]
    _batch = (
        _sizes.assign(
            bucket=pd.cut(
                _sizes["batch_size"],
                [0, 1, 5, 20, 100, 1000, float("inf")],
                labels=["1 photo", "2-5", "6-20", "21-100", "101-1000", "1000+"],
            ),
        )
        .groupby("bucket", as_index=False, observed=True)["weight"]
        .sum()
        .rename(columns={"weight": "runs"})
    )
    _batch["bucket"] = _batch["bucket"].astype(str)
    _formats = exploded(runs, "file_types")
    pair(
        chart_card(
            hbar(
                _endpoints,
                "settings.endpoint",
                "events",
                title="Where the model server runs",
                subtitle="localhost, your network, a tailnet, or a public service (never the URL)",
                label="label",
            ),
            _endpoints,
        )
        if _endpoints is not None and not _endpoints.empty
        else empty("No endpoint data yet (schema v3 only)."),
        mo.vstack(
            [
                chart_card(
                    hbar(
                        _batch,
                        "bucket",
                        "runs",
                        title="Batch sizes",
                        subtitle="photos per run, small to large",
                    ),
                    _batch,
                ),
                chart_card(
                    hbar(
                        _formats,
                        "name",
                        "share",
                        title="File formats",
                        subtitle="share of runs that included each format",
                        label="label",
                        fmt=".0%",
                    ),
                    _formats,
                ),
            ],
        ),
    )


@app.cell
def _(chart_card, empty, heatmap, mo, v3_runs):
    mo.stop(v3_runs.empty, empty("No v3 runs to place in the week yet."))
    _week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _cells = (
        v3_runs.dropna(subset=["local_hour", "local_weekday"])
        .assign(
            day=lambda f: f["local_weekday"].astype(int).map(dict(enumerate(_week))),
            hour=lambda f: f["local_hour"].astype(int),
        )
        .groupby(["day", "hour"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "runs"})
    )
    chart_card(
        heatmap(
            _cells,
            "hour",
            "day",
            "runs",
            title="When runs happen",
            subtitle="runs by the user's local weekday and hour: are overnight batches common?",
            x_title="local hour",
            x_domain=list(range(24)),
            y_sort=_week,
        ),
        _cells,
    )


@app.cell
def _(mo):
    mo.md("## Sessions and time")


@app.cell
def _(
    COLORS,
    chart_card,
    empty,
    fmt_duration,
    fmt_pct,
    gui_sessions,
    mo,
    pd,
    ratio,
    stacked_share,
    quantile,
    total,
):
    mo.stop(gui_sessions.empty, empty("No GUI sessions from v3 installs in this window yet."))
    _total = total(gui_sessions, "session_seconds")
    _busy = total(gui_sessions, "busy_seconds")
    _engaged = total(gui_sessions, "engaged_seconds")
    _unattended = total(gui_sessions, "unattended_seconds")
    _watched = max(0.0, _busy - _unattended)
    _reviewing = max(0.0, _engaged - _watched)
    _idle = max(0.0, _total - _busy - _reviewing)
    _split = pd.DataFrame(
        {
            "part": ["Run, watched", "Run, unattended", "In use, no run", "Idle"],
            "seconds": [_watched, _unattended, _reviewing, _idle],
            "color": [COLORS["hue"], COLORS["slot2"], COLORS["slot3"], COLORS["rest"]],
        },
    )
    _split["share"] = _split["seconds"] / _split["seconds"].sum() if _total else 0.0
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        fmt_duration(quantile(gui_sessions, "session_seconds")),
                        label="Median session",
                        caption="window open to close",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_pct(ratio(_busy, _total)),
                        label="Busy",
                        caption="share of window time with a run in flight",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_pct(ratio(_idle, _total)),
                        label="Idle",
                        caption="no run and no input for a minute or more",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_duration(
                            quantile(
                                gui_sessions[gui_sessions["first_run_seconds"] >= 0],
                                "first_run_seconds",
                            ),
                        ),
                        label="Time to the first run",
                        caption="median, sessions that ran something",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1,
                wrap=True,
            ),
            chart_card(
                stacked_share(
                    _split[["part", "share", "color"]],
                    title="Where GUI window time goes",
                    subtitle="all sessions together; in use means a click or key press "
                    "in the last minute",
                ),
                _split.drop(columns="color").assign(
                    time=_split["seconds"].map(fmt_duration),
                ),
            ),
        ],
        gap=1,
    )


@app.cell
def _(chart_card, empty, gui_sessions, hbar, mo, pair, pd):
    mo.stop(gui_sessions.empty, empty("No GUI sessions from v3 installs in this window yet."))
    _runs = (
        gui_sessions.assign(
            bucket=pd.cut(
                gui_sessions["runs"].fillna(0),
                [-1, 0, 1, 3, 10, float("inf")],
                labels=["no run", "1 run", "2-3 runs", "4-10 runs", "11+ runs"],
            ),
        )
        .groupby("bucket", as_index=False, observed=True)["weight"]
        .sum()
        .rename(columns={"weight": "sessions"})
    )
    _runs["bucket"] = _runs["bucket"].astype(str)
    _length = (
        gui_sessions.assign(
            bucket=pd.cut(
                gui_sessions["session_seconds"].fillna(0),
                [-1, 300, 1800, 3600, 4 * 3600, 12 * 3600, float("inf")],
                labels=["< 5 min", "5-30 min", "30-60 min", "1-4 h", "4-12 h", "12 h+"],
            ),
        )
        .groupby("bucket", as_index=False, observed=True)["weight"]
        .sum()
        .rename(columns={"weight": "sessions"})
    )
    _length["bucket"] = _length["bucket"].astype(str)
    pair(
        chart_card(
            hbar(
                _runs,
                "bucket",
                "sessions",
                title="Runs per GUI session",
                subtitle="a window opened only to look, or a working session",
            ),
            _runs,
        ),
        chart_card(
            hbar(
                _length,
                "bucket",
                "sessions",
                title="GUI session length",
                subtitle="window open to close, idle time included",
            ),
            _length,
        ),
    )


@app.cell
def _(mo):
    mo.md("## GUI workflows")


@app.cell
def _(
    chart_card,
    empty,
    fmt_duration,
    fmt_num,
    fmt_pct,
    gui_runs,
    gui_sessions,
    hbar,
    mo,
    pair,
    pd,
    ratio,
    save_runs,
    quantile,
    total,
):
    mo.stop(
        gui_runs.empty and gui_sessions.empty,
        empty("No GUI runs or sessions from v3 installs in this window yet."),
    )
    # A generation saves by itself when Save After Generating is on or a watched folder started it;
    # gui.saved counts the photos that chained save wrote.
    _chained = gui_runs[gui_runs["gui.autosave"] == 1]
    _watched = _chained["trigger"] == "watch"
    _paths = pd.DataFrame(
        {
            "path": [
                "Generate and Save (one click)",
                "Save Selected, after a review",
                "Saved one photo at a time",
                "Watched folder, saved by itself",
            ],
            "photos": [
                total(_chained[~_watched], "gui.saved"),
                total(save_runs, "success_count"),
                total(gui_sessions, "single_saves"),
                total(_chained[_watched], "gui.saved"),
            ],
        },
    )
    _written = _paths["photos"].sum()
    _paths["share"] = _paths["photos"] / _written if _written else 0.0
    _paths["label"] = _paths.apply(lambda r: f"{r['photos']:,.0f} ({r['share']:.0%})", axis=1)
    _saved = total(gui_sessions, "review.saved")
    _edited = total(gui_sessions, "review.edited")
    _edits = pd.DataFrame(
        {
            "field": ["Any field", "Title", "Description", "Keywords"],
            "photos": [
                _edited,
                total(gui_sessions, "review.edited_title"),
                total(gui_sessions, "review.edited_description"),
                total(gui_sessions, "review.edited_keywords"),
            ],
        },
    )
    _edits["share"] = _edits["photos"] / _saved if _saved else 0.0
    _edits["label"] = _edits["share"].map(lambda share: f"{share:.0%}")
    _by_hand = gui_runs[gui_runs["trigger"] != "watch"]
    _hinted = ratio(total(gui_runs, "gui.hinted"), total(gui_runs, "batch_size"))
    _added = ratio(total(gui_sessions, "review.keywords_added"), _edited)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        fmt_pct(ratio(total(_chained[~_watched]), total(_by_hand))),
                        label="Generate and Save",
                        caption="generations that saved by themselves (watched folders left out)",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_pct(ratio(_edited, _saved)),
                        label="Edited before saving",
                        caption="saved photos changed by hand",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_pct(_hinted),
                        label="Generated with a hint",
                        caption="photos with text in the hint box",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_duration(quantile(gui_sessions, "review.review_seconds_median")),
                        label="Time from proposal to save",
                        caption="median of the sessions' medians",
                        bordered=True,
                    ),
                    mo.stat(
                        fmt_num(ratio(total(gui_sessions, "review.keywords_removed"), _edited)),
                        label="Keywords removed per edited photo",
                        caption=f"{fmt_num(_added)} added",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1,
                wrap=True,
            ),
            pair(
                chart_card(
                    hbar(
                        _paths,
                        "path",
                        "photos",
                        title="How photos get saved",
                        subtitle="photos written by each path",
                        label="label",
                    ),
                    _paths,
                ),
                chart_card(
                    hbar(
                        _edits,
                        "field",
                        "share",
                        title="What gets edited before saving",
                        subtitle="share of saved photos with a manual change",
                        label="label",
                        fmt=".0%",
                    ),
                    _edits,
                ),
            ),
        ],
        gap=1,
    )


@app.cell
def _(chart_card, counts, empty, gui_runs, gui_sessions, hbar, mo, pair, pd, total):
    mo.stop(gui_runs.empty, empty("No GUI runs from v3 installs in this window yet."))
    _triggers = counts(gui_runs, "trigger")
    # In the order a photo meets them: the repair call inside the run, the automatic pass at its
    # end, what was still mixed after both, and then what the person did about it.
    _script = pd.DataFrame(
        {
            "step": [
                "Fixed by the language repair call",
                "Asked again by the automatic pass",
                "Still mixed when the run ended",
                "Asked again by hand",
                "Saved while still mixed",
            ],
            "photos": [
                total(gui_runs, "language_repairs"),
                total(gui_runs, "gui.mixed_pass"),
                total(gui_runs, "flagged"),
                total(gui_runs, "gui.regenerated_flagged"),
                total(gui_sessions, "review.flagged"),
            ],
        },
    )
    pair(
        chart_card(
            hbar(
                _triggers,
                "trigger",
                "events",
                title="What starts a GUI generation",
                subtitle="batch, one photo, Enter in the hint box, Retry Failed, a watched folder",
                label="label",
            ),
            _triggers,
        ),
        chart_card(
            hbar(
                _script,
                "step",
                "photos",
                title="Mixed writing systems: repaired, asked again, or saved as is",
                subtitle="GUI photos at each step",
            ),
            _script,
        ),
    )


@app.cell
def _(chart_card, empty, gui_sessions, hbar, mo, pd):
    _columns = [column for column in gui_sessions.columns if column.startswith("features.")]
    mo.stop(
        gui_sessions.empty or not _columns,
        empty("No feature counts from v3 GUI sessions in this window yet."),
    )
    _total = gui_sessions["weight"].sum()
    _features = pd.DataFrame(
        {
            "feature": [column.removeprefix("features.") for column in _columns],
            "sessions": [
                float(gui_sessions.loc[gui_sessions[column].fillna(0) > 0, "weight"].sum())
                for column in _columns
            ],
        },
    )
    _features["share"] = _features["sessions"] / _total if _total else 0.0
    _features["label"] = _features["share"].map(lambda share: f"{share:.0%}")
    _features = _features.sort_values("share", ascending=False).head(30)
    chart_card(
        hbar(
            _features,
            "feature",
            "share",
            title="GUI features",
            subtitle="share of sessions that used each one at least once",
            label="label",
            fmt=".0%",
        ),
        _features,
    )


@app.cell
def _(mo):
    mo.md("## Setup and configuration")


@app.cell
def _(chart_card, counts, empty, exploded, hbar, mo, pair, v3_runs, total):
    mo.stop(v3_runs.empty, empty("No v3 runs with settings in this window yet."))
    _changed = exploded(v3_runs, "setup.changed", of=total(v3_runs)).head(25)
    _latest = v3_runs.sort_values("ts").groupby("install_id").tail(1)
    _files = counts(_latest, "setup.config_file")
    pair(
        chart_card(
            hbar(
                _changed,
                "name",
                "share",
                title="Settings changed from the defaults",
                subtitle="share of runs with each setting off its built-in default",
                label="label",
                fmt=".0%",
            ),
            _changed,
        )
        if not _changed.empty
        else empty("Every run in view used the defaults."),
        chart_card(
            hbar(
                _files,
                "setup.config_file",
                "events",
                title="Config file",
                subtitle="installs by where their config came from (none, env, cwd, user)",
                label="label",
            ),
            _files,
        ),
    )


@app.cell
def _(chart_card, empty, exploded, hbar, mo, pair, v3_runs, total):
    mo.stop(v3_runs.empty, empty("No v3 runs with settings in this window yet."))
    _cli = v3_runs[v3_runs["interface"] == "cli"]
    _flags = exploded(_cli, "setup.cli_flags", of=total(_cli)).head(25)
    _keys = exploded(v3_runs, "setup.config_keys", of=total(v3_runs)).head(25)
    pair(
        chart_card(
            hbar(
                _flags,
                "name",
                "share",
                title="Flags typed on the command line",
                subtitle="share of CLI runs using each flag (names only, never values)",
                label="label",
                fmt=".0%",
            ),
            _flags,
        )
        if not _flags.empty
        else empty("No CLI flags in view."),
        chart_card(
            hbar(
                _keys,
                "name",
                "share",
                title="Keys set in config files",
                subtitle="share of runs whose config file sets each key",
                label="label",
                fmt=".0%",
            ),
            _keys,
        )
        if not _keys.empty
        else empty("No config keys in view."),
    )


@app.cell
def _(mo, v3_runs):
    _settings = sorted(column for column in v3_runs.columns if column.startswith("settings."))
    setting_picker = mo.ui.dropdown(
        options={column.removeprefix("settings."): column for column in _settings},
        value="sidecar_mode" if "settings.sidecar_mode" in _settings else None,
        label="Setting",
        searchable=True,
    )
    return (setting_picker,)


@app.cell
def _(chart_card, counts, empty, hbar, mo, setting_picker, v3_runs):
    mo.stop(
        v3_runs.empty or setting_picker.value is None,
        empty("Pick a setting to see how runs use it (v3 runs only)."),
    )
    _column = setting_picker.value
    _values = counts(
        v3_runs.assign(**{_column: v3_runs[_column].astype(str).replace("nan", "")}),
        _column,
        top=15,
    )
    mo.vstack(
        [
            mo.md("### Setting explorer\nHow the runs in view set any one setting."),
            setting_picker,
            chart_card(
                hbar(
                    _values,
                    _column,
                    "events",
                    title=f"{setting_picker.selected_key}",
                    subtitle="runs per value",
                    label="label",
                ),
                _values,
            ),
        ],
        gap=1,
    )


@app.cell
def _(mo):
    mo.md("## Hardware and performance")


@app.cell
def _(chart_card, counts, empty, ev, hbar, mo, pair):
    mo.stop(ev.empty, empty("No events in this window yet."))
    # Every beacon carries the hardware from schema v2 on; each install counts once, as it was last.
    _latest = ev[ev["cpu"] != ""].sort_values("ts").groupby("install_id").tail(1)
    _cpu_models = counts(_latest, "cpu", top=12)
    _gpu_models = counts(_latest, "gpu", top=12)
    pair(
        chart_card(
            hbar(_cpu_models, "cpu", "events", title="CPUs", subtitle="installs", label="label"),
            _cpu_models,
        )
        if not _cpu_models.empty
        else empty("No CPU data yet (schema v2 on)."),
        chart_card(
            hbar(
                _gpu_models,
                "gpu",
                "events",
                title="GPUs",
                subtitle="installs; inference may run on another machine",
                label="label",
            ),
            _gpu_models,
        )
        if not _gpu_models.empty
        else empty("No GPU data yet (schema v2 on)."),
    )


@app.cell
def _(alt, chart_card, empty, fmt_duration, hbar, mo, pair, pd, runs, quantile):
    mo.stop(runs.empty, empty("No runs in this window."))
    # Built row by row with fixed columns, so a window with no such runs is an empty frame, not
    # an error: runs from before v3 have no per-photo time at all.
    _by_arch = pd.DataFrame(
        [
            {
                "group": f"{_arch} · {_interface.upper() or '?'}",
                "median_seconds": quantile(_group, "duration_seconds"),
                "runs": float(_group["weight"].sum()),
            }
            for (_arch, _interface), _group in runs.groupby(["arch", "interface"])
            if _arch
        ],
        columns=["group", "median_seconds", "runs"],
    ).sort_values("runs", ascending=False)
    _per_photo = (
        pd.DataFrame(
            [
                {"model": _model, "median_seconds": quantile(_group, "photo_seconds_median")}
                for _model, _group in runs[runs["photo_seconds_median"] > 0].groupby("model")
                if _model
            ],
            columns=["model", "median_seconds"],
        )
        .sort_values("median_seconds")
        .head(12)
    )
    _by_arch["label"] = _by_arch["median_seconds"].map(fmt_duration)
    _per_photo["label"] = _per_photo["median_seconds"].map(fmt_duration)
    pair(
        chart_card(
            hbar(
                _by_arch,
                "group",
                "median_seconds",
                title="Median run duration by CPU architecture",
                subtitle="CLI batches and GUI generations, timed from start to end",
                label="label",
                tooltips=[
                    alt.Tooltip("group:N"),
                    alt.Tooltip("median_seconds:Q", title="Median seconds", format=",.1f"),
                    alt.Tooltip("runs:Q", title="Runs", format=",.0f"),
                ],
            ),
            _by_arch,
        )
        if not _by_arch.empty
        else empty("No architecture data."),
        chart_card(
            hbar(
                _per_photo,
                "model",
                "median_seconds",
                title="Model time per photo",
                subtitle="median seconds per photo, fastest first (v3 runs)",
                label="label",
            ),
            _per_photo,
        )
        if not _per_photo.empty
        else empty("No per-photo timing yet (schema v3 only)."),
    )


@app.cell
def _(chart_card, counts, empty, ev, hbar, launches, mo, pair, pd, sessions, v3_runs):
    mo.stop(ev.empty, empty("No events in this window yet."))
    _latest = ev.sort_values("ts").groupby("install_id").tail(1)
    _os = counts(_latest, "os")
    _memory = (
        _latest[_latest["memory_gb"] > 0]
        .groupby("memory_gb", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "installs"})
        .sort_values("memory_gb")
    )
    _memory["ram"] = _memory["memory_gb"].map(lambda gb: f"{gb:,.0f} GB")
    _screens = counts(pd.concat([launches, sessions]), "display.screen", top=10)
    _power = counts(v3_runs, "power")
    pair(
        mo.vstack(
            [
                chart_card(
                    hbar(
                        _os,
                        "os",
                        "events",
                        title="Operating systems",
                        subtitle="installs",
                        label="label",
                    ),
                    _os,
                ),
                chart_card(
                    hbar(
                        _memory,
                        "ram",
                        "installs",
                        title="RAM",
                        subtitle="installs, small to large",
                    ),
                    _memory,
                ),
            ],
        ),
        mo.vstack(
            [
                chart_card(
                    hbar(
                        _screens,
                        "display.screen",
                        "events",
                        title="Screens",
                        subtitle="primary screen in logical pixels and scale (GUI)",
                        label="label",
                    ),
                    _screens,
                )
                if not _screens.empty
                else empty("No screen data yet (GUI, schema v3 only)."),
                chart_card(
                    hbar(
                        _power,
                        "power",
                        "events",
                        title="Power at the end of a run",
                        subtitle="charger, battery, or a machine with no battery",
                        label="label",
                    ),
                    _power,
                )
                if not _power.empty
                else empty("No power data yet (schema v3 only)."),
            ],
        ),
    )


@app.cell
def _(mo):
    mo.md("## Reliability and crashes")


@app.cell
def _(COLORS, chart_card, counts, crashes, ev, filled, hbar, mo, pair, trend_line, version_key):
    mo.stop(crashes.empty, mo.md("_No crashes in this window._").callout(kind="success"))
    _daily = filled(
        crashes.groupby("day")["weight"].sum().rename("crashes"),
        start=ev["day"].min(),
        end=ev["day"].max(),
    )
    _versions = counts(crashes, "app_version", top=12).sort_values(
        "app_version",
        key=lambda s: s.map(version_key),
        ascending=False,
    )
    pair(
        chart_card(
            trend_line(
                _daily,
                "crashes",
                title="Crashes per day",
                subtitle="crash beacons (status color: this is a state, not a series)",
                val_title="Crashes",
                color=COLORS["critical"],
            ),
            _daily,
        ),
        chart_card(
            hbar(
                _versions,
                "app_version",
                "events",
                title="Crashes by app version",
                subtitle="newest first: did the last release make things better or worse?",
                color=COLORS["critical"],
                label="label",
            ),
            _versions,
        ),
    )


@app.cell
def _(pd):
    def tallied(frame: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
        """Sum a "name:count,name:count" *column* over *frame*, each event by its weight."""
        pairs = [
            (label, int(count) * weight)
            for cell, weight in zip(frame[column], frame["weight"], strict=True)
            for label, _, count in (entry.rpartition(":") for entry in str(cell).split(","))
            if label and count.isdigit()
        ]
        return (
            pd.DataFrame(pairs, columns=[name, "photos"])
            .groupby(name, as_index=False)["photos"]
            .sum()
            .sort_values("photos", ascending=False)
        )

    return (tallied,)


@app.cell
def _(COLORS, chart_card, counts, empty, hbar, mo, pair, pd, runs, save_runs, tallied):
    mo.stop(runs.empty, empty("No runs in this window."))
    _every = pd.concat([runs, save_runs])
    _kinds = tallied(_every, "failure_kinds", "kind")
    _errors = tallied(_every, "failure_types", "error")
    _setup = counts(runs[runs["outcome"] == "setup-failed"], "error_kind")
    _none = mo.md("_No per-photo failures in this window._").callout(kind="success")
    mo.vstack(
        [
            pair(
                chart_card(
                    hbar(
                        _kinds,
                        "kind",
                        "photos",
                        title="Why photos fail",
                        subtitle="final failures after the retry pass, by coarse cause",
                        color=COLORS["critical"],
                    ),
                    _kinds,
                )
                if not _kinds.empty
                else _none,
                chart_card(
                    hbar(
                        _errors,
                        "error",
                        "photos",
                        title="The errors behind them",
                        subtitle="error class, and the HTTP status of an HTTP error (v3 on)",
                        color=COLORS["critical"],
                    ),
                    _errors,
                )
                if not _errors.empty
                else empty("No error classes yet (schema v3 only)."),
            ),
            chart_card(
                hbar(
                    _setup,
                    "error_kind",
                    "events",
                    title="Why runs never start",
                    subtitle="the error that stopped a run before its first photo",
                    color=COLORS["critical"],
                    label="label",
                ),
                _setup,
            )
            if not _setup.empty
            else mo.md("_Every run in view got to its first photo._").callout(kind="success"),
        ],
        gap=1,
    )


@app.cell
def _(crashes, empty, mo):
    mo.stop(crashes.empty, empty("No crash signatures in this window."))
    _signatures = (
        crashes.groupby(["exception_type", "crash_location", "app_version", "os"], as_index=False)
        .agg(
            crashes=("weight", "sum"),
            installs=("install_id", "nunique"),
            chain=("exception_chain", "first"),
            stack=("crash_stack", "first"),
            frames=("crash_frames", "first"),
            libraries=("libraries", "first"),
        )
        .sort_values("crashes", ascending=False)
        .head(50)
    )
    mo.vstack(
        [
            mo.md(
                "### Crash signatures\n"
                "The exception type and the deepest photo-tagger frame (`module:function:line`), "
                "with the chain of causes, the whole stack by module (v3 on), and the library "
                "versions. Messages are never collected.",
            ),
            mo.ui.table(_signatures, page_size=10),
        ],
        gap=1,
    )


@app.cell
def _(mo):
    mo.md("## Languages")


@app.cell
def _(chart_card, counts, empty, ev, hbar, mo, pair, runs):
    mo.stop(ev.empty, empty("No events in this window yet."))
    _output = counts(runs, "output_language", top=12)
    _latest = ev.sort_values("ts").groupby("install_id").tail(1)
    _locale = counts(_latest, "system_locale", top=12)
    pair(
        chart_card(
            hbar(
                _output,
                "output_language",
                "events",
                title="Metadata languages",
                subtitle="language the model writes titles and keywords in (runs)",
                label="label",
            ),
            _output,
        ),
        chart_card(
            hbar(
                _locale,
                "system_locale",
                "events",
                title="System locales",
                subtitle="installs by OS language and region (v3 only)",
                label="label",
            ),
            _locale,
        )
        if not _locale.empty
        else empty("No locale data yet (schema v3 only)."),
    )


@app.cell
def _(install_labels, mo):
    install_picker = mo.ui.dropdown(
        options=install_labels,
        label="Install",
        searchable=True,
    )
    return (install_picker,)


@app.cell
def _(ev, install_picker, mo):
    _columns = [
        "ts",
        "event",
        "interface",
        "command",
        "job",
        "trigger",
        "outcome",
        "batch_size",
        "duration_seconds",
        "session_seconds",
        "model",
        "app_version",
    ]
    if install_picker.value:
        _rows = ev[ev["install_id"] == install_picker.value].sort_values("ts", ascending=False)
        _view = mo.ui.table(_rows[_columns], page_size=15)
    else:
        _view = mo.md("_Pick an install to list its events, newest first._")
    mo.vstack(
        [
            mo.md(
                "## Install explorer\n"
                "Every event one install sent in the window, for following a single user's path "
                "(the filters above still apply).",
            ),
            install_picker,
            _view,
        ],
        gap=1,
    )


@app.cell
def _(mo, source):
    _default_d1 = """\
    -- D1 is SQLite: payload is the beacon as JSON, read any field with ->>.
    SELECT payload ->> '$.settings.sidecar_mode' AS sidecar_mode, COUNT(*) AS runs
    FROM events
    WHERE event = 'run'
    GROUP BY sidecar_mode
    ORDER BY runs DESC;
    """
    _default_ae = """\
    -- Column map in queries.sql. Run rows: blob12 IN ('run', ''); blob19 is the beacon as JSON.
    SELECT blob3 AS provider, SUM(_sample_interval) AS runs
    FROM photo_tagger_telemetry
    WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '')
    GROUP BY provider
    ORDER BY runs DESC;
    """
    sql_editor = mo.ui.code_editor(
        value=_default_d1 if source in {"d1", "demo"} else _default_ae,
        language="sql",
    )
    run_sql = mo.ui.run_button(label="Run query")
    mo.vstack(
        [
            mo.md(
                "## SQL console\n"
                + (
                    "Runs against the D1 `events` table (SQLite). Starter statements live in "
                    "`queries.sql`."
                    if source != "ae"
                    else "Anything the [AE SQL dialect]("
                    "https://developers.cloudflare.com/analytics/analytics-engine/sql-reference/"
                    ") supports; `FORMAT JSON` is appended automatically."
                ),
            ),
            sql_editor,
            run_sql,
        ],
        gap=1,
    )
    return run_sql, sql_editor


@app.cell
def _(mo, pd, query_ae, query_d1, re, run_sql, source, sql_editor):
    mo.stop(not run_sql.value, mo.md("_Edit the SQL above and click **Run query**._"))
    mo.stop(source == "demo", mo.md("_The SQL console needs real data: it has none in demo mode._"))
    if source == "d1":
        _rows = query_d1(sql_editor.value.strip().removesuffix(";"))
    else:
        _sql = sql_editor.value.strip().removesuffix(";")
        if not re.search(r"\bFORMAT\b", _sql, flags=re.IGNORECASE):
            _sql += "\nFORMAT JSON"
        _rows = query_ae(_sql)
    _result = pd.DataFrame(_rows)
    mo.stop(_result.empty, mo.md("_The query returned no rows._"))
    mo.ui.table(_result, page_size=15)


@app.cell
def _(mo):
    mo.md(
        """
        ---
        Counts weigh each event by its sample weight (1 unless Analytics Engine sampled). The
        collector stores no IPs and no cookies; crash beacons carry the exception type and in-app
        code location only, never messages. See [README.md](README.md) in this directory and
        [docs/telemetry.md](../docs/telemetry.md) for every field.
        """,
    )


if __name__ == "__main__":
    app.run()
