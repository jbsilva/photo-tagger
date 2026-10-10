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
# cannot resolve them here (same escape hatch as gui.py). Altair 6 is required: 5.x fails to import
# on Python 3.14 (it declares `closed=True` TypedDicts that 3.14's stdlib typing does not accept).
"""
Marimo dashboard over the photo-tagger telemetry dataset.

Reads the Cloudflare Analytics Engine SQL API directly; nothing here can write. Sections: adoption,
usage, performance and hardware, reliability and crashes, languages, plus a free-form SQL console.
Every chart pairs with a table twin, and a window + interface filter scopes everything at once.

Run it (read-only app view, dependencies resolved into an isolated venv):

    uv run poe dashboard

or open it as an editable notebook with `uv run poe dashboard-edit`. Requires
CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (Account Analytics: Read) in the environment; see
telemetry/README.md.
"""

import marimo


__generated_with = "0.23.13"
app = marimo.App(width="full", app_title="photo-tagger telemetry")


@app.cell
def _(mo):
    mo.md(
        """
        # photo-tagger telemetry
        Anonymous usage and crash beacons from the [collector](README.md), one data point per
        event. All totals weight rows by `_sample_interval`, so they stay honest once Analytics
        Engine starts sampling. Fields added in schema v2 (hardware, outcome counters, crashes)
        are empty on rows sent by older installs; those views quietly cover v2 rows only.
        """,
    )


@app.cell
def _():
    import os
    import re
    from contextlib import suppress

    import altair as alt
    import httpx2
    import marimo as mo
    import pandas as pd

    return alt, httpx2, mo, os, pd, re, suppress


@app.cell
def _(mo):
    # Validated categorical/ink palette (the project's reference palette), stepped per theme.
    # Marks wear the hue; text and chrome wear the ink tokens, never the series color. Crash
    # views wear the reserved status hues (critical/good), which clear 3:1 on both surfaces and
    # never impersonate a data series.
    _dark = mo.app_meta().theme == "dark"
    COLORS = {
        "hue": "#3987e5" if _dark else "#2a78d6",
        "critical": "#d03b3b",
        "good": "#0ca30c",
        "surface": "#1a1a19" if _dark else "#fcfcfb",
        "grid": "#2c2c2a" if _dark else "#e1e0d9",
        "baseline": "#383835" if _dark else "#c3c2b7",
        "ink": "#ffffff" if _dark else "#0b0b0b",
        "ink2": "#c3c2b7" if _dark else "#52514e",
        "muted": "#898781",
    }
    return (COLORS,)


@app.cell
def _(mo, os):
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    mo.stop(
        not account_id or not api_token,
        mo.md(
            """
            **Credentials missing.** The dashboard reads the Analytics Engine SQL API, which needs:

            1. `CLOUDFLARE_ACCOUNT_ID`: the account id shown on the Cloudflare dashboard overview.
            2. `CLOUDFLARE_API_TOKEN`: an [API token](https://dash.cloudflare.com/profile/api-tokens)
               with the **Account Analytics: Read** permission and nothing else.

            Export both and restart:

            ```bash
            export CLOUDFLARE_ACCOUNT_ID=...
            export CLOUDFLARE_API_TOKEN=...
            uv run poe dashboard
            ```
            """,
        ).callout(kind="warn"),
    )
    return account_id, api_token


@app.cell
def _(mo):
    window = mo.ui.dropdown(
        options={"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90},
        value="Last 30 days",
        label="Window",
    )
    interface_filter = mo.ui.dropdown(
        options={"All interfaces": "", "CLI only": "cli", "GUI only": "gui"},
        value="All interfaces",
        label="Interface",
    )
    refresh = mo.ui.button(label="Refresh", value=0, on_click=lambda count: count + 1)
    mo.hstack([window, interface_filter, refresh], justify="start", align="center", gap=1)
    return interface_filter, refresh, window


@app.cell
def _(interface_filter, window):
    days = int(window.value)
    # Composable WHERE fragments. Run rows are everything that is not a crash (v1 rows have ''
    # in the event column); v2-only metrics additionally require double1 >= 2.
    _iface = f" AND blob2 = '{interface_filter.value}'" if interface_filter.value else ""
    since = f"timestamp > NOW() - INTERVAL '{days}' DAY"
    runs_where = f"{since} AND blob12 != 'crash'{_iface}"
    runs_v2_where = f"{runs_where} AND double1 >= 2"
    crash_where = f"{since} AND blob12 = 'crash'{_iface}"
    return crash_where, days, runs_v2_where, runs_where, since


@app.cell
def _(account_id, api_token, httpx2, pd, re, refresh, suppress):
    _ = refresh.value  # Depending on the button makes query re-run (and thus re-fetch) on click.

    def query(sql: str) -> pd.DataFrame:
        """Run one statement against the Analytics Engine SQL API and return the rows."""
        # Drop a trailing ';' (the queries.sql samples have one) so the appended clause stays
        # part of the same statement, then ask for JSON unless the caller already chose a format.
        sql = sql.strip().removesuffix(";")
        if not re.search(r"\bFORMAT\b", sql, flags=re.IGNORECASE):
            sql += "\nFORMAT JSON"
        response = httpx2.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/analytics_engine/sql",
            content=sql,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=30.0,
        )
        if response.status_code != httpx2.codes.OK:
            msg = f"AE SQL API returned {response.status_code}: {response.text[:500]}"
            raise RuntimeError(msg)
        frame = pd.DataFrame(response.json().get("data", []))
        for column in frame.columns:
            # Numeric blobs (counts, durations) become numbers; text blobs stay strings.
            with suppress(ValueError, TypeError):
                frame[column] = pd.to_numeric(frame[column])
        return frame

    return (query,)


@app.cell
def _(pd, re):
    def fmt_int(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.0f}"

    def fmt_secs(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:,.1f} s"

    def fmt_pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:.1%}"

    def version_key(text: str) -> tuple[int, ...]:
        """Sort key so '0.10.2' orders above '0.9.0' (string sort would not)."""
        return tuple(int(part) for part in re.findall(r"\d+", str(text))[:4]) or (0,)

    def ratio(numerator: float, denominator: float) -> float:
        """Divide safely, yielding NaN (rendered "n/a") instead of exploding on zero."""
        return numerator / denominator if denominator else float("nan")

    return fmt_int, fmt_pct, fmt_secs, ratio, version_key


@app.cell
def _(COLORS, alt, mo, pd):
    def _themed(chart: alt.LayerChart, title: str, subtitle: str) -> alt.LayerChart:
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
                padding={"left": 4, "right": 48, "top": 4, "bottom": 4},
            )
            .configure(background="transparent", font="system-ui, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(
                gridColor=COLORS["grid"],
                gridWidth=1,
                domainColor=COLORS["baseline"],
                tickColor=COLORS["baseline"],
                labelColor=COLORS["muted"],
                labelFontSize=11,
            )
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
    ) -> alt.LayerChart:
        """
        Ranked horizontal bars in a single hue, magnitude read by length.

        Values are labeled at the tips, so the value axis and its grid can go; each bar has a
        4px rounded data-end and stays square at the baseline.
        """
        y = alt.Y(
            f"{cat}:N",
            sort=None,  # Keep the frame's order; the SQL or the caller already ranked it.
            title=None,
            axis=alt.Axis(ticks=False, domain=False, labelLimit=220, labelPadding=8),
        )
        x = alt.X(f"{val}:Q", axis=None)
        bars = (
            alt.Chart(frame)
            .mark_bar(size=18, cornerRadiusEnd=4, color=color or COLORS["hue"])
            .encode(
                y=y,
                x=x,
                tooltip=tooltips
                or [
                    alt.Tooltip(f"{cat}:N"),
                    alt.Tooltip(f"{val}:Q", format=",.0f"),
                ],
            )
        )
        # A pre-formatted label column rides as-is; otherwise format the raw value at the tip.
        text = alt.Text(f"{label}:N") if label else alt.Text(f"{val}:Q", format=",.0f")
        tips = (
            alt.Chart(frame)
            .mark_text(align="left", dx=6, color=COLORS["ink2"], fontSize=11)
            .encode(y=y, x=x, text=text)
        )
        return _themed((bars + tips).properties(height=alt.Step(30)), title, subtitle)

    def trend_line(
        frame: pd.DataFrame,
        val: str,
        *,
        title: str,
        subtitle: str = "",
        val_title: str | None = None,
        color: str | None = None,
    ) -> alt.LayerChart:
        """
        Single-series daily trend as a 2px line over a 10% area wash.

        A hover tooltip and end-dot mark the reading, and the last value is labeled (no legend,
        since the title already names the one series).
        """
        hue = color or COLORS["hue"]
        x = alt.X("day:T", title=None, axis=alt.Axis(grid=False, format="%b %d"))
        y = alt.Y(
            f"{val}:Q",
            title=None,
            axis=alt.Axis(domain=False, ticks=False, tickCount=4, tickMinStep=1, format=","),
        )
        base = alt.Chart(frame).encode(x=x, y=y)
        tooltip = [
            alt.Tooltip("day:T", title="Day", format="%b %d"),
            alt.Tooltip(f"{val}:Q", title=val_title or val, format=","),
        ]
        hover = alt.selection_point(fields=["day"], nearest=True, on="pointerover", empty=False)
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
            .encode(x=x, y=y, text=alt.Text(f"{val}:Q", format=","))
        )
        chart = (wash + line + targets + dot + end_dot + end_label).properties(height=260)
        return _themed(chart, title, subtitle)

    def chart_card(chart: alt.LayerChart, frame: pd.DataFrame) -> object:
        """Pair every chart with its table twin, so no value is gated behind color or hover."""
        return mo.ui.tabs({"Chart": chart, "Table": mo.ui.table(frame, page_size=10)})

    return chart_card, hbar, trend_line


@app.cell
def _(days, fmt_int, fmt_secs, mo, query, runs_where):
    _totals = query(
        f"""
        SELECT
          COUNT(DISTINCT index1) AS installs,
          SUM(_sample_interval) AS runs,
          quantileWeighted(0.5)(double2, _sample_interval) AS median_batch
        FROM photo_tagger_telemetry
        WHERE {runs_where}
        """,
    ).iloc[0]
    _cli = query(
        f"""
        SELECT quantileWeighted(0.5)(double3, _sample_interval) AS median_seconds
        FROM photo_tagger_telemetry
        WHERE {runs_where} AND blob2 = 'cli'
        """,
    ).iloc[0]
    mo.hstack(
        [
            mo.stat(
                fmt_int(_totals["installs"]),
                label="Active installs",
                caption=f"last {days} days",
                bordered=True,
            ),
            mo.stat(
                fmt_int(_totals["runs"]),
                label="Runs",
                caption=f"weighted, last {days} days",
                bordered=True,
            ),
            mo.stat(
                fmt_int(_totals["median_batch"]),
                label="Median batch size",
                caption="photos per run",
                bordered=True,
            ),
            mo.stat(
                fmt_secs(_cli["median_seconds"]),
                label="Median run duration",
                caption="CLI runs only",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
        wrap=True,
    )


@app.cell
def _(crash_where, days, fmt_int, fmt_pct, mo, pd, query, ratio, runs_v2_where):
    _outcome = query(
        f"""
        SELECT
          SUM(double4 * _sample_interval) AS ok,
          SUM(double5 * _sample_interval) AS failed,
          SUM(double6 * _sample_interval) AS cache_hits,
          SUM(double2 * _sample_interval) AS photos
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND double13 = 0
        """,
    ).iloc[0]
    # SUM over zero crash rows comes back null; a quiet window means 0 crashes, not "n/a".
    _crashes = (
        query(
            f"""
            SELECT SUM(_sample_interval) AS crashes, COUNT(DISTINCT index1) AS affected
            FROM photo_tagger_telemetry
            WHERE {crash_where}
            """,
        )
        .iloc[0]
        .map(lambda v: 0 if pd.isna(v) else v)
    )
    mo.hstack(
        [
            mo.stat(
                fmt_pct(ratio(_outcome["ok"], _outcome["ok"] + _outcome["failed"])),
                label="Photo success rate",
                caption="written / attempted, real writes only",
                bordered=True,
            ),
            mo.stat(
                fmt_pct(ratio(_outcome["cache_hits"], _outcome["photos"])),
                label="Cache hit rate",
                caption="photos answered without a model call",
                bordered=True,
            ),
            mo.stat(
                fmt_int(_crashes["crashes"]),
                label="Crashes",
                caption=f"weighted, last {days} days",
                bordered=True,
            ),
            mo.stat(
                fmt_int(_crashes["affected"]),
                label="Crash-affected installs",
                caption="distinct installs that crashed",
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
def _(chart_card, mo, pd, query, since, trend_line):
    installs_frame = query(
        f"""
        SELECT
          toStartOfInterval(timestamp, INTERVAL '1' DAY) AS day,
          COUNT(DISTINCT index1) AS active_installs
        FROM photo_tagger_telemetry
        WHERE {since}
        GROUP BY day
        ORDER BY day
        """,
    )
    mo.stop(installs_frame.empty, mo.md("_No data points in this window yet._"))
    installs_frame["day"] = pd.to_datetime(installs_frame["day"])
    chart_card(
        trend_line(
            installs_frame,
            "active_installs",
            title="Active installs per day",
            subtitle="distinct install ids seen each day (runs and crashes)",
            val_title="Active installs",
        ),
        installs_frame,
    )


@app.cell
def _(chart_card, days, hbar, mo, query, runs_where, version_key):
    app_version_frame = query(
        f"""
        SELECT blob1 AS app_version, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where}
        GROUP BY app_version
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    # Guard before sorting: an empty AE result has no columns at all, so sort_values would
    # KeyError ahead of the empty-window message.
    mo.stop(app_version_frame.empty, mo.md("_No version data in this window._"))
    app_version_frame = app_version_frame.sort_values(
        "app_version",
        key=lambda s: s.map(version_key),
        ascending=False,
    )
    app_version_view = chart_card(
        hbar(
            app_version_frame,
            "app_version",
            "runs",
            title="photo-tagger versions",
            subtitle=f"newest first, last {days} days: is the latest release being picked up?",
        ),
        app_version_frame,
    )
    return (app_version_view,)


@app.cell
def _(chart_card, hbar, mo, query, runs_where, version_key):
    python_version_frame = query(
        f"""
        SELECT blob8 AS python_version, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where}
        GROUP BY python_version
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    # Guard before sorting; see the app-version cell.
    mo.stop(python_version_frame.empty, mo.md("_No Python version data in this window._"))
    python_version_frame = python_version_frame.sort_values(
        "python_version",
        key=lambda s: s.map(version_key),
        ascending=False,
    )
    python_version_view = chart_card(
        hbar(
            python_version_frame,
            "python_version",
            "runs",
            title="Python versions",
            subtitle="newest first: informs when old interpreters can be dropped",
        ),
        python_version_frame,
    )
    return (python_version_view,)


@app.cell
def _(app_version_view, mo, python_version_view):
    mo.hstack([app_version_view, python_version_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(mo):
    mo.md("## Usage")


@app.cell
def _(chart_card, days, hbar, mo, query, runs_where):
    models_frame = query(
        f"""
        SELECT blob4 AS model, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where}
        GROUP BY model
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(models_frame.empty, mo.md("_No model data in this window._"))
    models_view = chart_card(
        hbar(
            models_frame,
            "model",
            "runs",
            title="Most-used models",
            subtitle=f"top 12 by weighted runs, last {days} days",
        ),
        models_frame,
    )
    return (models_view,)


@app.cell
def _(chart_card, days, hbar, mo, query, runs_where):
    provider_frame = query(
        f"""
        SELECT blob3 AS provider, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where}
        GROUP BY provider
        ORDER BY runs DESC
        """,
    )
    mo.stop(provider_frame.empty, mo.md("_No provider data in this window._"))
    provider_view = chart_card(
        hbar(
            provider_frame,
            "provider",
            "runs",
            title="Providers",
            subtitle=f"weighted runs, last {days} days",
        ),
        provider_frame,
    )
    return (provider_view,)


@app.cell
def _(mo, models_view, provider_view):
    mo.hstack([models_view, provider_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, days, hbar, mo, query, since):
    # Deliberately unfiltered by the interface control: this chart IS the interface split.
    interface_frame = query(
        f"""
        SELECT blob2 AS interface, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {since} AND blob12 != 'crash'
        GROUP BY interface
        ORDER BY runs DESC
        """,
    )
    mo.stop(interface_frame.empty, mo.md("_No interface data in this window._"))
    interface_frame["interface"] = interface_frame["interface"].str.upper()
    _total = interface_frame["runs"].sum()
    interface_frame["share"] = interface_frame["runs"] / _total if _total else 0.0
    interface_frame["label"] = interface_frame.apply(
        lambda row: f"{row['runs']:,.0f} ({row['share']:.0%})",
        axis=1,
    )
    interface_view = chart_card(
        hbar(
            interface_frame,
            "interface",
            "runs",
            title="CLI vs GUI",
            subtitle=f"weighted runs, last {days} days (ignores the interface filter)",
            label="label",
        ),
        interface_frame,
    )
    return (interface_view,)


@app.cell
def _(chart_card, hbar, mo, pd, query, runs_where):
    _sizes_raw = query(
        f"""
        SELECT double2 AS batch_size, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where} AND double2 > 0
        GROUP BY batch_size
        """,
    )
    mo.stop(_sizes_raw.empty, mo.md("_No batch-size data in this window._"))
    _bucket_edges = [0, 1, 5, 20, 100, float("inf")]
    _bucket_names = ["1 photo", "2-5", "6-20", "21-100", "100+"]
    batch_frame = (
        _sizes_raw.assign(
            bucket=pd.cut(_sizes_raw["batch_size"], _bucket_edges, labels=_bucket_names),
        )
        .groupby("bucket", as_index=False, observed=True)["runs"]
        .sum()
    )
    # Ordered small-to-large (an ordinal axis), not ranked by magnitude like the other bars.
    batch_frame["bucket"] = batch_frame["bucket"].astype(str)
    batch_view = chart_card(
        hbar(
            batch_frame,
            "bucket",
            "runs",
            title="Batch sizes",
            subtitle="photos per run, weighted runs per bucket",
        ),
        batch_frame,
    )
    return (batch_view,)


@app.cell
def _(batch_view, interface_view, mo):
    mo.hstack([interface_view, batch_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, hbar, mo, query, runs_where):
    _formats_raw = query(
        f"""
        SELECT blob11 AS file_types, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where} AND blob11 != ''
        GROUP BY file_types
        ORDER BY runs DESC
        """,
    )
    mo.stop(_formats_raw.empty, mo.md("_No file-type data in this window yet._"))
    # blob11 is a per-run set like "cr3,jpg"; AE SQL cannot split it, so explode client-side into
    # one row per format. runs then counts runs that used each format (a CR3+JPEG run hits both).
    formats_frame = (
        _formats_raw.assign(fmt=_formats_raw["file_types"].str.split(","))
        .explode("fmt")
        .groupby("fmt", as_index=False)["runs"]
        .sum()
        .sort_values("runs", ascending=False)
    )
    formats_view = chart_card(
        hbar(
            formats_frame,
            "fmt",
            "runs",
            title="File formats",
            subtitle="runs that included each extension (a CR3+JPEG run counts toward both)",
        ),
        formats_frame,
    )
    return (formats_view,)


@app.cell
def _(chart_card, days, fmt_int, hbar, mo, query, runs_v2_where):
    _tokens_frame = query(
        f"""
        SELECT
          blob4 AS model,
          quantileWeighted(0.5)(double9, _sample_interval) AS median_tokens,
          SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND double9 > 0
        GROUP BY model
        ORDER BY runs DESC
        LIMIT 10
        """,
    )
    mo.stop(_tokens_frame.empty, mo.md("_No token data in this window yet (schema v2 rows only)._"))
    _tokens_frame["label"] = _tokens_frame["median_tokens"].map(fmt_int)
    tokens_view = chart_card(
        hbar(
            _tokens_frame,
            "model",
            "median_tokens",
            title="Token appetite by model",
            subtitle=f"median tokens per run, top models, last {days} days",
            label="label",
        ),
        _tokens_frame,
    )
    return (tokens_view,)


@app.cell
def _(formats_view, mo, tokens_view):
    mo.hstack([formats_view, tokens_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(mo):
    mo.md("## Hardware & performance")


@app.cell
def _(chart_card, hbar, mo, query, runs_v2_where):
    gpu_frame = query(
        f"""
        SELECT blob14 AS gpu, SUM(_sample_interval) AS runs, COUNT(DISTINCT index1) AS installs
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND blob14 != ''
        GROUP BY gpu
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(gpu_frame.empty, mo.md("_No GPU data in this window yet (schema v2 rows only)._"))
    gpu_view = chart_card(
        hbar(
            gpu_frame,
            "gpu",
            "runs",
            title="GPUs",
            subtitle=(
                "client-machine hardware; inference may run on a remote server "
                "(Apple Silicon reports the SoC)"
            ),
        ),
        gpu_frame,
    )
    return (gpu_view,)


@app.cell
def _(chart_card, hbar, mo, query, runs_v2_where):
    cpu_frame = query(
        f"""
        SELECT blob13 AS cpu, SUM(_sample_interval) AS runs, COUNT(DISTINCT index1) AS installs
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND blob13 != ''
        GROUP BY cpu
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(cpu_frame.empty, mo.md("_No CPU data in this window yet (schema v2 rows only)._"))
    cpu_view = chart_card(
        hbar(
            cpu_frame,
            "cpu",
            "runs",
            title="CPUs",
            subtitle="weighted runs per processor model",
        ),
        cpu_frame,
    )
    return (cpu_view,)


@app.cell
def _(cpu_view, gpu_view, mo):
    mo.hstack([cpu_view, gpu_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, hbar, mo, query, runs_v2_where):
    memory_frame = query(
        f"""
        SELECT double12 AS memory_gb, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND double12 > 0
        GROUP BY memory_gb
        ORDER BY memory_gb
        """,
    )
    mo.stop(memory_frame.empty, mo.md("_No RAM data in this window yet (schema v2 rows only)._"))
    # Ordered small-to-large (an ordinal axis), so the RAM ladder reads top-down.
    memory_frame["ram"] = memory_frame["memory_gb"].map(lambda gb: f"{gb:,.0f} GB")
    memory_view = chart_card(
        hbar(
            memory_frame,
            "ram",
            "runs",
            title="RAM",
            subtitle="weighted runs per memory size",
        ),
        memory_frame,
    )
    return (memory_view,)


@app.cell
def _(alt, chart_card, hbar, mo, query, runs_where):
    duration_frame = query(
        f"""
        SELECT
          blob5 AS arch,
          quantileWeighted(0.5)(double3, _sample_interval) AS median_seconds,
          SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where} AND blob2 = 'cli'
        GROUP BY arch
        ORDER BY runs DESC
        """,
    )
    mo.stop(duration_frame.empty, mo.md("_No CLI runs in this window._"))
    duration_frame["label"] = duration_frame["median_seconds"].map(lambda s: f"{s:,.1f} s")
    duration_view = chart_card(
        hbar(
            duration_frame,
            "arch",
            "median_seconds",
            title="Median run duration by CPU architecture",
            subtitle="CLI only: GUI durations include idle review time",
            label="label",
            tooltips=[
                alt.Tooltip("arch:N"),
                alt.Tooltip("median_seconds:Q", title="Median seconds", format=",.1f"),
                alt.Tooltip("runs:Q", title="Runs", format=",.0f"),
            ],
        ),
        duration_frame,
    )
    return (duration_view,)


@app.cell
def _(duration_view, memory_view, mo):
    mo.hstack([memory_view, duration_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(chart_card, days, hbar, mo, query, runs_where):
    os_frame = query(
        f"""
        SELECT blob6 AS os, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where}
        GROUP BY os
        ORDER BY runs DESC
        """,
    )
    mo.stop(os_frame.empty, mo.md("_No OS data in this window._"))
    os_view = chart_card(
        hbar(
            os_frame,
            "os",
            "runs",
            title="Operating systems",
            subtitle=f"weighted runs, last {days} days",
        ),
        os_frame,
    )
    return (os_view,)


@app.cell
def _(chart_card, hbar, mo, query, runs_v2_where):
    cores_frame = query(
        f"""
        SELECT double11 AS cores, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND double11 > 0
        GROUP BY cores
        ORDER BY cores
        """,
    )
    mo.stop(cores_frame.empty, mo.md("_No core-count data in this window yet (schema v2 only)._"))
    cores_frame["cores_label"] = cores_frame["cores"].map(lambda n: f"{n:,.0f} cores")
    cores_view = chart_card(
        hbar(
            cores_frame,
            "cores_label",
            "runs",
            title="Logical CPU cores",
            subtitle="weighted runs per core count",
        ),
        cores_frame,
    )
    return (cores_view,)


@app.cell
def _(cores_view, mo, os_view):
    mo.hstack([os_view, cores_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(mo):
    mo.md("## Reliability & crashes")


@app.cell
def _(COLORS, chart_card, crash_where, mo, pd, query, trend_line):
    crash_trend_frame = query(
        f"""
        SELECT
          toStartOfInterval(timestamp, INTERVAL '1' DAY) AS day,
          SUM(_sample_interval) AS crashes
        FROM photo_tagger_telemetry
        WHERE {crash_where}
        GROUP BY day
        ORDER BY day
        """,
    )
    mo.stop(
        crash_trend_frame.empty,
        mo.md("_No crashes in this window._ :tada:").callout(kind="success"),
    )
    crash_trend_frame["day"] = pd.to_datetime(crash_trend_frame["day"])
    crash_trend_view = chart_card(
        trend_line(
            crash_trend_frame,
            "crashes",
            title="Crashes per day",
            subtitle="weighted crash beacons (status color: this is a state, not a series)",
            val_title="Crashes",
            color=COLORS["critical"],
        ),
        crash_trend_frame,
    )
    return (crash_trend_view,)


@app.cell
def _(COLORS, chart_card, crash_where, hbar, mo, query, version_key):
    crash_version_frame = query(
        f"""
        SELECT blob1 AS app_version, SUM(_sample_interval) AS crashes
        FROM photo_tagger_telemetry
        WHERE {crash_where}
        GROUP BY app_version
        ORDER BY crashes DESC
        LIMIT 12
        """,
    )
    # Guard before sorting; see the app-version cell.
    mo.stop(crash_version_frame.empty, mo.md("_No crashes to attribute to versions._"))
    crash_version_frame = crash_version_frame.sort_values(
        "app_version",
        key=lambda s: s.map(version_key),
        ascending=False,
    )
    crash_version_view = chart_card(
        hbar(
            crash_version_frame,
            "app_version",
            "crashes",
            title="Crashes by app version",
            subtitle="newest first: did the last release make things better or worse?",
            color=COLORS["critical"],
        ),
        crash_version_frame,
    )
    return (crash_version_view,)


@app.cell
def _(crash_trend_view, crash_version_view, mo):
    mo.hstack([crash_trend_view, crash_version_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(COLORS, chart_card, hbar, mo, pd, query, runs_v2_where):
    _kinds_raw = query(
        f"""
        SELECT blob18 AS failure_kinds, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_v2_where} AND blob18 != ''
        GROUP BY failure_kinds
        """,
    )
    mo.stop(
        _kinds_raw.empty,
        mo.md("_No per-photo failures in this window._ :tada:").callout(kind="success"),
    )

    # blob18 is a per-run set like "timeout:3,other:1"; AE SQL cannot split it, so explode
    # client-side and weight each bucket by its count times the run weight.
    _pairs: list[tuple[str, float]] = []
    for _kinds, _runs in zip(_kinds_raw["failure_kinds"], _kinds_raw["runs"], strict=True):
        for _entry in str(_kinds).split(","):
            _kind, _, _count = _entry.partition(":")
            if _kind and _count.isdigit():
                _pairs.append((_kind, int(_count) * _runs))
    failure_kinds_frame = (
        pd.DataFrame(_pairs, columns=["kind", "photos"])
        .groupby("kind", as_index=False)["photos"]
        .sum()
        .sort_values("photos", ascending=False)
    )
    chart_card(
        hbar(
            failure_kinds_frame,
            "kind",
            "photos",
            title="Why photos fail",
            subtitle="final failures after the retry pass, bucketed by coarse cause",
            color=COLORS["critical"],
        ),
        failure_kinds_frame,
    )


@app.cell
def _(crash_where, mo, query):
    signatures_frame = query(
        f"""
        SELECT
          blob15 AS exception_type,
          blob16 AS crash_location,
          blob1 AS app_version,
          blob6 AS os,
          SUM(_sample_interval) AS crashes,
          COUNT(DISTINCT index1) AS installs
        FROM photo_tagger_telemetry
        WHERE {crash_where}
        GROUP BY exception_type, crash_location, app_version, os
        ORDER BY crashes DESC
        LIMIT 50
        """,
    )
    mo.stop(signatures_frame.empty, mo.md("_No crash signatures in this window._"))
    mo.vstack(
        [
            mo.md(
                "### Crash signatures\n"
                "Exception type and the deepest photo-tagger frame (`module:function:line`); "
                "messages are never collected, so a signature is the whole story the beacon "
                "tells. The full in-app frame chain is in `blob17` via the SQL console.",
            ),
            mo.ui.table(signatures_frame, page_size=10),
        ],
        gap=1,
    )


@app.cell
def _(mo):
    mo.md("## Languages")


@app.cell
def _(chart_card, hbar, mo, query, runs_where):
    output_language_frame = query(
        f"""
        SELECT blob9 AS output_language, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where} AND blob9 != ''
        GROUP BY output_language
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(output_language_frame.empty, mo.md("_No metadata-language data in this window yet._"))
    output_language_view = chart_card(
        hbar(
            output_language_frame,
            "output_language",
            "runs",
            title="Metadata languages",
            subtitle="language the model writes titles and keywords in",
        ),
        output_language_frame,
    )
    return (output_language_view,)


@app.cell
def _(chart_card, hbar, mo, query, runs_where):
    ui_language_frame = query(
        f"""
        SELECT blob10 AS ui_language, SUM(_sample_interval) AS runs
        FROM photo_tagger_telemetry
        WHERE {runs_where} AND blob10 != ''
        GROUP BY ui_language
        ORDER BY runs DESC
        LIMIT 12
        """,
    )
    mo.stop(ui_language_frame.empty, mo.md("_No UI-language data in this window yet._"))
    ui_language_view = chart_card(
        hbar(
            ui_language_frame,
            "ui_language",
            "runs",
            title="UI languages",
            subtitle="the app interface language actually in effect",
        ),
        ui_language_frame,
    )
    return (ui_language_view,)


@app.cell
def _(mo, output_language_view, ui_language_view):
    mo.hstack([output_language_view, ui_language_view], widths="equal", gap=1, wrap=True)


@app.cell
def _(mo):
    _default_sql = """\
    -- Column map (see worker.js). Run rows: blob12 != 'crash'; crash rows: blob12 = 'crash';
    -- v2-only fields (hardware, outcome counters): add AND double1 >= 2.
    --   index1  = install_id         blob1  = app_version    blob2  = interface (cli|gui)
    --   blob3   = provider           blob4  = model          blob5  = arch
    --   blob6   = os                 blob7  = os_release     blob8  = python_version
    --   blob9   = output_language    blob10 = ui_language    blob11 = file_types
    --   blob12  = event (run|crash)  blob13 = cpu            blob14 = gpu
    --   blob15  = exception_type     blob16 = crash_location blob17 = crash_frames
    --   double1 = schema_version     double2 = batch_size    double3 = duration_seconds
    --   double4 = success_count      double5 = failure_count double6 = cache_hits
    --   double7 = retry_successes    double8 = workers       double9 = total_tokens
    --   double10 = inference_seconds double11 = cpu_count    double12 = memory_gb
    --   double13 = dry_run (0|1)
    SELECT blob3 AS provider, SUM(_sample_interval) AS runs
    FROM photo_tagger_telemetry
    WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 != 'crash'
    GROUP BY provider
    ORDER BY runs DESC;
    """
    sql_editor = mo.ui.code_editor(value=_default_sql, language="sql")
    run_sql = mo.ui.run_button(label="Run query")
    mo.vstack(
        [
            mo.md(
                "### SQL console\n"
                "Anything the [AE SQL dialect]("
                "https://developers.cloudflare.com/analytics/analytics-engine/sql-reference/"
                ") supports; `FORMAT JSON` is appended automatically. Starter statements "
                "live in `queries.sql`.",
            ),
            sql_editor,
            run_sql,
        ],
        gap=1,
    )
    return run_sql, sql_editor


@app.cell
def _(mo, query, run_sql, sql_editor):
    mo.stop(not run_sql.value, mo.md("_Edit the SQL above and click **Run query**._"))
    result_frame = query(sql_editor.value)
    mo.stop(result_frame.empty, mo.md("_The query returned no rows._"))
    mo.ui.table(result_frame, page_size=15)


@app.cell
def _(mo):
    mo.md(
        """
        ---
        Counts are `SUM(_sample_interval)` and medians are `quantileWeighted`, so numbers stay
        correct once Analytics Engine samples (the weight is 1 until then). The collector stores
        no IPs and no cookies; crash beacons carry the exception type and in-app code location
        only, never messages. See [README.md](README.md) in this directory.
        """,
    )


if __name__ == "__main__":
    app.run()
