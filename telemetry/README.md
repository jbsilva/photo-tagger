# photo-tagger telemetry collector

This directory is the **entire** server side of photo-tagger's anonymous usage telemetry: a small
Cloudflare Worker that stores each beacon it receives. There is no third-party analytics service. It
is kept in the repo so anyone can read exactly what the collector does.

What the client sends, when, and how to turn it off is documented in
[docs/telemetry.md](../docs/telemetry.md). The client side lives in
[`src/photo_tagger/telemetry.py`](../src/photo_tagger/telemetry.py).

## Files

| File                                 | Purpose                                                          |
| ------------------------------------ | ---------------------------------------------------------------- |
| [`worker.js`](worker.js)             | The Worker: checks and sanitizes a beacon, then stores it.       |
| [`worker.test.mjs`](worker.test.mjs) | Tests for the Worker (`uv run poe test-worker`, needs Node).     |
| [`wrangler.toml`](wrangler.toml)     | Deploy config (Worker name, bindings, custom domain route).      |
| [`migrations/`](migrations/)         | The D1 table the Worker writes to.                               |
| [`backfill.py`](backfill.py)         | Copies the events AE still holds into D1 (see below).            |
| [`queries.sql`](queries.sql)         | Starter SQL answering each question the telemetry exists to ask. |
| [`dashboard.py`](dashboard.py)       | A marimo dashboard over the stored events.                       |

## What it stores

Each beacon is one event: `run`, `crash`, `session`, `command`, or `launch`. The fields are listed
in [docs/telemetry.md](../docs/telemetry.md). The Worker stores every event in two places:

- **Analytics Engine** (AE), in one shared column layout (see the map in
  [`queries.sql`](queries.sql)). Column `blob19` holds the whole beacon as JSON. AE keeps data for
  three months.
- **D1**, when the `DB` binding exists: one row per event in the `events` table, with the same JSON
  in `payload`. D1 keeps everything, so the dashboard can look back further than three months.

Before it stores a beacon, the Worker drops any key that is not `snake_case`, any list, and any
nesting deeper than one level, and it clamps every string. A hostile client cannot bloat or reshape
a row. Older clients still send schema v1 or v2, and the Worker accepts both.

It does **not** store IP addresses, set cookies, or read anything back. The endpoint is write-only
and replies `204 No Content`.

## Deploy

Prerequisites: a Cloudflare account, `tagger.photo` as a zone on it, and
[Wrangler](https://developers.cloudflare.com/workers/wrangler/) installed.

```bash
cd telemetry
export CLOUDFLARE_ACCOUNT_ID=...   # or run `wrangler login`
wrangler deploy
```

Wrangler creates the `photo_tagger_telemetry` Analytics Engine dataset on first write and provisions
the DNS record and TLS certificate for `telemetry.tagger.photo`. A push to `main` that changes the
Worker deploys it through [`deploy-telemetry.yml`](../.github/workflows/deploy-telemetry.yml), after
the Worker tests pass.

### Keep every event

Do this once, from the `telemetry` directory, to add the D1 table:

1. Create the database: `wrangler d1 create photo-tagger-telemetry`. It prints a `database_id`.
2. In [`wrangler.toml`](wrangler.toml), uncomment the `[[d1_databases]]` block and paste the id.
3. Create the table: `wrangler d1 migrations apply photo-tagger-telemetry --remote`.
4. Deploy: `wrangler deploy`, or commit `wrangler.toml` and push to `main`.
5. Copy what AE still holds into D1, so the history starts before today:
   `uv run backfill.py > backfill.sql`, then
   `wrangler d1 execute photo-tagger-telemetry --remote --file backfill.sql`.

Do step 5 after step 4. The backfill copies only the AE rows older than the first D1 row, so no
event is stored twice and none goes missing.

## Query

Use the
[Analytics Engine SQL API](https://developers.cloudflare.com/analytics/analytics-engine/sql-api/)
(or the Cloudflare dashboard SQL console) with the statements in [`queries.sql`](queries.sql). For
D1, use `wrangler d1 execute photo-tagger-telemetry --remote --command "SELECT ..."`. The D1 table
is plain SQLite, so `payload ->> '$.settings.sidecar_mode'` reads any field of the JSON.

## Dashboard

[`dashboard.py`](dashboard.py) is a [marimo](https://marimo.io) notebook over the stored events. It
reads the D1 table when `CLOUDFLARE_D1_DATABASE_ID` is set, and Analytics Engine when it is not.
Analytics Engine keeps three months, so a longer window then shows three months only. The notebook
never writes.

It fetches the events once per refresh and computes every view from them, so each filter applies at
once to every view:

- Window: the last 7, 30, 90, or 365 days, or all time.
- Interface: CLI, GUI, or both.
- Installs: show only some install ids, or leave some out. The install id of the machine that runs
  the dashboard is left out by default, so your own test runs do not count. To leave out more ids by
  default, list them in `PHOTO_TAGGER_DASHBOARD_EXCLUDE`, separated by commas.
- Development checkouts: events from editable installs and uninstalled checkouts are hidden by
  default.

The sections are Adoption, Usage, Sessions and time, GUI workflows, Setup and settings, Hardware and
performance, Reliability and crashes, and Languages. An install explorer lists the events of one
install, and a SQL console runs your own statements against the same store. Each chart has a table
view, so no value depends on color or a hover.

The views of the configuration, the GUI sessions, and the review of proposals read fields that
schema v3 added. Events from older clients do not have these fields, so those views count v3 events
only.

The notebook is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script, so
[uv](https://docs.astral.sh/uv/) resolves its dependencies into a throwaway venv. It reads these
environment variables:

1. `CLOUDFLARE_ACCOUNT_ID`: the same account id as the deploy step.
2. `CLOUDFLARE_API_TOKEN`: a [Cloudflare API token](https://dash.cloudflare.com/profile/api-tokens)
   with **Account Analytics: Read**. To read the D1 table, the token also needs **D1: Read**. Give
   it no other permission.
3. `CLOUDFLARE_D1_DATABASE_ID`: the id that `wrangler d1 create` printed. Without it, the dashboard
   reads Analytics Engine.

```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
export CLOUDFLARE_D1_DATABASE_ID=...
uv run poe dashboard         # uvx marimo run --sandbox telemetry/dashboard.py
uv run poe dashboard-edit    # The same notebook in marimo edit, to change it
uv run poe dashboard-demo    # Made-up events, no credentials needed
```

If the variables are missing, the dashboard opens to setup instructions instead of failing. The demo
task sets `PHOTO_TAGGER_DASHBOARD_DEMO=1`, which shows made-up v3 events, so you can see every view
before real v3 data arrives.
