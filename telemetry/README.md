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

[`dashboard.py`](dashboard.py) is a [marimo](https://marimo.io) notebook that runs those queries and
turns them into sections: **adoption** (KPI tiles, daily active-installs trend, app/Python
versions), **usage** (models, providers, CLI vs GUI, batch sizes, file formats, token appetite),
**hardware & performance** (CPU/GPU models, RAM, core counts, OS, duration by architecture),
**reliability & crashes** (success and cache-hit rates, crash trend, crashes by version, a crash
signature table), **languages**, and a free-form SQL console. It reads the SQL API directly and
never writes.

It is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script, so
[uv](https://docs.astral.sh/uv/) resolves its dependencies into a throwaway venv. It needs two
environment variables: the same `CLOUDFLARE_ACCOUNT_ID` as the deploy step, and a
[Cloudflare API token](https://dash.cloudflare.com/profile/api-tokens) scoped to **Account
Analytics: Read** (read-only, nothing else):

```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
uv run poe dashboard         # uvx marimo run --sandbox telemetry/dashboard.py
uv run poe dashboard-edit    # The same notebook in marimo edit, to change it
```

If the variables are missing the dashboard opens to setup instructions instead of failing. A window
control (last 7/30/90 days) and an interface filter (all/CLI/GUI) scope every chart, and each chart
has a table view so no value is gated behind color or a hover.
