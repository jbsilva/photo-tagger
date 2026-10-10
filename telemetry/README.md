# photo-tagger telemetry collector

This directory holds the whole server side of photo-tagger's anonymous usage telemetry. It is a
small Cloudflare Worker that stores each beacon it receives, and no third-party analytics service is
involved. The code is in the repository, so anyone can read what the collector does.

[docs/telemetry.md](../docs/telemetry.md) describes the data that the client sends, the events that
make it send, and the ways to turn it off. The client code is in
[`src/photo_tagger/telemetry.py`](../src/photo_tagger/telemetry.py).

## Files

| File                                 | Purpose                                                            |
| ------------------------------------ | ------------------------------------------------------------------ |
| [`worker.js`](worker.js)             | The Worker. It checks and cleans a beacon, then stores it.         |
| [`worker.test.mjs`](worker.test.mjs) | Tests for the Worker (`uv run poe test-worker`, needs Node).       |
| [`wrangler.toml`](wrangler.toml)     | The Worker's configuration: its name, bindings, and custom domain. |
| [`migrations/`](migrations/)         | The schema of the D1 table that the Worker writes to.              |
| [`infra/`](infra/)                   | OpenTofu configuration for the Cloudflare resources.               |
| [`backfill.py`](backfill.py)         | Copies the events that Analytics Engine still holds into D1.       |
| [`queries.sql`](queries.sql)         | Starter SQL for each question the telemetry exists to answer.      |
| [`dashboard.py`](dashboard.py)       | A marimo dashboard over the stored events.                         |

## What it stores

Each beacon is one event: `run`, `crash`, `session`, `command`, or `launch`.
[docs/telemetry.md](../docs/telemetry.md) lists the fields. The Worker stores every event in two
places. Analytics Engine (AE) keeps it for three months, in one shared column layout. The map of the
columns is in [`queries.sql`](queries.sql), and column `blob19` holds the whole beacon as JSON. D1
keeps it with no expiry date, as one row of the `events` table, with the same JSON in the `payload`
column. The dashboard reads D1, so it can look back further than three months.

Before it stores a beacon, the Worker removes each key that is not `snake_case` and each list. It
also removes each object nested more than one level deep, and it cuts every string to a fixed
length. Thus a hostile client cannot make a row bigger or change its shape. Older clients still send
schema v1 or v2, and the Worker accepts both.

The Worker does not store IP addresses, set cookies, or send data back. The endpoint accepts writes
only, and it replies `204 No Content`.

## Infrastructure

Two tools manage the Cloudflare side, and both read files in this directory.
[OpenTofu](https://opentofu.org), in [`infra/`](infra/), manages the resources of the account, which
change rarely:

- the `photo-tagger-telemetry` D1 database,
- a deploy token for CI, which can deploy the Worker and change the schema of the database,
- a read-only token for the dashboard and the backfill.

Wrangler deploys the Worker from [`wrangler.toml`](wrangler.toml), with its bindings and the
`telemetry.tagger.photo` custom domain. It also applies the migrations in
[`migrations/`](migrations/) to the D1 table. The
[`deploy-telemetry.yml`](../.github/workflows/deploy-telemetry.yml) workflow does both steps after
the Worker tests pass. It runs on each push to `main` that changes the Worker, its tests,
`wrangler.toml`, or a migration.

Analytics Engine needs no setup. The `photo_tagger_telemetry` dataset starts to exist with the first
write.

### Change the Cloudflare resources

Install OpenTofu 1.8 or later, for example the `opentofu` package of Homebrew or nixpkgs. Then
export these three variables:

1. `CLOUDFLARE_ACCOUNT_ID`: the Cloudflare account id.
2. `CLOUDFLARE_API_TOKEN`: a token with D1 Write, Account API Tokens Write, and Zone Read on the
   `tagger.photo` zone.
3. `TF_VAR_state_passphrase`: the passphrase that encrypts the state, at least 16 characters.

Then run these tasks from the root of the repository:

```bash
uv run poe infra-plan    # Show what OpenTofu would change
uv run poe infra-apply   # Make the changes, after you type "yes"
```

OpenTofu keeps its state in `infra/terraform.tfstate`, and git ignores that file. The state holds
the values of the two tokens, so OpenTofu encrypts it with a key made from the passphrase. It
refuses to write the state or a saved plan in plain text. Keep the passphrase with your tokens.
Without the passphrase, the state cannot be read. If you lose the state or the passphrase, import
the database and apply again:

```bash
export TF_VAR_account_id="$CLOUDFLARE_ACCOUNT_ID"
tofu -chdir=telemetry/infra import cloudflare_d1_database.telemetry \
  "$CLOUDFLARE_ACCOUNT_ID/fb3c2ad8-1808-4e07-bb8e-ea5b936ad7d8"
uv run poe infra-apply
```

That apply makes two new tokens. Delete the old ones on the API tokens page of the Cloudflare
dashboard.

After an apply that makes a new token, put its value where it is used:

```bash
# The deploy token is the CLOUDFLARE_API_TOKEN secret of the CI workflow's production environment.
tofu -chdir=telemetry/infra output -raw deploy_token | gh secret set CLOUDFLARE_API_TOKEN --env production
# The read-only token is the CLOUDFLARE_API_TOKEN of the dashboard and the backfill.
tofu -chdir=telemetry/infra output -raw dashboard_token
# The database id goes in wrangler.toml and in CLOUDFLARE_D1_DATABASE_ID. It is not a secret.
tofu -chdir=telemetry/infra output -raw d1_database_id
```

The production environment protects the secret only after two more steps. In the repository
settings, under Environments > production > Deployment branches, allow only `main`. Then delete the
repository-level `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`
(`gh secret delete CLOUDFLARE_API_TOKEN`), and set the account id on the environment too. Until
then, every workflow can read the old repository-level secret.

### Deploy from your machine

CI deploys the Worker on each push. To deploy from your machine, put the deploy token in
`CLOUDFLARE_API_TOKEN`. Then run these commands from this directory:

```bash
npx wrangler d1 migrations apply DB --remote
npx wrangler deploy
```

### Copy the Analytics Engine history into D1

A new D1 database starts empty, but Analytics Engine still has the last three months. After the
Worker starts to write to a new database, copy that history. Run these commands from this directory,
the script with the read-only token and wrangler with the deploy token:

```bash
uv run backfill.py > backfill.sql
npx wrangler d1 execute DB --remote --file backfill.sql
```

The backfill copies only the AE rows that are older than the first D1 row. Thus no event is stored
twice, and no event goes missing. The script also needs `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_D1_DATABASE_ID`. The current database got its history on 2026-09-23.

## Query

To query Analytics Engine, use the statements in [`queries.sql`](queries.sql). Run them with the
[Analytics Engine SQL API](https://developers.cloudflare.com/analytics/analytics-engine/sql-api/) or
with the SQL console of the Cloudflare dashboard. To query D1, run
`npx wrangler d1 execute DB --remote --command "SELECT ..."` from this directory. The D1 table is
plain SQLite, so `payload ->> '$.settings.sidecar_mode'` reads any field of the JSON.

## Dashboard

[`dashboard.py`](dashboard.py) is a [marimo](https://marimo.io) notebook over the stored events.
With `CLOUDFLARE_D1_DATABASE_ID` set, it reads the D1 table. Without it, it reads Analytics Engine,
which keeps three months, so a longer window then shows three months only. The notebook never
writes.

It fetches the events once per refresh and computes every view from them. Thus each filter applies
at once to every view:

- Window: the last 7, 30, 90, or 365 days, or all time.
- Interface: CLI, GUI, or both.
- Installs: show only some install ids, or leave some out. By default, the dashboard leaves out the
  install id of the machine that runs it, so your own test runs do not count. It also leaves out the
  ids in `PHOTO_TAGGER_DASHBOARD_EXCLUDE` (separated by commas) and the all-zero id of the deploy
  checks.
- Development checkouts: by default, the dashboard hides the events of editable installs and of
  checkouts that are not installed.

The sections are Adoption, Usage, Sessions and time, GUI workflows, Setup and configuration,
Hardware and performance, Reliability and crashes, and Languages. An install explorer lists the
events of one install. A SQL console runs your own statements against the same store. Each chart has
a table view, so no value depends on color or on a hover.

The views of the configuration, the GUI sessions, and the review of proposals read fields that
schema v3 added. Events from older clients do not have these fields, so those views count v3 events
only.

The notebook is a self-contained [PEP 723](https://peps.python.org/pep-0723/) script, so
[uv](https://docs.astral.sh/uv/) installs its dependencies into a temporary virtual environment. It
reads these environment variables:

1. `CLOUDFLARE_ACCOUNT_ID`: the Cloudflare account id.
2. `CLOUDFLARE_API_TOKEN`: the read-only token from OpenTofu. It has Account Analytics Read and D1
   Read, and nothing more.
3. `CLOUDFLARE_D1_DATABASE_ID`: the id of the database. Without it, the dashboard reads Analytics
   Engine.

```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN="$(tofu -chdir=telemetry/infra output -raw dashboard_token)"
export CLOUDFLARE_D1_DATABASE_ID="$(tofu -chdir=telemetry/infra output -raw d1_database_id)"
uv run poe dashboard         # uvx marimo run --sandbox telemetry/dashboard.py
uv run poe dashboard-edit    # The same notebook in marimo edit, to change it
uv run poe dashboard-demo    # Made-up events, no credentials needed
```

If a variable is missing, the dashboard opens to setup instructions and does not fail. The demo task
sets `PHOTO_TAGGER_DASHBOARD_DEMO=1`, which shows made-up v3 events. You can then see every view
before real v3 data arrives.
