# Photo Tagger

[![MIT License](https://img.shields.io/badge/license-MIT-007EC7.svg?style=flat-square)](https://github.com/jbsilva/photo-tagger/blob/main/LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/photo-tagger.svg?style=flat-square)](https://pypi.org/project/photo-tagger)
[![Documentation](https://img.shields.io/badge/docs-jbsilva.github.io-blue.svg?style=flat-square)](https://jbsilva.github.io/photo-tagger/)
[![Conda Forge version](https://anaconda.org/conda-forge/photo-tagger/badges/version.svg)](https://anaconda.org/conda-forge/photo-tagger)
[![Downloads](https://pepy.tech/badge/photo-tagger)](https://pepy.tech/project/photo-tagger)
[![tests](https://github.com/jbsilva/photo-tagger/actions/workflows/tests.yml/badge.svg)](https://github.com/jbsilva/photo-tagger/actions/workflows/tests.yml)
[![CodeQL](https://github.com/jbsilva/photo-tagger/actions/workflows/codeql.yml/badge.svg)](https://github.com/jbsilva/photo-tagger/actions/workflows/codeql.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jbsilva_photo-tagger&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=jbsilva_photo-tagger)
[![codecov](https://codecov.io/github/jbsilva/photo-tagger/graph/badge.svg?token=G3EFTL5S9Z)](https://codecov.io/github/jbsilva/photo-tagger)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![BuyMeACoffee](https://img.shields.io/badge/%E2%98%95-buymeacoffee-ffdd00?style=flat-square)](https://www.buymeacoffee.com/jbsilva)

Photo Tagger asks a vision-language model to analyze your photos and writes Lightroom-compatible
metadata. Use it as a command-line tool (`photo-tagger`) or as an optional desktop app
(`photo-tagger gui`); both run the same pipeline and read the same config.

By default it keeps your originals untouched by creating XMP sidecars, but you can embed the updates
directly into each photo with `--embed-in-photo`, or split the two by file type with
`--sidecar-mode raw`.

**Full documentation:** <https://jbsilva.github.io/photo-tagger/>

## Highlights

- Works with RAW and standard image formats (CR3, CR2, NEF, JPG, PNG, and more)
- Generates a title, a concise description, and hierarchical keywords, in any language you ask for
  (`--output-language`)
- Merges with existing keywords unless you opt-in to overwrite, and decides per field (title,
  description, keywords) what a photo keeps and what gets replaced
- Snaps keywords onto your own keyword list, so a run cannot fill your catalog with near-duplicates
  of terms you already curate, and builds that list for you from the photos you already have
  (`photo-tagger vocabulary`)
- Harmonizes each shoot: every frame of the same subject gets the same keyword and the same
  hierarchy, instead of "Osprey" here and "Ospreys" there
- Works with Ollama, LM Studio, llama.cpp, and any hosted OpenAI-compatible API
- Ships a `doctor` command that pre-flights ExifTool and your model provider
- Records what each run writes, so `photo-tagger undo` puts a bad batch back
- Watches a folder and tags photos as they arrive (`photo-tagger watch`)
- Optional desktop GUI (`photo-tagger gui`) for a point-and-click workflow, with the same
  vocabulary, shoot-harmonization, watch, and undo features, translated into your own language
  (English and Brazilian Portuguese ship today)
- Converts images to compact JPEG bytes to minimize token usage
- Reports what it did: a JSON run summary, NDJSON on stdout, or a per-photo CSV (`--csv-file`)
- Generates detailed log files for easy debugging and auditing
- Highly configurable via CLI flags and environment variables

## Requirements

- Python 3.14+
- [ExifTool](https://exiftool.org/) available on `PATH`
- A running Ollama, LM Studio, or llama.cpp server exposing a vision-language model (for example
  Qwen-VL)
- `libraw` support for `rawpy` (install via Homebrew on macOS: `brew install libraw`)

## Installation

For end-users, the recommended installation method is via
[uv](https://docs.astral.sh/uv/guides/tools/):

```bash
uv tool install photo-tagger
```

To include the optional desktop GUI, install the `gui` extra:

```bash
uv tool install 'photo-tagger[gui]'
```

Or install from [conda-forge](https://anaconda.org/conda-forge/photo-tagger) (works with conda,
mamba, or pixi). The GUI is a separate package that bundles PySide6:

```bash
conda install -c conda-forge photo-tagger        # CLI only
conda install -c conda-forge photo-tagger-gui     # CLI + desktop GUI
```

For development (tests, linting):

```bash
uv sync --group dev --group test
```

## Configuration

Environment variables provide defaults so you can keep the CLI concise:

- `OLLAMA_BASE_URL` – override the Ollama HTTP endpoint (default `http://localhost:11434/v1`)
- `OLLAMA_API_KEY` – optional API key passed to Ollama requests
- `LM_STUDIO_BASE_URL` – override the LM Studio endpoint (default `http://localhost:1234/v1`)
- `LM_STUDIO_API_KEY` / `OPENAI_API_KEY` – API key for LM Studio’s OpenAI-compatible server
- `LLAMA_CPP_BASE_URL` – override the llama.cpp `llama-server` endpoint (default
  `http://localhost:8080/v1`)
- `LLAMA_CPP_API_KEY` – API key for llama.cpp, only needed when `llama-server` was started with
  `--api-key`
- `OPENAI_BASE_URL` – endpoint for the `openai` provider (default `https://api.openai.com/v1`)
- `OPENAI_API_KEY` – API key for the `openai` provider (required for that provider)
- `MODEL_NAME` – default model name (default `qwen/qwen3-vl-30b`)
- `JPEG_DIMENSIONS`, `JPEG_QUALITY`, `TEMPERATURE`, `MAX_TOKENS`, `RETRIES`, `TIMEOUT_SECONDS`,
  `FREQUENCY_PENALTY` – fine-tune runtime
- `PHOTO_TAGGER_EXIFTOOL` – path to the ExifTool binary, for installs not on `PATH` (overrides the
  config file's `exiftool_path`)
- `PHOTO_TAGGER_LANG` – language of the desktop app's own interface (`en`, `pt_BR`, or `auto`); the
  language of the generated metadata is `--output-language`, a separate setting

Any CLI flag takes precedence over the environment.

### Config file

You can persist CLI defaults in a TOML file so they apply automatically. Search order:

1. `$PHOTO_TAGGER_CONFIG` environment variable (explicit path)
2. `.photo-tagger.toml` in the current working directory (project-local)
3. `~/.config/photo-tagger/config.toml` (user-wide)

CLI flags override config file values, and the config file overrides built-in defaults.

Example `.photo-tagger.toml`:

```toml
extensions = "cr3,jpg,dng"
recursive = true
workers = 2

[provider]
model_name = "qwen/qwen3-vl-30b"
provider_name = "lmstudio"
api_base_url = "http://localhost:1234/v1"

[inference]
temperature = 0.2
max_tokens = 32768

[output]
preserve_keywords = true
max_keywords = 15

[artifacts]
cache_file = ".photo-tagger-cache.db"

[telemetry]
enabled = true
```

The section names match the internal option groups: `provider`, `inference`, `output`, `log`,
`display`, `filter`, `artifacts`, and `telemetry`. Top-level keys cover `extensions`, `recursive`,
`workers`, and `exiftool_path` (set this to your ExifTool binary if it lives somewhere unusual).
Unknown keys are silently ignored, so the file stays forward-compatible.

## Usage

The CLI is exposed as `photo-tagger` once installed, or you can invoke it directly:

```bash
photo-tagger -i ./photos --ext cr3,jpg -r
```

Key options:

- `-i/--input PATH` – repeatable; mix files and directories
- `--ext` – comma-separated extension list used when scanning directories (default `cr3,jpg`)
- `-r/--recursive` – recurse into subdirectories while scanning inputs
- `-m/--model` – model identifier understood by your provider
- `--provider` – `ollama`, `lmstudio`, `llamacpp`, or `openai` (defaults to `lmstudio`)
- `--url` / `--api-key` – override provider endpoint and credentials
- `--output-language` / `--lang` – language of the generated title, description, and keywords (any
  language name the model understands, for example `German` or `"Brazilian Portuguese"`)
- `--hint` – a note about every photo in the run that the model must trust over its own reading of
  the image, for example `"The animal in these photos is a deer"`
- `--overwrite-keywords` – replace instead of merge existing keyword metadata
- `--preserve-title` / `--preserve-description` – keep the title or description a photo already has
  and write the generated one only where there is none (both are replaced by default, which is what
  clears a camera-written placeholder)
- `--no-write-title` / `--no-write-description` / `--no-write-keywords` – skip writing those fields
- `--no-backup-xmp` – avoid creating `*_original` snapshot before writing
- `--embed-in-photo` – write metadata directly into the image instead of creating an XMP sidecar
- `--sidecar-mode {all,none,raw,both}` – where metadata goes: sidecars for every photo (`all`, the
  default), inside every photo (`none`, the same as `--embed-in-photo`), one of each per file type
  (`raw`: a sidecar for RAW files, embedded for JPEG and friends), or `both` (each photo *and* a
  sidecar beside it)
- `--dry-run` – run the model and log the proposed metadata without writing XMP
- `-w/--workers N` – process N photos concurrently using a thread pool (default 1)
- `--no-progress` – hide the live rich progress bar (auto-disabled on non-interactive stdouts)
- `--max-keywords N` – cap how many AI-generated keywords are kept per photo before merging
- `--vocabulary PATH` – restrict generated keywords to the terms in a Lightroom keyword-list export
  (either the `.txt` or the `.csv` shape of _Metadata > Export Keywords_, or a plain list of terms
  and `Animal|Bird|Osprey` paths). Matching ignores case and punctuation, and every match is
  rewritten to the file's own spelling and hierarchy. Works in any language: write the file in the
  language you generate in (see `--output-language`), and use the `{synonym}` syntax for inflected
  forms. Add `--vocabulary-strict` to drop keywords the file does not cover; the run summary then
  lists every dropped term and how often it came up
- `--session-gap MINUTES` – group photos into shoots separated by this much idle time (by capture
  time, falling back to mtime) and make each shoot's keywords agree with itself: the spelling and
  hierarchy most of the session used win for all of it. Nothing in a session is written until every
  photo in it has been analyzed
- `--prompt-file PATH` – override the default user prompt with the contents of `PATH`
- `--summary-file PATH` – write a JSON run summary (token usage, success/failure counts) to `PATH`
  on completion
- `--csv-file PATH` – write a spreadsheet-friendly report with one row per photo: what was
  generated, the keywords already on the file, the camera and location EXIF read as context, and
  per-photo tokens and timing. Rows stream as photos finish, so a stopped run still leaves a valid
  file
- `--cache-file PATH` – persistent SQLite cache of model outputs keyed by an image-data hash
  (ExifTool's `ImageDataHash`, which ignores metadata so it survives `--embed-in-photo`) plus
  model+prompt+settings. Reruns skip the model call when nothing relevant has changed
- `--lock-file PATH` – acquire an exclusive file lock on `PATH` before running and refuse to start
  if another `photo-tagger` already holds it (prevents two runs racing on the same folder). Works on
  Linux, macOS, and Windows
- `--json` – emit one NDJSON line per processed photo to stdout (file, status, title, description,
  keywords, token usage, cache flag); logs and progress stay on stderr so you can pipe straight into
  `jq` or your own tools
- `--notify` – raise a desktop notification when the batch ends, so a long run can be started and
  walked away from. It carries the tally (`38 of 42 photos tagged, 4 failed.`) and fires whether the
  run fully succeeded or only partly did; under `watch`, every batch gets one. Add `--notify-sound`
  for an audible alert. Uses whatever notifier your OS already ships (`osascript`, `notify-send`, a
  PowerShell toast), so there is nothing to install and nothing breaks where there is none
- `--newer-than DATE` / `--older-than DATE` – filter the input batch by file mtime. Accepts ISO 8601
  like `2024-01-01` or `2024-01-01T14:30`; naive timestamps use local time
- `--no-undo-log` – do not record this run's writes (recording is on by default, and is what
  `photo-tagger undo` puts back)
- `--no-telemetry` – turn off anonymous usage telemetry for this run (see [Telemetry](#telemetry))
- `--jpeg-dimensions`, `--jpeg-quality`, `--temperature`, `--max-tokens`, `--retries`,
  `--timeout-seconds`, `--frequency-penalty` – control inference behavior

### Skipping and resuming

Three flags work together so you can re-run on a folder without redoing finished work:

- `--skip-from FILE` – skip filenames listed in `FILE` (one per line; `#` lines are comments).
- `--append-to-skip-file FILE` – append each successfully tagged filename to `FILE` as the run
  progresses. The file is created if missing, so the same path can be passed to both flags from the
  very first run.
- `--skip-tagged` – skip files that already have keywords, a description, or a title in either the
  image or its XMP sidecar. Catches photos tagged in Lightroom or by hand without needing a skip
  list at all.

Resume-on-failure pattern: pass the same path to both flags so a killed run can be restarted with a
single command.

```bash
photo-tagger -i ~/Pictures/Shoot -r \
  --skip-from processed.txt \
  --append-to-skip-file processed.txt
```

To process a folder mixing already-tagged and untagged photos:

```bash
photo-tagger -i ~/Pictures/Mixed --skip-tagged
```

A successful run creates or updates an `.xmp` sidecar for every processed image (unless you embed
the metadata). Existing metadata is merged so Lightroom keeps hierarchical keywords such as
`Animal|Bird|Osprey` intact.

### Examples

Process a folder of RAW and JPEG files recursively:

```bash
photo-tagger -i ~/Pictures/Portfolio --ext cr3,jpg -r
```

Tag a few explicit files and overwrite existing keywords:

```bash
photo-tagger \
  -i IMG_0001.CR3 \
  -i IMG_0002.CR3 \
  --overwrite-keywords
```

Embed metadata directly into a set of JPEGs:

```bash
photo-tagger -i ./exports --ext jpg --embed-in-photo
```

Tag a folder of RAW + JPEG pairs, sidecars for the RAWs and embedded metadata for the JPEGs:

```bash
photo-tagger -i ~/Pictures/shoot --ext cr3,jpg --sidecar-mode raw
```

Build a keyword list from the photos you already have, then tag against it:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --report dropped.csv
photo-tagger -i ~/Pictures/Shoot --vocabulary vocabulary.txt --vocabulary-strict
```

The keywords are read from the photos themselves with ExifTool, so it works whatever wrote them:
Lightroom, digiKam, darktable, Immich, PhotoPrism, Synology Photos, and anything else that writes
XMP or IPTC. Photos and catalog are left untouched; the output is a text file to review and edit,
and `--report` names every keyword it dropped and why.

Add `--organize` to let the model fold synonyms together (`Golden Light` becomes a `{synonym}` of
`Golden Hour`, so it still matches) and file the keywords under categories. It never decides what to
keep, and never invents a keyword: everything it returns is matched back to one your library already
uses.

Tag photos as they land in an import folder, until you stop it with Ctrl-C:

```bash
photo-tagger watch -i ~/Pictures/Inbox -r --skip-tagged
```

Put back what the last run wrote (a bad prompt across 500 photos is one command to revert):

```bash
photo-tagger undo            # or: photo-tagger undo --list / --dry-run
```

Check your setup before a big run (verifies ExifTool and that the provider serves the model):

```bash
photo-tagger doctor --provider lmstudio --model qwen/qwen3-vl-30b
```

Send requests to a remote Ollama host with a custom model:

```bash
photo-tagger -i ./shoot --provider ollama --model llava:34b --url http://ollama-box:11434/v1
```

Use a hosted OpenAI-compatible API (key read from `OPENAI_API_KEY`):

```bash
photo-tagger -i ./shoot --provider openai --model gpt-4o-mini
```

Preview proposed metadata without writing anything (useful when iterating on prompts):

```bash
photo-tagger -i ./sample --dry-run
```

Process a large folder concurrently with a live progress bar and a JSON summary:

```bash
photo-tagger -i ~/Pictures/Trip -r --workers 4 --summary-file ~/Pictures/Trip/run.json
```

Keep generated keywords inside the vocabulary your Lightroom catalog already uses:

```bash
photo-tagger -i ./shoot --vocabulary ~/lightroom-keywords.txt --vocabulary-strict
```

Tag a trip so every frame of the same subject agrees, grouping shoots an hour apart:

```bash
photo-tagger -i ~/Pictures/Trip -r --session-gap 60
```

Use a custom prompt tuned for wildlife photography:

```bash
photo-tagger -i ./shoot --prompt-file prompts/wildlife.txt --max-keywords 12
```

Cache model outputs so reruns on the same folder skip the inference cost:

```bash
photo-tagger -i ~/Pictures/Shoot -r --cache-file ~/.cache/photo-tagger.db
```

Tag only photos from a specific trip and stream NDJSON for downstream tools:

```bash
photo-tagger -i ~/Pictures/Camera -r \
  --newer-than 2026-04-01 --older-than 2026-05-01 \
  --json --no-progress | jq -c 'select(.status == "ok") | {file, title}'
```

Refuse to start if another run is already in flight on this folder:

```bash
photo-tagger -i ~/Pictures/Camera --lock-file /tmp/photo-tagger.lock
```

## Desktop GUI

Prefer a point-and-click workflow? Install the optional `gui` extra (or the `photo-tagger-gui`
conda-forge package) and launch the desktop app:

```bash
uv tool install 'photo-tagger[gui]'        # or: conda install -c conda-forge photo-tagger-gui
photo-tagger gui
```

The GUI is a review-before-write frontend over the same building blocks as the CLI. Drag in photos
or folders, pick what to process from a checkable tree (with per-file type, status, and
already-tagged columns), choose a provider and model (**Connection...** holds the URL, API key, and
the same checks as `doctor`), then **Generate Selected** to run the model on a background thread
with a progress bar; results are cached so re-runs are free. Each photo's proposed title,
description, and keywords appear next to the existing values in a side-by-side detail pane, where
you can edit any field before you **Save**. A photo that fails turns red, shows why, and can be
retried from its right-click menu, and **Help > Open Logs** opens the run log folder. It reads the
same config file and environment variables as the CLI, and **Settings > Save Settings as Defaults**
writes them back. PySide6 is only pulled in by the `gui` extra, so the plain CLI install stays
lightweight.

The window covers the same keyword workflow as the CLI. **Settings > Keyword Rules...** picks the
controlled vocabulary generated keywords are snapped onto (with a strict toggle) and the session gap
that harmonizes each shoot, applied to the proposals so you still review before saving. **Tools**
holds the jobs that act on a whole library: **Build Vocabulary...** writes a keyword file out of the
keywords your photos already carry (optionally organized by the model), **Watch Folder...** picks up
and generates photos as they land, and **Undo Writes...** puts back what a recorded run wrote,
whether it came from the window or the command line.

### Build a standalone macOS app

For a double-clickable `Photo Tagger.app` that needs no Python install to run:

```bash
./packaging/build_macos_app.sh
```

This bundles Python, PySide6, and everything else into `dist/Photo Tagger.app` with PyInstaller (its
source and spec live in [`packaging/`](packaging/)). Two things to know:

- It is **unsigned**, so the first launch needs a right-click > **Open** (Gatekeeper asks once); it
  opens normally after that.
- ExifTool is **not** bundled - install it however you like (`brew install exiftool`, Nix, MacPorts,
  ...). A Finder launch gives apps a minimal `PATH`, so on startup the app inherits the `PATH` your
  login shell would set, finding exiftool wherever your package manager put it. **Test Connection**
  reports a clear error if it is still missing.

## Telemetry

photo-tagger sends a single anonymous beacon at the end of each run so development can be guided by
how the tool is actually used (which models, platforms, and hardware are common, typical batch
sizes), and one when the app crashes so breakage is visible without waiting for bug reports. It is
**opt-out**: on by default, with a one-time notice on the first run, and easy to disable.

**What is collected**, and nothing else:

- app version and interface (`cli` or `gui`)
- provider and model name
- batch size (photo count), run duration, and outcome counts (successes, failures, cache hits, retry
  recoveries, worker count, token totals, model time, dry-run flag), plus coarse failure buckets
  (fixed labels like `timeout` or `metadata-write`, never error text)
- CPU architecture, OS, OS release, and Python version
- CPU and GPU model names (for example "Apple M3 Pro" or "NVIDIA GeForce RTX 4070"), logical core
  count, and RAM size in whole gigabytes: generic values shared by millions of machines
- on a crash: the exception *type* and its code location inside photo-tagger
  (`module:function:line`), never the error message (messages can embed paths)
- a random install id (a UUID generated once, **not** derived from any hardware identifier)

**What is never collected:** file paths, filenames, photo contents, generated tags/titles/
descriptions, prompts, API keys, IP addresses, error messages, or anything else that identifies you.
The exact, closed payloads are the
[`build_payload` and `build_crash_payload`](src/photo_tagger/telemetry.py) functions; there is
nothing else to leak.

**Where it goes:** our own Cloudflare Worker at `telemetry.tagger.photo`. No third-party analytics
service is involved, and the collector's full source (and the queries run against it) lives in
[`telemetry/`](telemetry/). Sending happens on a background thread with a short timeout and is
wrapped so it can never crash or slow down a run.

**How to disable it** (any one of these):

- pass `--no-telemetry` on the command line
- set `PHOTO_TAGGER_NO_TELEMETRY=1` (or the cross-tool `DO_NOT_TRACK=1`) in your environment
- put `enabled = false` under `[telemetry]` in your config file
- in the desktop app, uncheck **Settings > Send Anonymous Telemetry** (the first-run dialog also
  offers a one-click "Turn It Off"); the choice is remembered for next launch

The environment variables win over everything, so exporting `PHOTO_TAGGER_NO_TELEMETRY=1` once
disables telemetry everywhere, including the GUI.

## Logging

Logs are written to stderr and to a timestamped file (for example `20260101...-photo_tagger.log`).
Adjust levels with `--console-log-level` and `--file-log-level`, or disable either by setting the
value to `OFF`.

## Testing

Run the unit tests with:

```bash
uv run pytest
```

The GUI tests are skipped automatically unless the `gui` extra is installed. To exercise them, sync
the extra and run headless via Qt's offscreen platform:

```bash
uv sync --extra gui --group test
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py
```

If you plan to contribute, also run `uv run ruff check` for linting before opening a PR.
