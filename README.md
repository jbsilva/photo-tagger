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
(`photo-tagger gui`). Both run the same pipeline and read the same configuration.

By default, Photo Tagger keeps your originals untouched by creating XMP sidecars. You can embed
updates directly into each photo with `--embed-in-photo`. You can also split sidecar and embedded
output by file type with `--sidecar-mode raw`.

Full documentation: <https://jbsilva.github.io/photo-tagger/>

## Highlights

- Works with RAW and standard image formats (CR3, CR2, NEF, JPG, PNG, and more)
- Generates a title, a concise description, and hierarchical keywords, in any language you ask for
  (`--output-language`)
- Merges with existing keywords unless you opt-in to overwrite, and decides per field (title,
  description, keywords) what a photo keeps and what gets replaced.
- Snaps keywords onto your own keyword list, so a run cannot fill your catalog with near-duplicate
  terms
- Builds that keyword list for you from the photos you already have (`photo-tagger vocabulary`)
- Harmonizes each shoot: every frame of the same subject gets the same keyword and the same
  hierarchy, instead of "Osprey" here and "Ospreys" there
- Writes the fields no model can guess: your credit and copyright notice, dated by the year of each
  photo (`--artist`, `--copyright`)
- Writes any other tag you name too (`--set`)
- Prints the metadata a photo already carries, tag by tag (`photo-tagger show`)
- Works with Ollama, LM Studio, llama.cpp, and any hosted OpenAI-compatible API
- Ships a `doctor` command that pre-flights ExifTool and your model provider
- Records what each run writes, so `photo-tagger undo` puts a bad batch back
- Watches a folder and tags photos as they arrive (`photo-tagger watch`)
- Optional desktop GUI (`photo-tagger gui`) for a point-and-click workflow, with the same
  vocabulary, shoot-harmonization, watch, and undo features
- The GUI is translated into your own language (English and Brazilian Portuguese ship today)
- Converts images to compact JPEG bytes to minimize token usage
- Reports what it did: a JSON run summary, NDJSON on stdout, or a per-photo CSV (`--csv-file`)
- Generates detailed log files for easy debugging and auditing
- Highly configurable via CLI flags and environment variables

## Requirements

- Python 3.14+
- [ExifTool](https://exiftool.org/) available on `PATH`, or its path in the `exiftool_path` key or
  the `PHOTO_TAGGER_EXIFTOOL` variable
- A vision-language model (for example Qwen-VL). It can run in Ollama, LM Studio, or llama.cpp on
  your machine, on your own server, or at any OpenAI-compatible service, free or paid.
- `libraw` support for `rawpy`. The rawpy wheels for Linux, Windows, and Apple silicon include it.
  An Intel Mac may need `brew install libraw`

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

- `OLLAMA_BASE_URL`: override the Ollama HTTP endpoint (default `http://localhost:11434/v1`)
- `OLLAMA_API_KEY`: optional API key passed to Ollama requests
- `LM_STUDIO_BASE_URL`: override the LM Studio endpoint (default `http://localhost:1234/v1`)
- `LM_STUDIO_API_KEY`: API key for LM Studio's OpenAI-compatible server. `OPENAI_API_KEY` is never
  sent to it
- `LLAMA_CPP_BASE_URL`: override the llama.cpp `llama-server` endpoint (default
  `http://localhost:8080/v1`)
- `LLAMA_CPP_API_KEY`: API key for llama.cpp. Set this only for a `llama-server` that runs with
  `--api-key`
- `OPENAI_BASE_URL`: endpoint for the `openai` provider (default `https://api.openai.com/v1`)
- `OPENAI_API_KEY`: API key for the `openai` provider (required for that provider)
- `MODEL_NAME`: default model name (default `qwen/qwen3-vl-30b`)
- `JPEG_DIMENSIONS`, `JPEG_QUALITY`, `TEMPERATURE`, `MAX_TOKENS`, `RETRIES`, `TIMEOUT_SECONDS`,
  `FREQUENCY_PENALTY`: fine-tune runtime
- `PHOTO_TAGGER_EXIFTOOL`: path to the ExifTool binary, for installs not on `PATH` (overrides the
  configuration file's `exiftool_path`)
- `PHOTO_TAGGER_LANG`: language of the desktop app's own interface (`en`, `pt_BR`, or `auto`). The
  language of the generated metadata is a separate configuration, `--output-language`

Any CLI flag takes precedence over the environment.

### Config file

You can persist CLI defaults in a TOML file so they apply automatically. Search order:

1. `$PHOTO_TAGGER_CONFIG` environment variable (explicit path)
2. `.photo-tagger.toml` in the current working directory (project-local)
3. `~/.config/photo-tagger/config.toml` (user-wide)

CLI flags override configuration file values, and the configuration file overrides built-in
defaults. If `$PHOTO_TAGGER_CONFIG` names a file that does not exist, photo-tagger logs a warning
and reads no configuration file at all.

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

The section names match the internal option groups: `provider`, `inference`, `output`, `copyright`,
`log`, `display`, `filter`, `artifacts`, `telemetry`, and `pacing`. The `gui` section holds the
settings of the desktop window. Top-level keys cover `extensions`, `recursive`, `workers`,
`exiftool_path`, and `language` (the language of the desktop window). If your ExifTool binary lives
somewhere unusual, set `exiftool_path` to its path. Unknown keys are silently ignored, so the file
stays forward-compatible.

## Usage

The CLI is exposed as `photo-tagger` once installed, or you can invoke it directly:

```bash
photo-tagger -i ./photos --ext cr3,jpg -r
```

Key options:

- `-i/--input PATH`: repeatable. Mix files and directories
- `--ext`: comma-separated extension list for scanning directories (default `cr3,jpg`)
- `-r/--recursive`: recurse into subdirectories while scanning inputs
- `-m/--model`: model identifier understood by your provider
- `--provider`: `ollama`, `lmstudio`, `llamacpp`, or `openai` (defaults to `lmstudio`)
- `--url` / `--api-key`: override provider endpoint and credentials
- `--output-language` / `--lang`: language of the generated title, description, and keywords (any
  language name the model understands, for example `German` or `"Brazilian Portuguese"`)
- `--hint`: a note about every photo in the run that the model must trust over its own reading of
  the image, for example `"The animal in these photos is a deer"`
- `--overwrite-keywords`: replace instead of merge existing keyword metadata
- `--preserve-title` / `--preserve-description`: keep the title or description a photo already has,
  and write the generated one only where there is none. Both are replaced by default, which is what
  clears a camera-written placeholder
- `--no-write-title` / `--no-write-description` / `--no-write-keywords`: skip writing those fields
- `--no-backup-xmp`: avoid creating `*_original` snapshot before writing
- `--embed-in-photo`: write metadata directly into the image instead of creating an XMP sidecar
- `--sidecar-mode {all,none,raw,both}`: where metadata goes
  - `all` (the default): a sidecar for every photo
  - `none` (the same as `--embed-in-photo`): metadata inside every photo
  - `raw`: a sidecar for RAW files, embedded metadata for JPEG and friends
  - `both`: each photo *and* a sidecar beside it
- `--sidecar-override EXT=TARGET`: send one file type elsewhere, whatever the mode. For example,
  `dng=photo` writes into each DNG, which is where Lightroom Classic keeps the metadata of a DNG
- `--dry-run`: run the model and log the proposed metadata without writing XMP
- `-w/--workers N`: process N photos concurrently using a thread pool (default 1)
- `--no-progress`: hide the live rich progress bar (auto-disabled when stderr is not a terminal)
- `--max-keywords N`: cap how many AI-generated keywords are kept per photo before merging
- `--vocabulary PATH`: restrict generated keywords to the terms in a Lightroom keyword-list export.
  Use either the `.txt` or the `.csv` shape of _Metadata > Export Keywords_, or a plain list of
  terms and `Animal|Bird|Osprey` paths. Matching ignores case and punctuation, and every match is
  rewritten to the file's own spelling and hierarchy. It works in any language: write the file in
  the language you generate in (see `--output-language`), and use the `{synonym}` syntax for
  inflected forms. Add `--vocabulary-strict` to drop keywords the file does not cover. The run
  summary then lists every dropped term and how often it came up
- `--session-gap MINUTES`: group photos into shoots separated by this much idle time (capture time,
  with mtime as a fallback). Each shoot's keywords then agree with itself: the spelling and
  hierarchy most of the session used win for all of it. Nothing in a session is written until
  photo-tagger analyzes every photo in it
- `--prompt-file PATH`: override the default user prompt with the contents of `PATH`
- `--summary-file PATH`: write a JSON run summary (token usage, success/failure counts) to `PATH` on
  completion
- `--csv-file PATH`: write a spreadsheet-friendly report with one row per photo. Each row holds what
  was generated, the keywords already on the file, the camera and location EXIF read as context, and
  per-photo tokens and timing. Rows stream as photos finish, so a stopped run still leaves a valid
  file
- `--cache-file PATH`: persistent SQLite cache of model outputs, keyed by an image-data hash and by
  the model, the prompt, and the configuration used. The image-data hash comes from ExifTool's
  `ImageDataHash`, which ignores metadata, so it survives `--embed-in-photo`. If nothing relevant
  changes, a rerun skips the model call
- `--lock-file PATH`: acquire an exclusive file lock on `PATH` before running. If another
  `photo-tagger` already holds it, the run refuses to start (this prevents two runs racing on the
  same folder). It works on Linux, macOS, and Windows
- `--json`: emit one NDJSON line per processed photo to stdout (file, status, title, description,
  keywords, token usage, cache flag). Logs and progress stay on stderr, so you can pipe straight
  into `jq` or your own tools
- `--notify`: raise a desktop notification at the end of the batch, so you can start a long run and
  walk away. It carries the tally (`38 of 42 photos tagged, 4 failed.`), and it fires whether the
  run fully succeeded or only partly did. Under `watch`, every batch gets one notification. Add
  `--notify-sound` for an audible alert too. It uses whatever notifier your OS already ships
  (`osascript`, `notify-send`, a PowerShell toast). There is nothing to install, and nothing breaks
  on a system without one
- `--newer-than DATE` / `--older-than DATE`: filter the input batch by file mtime. It accepts ISO
  8601 dates like `2024-01-01` or `2024-01-01T14:30`. Naive timestamps use local time
- `--no-undo-log`: do not record this run's writes (recording is on by default, and is what
  `photo-tagger undo` puts back)
- `--no-telemetry`: turn off anonymous usage telemetry for this run (see [Telemetry](#telemetry))
- `--jpeg-dimensions`, `--jpeg-quality`, `--temperature`, `--max-tokens`, `--retries`,
  `--timeout-seconds`, `--frequency-penalty`: control inference behavior

### Skipping and resuming

Three flags work together so you can re-run on a folder without redoing finished work:

- `--skip-from FILE`: skip the photos listed in `FILE`, one per line, with `#` marking a comment. A
  line is a bare filename or a full path, and the match ignores case.
- `--append-to-skip-file FILE`: append the full path of each successfully tagged photo to `FILE` as
  the run progresses. If `FILE` is missing, photo-tagger creates it, so you can pass the same path
  to both flags from the very first run.
- `--skip-tagged`: skip files that already have keywords, a description, or a title in either the
  image or its XMP sidecar. It catches photos tagged in Lightroom or by hand without needing a skip
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

### Copyright and other metadata

Title, description, and keywords come from the model. Your name and your copyright notice do not,
and the file is open anyway, so photo-tagger writes them in the same pass:

```bash
photo-tagger -i ~/Pictures/2025 \
  --artist "Ada Lovelace" \
  --copyright "Copyright (c) {year} {artist}, All rights reserved" \
  --rights "© {year} {artist}, All rights reserved" \
  --marked
```

The credit goes to `EXIF:Artist`, `XMP-dc:Creator`, and `IPTC:By-line`. The notice goes to
`EXIF:Copyright` and `IPTC:CopyrightNotice`, and `--rights` gives `XMP-dc:Rights` its own UTF-8
wording. `--marked` sets `XMP-xmpRights:Marked`. `{year}` is the year of the photo, read from its
capture date, so a folder of ten years gets ten different notices. A placeholder can also read any
tag and fall back, as in `{EXIF:Artist|XMP-dc:Creator|"Unknown"}`. A photo that already has a notice
or a creator keeps it, unless you pass `--no-preserve-copyright`. `--set 'TAG=VALUE'` writes any
other ExifTool tag.

To stamp photos that are already tagged, `photo-tagger set` does the same without calling a model:

```bash
photo-tagger set -i ~/Pictures -r --embed-in-photo --artist "Ada Lovelace" --marked
```

Put the values under `[copyright]` in the configuration file and every run credits your photos, with
no flags to remember. `photo-tagger undo` puts back a `set` run, the same as a tagging run.

To see what a photo already carries, tag by tag:

```bash
photo-tagger show -i IMG_0001.CR3
```

Each line names the ExifTool tag behind the value, and marks a value that lives in the XMP sidecar
rather than in the photo. Add `--all` for every tag ExifTool reports, or `--json` to pipe the report
into another tool.

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

Match Lightroom Classic: a sidecar for proprietary RAW files, embedded metadata in DNG and JPEG:

```bash
photo-tagger -i ~/Pictures/shoot --ext cr3,dng,jpg --sidecar-mode raw --sidecar-override dng=photo
```

Build a keyword list from the photos you already have, then tag against it:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --report dropped.csv
photo-tagger -i ~/Pictures/Shoot --vocabulary vocabulary.txt --vocabulary-strict
```

The keywords are read from the photos themselves with ExifTool, so it works whatever wrote them.
That includes Lightroom, digiKam, darktable, Immich, PhotoPrism, Synology Photos, and anything else
that writes XMP or IPTC. Photos and catalog are left untouched. The output is a text file to review
and edit, and `--report` names every keyword it dropped and why.

Add `--organize` to let the model fold synonyms together, and file the keywords under categories.
For example, `Golden Light` becomes a `{synonym}` of `Golden Hour`, so it still matches. It never
decides what to keep, and never invents a keyword: everything it returns is matched back to one your
library already uses.

Tag photos as they land in an import folder, until you stop it with Ctrl-C:

```bash
photo-tagger watch -i ~/Pictures/Inbox -r --skip-tagged
```

Put back what the last run wrote (a bad prompt across 500 photos is one command to revert):

```bash
photo-tagger undo            # or: photo-tagger undo --list / --dry-run
```

Make sure that your setup is ready before a big run: `doctor` pre-flights ExifTool and your model
provider:

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

Preview proposed metadata without writing anything (useful while you iterate on prompts):

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

Tag a trip so every frame of the same subject agrees, with shoots grouped an hour apart:

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

If another run is already in flight on this folder, refuse to start:

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
or folders. Pick what to process from a checkbox tree, with columns for file type, status, and
whether a photo already carries tags. Choose a provider and model: **Connection...** holds the URL,
the API key, and the same pre-flight as `doctor`. Then click **Generate Selected** to run the model
on a background thread with a progress bar. Results are cached, so re-runs are free.

Each photo's proposed title, description, and keywords appear next to the existing values in a
side-by-side detail pane. You can edit any field there before you **Save**. For a batch nobody will
sit through, use **Save After Generating** (on either **Generate** button's arrow). It writes each
run's photos as soon as that run finishes.

A photo that fails turns red and shows why. You can retry it from its right-click menu. **Help >
Open Logs** opens the run log folder.

The GUI reads the same configuration file and environment variables as the CLI, and **Settings >
Save Settings as Defaults...** writes them back. PySide6 is only pulled in by the `gui` extra, so
the plain CLI install stays lightweight.

**Settings > Copyright...** holds your credit and copyright notice, written by every save next to
whatever the model produced. `{year}` in a notice becomes the year of that photo, so the same notice
fits every shoot. A photo that already has a notice keeps it, unless you uncheck the option. The
panel under the fields shows the tags a save writes, before it writes them. With a copyright set, a
save writes it even with Title, Description, and Keywords all unchecked.

The window covers the same keyword workflow as the CLI. **Settings > Keyword Rules...** picks the
controlled vocabulary that generated keywords snap onto (with a strict toggle), and the session gap
that harmonizes each shoot. Both apply to the proposals, so you still review them before saving.

**Tools** holds the jobs that act on a whole library. **Build Vocabulary...** writes a keyword file
out of the keywords your photos already carry (optionally organized by the model). **Watch
Folder...** picks up and generates photos as they land. **Undo Writes...** puts back what a recorded
run wrote, whether that run came from the window or the command line.

### Build a standalone macOS app

For a double-clickable `Photo Tagger.app` that needs no Python install to run:

```bash
./packaging/build_macos_app.sh
```

This bundles Python, PySide6, and everything else into `dist/Photo Tagger.app` with PyInstaller (its
source and spec live in [`packaging/`](packaging/)). Two things to know:

- It is unsigned, so the first launch needs a right-click > **Open** (Gatekeeper asks once). It
  opens normally after that.
- ExifTool is not bundled: install it however you like (`brew install exiftool`, Nix, MacPorts,
  ...). A Finder launch gives apps a minimal `PATH`. On startup, the app inherits the `PATH` that
  your login shell sets, so it finds ExifTool wherever your package manager put it. If ExifTool is
  still missing, **Test Connection** reports a clear error.

## Telemetry

photo-tagger sends small anonymous beacons that show how the tool is used. A beacon goes out at the
end of each run and each other command, as the GUI window opens and closes, and on a crash. It is
opt-out. It is on by default. You see a notice once on the first run, and you can turn it off at any
time. [docs/telemetry.md](docs/telemetry.md) lists every field.

What is collected:

- the app version, the interface (`cli` or `gui`), and the command
- the provider and model, and where the model server runs: `localhost`, your network, or the name of
  a known public service, never the URL
- the configuration and features you use, as switches, counts, and mode names, never the text or
  paths they hold
- which parts of your configuration differ from the defaults
- per run: batch size, file types, duration, outcome counts, tokens, and model time
- in the GUI: how long the window is busy, in use, or idle
- in the GUI: how many photos you edit before a save (counts only), and your screen size
- the OS, CPU and GPU model names, core count, RAM, power source and battery level, locale, time
  zone offset, and install method
- on a crash: the exception type and a stack trace of code locations (module, function, line, and
  library frames), plus the chain of exceptions that caused it. Also the versions of key libraries,
  but never the message and never a file path.
- a random install id (a UUID made once, not derived from any hardware identifier)

What is never collected: photos, file paths, and filenames. The generated tags, titles, and
descriptions, your prompts and hints, and server URLs are never sent either. Neither are API keys,
IP addresses, error messages, or anything else that identifies you. The dataclasses in
[`telemetry.py`](src/photo_tagger/telemetry.py) are the closed list of what leaves the machine.

Where it goes: our own Cloudflare Worker at `telemetry.tagger.photo`. No third-party analytics
service is involved, and the collector's full source (and the queries run against it) lives in
[`telemetry/`](telemetry/). Events are kept with no expiry date, so trends can be compared across
releases. Sending happens on a background thread with a short timeout and is wrapped so it can never
crash or slow down a run.

How to disable it (any one of these):

- pass `--no-telemetry` on the command line
- set `PHOTO_TAGGER_TELEMETRY=0` (or the cross-tool `DO_NOT_TRACK=1`) in your environment
- put `enabled = false` under `[telemetry]` in your configuration file
- in the desktop app, clear **Settings > Send Anonymous Telemetry**. The first-run dialog also
  offers **Turn It Off**. The app keeps the choice for the next launch.

An opt-out in the environment or on the command line wins over every opt-in, so exporting
`PHOTO_TAGGER_TELEMETRY=0` once disables telemetry everywhere, including the GUI. In the desktop
app, a toggle checked by hand wins over `enabled = false` in the configuration file, because the app
writes that key itself when it saves your settings. The `PHOTO_TAGGER_NO_TELEMETRY=1` of version 0.9
still works too.

`DO_NOT_TRACK=1` is the one exception, because it speaks for every tool at once. If you set it but
still want to send photo-tagger's stats, set `PHOTO_TAGGER_TELEMETRY=1`. It works for the CLI and
the GUI. In the desktop app you can also check **Settings > Send Anonymous Telemetry** by hand. The
toggle shows unchecked while `DO_NOT_TRACK` stops the beacons, so what you see is what is sent.

## Logging

Logs are written to stderr and to a timestamped file (for example `20260101...-photo_tagger.log`).
Adjust levels with `--console-log-level` and `--file-log-level`, or disable either by setting the
value to `OFF`.

## Testing

Run the unit tests with:

```bash
uv run poe test
```

The GUI tests are skipped automatically unless the `gui` extra is installed. To run them headless,
use the `test-gui` task. It installs the extra and uses Qt's offscreen platform:

```bash
uv run poe test-gui
```

If you plan to contribute, also run `uv run poe check` before opening a PR. `uv run poe` lists the
other dev tasks.
