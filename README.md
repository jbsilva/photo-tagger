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

Photo Tagger is a command-line helper that asks a vision-language model to analyze your photos and
writes Lightroom-compatible metadata.

By default it keeps your originals untouched by creating XMP sidecars, but you can embed the updates
directly into each photo with `--embed-in-photo`.

**Full documentation:** <https://jbsilva.github.io/photo-tagger/>

## Highlights

- Works with RAW and standard image formats (CR3, CR2, NEF, JPG, PNG, and more)
- Generates a title, a concise description, and hierarchical keywords
- Merges with existing metadata unless you opt-in to overwrite
- Works with Ollama, LM Studio, and any hosted OpenAI-compatible API
- Ships a `doctor` command that pre-flights ExifTool and your model provider
- Optional desktop GUI (`photo-tagger gui`) for a point-and-click workflow
- Converts images to compact JPEG bytes to minimize token usage
- Generates detailed log files for easy debugging and auditing
- Highly configurable via CLI flags and environment variables

## Requirements

- Python 3.14+
- [ExifTool](https://exiftool.org/) available on `PATH`
- A running Ollama or LM Studio server exposing a vision-language model (for example Qwen-VL)
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
- `OPENAI_BASE_URL` – endpoint for the `openai` provider (default `https://api.openai.com/v1`)
- `OPENAI_API_KEY` – API key for the `openai` provider (required for that provider)
- `MODEL_NAME` – default model name (default `qwen/qwen3-vl-30b`)
- `JPEG_DIMENSIONS`, `JPEG_QUALITY`, `TEMPERATURE`, `MAX_TOKENS`, `RETRIES` – fine-tune runtime

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
```

The section names match the internal option groups: `provider`, `inference`, `output`, `log`,
`display`, `filter`, and `artifacts`. Top-level keys cover `extensions`, `recursive`, and `workers`.
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
- `--provider` – `ollama`, `lmstudio`, or `openai` (defaults to `lmstudio`)
- `--url` / `--api-key` – override provider endpoint and credentials
- `--overwrite-keywords` – replace instead of merge existing keyword metadata
- `--no-write-title` / `--no-write-description` – skip writing those fields
- `--no-backup-xmp` – avoid creating `*_original` snapshot before writing
- `--embed-in-photo` – write metadata directly into the image instead of creating an XMP sidecar
- `--dry-run` – run the model and log the proposed metadata without writing XMP
- `-w/--workers N` – process N photos concurrently using a thread pool (default 1)
- `--no-progress` – hide the live rich progress bar (auto-disabled on non-interactive stdouts)
- `--max-keywords N` – cap how many AI-generated keywords are kept per photo before merging
- `--prompt-file PATH` – override the default user prompt with the contents of `PATH`
- `--summary-file PATH` – write a JSON run summary (token usage, success/failure counts) to `PATH`
  on completion
- `--cache-file PATH` – persistent SQLite cache of model outputs keyed by image content hash and
  model+prompt+settings. Reruns skip the model call when nothing relevant has changed
- `--lock-file PATH` – acquire an exclusive file lock on `PATH` before running and refuse to start
  if another `photo-tagger` already holds it (prevents two runs racing on the same folder). Works on
  Linux, macOS, and Windows
- `--json` – emit one NDJSON line per processed photo to stdout (file, status, title, description,
  keywords, token usage, cache flag); logs and progress stay on stderr so you can pipe straight into
  `jq` or your own tools
- `--newer-than DATE` / `--older-than DATE` – filter the input batch by file mtime. Accepts ISO 8601
  like `2024-01-01` or `2024-01-01T14:30`; naive timestamps use local time
- `--jpeg-dimensions`, `--jpeg-quality`, `--temperature`, `--max-tokens`, `--retries` – control
  inference behavior

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
or folders, pick what to process from a checkable tree, choose a provider and model (hit **Test
connection** to run the same checks as `doctor`), then **Generate selected** to run the model on a
background thread. Each photo's proposed title, description, and keywords appear next to the
existing values in a side-by-side detail pane, where you can edit any field before you **Save**. A
photo that fails shows why and can be retried, and **Open logs** opens the run log folder. It reads
the same config file and environment variables as the CLI. PySide6 is only pulled in by the `gui`
extra, so the plain CLI install stays lightweight.

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
