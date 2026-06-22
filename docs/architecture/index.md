---
icon: lucide/network
---

# Architecture

photo-tagger turns a set of input paths into Lightroom-compatible metadata by running each photo
through a local vision-language model and writing the result back as an XMP sidecar (the default) or
embedded in the image file. The work is split into small, focused modules so that discovery,
prompting, inference, caching, keyword merging, and metadata writing can each be tested and reasoned
about on their own.

This section explains how those pieces fit together. The pages below go module by module; this
overview shows the high-level flow and a map of every module under `src/photo_tagger/`.

## High-level flow

The orchestrator resolves a batch of images, then processes each photo through the same sequence. A
cache hit short-circuits the model call, which is the slowest step.

```mermaid
flowchart TD
    A[Input paths] --> B[Discovery and filters]
    B --> C{Per photo}
    C --> D[Read existing metadata]
    D --> E[Build prompt]
    E --> F{Cache lookup}
    F -->|hit| K[Merge keywords]
    F -->|miss| G[Prepare JPEG]
    G --> H[Model inference]
    H --> K
    K --> L[Write XMP sidecar or embed]
    L --> M[Run summary]
```

The per-photo steps run inside `run_batch()`. Photos that fail are retried once in a second pass,
and the final run summary reports success and failure counts, token usage, and wall time.

## Modules

Each module under `src/photo_tagger/` owns one part of the flow. Names link to the source on `main`;
table cells cannot hard-wrap, so the link targets live in the reference block below.

| Module                              | Responsibility                                                                                                                                                                                                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`main.py`][main]                   | cyclopts entry point: the `tag()` default command (orchestration) and the `doctor` command. Resolves the batch, acquires the lock, runs the batch, emits NDJSON, and writes the summary.                                                                                                          |
| [`cli_options.py`][cli_options]     | The CLI schema: the seven `@dataclass` flag groups, their config-file defaults (`load_defaults`), and the `ProcessingOptions` mapping.                                                                                                                                                            |
| [`gui.py`][gui]                     | Optional PySide6 desktop frontend (`photo-tagger gui`): a review-before-write window that generates proposals on a background `QThread` (reusing `create_agent`/`analyze_image_with_ai`, not `run_batch`) and writes on demand. Imported lazily; excluded from coverage and the static analyzers. |
| [`gui_state.py`][gui_state]         | The GUI's Qt-free logic (per-photo `PhotoItem` state, the folder tree, keyword parsing and diffing, and the hierarchy preview), unit-tested without a display or PySide6.                                                                                                                         |
| [`config.py`][config]               | Defaults, provider base URLs and keys, tag names, and the system and user prompts.                                                                                                                                                                                                                |
| [`config_file.py`][config_file]     | TOML loader: `find_config_file`, `load_config`, `apply_overrides`.                                                                                                                                                                                                                                |
| [`providers.py`][providers]         | Backend registry: a `ProviderBackend` (Strategy) per backend bundles its listing URL, listing parser, and provider factory; shared HTTP plumbing lives once. Hosts the `openai` backend.                                                                                                          |
| [`discovery.py`][discovery]         | `resolve_image_batch()` expands and extension-filters inputs; `apply_skip_file`, `apply_date_filter`, and `apply_skip_tagged` narrow the batch.                                                                                                                                                   |
| [`image_io.py`][image_io]           | `prepare_image_for_agent()` loads the image (rawpy for RAW, Pillow otherwise), fixes orientation, flattens alpha to white, resizes, and JPEG-encodes the bytes sent to the model.                                                                                                                 |
| [`ai.py`][ai]                       | `create_agent()` builds a pydantic-ai Agent over an OpenAI-compatible model and checks it exists; `analyze_image_with_ai()` runs it and returns an `InferenceResult`.                                                                                                                             |
| [`models.py`][models]               | `GeneratedMetadata` (caps title and description length and keyword count), `InferenceResult`, and `KeywordSet` (the typed keyword value object).                                                                                                                                                  |
| [`keywords.py`][keywords]           | `parse_hierarchical_keyword()` converts leaf-first form to Lightroom root-to-leaf pipe form; `merge_keywords()` merges and dedups case-insensitively, preserving hierarchy.                                                                                                                       |
| [`diagnostics.py`][diagnostics]     | The checks behind `photo-tagger doctor`: ExifTool on PATH and provider/model reachability, returned as `CheckResult`s.                                                                                                                                                                            |
| [`metadata.py`][metadata]           | `read_image_context()` batches one exiftool read; `build_contextual_prompt()` appends it; `write_metadata()` writes the sidecar or embeds; `find_tagged_images()` detects tagged files.                                                                                                           |
| [`cache.py`][cache]                 | `InferenceCache`, a SQLite database (WAL mode) keyed by image content hash plus a namespace digest; a cache hit skips the model call entirely.                                                                                                                                                    |
| [`locking.py`][locking]             | `FileLock` wraps the filelock library for a cross-platform, non-blocking exclusive lock.                                                                                                                                                                                                          |
| [`pipeline.py`][pipeline]           | `run_batch()` orchestrates everything: serial or concurrent processing, the per-photo steps, retries, and streaming outcomes through callbacks.                                                                                                                                                   |
| [`csv_report.py`][csv_report]       | `ReportRow` plus the streaming `CsvReportWriter` (CLI `--csv-file`) and batch `write_report` (GUI export): one spreadsheet row per photo, shared column schema across both frontends.                                                                                                             |
| [`progress.py`][progress]           | rich progress bar on a TTY (stderr), no-op otherwise.                                                                                                                                                                                                                                             |
| [`logging_setup.py`][logging_setup] | loguru config: console sink to stderr (so stdout NDJSON stays clean) and a rotating, compressed file sink.                                                                                                                                                                                        |
| [`errors.py`][errors]               | `PhotoTaggerError` (base), `ProviderError`, `DiscoveryError`, `BatchError`.                                                                                                                                                                                                                       |

!!! note

    Serial runs (`--workers 1`) reuse one long-lived ExifTool process (`-stay_open`). Concurrent runs
    give each worker its own short-lived helper because pyexiftool's pipe is not thread-safe.

## Frontends

The CLI and the GUI are two frontends over the same building blocks, but they consume them at
different levels because they have different goals.

The **CLI** ([`main.py`][main]) drives the batch through `run_batch()`, the pipeline's UI-agnostic
seam. `run_batch()` takes a `ProcessingOptions` bundle and reports back only through callbacks:
`on_image_result` streams one `ImageOutcome` per photo as it finishes, `progress` ticks once per
completed file, and `on_complete` delivers the final `BatchTotals`. Its `--json` mode turns each
`ImageOutcome` into one NDJSON line on stdout. No pipeline code knows what is on the other end of
those callbacks, so a TUI or web service could drive a batch the same way.

The **GUI** ([`gui.py`][gui]) is a review-before-write app, so it deliberately does *not* use
`run_batch()` (which generates and writes in a single pass). Instead it reuses the lower-level
building blocks directly: a background `QThread` worker calls `create_agent()` and
`analyze_image_with_ai()` once per photo and emits a `Proposal`; the user reviews and edits each
proposal, and only then does the window call `merge_keywords()` and `write_metadata()` to save. The
worker reaches the UI thread only through Qt signals, so the pipeline and the building blocks stay
free of any Qt dependency. The Qt-free parts of the GUI live in [`gui_state.py`][gui_state] so this
logic stays testable without a display.

## Detail pages

- [Processing pipeline](pipeline.md): how `run_batch()` orders the per-photo steps, handles serial
    versus concurrent execution, retries failures, and streams outcomes.
- [AI providers](ai-providers.md): the backend registry (Ollama, LM Studio, OpenAI), agent
    construction, model validation, and schema-validation retries.
- [Metadata and keywords](metadata.md): reading existing context, the exact tags written, and how
    hierarchical keywords are parsed and merged.
- [Caching and locking](caching.md): the SQLite inference cache, its namespace digest, and the
    cross-platform file lock.

[ai]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/ai.py
[cache]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/cache.py
[cli_options]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/cli_options.py
[config]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/config.py
[config_file]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/config_file.py
[csv_report]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/csv_report.py
[diagnostics]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/diagnostics.py
[discovery]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/discovery.py
[errors]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/errors.py
[gui]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/gui.py
[gui_state]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/gui_state.py
[image_io]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/image_io.py
[keywords]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/keywords.py
[locking]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/locking.py
[logging_setup]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/logging_setup.py
[main]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/main.py
[metadata]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/metadata.py
[models]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/models.py
[pipeline]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/pipeline.py
[progress]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/progress.py
[providers]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/providers.py
