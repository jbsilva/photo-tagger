# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional desktop GUI (`photo-tagger gui`), installed via the `gui` extra
  (`pip install 'photo-tagger[gui]'`). A PySide6 frontend with a review-before-write workflow: drag
  in photos or folders, pick what to process from a checkable, nested tree (remove entries with the
  button or ++delete++), generate proposals with the model, then review each photo's preview
  alongside its existing and proposed title, description, and keywords, edit any field with a live
  Lightroom-hierarchy preview, and save. The provider list is capitalized, the model field can query
  the provider for available models (vision-likely first), and the columns are resizable. The
  command imports PySide6 lazily, so the base CLI never depends on Qt and prints an install hint if
  the extra is missing. Ships with a custom app icon.
- GUI failure handling and logs. A photo that fails to generate now shows why: hover its `failed ✗`
  status for the reason, or open it to see a red banner above the preview. **Retry failed** re-runs
  the model on every failed photo (or use **Generate this photo** to retry just one); a successful
  retry clears the banner. The GUI now writes a rotating log file to `~/.photo-tagger/logs/` (with
  full tracebacks) instead of silencing logging, and an **Open logs** button reveals that folder in
  the file browser.
- GUI **API key** field. The toolbar now has a masked key field alongside the provider, model, and
  URL, so a key can be typed straight into the window. Leaving it blank falls back to the provider's
  environment variable (the previous behavior); a typed key is used for that session only and is
  never written to disk.
- New `openai` provider for any hosted OpenAI-compatible endpoint (the real OpenAI API or a drop-in
  gateway). It fails fast with a clear message when no API key is configured. Set the endpoint with
  `OPENAI_BASE_URL` / `--url` and the key with `OPENAI_API_KEY` / `--api-key`.
- New `photo-tagger doctor` command: a pre-flight checklist that verifies ExifTool is on PATH and
  the configured provider is reachable and serves the requested model, exiting non-zero on failure.
- A PEP 561 `py.typed` marker so downstream projects can consume the package's type hints.

### Changed

- Backends now live in a `photo_tagger.providers` registry. Each `ProviderBackend` bundles the three
  things that differ per backend (model-listing URL, listing parser, provider factory); the shared
  HTTP plumbing is written once. Adding a backend is a single registry entry.
- Existing keywords are modeled by a typed `KeywordSet` value object instead of a
  `dict[str, list[str]]` keyed by bare strings, removing a silent-typo failure mode.
- The package version is read from installed distribution metadata (`importlib.metadata.version`),
  so `pyproject.toml` is the only code-side source of truth.
- CLI option groups moved out of `main.py` into `photo_tagger.cli_options`, separating the CLI
  schema from the orchestration logic.

### Fixed

- Hierarchical keywords are generated reliably again. The model schema now has a dedicated
  `hierarchies` field, so the vision model is given an explicit place to put taxonomy chains
  (`Golden Eagle<Bird of Prey<Animal`) instead of being expected to embed `<` inside the flat
  `keywords` list, which it had stopped doing. The chains are folded back into the keyword list and
  written to `XMP-lr:HierarchicalSubject` as before. (The CLI's `--cache-file`, if used, is keyed on
  the user prompt and sampling only, not the system prompt, so delete a stale cache to pick up
  hierarchies on already-processed photos; the GUI never caches and is unaffected.)

## [0.2.2] - 2026-05-30

### Added

- MIT License. The `LICENSE` file now ships in the sdist via PEP 639 `license-files`; the deprecated
  `License :: OSI Approved :: MIT License` classifier was dropped.

### Changed

- System prompt now forbids emitting the camera body, lens model, or capture timestamp as keywords;
  these describe equipment, not subject content. Earlier runs sometimes copied literal EXIF strings
  (e.g. `Canon Eos R5M2`, `Rf200-800Mm F6.3-9 Is Usm`) into the keyword list.

### Fixed

- `create_agent` no longer has a code path where the provider could be left unbound for a value
  outside the supported set. Provider construction moved into `_build_provider`, which returns from
  each branch and ends in `assert_never`, keeping the match exhaustive for the type checker.

## [0.2.1] - 2026-05-27

### Fixed

- `AttributeError: 'str' object has no attribute 'parent'` when writing the summary file after a
  successful run if `summary_file` (or any other `Path` field) was set via the TOML config rather
  than the CLI. `apply_overrides` now coerces string TOML values to `Path` for fields annotated as
  `Path` or `Path | None`.

## [0.2.0] - 2026-05-26

### Added

- TOML config file. Search order: `$PHOTO_TAGGER_CONFIG`, `./.photo-tagger.toml`,
  `~/.config/photo-tagger/config.toml`. CLI flags still win. See
  [`.photo-tagger.example.toml`](.photo-tagger.example.toml) for a template.
- `--workers N` for thread-pool concurrency (default 1; the model server is usually the bottleneck).
- `--cache-file PATH` SQLite cache of model outputs keyed by image content hash plus model, prompt,
  and sampling settings. Reruns skip the model entirely when nothing relevant changed. WAL mode is
  enabled.
- `--lock-file PATH` exclusive file lock that refuses to start if another `photo-tagger` already
  holds it. Cross-platform (Linux, macOS, Windows).
- `--summary-file PATH` writes a JSON run summary on completion (success counts, failed files, token
  usage, wall time). Atomic write; parent dir is created.
- `--json` emits one NDJSON line per processed photo on stdout. Logs and progress stay on stderr so
  `| jq` works.
- `--skip-tagged` skips files whose image or sidecar already has keywords, description, or title
  (catches photos tagged in Lightroom or by hand).
- `--append-to-skip-file PATH` records each successful filename so a later run with
  `--skip-from PATH` resumes where this one stopped.
- `--newer-than` / `--older-than` ISO 8601 mtime filters. Naive timestamps are read as local time.
- `--prompt-file PATH` replaces the default user prompt with file contents; existing photo metadata
  is still appended.
- `--max-keywords N` caps the AI keyword count per photo before merging with existing tags.
- `--dry-run` runs the model and logs the proposed metadata without writing.
- `--timeout-seconds` per-image hard cap; the retry loop handles the abort.
- `--frequency-penalty` (default 0.5) suppresses chant-style token loops observed with Qwen3-VL at
  low temperature.
- `--progress` / `--no-progress` rich progress bar (auto-disabled on non-tty stderr).
- Graceful Ctrl-C in batch runs.
- Token usage tracking per call (`InferenceResult`) and per batch (`BatchTotals`).

### Changed

- System prompt rewritten: anchors on visible image content, treats EXIF/GPS as corroborative
  evidence only, refuses to copy existing keywords as filler.
- Default console log level is now `INFO` (was `DEBUG`).
- Default `MAX_TOKENS` raised to 1200.
- Progress bar routed to stderr; stays clean alongside `--json`.
- Existing keywords are de-duplicated case-insensitively at read time.
- `WeightedFlatSubject` is now written back when persisting merged keywords.
- `--ext` matches case-insensitively; default aligned with the README.

### Fixed

- Lock leak when the PID file write failed after lock acquire.
- `parse_hierarchical_keyword` returning `['']` for empty input.
- StubAgent exposing `usage` as a callable instead of an attribute.
- Pydantic-AI deprecation: access `result.usage` as a property.
- EXIF orientation now honored; PIL file handles closed eagerly.
- Over-long keyword lists are truncated rather than failing validation.
- Cache and lock startup errors degrade to warnings instead of failing the run.
- Cache I/O errors are treated as warnings, not photo failures.
- Skip-list appender is now thread-safe under `--workers > 1`.
- Numeric env vars (`JPEG_QUALITY`, `TEMPERATURE`, etc.) parse safely with a warning instead of
  crashing.

### Security

- `--api-key` warns that CLI args are visible in process listings; prefer env vars
  (`OLLAMA_API_KEY`, `LM_STUDIO_API_KEY`, `OPENAI_API_KEY`).
- Lock file permissions tightened to `0o600`.

## [0.1.0] - 2026-02-16

Initial release.

### Added

- `photo-tagger` CLI that asks a vision-language model to analyze each photo and writes
  Lightroom-compatible metadata (title, one-sentence description, and hierarchical keywords).
- RAW and standard image support: CR3, CR2, NEF, JPG, PNG, and more.
- XMP sidecars by default; `--embed-in-photo` writes metadata directly into the image instead.
- Keyword merging with existing metadata by default; `--overwrite-keywords` replaces.
  `--no-write-title` / `--no-write-description` skip those fields.
- `--no-backup-xmp` to skip the ExifTool `_original` snapshot.
- Provider support for Ollama and LM Studio via their OpenAI-compatible APIs. Selected with
  `--provider`; endpoint and credentials via `--url` / `--api-key` or env vars (`OLLAMA_BASE_URL`,
  `OLLAMA_API_KEY`, `LM_STUDIO_BASE_URL`, `LM_STUDIO_API_KEY`, `OPENAI_API_KEY`).
- Repeatable `-i/--input` accepting files and directories; `--ext` filters by extension,
  `-r/--recursive` walks subdirectories.
- Inference knobs: `-m/--model`, `--temperature`, `--max-tokens`, `--retries`, `--jpeg-dimensions`,
  `--jpeg-quality`. Each also has an env-var override (`MODEL_NAME`, `TEMPERATURE`, etc.).
- In-memory JPEG conversion to keep token usage low.
- Structured log files for debugging and auditing.

[0.1.0]: https://github.com/jbsilva/photo-tagger/releases/tag/v0.1.0
[0.2.0]: https://github.com/jbsilva/photo-tagger/compare/v0.1.0...v0.2.0
[0.2.1]: https://github.com/jbsilva/photo-tagger/compare/v0.2.0...v0.2.1
[0.2.2]: https://github.com/jbsilva/photo-tagger/compare/v0.2.1...v0.2.2
[unreleased]: https://github.com/jbsilva/photo-tagger/compare/v0.2.2...HEAD
