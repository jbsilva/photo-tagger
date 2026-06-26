# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The `--cache-file` inference cache is now keyed on the image data only (ExifTool's
  `ImageDataHash`) instead of a hash of the whole file. Writing metadata into a photo no longer
  changes the key, so a rerun over the same folder hits the cache even after `--embed-in-photo`
  wrote tags on the first pass. Formats ExifTool cannot hash that way fall back to the whole-file
  hash. Existing cache files are simply repopulated on the next run.

## [0.3.0] - 2026-06-22

### Added

- Optional desktop GUI (`photo-tagger gui`), via the `gui` extra
  (`pip install 'photo-tagger[gui]'`). A PySide6 review-before-write frontend: drag in photos or
  folders, generate proposals, then review and edit each photo's title, description, and keywords
  (with a live Lightroom-hierarchy preview) before saving. PySide6 is imported lazily, so the base
  CLI never depends on Qt.
- New `openai` provider for any hosted OpenAI-compatible endpoint. Set `OPENAI_BASE_URL` / `--url`
  and `OPENAI_API_KEY` / `--api-key`; fails fast when no key is configured.
- New `photo-tagger doctor` command: a pre-flight check that ExifTool is on PATH and the provider is
  reachable and serves the requested model, exiting non-zero on failure.
- `--csv-file PATH` on the `tag` command writes a per-photo CSV report (existing/written metadata,
  camera, location, GPS, usage, timing), alongside `--summary-file` and `--json`.
- `--write-keywords` / `--no-write-keywords` (default on) refreshes the title and description while
  leaving existing keywords on disk untouched.
- A PEP 561 `py.typed` marker so downstream projects can consume the package's type hints.

### Changed

- Backends now live in a `photo_tagger.providers` registry; adding a backend is a single entry.
- Existing keywords use a typed `KeywordSet` value object instead of a bare `dict[str, list[str]]`.
- The package version is read from installed distribution metadata, so `pyproject.toml` is the only
  code-side source of truth.
- CLI option groups moved out of `main.py` into `photo_tagger.cli_options`.

### Fixed

- Hierarchical keywords are generated reliably again: the model schema now has a dedicated
  `hierarchies` field for taxonomy chains (`Golden Eagle<Bird of Prey<Animal`) instead of expecting
  `<` embedded in the flat `keywords` list, which the model had stopped doing. (The CLI cache is
  keyed on the user prompt only, so delete a stale `--cache-file` to pick this up on
  already-processed photos; the GUI never caches.)

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
[0.3.0]: https://github.com/jbsilva/photo-tagger/compare/v0.2.2...v0.3.0
