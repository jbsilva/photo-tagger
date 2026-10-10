# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--duty` and `--battery-floor` pace a long run: it works for a window, then rests, so the battery
  charges back, the fans wind down, and a library of hundreds of thousands of photos no longer needs
  the machine at full tilt for days. Rests happen at photo boundaries, are capped by
  `--max-pause-minutes`, and the summary reports what they cost. Also a `[pacing]` config table.

- GUI: **Settings > Pacing...** sets the same thing for the window's own runs, and the status bar
  says why a run is resting and for how long.

- `--artist` and `--copyright` write your credit and copyright notice alongside the AI fields, into
  EXIF, XMP, and IPTC, with `{year}` taken from each photo's own capture date. Also `--rights`,
  `--marked`, `--set TAG=VALUE`, `--no-write-copyright`, and a `[copyright]` config table.

- `photo-tagger set` writes those same fields on already-tagged photos, with no model call.

- `photo-tagger show` prints what a photo carries, naming the ExifTool tag behind every value.
  `--all` dumps every tag, `--json` pipes it elsewhere.

- GUI: **Save After Generating** writes each run's photos as soon as it finishes, so an overnight
  batch is one click. The toggle sits on the arrow of either **Generate** button, which then reads
  **Generate and Save**. The one set of **Save Options** sits beside it, because the write it makes
  is the write **Save** makes. Before saving, the run retries the photos that failed. Then it asks
  again for the photos whose text came back in two writing systems, where a second answer usually
  comes back in one. That call reads past the result cache and replaces what it finds. The closing
  line counts any photo still failing. The choice persists to `[gui]`.

- GUI: **Settings > Copyright...** sets the same credit and notice, previews the tags a save writes,
  and persists them to `[copyright]`. A copyright alone is enough to save a photo, and **Insert**
  buttons type `©`, `{year}`, and `{artist}` into the field holding the cursor. The save menu's
  **Write Copyright** skips the block without clearing it.

- `--otel` exports an OpenTelemetry trace of every model call (timings, tokens, failures) to a
  collector you run, so a slow or expensive batch is something you can read rather than guess at.
  Off by default and pointed only where `--otel-endpoint` says. The photo itself is never exported;
  generated text needs `--otel-content`. Needs the new `[otel]` extra.

- `--notify` raises a desktop notification when a batch ends, carrying its tally
  (`38 of 42 photos tagged, 4 failed.`); `--notify-sound` adds an audible alert. `watch` alerts per
  batch. In the GUI: **Settings > Notify When a Run Ends**. Uses the notifier the OS already ships
  (`osascript`, `notify-send`, a PowerShell toast), so no new dependency and no failure where there
  is none.

- Every run now checks the generated text for mixed writing systems and flags the photos that have
  them, so a stray `陶笛` in English output is something you are told about instead of something you
  find in the catalog later. The photo is still tagged: the flag names the words in a
  `foreign_terms` CSV column and `--json` key, a line in the log, and `flagged_files` in the
  `--summary-file`. No language list is involved, so it works for any `--output-language`. In the
  GUI a flagged photo gets an amber note over its editable fields and a `⚠` in the **Status**
  column. Its thumbnail gets an amber `!`. **Mixed writing systems** joins the **Select > Status**
  criteria and the folder grid's **Show** box, so the closing count turns into rows you can open.

- `--repair-language` asks the model again, with the same photo, for the flagged photos only, so
  `Decorated陶笛` comes back as `Decorated Ocarina`. One extra call per affected photo and nothing on
  the rest, so it is off by default; a repair that fails or comes back mixed keeps the original
  answer and the flag. In the GUI: **Settings > Rewrite Mixed-Language Output**.

- GUI: the **Remove** button's arrow offers **Remove Checked** and **Remove Unchecked**, so you can
  take photos off the list by checkbox, not only by selection.

### Changed

- `PHOTO_TAGGER_TELEMETRY` replaces `PHOTO_TAGGER_NO_TELEMETRY`. `0` turns telemetry off, and `1`
  sends it even with `DO_NOT_TRACK=1` set. A GUI toggle checked by hand also overrules
  `DO_NOT_TRACK`, and the toggle now shows what is actually sent.

- `--jpeg-dimensions` now defaults to 640 px instead of 1280. Prefill scales with the pixel count,
  so this is about a second less per photo on a local 30B model, and 640 px carries everything but
  small text in the frame. Raise it to 896 or 1024 for photos of signs, documents, or labels. A run
  at a new size starts a fresh cache namespace, so pin `jpeg_dimensions = 1280` in your config to
  keep both the old detail level and your existing cache.

### Fixed

- The desktop window now sends the configured `jpeg_dimensions` to the model instead of its own
  hardcoded 640 px preview size, so a GUI run and a CLI run ask the same question about a photo.
  Windows left on the default are unaffected, cache included.
- `max_tokens` no longer takes part in the cache namespace. A call that hits the cap yields nothing
  to store, so the cap cannot have shaped any cached answer, and raising it to escape a repetition
  loop no longer throws the cache away.
- A photo whose answer ran past `--max-tokens` is now asked again automatically, twice, instead of
  failing. The cap is only ever reached by a model that fell into a repetition loop (an endless
  `Animal<Animal` chain in its hierarchies), which the next attempt almost never repeats.
- The schema handed to the model now states the keyword and hierarchy caps it already enforced, and
  the prompt rules out repeated chains, a term as its own parent, and climbing past a broad
  category.
- pydantic-ai's first-run observability banner no longer prints. It landed in the middle of a batch,
  in the stdout stream `--json` consumers parse, and in the terminal behind the GUI window.
- The system prompt now rules out a foreign word used *as* the concept name (a title reading
  `Decorated陶笛 with Geometric Pattern` when the output language is English). Its language rule only
  covered mixed spelling before.
- GUI: **Uncheck Already Tagged** no longer freezes the window. It now goes by the **Tagged**
  column's background scan instead of reading every photo from the click handler, and waits for that
  scan when it is still running.
- GUI: the metadata scan behind the **Tagged** column reads in small batches, so the column fills in
  as it goes and a long scan (thousands of photos, or a network share) shows a progress bar with a
  **Stop** button under the list.
- GUI: removing a folder while that scan is running stops it from reading that folder, and takes
  those photos out of the progress bar's total.

## [0.9.0] - 2026-09-18

### Added

- `--sidecar-mode {all,none,raw,both}` picks the write target: a sidecar per photo, inside each
  photo, `raw` to split by file type (sidecar for RAW, embedded for the rest), or `both` to write
  each photo and a sidecar beside it. `--write-sidecar` / `--embed-in-photo` are now shorthand. In
  the GUI: the save menu's **Write To** submenu.
- `--preserve-title` and `--preserve-description` keep a title or description the photo already has,
  filling in only the photos without one. Both still default to replacing. In the GUI: the save
  menu's **Overwrite Existing** submenu, one entry per field.
- GUI: a search box above the photo list, matching by text or glob (`*_edit.jpg`). It only narrows
  what the list shows; hidden photos are still generated and saved.
- GUI: the photo list can be shown flat (**View as List**), so a sort runs across every photo
  instead of inside each folder.
- GUI: the **Select** menu can check, uncheck, or check only the photos matching a file type,
  status, or name pattern.
- GUI: clicking the preview opens the photo in a window that zooms up to 8:1 and pans.
- GUI: both keyword columns now render and sort the same way, so they read line by line. Each
  hierarchy shows once, as its deepest chain, instead of every leaf appearing twice.
- GUI: each existing title and description says which file it came from, and names a value the
  sidecar shadows (`shadows "default" in the image file`).

### Fixed

- Embedding metadata works on photos whose maker notes ExifTool cannot parse, which some cameras
  write. ExifTool refused the write; it is retried once ignoring minor errors, and logged. Maker
  notes and image data come through byte for byte.
- Embedding a description replaces `EXIF:ImageDescription` too, not only its XMP copies. A
  camera-written placeholder used to survive every save in IFD0.
- An existing title and description are read from the XMP sidecar first, then the image file. A
  stale caption in the photo shadowed the one just written beside it.
- A tag that is present but empty no longer shadows the tag photo-tagger falls back to.
- `photo-tagger vocabulary --report` quotes a keyword a spreadsheet would run as a formula, as
  `--csv-file` always has.
- GUI: saving settings into a config file with a syntax error reports the problem instead of
  crashing the window.
- Merging keywords no longer writes a duplicate to `XMP-lr:WeightedFlatSubject`.
- A run that fails while starting up closes the `--cache-file` it had already opened.
- The telemetry probes and the GUI's **Reveal in Finder** run their helper program by absolute path,
  so Windows cannot pick one up from the photo folder instead.
- GUI: a generate run that dies while starting up no longer leaves the window disabled.

## [0.8.0] - 2026-09-03

### Added

- `--vocabulary PATH`: keep generated keywords inside a controlled vocabulary, so a curated catalog
  stops collecting near-duplicates. Reads a Lightroom keyword export or a plain list. Matching
  ignores case, punctuation, and typos, and every match takes the file's own spelling and hierarchy.
- `--vocabulary-strict`: drop a keyword the vocabulary does not cover instead of writing it as it
  came.
- `--session-gap MINUTES`: group the batch into shoots by capture time and make each shoot agree
  with itself, so forty frames of one bird stop reaching the catalog as `Osprey`, `Ospreys`, and
  `Bird|Osprey`.
- `photo-tagger vocabulary`: build that vocabulary from the keywords your library already uses, read
  off the photos with ExifTool. `--organize` adds an opt-in model pass that folds synonyms and adds
  a hierarchy. Nothing is written to your photos.
- `photo-tagger undo`: put back what a run wrote, from the journal every run now records. On by
  default (`--no-undo-log`).
- `photo-tagger watch`: tag photos as they arrive, each one once it stops changing.
- GUI: all of the above, without a terminal. Keyword rules live in **Settings**, the three commands
  in **Tools**.
- GUI: **Back** and **Forward** over the photos and folder grids you have visited, as arrows above
  the pane and in a new **Go** menu. Opening a photo from a thumbnail grid used to be a one-way
  trip.

### Changed

- The HTTP dependency moved from `httpx` to `httpx2`.

### Fixed

- `--skip-tagged` and the GUI's **Tagged** column credit every photo that shares one XMP sidecar, so
  a RAW and JPEG pair no longer reads as half untagged.
- `>` separates a keyword hierarchy even when mixed with `<`.
- Keyword files, skip lists, and the prompt file are read as `utf-8-sig`, so a byte-order mark no
  longer corrupts the first line.
- `--append-to-skip-file` is safe when two runs share one file.
- A rejected credential (401 or 403) skips the retry pass instead of paying for a second round of
  failing requests.
- Tokens spent on failed calls are counted, so the summary no longer understates spend.
- Tagging a batch that turns out to be empty no longer truncates the previous `--csv-file` report.
- `--log-folder` is validated when the flags are parsed, and a logging failure exits cleanly.
- `photo-tagger doctor` reports its warning about a redirected ExifTool binary.
- A missing `exiftool` binary no longer raises a bare traceback out of a metadata read or write.
  Reads degrade with a logged warning, as they always did for other ExifTool errors, and a write
  reports the photo as failed.
- The first run no longer stalls on exit while telemetry probes the hardware.
- GUI: removing a folder from a large list, quitting the window, and the first drag-and-drop no
  longer crash.
- GUI: closing with unsaved proposals asks first.
- GUI: the window no longer freezes on a folder that is still loading or a metadata scan of a large
  folder, **Cancel** during a save is honored, and **Clear List** is refused while a save runs.
- GUI: unchecking a subfolder repaints the folders above it.
- GUI: a single-photo **Save** reports an ExifTool error instead of dying silently.
- GUI: a per-photo **hint** no longer enters the shared cache.

### Security

- The LM Studio provider no longer falls back to `OPENAI_API_KEY`. It is the default provider, so a
  real OpenAI key went as a bearer token to whatever `--url` pointed at.
- The API key is redacted from provider error logs and messages.
- CSV report cells that start with `=`, `+`, `-`, or `@` are quoted, so a file name or a generated
  title cannot run as a spreadsheet formula (CWE-1236).
- Log files are created owner-only (mode 0600) on POSIX systems.

## [0.7.0] - 2026-08-08

### Added

- GUI: the bottom bar now shows the time **elapsed** and the estimated time **left** next to the
  progress bar, for both generating and saving (`2:30 elapsed · 8:10 left`).

- GUI: a **Keep ExifTool Backup** toggle in the Save options menu (the CLI's `--no-backup-xmp`).
  Saving used to always leave a `*_original` copy of every photo behind, which on a large batch
  doubles the disk space used. It defaults to on and persists via *Save Settings as Defaults*.

### Changed

- GUI: **Save Selected** now writes on a background thread (sharing one ExifTool process for the
  batch) instead of blocking the window. Saving a large batch used to freeze the interface behind
  the OS busy cursor with no sign of progress, which looked like a crash. It now reports per photo,
  and **Cancel** stops it after the file in flight.

## [0.6.0] - 2026-07-11

### Added

- Anonymous crash reports: an unhandled exception sends the exception *type* and its code location
  inside photo-tagger (`module:function:line`), never the error message (messages can embed paths).
  Same opt-outs as the usage telemetry; capped per process. The one-time telemetry notice re-shows
  once for existing installs so the expanded disclosure is seen.
- Telemetry now reports coarse hardware facts (CPU/GPU model, logical cores, RAM in whole GB; probed
  once and cached), per-run outcome counters (successes, failures, cache hits, retry recoveries,
  workers, token totals, model time, dry-run flag), and failure buckets (fixed labels like `timeout`
  or `metadata-write`) so breakage is measurable, not anecdotal.
- The telemetry dashboard grew sections for adoption, usage, hardware & performance, and reliability
  & crashes, plus a global interface filter (all/CLI/GUI).
- The inference cache prunes entries older than 180 days when it opens, so `cache.sqlite` stops
  growing forever (every settings change orphans its old namespace).
- A short pause before the retry pass, so an overloaded model server is not immediately re-hit.

### Changed

- `--append-to-skip-file` records full paths instead of bare filenames, so duplicate camera names
  across folders (`A/IMG_0001.CR3`, `B/IMG_0001.CR3`) can no longer make a resumed run silently skip
  unprocessed photos. Old name-only skip files keep working.
- Config-file values now pass through the same conversion and validation as CLI flags, so a mistyped
  value fails with a clean error at startup instead of a traceback mid-run.
- Photos with identical pixel content in one concurrent run share a single model call instead of
  each paying their own.
- CSV reports are written as `utf-8-sig` so Excel renders accented titles and keywords correctly.

### Fixed

- Passing any CLI flag from an option group no longer silently resets the group's other config-file
  values to built-in defaults (e.g. `--temperature` used to revert a configured `max_tokens`).
- `--dry-run` no longer appends previewed photos to `--append-to-skip-file`, which made later real
  runs skip photos that never got metadata.
- The description mirror is written to `XMP-tiff:ImageDescription`; the previous `XMP-exif` tag was
  silently rejected by ExifTool, so the mirror never reached the file.
- Keyword capitalization no longer corrupts acronyms and apostrophes (`NYC` stays `NYC` instead of
  becoming `Nyc`; `bird's nest` no longer becomes `Bird'S Nest`).
- One corrupt or zero-byte photo no longer makes the whole folder read as untagged (and get
  re-tagged): the batched ExifTool reads salvage the healthy files, and `--skip-tagged` and the
  GUI's Tagged column now distinguish "could not read" from "no metadata".
- Windows: file locking works again (the PID note into the held lock file raised on every
  acquisition), and the batched metadata reads no longer miss every file over path-separator
  differences.
- Ctrl-C during concurrent batch start-up is honored instead of grinding through every queued photo;
  never-submitted photos are reported as pending.
- A single over-long or blank keyword from the model no longer burns the whole validation retry
  budget (full vision calls); items are cleaned instead.
- GUI: streamed-in thumbnails no longer swallow the next grid click; a non-domain error during
  generation no longer leaves the window stuck in the running state; removing photos prunes the
  visible grid and thumbnail cache; bulk actions no longer kick you out of the photo or folder being
  reviewed.
- The `gui` command only suggests installing the `[gui]` extra when PySide6/shiboken6 is actually
  missing; other import errors surface as themselves.
- A failed cache initialization no longer leaks its SQLite connection.

## [0.5.0] - 2026-07-09

### Added

- The GUI is translated at runtime (gettext) and ships a complete Brazilian Portuguese (`pt_BR`)
  translation. Pick a language with **Settings > Language**, the `language` config key, or
  `PHOTO_TAGGER_LANG`; the default follows the OS locale.
- New `llamacpp` provider for llama.cpp's `llama-server`. Default endpoint
  `http://localhost:8080/v1`; env vars `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_API_KEY` (the key is only
  needed when `llama-server` was started with `--api-key`).
- Anonymous, opt-out usage telemetry with a one-time first-run notice. Disable it with
  `--no-telemetry`, `PHOTO_TAGGER_NO_TELEMETRY=1` (or `DO_NOT_TRACK=1`), `enabled = false` under
  `[telemetry]` in the config file, or the GUI's **Settings > Send Anonymous Telemetry** toggle.
- GUI redesign: split **Generate**/**Save** buttons with option menus, thumbnail badges,
  multi-selection with bulk actions, tree columns for file type, status, and already-tagged,
  collapsible keyword-change details, reveal-in-file-manager on Windows, Linux, and macOS, and CSV
  export.
- The GUI now caches results by default (sharing the CLI's cache format), with skip-cache actions
  for forcing a fresh generation.
- **Settings > Save Settings as Defaults** merges into the existing config file instead of rewriting
  it, preserving comments and unknown keys.
- Windowed `photo-tagger-gui` entry point, so the desktop app launches without a console window.
- `exiftool_path` config key and `PHOTO_TAGGER_EXIFTOOL` env var for ExifTool installs not on
  `PATH`. A GUI launched from Finder also inherits the login shell's `PATH` automatically.
- `packaging/build_macos_app.sh` builds a standalone, double-clickable macOS app with PyInstaller.

### Changed

- Console logging is capped at `INFO` from import (so nothing above it leaks before the CLI flags
  apply), and file logs are serialized as JSON lines.
- System prompt now demands English-only, single-script output and at most one hierarchy chain per
  keyword.

### Fixed

- The GUI's telemetry beacon is flushed before the process exits instead of being lost.
- The GUI result cache is keyed on the image content hash, so **Embed in Photo** no longer
  invalidates it.
- Hierarchical keywords written with `>` separators are parsed as hierarchies instead of being
  mangled, and model keywords are deduplicated.
- Ctrl-C during a concurrent batch no longer miscounts photos that finished while the pool drained.

## [0.4.0] - 2026-06-26

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
[0.4.0]: https://github.com/jbsilva/photo-tagger/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/jbsilva/photo-tagger/compare/v0.4.0...v0.5.0
[0.6.0]: https://github.com/jbsilva/photo-tagger/compare/v0.5.0...v0.6.0
[0.7.0]: https://github.com/jbsilva/photo-tagger/compare/v0.6.0...v0.7.0
[0.8.0]: https://github.com/jbsilva/photo-tagger/compare/v0.7.0...v0.8.0
[0.9.0]: https://github.com/jbsilva/photo-tagger/compare/v0.8.0...v0.9.0
