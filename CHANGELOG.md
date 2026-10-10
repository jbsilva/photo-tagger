# Changelog

This file lists all notable changes to this project. The format follows the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) guide. The version numbers follow the
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) rules.

## [Unreleased]

### Added

- `--sidecar-override EXT=TARGET` (`sidecar_overrides` in the configuration file) sends one file
  type elsewhere, whatever `--sidecar-mode` says. `dng=photo` writes into each DNG, as Lightroom
  Classic does.

- `--duty` and `--battery-floor` pace a long run. The run works for a window, then rests, so the
  battery charges back and the fans wind down. This means a library of hundreds of thousands of
  photos no longer needs the machine at full tilt for days. Rests happen at photo boundaries and
  `--max-pause-minutes` caps them. The summary reports what the rests cost. There is also a
  `[pacing]` configuration table.

- GUI: **Settings > Pacing...** sets the same thing for the window's own runs. The status bar says
  why a run is resting and for how long.

- `--artist` and `--copyright` write your credit and copyright notice alongside the AI fields, into
  EXIF, XMP, and IPTC. `{year}` comes from each photo's own capture date. Also `--rights`,
  `--marked`, `--set TAG=VALUE`, `--no-write-copyright`, and a `[copyright]` configuration table.

- `photo-tagger set` writes those same fields on already-tagged photos, with no model call.

- `photo-tagger show` prints what a photo carries. It names the ExifTool tag behind every value.
  `--all` dumps every tag, and `--json` pipes it elsewhere.

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

- GUI: **No copyright** joins the **Select > Status** criteria and the grid's **Show** box. It finds
  the photos that have no copyright notice in EXIF, IPTC, or XMP.

- GUI: with a copyright set, **Save Selected** also stamps it on the checked photos that you did not
  generate, and leaves their text alone. You can add a notice to tagged photos without a model call.

- GUI: a `C` in the **Tagged** column and a `C` badge on the thumbnail mark a photo that has a
  copyright notice. A **Copyright** row in the detail pane shows the notice on the file next to the
  one that a save writes.

- `--otel` exports an OpenTelemetry trace of every model call (timings, tokens, failures) to a
  collector you run, so a slow or expensive batch is something you can read rather than guess at.
  Off by default and pointed only where `--otel-endpoint` says. The photo itself is never exported.
  Generated text needs `--otel-content`. Needs the new `[otel]` extra.

- When a batch ends, `--notify` raises a desktop notification that carries its tally
  (`38 of 42 photos tagged, 4 failed.`). `--notify-sound` adds an audible alert, and `watch` alerts
  per batch. In the GUI: **Settings > Notify When a Run Ends**. It uses the notifier the OS already
  ships (`osascript`, `notify-send`, or a PowerShell toast). That means no new dependency and no
  failure where there is none.

- Every run now checks the generated text for mixed writing systems and flags the photos that have
  them, so a stray `陶笛` in English output is something you are told about instead of something you
  find in the catalog later. The photo is still tagged: the flag names the words in a
  `foreign_terms` CSV column and `--json` key, a line in the log, and `flagged_files` in the
  `--summary-file`. No language list is involved, so it works for any `--output-language`. In the
  GUI a flagged photo gets an amber note over its editable fields and a `⚠` in the **Status**
  column. Its thumbnail gets an amber `!`. **Mixed writing systems** joins the **Select > Status**
  criteria and the folder grid's **Show** box, so the closing count turns into rows you can open.

- `--repair-language` asks the model again, with the same photo, for the flagged photos only, so
  `Decorated陶笛` comes back as `Decorated Ocarina`. It makes one extra call per affected photo and
  nothing on the rest, so it is off by default. A repair that fails or comes back mixed keeps the
  original answer and the flag. In the GUI: **Settings > Rewrite Mixed-Language Output**.

- GUI: the **Remove** button's arrow offers **Remove Checked** and **Remove Unchecked**, so you can
  take photos off the list by checkbox, not only by selection.

- Telemetry now reports the configuration and features you use, as switches, counts, and mode names,
  and how each run ends. In the GUI, each generation and save is a run of its own. The session tells
  how long the window was busy, in use, or idle. [docs/telemetry.md](docs/telemetry.md) lists every
  field. You see the notice once more.

- Crash reports now carry the stack trace as modules, functions, and lines (never a path or the
  message), and a failed photo reports its error class and HTTP status.

### Changed

- `PHOTO_TAGGER_TELEMETRY` replaces `PHOTO_TAGGER_NO_TELEMETRY`. `0` turns telemetry off, and `1`
  sends it even with `DO_NOT_TRACK=1` set. A GUI toggle checked by hand also overrules
  `DO_NOT_TRACK`, and the toggle now shows what is actually sent.

- The telemetry collector now keeps events with no expiry date. Before, they expired after three
  months.

- The telemetry dashboard reads the whole D1 history and filters by install. It leaves out the
  machine it runs on, and it charts GUI runs, sessions, and configuration.

- `--jpeg-dimensions` now defaults to 640 px instead of 1280. Prefill scales with the pixel count,
  so this is about a second less per photo on a local 30B model. 640 px carries everything but small
  text in the frame. Raise it to 896 or 1024 for photos of signs, documents, or labels. A run at a
  new size starts a fresh cache namespace. Pin `jpeg_dimensions = 1280` in your configuration to
  keep both the old detail level and your existing cache.

### Fixed

- A save writes only the tags whose value changed. If the photo already has every value, the file is
  not touched, so its timestamp stays and a backup tool such as rsync does not copy it again.
- The desktop window now sends the configured `jpeg_dimensions` to the model instead of its own
  hardcoded 640 px preview size, so a GUI run and a CLI run ask the same question about a photo.
  Windows left on the default are unaffected, cache included.
- `max_tokens` no longer takes part in the cache namespace. A call that hits the cap yields nothing
  to store, so the cap never shaped any cached answer. Raising the cap to escape a repetition loop
  no longer throws the cache away.
- A photo whose answer ran past `--max-tokens` is now asked again automatically, twice, instead of
  failing. The cap is only ever reached by a model that fell into a repetition loop, an endless
  `Animal<Animal` chain in its hierarchies. The next attempt almost never repeats it.
- The schema handed to the model now states the keyword and hierarchy caps it already enforced. The
  prompt also rules out repeated chains, a term as its own parent, and climbing past a broad
  category.
- pydantic-ai's first-run observability banner no longer prints. It landed on stderr in the middle
  of a batch, between the log lines, and in the terminal behind the GUI window.
- The system prompt now rules out a foreign word used *as* the concept name. When the output
  language is English, a title cannot read `Decorated陶笛 with Geometric Pattern`. Its language rule
  only covered mixed spelling before.
- GUI: **Uncheck Already Tagged** no longer freezes the window. It now uses the **Tagged** column's
  background scan instead of reading every photo from the click handler. If the scan is still
  running, it waits for the scan to finish.
- GUI: the metadata scan behind the **Tagged** column reads in small batches, so the column fills in
  as it goes. A long scan (thousands of photos, or a network share) shows a progress bar with a
  **Stop** button under the list.
- GUI: removing a folder while that scan is running stops it from reading that folder, and takes
  those photos out of the progress bar's total.

## [0.9.0] - 2026-09-18

### Added

- `--sidecar-mode {all,none,raw,both}` picks the write target. `all` writes a sidecar per photo, and
  `none` writes inside each photo. `raw` splits by file type: a sidecar for RAW, embedded metadata
  for the rest. `both` writes each photo and a sidecar beside it. `--write-sidecar` /
  `--embed-in-photo` are now shorthand. In the GUI: the save menu's **Write To** submenu.
- `--preserve-title` and `--preserve-description` keep a title or description the photo already has.
  They fill in only the photos without one. Both still default to replacing. In the GUI: the save
  menu's **Overwrite Existing** submenu, one entry per field.
- GUI: a search box above the photo list matches by text or glob (`*_edit.jpg`) and only narrows
  what the list shows. Hidden photos are still generated and saved.
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

- Embedding metadata now works on photos whose maker notes ExifTool cannot parse, which some cameras
  write. Before, ExifTool refused the write. Now, the write is retried once with minor errors
  ignored, and the event is logged. Maker notes and image data come through byte for byte.
- Embedding a description replaces `EXIF:ImageDescription` too, not only its XMP copies. A
  camera-written placeholder used to survive every save in IFD0.
- An existing title and description are read from the XMP sidecar first, then the image file. A
  stale caption in the photo shadowed the one just written beside it.
- A tag that is present but empty no longer shadows the tag photo-tagger falls back to.
- `photo-tagger vocabulary --report` now quotes a keyword that a spreadsheet can run as a formula.
  `--csv-file` already quotes it the same way.
- GUI: saving configuration into a configuration file with a syntax error reports the problem
  instead of crashing the window.
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
- GUI: **Back** and **Forward** over the photos and folder grids you visited, as arrows above the
  pane and in a new **Go** menu. Opening a photo from a thumbnail grid used to be a one-way trip.

### Changed

- The HTTP dependency moved from `httpx` to `httpx2`.

### Fixed

- `--skip-tagged` and the GUI's **Tagged** column credit every photo that shares one XMP sidecar, so
  a RAW and JPEG pair no longer reads as half untagged.
- When `<` is also mixed in, `>` still separates a keyword hierarchy.
- Keyword files, skip lists, and the prompt file are read as `utf-8-sig`, so a byte-order mark no
  longer corrupts the first line.
- When two runs share one file, `--append-to-skip-file` is safe.
- A rejected credential (401 or 403) skips the retry pass instead of paying for a second round of
  failing requests.
- Tokens spent on failed calls are counted, so the summary no longer understates spend.
- Tagging a batch that turns out to be empty no longer truncates the previous `--csv-file` report.
- When photo-tagger parses the flags, it makes sure that `--log-folder` is valid, and a logging
  failure exits cleanly.
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
  Same opt-outs as the usage telemetry. It is capped per process. The one-time telemetry notice
  re-shows once for existing installs so the expanded disclosure is seen.
- Telemetry now reports coarse hardware facts: CPU/GPU model, logical cores, and RAM in whole GB. It
  probes these once and caches them. Telemetry also reports per-run outcome counters: successes,
  failures, cache hits, retry recoveries, workers, token totals, model time, and the dry-run flag.
  It reports failure buckets too, with fixed labels like `timeout` or `metadata-write`. This makes
  breakage measurable instead of anecdotal.
- The telemetry dashboard grew sections for adoption, usage, hardware & performance, and reliability
  & crashes, plus a global interface filter (all/CLI/GUI).
- When it opens, the inference cache prunes entries older than 180 days, so `cache.sqlite` stops
  growing forever (every configuration change orphans its old namespace).
- A short pause before the retry pass, so an overloaded model server is not immediately re-hit.

### Changed

- `--append-to-skip-file` records full paths instead of bare filenames. This means duplicate camera
  names across folders (`A/IMG_0001.CR3`, `B/IMG_0001.CR3`) can no longer make a resumed run
  silently skip unprocessed photos. Old name-only skip files keep working.
- Configuration file values now pass through the same conversion and validation as CLI flags. A
  mistyped value now fails with a clean error at startup instead of a traceback mid-run.
- Photos with identical pixel content in one concurrent run share a single model call instead of
  each paying their own.
- CSV reports are written as `utf-8-sig` so Excel renders accented titles and keywords correctly.

### Fixed

- Passing any CLI flag from an option group no longer silently resets the group's other
  configuration file values to built-in defaults. For example, `--temperature` used to revert a
  configured `max_tokens`.
- `--dry-run` no longer appends previewed photos to `--append-to-skip-file`, which made later real
  runs skip photos that never got metadata.
- The description mirror is written to `XMP-tiff:ImageDescription`. The previous `XMP-exif` tag was
  silently rejected by ExifTool, so the mirror never reached the file.
- Keyword capitalization no longer corrupts acronyms and apostrophes. `NYC` stays `NYC` instead of
  becoming `Nyc`, and `bird's nest` no longer becomes `Bird'S Nest`.
- One corrupt or zero-byte photo no longer makes the whole folder read as untagged, and it no longer
  gets re-tagged as a result. The batched ExifTool reads salvage the healthy files. `--skip-tagged`
  and the GUI's Tagged column now tell a file that ExifTool cannot read apart from a file with no
  metadata.
- Windows: file locking works again. Before, the PID note written into the held lock file raised an
  error on every acquisition. The batched metadata reads no longer miss every file because of
  path-separator differences.
- Ctrl-C during concurrent batch start-up is honored instead of grinding through every queued photo.
  Never-submitted photos are reported as pending.
- A single over-long or blank keyword from the model no longer burns the whole validation retry
  budget (full vision calls). The items are cleaned instead.
- GUI: streamed-in thumbnails no longer swallow the next grid click. A non-domain error during
  generation no longer leaves the window stuck in the running state. Removing photos prunes the
  visible grid and thumbnail cache. Bulk actions no longer kick you out of the photo or folder you
  are reviewing.
- When PySide6/shiboken6 is actually missing, the `gui` command suggests installing the `[gui]`
  extra. Other import errors surface as themselves.
- A failed cache initialization no longer leaks its SQLite connection.

## [0.5.0] - 2026-07-09

### Added

- The GUI is translated at runtime (gettext) and ships a complete Brazilian Portuguese (`pt_BR`)
  translation. Pick a language with **Settings > Language**, the `language` configuration key, or
  `PHOTO_TAGGER_LANG`. The default follows the OS locale.
- New `llamacpp` provider for llama.cpp's `llama-server`. Default endpoint
  `http://localhost:8080/v1`. Env vars are `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_API_KEY`. When
  `llama-server` starts with `--api-key`, it needs `LLAMA_CPP_API_KEY`.
- Anonymous, opt-out usage telemetry with a one-time first-run notice. Disable it with
  `--no-telemetry`, `PHOTO_TAGGER_NO_TELEMETRY=1` (or `DO_NOT_TRACK=1`), `enabled = false` under
  `[telemetry]` in the configuration file, or the GUI's **Settings > Send Anonymous Telemetry**
  toggle.
- GUI redesign: split **Generate**/**Save** buttons with option menus, and thumbnail badges.
  Multi-selection now has bulk actions, and tree columns cover file type, status, and
  already-tagged. Keyword-change details are collapsible. Reveal-in-file-manager works on Windows,
  Linux, and macOS. CSV export is included.
- The GUI now caches results by default (sharing the CLI's cache format), with skip-cache actions
  for forcing a fresh generation.
- **Settings > Save Settings as Defaults** merges into the existing configuration file instead of
  rewriting it. It keeps comments and unknown keys.
- Windowed `photo-tagger-gui` entry point, so the desktop app launches without a console window.
- `exiftool_path` configuration key and `PHOTO_TAGGER_EXIFTOOL` env var for ExifTool installs not on
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
  changes the key. A rerun over the same folder hits the cache, even after `--embed-in-photo` wrote
  tags on the first pass. Formats ExifTool cannot hash that way fall back to the whole-file hash.
  Existing cache files are repopulated on the next run.

## [0.3.0] - 2026-06-22

### Added

- Optional desktop GUI (`photo-tagger gui`), via the `gui` extra
  (`pip install 'photo-tagger[gui]'`). A PySide6 review-before-write frontend: drag in photos or
  folders, generate proposals, then review and edit each photo's title, description, and keywords
  (with a live Lightroom-hierarchy preview) before saving. PySide6 is imported lazily, so the base
  CLI never depends on Qt.
- New `openai` provider for any hosted OpenAI-compatible endpoint. Set `OPENAI_BASE_URL` / `--url`
  and `OPENAI_API_KEY` / `--api-key`. When no key is configured, it fails fast.
- New `photo-tagger doctor` command: a pre-flight check that ExifTool is on PATH and the provider is
  reachable and serves the requested model. It exits non-zero on failure.
- `--csv-file PATH` on the `tag` command writes a per-photo CSV report (existing/written metadata,
  camera, location, GPS, usage, timing), alongside `--summary-file` and `--json`.
- `--write-keywords` / `--no-write-keywords` (default on) refreshes the title and description while
  leaving existing keywords on disk untouched.
- A PEP 561 `py.typed` marker so downstream projects can consume the package's type hints.

### Changed

- Backends now live in a `photo_tagger.providers` registry. Adding a backend is a single entry.
- Existing keywords use a typed `KeywordSet` value object instead of a bare `dict[str, list[str]]`.
- The package version is read from installed distribution metadata, so `pyproject.toml` is the only
  code-side source of truth.
- CLI option groups moved out of `main.py` into `photo_tagger.cli_options`.

### Fixed

- Hierarchical keywords are generated reliably again. The model schema has a dedicated `hierarchies`
  field for taxonomy chains (`Golden Eagle<Bird of Prey<Animal`), instead of expecting embedded `<`
  in the flat `keywords` list. The model had stopped doing that. (The CLI cache is keyed on the user
  prompt only, so delete a stale `--cache-file` to pick this up on already-processed photos. The GUI
  never caches.)

## [0.2.2] - 2026-05-30

### Added

- MIT License. The `LICENSE` file now ships in the sdist via PEP 639 `license-files`. The deprecated
  `License :: OSI Approved :: MIT License` classifier was dropped.

### Changed

- System prompt now forbids emitting the camera body, lens model, or capture timestamp as keywords.
  These describe equipment, not subject content. Earlier runs sometimes copied literal EXIF strings,
  for example `Canon Eos R5M2` and `Rf200-800Mm F6.3-9 Is Usm`, into the keyword list.

### Fixed

- `create_agent` no longer has a code path that used to leave the provider unbound for a value
  outside the supported set. Provider construction moved into `_build_provider`, which returns from
  each branch and ends in `assert_never`. This keeps the match exhaustive for the type checker.

## [0.2.1] - 2026-05-27

### Fixed

- When the TOML file, not the CLI, set `summary_file` (or another `Path` field), a successful run
  raised an error while writing the summary file. The error was
  `AttributeError: 'str' object has no attribute 'parent'`. `apply_overrides` now coerces string
  TOML values to `Path` for fields annotated as `Path` or `Path | None`.

## [0.2.0] - 2026-05-26

### Added

- TOML configuration file. Search order: `$PHOTO_TAGGER_CONFIG`, `./.photo-tagger.toml`,
  `~/.config/photo-tagger/config.toml`. CLI flags still win. See
  [`.photo-tagger.example.toml`](.photo-tagger.example.toml) for a template.
- `--workers N` for thread-pool concurrency (default 1). The model server is usually the bottleneck.
- `--cache-file PATH` SQLite cache of model outputs keyed by image content hash plus model, prompt,
  and sampling configuration. When nothing relevant changed, reruns skip the model entirely. WAL
  mode is enabled.
- `--lock-file PATH` is an exclusive file lock. If another `photo-tagger` already holds it, the new
  run refuses to start. Cross-platform (Linux, macOS, Windows).
- `--summary-file PATH` writes a JSON run summary on completion (success counts, failed files, token
  usage, wall time). The write is atomic, and photo-tagger creates the parent directory.
- `--json` emits one NDJSON line per processed photo on stdout. Logs and progress stay on stderr so
  `| jq` works.
- `--skip-tagged` skips files whose image or sidecar already has keywords, description, or title
  (catches photos tagged in Lightroom or by hand).
- `--append-to-skip-file PATH` records each successful filename so a later run with
  `--skip-from PATH` resumes where this one stopped.
- `--newer-than` / `--older-than` ISO 8601 mtime filters. Naive timestamps are read as local time.
- `--prompt-file PATH` replaces the default user prompt with file contents. Existing photo metadata
  is still appended.
- `--max-keywords N` caps the AI keyword count per photo before merging with existing tags.
- `--dry-run` runs the model and logs the proposed metadata without writing.
- `--timeout-seconds` is a per-image hard cap. The retry loop handles the abort.
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
- Progress bar routed to stderr. It stays clean alongside `--json`.
- Existing keywords are de-duplicated case-insensitively at read time.
- When persisting merged keywords, `WeightedFlatSubject` is now written back.
- `--ext` matches case-insensitively. Its default is aligned with the README.

### Fixed

- When the PID file write failed after lock acquire, the lock leaked.
- `parse_hierarchical_keyword` returning `['']` for empty input.
- StubAgent exposing `usage` as a callable instead of an attribute.
- Pydantic-AI deprecation: access `result.usage` as a property.
- EXIF orientation is now honored. PIL file handles are closed eagerly.
- Over-long keyword lists are truncated rather than failing validation.
- Cache and lock startup errors degrade to warnings instead of failing the run.
- Cache I/O errors are treated as warnings, not photo failures.
- Skip-list appender is now thread-safe under `--workers > 1`.
- Numeric env vars (`JPEG_QUALITY`, `TEMPERATURE`, and others) parse safely with a warning instead
  of crashing.

### Security

- `--api-key` warns that CLI args are visible in process listings. Prefer env vars instead
  (`OLLAMA_API_KEY`, `LM_STUDIO_API_KEY`, `OPENAI_API_KEY`).
- Lock file permissions tightened to `0o600`.

## [0.1.0] - 2026-02-16

Initial release.

### Added

- `photo-tagger` CLI that asks a vision-language model to analyze each photo and writes
  Lightroom-compatible metadata (title, one-sentence description, and hierarchical keywords).
- RAW and standard image support: CR3, CR2, NEF, JPG, PNG, and more.
- XMP sidecars are used by default. `--embed-in-photo` writes metadata directly into the image
  instead.
- By default, keywords merge with existing metadata. `--overwrite-keywords` replaces them instead.
  `--no-write-title` / `--no-write-description` skip those fields.
- `--no-backup-xmp` to skip the ExifTool `_original` snapshot.
- Provider support for Ollama and LM Studio via their OpenAI-compatible APIs. Select the provider
  with `--provider`. Set the endpoint and credentials via `--url` / `--api-key`, or with env vars
  (`OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `LM_STUDIO_BASE_URL`, `LM_STUDIO_API_KEY`,
  `OPENAI_API_KEY`).
- Repeatable `-i/--input` accepts files and directories. `--ext` filters by extension, and
  `-r/--recursive` walks subdirectories.
- Inference knobs: `-m/--model`, `--temperature`, `--max-tokens`, `--retries`, `--jpeg-dimensions`,
  `--jpeg-quality`. Each also has an env-var override (`MODEL_NAME`, `TEMPERATURE`, and more).
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
