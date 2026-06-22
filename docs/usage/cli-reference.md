---
icon: lucide/square-terminal
---

# CLI reference

photo-tagger runs as a single command, `photo-tagger`, built with
[cyclopts](https://github.com/BrianPugh/cyclopts). Its flags are grouped into logical option groups;
this page documents every flag, its default, the matching environment variable (or `-` when there is
none), and what it does.

Any flag you pass on the command line overrides the corresponding config-file value and
environment-informed default. See [Configuration](../getting-started/configuration.md) for the full
precedence rules and TOML layout.

## Commands

Running `photo-tagger` with image inputs tags them (the default command). Two subcommands exist:

| Command               | Description                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| `photo-tagger`        | Tag the given images (default). Documented by the option groups below.                              |
| `photo-tagger doctor` | Pre-flight check: verifies ExifTool is on `PATH` and the provider serves the model, then exits 0/1. |
| `photo-tagger gui`    | Launch the optional desktop GUI. Requires the `gui` extra; see [Desktop GUI](gui.md).               |

`doctor` accepts `--provider`, `-m/--model`, `-u/--url`, and `-k/--api-key` (same meanings as below)
and honors the same config file and environment variables. Run it first when a tagging run will not
start:

```console
$ photo-tagger doctor --provider lmstudio --model qwen/qwen3-vl-30b
photo-tagger 0.2.2 environment check

  OK    ExifTool: /usr/bin/exiftool
  OK    Model 'qwen/qwen3-vl-30b' on lmstudio: available at http://localhost:1234/v1

All checks passed.
```

## Input and scanning

`-i/--input` is required and repeatable: pass it once per file or directory you want to process.

| Flag                         | Default    | Env var | Description                                                                                   |
| ---------------------------- | ---------- | ------- | --------------------------------------------------------------------------------------------- |
| `-i`, `--input` PATH         | (required) | `-`     | One or more files and/or directories; repeat the flag.                                        |
| `--ext`, `--extensions` LIST | `cr3,jpg`  | `-`     | Comma-separated extensions used when scanning directories (case-insensitive).                 |
| `-r`, `--recursive`          | `false`    | `-`     | Recurse into subdirectories while scanning input directories.                                 |
| `-w`, `--workers` N          | `1`        | `-`     | Process N photos concurrently with a thread pool. The model server is usually the bottleneck. |
| `--skip-from` PATH           | none       | `-`     | Skip filenames listed in PATH (one per line; lines starting with `#` are comments).           |
| `--append-to-skip-file` PATH | none       | `-`     | Append each successfully tagged filename to PATH as the run progresses (created if missing).  |

## Provider

The provider group selects the backend and how to reach it. Prefer the API-key environment variables
over `--api-key` so the key never lands in your shell history.

| Flag                  | Default                                                                     | Env var                                                      | Description                                                        |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `--provider` NAME     | `lmstudio`                                                                  | `-`                                                          | Backend: `ollama`, `lmstudio`, or `openai`.                        |
| `-m`, `--model` NAME  | `qwen/qwen3-vl-30b`                                                         | `MODEL_NAME`                                                 | Vision-language model identifier.                                  |
| `-u`, `--url` URL     | `http://localhost:1234/v1` (lmstudio), `http://localhost:11434/v1` (ollama) | `LM_STUDIO_BASE_URL` / `OLLAMA_BASE_URL` / `OPENAI_BASE_URL` | Provider API base URL.                                             |
| `-k`, `--api-key` KEY | none                                                                        | `OLLAMA_API_KEY` / `LM_STUDIO_API_KEY` / `OPENAI_API_KEY`    | API key; prefer the env vars over the flag. Required for `openai`. |
| `--retries` N         | `5`                                                                         | `RETRIES`                                                    | Automatic retries when the model output fails schema validation.   |

## Inference

These flags tune sampling and the image sent to the model. Lower temperature and a frequency penalty
keep the output focused; the JPEG settings control how much detail the model sees.

| Flag                        | Default | Env var             | Description                                                      |
| --------------------------- | ------- | ------------------- | ---------------------------------------------------------------- |
| `--temperature` FLOAT       | `0.2`   | `TEMPERATURE`       | Sampling temperature.                                            |
| `--max-tokens` N            | `1200`  | `MAX_TOKENS`        | Maximum tokens to generate.                                      |
| `--timeout-seconds` FLOAT   | `60.0`  | `TIMEOUT_SECONDS`   | Per-image inference timeout; on timeout the retry loop steps in. |
| `--frequency-penalty` FLOAT | `0.5`   | `FREQUENCY_PENALTY` | Penalty on repeated tokens; discourages repetitive output loops. |
| `--jpeg-dimensions` N       | `1280`  | `JPEG_DIMENSIONS`   | Max dimension (px) of the JPEG sent to the model.                |
| `--jpeg-quality` N          | `80`    | `JPEG_QUALITY`      | JPEG quality (1-100) of the image sent to the model.             |

## Output

The output group decides what metadata is written and where. By default photo-tagger writes an XMP
sidecar next to each image and leaves the original untouched.

| Flag                                             | Default           | Env var | Description                                                                                         |
| ------------------------------------------------ | ----------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `--preserve-keywords` / `--overwrite-keywords`   | preserve (`true`) | `-`     | Merge with existing keywords vs replace them.                                                       |
| `--write-title` / `--no-write-title`             | write (`true`)    | `-`     | Generate and write a title.                                                                         |
| `--write-description` / `--no-write-description` | write (`true`)    | `-`     | Generate and write a description.                                                                   |
| `--write-keywords` / `--no-write-keywords`       | write (`true`)    | `-`     | Write keywords (merged per `--preserve-keywords`); `--no-write-keywords` leaves existing ones.      |
| `--write-sidecar` / `--embed-in-photo`           | sidecar (`true`)  | `-`     | Write an XMP sidecar (default) vs embed metadata into the image file.                               |
| `--backup-xmp` / `--no-backup-xmp`               | backup (`true`)   | `-`     | Keep ExifTool's `*_original` backup before writing; `--no-backup-xmp` passes `-overwrite_original`. |
| `--max-keywords` N                               | none (keep all)   | `-`     | Cap AI-generated keywords kept per photo before merging.                                            |
| `--dry-run`                                      | `false`           | `-`     | Run the model and log the proposed metadata, but write nothing.                                     |

## Filter

Filters narrow the resolved batch before any model call. Timestamps use ISO 8601, such as
`2024-01-01` or `2024-01-01T14:30`; naive timestamps use local time.

| Flag                   | Default | Env var | Description                                                                                       |
| ---------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------- |
| `--skip-tagged`        | `false` | `-`     | Skip files that already have keywords, a title, or a description in the image or its XMP sidecar. |
| `--newer-than` ISO8601 | none    | `-`     | Drop files whose mtime is on/before this timestamp.                                               |
| `--older-than` ISO8601 | none    | `-`     | Drop files whose mtime is on/after this timestamp.                                                |

## Log

photo-tagger writes a timestamped log file and mirrors messages to stderr, so stdout stays clean for
[`--json`](#display) output.

| Flag                        | Default | Env var | Description                                                        |
| --------------------------- | ------- | ------- | ------------------------------------------------------------------ |
| `--console-log-level` LEVEL | `INFO`  | `-`     | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`/`OFF`. `OFF` disables. |
| `--file-log-level` LEVEL    | `DEBUG` | `-`     | Same levels; `OFF` disables the file log.                          |
| `--log-folder` PATH         | `logs`  | `-`     | Folder for timestamped log files.                                  |

## Display

The display group controls the progress bar and machine-readable output.

| Flag                           | Default           | Env var | Description                                                                                                                                                                                                                                  |
| ------------------------------ | ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--progress` / `--no-progress` | progress (`true`) | `-`     | Live rich progress bar; auto-disabled when stderr is not a TTY.                                                                                                                                                                              |
| `--json`                       | `false`           | `-`     | Emit one NDJSON line per processed photo to stdout (`file`, `status`, `from_cache`, `retry`, `title`, `description`, `keywords`, input/output/total tokens, `seconds`). Logs and progress stay on stderr, so stdout pipes cleanly into `jq`. |

## Artifacts

The artifacts group points at side files: a custom prompt, a run summary, a per-photo CSV report, a
result cache, and a lock.

| Flag                  | Default | Env var | Description                                                                                                                                  |
| --------------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--prompt-file` PATH  | none    | `-`     | Replace the default user prompt with the contents of PATH; existing photo metadata is still appended automatically.                          |
| `--summary-file` PATH | none    | `-`     | Write a JSON run summary (success/failure counts, failed files, token usage, wall time) on completion.                                       |
| `--csv-file` PATH     | none    | `-`     | Write a CSV report with one row per photo (see below). Rows stream as photos finish, so a stopped run still leaves a valid file.             |
| `--cache-file` PATH   | none    | `-`     | SQLite cache of model outputs; reruns skip the model call when nothing relevant changed. Created if missing; safe to delete.                 |
| `--lock-file` PATH    | none    | `-`     | Acquire an exclusive file lock before running; refuse to start if another photo-tagger already holds it. Works on Linux, macOS, and Windows. |

### CSV report

Where `--summary-file` writes one JSON object for the whole run and `--json` streams NDJSON to
stdout, `--csv-file` writes a spreadsheet-friendly table with **one row per photo**. It is the
single file that gathers everything extracted and computed for each image:

- `filename`, `file`, `status`
- `title`, `description`, `keywords` (the keywords actually written), `hierarchical_keywords`
- `existing_keywords` (what was already on the file)
- `camera_model`, `lens_model`, `capture_date`, `gps_position`, `city`, `country` (read EXIF)
- `input_tokens`, `output_tokens`, `total_tokens`, `seconds`, `from_cache`, `retry`

Multi-value cells (the keyword lists) are joined with a semicolon and a space. Rows are flushed as
each photo completes, so interrupting the run with Ctrl-C still leaves a complete, openable CSV of
the work done so far. `--csv-file` and `--json` can be used together; both observe every photo. A
`--dry-run` still fills the report, which makes it handy for previewing a batch before writing any
metadata.

## Skipping and resuming

Three flags cooperate to skip work you have already done and to resume a run that stopped partway
through:

- `--skip-from PATH` reads a list of filenames (one per line, `#` comments allowed) and drops any
    matching files from the batch before processing starts.
- `--append-to-skip-file PATH` appends each successfully tagged filename to PATH as the run
    progresses, creating the file if it does not exist.
- `--skip-tagged` inspects each file's existing metadata and skips anything that already has
    keywords, a title, or a description (in the image or its XMP sidecar). Use it when you want the
    skip decision to come from the files themselves rather than from a list.

For resume-on-failure, pass the **same path** to both `--skip-from` and `--append-to-skip-file`. The
first run appends every success to the file; if the run dies partway through, re-running with the
same arguments reads that file back through `--skip-from` and continues from where it left off,
without re-tagging the photos that already succeeded.

!!! tip

    Combine the skip file with `--cache-file` for an even cheaper resume: the skip file removes finished
    photos from the batch entirely, while the cache avoids re-calling the model for any photo that does
    slip back in unchanged.

See [Recipes](recipes.md) for runnable resume and skip examples.
