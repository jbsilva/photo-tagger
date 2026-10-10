---
icon: lucide/radio
---

# Telemetry

photo-tagger sends small anonymous beacons that show how the tool is used. They show which models,
configuration values, and features people use, how runs go, and on what machines. Development
follows what the beacons show. Telemetry is **opt-out**. It is on by default, you see a notice once
on the first run, and you can turn it off at any time.

This page is the full disclosure. It lists what is sent, when, where it goes, and every way to turn
it off.

## When a beacon is sent

| Event     | When                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------- |
| `run`     | A run ends: a CLI batch, one `watch` batch, or one GUI generation or save.                                  |
| `session` | A GUI window closes, or a `watch` stops.                                                                    |
| `launch`  | A GUI window opens.                                                                                         |
| `command` | Any other command ends (`show`, `set`, `vocabulary`, `undo`, `doctor`), or a run stops at the command line. |
| `crash`   | The app hits an error it did not expect.                                                                    |

## How runs and sessions are counted

A **session** is one launch: a GUI window from open to close, or one `watch` from start to stop.

A **run** is one piece of work that you start. In the CLI, that is a `tag` batch or one `watch`
batch. In the GUI, it is one click on **Generate**, together with what the click starts by itself:
the retry pass, the second try for photos with mixed writing systems, shoot harmonization, and the
**Save After Generating** write. A later **Save Selected** is a run of its own. Photos you save one
at a time count toward the session, not as runs.

For example, you open the GUI, add a folder, and click **Generate** before you go to sleep. The run
takes 2 hours. You come back 8 hours later, review the photos, and click **Save Selected**. Then you
start on another folder. That is one session and three runs: the first generation, the save, and the
second generation.

The time is split, so a window left open overnight does not look busy:

- A run's `duration_seconds` counts that run only, from its start to its end. Pacing rests are
    included, and `rested_seconds` tells how long they were.
- The session's `session_seconds` counts the whole time the window was open. `busy_seconds` is the
    time a run was in flight. `engaged_seconds` is the time with keyboard or mouse activity: each
    click or key press counts the next 60 seconds as engaged. `idle_seconds` is the rest.
- `unattended_seconds` is busy time with nobody at the keyboard.

In the example, the session lasts about 10 hours: 2 hours busy, a few minutes engaged, and close to
8 hours idle. Most of the 2 busy hours are unattended.

## What is collected

Every value is a switch, a number, a mode name, a hardware model name, or one of two random ids.
Configuration values are sent as switches, counts, and mode names, never as the text or the path
that they hold. For example, a vocabulary file becomes `vocabulary` (set or not) and
`vocabulary_terms` (a count). A hint becomes `hint` and `hint_chars`. The dataclasses in
[`telemetry.py`](https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/telemetry.py)
are the closed list of what leaves the machine.

### Every beacon

| Field                | Example        | What it is                                                                                               |
| -------------------- | -------------- | -------------------------------------------------------------------------------------------------------- |
| `schema_version`     | `3`            | The payload format version.                                                                              |
| `event`              | `run`          | The event name, from the table above.                                                                    |
| `install_id`         | a UUID         | A random id, made once per install (see below).                                                          |
| `session_id`         | a UUID         | A random id for one launch of the app. Your machine never saves it, so the next launch gets a new one.   |
| `app_version`        | `0.10.0`       | The photo-tagger version.                                                                                |
| `interface`          | `cli`          | `cli` or `gui`.                                                                                          |
| `command`            | `tag`          | The command: `tag`, `watch`, `show`, `set`, `vocabulary`, `undo`, `doctor`, `gui`, `help`, or `version`. |
| `consent`            | `default`      | Why sending is allowed: `default`, `env-opt-in`, or `gui-opt-in`.                                        |
| `uptime_seconds`     | `12.5`         | The time since the app started.                                                                          |
| `arch`               | `arm64`        | The CPU architecture.                                                                                    |
| `os`                 | `Darwin`       | The operating system.                                                                                    |
| `os_release`         | `25.5.0`       | The OS release string.                                                                                   |
| `python_version`     | `3.14.0`       | The Python version.                                                                                      |
| `cpu`, `gpu`         | `Apple M3 Pro` | The CPU and GPU model names.                                                                             |
| `cpu_count`          | `12`           | The number of logical CPU cores.                                                                         |
| `memory_gb`          | `36`           | The RAM size in whole gigabytes.                                                                         |
| `install_method`     | `uv-tool`      | `app`, `editable`, `git`, `conda`, `uv-tool`, `pipx`, `venv`, `system`, or `source`.                     |
| `installer`          | `uv`           | The tool that installed photo-tagger, from a fixed list of names.                                        |
| `system_locale`      | `de_DE`        | The OS language and region.                                                                              |
| `utc_offset_minutes` | `120`          | The time zone offset.                                                                                    |
| `local_hour`         | `23`           | The local hour of the event.                                                                             |
| `local_weekday`      | `4`            | The local day of the week of the event (Monday is 0).                                                    |
| `ci`                 | `false`        | A CI system runs the command.                                                                            |
| `terminal`           | `true`         | A terminal is attached, so a person can watch.                                                           |

### Run

| Field                                                         | Example                 | What it is                                                                                                                                                   |
| ------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `job`                                                         | `tag`                   | `tag` for a CLI batch, `generate` or `save` for a GUI run.                                                                                                   |
| `trigger`                                                     | `batch`                 | What started the run: `batch`, `single`, `hint`, `retry`, or `watch`.                                                                                        |
| `outcome`                                                     | `ok`                    | `ok`, `partial`, `failed`, `cancelled`, `interrupted`, `empty`, or `setup-failed`.                                                                           |
| `error_kind`                                                  | `ProviderError`         | The error class that stopped a run before its first photo.                                                                                                   |
| `provider`, `model`                                           | `lmstudio`              | The backend and the model name. A path keeps only its file name, and a fine-tuned id only its base model.                                                    |
| `batch_size`                                                  | `42`                    | The number of photos in the run.                                                                                                                             |
| `duration_seconds`                                            | `183.5`                 | The run from start to end.                                                                                                                                   |
| `output_language`, `ui_language`                              | `English`, `pt_BR`      | The language of the metadata (a known language name, else `other`) and of the app.                                                                           |
| `file_types`                                                  | `cr3,jpg`               | The image formats in the batch. An unknown extension counts as `other`.                                                                                      |
| `success_count`, `failure_count`, `initial_failures`          | `41`, `1`, `3`          | Photos done, photos still failing, and photos that failed at first. A photo is done when it is written, or for a GUI `generate` job, when it has a proposal. |
| `cache_hits`, `retry_successes`, `workers`                    | `10`, `2`, `1`          | Answers from the cache, photos the retry saved, and the thread count.                                                                                        |
| `input_tokens`, `output_tokens`, `total_tokens`               | `52000`                 | The tokens the model used.                                                                                                                                   |
| `inference_calls`, `inference_seconds`                        | `33`, `120.2`           | The model calls, and the time inside them.                                                                                                                   |
| `dry_run`                                                     | `0`                     | `1` for a run that wrote nothing.                                                                                                                            |
| `failure_kinds`                                               | `timeout:3`             | Fixed failure categories, never error text.                                                                                                                  |
| `failure_types`                                               | `ModelHTTPError(404):3` | The error class behind each failure, with the status of an HTTP error.                                                                                       |
| `language_repairs`, `flagged`                                 | `1`, `2`                | Extra calls for text in mixed writing systems, and photos still flagged.                                                                                     |
| `vocabulary_mapped`, `vocabulary_dropped`                     | `12`, `3`               | Keywords the vocabulary rewrote, and distinct terms it dropped.                                                                                              |
| `rested_seconds`, `rests_taken`                               | `600`, `2`              | Pacing rests.                                                                                                                                                |
| `shoots`                                                      | `3`                     | The shoots the batch split into (`--session-gap`).                                                                                                           |
| `photo_seconds_median`, `photo_seconds_p90`                   | `3.1`, `5.4`            | The model time per photo.                                                                                                                                    |
| `keywords_mean`, `title_chars_mean`, `description_chars_mean` | `18.2`                  | The size of the generated text, never the text.                                                                                                              |
| `power`, `battery_percent`                                    | `ac`, `80`              | At the end of the run: charger or battery, and the charge.                                                                                                   |
| `settings`, `setup`                                           |                         | See [`settings`](#settings) and [`setup`](#setup).                                                                                                           |
| `gui`, `review`                                               |                         | GUI runs only. See [GUI runs](#gui-runs) and [Review](#review).                                                                                              |

### Session

| Field                                                                   | Example     | What it is                                                       |
| ----------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------- |
| `session_seconds`                                                       | `36000`     | The window was open, or the watch ran, this long.                |
| `busy_seconds`, `engaged_seconds`, `idle_seconds`, `unattended_seconds` | `7200`      | The split of that time (see above).                              |
| `runs`, `save_runs`, `single_saves`                                     | `2`, `1`    | Generation runs, batch saves, and photos saved one at a time.    |
| `first_run_seconds`                                                     | `45`        | The time from opening to the first run, or `-1` without a run.   |
| `photos_added`, `photos_generated`, `photos_saved`, `photos_unsaved`    | `120`       | Photo counts.                                                    |
| `closed_with_unsaved`                                                   | `false`     | The window closed with proposals still unsaved.                  |
| `view_mode`                                                             | `tree`      | The layout of the photo list.                                    |
| `features`                                                              | `viewer: 3` | How often each GUI feature was used, from a fixed list of names. |
| `provider`, `model`, `output_language`, `ui_language`                   |             | As for a run.                                                    |
| `settings`, `setup`, `review`, `display`                                |             | The groups below.                                                |

### Command

| Field              | Example          | What it is                                                                                  |
| ------------------ | ---------------- | ------------------------------------------------------------------------------------------- |
| `outcome`          | `ok`             | `ok`, `failed`, `usage-error`, or `interrupted`.                                            |
| `exit_code`        | `0`              | The exit status of the command.                                                             |
| `duration_seconds` | `1.2`            | The time the command took.                                                                  |
| `error_kind`       | `DiscoveryError` | The error class that stopped the command.                                                   |
| `details`          | `photos: 12`     | A few counts from a fixed list of names. `problems` holds fixed codes, such as `not-found`. |
| `setup`            |                  | See [Setup](#setup).                                                                        |

The names in `details` depend on the command. `show` and `set` report `photos`, `written`, and
`failed`. `undo` reports `entries`, `put_back`, and `left_alone`. `vocabulary` reports `source`,
`keywords_read`, `keywords_kept`, and `keywords_dropped`. `doctor` reports `checks_failed` and
`problems`.

### Launch

| Field                          | Example | What it is                                        |
| ------------------------------ | ------- | ------------------------------------------------- |
| `startup_seconds`              | `1.4`   | The time from the start of the app to the window. |
| `settings`, `setup`, `display` |         | The groups below.                                 |

### Crash

| Field              | Example                                                    | What it is                                                                    |
| ------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `exception_type`   | `KeyError`                                                 | The class of the error, never its message.                                    |
| `exception_module` | `builtins`                                                 | The module that defines that class.                                           |
| `exception_chain`  | `FieldError<KeyError`                                      | The error and the errors that caused it, the outermost first.                 |
| `crash_location`   | `pipeline:run_batch:714`                                   | The deepest photo-tagger code location.                                       |
| `crash_frames`     | `main:tag:1400>...`                                        | The last few photo-tagger code locations.                                     |
| `crash_stack`      | `photo_tagger.main:tag:1400>...>pydantic_ai.agent:run:512` | The stack trace: each frame as module, function, and line, at most 40 frames. |
| `libraries`        | `openai=2.1.0,pydantic=2.13.5`                             | The versions of the libraries that crashes most often come from.              |

The stack trace names library frames too, but only by module name. A frame outside the installed
packages and the Python library shows as `?`, so a folder name never gets into the report. A crash
beacon never carries the error message, because messages can contain file paths.

### Settings

| Area                    | Fields                                                                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The model server        | `endpoint`, `secure`, `custom_url`, `api_key` (a key is set), `retries`, `timeout_seconds`                                                                                                                                |
| What the model is asked | `temperature`, `max_tokens`, `frequency_penalty`, `jpeg_dimensions`, `jpeg_quality`, `repair_language`, `hint`, `hint_chars`, `custom_prompt`, `workers`, `cache`                                                         |
| Keywords                | `max_keywords`, `vocabulary`, `vocabulary_terms`, `vocabulary_strict`, `session_gap_minutes`                                                                                                                              |
| What a save writes      | `sidecar_mode`, `sidecar_overrides` (a count), `write_title`, `write_description`, `write_keywords`, `write_copyright`, `overwrite_title`, `overwrite_description`, `overwrite_keywords`, `backup`, `undo_log`, `dry_run` |
| Copyright               | `artist`, `copyright_notice`, `rights` (each one filled in or not), `marked`, `overwrite_copyright`, `set_tags` (a count)                                                                                                 |
| Pacing                  | `paced`, `duty`, `battery_floor`, `battery_resume`, `work_minutes`, `max_pause_minutes`                                                                                                                                   |
| Alerts                  | `notify`, `notify_sound`                                                                                                                                                                                                  |
| Which photos            | `extensions` (known formats only), `recursive`, `skip_tagged`, `date_filter`, `skip_list`, `resume_list`                                                                                                                  |
| Reports (CLI)           | `summary_file`, `csv_report`, `json_output`, `progress_bar`, `lock_file`, `otel`, `otel_content`, `console_log_level`, `file_log_level`                                                                                   |
| Watching                | `watch_interval`, `watch_settle`                                                                                                                                                                                          |
| GUI                     | `autosave` (**Save After Generating**)                                                                                                                                                                                    |

The `duty`, `battery_floor`, and `battery_resume` fields are percentages, and `watch_interval` and
`watch_settle` are seconds. Each field has the name of the configuration key it reports.

The `endpoint` tells where the model server runs, never its URL. It is `localhost`, `lan` (your own
network), `tailnet`, `public`, or the name of a known service, such as `api.openai.com`.

### Setup

| Field                 | What it is                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `config_file`         | `none`, `env`, `cwd`, or `user`: which configuration file was used, never its path.                                    |
| `config_keys`         | The `table.key` names that the configuration file sets.                                                                |
| `unknown_config_keys` | The number of keys photo-tagger does not read. Their names are not sent.                                               |
| `cli_flags`           | The options typed on the command line, never their values.                                                             |
| `env_vars`            | The photo-tagger environment variables that are set, never their values.                                               |
| `changed`             | The settings that differ from the defaults, by name. One moved by an environment variable shows in `env_vars` instead. |

### GUI runs

| Field                                | What it is                                                               |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `use_cache`, `autosave`              | The cache was on, and **Save After Generating** was on.                  |
| `hinted`                             | Photos generated with a hint.                                            |
| `regenerated_flagged`                | Photos generated again because their text mixed writing systems.         |
| `retry_pass`, `retry_pass_recovered` | Photos the automatic retry ran again, and how many of them it saved.     |
| `mixed_pass`, `mixed_pass_fixed`     | Photos the automatic mixed-script pass ran again, and how many it fixed. |
| `harmonized`                         | Photos whose keywords shoot harmonization changed.                       |
| `saved`, `save_failures`             | Photos the automatic save wrote, and photos it failed to write.          |
| `generate_seconds`, `save_seconds`   | The time spent generating and saving.                                    |
| `attended_seconds`                   | Keyboard or mouse activity while the run was in flight.                  |

### Review

This group tells what happens between generating photos and saving them. The edits are counts. The
edited text is never sent.

| Field                                                   | What it is                                                                  |
| ------------------------------------------------------- | --------------------------------------------------------------------------- |
| `saved`                                                 | Photos saved.                                                               |
| `edited`                                                | Photos with a manual change to the proposal.                                |
| `edited_title`, `edited_description`, `edited_keywords` | Which fields were changed.                                                  |
| `keywords_added`, `keywords_removed`                    | Keywords added to, or removed from, the proposals.                          |
| `hinted`, `from_cache`                                  | Saved photos that had a hint, or came from the cache.                       |
| `flagged`                                               | Photos saved while their text still mixed writing systems.                  |
| `manual_only`                                           | Photos saved without a proposal: the GUI used as a plain editor.            |
| `copyright_only`                                        | Photos whose save wrote only the copyright fields, without being generated. |
| `review_seconds_median`, `review_seconds_max`           | The time from the proposal to the save.                                     |

### Display

GUI only.

| Field                          | Example      | What it is                                                                     |
| ------------------------------ | ------------ | ------------------------------------------------------------------------------ |
| `qt_version`, `pyside_version` | `6.10.0`     | The versions of the GUI toolkit.                                               |
| `screen`, `screens`            | `1512x982@2` | The primary screen in logical pixels and its scale, and the number of screens. |
| `color_scheme`                 | `dark`       | `dark` or `light`.                                                             |
| `window`                       | `1180x760`   | The window size, or `maximized` or `full-screen`.                              |

### The install id

The `install_id` is a random UUID. photo-tagger makes it once and stores it in the user state
directory. It does **not** come from a hardware identifier, so it cannot fingerprint a machine. It
only lets the runs of one install count as one active user.

### What is never collected

File paths, filenames, photo contents, generated tags, titles, descriptions, prompts, hints, server
URLs, API keys, IP addresses, error messages, and anything else that identifies you.

## Where it goes

The collector is our own Cloudflare Worker at `telemetry.tagger.photo`. It is write-only, stores no
IP addresses, sets no cookies, and uses no third-party analytics service. Its full source and the
queries that run against it are in the
[`telemetry/`](https://github.com/jbsilva/photo-tagger/tree/main/telemetry) directory of the
repository.

Events are kept with no expiry date, so trends can be compared across releases.

Sending happens on a background thread with a short timeout, and every failure is ignored. Telemetry
cannot crash, slow down, or block a tagging run.

## The first-run notice

The first time telemetry is active, photo-tagger prints a notice that tells what is collected and
how to turn it off. The GUI shows a dialog with a **Turn It Off** button. After that, it stays
quiet. When a release adds a new kind of data, the notice shows once more.

## How to disable it

Any one of these turns telemetry off:

- Pass `--no-telemetry` on the command line.
- Set `PHOTO_TAGGER_TELEMETRY=0` (or the cross-tool `DO_NOT_TRACK=1`) in your environment.
- Put `enabled = false` under `[telemetry]` in your
    [configuration file](getting-started/configuration.md).
- In the desktop GUI, clear **Settings > Send Anonymous Telemetry**. The GUI keeps the choice for
    the next launch.

An opt-out anywhere wins over an opt-in, so exporting `PHOTO_TAGGER_TELEMETRY=0` once disables
telemetry everywhere, including the GUI. Any value other than `1`, `true`, `yes`, or `on` also turns
it off, so `PHOTO_TAGGER_TELEMETRY=disabled` does what it says. The `PHOTO_TAGGER_NO_TELEMETRY=1` of
version 0.9 and earlier still turns telemetry off too.

## Sending anyway when DO_NOT_TRACK is set

`DO_NOT_TRACK=1` asks every tool on the machine not to send telemetry, and photo-tagger honors it.
It speaks for every tool at once, so a choice made for photo-tagger alone overrules it:

- Set `PHOTO_TAGGER_TELEMETRY=1` in your environment. This works for the CLI and the GUI.
- In the desktop GUI, check **Settings > Send Anonymous Telemetry** by hand. The GUI keeps the
    choice for the next launch.

While `DO_NOT_TRACK` stops the beacons, the GUI toggle shows unchecked, and its tooltip tells why.
The `enabled = true` of the configuration file does not overrule `DO_NOT_TRACK`. It is the default
value, and the GUI writes it each time you save your configuration, so it is not a deliberate
choice.

The full order, first match wins:

1. `PHOTO_TAGGER_TELEMETRY` set to anything but a "yes", or `PHOTO_TAGGER_NO_TELEMETRY=1`: off.
2. `--no-telemetry`, `enabled = false` in the configuration file, or the GUI toggle cleared: off. In
    the GUI, a toggle checked by hand overrules `enabled = false`, because the GUI writes that key
    itself when it saves your settings.
3. `PHOTO_TAGGER_TELEMETRY=1`: on.
4. The GUI toggle checked by hand: on.
5. `DO_NOT_TRACK=1`: off.
6. Otherwise: on.
