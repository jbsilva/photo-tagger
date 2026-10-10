---
icon: lucide/radio
---

# Telemetry

photo-tagger sends a single anonymous beacon at the end of each run so development can be guided by
how the tool is actually used (which models, platforms, languages, and file formats are common, and
typical batch sizes). It is **opt-out**: on by default, with a one-time notice on the first run, and
easy to disable. This page is the full disclosure: exactly what is sent, where it goes, and every
way to turn it off.

## What is collected

The beacon contains these fields, and nothing else:

| Field              | Example             | What it is                                                   |
| ------------------ | ------------------- | ------------------------------------------------------------ |
| `schema_version`   | `1`                 | Payload format version, so the collector can branch.         |
| `install_id`       | a UUID              | Random id generated once per install (see below).            |
| `app_version`      | `0.5.0`             | The photo-tagger version.                                    |
| `interface`        | `cli` or `gui`      | Which frontend ran the batch.                                |
| `provider`         | `lmstudio`          | The selected backend.                                        |
| `model`            | `qwen/qwen3-vl-30b` | The model identifier.                                        |
| `batch_size`       | `42`                | How many photos were in the run.                             |
| `duration_seconds` | `183.5`             | Wall-clock run duration.                                     |
| `output_language`  | `English`           | Language the model writes titles, descriptions, keywords in. |
| `ui_language`      | `pt_BR`             | The resolved app interface language.                         |
| `file_types`       | `cr3,jpg`           | Distinct file extensions in the batch (never filenames).     |
| `arch`             | `arm64`             | CPU architecture.                                            |
| `os`               | `Darwin`            | Operating system.                                            |
| `os_release`       | `25.5.0`            | OS release string.                                           |
| `python_version`   | `3.14.0`            | The Python runtime version.                                  |

The `install_id` is a random UUID generated once and stored in the user state directory. It is
**not** derived from any hardware identifier, so it cannot fingerprint a machine; it only lets
repeated runs from the same install count as one active user over time.

**What is never collected:** file paths, filenames, photo contents, generated tags, titles,
descriptions, prompts, API keys, IP addresses, or anything else that identifies you. The exact,
closed payload is the
[`build_payload`](https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/telemetry.py)
function; there is nothing else to leak.

## Where it goes

The collector is our own self-hosted Cloudflare Worker at `telemetry.tagger.photo`. It is
write-only, stores no IP addresses, sets no cookies, and involves no third-party analytics service.
Its full source (and the queries run against it) lives in the
[`telemetry/`](https://github.com/jbsilva/photo-tagger/tree/main/telemetry) directory of the
repository.

Events are kept with no expiry date, so trends can be compared across releases.

Sending happens on a background thread with a short timeout and every failure is swallowed, so
telemetry can never crash, slow down, or block a tagging run.

## The first-run notice

The first time telemetry is active, photo-tagger prints a one-time notice (the GUI shows a
dismissible dialog with a one-click "Turn It Off") explaining what is collected and how to disable
it. After that, it stays quiet.

## How to disable it

Any one of these turns telemetry off:

- Pass `--no-telemetry` on the command line.
- Set `PHOTO_TAGGER_TELEMETRY=0` (or the cross-tool `DO_NOT_TRACK=1`) in your environment.
- Put `enabled = false` under `[telemetry]` in your [config file](getting-started/configuration.md).
- In the desktop GUI, uncheck **Settings > Send Anonymous Telemetry**; the choice is remembered for
    the next launch.

An opt-out anywhere wins over an opt-in, so exporting `PHOTO_TAGGER_TELEMETRY=0` once disables
telemetry everywhere, including the GUI. Any value other than `1`, `true`, `yes`, or `on` also turns
it off, so `PHOTO_TAGGER_TELEMETRY=disabled` does what it says. The `PHOTO_TAGGER_NO_TELEMETRY=1` of
version 0.9 and earlier still turns telemetry off too.

## Sending anyway when DO_NOT_TRACK is set

`DO_NOT_TRACK=1` asks every tool on the machine not to send telemetry, and photo-tagger honors it.
Because it speaks for every tool at once, a choice made for photo-tagger alone overrules it:

- Set `PHOTO_TAGGER_TELEMETRY=1` in your environment. This works for the CLI and the GUI.
- In the desktop GUI, check **Settings > Send Anonymous Telemetry** by hand. The choice is
    remembered for the next launch.

While `DO_NOT_TRACK` stops the beacons, the GUI toggle shows unchecked, and its tooltip says why.
The config file's `enabled = true` does not overrule `DO_NOT_TRACK`. It is the default value, and
the GUI writes it every time you save your settings, so it is not a deliberate choice.

The full order, first match wins:

1. `PHOTO_TAGGER_TELEMETRY` set to anything but a "yes", or `PHOTO_TAGGER_NO_TELEMETRY=1`: off.
2. `--no-telemetry`, `enabled = false` in the config file, or the GUI toggle unchecked: off. In the
    GUI, a toggle checked by hand overrules `enabled = false`, because the GUI writes that key
    itself when it saves your settings.
3. `PHOTO_TAGGER_TELEMETRY=1`: on.
4. The GUI toggle checked by hand: on.
5. `DO_NOT_TRACK=1`: off.
6. Otherwise: on.
