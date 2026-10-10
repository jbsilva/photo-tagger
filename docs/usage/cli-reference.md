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

Running `photo-tagger` with image inputs tags them (the default command). Seven subcommands exist:

| Command                   | Description                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `photo-tagger`            | Tag the given images (default). Documented by the option groups below.                                      |
| `photo-tagger doctor`     | Pre-flight check: verifies ExifTool is on `PATH` and the provider serves the model, then exits 0/1.         |
| `photo-tagger show`       | Print the metadata a photo already carries (see [Copyright and other fields](#copyright-and-other-fields)). |
| `photo-tagger set`        | Write copyright and other fields without calling a model (same section).                                    |
| `photo-tagger vocabulary` | Build a keyword file from a library's own keywords (see [Building a vocabulary](#building-a-vocabulary)).   |
| `photo-tagger watch`      | Watch folders and tag photos as they arrive (see [Watching a folder](#watching-a-folder)).                  |
| `photo-tagger undo`       | Put back what the last run wrote (see [Undoing a run](#undoing-a-run)).                                     |
| `photo-tagger gui`        | Launch the optional desktop GUI. Requires the `gui` extra; see [Desktop GUI](gui.md).                       |

`doctor` accepts `--provider`, `-m/--model`, `-u/--url`, and `-k/--api-key` (same meanings as below)
and honors the same config file and environment variables. Run it first when a tagging run will not
start:

```console
$ photo-tagger doctor --provider lmstudio --model qwen/qwen3-vl-30b
photo-tagger 0.5.0 environment check

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

| Flag                  | Default                                                                                                            | Env var                                                                             | Description                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `--provider` NAME     | `lmstudio`                                                                                                         | `-`                                                                                 | Backend: `ollama`, `lmstudio`, `llamacpp`, or `openai`.            |
| `-m`, `--model` NAME  | `qwen/qwen3-vl-30b`                                                                                                | `MODEL_NAME`                                                                        | Vision-language model identifier.                                  |
| `-u`, `--url` URL     | `http://localhost:1234/v1` (lmstudio), `http://localhost:11434/v1` (ollama), `http://localhost:8080/v1` (llamacpp) | `LM_STUDIO_BASE_URL` / `OLLAMA_BASE_URL` / `LLAMA_CPP_BASE_URL` / `OPENAI_BASE_URL` | Provider API base URL.                                             |
| `-k`, `--api-key` KEY | none                                                                                                               | `OLLAMA_API_KEY` / `LM_STUDIO_API_KEY` / `LLAMA_CPP_API_KEY` / `OPENAI_API_KEY`     | API key; prefer the env vars over the flag. Required for `openai`. |
| `--retries` N         | `5`                                                                                                                | `RETRIES`                                                                           | Automatic retries when the model output fails schema validation.   |

## Inference

These flags tune sampling and the image sent to the model. Lower temperature and a frequency penalty
keep the output focused; the JPEG settings control how much detail the model sees.

| Flag                                         | Default   | Env var             | Description                                                                                                                                                                                                                                                    |
| -------------------------------------------- | --------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output-language`, `--lang` NAME           | `English` | `-`                 | Language of the generated title, description, and keywords (any language name the model understands, e.g. `German`, `"Brazilian Portuguese"`).                                                                                                                 |
| `--hint` TEXT                                | none      | `-`                 | A note about every photo in the run that the model trusts over its own reading of the image, e.g. `"The animal in these photos is a deer"`. Changes the cache namespace, so hinted runs never replay hint-less results.                                        |
| `--repair-language` / `--no-repair-language` | `false`   | `-`                 | When the generated text mixes writing systems, spend one more model call asking for it in `--output-language` (`Ocarina` for a stray `陶笛`). Off by default; the affected photos are flagged either way. See [Mixed-language output](#mixed-language-output). |
| `--temperature` FLOAT                        | `0.2`     | `TEMPERATURE`       | Sampling temperature.                                                                                                                                                                                                                                          |
| `--max-tokens` N                             | `1200`    | `MAX_TOKENS`        | Maximum tokens to generate.                                                                                                                                                                                                                                    |
| `--timeout-seconds` FLOAT                    | `60.0`    | `TIMEOUT_SECONDS`   | Timeout of one model request; on timeout the retry loop steps in.                                                                                                                                                                                              |
| `--frequency-penalty` FLOAT                  | `0.5`     | `FREQUENCY_PENALTY` | Penalty on repeated tokens; discourages repetitive output loops.                                                                                                                                                                                               |
| `--jpeg-dimensions` N                        | `640`     | `JPEG_DIMENSIONS`   | Longest edge (px) of the JPEG sent to the model, and the biggest lever on speed. See [Image size](#image-size).                                                                                                                                                |
| `--jpeg-quality` N                           | `80`      | `JPEG_QUALITY`      | JPEG quality (1-100) of the image sent to the model.                                                                                                                                                                                                           |

### Image size

`--jpeg-dimensions` sets the longest edge of the JPEG the model is shown, and it is the biggest
single lever on how long a photo takes. Prefill scales with the pixel count while the rest of the
call does not move, so on a local Qwen3-VL-30B one photo costs about 0.49 s of prefill at 640 px,
0.83 s at 896, 1.11 s at 1024, and 1.80 s at 1280, against roughly 1.9 s to write the answer.

What the extra pixels buy is text inside the frame, not subject recognition: in a controlled probe,
a line of text 30 px tall in a 4032 px photo was transcribed reliably at 896 px and never at 768 px
or below. So 640 is the fast choice, 768 sees more detail, 896 reads large text such as signs, and
1024 reads fine text such as documents and labels. Multiples of 128 line up with the vision tower's
patch grid.

Changing this value starts a fresh cache namespace, because the answers already stored describe what
the model saw at the old size. Pick it before a long run rather than during one.

## Output

The output group decides what metadata is written and where. By default photo-tagger writes an XMP
sidecar next to each image and leaves the original untouched.

| Flag                                                 | Default             | Env var | Description                                                                                                                                                                                  |
| ---------------------------------------------------- | ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--preserve-keywords` / `--overwrite-keywords`       | preserve (`true`)   | `-`     | Merge with existing keywords vs replace them.                                                                                                                                                |
| `--preserve-title` / `--overwrite-title`             | overwrite (`false`) | `-`     | Keep the title a photo already has (write the generated one only where there is none) vs replace it.                                                                                         |
| `--preserve-description` / `--overwrite-description` | overwrite (`false`) | `-`     | Keep the description a photo already has vs replace it. Replacing is what clears a camera-written placeholder.                                                                               |
| `--write-title` / `--no-write-title`                 | write (`true`)      | `-`     | Generate and write a title.                                                                                                                                                                  |
| `--write-description` / `--no-write-description`     | write (`true`)      | `-`     | Generate and write a description.                                                                                                                                                            |
| `--write-keywords` / `--no-write-keywords`           | write (`true`)      | `-`     | Write keywords (merged per `--preserve-keywords`); `--no-write-keywords` leaves existing ones.                                                                                               |
| `--write-copyright` / `--no-write-copyright`         | write (`true`)      | `-`     | Write the copyright block below. Nothing is written unless it is configured; the flag is how one run skips a configured one.                                                                 |
| `--write-sidecar` / `--embed-in-photo`               | sidecar (`true`)    | `-`     | Shorthand for `--sidecar-mode all` vs `--sidecar-mode none`.                                                                                                                                 |
| `--sidecar-mode` {all,none,raw,both}                 | `all`               | `-`     | Where metadata goes: a sidecar per photo, inside each photo, (`raw`) a sidecar for RAW files and embedded for the rest, or (`both`) each photo and a sidecar. Wins over the two flags above. |
| `--backup-xmp` / `--no-backup-xmp`                   | backup (`true`)     | `-`     | Keep ExifTool's `*_original` backup before writing; `--no-backup-xmp` passes `-overwrite_original`.                                                                                          |
| `--max-keywords` N                                   | none (keep all)     | `-`     | Cap AI-generated keywords kept per photo before merging.                                                                                                                                     |
| `--vocabulary` PATH                                  | none                | `-`     | Restrict generated keywords to the terms in PATH (see below).                                                                                                                                |
| `--vocabulary-strict`                                | `false`             | `-`     | Drop generated keywords the vocabulary does not cover instead of writing them as-is.                                                                                                         |
| `--session-gap` MINUTES                              | `0` (off)           | `-`     | Group photos into shoots and make each shoot's keywords agree with itself (see below).                                                                                                       |
| `--dry-run`                                          | `false`             | `-`     | Run the model and log the proposed metadata, but write nothing.                                                                                                                              |

### Controlled vocabulary

`--vocabulary` points at the keyword list your catalog already uses, so a run cannot seed it with
near-duplicates of keywords you have curated by hand. Two file shapes are accepted:

- A **Lightroom keyword-list export** (_Metadata > Export Keywords_), in either shape that menu
    offers: the `.txt` (_Exclude Keyword Tag Options_) is one keyword per line, children indented
    under their parent, `{braces}` for synonyms, `[brackets]` for keywords marked "do not export";
    the `.csv` (_Include Keyword Tag Options_) is the same list behind four option columns, and the
    keyword column is lifted out of it automatically.
- A **plain list**: one term per line, optionally as a full path in either `Animal|Bird|Osprey` or
    `Osprey<Bird<Animal` form. Blank lines are ignored, and so are `#` comments: a comment needs a
    space after the hash, so a hashtag-style keyword such as `#Diversity` is kept as a keyword.

```text
Animal
	Bird
		Osprey
		{Sea Hawk}
	Mammal
Landscape
```

Every generated keyword is matched against the file, ignoring case and punctuation, with a
conservative fuzzy pass for typos and longer variants. A match is rewritten to the file's own
spelling **and hierarchy**, so `ospreys`, `Sea Hawk`, and `Osprey<Raptor<Wildlife` all land as
`Animal|Bird|Osprey`. The file's spelling is used exactly as written, so a catalog that keeps its
keywords in lower case stays that way. Keywords the file does not cover pass through untouched
unless `--vocabulary-strict` is set, in which case they are dropped and reported: the run summary's
`vocabulary_dropped` names every rejected term and how often it came up, which is the list to work
from when growing the vocabulary. `vocabulary_mapped` counts the rewrites.

#### Languages other than English

Write the vocabulary in the same language you generate in ([`--output-language`](#inference)); a
German run cannot match an English catalog whatever the matcher does.

Matching itself is language-neutral except in one place: folding a plural onto its singular
(`Ospreys` → `Osprey`) uses English rules, so it is applied for English output and skipped for every
other language. Applying it to German would merge `Alles` into `Alle`, and it can say nothing at all
about `птицы`. Everything else works in any script: case folding (including `Straße` and `Strasse`),
punctuation and spacing, and the fuzzy pass, which still unifies longer inflections such as
`Landschaften` with `Landschaft` or `Закаты` with `Закат`.

For the short inflected forms no ratio can safely catch, declare them in the file with the synonym
syntax, which is exact and needs no guessing:

```text
Животное
	Птица
	{птицы}
	{птиц}
```

The vocabulary is listed in the prompt as well, so the model prefers your terms in the first place
instead of being corrected afterwards. That listing is part of the cache namespace: swapping
vocabulary files starts a fresh cache slice rather than replaying keywords chosen under the old one.

### Sessions

Each photo is analyzed on its own, so forty frames of the same bird can come back as `Osprey` here
and `Ospreys` there, filed under `Bird<Animal` on one frame and `Raptor<Wildlife` on the next.
Lightroom then shows four keywords where there is one subject.

`--session-gap MINUTES` fixes that without needing a vocabulary file. The batch is split into shoots
wherever the capture time (EXIF `DateTimeOriginal`, falling back to file mtime) jumps by more than
the given gap. Every photo in a shoot is analyzed first, then the session's own output becomes its
vocabulary: the spelling most of the session used wins, and so does the hierarchy most of it used.
Only then is anything written. It reads `--output-language` for the same reason the vocabulary file
does, so a German or Russian shoot is harmonized under that language's rules rather than English
ones, and it folds close-enough variants together on top of that: a shoot that said `Закаты` twice
and `Закат` once writes `Закаты` throughout, and one that said `Landschaften` and `Landschaft`
settles on one of them. Short words are never folded this way, so `Alle` and `Alles` stay two
keywords.

```bash
photo-tagger -i ~/Pictures/Trip -r --session-gap 60
```

Because the vocabulary is derived from the finished results rather than fed to the model, the
outcome does not depend on which photo finished first: the same batch harmonizes the same way every
time, at any `--workers` setting. It composes with `--vocabulary`, which is applied per photo first.

Two details worth knowing:

- Photos are processed in capture order rather than the order they were listed.
- Sessions run one after another (the photos inside one still run in parallel), and a session's
    writes happen in a burst at its end. If a write fails there it is reported as failed and **not**
    retried: the model work is already done and harmonized, and an ExifTool write that failed for a
    filesystem reason is not the kind of failure a second attempt clears. Analysis failures are
    still retried as usual, and a photo recovered by the retry pass lands on its shoot's agreed
    terms.

## Copyright and other fields

No model can guess who took a photo or who owns it. You write those fields yourself, and they go
into the same ExifTool call as the generated metadata. They belong to the output group, so
`photo-tagger` and `photo-tagger watch` both take them.

| Flag                         | Default | Env var | Description                                                                                         |
| ---------------------------- | ------- | ------- | --------------------------------------------------------------------------------------------------- |
| `--artist`, `--creator` NAME | none    | `-`     | Who took the photo. Goes to `EXIF:Artist`, `XMP-dc:Creator`, and `IPTC:By-line`.                    |
| `--copyright` TEXT           | none    | `-`     | The copyright notice. Goes to `EXIF:Copyright`, `IPTC:CopyrightNotice`, and `XMP-dc:Rights`.        |
| `--rights` TEXT              | none    | `-`     | `XMP-dc:Rights` alone, for when it reads differently from `--copyright`.                            |
| `--marked` / `--not-marked`  | unset   | `-`     | `XMP-xmpRights:Marked`: under copyright, or in the public domain. Left alone when you pass neither. |
| `--set` TAG=VALUE            | none    | `-`     | Any other tag, under its ExifTool name. Repeatable. An empty value deletes the tag.                 |

`--copyright` and `--rights` are templates. `{year}` becomes the year of the photo. `{artist}`
becomes the name from `--artist`. The year is the photo's own `EXIF:DateTimeOriginal`. If the photo
carries no capture date, the year comes from the file's timestamp. One command therefore stamps a
decade of shoots, each with its own year.

EXIF types its copyright and artist fields as ASCII. UTF-8 only became legal there in Exif 3.0 (CIPA
DC-008-2023). Nothing in a file says which of the two a value uses, so a `©` in `EXIF:Copyright` is
a bet on the reader. XMP is UTF-8 by definition and carries anything. That is what `--rights` is
for: write `©` there and `(c)` in `--copyright`. Without `--rights`, `XMP-dc:Rights` gets the
`--copyright` text.

Put the values in the config file to credit every run without typing them again:

```toml
[copyright]
artist = "Ada Lovelace"
copyright_notice = "Copyright (c) {year} {artist}, All rights reserved"
rights = "© {year} {artist}, All rights reserved"
marked = true
```

### Stamping photos that are already tagged

`photo-tagger set` writes the same fields on their own. It calls no model, reads no keyword, and
rewrites no title or description. Use it on a library that is already tagged, or when you want the
credit and nothing else.

```console
$ photo-tagger set -i ~/Pictures/2025 -r --embed-in-photo \
    --artist "Ada Lovelace" \
    --copyright "Copyright (c) {year} {artist}, All rights reserved" \
    --marked
Wrote 412 of 412 photo(s).
```

It takes `-i/--input`, `--ext`, `-r/--recursive`, and the five flags above. It also takes
`--sidecar-mode`, `--write-sidecar` / `--embed-in-photo`, `--backup-xmp` / `--no-backup-xmp`,
`--undo-log` / `--no-undo-log`, and `--dry-run`, with the meanings they have everywhere else. The
command records every write, so `photo-tagger undo` puts back a run that went to the wrong folder.
`--dry-run` prints the tags for each photo and writes none.

### Reading metadata back

`photo-tagger show` prints what a photo already carries and changes nothing. Every line names the
ExifTool tag the value came from. A value that lives in the XMP sidecar says so. Two tags that
disagree get a line each, which is how a stale caption under a sidecar becomes visible.

```console
$ photo-tagger show -i IMG_0001.jpg
/photos/IMG_0001.jpg

  Title           Osprey in Flight                              XMP:Title (sidecar)
  Creator         Ada Lovelace                                  XMP:Creator
  Copyright       © 2019 Ada Lovelace, All rights reserved      XMP:Rights
                  Copyright (c) 2019 Ada Lovelace, All rig...   EXIF:Copyright
  Rights marked   True                                          XMP:Marked
  Captured        2019:07:14 10:32:11                           EXIF:DateTimeOriginal
```

| Flag     | Default | Env var | Description                                                                           |
| -------- | ------- | ------- | ------------------------------------------------------------------------------------- |
| `--all`  | `false` | `-`     | Print every tag ExifTool reports instead of the curated list. Leaves out binary data. |
| `--json` | `false` | `-`     | Emit one JSON object per photo instead of a table.                                    |

`show` also takes `-i/--input`, `--ext`, and `-r/--recursive`. With `--json` each photo is one line,
so the output pipes straight into `jq`.

## Pacing

A vision model at full tilt can draw more power than a laptop charger supplies, so a run over a
large library discharges the battery even while plugged in. It also runs the machine hot and loud
and leaves little of it for anything else. Pacing fixes all three by spending part of the time not
working: the battery charges back, the fans wind down, and the average draw lands under the supply.

Everything here is off unless you ask for it: without `--duty` or `--battery-floor` no pacer is
built at all and a run works flat out, so `--work-minutes` and `--max-pause-minutes` describe the
shape of a rest rather than causing one. A rest falls at a photo boundary, after a whole work
window, never after each photo: a rest between two photos is too short to charge anything and gives
the model server the chance to unload the model. Every rest is capped for the same reason.

| Flag                        | Default | Env var             | Description                                                                                                                                                                                                                                                                                                                                           |
| --------------------------- | ------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--duty` N                  | `100`   | `DUTY_PERCENT`      | Percentage of the time the run may work, resting for the remainder. `100` works flat out. `70` rests roughly four minutes per ten-minute window.                                                                                                                                                                                                      |
| `--battery-floor` N         | `0`     | `BATTERY_FLOOR`     | Rest until the charge recovers whenever the battery falls to N percent. `0` never looks at the battery. Above 90 the run warns and carries on: a plugged-in laptop is rarely brim full, so a floor up there rests after nearly every window. 99 is the ceiling, since the resume level has to fit above it. Only rests while a charger is plugged in. |
| `--battery-resume` N        | `0`     | `BATTERY_RESUME`    | Charge that ends a rest started by `--battery-floor`, and must be above it. `0` sits 15 points above the floor.                                                                                                                                                                                                                                       |
| `--work-minutes` FLOAT      | `10.0`  | `WORK_MINUTES`      | Minutes of work between rests, for a run that is already resting. Changes nothing on its own.                                                                                                                                                                                                                                                         |
| `--max-pause-minutes` FLOAT | `15.0`  | `MAX_PAUSE_MINUTES` | Longest a single rest may last. Bounds the wait for a slow charger. Changes nothing on its own.                                                                                                                                                                                                                                                       |

`--battery-floor` needs no wattage numbers and adapts to screen brightness and whatever else the
machine is doing. It reads the charge from `pmset` on macOS, from `/sys/class/power_supply` on
Linux, and from `GetSystemPowerStatus` on Windows. A machine with no battery, or one the reading
fails on, is paced by `--duty` alone.

A paced run reports what the rests cost: `rests_taken` and `rested_seconds` appear in the summary
file written by `--summary-file`, and in the log as `pacing_summary`.

```bash
# Tag a large library overnight without flattening the battery.
photo-tagger -i ~/Pictures/Archive -r --skip-tagged --battery-floor 30 --duty 70
```

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

### Tracing model calls

`--otel` exports an OpenTelemetry span per model call (the request, how long it took, how many
tokens it cost, whether it failed) to a collector you run. It answers the questions a log file makes
you piece together by hand: which photo was slow, where a batch stalled, what a run actually cost.

It is off by default and there is no service behind it. Nothing is exported until you pass `--otel`
**and** name an endpoint, and the endpoint is always yours. This is separate from
[telemetry](#telemetry), which is the project's own anonymous beacon and carries none of this.

| Flag                                   | Default | Env var                                                              | Description                                                                                                                |
| -------------------------------------- | ------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `--otel` / `--no-otel`                 | `false` | `-`                                                                  | Export a trace of every model call over OTLP/HTTP.                                                                         |
| `--otel-endpoint` URL                  | none    | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_ENDPOINT` | Where to send traces. The flag wins; the signal-specific variable beats the general one, which gets `/v1/traces` appended. |
| `--otel-content` / `--no-otel-content` | `false` | `-`                                                                  | Also export the prompt and the generated title, description, and keywords.                                                 |

Install the exporter with the optional extra, then point it somewhere:

```bash
pip install 'photo-tagger[otel]'
photo-tagger -i ~/Pictures/Trip --otel --otel-endpoint http://localhost:4318/v1/traces
```

Any OTLP/HTTP backend works: [Logfire](https://pydantic.dev/logfire) (pydantic-ai's own, which
renders these spans natively), Jaeger, Grafana Tempo, or an OpenTelemetry Collector you already run.
For a hosted backend, set its auth header the standard way, with `OTEL_EXPORTER_OTLP_HEADERS`.

**What never leaves the machine.** The photo itself is never exported. Spans record that an image of
a given media type was sent, never its bytes, and there is no flag to change that. The text the
model generates is also kept out by default, because it describes your photos by name; add
`--otel-content` when you are debugging a prompt and trust the collector. Turning `--otel` on
without an endpoint is an error rather than a silent fallback to `localhost:4318`, so a run never
looks traced when nothing is listening.

Spans are batched and flushed when the process exits. Setting `OTEL_SERVICE_NAME` renames the
service; otherwise spans arrive as `photo-tagger`, tagged with the version that produced them. The
GUI has no command line, so it reads these three settings from the `[log]` table of the
[config file](../getting-started/configuration.md); a bad endpoint there is logged and the window
still opens.

## Display

The display group controls how a run makes itself visible: the progress bar, machine-readable
output, and the desktop alert when the batch ends.

| Flag                                   | Default           | Env var | Description                                                                                                                                                                                                                                                   |
| -------------------------------------- | ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--progress` / `--no-progress`         | progress (`true`) | `-`     | Live rich progress bar; auto-disabled when stderr is not a TTY.                                                                                                                                                                                               |
| `--json`                               | `false`           | `-`     | Emit one NDJSON line per processed photo to stdout (`file`, `status`, `from_cache`, `retry`, `title`, `description`, `keywords`, input/output/total tokens, `seconds`, `foreign_terms`). Logs and progress stay on stderr, so stdout pipes cleanly into `jq`. |
| `--notify` / `--no-notify`             | `false`           | `-`     | Raise a desktop notification when the batch ends, carrying the run's tally (`38 of 42 photos tagged, 4 failed.`).                                                                                                                                             |
| `--notify-sound` / `--no-notify-sound` | `false`           | `-`     | Play a sound with the notification. Implies `--notify`.                                                                                                                                                                                                       |

### End-of-batch notification

A long run is meant to be walked away from, so `--notify` calls you back when it is over. The alert
fires whether the batch fully succeeded or only partly did: a run that half failed is exactly the
one worth being told about. Under `watch`, every batch gets its own alert.

It uses whatever notifier the operating system already ships, so there is nothing to install:

| Platform      | Notification              | Sound (`--notify-sound`)           |
| ------------- | ------------------------- | ---------------------------------- |
| macOS         | `osascript`               | the notification's own alert sound |
| Linux and BSD | `notify-send` (libnotify) | `canberra-gtk-play`, else `paplay` |
| Windows       | a PowerShell toast        | the system alert sound             |

Everything about it is best-effort. On a machine with no notifier on `PATH`, in a headless session,
or where notification permission was denied, nothing appears and the run is unaffected; the failure
is recorded in the debug log only. When a sound was asked for and the platform has nothing to play
it with, the terminal bell is rung instead (skipped when stderr is redirected).

## Telemetry

photo-tagger sends one anonymous beacon per run, and one on a crash. It is opt-out and carries no
photos, paths, filenames, tags, or error messages; see [Telemetry](../telemetry.md) for the exact
payload and every way to switch it off.

| Flag                             | Default     | Env var                                   | Description                                                                                                                                                                                                    |
| -------------------------------- | ----------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--telemetry` / `--no-telemetry` | on (`true`) | `PHOTO_TAGGER_TELEMETRY` / `DO_NOT_TRACK` | Send anonymous usage stats and crash reports. An opt-out anywhere wins over an opt-in. `PHOTO_TAGGER_TELEMETRY=0` turns it off, and `PHOTO_TAGGER_TELEMETRY=1` overrules `DO_NOT_TRACK` for photo-tagger only. |

## Artifacts

The artifacts group points at side files: a custom prompt, a run summary, a per-photo CSV report, a
result cache, and a lock.

| Flag                           | Default     | Env var | Description                                                                                                                                                                                                         |
| ------------------------------ | ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--prompt-file` PATH           | none        | `-`     | Replace the default user prompt with the contents of PATH; existing photo metadata is still appended automatically.                                                                                                 |
| `--summary-file` PATH          | none        | `-`     | Write a JSON run summary (success/failure counts, failed files, token usage, wall time) on completion.                                                                                                              |
| `--csv-file` PATH              | none        | `-`     | Write a CSV report with one row per photo (see below). Rows stream as photos finish, so a stopped run still leaves a valid file.                                                                                    |
| `--cache-file` PATH            | none        | `-`     | SQLite cache of model outputs, keyed on an image-data hash that ignores metadata (so it survives `--embed-in-photo`). Reruns skip the model call when nothing relevant changed. Created if missing; safe to delete. |
| `--lock-file` PATH             | none        | `-`     | Acquire an exclusive file lock before running; refuse to start if another photo-tagger already holds it. Works on Linux, macOS, and Windows.                                                                        |
| `--undo-log` / `--no-undo-log` | on (`true`) | `-`     | Record every file the run writes so `photo-tagger undo` can put it back.                                                                                                                                            |

### CSV report

Where `--summary-file` writes one JSON object for the whole run and `--json` streams NDJSON to
stdout, `--csv-file` writes a spreadsheet-friendly table with **one row per photo**. It is the
single file that gathers everything extracted and computed for each image:

- `filename`, `file`, `status`
- `title`, `description`, `keywords` (the keywords actually written), `hierarchical_keywords`
- `existing_keywords` (what was already on the file)
- `camera_model`, `lens_model`, `capture_date`, `gps_position`, `city`, `country` (read EXIF)
- `input_tokens`, `output_tokens`, `total_tokens`, `seconds`, `from_cache`, `retry`
- `foreign_terms`: filled only when the generated text mixes writing systems, with the words in the
    odd script out (see [Mixed-language output](#mixed-language-output))

Multi-value cells (the keyword lists) are joined with a semicolon and a space. Rows are flushed as
each photo completes, so interrupting the run with Ctrl-C still leaves a complete, openable CSV of
the work done so far. `--csv-file` and `--json` can be used together; both observe every photo. A
`--dry-run` still fills the report, which makes it handy for previewing a batch before writing any
metadata.

## Mixed-language output

Asked for English, a model now and then reaches for a word in its own script anyway, and a title
comes back as `Decorated陶笛 with Geometric Pattern` where `Ocarina` would have said the same thing.
Every run checks the generated text for this: whatever script the bulk of the title, description,
and keywords is written in is taken as the one it meant to use, and anything in another script is
reported. No language list is involved, so it works for whatever `--output-language` is set to.

A flagged photo is still tagged. Mixed script is right often enough (the photo shows a sign, a book
cover, a printed character) that only the person looking at the photo can say. What the flag does is
point you at it:

- a `foreign_terms` cell in the `--csv-file` report and key in the `--json` stream, naming the words
- a `files_flagged_for_review` line in the log at the end of the run
- `flagged_files` in the `--summary-file` JSON
- in the GUI, an amber note over the editable fields and a count in the closing status line

`--repair-language` goes one step further: on a flagged photo only, it asks the same model again,
with the same image, quoting its previous answer and naming the offending words. The image goes
along because the question that decides the right fix is whether the photo shows that text, which no
text-only pass can answer. That costs one extra call per affected photo, so it is off by default; a
clean answer never pays for it. If the repair fails or comes back mixed again, the original answer
is kept and the photo stays flagged. The repaired answer is what goes into the `--cache-file`, so a
rerun replays the fix instead of paying for it twice. A mixed answer that a run without
`--repair-language` cached is repaired when it is replayed, and the fix replaces it in the cache. A
photo whose repair comes back mixed again keeps its original answer in the cache, so each run with
`--repair-language` asks once more for it.

In the GUI: **Settings > Rewrite Mixed-Language Output**.

## Building a vocabulary

`--vocabulary-strict` is what stops a catalog sprawling, and it needs a keyword file worth
enforcing. `photo-tagger vocabulary` writes one from the library you already have:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --report dropped.csv
```

It reads the keywords your photos already carry, counts how often each one is used, and keeps the
ones that earn their place. The read goes through exiftool, so **the application does not matter**:
digiKam, darktable, Immich, PhotoPrism, Synology Photos, Piwigo and the rest all write XMP/IPTC
keywords, and only Lightroom offers a keyword-list export at all. `XMP-lr:HierarchicalSubject`
carries the hierarchy those photos really use, so the generated file keeps it.

Nothing is written to your photos or your catalog. The output is a text file to read and edit.

| Flag                   | Default    | Description                                                                               |
| ---------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| `-i`, `--input` PATH   | none       | Photos or folders to read keywords from; repeat the flag. Honors `--ext` and `-r`.        |
| `-o`, `--output` PATH  | (required) | Where to write the vocabulary file.                                                       |
| `--from-export` PATH   | none       | Read a Lightroom keyword export (`.txt` or `.csv`) instead of, or as well as, the photos. |
| `--min-uses` N         | `2`        | Keep a keyword only when the library uses it at least this often.                         |
| `--max-terms` N        | `4800`     | Cap the file, dropping the least-used first. `0` means no cap.                            |
| `--allow-digits`       | `false`    | Keep keywords containing digits (dropped by default as measurements and model numbers).   |
| `--output-language`    | `English`  | Language the catalog's keywords are in; only English plurals fold onto their singular.    |
| `--flat`               | `false`    | Write bare keywords instead of their hierarchies.                                         |
| `--report` PATH        | none       | Write a CSV of every dropped keyword, its count, and the rule that cut it.                |
| `--organize`           | `false`    | Model pass for synonyms and a hierarchy (see [below](#organizing-with-the-model)).        |
| `--organize-workers` N | `1`        | Model requests to run at once while organizing.                                           |

`--organize` also takes the provider flags (`--provider`, `-m/--model`, `-u/--url`, `-k/--api-key`),
with the same meanings and the same config-file and environment defaults as a tagging run.

### Which source to use

Prefer the photos. An export has no usage data at all, so `--from-export` counts *occurrences in the
keyword tree* instead: a term filed under forty parents scores forty, however many photos carry it.
It is a usable proxy for a catalog that is not on this machine, not the same measure. Pass both and
the counts are added together.

### What the rules do

Applied in this order, each one reported in `--report`:

1. **Shape.** A keyword with digits, odd punctuation, more than three words, or over 30 characters
    is a measurement, a path, or a sentence, not a subject.
2. **Rarity.** `--min-uses` drops the one-offs. In a catalog an AI has been writing to this is the
    rule that does the work: most keywords are used exactly once.
3. **Variants.** One concept keeps one spelling, the most-used one. Nothing is lost by this, since
    vocabulary matching folds case, punctuation, and plurals anyway: a photo tagged `Animals` still
    snaps onto `Animal`.
4. **The cap.** `--max-terms` removes the least-used survivors last, so it never cuts a term an
    earlier rule would have kept.

The result is deterministic: the same library gives the same file, ties broken alphabetically.

!!! tip

    Read the file before you trust it, and tune from the report rather than by guesswork. If a keyword
    you care about was dropped, `dropped.csv` names it with the count that would have kept it. The
    hierarchy is worth a look too: a catalog a tool has been writing to can file `Beach` under `Sand`,
    and a vocabulary imposes its hierarchy on every photo it matches. `--flat` drops the hierarchies
    when the source is not worth keeping.

### Organizing with the model

Counting settles which keywords are worth keeping. It cannot settle two things, and `--organize`
asks the model for exactly those two, over the keywords that already survived:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --organize --organize-workers 4
```

- **Synonyms.** Matching already folds case, punctuation, and plurals, so `Animal` and `Animals` are
    one keyword without any help. It cannot know that `Golden Light` is `Golden Hour`. The winner
    keeps the entry and the others are written as `{braces}` on it, so a photo tagged with a folded
    spelling still matches; each fold is listed in `--report` with the reason `synonym`.
- **A hierarchy.** The categories are chosen once, from the most-used keywords, and every chunk of
    the list is then filed against that one fixed set. Asking each chunk to invent its own would
    give `Animal` in one and `Animals` in the next, which is the sprawl this command exists to end.

Five properties keep the pass from making the file worse:

- **It never decides what to keep.** That is already settled, by counting, before the model sees
    anything.
- **It never invents a keyword.** Every string the model returns is matched back to a keyword that
    was sent, loosely enough to survive a retyped capital; anything else is discarded and counted.
- **A failure costs nothing but the organizing.** A chunk that errors, or a keyword the model
    forgets, keeps the shape the deterministic pass gave it.
- **A group cannot swallow a category.** At most three synonyms are accepted per keyword; a group
    claiming more is refused whole. A model listing five is not naming synonyms, it is emptying a
    category into one keyword (`People` taking `Person`, `Human`, `Woman`, and `Man`), and each one
    it takes is a keyword your catalog loses.
- **A degenerate category list is refused.** Asked for six to twenty top-level categories, a weak
    model sometimes echoes the keyword list back. Keeping the first twenty of that would look like
    an answer and behave like noise, so the hierarchy is skipped and only synonyms are folded.

### Choosing a model for it

This pass is **text only**: it reads a list of words, not an image. Your tagging model is a
vision-language model, and its vision half buys nothing here, so it is worth pointing `--model` at
something else:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --organize --model openai/gpt-oss-20b
```

What matters is instruction-following and reliable structured output, not size. Three failure modes
tell you a model is the wrong choice, and all three are visible in the run log:

- `vocabulary_organize_request_failed` on every chunk, with a token-limit message: a reasoning model
    spending its whole budget thinking before it answers. The budget is already generous; a model
    that still cannot finish inside it is not usable here.
- **Chunks that take minutes each.** Grouping words is recall, not deduction, but a reasoning model
    left to itself will spend thousands of tokens deliberating over a list of sixty of them. Every
    request therefore asks for no reasoning (`reasoning_effort: "none"`), which servers that do not
    know the setting simply ignore. On one local 31B model that setting was the difference between
    **13 minutes for sixteen keywords and 12 seconds**. If chunks are still slow, the server is
    probably not honoring it; check whether your provider exposes its own switch.
- `vocabulary_synonym_group_refused` many times over: the model is folding categories into keywords,
    and the guardrail is the only thing between it and your catalog.

!!! tip

    Time one chunk before committing a whole library to it. Add `--max-terms 60` so exactly one chunk is
    sent, and watch the clock between `vocabulary_organize_started` and `vocabulary_organized`. Multiply
    by the chunks your real list needs, then divide by `--organize-workers`.

!!! warning "Categories are new keywords"

    A category the model names becomes a parent in the file, so it will be written to your photos as a
    hierarchical keyword even if your library never used that word. The file's header lists them for
    exactly this reason. If you would rather not have any, use `--flat` or drop the parents by hand.

The list is sent in chunks of 60 keywords: roughly one request per 60 keywords plus one for the
categories, so about 80 requests for a 4,800-keyword file. `--organize-workers` runs several at
once. Sampling is fixed at temperature 0 and chunks are reassembled by position, so the same list
organizes the same way whatever order the replies arrive in, but a model is not a pure function:
treat the output as a proposal to read, which is what the whole file is anyway.

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

## Watching a folder

`photo-tagger watch` is the import-time workflow: point it at the folder your card reader, tethered
capture, or sync client fills, and leave it running.

```bash
photo-tagger watch -i ~/Pictures/Inbox --recursive --skip-tagged
```

Photos already in the folder are tagged first, then each new one as it lands. Every flag the tagging
command takes works here too and applies to each batch, with a single agent, cache, and CSV/NDJSON
file shared by the whole session. Each batch records its own undo journal, so
[`photo-tagger undo`](#undoing-a-run) puts back the last import rather than everything since the
watch started. Stop it with Ctrl-C.

| Flag             | Default | Description                                            |
| ---------------- | ------- | ------------------------------------------------------ |
| `--interval` SEC | `5.0`   | Seconds between folder scans.                          |
| `--settle` SEC   | `2.0`   | Seconds a file must sit unchanged before it is tagged. |

Two behaviors worth knowing:

- **A file is only tagged once it stops changing.** It must be unchanged across two scans *and* its
    last modification must be at least `--settle` seconds old, so a photo still being copied is left
    alone until the copy finishes. This also means the first batch appears one `--interval` after
    the watch starts, not instantly.
- **A failing batch does not stop the watch.** The failure is logged and the next photo to land gets
    its turn. Scanning is a plain directory listing rather than a filesystem-event API, so it
    behaves the same on every platform and over network shares.

## Undoing a run

Every run records the files it writes, so a batch tagged with the wrong prompt, the wrong model, or
the wrong vocabulary can be reverted in one command instead of by hand:

```console
$ photo-tagger undo
Undoing run 20260501142233-8421.jsonl
Undoing 412 write(s)

  deleted    /Users/you/Pictures/Trip/IMG_0001.xmp
  restored   /Users/you/Pictures/Trip/IMG_0002.xmp
  ...

Every recorded write was put back.
```

Sidecars the run **created** are deleted; files it **overwrote** are restored from ExifTool's
`*_original` backup. The journals are small JSON-lines files under the state directory
(`$XDG_STATE_HOME/photo-tagger/runs`, or `~/.local/state/photo-tagger/runs`), pruned to the 50 most
recent runs and 90 days.

| Flag         | Default    | Description                                               |
| ------------ | ---------- | --------------------------------------------------------- |
| `--run` PATH | newest run | Undo this journal instead of the most recent one.         |
| `--list`     | `false`    | List the recorded runs (name, file count, path) and exit. |
| `--dry-run`  | `false`    | Report what would be put back, without touching anything. |
| `--force`    | `false`    | Also revert files that changed after the run wrote them.  |

Three things are deliberately left alone:

- **Files changed since the run.** A different size or mtime means someone edited the file
    afterwards, so undo reports it and moves on. `--force` overrides this.
- **Writes made with `--no-backup-xmp`.** There is no copy of the previous contents, so an
    overwritten file cannot be restored. Newly created sidecars are still deleted.
- **Runs from the desktop GUI**, which does not record a journal.

`photo-tagger undo` exits 1 when there is nothing to undo or when any entry was left alone, and 0
when every recorded write was put back. Dry runs are not written to a journal (they change nothing),
and `--no-undo-log` turns recording off for a run.
