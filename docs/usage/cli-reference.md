---
icon: lucide/square-terminal
---

# CLI reference

photo-tagger runs as a single command, `photo-tagger`, built with
[cyclopts](https://github.com/BrianPugh/cyclopts). Its flags are grouped into logical option groups.
This page documents every flag: its default value, its environment variable (`-` means none), and
what the flag does.

Any flag you pass on the command line overrides the value in the configuration file, and the default
that the environment sets. See [Configuration](../getting-started/configuration.md) for the full
precedence rules and TOML layout.

## Commands

Running `photo-tagger` with image inputs tags them (the default command). Seven subcommands exist:

| Command                   | Description                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `photo-tagger`            | Tag the given images (default). Documented by the option groups below.                                      |
| `photo-tagger doctor`     | Pre-flight check: makes sure that ExifTool is on `PATH` and the provider serves the model, then exits 0/1.  |
| `photo-tagger show`       | Print the metadata a photo already carries (see [Copyright and other fields](#copyright-and-other-fields)). |
| `photo-tagger set`        | Write copyright and other fields without calling a model (same section).                                    |
| `photo-tagger vocabulary` | Build a keyword file from a library's own keywords (see [Building a vocabulary](#building-a-vocabulary)).   |
| `photo-tagger watch`      | Watch folders and tag photos as they arrive (see [Watching a folder](#watching-a-folder)).                  |
| `photo-tagger undo`       | Put back what the last run wrote (see [Undoing a run](#undoing-a-run)).                                     |
| `photo-tagger gui`        | Launch the optional desktop GUI. Requires the `gui` extra. See [Desktop GUI](gui.md).                       |

`doctor` accepts `--provider`, `-m/--model`, `-u/--url`, and `-k/--api-key` (same meanings as below)
and honors the same configuration file and environment variables. If a tagging run will not start,
run `doctor` first:

```console
$ photo-tagger doctor --provider lmstudio --model qwen/qwen3-vl-30b
photo-tagger 0.5.0 environment check

  OK    ExifTool: /usr/bin/exiftool
  OK    Model 'qwen/qwen3-vl-30b' on lmstudio: available at http://localhost:1234/v1

All checks passed.
```

## Input and scanning

`-i/--input` is required and repeatable: pass it once per file or directory you want to process.

| Flag                         | Default    | Env var | Description                                                                                                             |
| ---------------------------- | ---------- | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `-i`, `--input` PATH         | (required) | `-`     | One or more files, directories, or both. Repeat the flag for each one.                                                  |
| `--ext`, `--extensions` LIST | `cr3,jpg`  | `-`     | Comma-separated list of extensions to scan for in directories (case-insensitive).                                       |
| `-r`, `--recursive`          | `false`    | `-`     | Recurse into subdirectories while scanning input directories.                                                           |
| `-w`, `--workers` N          | `1`        | `-`     | Process N photos at once with a thread pool. The model server is usually the bottleneck.                                |
| `--skip-from` PATH           | none       | `-`     | Skip the filenames listed in PATH, one per line. A line that starts with `#` is a comment.                              |
| `--append-to-skip-file` PATH | none       | `-`     | Append each successfully tagged filename to PATH as the run continues. If PATH does not exist, photo-tagger creates it. |

## Provider

The provider group selects the backend and how to reach it. Prefer the API-key environment variables
over `--api-key` so the key never lands in your shell history.

| Flag                  | Default                                                                                                            | Env var                                                                             | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `--provider` NAME     | `lmstudio`                                                                                                         | `-`                                                                                 | Backend: `ollama`, `lmstudio`, `llamacpp`, or `openai`.                         |
| `-m`, `--model` NAME  | `qwen/qwen3-vl-30b`                                                                                                | `MODEL_NAME`                                                                        | Vision-language model identifier.                                               |
| `-u`, `--url` URL     | `http://localhost:1234/v1` (lmstudio), `http://localhost:11434/v1` (ollama), `http://localhost:8080/v1` (llamacpp) | `LM_STUDIO_BASE_URL` / `OLLAMA_BASE_URL` / `LLAMA_CPP_BASE_URL` / `OPENAI_BASE_URL` | Provider API base URL.                                                          |
| `-k`, `--api-key` KEY | none                                                                                                               | `OLLAMA_API_KEY` / `LM_STUDIO_API_KEY` / `LLAMA_CPP_API_KEY` / `OPENAI_API_KEY`     | API key. Prefer the environment variables over the flag. Required for `openai`. |
| `--retries` N         | `5`                                                                                                                | `RETRIES`                                                                           | Automatic retries after the model's output fails schema validation.             |

## Inference

These flags tune the sampling and the image sent to the model. A lower temperature and a frequency
penalty keep the output focused. The JPEG flags control how much detail the model sees.

| Flag                                         | Default   | Env var             | Description                                                                                                                                                                                                                                                           |
| -------------------------------------------- | --------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output-language`, `--lang` NAME           | `English` | `-`                 | Language of the generated title, description, and keywords. Use any language name the model understands, for example `German` or `"Brazilian Portuguese"`.                                                                                                            |
| `--hint` TEXT                                | none      | `-`                 | A note about every photo in the run that the model trusts over its own reading of the image, for example `"The animal in these photos is a deer"`. Changes the cache namespace, so hinted runs never replay hint-less results.                                        |
| `--repair-language` / `--no-repair-language` | `false`   | `-`                 | When the generated text mixes writing systems, spend one more model call to ask for it in `--output-language` (`Ocarina` for a stray `陶笛`). It is off by default. Either way, the affected photos are flagged. See [Mixed-language output](#mixed-language-output). |
| `--temperature` FLOAT                        | `0.2`     | `TEMPERATURE`       | Sampling temperature.                                                                                                                                                                                                                                                 |
| `--max-tokens` N                             | `1200`    | `MAX_TOKENS`        | Maximum tokens to generate.                                                                                                                                                                                                                                           |
| `--timeout-seconds` FLOAT                    | `60.0`    | `TIMEOUT_SECONDS`   | Timeout of one model request. If a request times out, the retry loop takes over.                                                                                                                                                                                      |
| `--frequency-penalty` FLOAT                  | `0.5`     | `FREQUENCY_PENALTY` | Penalty on repeated tokens. It discourages repetitive output loops.                                                                                                                                                                                                   |
| `--jpeg-dimensions` N                        | `640`     | `JPEG_DIMENSIONS`   | Longest edge (px) of the JPEG sent to the model, and the biggest lever on speed. See [Image size](#image-size).                                                                                                                                                       |
| `--jpeg-quality` N                           | `80`      | `JPEG_QUALITY`      | JPEG quality (1-100) of the image sent to the model.                                                                                                                                                                                                                  |

### Image size

`--jpeg-dimensions` sets the longest edge of the JPEG that the model sees. It is the biggest lever
on how long a photo takes. Prefill time scales with the pixel count, and the rest of the model call
does not change with size. On a local Qwen3-VL-30B model, one photo costs roughly this much prefill
time, against about 1.9 s to write the answer:

- 640 px: about 0.49 s
- 896 px: about 0.83 s
- 1024 px: about 1.11 s
- 1280 px: about 1.80 s

A controlled probe found what the extra pixels really help with: reading text inside the frame, not
recognizing the subject. A 30 px line of text in a 4032 px photo was readable at 896 px, and not at
768 px or below. So the choice depends on what you need:

- 640: the fast choice
- 768: sees more detail
- 896: reads large text, such as signs
- 1024: reads fine text, such as documents and labels

Multiples of 128 line up with the vision tower's patch grid.

Changing this value starts a fresh cache namespace, because the answers already stored describe what
the model saw at the old size. Pick it before a long run rather than during one.

## Output

The output group decides what metadata is written and where. By default photo-tagger writes an XMP
sidecar next to each image and leaves the original untouched.

| Flag                                                 | Default             | Env var | Description                                                                                                                                                                                                                                       |
| ---------------------------------------------------- | ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--preserve-keywords` / `--overwrite-keywords`       | preserve (`true`)   | `-`     | Merge with existing keywords instead of replacing them.                                                                                                                                                                                           |
| `--preserve-title` / `--overwrite-title`             | overwrite (`false`) | `-`     | Keep the title a photo already has, and write the generated one only where there is none, instead of replacing it.                                                                                                                                |
| `--preserve-description` / `--overwrite-description` | overwrite (`false`) | `-`     | Keep the description a photo already has, instead of replacing it. Replacing is what clears a camera-written placeholder.                                                                                                                         |
| `--write-title` / `--no-write-title`                 | write (`true`)      | `-`     | Generate and write a title.                                                                                                                                                                                                                       |
| `--write-description` / `--no-write-description`     | write (`true`)      | `-`     | Generate and write a description.                                                                                                                                                                                                                 |
| `--write-keywords` / `--no-write-keywords`           | write (`true`)      | `-`     | Write keywords, merged as `--preserve-keywords` sets. `--no-write-keywords` leaves the existing keywords alone.                                                                                                                                   |
| `--write-copyright` / `--no-write-copyright`         | write (`true`)      | `-`     | Write the copyright block below. Nothing is written until copyright values are set, by flags or in the configuration file. Use `--no-write-copyright` to skip values you already set, for one run.                                                |
| `--write-sidecar` / `--embed-in-photo`               | sidecar (`true`)    | `-`     | Shorthand: `--write-sidecar` means `--sidecar-mode all`, and `--embed-in-photo` means `--sidecar-mode none`.                                                                                                                                      |
| `--sidecar-mode` {all,none,raw,both}                 | `all`               | `-`     | Where metadata goes: `all` writes a sidecar, `none` embeds it, `raw` writes a sidecar for RAW files and embeds the rest, and `both` does both. Wins over the two flags above.                                                                     |
| `--sidecar-override` EXT=TARGET                      | none                | `-`     | Send one file type elsewhere: TARGET is `sidecar`, `photo`, or `both`. Wins over `--sidecar-mode` for that extension. Repeat it for more types. `dng=photo` matches Lightroom Classic ([recipe](recipes.md#tag-dng-files-for-lightroom-classic)). |
| `--backup-xmp` / `--no-backup-xmp`                   | backup (`true`)     | `-`     | Keep ExifTool's `*_original` backup before writing. `--no-backup-xmp` passes `-overwrite_original` instead.                                                                                                                                       |
| `--max-keywords` N                                   | none (keep all)     | `-`     | Cap AI-generated keywords kept per photo before merging.                                                                                                                                                                                          |
| `--vocabulary` PATH                                  | none                | `-`     | Restrict generated keywords to the terms in PATH (see below).                                                                                                                                                                                     |
| `--vocabulary-strict`                                | `false`             | `-`     | Drop generated keywords the vocabulary does not cover instead of writing them as-is.                                                                                                                                                              |
| `--session-gap` MINUTES                              | `0` (off)           | `-`     | Group photos into shoots and make each shoot's keywords agree with itself (see below).                                                                                                                                                            |
| `--dry-run`                                          | `false`             | `-`     | Run the model and log the proposed metadata, but write nothing.                                                                                                                                                                                   |

### Controlled vocabulary

`--vocabulary` points at the keyword list your catalog already uses, so a run cannot seed it with
near-duplicates of keywords you curated by hand. Two file shapes are accepted:

- A Lightroom keyword-list export (_Metadata > Export Keywords_). Lightroom offers two shapes for
    this export:
    - `.txt` (_Exclude Keyword Tag Options_): one keyword per line, with children indented under their
        parent. Use `{braces}` for synonyms and `[brackets]` for keywords marked "do not export".
    - `.csv` (_Include Keyword Tag Options_): the same list, behind four option columns. photo-tagger
        pulls the keyword column out on its own.
- A plain list: one term per line, optionally as a full path, in either `Animal|Bird|Osprey` or
    `Osprey<Bird<Animal` form. Blank lines and `#` comments are ignored. A comment needs a space
    after the hash, so a hashtag-style keyword such as `#Diversity` stays a keyword.

```text
Animal
	Bird
		Osprey
		{Sea Hawk}
	Mammal
Landscape
```

Every generated keyword is matched against the file. Matching ignores case and punctuation, with a
conservative fuzzy pass for typos and longer variants. A match is rewritten to the file's own
spelling and hierarchy, so `ospreys`, `Sea Hawk`, and `Osprey<Raptor<Wildlife` all land as
`Animal|Bird|Osprey`. The file's spelling is used exactly as written, so a catalog that keeps its
keywords in lower case stays that way.

Keywords the file does not cover pass through untouched, unless `--vocabulary-strict` is set. Then
they are dropped and reported. The run summary's `vocabulary_dropped` field names every rejected
term and how often it came up: that is the list to check as you grow the vocabulary.
`vocabulary_mapped` counts the rewrites.

#### Languages other than English

Write the vocabulary in the same language you generate in ([`--output-language`](#inference)). A
German run cannot match an English catalog, whatever the matcher does.

Matching itself is language-neutral except in one place. Folding a plural onto its singular, for
example `Ospreys` to `Osprey`, uses English rules. So it applies to English output and not to any
other language. Applied to German, the same rule merges `Alles` into `Alle` by mistake, and it says
nothing about `птицы` at all. Everything else works in any script:

- Case folding, including `Straße` and `Strasse`
- Punctuation and spacing
- The fuzzy pass, which still unifies longer inflections, such as `Landschaften` with `Landschaft`
    or `Закаты` with `Закат`

No ratio can safely catch the short inflected forms. Declare them in the file with the synonym
syntax instead, which is exact and needs no guessing:

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

Each photo is analyzed on its own. Forty frames of the same bird can come back under different
spellings and hierarchies: one frame as `Osprey` under `Bird<Animal`, the next as `Ospreys` under
`Raptor<Wildlife`. Lightroom then shows four keywords where there is one subject.

`--session-gap MINUTES` fixes that without needing a vocabulary file. The batch is split into shoots
wherever the capture time (EXIF `DateTimeOriginal`, or else the file's mtime) jumps by more than the
given gap. Every photo in a shoot is analyzed first. Then the session's own output becomes its
vocabulary: the spelling most of the session used wins, and so does the hierarchy most of it used.
Only then does photo-tagger write anything.

It reads `--output-language` for the same reason the vocabulary file does. So a German or Russian
shoot is harmonized under that language's rules, not English ones. On top of that, it folds
close-enough variants together: a shoot that said `Закаты` twice and `Закат` once writes `Закаты`
throughout, and one that said `Landschaften` and `Landschaft` settles on one of them. Short words
are never folded this way, so `Alle` and `Alles` stay two keywords.

```bash
photo-tagger -i ~/Pictures/Trip -r --session-gap 60
```

Because the vocabulary is derived from the finished results rather than fed to the model, the
outcome does not depend on which photo finished first: the same batch harmonizes the same way every
time, at any `--workers` setting. It composes with `--vocabulary`, which is applied per photo first.

Two details worth knowing:

- Photos are processed in capture order rather than the order they were listed.
- Sessions run one after another. The photos inside one session still run in parallel, and a
    session's writes happen in a burst at its end. If a write fails there, it is reported as failed
    and not retried. The model's work is already done and harmonized. An ExifTool write that failed
    for a filesystem reason is not the kind of failure that a second attempt clears. Analysis
    failures are still retried as usual, and a photo that the retry pass recovers lands on its
    shoot's agreed terms.

## Copyright and other fields

No model can guess who took a photo or who owns it. You write those fields yourself, and they go
into the same ExifTool call as the generated metadata. They belong to the output group, so
`photo-tagger` and `photo-tagger watch` both take them.

| Flag                                               | Default | Env var | Description                                                                                                          |
| -------------------------------------------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `--artist`, `--creator` NAME                       | none    | `-`     | Who took the photo. Goes to `EXIF:Artist`, `XMP-dc:Creator`, and `IPTC:By-line`.                                     |
| `--copyright` TEXT                                 | none    | `-`     | The copyright notice. Goes to `EXIF:Copyright`, `IPTC:CopyrightNotice`, and `XMP-dc:Rights`.                         |
| `--rights` TEXT                                    | none    | `-`     | `XMP-dc:Rights` alone, for text that reads differently from `--copyright`.                                           |
| `--marked` / `--not-marked`                        | unset   | `-`     | `XMP-xmpRights:Marked`: under copyright, or in the public domain. If you pass neither, photo-tagger leaves it alone. |
| `--preserve-copyright` / `--no-preserve-copyright` | keep    | `-`     | Keep the name and the notice a photo already has. `--no-preserve-copyright` replaces them.                           |
| `--set` TAG=VALUE                                  | none    | `-`     | Any other tag, under its ExifTool name. Repeatable. An empty value deletes the tag.                                  |

`--copyright` and `--rights` are templates. `{year}` becomes the year of the photo, from its capture
date: `EXIF:DateTimeOriginal`, then the XMP and IPTC dates, then `EXIF:CreateDate`. One command
therefore stamps a decade of shoots, each with its own year. `{artist}` becomes the name from
`--artist`, or else the name the photo already carries.

A placeholder is a chain of sources, separated by `|`. photo-tagger tries them from left to right
and uses the first one with a value:

- A tag name with its group reads that tag from the photo, for example `{EXIF:Artist}`.
- A `%` after a tag formats a date with strftime codes, for example `{EXIF:DateTimeOriginal%Y}`.
- Text in double quotes is a fallback. It always wins when the chain reaches it, and it can hold
    placeholders of its own.
- If a chain ends in `""` and no source answers, the placeholder drops out with the space before it.
- If a chain ends with no quoted text and no source answers, photo-tagger writes no notice on that
    photo and warns.

`{year}` is short for the capture-date chain with `""` at the end. So is `{artist}` for `--artist`,
`EXIF:Artist`, `XMP-dc:Creator`, and `IPTC:By-line`. A photo without a capture date therefore gets
`Copyright (c) Ada Lovelace, All rights reserved` from the example below, when `--artist` is
`Ada Lovelace`. Write `{year!}` or `{artist!}` to write no notice on such a photo instead. Its
creator and rights flag are still written. photo-tagger never uses the file's timestamp, because a
copy or an edit changes it. Name `{File:FileModifyDate%Y}` in a chain if you want it. Write `{{` and
`}}` for literal braces.

```text
Copyright (c) {year} {artist}, All rights reserved
Copyright (c) {year!} {EXIF:Artist|XMP-dc:Creator|"Studio Lovelace"}
```

By default, a photo that already has a notice in EXIF, XMP, or IPTC keeps it, with its XMP rights
and its rights flag. A photo that already names a creator keeps that name. A second run therefore
never dates an old notice again, and a folder with photos from other people keeps their credit. When
the creator is kept, `{artist}` in a new notice names that creator, not `--artist`. Only the files a
save writes count: a notice in a sidecar does not stop a save into the photo.
`--no-preserve-copyright` writes your values over the old ones. `--set` is always written.

EXIF types its copyright and artist fields as ASCII. UTF-8 only became legal there in Exif 3.0 (CIPA
DC-008-2023). Nothing in a file says which of the two a value uses, so a `©` in `EXIF:Copyright` is
a bet on the reader. XMP is UTF-8 by definition and carries anything. That is what `--rights` is
for: write `©` there and `(c)` in `--copyright`. Without `--rights`, `XMP-dc:Rights` gets the
`--copyright` text.

Put the values in the configuration file to credit every run without typing them again:

```toml
[copyright]
artist = "Ada Lovelace"
copyright_notice = "Copyright (c) {year} {artist}, All rights reserved"
rights = "© {year} {artist}, All rights reserved"
marked = true
```

### Stamping photos that are already tagged

`photo-tagger set` writes the same fields on their own. It calls no model, reads no keyword, and
rewrites no title or description. Use it on a library that is already tagged, or any time you want
just the credit and nothing else.

```console
$ photo-tagger set -i ~/Pictures/2025 -r --embed-in-photo \
    --artist "Ada Lovelace" \
    --copyright "Copyright (c) {year} {artist}, All rights reserved" \
    --marked
Wrote 412 of 412 photo(s).
```

It takes `-i/--input`, `--ext`, `-r/--recursive`, and the six flags above. It also takes these
flags, with the meanings they have everywhere else:

- `--sidecar-mode`
- `--sidecar-override`
- `--write-sidecar` / `--embed-in-photo`
- `--backup-xmp` / `--no-backup-xmp`
- `--undo-log` / `--no-undo-log`
- `--dry-run`

The command records every write, so `photo-tagger undo` puts back a run that went to the wrong
folder. `--dry-run` prints the tags for each photo and writes none.

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

A vision model at full tilt can draw more power than a laptop charger supplies. So a run over a
large library discharges the battery, even while it stays plugged in. It also runs the machine hot
and loud and leaves little of it for anything else. Pacing fixes all three by resting part of the
time: the battery charges back, the fans wind down, and the average draw lands under the supply.

Everything here is off unless you ask for it. Without `--duty` or `--battery-floor`, no pacer is
built at all, and a run works flat out. So `--work-minutes` and `--max-pause-minutes` describe the
shape of a rest, rather than causing one. A rest falls at a photo boundary, after a whole work
window, never after each photo: a rest between two photos is too short to charge anything and gives
the model server the chance to unload the model. Every rest is capped for the same reason.

| Flag                        | Default | Env var             | Description                                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--duty` N                  | `100`   | `DUTY_PERCENT`      | Percentage of the time the run can work. It rests for the remainder. `100` works flat out. `70` rests roughly four minutes per ten-minute window.                                                                                                                                                                                                    |
| `--battery-floor` N         | `0`     | `BATTERY_FLOOR`     | Whenever the battery falls to N percent, rest until the charge recovers. `0` never looks at the battery. Above 90 the run warns and carries on: a plugged-in laptop is rarely brim full, so a floor up there rests after nearly every window. 99 is the ceiling, since the resume level must fit above it. Only rests while a charger is plugged in. |
| `--battery-resume` N        | `0`     | `BATTERY_RESUME`    | Charge that ends a rest started by `--battery-floor`, and must be above it. `0` sits 15 points above the floor.                                                                                                                                                                                                                                      |
| `--work-minutes` FLOAT      | `10.0`  | `WORK_MINUTES`      | Minutes of work between rests, for a run that is already resting. Changes nothing on its own.                                                                                                                                                                                                                                                        |
| `--max-pause-minutes` FLOAT | `15.0`  | `MAX_PAUSE_MINUTES` | Longest a single rest can last. Bounds the wait for a slow charger. Changes nothing on its own.                                                                                                                                                                                                                                                      |

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
`2024-01-01` or `2024-01-01T14:30`. Naive timestamps use local time.

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
| `--file-log-level` LEVEL    | `DEBUG` | `-`     | Same levels. `OFF` disables the file log.                          |
| `--log-folder` PATH         | `logs`  | `-`     | Folder for timestamped log files.                                  |

### Tracing model calls

`--otel` exports an OpenTelemetry span per model call: the request, how long it took, how many
tokens it cost, and whether it failed. It goes to a collector you run. It answers the questions a
log file makes you piece together by hand: which photo was slow, where a batch stalled, what a run
actually cost.

It is off by default and there is no service behind it. Nothing is exported until you pass `--otel`
and also name an endpoint. The endpoint is always yours. This is separate from
[telemetry](#telemetry), which is the project's own anonymous beacon and carries none of this.

| Flag                                   | Default | Env var                                                              | Description                                                                                                                |
| -------------------------------------- | ------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `--otel` / `--no-otel`                 | `false` | `-`                                                                  | Export a trace of every model call over OTLP/HTTP.                                                                         |
| `--otel-endpoint` URL                  | none    | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_ENDPOINT` | Where to send traces. The flag wins. The signal-specific variable beats the general one, which gets `/v1/traces` appended. |
| `--otel-content` / `--no-otel-content` | `false` | `-`                                                                  | Also export the prompt and the generated title, description, and keywords.                                                 |

Install the exporter with the optional extra, then point it somewhere:

```bash
pip install 'photo-tagger[otel]'
photo-tagger -i ~/Pictures/Trip --otel --otel-endpoint http://localhost:4318/v1/traces
```

Any OTLP/HTTP backend works: [Logfire](https://pydantic.dev/logfire) (pydantic-ai's own, which
renders these spans natively), Jaeger, Grafana Tempo, or an OpenTelemetry Collector you already run.
For a hosted backend, set its auth header the standard way, with `OTEL_EXPORTER_OTLP_HEADERS`.

The photo itself never leaves the machine. It is never exported: spans record that an image of a
given media type was sent, never its bytes, and there is no flag to change that.

The text the model generates is also kept out by default, because it describes your photos by name.
When you debug a prompt and trust the collector, add `--otel-content`. If you turn on `--otel`
without an endpoint, photo-tagger raises an error instead of a silent fallback to `localhost:4318`.
This way, a run never looks traced while nothing is listening.

photo-tagger sends the spans in batches during the run, and flushes the rest when the process exits.
Setting `OTEL_SERVICE_NAME` renames the service. Otherwise, spans arrive as `photo-tagger`, tagged
with the version that produced them. The GUI has no command line, so it reads the same three
settings (`otel`, `otel_endpoint`, and `otel_content`) from the `[log]` table of the
[configuration file](../getting-started/configuration.md). A bad endpoint there is logged, and the
window still opens.

## Display

The display group controls how a run makes itself visible: the progress bar, machine-readable
output, and the desktop alert at the end of the batch.

| Flag                                   | Default           | Env var | Description                                                                                                                                                                                                                                                                     |
| -------------------------------------- | ----------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--progress` / `--no-progress`         | progress (`true`) | `-`     | Live, rich progress bar. When stderr is not a TTY, it turns off automatically.                                                                                                                                                                                                  |
| `--json`                               | `false`           | `-`     | Emit one NDJSON line per processed photo to stdout, with these fields: `file`, `status`, `from_cache`, `retry`, `title`, `description`, `keywords`, input/output/total tokens, `seconds`, `foreign_terms`. Logs and progress stay on stderr, so stdout pipes cleanly into `jq`. |
| `--notify` / `--no-notify`             | `false`           | `-`     | When the batch ends, raise a desktop notification with the run's tally, such as `38 of 42 photos tagged, 4 failed.`.                                                                                                                                                            |
| `--notify-sound` / `--no-notify-sound` | `false`           | `-`     | Play a sound with the notification. Implies `--notify`, unless `--no-notify` is set.                                                                                                                                                                                            |

### End-of-batch notification

A long run is meant to be walked away from. When it ends, `--notify` calls you back. The alert fires
whether the batch fully succeeded or only partly did: a run that half failed is exactly the one you
want to hear about. Under `watch`, every batch gets its own alert.

It uses whatever notifier the operating system already ships, so there is nothing to install:

| Platform      | Notification              | Sound (`--notify-sound`)           |
| ------------- | ------------------------- | ---------------------------------- |
| macOS         | `osascript`               | the notification's own alert sound |
| Linux and BSD | `notify-send` (libnotify) | `canberra-gtk-play`, else `paplay` |
| Windows       | a PowerShell toast        | the system alert sound             |

Everything about it is best-effort. Nothing appears, and the run is unaffected, on a machine with no
notifier on `PATH`, in a headless session, or with notification permission denied. The failure is
recorded in the debug log only. When a sound was asked for and the platform has nothing to play it
with, the terminal bell rings instead (skipped when stderr is redirected).

## Telemetry

photo-tagger sends one anonymous beacon per run, and one on a crash. It is opt-out and carries no
photos, paths, filenames, tags, or error messages. See [Telemetry](../telemetry.md) for the exact
payload and every way to switch it off.

| Flag                             | Default     | Env var                                   | Description                                                                                                                                                                                                    |
| -------------------------------- | ----------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--telemetry` / `--no-telemetry` | on (`true`) | `PHOTO_TAGGER_TELEMETRY` / `DO_NOT_TRACK` | Send anonymous usage stats and crash reports. An opt-out anywhere wins over an opt-in. `PHOTO_TAGGER_TELEMETRY=0` turns it off, and `PHOTO_TAGGER_TELEMETRY=1` overrules `DO_NOT_TRACK` for photo-tagger only. |

## Artifacts

The artifacts group points at side files: a custom prompt, a run summary, a per-photo CSV report, a
result cache, and a lock.

| Flag                           | Default     | Env var | Description                                                                                                                                                                                                                                             |
| ------------------------------ | ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--prompt-file` PATH           | none        | `-`     | Replace the default user prompt with the contents of PATH. photo-tagger still appends existing photo metadata automatically.                                                                                                                            |
| `--summary-file` PATH          | none        | `-`     | Write a JSON run summary (success/failure counts, failed files, token usage, wall time) on completion.                                                                                                                                                  |
| `--csv-file` PATH              | none        | `-`     | Write a CSV report with one row per photo (see below). Rows stream as photos finish, so a stopped run still leaves a valid file.                                                                                                                        |
| `--cache-file` PATH            | none        | `-`     | SQLite cache of model outputs, keyed on an image-data hash that ignores metadata (so it survives `--embed-in-photo`). If nothing relevant changed, a rerun skips the model call. If the file is missing, photo-tagger creates it. It is safe to delete. |
| `--lock-file` PATH             | none        | `-`     | Acquire an exclusive file lock before running. If another photo-tagger already holds it, refuse to start. Works on Linux, macOS, and Windows.                                                                                                           |
| `--undo-log` / `--no-undo-log` | on (`true`) | `-`     | Record every file the run writes so `photo-tagger undo` can put it back.                                                                                                                                                                                |

### CSV report

`--summary-file` writes one JSON object for the whole run. `--json` streams NDJSON to stdout.
`--csv-file` writes a spreadsheet-friendly table with one row per photo: the single file that
gathers everything extracted and computed for each image:

- `filename`, `file`, `status`
- `title`, `description`, `keywords` (the keywords actually written), `hierarchical_keywords`
- `existing_keywords` (what was already on the file)
- `camera_model`, `lens_model`, `capture_date`, `gps_position`, `city`, `country` (read EXIF)
- `input_tokens`, `output_tokens`, `total_tokens`, `seconds`, `from_cache`, `retry`
- `foreign_terms`: the words in the odd script, for output that mixes writing systems (see
    [Mixed-language output](#mixed-language-output))

Multi-value cells (the keyword lists) are joined with a semicolon and a space. Rows are flushed as
each photo completes, so interrupting the run with Ctrl-C still leaves a complete, openable CSV of
the work done so far. You can use `--csv-file` and `--json` together. Both observe every photo. A
`--dry-run` still fills the report, which makes it handy for previewing a batch before writing any
metadata.

## Mixed-language output

Asked for English, a model now and then reaches for a word in its own script anyway. A title can
come back as `Decorated陶笛 with Geometric Pattern`, where `Ocarina` says the same thing in English.

Every run checks the generated text for this. It takes whatever script most of the title,
description, and keywords use as the one the model meant, and it reports anything in another script.
No language list is involved, so it works for whatever `--output-language` is set to.

A flagged photo is still tagged. Mixed script is often correct: a sign, a book cover, or a printed
character in the photo can explain it. Only the person looking at the photo can say for sure. What
the flag does is point you at it:

- a `foreign_terms` cell in the `--csv-file` report and key in the `--json` stream, naming the words
- a `files_flagged_for_review` line in the log at the end of the run
- `flagged_files` in the `--summary-file` JSON
- in the GUI, an amber note over the editable fields and a count in the closing status line

`--repair-language` goes one step further: on a flagged photo only, it asks the same model again,
with the same image, quoting its previous answer and naming the offending words. The image goes
along because the question that decides the right fix is whether the photo shows that text, which no
text-only pass can answer. That costs one extra call per affected photo, so it is off by default. A
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
ones that earn their place. The read goes through exiftool, so the application does not matter:
digiKam, darktable, Immich, PhotoPrism, Synology Photos, Piwigo and the rest all write XMP/IPTC
keywords, and only Lightroom offers a keyword-list export at all. `XMP-lr:HierarchicalSubject`
carries the hierarchy those photos really use, so the generated file keeps it.

Nothing is written to your photos or your catalog. The output is a text file to read and edit.

| Flag                   | Default    | Description                                                                               |
| ---------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| `-i`, `--input` PATH   | none       | Photos or folders to read keywords from. Repeat the flag. Honors `--ext` and `-r`.        |
| `-o`, `--output` PATH  | (required) | Where to write the vocabulary file.                                                       |
| `--from-export` PATH   | none       | Read a Lightroom keyword export (`.txt` or `.csv`) instead of, or as well as, the photos. |
| `--min-uses` N         | `2`        | The library must use a keyword at least this often to keep it.                            |
| `--max-terms` N        | `4800`     | Cap the file, dropping the least-used first. `0` means no cap.                            |
| `--allow-digits`       | `false`    | Keep keywords containing digits (dropped by default as measurements and model numbers).   |
| `--output-language`    | `English`  | Language the catalog's keywords are in. Only English plurals fold onto their singular.    |
| `--flat`               | `false`    | Write bare keywords instead of their hierarchies.                                         |
| `--report` PATH        | none       | Write a CSV of every dropped keyword, its count, and the rule that cut it.                |
| `--organize`           | `false`    | Model pass for synonyms and a hierarchy (see [below](#organizing-with-the-model)).        |
| `--organize-workers` N | `1`        | Model requests to run at once while organizing.                                           |

`--organize` also takes the provider flags, with the same meanings and the same configuration file
and environment defaults as a tagging run:

- `--provider`
- `-m/--model`
- `-u/--url`
- `-k/--api-key`

### Which source to use

Prefer the photos. An export has no usage data at all, so `--from-export` counts *occurrences in the
keyword tree* instead: a term filed under forty parents scores forty, however many photos carry it.
It is a usable proxy for a catalog that is not on this machine, not the same measure. Pass both and
the counts are added together.

### What the rules do

Applied in this order, each one reported in `--report`:

1. Shape. A keyword with digits, odd punctuation, more than three words, or over 30 characters is a
    measurement, a path, or a sentence, not a subject.
2. Rarity. `--min-uses` drops the one-offs. In a catalog an AI writes to, this is the rule that does
    the work: most keywords are used exactly once.
3. Variants. One concept keeps one spelling, the most-used one. Nothing is lost by this, since
    vocabulary matching folds case, punctuation, and plurals anyway: a photo tagged `Animals` still
    snaps onto `Animal`.
4. The cap. `--max-terms` runs last, on the terms the other rules kept, and removes the least-used
    of them.

The result is deterministic: the same library gives the same file, ties broken alphabetically.

!!! tip

    Read the file before you trust it, and tune from the report rather than by guesswork. If a keyword
    you care about was dropped, `dropped.csv` names it, its count, and the rule that dropped it. If the
    rule was `rare`, set `--min-uses` to that count or lower to keep it. The hierarchy is worth a look
    too: a catalog that a tool writes to can file `Beach` under `Sand`, and a vocabulary imposes its
    hierarchy on every photo it matches. When the source is not worth keeping, `--flat` drops the
    hierarchies.

### Organizing with the model

Counting settles which keywords are worth keeping. It cannot settle two things, and `--organize`
asks the model for exactly those two, over the keywords that already survived:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --organize --organize-workers 4
```

- Synonyms. Matching already folds case, punctuation, and plurals, so `Animal` and `Animals` are one
    keyword without any help. It cannot know that `Golden Light` is `Golden Hour`. The winner keeps
    the entry, and the others are written as `{braces}` on it. So a photo tagged with a folded
    spelling still matches. Each fold is listed in `--report` with the reason `synonym`.
- A hierarchy. The categories are chosen once, from the most-used keywords, and every chunk of the
    list is then filed against that one fixed set. Letting each chunk invent its own gives `Animal`
    in one chunk and `Animals` in the next: exactly the sprawl this command exists to end.

Five properties keep the pass from making the file worse:

- It never decides what to keep. That is already settled, by counting, before the model sees
    anything.
- It never invents a keyword. Every string the model returns is matched back to a keyword that was
    sent, loosely enough to survive a retyped capital. Everything else is discarded and counted.
- A failure costs nothing but the organizing. A chunk that errors, or a keyword the model forgets,
    keeps the shape the deterministic pass gave it.
- A group cannot swallow a category. At most three synonyms are accepted per keyword. A group
    claiming more is refused whole. A model listing five is not naming synonyms. It is emptying a
    category into one keyword, for example `People` taking `Person`, `Human`, `Woman`, and `Man`.
    Each one it takes is a keyword your catalog loses.
- A degenerate category list is refused. Asked for six to twenty top-level categories, a weak model
    sometimes echoes the keyword list back. Keeping the first twenty of that looks like an answer
    and behaves like noise. So the hierarchy is skipped, and only synonyms are folded.

### Choosing a model for it

This pass is text only: it reads a list of words, not an image. Your tagging model is a
vision-language model, and its vision half buys nothing here, so it is worth pointing `--model` at
something else:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --organize --model openai/gpt-oss-20b
```

What matters is instruction-following and reliable structured output, not size. Three failure modes
tell you a model is the wrong choice, and all three are visible in the run log:

- `vocabulary_organize_request_failed` on every chunk, with a token-limit message: a reasoning model
    spending its whole budget thinking before it answers. The budget is already generous. A model
    that still cannot finish inside it is not usable here.
- Chunks that take minutes each. Grouping words is recall, not deduction, but a reasoning model left
    to itself will spend thousands of tokens deliberating over a list of sixty of them. Every
    request therefore asks for no reasoning (`reasoning_effort: "none"`), which servers that do not
    know the setting ignore. On one local 31B model, that setting was the difference between 13
    minutes for sixteen keywords and 12 seconds. If chunks are still slow, the server is probably
    not honoring it. Check whether your provider exposes its own switch.
- `vocabulary_synonym_group_refused` many times over: the model is folding categories into keywords,
    and the guardrail is the only thing between it and your catalog.

!!! tip

    Time one chunk before committing a whole library to it. Add `--max-terms 60` so exactly one chunk is
    sent, and watch the clock between `vocabulary_organize_started` and `vocabulary_organized`. Multiply
    by the chunks your real list needs, then divide by `--organize-workers`.

!!! warning "Categories are new keywords"

    A category the model names becomes a parent in the file. photo-tagger writes it to your photos as a
    hierarchical keyword, whether or not your library ever used that word. The file's header lists them
    for exactly this reason. If you do not want any, use `--flat` or drop the parents by hand.

The list is sent in chunks of 60 keywords: roughly one request per 60 keywords plus one for the
categories, so about 80 requests for a 4,800-keyword file. `--organize-workers` runs several at
once. Sampling is fixed at temperature 0, and chunks are reassembled by position. So the same list
organizes the same way, whatever order the replies arrive in. A model is not a pure function,
though: treat the output as a proposal to read, which is what the whole file is anyway.

## Skipping and resuming

Three flags cooperate to skip work you already did and to resume a run that stopped partway through:

- `--skip-from PATH` reads a list of filenames (one per line, `#` comments allowed) and drops any
    matching files from the batch before processing starts.
- `--append-to-skip-file PATH` appends each successfully tagged filename to PATH as the run
    continues. If the file does not exist, it creates one.
- `--skip-tagged` inspects each file's existing metadata and skips anything that already has
    keywords, a title, or a description (in the image or its XMP sidecar). When you want the skip
    decision to come from the files themselves, rather than from a list, use it instead.

For resume-on-failure, pass the same path to both `--skip-from` and `--append-to-skip-file`. The
first run appends every success to the file. If the run dies partway through, the next run with the
same arguments reads that file back through `--skip-from`. It continues from where it left off,
without re-tagging the photos that already succeeded.

!!! tip

    Combine the skip file with `--cache-file` for an even cheaper resume. The skip file removes finished
    photos from the batch entirely. The cache then avoids re-calling the model for any photo that still
    slips back in unchanged.

See [Recipes](recipes.md) for runnable resume and skip examples.

## Watching a folder

`photo-tagger watch` is the import-time workflow: point it at the folder your card reader, tethered
capture, or sync client fills, and leave it running.

```bash
photo-tagger watch -i ~/Pictures/Inbox --recursive --skip-tagged
```

Photos already in the folder are tagged first, then each new one as it lands. Every flag the tagging
command takes works here too, and applies to each batch. A single agent, cache, and CSV/NDJSON file
are shared by the whole session. Each batch records its own undo journal, so
[`photo-tagger undo`](#undoing-a-run) puts back the last import rather than everything since the
watch started. Stop it with Ctrl-C.

| Flag             | Default | Description                                            |
| ---------------- | ------- | ------------------------------------------------------ |
| `--interval` SEC | `5.0`   | Seconds between folder scans.                          |
| `--settle` SEC   | `2.0`   | Seconds a file must sit unchanged before it is tagged. |

Two behaviors worth knowing:

- A file is only tagged once it stops changing. It must be unchanged across two scans *and* its last
    modification must be at least `--settle` seconds old, so a photo still being copied is left
    alone until the copy finishes. This also means the first batch appears one `--interval` after
    the watch starts, not instantly.
- A failing batch does not stop the watch. The failure is logged and the next photo to land gets its
    turn. Scanning is a plain directory listing rather than a filesystem-event API, so it behaves
    the same on every platform and over network shares.

## Undoing a run

Every run records the files it writes. So a batch tagged with the wrong prompt, the wrong model, or
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

Sidecars the run created are deleted. Files it overwrote are restored from ExifTool's `*_original`
backup. The journals are small JSON-lines files under the state directory:
`$XDG_STATE_HOME/photo-tagger/runs`, or `~/.local/state/photo-tagger/runs`. photo-tagger prunes them
to the 50 most recent runs and 90 days.

| Flag         | Default    | Description                                                  |
| ------------ | ---------- | ------------------------------------------------------------ |
| `--run` PATH | newest run | Undo this journal instead of the most recent one.            |
| `--list`     | `false`    | List the recorded runs (name, file count, path) and exit.    |
| `--dry-run`  | `false`    | Show what an undo would put back, without changing anything. |
| `--force`    | `false`    | Also revert files that changed after the run wrote them.     |

Three things are deliberately left alone:

- Files changed since the run. A different size or mtime means someone edited the file afterwards,
    so undo reports it and moves on. `--force` overrides this.
- Writes made with `--no-backup-xmp`. There is no copy of the previous contents, so an overwritten
    file cannot be restored. Newly created sidecars are still deleted.
- Runs from the desktop GUI, which does not record a journal.

If there is nothing to undo, or if any entry was left alone, `photo-tagger undo` exits with code 1.
If every recorded write was put back, it exits with code 0. Dry runs are not written to a journal
(they change nothing), and `--no-undo-log` turns recording off for a run.
