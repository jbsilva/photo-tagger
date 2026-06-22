---
icon: lucide/terminal
---

# Usage

photo-tagger is a single command-line tool. You point it at photos, it sends each one to a local
vision-language model, and it writes Lightroom-compatible keywords, a title, and a description back
to your images.

## Basic invocation

Every run is one command with one or more `-i`/`--input` paths plus options:

```bash
photo-tagger -i PATH [-i PATH ...] [options]
```

`-i`/`--input` is repeatable and accepts both files and directories. Add `-r`/`--recursive` to
descend into subdirectories, and `--ext` to control which extensions are picked up when scanning
directories (default `cr3,jpg`, case-insensitive). The tool reads RAW and standard formats (CR3,
CR2, NEF, DNG, JPG, PNG, and more).

Before your first run, make sure ExifTool is on your `PATH` and a model server is running. See
[Installation](../getting-started/installation.md) for the full prerequisites, and
[Configuration](../getting-started/configuration.md) for env vars and the TOML config file.

## What a run produces

By default photo-tagger leaves your originals untouched: for each image it writes an XMP sidecar
(for example `IMG_1234.xmp`) next to the file. Pass `--embed-in-photo` to write the metadata into
the image file itself instead.

Either way the metadata is written through ExifTool. The fields generated are:

| Field       | Written when                    | Notes                                     |
| ----------- | ------------------------------- | ----------------------------------------- |
| Keywords    | always                          | Flat keywords plus a Lightroom hierarchy. |
| Title       | unless `--no-write-title`       | Short title for the photo.                |
| Description | unless `--no-write-description` | One-line caption.                         |

!!! note

    Existing metadata is merged, not clobbered. New keywords are combined with the ones already on the
    file, preserving Lightroom hierarchies and deduplicating case-insensitively. Use
    `--overwrite-keywords` if you would rather replace existing keywords instead of merging.

## Logs, progress, and NDJSON

photo-tagger keeps machine output and human output on separate streams so a run stays scriptable:

- Logs and the live progress bar go to **stderr**. The progress bar is shown on a TTY and is
    auto-disabled when stderr is redirected; disable it explicitly with `--no-progress`.
- With `--json`, one NDJSON line per processed photo goes to **stdout** (file, status, from_cache,
    retry, title, description, keywords, token counts, seconds). Because logs stay on stderr, stdout
    pipes cleanly into tools like `jq`.

## End-to-end example

Tag every CR3 and JPG under a folder, recursing into subfolders, against an Ollama server, and write
XMP sidecars (the default):

```bash
photo-tagger \
  --provider ollama \
  -i ~/Pictures/2024-trip \
  --recursive
```

Each photo gets an `.xmp` sidecar with the generated keywords, title, and description, ready to
import into Lightroom. Logs and a progress bar appear on stderr; nothing is written to stdout
because `--json` was not passed.

!!! tip

    Add `--dry-run` to run the model and log the proposed metadata without writing anything. It is the
    safest way to preview results before committing to a real run.

!!! tip "Prefer a window?"

    photo-tagger ships an optional [desktop GUI](gui.md) (`photo-tagger gui`) for a review-before-write
    workflow: drag in photos, generate proposals, then review and edit each title, description, and
    keyword set before saving. Install it with the `gui` extra.

## Next steps

- [CLI reference](cli-reference.md): every flag, its default, and the matching environment variable.
- [Desktop GUI](gui.md): the optional point-and-click frontend.
- [Recipes](recipes.md): ready-made command lines for common workflows.
