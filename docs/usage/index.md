---
icon: lucide/terminal
---

# Usage

photo-tagger is a single command-line tool. You point it at photos. It sends each one to a
vision-language model, which can be a local model, a self-hosted one, or any OpenAI-compatible
service, free or paid. It writes Lightroom-compatible keywords, a title, and a description back to
your images.

## Basic invocation

Every run is one command with one or more `-i`/`--input` paths plus options:

```bash
photo-tagger -i PATH [-i PATH ...] [options]
```

`-i`/`--input` is repeatable and accepts both files and directories. Add `-r`/`--recursive` to
descend into subdirectories. When scanning directories, `--ext` controls which extensions are picked
up (default `cr3,jpg`, case-insensitive). The tool reads RAW and standard formats (CR3, CR2, NEF,
DNG, JPG, PNG, and more).

Before your first run, make sure that photo-tagger can find ExifTool (on your `PATH`, or through the
`exiftool_path` key) and that a model server is running. See
[Installation](../getting-started/installation.md) for the full prerequisites, and
[Configuration](../getting-started/configuration.md) for env vars and the TOML configuration file.

## What a run produces

By default photo-tagger leaves your originals untouched: for each image it writes an XMP sidecar
(for example `IMG_1234.xmp`) next to the file. Pass one of these flags to change that:

- `--embed-in-photo` writes the metadata into the image file itself.
- `--sidecar-mode raw` splits by file type: a sidecar for each RAW file, embedded metadata for
    everything else.
- `--sidecar-mode both` writes each photo and a sidecar beside it.
- `--sidecar-override dng=photo` sends one file type elsewhere, whatever the mode. Lightroom Classic
    keeps the metadata of a DNG inside the DNG, so this is the setting for a Lightroom library. See
    [Tag DNG files for Lightroom Classic](recipes.md#tag-dng-files-for-lightroom-classic).

Either way the metadata is written through ExifTool. The fields generated are:

| Field       | Written when                    | Notes                                     |
| ----------- | ------------------------------- | ----------------------------------------- |
| Keywords    | unless `--no-write-keywords`    | Flat keywords plus a Lightroom hierarchy. |
| Title       | unless `--no-write-title`       | Short title for the photo.                |
| Description | unless `--no-write-description` | One-line caption.                         |

!!! note

    Only keywords are merged. New keywords are combined with the ones already on the file. This keeps
    Lightroom hierarchies and removes duplicates without case sensitivity. To replace the existing
    keywords instead, use `--overwrite-keywords`.

    A generated title or description replaces the one that the file already has. To keep the existing
    one, use `--preserve-title` or `--preserve-description`.

## Logs, progress, and NDJSON

photo-tagger keeps machine output and human output on separate streams so a run stays scriptable:

- Logs and the live progress bar go to stderr. The progress bar is shown on a TTY. When stderr is
    redirected, it is auto-disabled. Disable it explicitly with `--no-progress`.
- With `--json`, one NDJSON line per processed photo goes to stdout (file, status, from_cache,
    retry, title, description, keywords, token counts, seconds, foreign_terms). Because logs stay on
    stderr, stdout pipes cleanly into tools like `jq`.

## End-to-end example

This command tags every CR3 and JPG under a folder. It recurses into subfolders, runs against an
Ollama server, and writes XMP sidecars (the default):

```bash
photo-tagger \
  --provider ollama \
  -i ~/Pictures/2024-trip \
  --recursive
```

Each photo gets an `.xmp` sidecar with the generated keywords, title, and description, ready to
import into Lightroom. Logs and a progress bar appear on stderr. Nothing is written to stdout
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
