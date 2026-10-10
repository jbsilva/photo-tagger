---
icon: lucide/house
---

# photo-tagger

photo-tagger sends each photo to a vision-language model (an AI model that reads images) and writes
Lightroom-compatible metadata: a title, a short description, and hierarchical keywords. The model
can run on your own machine or server, or come from any OpenAI-compatible service, free or paid. It
comes as a command-line tool and an optional desktop app. The two share one pipeline and one
configuration file. By default, photo-tagger does not change your original files. It writes an XMP
sidecar (a separate metadata file) next to each image instead.

## Highlights

- Works with RAW and standard formats (CR3, CR2, NEF, DNG, JPG, PNG, and more). It decodes RAW files
    with rawpy/libraw and all other files with Pillow.
- Generates a `title`, a short `description`, and hierarchical keywords in the root-to-leaf pipe
    form of Lightroom. It writes them in any language that you ask for with `--output-language`.
- Merges new keywords with existing ones by default, or replaces them with `--overwrite-keywords`.
- Snaps generated keywords onto your own Lightroom keyword list with `--vocabulary`.
- Harmonizes each shoot with `--session-gap`, so every frame of one subject gets the same term.
- Talks to local Ollama, LM Studio, or llama.cpp servers, or any hosted OpenAI-compatible API.
- Has a `doctor` command that tests ExifTool and the model provider before a run.
- Has a `watch` command that tags photos as they arrive in an import folder.
- Has an `undo` command that puts back what a run wrote, down to the last sidecar.
- Has an optional PySide6 desktop GUI (`photo-tagger gui`) for a point-and-click workflow. The GUI
    has the same vocabulary, shoot-harmonization, watch, and undo features. Its interface is in
    English and Brazilian Portuguese.
- Sends a compact, resized JPEG to the model to save tokens. You can set the dimensions and the
    quality of this JPEG.
- Has an optional SQLite cache. If nothing relevant changed, a rerun skips the model call.
- Processes several photos at the same time with a thread pool, for model servers that can keep up.
- Supports skip and resume. It can skip files that already have tags, skip the names in a list, and
    add each successful file to a skip file during the run.
- Writes optional NDJSON output on stdout (one line per photo) that you can pipe into `jq`. It can
    also write a JSON run summary and a CSV report with one row per photo.
- Writes timestamped, rotating log files and shows a live progress bar in an interactive terminal.
- Takes its configuration from CLI flags, environment variables, and a TOML configuration file.

## Quick start

The recommended way to install photo-tagger is with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install photo-tagger
```

After you install it, tag a folder of CR3 and JPG files. The `-r` flag includes the subdirectories:

```bash
photo-tagger -i ./photos --ext cr3,jpg -r
```

!!! tip

    photo-tagger needs [ExifTool](https://exiftool.org/), on your `PATH` or at the path that you
    configure. It also needs a vision-language model: an Ollama or LM Studio server, or any
    OpenAI-compatible service. For the full setup (with `libraw` for RAW decoding on an Intel Mac), see
    [Installation](getting-started/installation.md).

## Where to go next

- [Getting started](getting-started/index.md): install photo-tagger and learn how to configure it.
- [Usage](usage/index.md): run the tool, with a full CLI reference and practical recipes.
- [Architecture](architecture/index.md): how the processing pipeline, AI providers, metadata, and
    caching fit together.
- [Development](development/index.md): set up the project, run tests, and keep code quality high.
- [License](license.md): photo-tagger is available under the MIT License.
