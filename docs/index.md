______________________________________________________________________

## icon: lucide/house

# photo-tagger

photo-tagger is a Python command-line tool that sends each photo to a local vision-language model
and writes Lightroom-compatible metadata: a title, a short description, and hierarchical keywords.
By default it leaves your originals untouched, writing an XMP sidecar next to each image instead of
modifying the file.

## Highlights

- Works with RAW and standard formats (CR3, CR2, NEF, DNG, JPG, PNG, and more). RAW files are
    decoded with rawpy/libraw and the rest with Pillow.
- Generates a `title`, a short `description`, and hierarchical keywords in Lightroom's root-to-leaf
    pipe form.
- Merges new keywords with existing ones by default, or replaces them with `--overwrite-keywords`.
- Talks to local Ollama or LM Studio servers over an OpenAI-compatible API.
- Sends a compact, resized JPEG to the model to save tokens, with configurable dimensions and
    quality.
- Optional SQLite cache so reruns skip the model call when nothing relevant changed.
- Processes photos concurrently with a thread pool when the model server can keep up.
- Skip and resume support: skip already-tagged files, skip names from a list, and append successes
    to a skip file as the run progresses.
- Optional NDJSON output on stdout, one line per photo, that pipes cleanly into `jq`.
- Timestamped, rotating log files plus a live progress bar on a TTY.
- Configurable through CLI flags, environment variables, and a TOML config file.

## Quick start

The recommended way to install photo-tagger is with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install photo-tagger
```

Once installed, tag a folder of CR3 and JPG files, recursing into subdirectories:

```bash
photo-tagger -i ./photos --ext cr3,jpg -r
```

!!! tip

    photo-tagger needs [ExifTool](https://exiftool.org/) on your `PATH` and a running Ollama or LM
    Studio server exposing a vision-language model. See [Installation](getting-started/installation.md)
    for the full setup, including `libraw` for RAW decoding.

## Where to go next

- [Getting started](getting-started/index.md): install photo-tagger and learn how to configure it.
- [Usage](usage/index.md): run the tool, with a full CLI reference and practical recipes.
- [Architecture](architecture/index.md): how the processing pipeline, AI providers, metadata, and
    caching fit together.
- [Development](development/index.md): set up the project, run tests, and keep code quality high.
- [License](license.md): photo-tagger is released under the MIT License.
