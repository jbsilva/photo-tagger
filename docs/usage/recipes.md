---
icon: lucide/book-open
---

# Recipes

Short, task-oriented examples you can copy and adapt. Each one builds on the flags documented in the
[CLI reference](cli-reference.md). For the full option list and defaults, see that page.

## Process a folder of RAW + JPEG recursively

Scan a directory tree for both Canon RAW and JPEG files and tag everything found.

```bash
photo-tagger \
  --input ~/Pictures/2026-trip \
  --recursive \
  --extensions cr3,jpg
```

By default, `--extensions` is `cr3,jpg`. If you want a different set, add `--extensions`. Matching
is case-insensitive.

## Tag a few explicit files and overwrite keywords

Point at individual files (repeat `-i`) and replace any keywords already present instead of merging.

```bash
photo-tagger \
  -i ~/Pictures/IMG_0001.cr3 \
  -i ~/Pictures/IMG_0002.jpg \
  --overwrite-keywords
```

By default photo-tagger preserves existing keywords. `--overwrite-keywords` discards them and keeps
only what the model proposes.

## Embed metadata into JPEGs

Write the title, description, and keywords into the image files themselves rather than into XMP
sidecars.

```bash
photo-tagger \
  -i ~/Pictures/export \
  --embed-in-photo
```

!!! warning

    `--embed-in-photo` modifies your originals. ExifTool keeps a `*_original` backup by default. If you
    are sure you do not need it, add `--no-backup-xmp`.

## Tag a folder of RAW + JPEG pairs

A camera set to RAW+JPEG leaves two files per shot. `--sidecar-mode raw` handles both in one pass:
the CR3 (or DNG, NEF, ARW) keeps its bytes and gets a `.xmp` beside it. The JPEG carries its own
metadata, so it travels intact. If you use Lightroom Classic, also read the next recipe: it keeps a
DNG's metadata inside the DNG.

```bash
photo-tagger \
  -i ~/Pictures/shoot \
  --ext cr3,jpg \
  --sidecar-mode raw
```

## Tag DNG files for Lightroom Classic

Lightroom Classic keeps the metadata of a DNG inside the DNG file, not in a `.xmp` sidecar.
photo-tagger counts DNG as a RAW format, so each sidecar mode except `none` gives a DNG a sidecar.
To write into each DNG instead, add an override for the DNG file type:

```bash
photo-tagger \
  -i ~/Pictures/shoot \
  --ext cr3,dng,jpg \
  --sidecar-mode raw \
  --sidecar-override dng=photo
```

This matches what Lightroom Classic does. A CR3 keeps its bytes and gets a `.xmp` beside it. A DNG
and a JPEG carry their own metadata. To make this your default, put it in the
[configuration file](../getting-started/configuration.md). The GUI then follows it too:

```toml
[output]
sidecar_mode = "raw"               # A sidecar for CR3, NEF, ARW. Embedded for JPEG and TIFF.
sidecar_overrides = ["dng=photo"]  # DNG counts as RAW above. This writes into each DNG instead.
```

An override has the form `EXT=TARGET`, where TARGET is `sidecar`, `photo`, or `both`. It applies to
one file extension and wins over `--sidecar-mode`. Without an override, photo-tagger still gives a
DNG a sidecar.

### Two overrides in one run

Repeat the flag once for each file type. This run keeps the default sidecar for most files. It
writes into each DNG for Lightroom Classic. It writes each TIFF twice: into the file itself, and
into a sidecar for other tools. The TIFF then carries its metadata to wherever you send it.

```bash
photo-tagger \
  -i ~/Pictures/shoot \
  --ext cr3,dng,tif,jpg \
  --sidecar-mode all \
  --sidecar-override dng=photo \
  --sidecar-override tif=both
```

In the configuration file, the same two overrides are two entries in one list:

```toml
[output]
sidecar_mode = "all"
sidecar_overrides = ["dng=photo", "tif=both"]
```

Each photo then gets these files:

```text
photos/
  duck.cr3        # the mode: original, never modified
  duck.xmp        # its sidecar
  heron.dng       # dng=photo: metadata written into the file itself
  swan.tif        # tif=both: metadata written into the file itself
  swan.xmp        # and the same metadata beside it
  owl.jpg         # the mode: original, never modified
  owl.xmp         # its sidecar
```

An override matches one extension exactly, and `tif=both` does not cover `.tiff` files. If you have
both, add `--sidecar-override tiff=both`. On the command line, the flags replace the whole list from
the configuration file. To keep an entry from the file, repeat it on the command line.

### Why a sidecar beside a DNG loses metadata

Lightroom Classic picks the place for XMP by file type. For a proprietary RAW file (CR2, CR3, NEF,
ARW), it writes a `.xmp` sidecar, because it never changes those files. For DNG, JPEG, TIFF, and
PSD, it writes the XMP inside the file. Lightroom Classic has no option to change this. See
[Adobe: metadata basics](https://helpx.adobe.com/lightroom-classic/help/metadata-basics-actions.html).

Lightroom Classic shows the metadata from its catalog, not the metadata from the newest file. When a
file changes outside Lightroom, it marks the thumbnail with a "metadata was changed externally"
badge. You then choose to read the file or to overwrite it. See
[Adobe: advanced metadata actions](https://helpx.adobe.com/lightroom-classic/help/advanced-metadata-actions.html).

When Lightroom Classic reads metadata from a DNG (on import, or with _Metadata > Read Metadata from
Files_), it also reads a sidecar beside it. If the sidecar is older than the DNG, Lightroom ignores
the sidecar. See the
[Adobe community thread](https://community.adobe.com/t5/lightroom-classic-discussions/p-lrc-reads-metadata-for-a-dng-from-an-associated-sidecar/m-p/13204500),
where Thomas Knoll of Adobe replies. This gives a sequence that loses the keywords:

1. photo-tagger writes `duck.xmp` beside `duck.dng`. The sidecar is now the newer file, but the
    catalog does not change.
2. You edit the photo in Lightroom Classic. With _Automatically write changes into XMP_ on,
    Lightroom writes the catalog metadata into `duck.dng`. The DNG is now the newer file.
3. Later, Lightroom Classic reads the metadata from the file. It ignores the older sidecar, so the
    keywords from photo-tagger never reach the catalog.

With `dng=photo`, the keywords are in the DNG itself, so no date comparison applies. To bring them
into the catalog, select the photos and choose _Metadata > Read Metadata from Files_.

!!! warning

    Save your catalog edits to the files (_Metadata > Save Metadata to Files_) before you run
    photo-tagger. Otherwise, _Read Metadata from Files_ replaces the catalog edits that are not in the
    file yet.

Lightroom Classic 15.0 (October 2025) and later also write `.acr` files. They hold large edit data,
such as AI masks, noise reduction, and Distraction Removal, so that the `.xmp` stays small. See
[Adobe: XMP and ACR files](https://helpx.adobe.com/lightroom-classic/help/create-xmp-acr-files.html).
Adobe does not say that they hold keywords or captions, and photo-tagger does not read or write
them. They do not change the choice above.

### Move old sidecars into their DNG files

If an earlier run gave your DNG files sidecars, copy each sidecar into its DNG with ExifTool. `%d%f`
is the directory and name of each DNG, so every DNG reads the `.xmp` with its own name:

```bash
exiftool -tagsFromFile %d%f.xmp -all:all -ext dng -r ~/Pictures/shoot
```

ExifTool keeps a `*_original` backup of each DNG. Make sure that the DNG files carry the keywords
(`photo-tagger show -i photo.dng`), then delete the old sidecars. photo-tagger does not delete them.
While a sidecar is there, photo-tagger reads it first, so its old values hide the new ones in the
DNG.

## Point at a remote Ollama host with a custom model

Use an Ollama server on another machine and a specific vision-language model.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --provider ollama \
  --url http://gpu-box.local:11434/v1 \
  --model qwen3-vl:32b
```

!!! tip

    Prefer environment variables for secrets. Set `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` instead of
    passing `--url` and `--api-key` on the command line.

## Dry-run preview

See the metadata the model proposes, without changing any files.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --dry-run
```

`--dry-run` still runs the model (and uses any cache), but writes neither sidecars nor embedded
metadata.

## Process concurrently and write a run summary

Run several photos at once with a thread pool. When the run finishes, save a JSON summary.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --workers 4 \
  --summary-file ~/Pictures/run-summary.json
```

When the server can serve requests in parallel, raising `--workers` helps most. The model server is
usually the bottleneck. The summary file records success and failure counts, failed files, token
usage, and wall time.

## Export a per-photo CSV report

Write one CSV row per photo with everything extracted and computed: the generated title,
description, and keywords, the metadata already on the file, the camera/location EXIF, and per-photo
token usage and timing.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --csv-file ~/Pictures/report.csv
```

Rows stream as each photo finishes, so a run stopped with Ctrl-C still leaves a valid file. Pair it
with `--dry-run` to preview a batch as a spreadsheet without writing any metadata. Pair it with
`--json` instead to get both the CSV and the NDJSON stream from one run.

## Custom prompt with a keyword cap

Replace the default user prompt and limit how many AI keywords are kept per photo.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --prompt-file ~/prompts/studio.txt \
  --max-keywords 15
```

Existing photo metadata (location, GPS, camera EXIF) is still appended to your custom prompt
automatically. `--max-keywords` caps the AI keywords before they are merged with existing ones.

## Keep keywords inside your Lightroom catalog

Export your keyword list from Lightroom (_Metadata > Export Keywords_), and point the run at it.
Generated keywords then use the names and hierarchy you already curate, instead of new
near-duplicates.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --vocabulary ~/lightroom-keywords.txt \
  --summary-file ~/Pictures/inbox/run.json
```

`ospreys`, `Sea Hawk`, and `Osprey<Raptor<Wildlife` all land as your `Animal|Bird|Osprey`. Add
`--vocabulary-strict` to drop anything the list does not cover. The summary file's
`vocabulary_dropped` then names each rejected term and how often it came up. When you decide what to
add to the catalog, this is the list to use.

## Make a shoot's keywords agree with themselves

Group the batch into shoots and let each one settle on one spelling and one hierarchy per subject.

```bash
photo-tagger \
  -i ~/Pictures/Trip \
  --recursive \
  --session-gap 60
```

Photos taken less than an hour apart are one session. Each session is analyzed in full. Its own
output becomes its vocabulary (majority spelling, majority hierarchy), and only then is anything
written. This way, forty frames of the same bird cannot land in the catalog as `Osprey`, `Ospreys`,
`Bird|Osprey`, and `Wildlife|Raptor|Osprey`. Photos are processed in capture order, and the result
is the same at any `--workers` setting.

## Cache model outputs across reruns

Store model outputs in a SQLite cache. Reruns then skip the model call for anything unchanged.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --cache-file ~/.cache/photo-tagger/cache.sqlite
```

The cache key combines a hash of the image data (ExifTool's `ImageDataHash`, which ignores metadata)
with the model, the prompt, the output language, and the sampling and image settings. Changing the
model or the temperature starts fresh entries. When the first pass embeds tags into the photo with
`--embed-in-photo`, a rerun still hits the cache, because the hash ignores metadata. If the file is
missing, photo-tagger creates it. It is safe to delete at any time. See
[Caching and locking](../architecture/caching.md#cache-key) for what the key includes and leaves
out.

## Tag photos as they are imported

Leave a watcher on the folder your card reader or sync client fills.

```bash
photo-tagger watch \
  -i ~/Pictures/Inbox \
  --recursive \
  --skip-tagged \
  --cache-file ~/.cache/photo-tagger/cache.sqlite
```

Photos already in the folder are tagged first, then each new one as it lands. A file is left alone
until it stops changing, so a photo still being copied is never tagged half-written. Tune the scan
frequency with `--interval` and the stability window with `--settle`. Ctrl-C stops the watch.

## Undo a run that went wrong

Revert everything the last run wrote: sidecars it created are deleted, files it overwrote are
restored from ExifTool's backup.

```bash
photo-tagger undo --dry-run   # see the plan first
photo-tagger undo             # carry it out
```

Recording happens automatically on every run that writes (not on `--dry-run`, and not with
`--no-undo-log`), so undo works even without advance planning. `photo-tagger undo --list` shows the
recorded runs and `--run PATH` picks one other than the newest. Files you edited after the run are
left alone unless you add `--force`.

## Resume a killed run

Track successes in a skip file, then resume from it after an interruption. Use the same path for
both flags so the next run skips what already succeeded.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --skip-from ~/Pictures/done.txt \
  --append-to-skip-file ~/Pictures/done.txt
```

`--append-to-skip-file` adds each successfully tagged filename as the run progresses, and
`--skip-from` drops those filenames on the next run. The file uses one filename per line. Lines
starting with `#` are comments.

## Tag a mixed folder, skipping already-tagged files

In a folder where some images are already tagged, process only the ones that still need metadata.

```bash
photo-tagger \
  -i ~/Pictures/library \
  --recursive \
  --skip-tagged
```

`--skip-tagged` drops files that already have keywords, a title, or a description in the image or
its XMP sidecar.

## Filter by date and stream NDJSON into jq

Tag only files modified within a date range and pipe one NDJSON line per photo into `jq`.

```bash
photo-tagger \
  -i ~/Pictures/library \
  --recursive \
  --newer-than 2026-01-01 \
  --older-than 2026-06-01 \
  --json \
| jq -r 'select(.status == "ok") | "\(.file)\t\(.title)"'
```

`--newer-than` drops files whose mtime is on or before the timestamp, and `--older-than` drops files
on or after it. Logs and the progress bar stay on stderr, so stdout pipes cleanly into `jq`.

## Refuse concurrent runs with a lock file

Guard a scheduled or shared job so a second photo-tagger cannot start while one is already running.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --lock-file ~/Pictures/.photo-tagger.lock
```

The lock is acquired before processing starts. If another process already holds it, photo-tagger
refuses to start. This works on Linux, macOS, and Windows.
