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

The default `--extensions` is `cr3,jpg`, so `--extensions` is only needed when you want a different
set. Matching is case-insensitive.

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

    `--embed-in-photo` modifies your originals. ExifTool keeps a `*_original` backup by default; add
    `--no-backup-xmp` only if you are sure you do not need it.

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

See the metadata the model would write without changing any files.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --dry-run
```

`--dry-run` still runs the model (and uses any cache), but writes neither sidecars nor embedded
metadata.

## Process concurrently and write a run summary

Run several photos at once with a thread pool and save a JSON summary when the run finishes.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --workers 4 \
  --summary-file ~/Pictures/run-summary.json
```

The model server is usually the bottleneck, so raising `--workers` helps most when the server can
serve requests in parallel. The summary file records success and failure counts, failed files, token
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
with `--dry-run` to preview a batch as a spreadsheet without writing any metadata, or with `--json`
to get both the CSV and the NDJSON stream from one run.

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

## Cache model outputs across reruns

Store model outputs in a SQLite cache so reruns skip the model call when nothing relevant changed.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --cache-file ~/.cache/photo-tagger/cache.sqlite
```

The cache key combines a hash of the image data (ExifTool's `ImageDataHash`, which ignores metadata)
with the model name and sampling settings, so changing the model or temperature invalidates stale
entries. Because the hash ignores metadata, a rerun still hits the cache even when the first pass
embedded tags into the photo with `--embed-in-photo`. The file is created if missing and safe to
delete.

## Resume a killed run

Track successes in a skip file, then resume from it after an interruption. Use the **same** path for
both flags so the next run skips what already succeeded.

```bash
photo-tagger \
  -i ~/Pictures/inbox \
  --recursive \
  --skip-from ~/Pictures/done.txt \
  --append-to-skip-file ~/Pictures/done.txt
```

`--append-to-skip-file` adds each successfully tagged filename as the run progresses, and
`--skip-from` drops those filenames on the next run. The file uses one filename per line; lines
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

`--newer-than` drops files whose mtime is on or before the timestamp; `--older-than` drops files on
or after it. Logs and the progress bar stay on stderr, so stdout pipes cleanly into `jq`.

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
