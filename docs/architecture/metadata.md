---
icon: lucide/tags
---

# Metadata and keywords

photo-tagger reads whatever metadata a photo already carries and feeds the useful parts into the
model prompt. It writes the result back as Lightroom-compatible tags. Two modules do this work:
[`metadata.py`](https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/metadata.py)
talks to ExifTool, and
[`keywords.py`](https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/keywords.py)
parses and merges hierarchical keywords. This page explains the round trip from read to write.

## Reading existing metadata

Before the model ever sees a photo, `read_image_context()` collects every read-only tag the pipeline
needs in a single `exiftool` call. Issuing one batched read instead of separate reads for keywords,
location, GPS, and camera EXIF keeps the per-photo IPC cost to one round trip. It returns an
`ImageContext`:

```python
@dataclass(slots=True, frozen=True)
class ImageContext:
    existing_keywords: KeywordSet  # typed subject / hierarchical / weighted views
    existing_title: str | None  # what --preserve-title would keep
    existing_description: str | None  # what --preserve-description would keep
    location_tags: dict[str, str]  # city/country from XMP-photoshop and IPTC
    gps_position: str | None  # Composite:GPSPosition, if present
    camera_info: dict[str, str]  # EXIF Model, LensModel, DateTimeOriginal
```

`KeywordSet` (in [`models.py`][models]) is a small dataclass with `subject`, `hierarchical`, and
`weighted` list fields. It replaced a bare `dict[str, list[str]]` keyed by those strings, so a
misspelled key is now a type error instead of a silently empty list.

The read targets both the image file and any adjacent `.xmp` sidecar, so metadata that lives only in
the sidecar is still picked up. Keywords from both are pooled. For the single-valued title and
description, the sidecar's answer wins (`_sidecar_first`). The sidecar is the later copy: it is the
one every XMP-aware catalog reads, and the only thing a sidecar-mode run writes. That is why the old
order, with the image first, was wrong. A camera's placeholder then hid the caption that
photo-tagger wrote into the sidecar during an earlier run.

`build_contextual_prompt()` turns that context into a short "Existing Metadata" block appended to
the user prompt. The model gets the first few existing keywords, a `City, Country` location hint,
the GPS position, and the camera, lens, and capture date. Camera details are corroborative only: the
prompt instructs the model to use them to disambiguate what is visible, never to assert content the
image does not show.

```mermaid
flowchart LR
    A[Photo + sidecar] -->|read_image_context| B[ImageContext]
    B -->|build_contextual_prompt| C[Prompt with Existing Metadata]
    C --> D[Vision-language model]
    D --> E[GeneratedMetadata]
    E -->|merge_keywords| F[Merged keyword buckets]
    F -->|write_metadata| G[XMP sidecar or embedded tags]
```

## Hierarchical keywords

The model returns taxonomy chains in a dedicated `hierarchies` field, leaf-first and joined with
`<`, for example `Duck<Bird<Animal`. `analyze_image_with_ai()` folds those chains into the keyword
list, so from here on a hierarchical keyword is just a keyword that contains `<`. Lightroom expects
the inverse of that leaf-first form: a root-to-leaf path joined with pipes, `Animal|Bird|Duck`.
`parse_hierarchical_keyword()` does the conversion and also returns each level as a flat keyword:

```python
parse_hierarchical_keyword("Duck<Bird<Animal")
# ('Animal|Bird|Duck', ['Animal', 'Bird', 'Duck'])
parse_hierarchical_keyword("Landscape")
# ('Landscape', ['Landscape'])
```

A plain keyword with no `<` separator passes through unchanged as a single-level entry. Stray `>`
characters the model sometimes emits are dropped before parsing.

### Merging with existing tags

`merge_keywords()` combines the new AI keywords with whatever already lives on the photo. It keeps
three parallel buckets and preserves the hierarchy:

- `subject`: every level flattened into a single keyword list.

- `hierarchical`: cumulative pipe paths. Lightroom needs each prefix, so a `Animal|Bird|Duck` leaf
    also contributes `Animal|Bird`.

- `weighted`: a flat list that mirrors `subject`.

Deduplication is case-insensitive (compared with `casefold`), so a photo that already carries `Bird`
will not gain a second `bird`. The first-seen casing wins. When a leaf appears in more than one
chain, the longest observed chain is kept.

Two flag pairs control the merge:

- `--preserve-keywords` (default) merges new keywords with the existing ones. `--overwrite-keywords`
    replaces them instead.
- `--max-keywords N` caps how many AI-generated keywords are kept per photo before merging. The
    default keeps all of them.

The title and the description hold a single value each, so they have no merge step. Their pairs
(`--preserve-title` / `--overwrite-title` and `--preserve-description` / `--overwrite-description`)
only decide whether the generated text replaces what the photo carries. Each field answers for
itself. Replacing is the default, so a camera-written placeholder (some fill `ImageDescription` with
one on every photo) does not outlive the first run. Preserving writes the generated text only where
the field is empty, which is how a hand-written caption survives a re-run.

!!! example

    Existing `Animal|Bird` plus the model output `Seagull<Bird<Animal` and `bird` merges to a
    `hierarchical` bucket of `['Animal|Bird', 'Animal|Bird|Seagull']`: the duplicate `Bird` is collapsed
    and the deeper chain is added.

## Writing metadata

`write_metadata()` builds one ExifTool payload and applies it in a single `set_tags` call. Each
piece of generated metadata is written to both an XMP tag and its IPTC or EXIF counterpart. This way
different tools agree on the value. Lightroom prioritizes `IPTC:Keywords` for JPEGs, which is why
the flat subject list is mirrored there.

| Generated field    | Tags written                                                                 |
| ------------------ | ---------------------------------------------------------------------------- |
| Flat keywords      | `XMP-dc:Subject`, `IPTC:Keywords`                                            |
| Keyword hierarchy  | `XMP-lr:HierarchicalSubject`                                                 |
| Weighted flat list | `XMP-lr:WeightedFlatSubject`                                                 |
| Title              | `XMP-dc:Title`, `IPTC:ObjectName`                                            |
| Description        | `XMP-dc:Description`, `XMP-tiff:ImageDescription`, `EXIF:ImageDescription`\* |

\* When the target is the photo itself, ExifTool also writes `EXIF:ImageDescription`. A sidecar
holds XMP and nothing else. When a sidecar is folded into an image, ExifTool maps the `XMP-tiff`
mirror back to IFD0, but a direct write does not touch IFD0. Writing the EXIF tag too is what keeps
a camera's placeholder from surviving underneath the new description.

When `--write-title` and `--write-description` are enabled (both are on by default), title and
description are written. If the payload is empty, nothing is written.

Before the write, `write_metadata()` reads the target's current values of the payload tags. It drops
each tag that already has the value, and it writes only the rest. If no tag is left, it does not
touch the file and returns `WriteResult.UNCHANGED`. ExifTool rewrites a file even for identical
values, which moves the file's mtime. A backup tool such as rsync then copies an unchanged photo
again. A file that a save did not touch is also left out of the undo journal. A tag that the
comparison cannot match to a read (a group other than EXIF, IPTC, or XMP) is always written.

### Fields the model has no part in

`photo_tagger.fields` holds the other half of the payload: the metadata only the photographer knows.
`--artist`, `--copyright`, `--rights`, and `--marked` become the tags below. `--set TAG=VALUE` adds
any other tag under its ExifTool name. A tagging run, the `set` command, and a save from the GUI's
**Settings > Copyright...** all write the same map. The `set` command calls no model.

| Field        | Flag          | Tags written                                                |
| ------------ | ------------- | ----------------------------------------------------------- |
| Creator      | `--artist`    | `EXIF:Artist`\*, `XMP-dc:Creator`, `IPTC:By-line`           |
| Copyright    | `--copyright` | `EXIF:Copyright`\*, `IPTC:CopyrightNotice`, `XMP-dc:Rights` |
| Rights (XMP) | `--rights`    | `XMP-dc:Rights`, over the line above                        |
| Rights flag  | `--marked`    | `XMP-xmpRights:Marked`                                      |

\* For a sidecar target the writer drops the EXIF tags, because a sidecar holds XMP and nothing
else. ExifTool ignores an EXIF tag there without a word, but a payload with nothing left in it fails
the write.

A notice is a template. `{year}` becomes the year of the photo, and `{artist}` becomes the name. The
year comes from `EXIF:DateTimeOriginal`. If the photo carries no capture date, the year comes from
the file's mtime. The pipeline reads that tag for the prompt anyway, so a dated notice costs no
extra ExifTool call.

### Sidecar or embedded

By default photo-tagger writes an XMP sidecar named after the image (`image.cr3` gets `image.xmp`).
This leaves the original file byte-for-byte untouched. `--sidecar-mode` picks between the four
answers. `--write-sidecar` and `--embed-in-photo` are shorthand for the first two.

=== "Sidecar (default, `--sidecar-mode all`)"

    ```text
    photos/
      duck.cr3        # original, never modified
      duck.xmp        # title, description, and keywords
    ```

=== "Embedded (`--embed-in-photo`)"

    ```text
    photos/
      duck.cr3        # metadata written into the file itself
    ```

=== "One of each (`--sidecar-mode raw`)"

    ```text
    photos/
      duck.dng        # RAW: original, never modified
      duck.xmp        # its title, description, and keywords
      duck.jpg        # not RAW: metadata written into the file itself
    ```

=== "Both (`--sidecar-mode both`)"

    ```text
    photos/
      duck.cr3        # metadata written into the file itself
      duck.xmp        # and the same metadata beside it
    ```

!!! note

    Sidecars keep your originals completely untouched, which is why they are the default. Lightroom
    Classic reads a `.xmp` sidecar only for a proprietary RAW file (CR3, NEF, ARW). It keeps the
    metadata of a DNG, JPEG, or TIFF inside the file. See
    [Tag DNG files for Lightroom Classic](../usage/recipes.md#tag-dng-files-for-lightroom-classic).

`--sidecar-mode raw` is for the folder that holds both, the RAW+JPEG pairs a camera set to that mode
produces: the RAWs keep their bytes, and each JPEG carries its own metadata so it travels intact.
RAW is decided by extension, by exclusion: a suffix Pillow reads natively (`.jpg`, `.png`, `.tif`,
`.heic`, ...) is not RAW, and anything else is. So an unfamiliar extension gets the cautious answer
and keeps its file untouched.

`--sidecar-mode both` is the safest and most redundant choice: every photo is written twice, into
itself and into a sidecar. The photo then carries its metadata wherever it is copied, while the
sidecar stays for the catalog to edit. It costs two ExifTool writes per photo, and `plan_writes` is
what turns a mode into that list of targets. Undo records each one, so reverting puts back both.

`--sidecar-override EXT=TARGET` (`sidecar_overrides` in `config.toml`) wins over the mode for one
extension. TARGET is `sidecar`, `photo`, or `both`, the per-file forms of `all`, `none`, and `both`.
A mode name reads wrong there: `dng=none` does not say "into the DNG". `plan_writes` checks the
override before the mode, so every writer (`tag`, `watch`, `set`, and the GUI) agrees.
`parse_sidecar_overrides` makes the extension lowercase and adds the dot. It refuses an unknown
target, and it refuses `xmp`, because a sidecar has no sidecar. The CLI checks the entries at parse
time, so a typo stops the run before the first photo. The GUI logs a bad entry and follows the mode.

### Cameras ExifTool will not write to on the first try

When ExifTool cannot parse a photo's maker notes, it refuses a write: it will not move a block whose
internal offsets it cannot fix up. Some cameras ship files that trip this. Their maker-note offsets
are already wrong from the camera. `exiftool -validate` says so before anything is written to them.
So every embedded write failed with `Error: [minor] Maker notes could not be parsed`.

Such a write is tried once more with ExifTool's `-m` (ignore minor errors), the only way to put
metadata inside those files. Each retry is logged per photo. The waiver is never asked for up front,
since the check is worth having for everything else. Maker notes and image data come through
byte-identical. What stays wrong is what the camera already had wrong.

### Backups and dry runs

When writing, ExifTool keeps a `*_original` backup of the target before changing it. This is on by
default. `--no-backup-xmp` passes `-overwrite_original` to ExifTool so no backup file is left
behind. The GUI exposes the same choice as **Keep ExifTool Backup** in the Save options menu.

`--dry-run` runs the model and logs the proposed title, description, and keywords, but writes
nothing. Use it to preview output before touching any files.

!!! warning

    `--no-backup-xmp` combined with `--embed-in-photo` modifies the original image file with no backup.
    Make sure that you have your own copies before running that combination.

## Detecting tagged images

`find_tagged_images()` powers the `--skip-tagged` filter. It batches one ExifTool read across all
candidate files. When the image or its sidecar has any indicator tag populated, it marks the photo
as already tagged. The indicator tags are the keyword tags above, a title, or a description. See
[Processing pipeline](pipeline.md) for where this filter runs in the batch.

## Related pages

- [Processing pipeline](pipeline.md): how the read, prompt, merge, and write steps are orchestrated
    per photo.
- [AI providers](ai-providers.md): how the prompt reaches the model and the structured output comes
    back.
- [CLI reference](../usage/cli-reference.md): the full list of output and filter flags referenced
    here.

[models]: https://github.com/jbsilva/photo-tagger/blob/main/src/photo_tagger/models.py
