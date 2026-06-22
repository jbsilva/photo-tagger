---
icon: lucide/app-window
---

# Desktop GUI

photo-tagger ships an optional desktop GUI for a point-and-click, **review-before-write** workflow
over the same pipeline the CLI uses. You drag in photos, generate proposals with the vision model,
then inspect and edit each photo's title, description, and keywords before saving.

The GUI is purely additive: it reads existing metadata, runs the model, and writes with ExifTool
through the same building blocks as the CLI, so what you learn in the
[CLI reference](cli-reference.md) maps directly onto it. See
[Architecture](../architecture/index.md#frontends) for how the frontends share one pipeline.

## Install and launch

The GUI lives behind the `gui` extra so the base CLI stays free of the Qt dependency:

```bash
uv tool install 'photo-tagger[gui]'
photo-tagger gui
```

Run without the extra and the command prints an install hint instead of a traceback:

```console
$ photo-tagger gui
The desktop GUI needs PySide6, which is not installed.
Install the optional extra with:
    pip install 'photo-tagger[gui]'
```

!!! note

    The GUI needs the same prerequisites as the CLI: ExifTool on your `PATH` and a reachable model
    server. Use **Test connection** in the window to verify both before a run.

## The workflow

### 1. Add photos

Drag photos or folders anywhere onto the window, or use **Add files...** / **Add folder...**.
Dropped folders are expanded using the **File types** field (comma-separated, case-insensitive) and
the **Recurse** toggle, exactly like the CLI's `--ext` and `--recursive`. Photos appear in a nested
tree on the left: subfolders are grouped under their parent folder, so a deep shoot stays organized
rather than flattened.

!!! note "File types are case-insensitive but variant-aware"

    `jpg` matches `.JPG` (case-insensitive), but it does **not** cover `.jpeg`: those are distinct
    extensions, so the default list includes both. Hover any control for a tooltip explaining it.

### 2. Choose what to process

Every folder and file has a checkbox; uncheck a folder to exclude everything under it, or uncheck
individual files. Only checked photos are generated. The two columns are resizable. To take an item
off the list entirely (rather than just deselect it), select it and click **Remove** or press
++delete++ / ++backspace++; **Clear** empties the whole list. The status column shows each photo's
state: blank (pending), `working...`, `ready`, `saved ✓`, or `failed ✗`.

Click the **Photos** or **Status** column header to sort by it; click again to reverse. The Status
column sorts by processing stage (pending, working, ready, saved, failed), not the label text, so
clicking it groups all the failures or all the ready photos together. Folders stay grouped above
their sibling files either way.

The **Deselect** row unchecks photos in bulk, mirroring the CLI's skip flags so you do not have to
hunt through a large list by hand:

- **Already tagged** opens a menu of criteria for what counts as "already done", the GUI's
    field-aware [`--skip-tagged`](cli-reference.md). It reads the metadata in one pass (so a large
    folder pauses briefly) and unchecks the matching photos:
    - **Has any metadata** unchecks any photo that has a title, description, *or* keywords (the broad,
        original behavior).
    - **Has a title**, **Has a description**, **Has keywords**, or **Has a title and a description**
        target specific fields. The combined criteria require *all* of their fields, so a photo that
        only has keywords survives **Has a title and a description** and stays selected, which is what
        you want when filling in the title and description on photos that are missing them.
- **From file...** lets you pick a plain-text file listing photos to skip, one per line (by bare
    filename or full path, `#` comments allowed), and unchecks the ones it names. This is the GUI's
    [`--skip-from`](cli-reference.md), and it pairs with the CLI's `--append-to-skip-file`: point it
    at the list a CLI run wrote to resume the same work in the window.

Deselecting only unchecks: the photos stay in the list, so you can see what was skipped and re-check
any of them. The status bar reports how many photos were deselected and how many are still selected.

Selecting a **folder** (rather than a file) shows a **thumbnail grid** of its photos on the right,
like a contact sheet. Thumbnails load in the background, so a large folder of RAW files stays
responsive. Click any thumbnail to open that photo's detail (the same as clicking its name in the
tree).

### 3. Generate proposals

Pick a **Provider** (Ollama, LM Studio, or OpenAI) and a **Model**. Press **Refresh** to query the
provider for the models it currently serves and pick from the dropdown instead of typing; likely
vision-capable models are listed first. Set a custom **URL** or **API key** in the same toolbar if
your provider needs them: the key field is masked, and leaving it blank falls back to the provider's
environment variable. **Generate selected** then runs the model on the checked photos on a
background thread, building the same contextual prompt as the CLI (existing keywords, location, GPS,
camera). Results stream in and the tree status updates per photo.

To regenerate a single photo without touching your selection, open it and press **Generate this
photo** in the detail pane: it runs the model on just that photo, regardless of which photos are
checked (the counterpart to **Save this photo**). If some photos ended up `failed ✗`, **Retry
failed** in the toolbar re-runs the model on every failed photo at once.

To stop a run early, press **Cancel** (next to *Generate selected*). The photo already in flight
finishes (a model request cannot be interrupted mid-call), then the run stops and the un-started
photos return to `pending` so you can resume them later with another **Generate selected**. Anything
already generated keeps its proposal.

### 4. Review, edit, and save

Click a photo to open it on the right. The detail pane is **side-by-side** for easy comparison: an
**Existing** column (read-only) next to a **New (editable)** column.

- a **preview** (RAW files are decoded just like a real run),
- a **Source** line saying whether the existing metadata came from the image file, an XMP sidecar,
    or both,
- **Existing** vs **New** Title, Description, and Keywords lined up row by row, with the New side
    editable and seeded from the proposal,
- a **Keyword changes** diff coloring what a save will do: green for added, red and struck-through
    for removed (only with overwrite), grey for unchanged,
- a **Hierarchy** preview of the Lightroom paths the save will write.

Keywords support hierarchy with `<` (specific to general), for example `Eagle<Bird<Animal`; the diff
and hierarchy update live as you edit. Adjust anything, then press **Save this photo** to write just
the open one, or **Save selected** to write the **checked** photos that have a proposal. The save
scope matches *Generate selected* (the same checkboxes), so check everything to save everything.

The **Write** row chooses which fields a save touches: **Title**, **Description**, and **Keywords**
(all on by default), the GUI's equivalent of the CLI's `--no-write-title` / `--no-write-description`
/ `--no-write-keywords`. Uncheck one to leave that field on the photo untouched, for example uncheck
**Keywords** to refresh only the title and description while keeping a curated Lightroom keyword
list as is (turning **Keywords** off also disables **Overwrite existing keywords**, since there is
nothing to write). By default photo-tagger writes an XMP sidecar and merges the keywords with the
existing ones, just like the CLI. Toggle **Embed in photo** to write into the image instead, or
**Overwrite existing keywords** to replace rather than merge. You can edit and save a photo even
without generating a proposal first: the editable fields then start from the existing values.

!!! tip "API keys: field or environment"

    The toolbar has a masked **API key** field. Leave it blank to use the provider's environment
    variable (`OPENAI_API_KEY`, `LM_STUDIO_API_KEY`, or `OLLAMA_API_KEY`), which keeps the secret out of
    the app entirely; or type a key to use it for this session only. A typed key is held in memory for
    the run and is never written to disk. OpenAI requires a key; local Ollama and LM Studio servers
    usually do not. To set it from the environment instead, launch with, for example,
    `OPENAI_API_KEY=sk-... photo-tagger gui`.

### 5. When a photo fails

A photo that the model could not process is marked `failed ✗` in the status column. To find out why:

- **Hover** the `failed ✗` cell for a tooltip with the error, or
- **open** the photo: a red banner above the preview shows the reason (for example
    `model unreachable` or a decode error).

Once you have addressed the cause (start the model server, fix the URL, free up memory), click
**Retry failed** to re-run every failed photo, or open one and press **Generate this photo** to
retry just that one. A successful retry clears the banner and flips the status back to `ready`.

For the full traceback behind a failure, click **Open logs** (next to the status bar). The GUI
writes a timestamped, rotating log file to `~/.photo-tagger/logs/` on every run and the button opens
that folder in your file browser.

## Configuration

The GUI reads the same TOML config file and environment variables as the CLI and pre-fills the
provider, model, URL, and extensions from them. Persisting those in `.photo-tagger.toml` (see
[Configuration](../getting-started/configuration.md)) means the window opens ready to go.

The form surfaces the most common options. Which fields are written is chosen with the **Write**
checkboxes (see [Review, edit, and save](#4-review-edit-and-save)), skip lists through the
**Deselect** buttons (see [Choose what to process](#2-choose-what-to-process)). An ExifTool backup
is always kept and inference settings use their defaults. For the full set of flags (custom prompts,
caching, date-range filters, sampling, logging), use the [CLI](cli-reference.md).

## Limitations

!!! warning

    Cancelling (or closing the window mid-run) stops at the next photo boundary: the photo **already in
    flight runs to completion** before the run halts, because a model request cannot be interrupted
    mid-call. With a slow model that one photo can take a while.

- Loading a photo's preview and existing metadata is synchronous, so selecting a large RAW file may
    pause briefly the first time (results are cached afterwards).
- A standalone, double-click app bundle (with the Dock icon and name) is planned via
    [Briefcase](https://briefcase.readthedocs.io/); for now launch the GUI with `photo-tagger gui`.

For headless machines, scripting, scheduling, or piping results into other tools, use the CLI with
[`--json`](cli-reference.md#display); the GUI is meant for interactive, local use.
