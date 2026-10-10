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

On conda-forge it is a separate package, `photo-tagger-gui`, that bundles PySide6:

```bash
conda install -c conda-forge photo-tagger-gui
photo-tagger gui
```

Run without the extra (or without the `photo-tagger-gui` package) and the command prints an install
hint instead of a traceback:

```console
$ photo-tagger gui
The desktop GUI needs PySide6, which is not installed.
Install the optional extra with:
    pip install 'photo-tagger[gui]'
```

!!! note

    The GUI needs the same prerequisites as the CLI: ExifTool on your `PATH` and a reachable model
    server. Use **Test Connection** in the window to verify both before a run.

## The workflow

### 1. Add photos

Drag photos or folders anywhere onto the window, or use the **Add Photos...** button: a click opens
the file picker, and its arrow menu holds **Add Folder...** plus the folder-scan options (the **File
types** list, comma-separated and case-insensitive, and **Include subfolders**), exactly like the
CLI's `--ext` and `--recursive`. Photos appear in a nested tree on the left: subfolders are grouped
under their parent folder, so a deep shoot stays organized rather than flattened.

!!! note "File types are case-insensitive but variant-aware"

    `jpg` matches `.JPG` (case-insensitive), but it does **not** cover `.jpeg`: those are distinct
    extensions, so the default list includes both. Hover any control for a tooltip explaining it.

### 2. Choose what to process

Every folder and file has a checkbox; uncheck a folder to exclude everything under it, or uncheck
individual files. Only checked photos are generated. Rows support multi-selection the usual way
(++shift++-click for a range, ++cmd++/++ctrl++-click to add single rows, ++shift++ + arrow keys from
the keyboard); right-clicking inside the selection then offers **Check / Uncheck / Check Only**
(checks these, unchecks the rest) **/ Generate** (with or without the cache) **/ Remove** for all of
it at once, and this works in the thumbnail grid too. To take items off the list entirely (rather
than just deselect them), select them and click **Remove**, press ++delete++ / ++backspace++, or use
**Remove From List** in the right-click menu; **File > Clear List** empties the whole list.

The tree has four resizable, sortable columns (click a header to sort; click again to reverse;
folders stay grouped above their sibling files either way):

- **Photos**: the file name.
- **Type**: the extension, with `+xmp` appended when an XMP sidecar sits next to the file.
- **Status**: blank (pending), `working...`, `ready` (or `ready (cached)` when the proposal was
    replayed from the result cache), `saved ✓` (green), or `failed ✗` (red). A trailing `⚠` means
    the generated text mixes writing systems; hover it for the words. Sorting uses the processing
    stage, not the label text, so clicking it groups all the failures or all the ready photos
    together.
- **Tagged**: which metadata the file already carries, filled in by a background scan after you add
    photos: `T` title, `D` description, `K` keywords, `-` for none (hover for the full list). The
    scan reads a few dozen photos at a time and fills the column as it goes. A scan that runs longer
    than a moment (a few thousand photos, or a network share) gets a progress bar and a **Stop**
    button under the list; stopping keeps whatever was read and leaves the rest blank.

Above the list is a **search box**: type into it and the list narrows to the photos whose name
matches. Plain text matches anywhere in the name (`IMG` finds `IMG_0001.dng`), and `*`, `?` and `[]`
are wildcards (`*_edit.jpg`), the same rule the Select menu's **Name Pattern...** uses. Matching
ignores case, and in the folder tree the folders holding a match stay so you can still see where the
photos live. This only changes what the list shows: a photo the box is hiding is still in the list
and, if it is checked, is still generated and saved. The status bar says `Showing 2 of 40 photos`
whenever a search is narrowing things, so a filtered list never reads as a list that lost photos.

The small button at the end of that row picks how the list is laid out: **View as Tree** (the
default, grouped under each folder) or **View as List**. As a list, every photo sits at one level,
so a sort runs across the whole list instead of inside each folder, which is what you want to see
every failure, or every `dng`, together no matter where it lives. Photos keep their filename when
they all come from one folder, and show the part of the path that tells them apart when they do not,
so two `DSC_0042.NEF` stay distinguishable (hover a row for the full path).

Only the folder rows go: every photo keeps its own checkbox, and the rest of the window is
unchanged. What you lose is checking a whole folder in one click and opening a folder's grid by
clicking its row, so **Go > Enclosing Folder** (++cmd+up++ / ++alt+up++) is how you reach the grid
while the list is flat. The choice lasts for the session and is not written to the config file.

The **Select** menu checks and unchecks photos in bulk, so you do not have to hunt through a large
list by hand:

- **Check All** / **Uncheck All** flip every checkbox at once; **Invert Checked** swaps checked and
    unchecked.
- **Check**, **Uncheck**, and **Check Only** each open the same three criteria, so you can act on
    part of a list without clicking through it. **Check Only** is the "just these" move: it checks
    what matched and unchecks everything else.
    - **File Type** lists the extensions actually in the list with a count each (`dng (120)`,
        `jpg (120)`), built fresh every time the menu opens.
    - **Status** works on where a photo is in the cycle: **Not generated**, **Generated**, **Saved**,
        **Failed**, **Untagged**, or **Mixed writing systems**. These are the same criteria as the
        thumbnail grid's filter, so **Uncheck > Status > Saved** before a rerun means the same thing
        in both places.
    - **Name Pattern...** asks for text to match against the filename. Plain text means "contains"
        (`IMG` finds `IMG_0001.dng`), and anything with `*`, `?`, or `[` is treated as a glob
        (`*_edit.jpg`). A pattern holding a `/` is matched against the whole path, so it can name a
        subfolder. Matching ignores case, and the box remembers your last pattern.
- **Uncheck Already Tagged** opens a menu of criteria for what counts as "already done", the GUI's
    field-aware [`--skip-tagged`](cli-reference.md). It goes by the **Tagged** column, so it
    unchecks exactly what the list shows. When the scan behind that column has not reached the end
    yet, the criterion waits for it and applies the moment it does: a progress bar with a **Stop**
    button appears under the list, and the window stays usable meanwhile. The criteria are:
    - **Has any metadata** unchecks any photo that has a title, description, *or* keywords (the broad,
        original behavior).
    - **Has a title**, **Has a description**, **Has keywords**, or **Has a title and a description**
        target specific fields. The combined criteria require *all* of their fields, so a photo that
        only has keywords survives **Has a title and a description** and stays selected, which is what
        you want when filling in the title and description on photos that are missing them.
- **Uncheck From Skip List...** lets you pick a plain-text file listing photos to skip, one per line
    (by bare filename or full path, `#` comments allowed), and unchecks the ones it names. This is
    the GUI's [`--skip-from`](cli-reference.md), and it pairs with the CLI's
    `--append-to-skip-file`: point it at the list a CLI run wrote to resume the same work in the
    window.

Deselecting only unchecks: the photos stay in the list, so you can see what was skipped and re-check
any of them. The status bar reports how many photos were deselected and how many are still selected.

Selecting a **folder** (rather than a file) shows a **thumbnail grid** of its photos on the right,
like a contact sheet. Thumbnails load in the background, so a large folder of RAW files stays
responsive. Click any thumbnail to open that photo's detail, tick its checkbox to select or deselect
it (kept in sync with the tree), or right-click it for the same menu as its row in the tree (retry,
skip cache, reveal, remove). Each thumbnail carries small badges (hover for the explanation): the
top-right one tracks the lifecycle (red ✗ failed, green ✓ saved, indigo dot for
generated-but-not-saved), and the top-left ones flag a photo that already has metadata (`M`), an XMP
sidecar (`S`), or text that mixes writing systems (an amber `!`). The **Show** box above the grid
filters it by any of the same states.

#### Getting back to where you were

The row above the right-hand pane retraces your steps, the way a browser does. **←** returns to the
place you were before this one, so a photo you opened from a grid goes back to that grid, and
hopping between photos goes back to the one you were just looking at; **→** replays a step you took
back. The label beside the arrows names what is open (`Shoot 1 / DSC_0042.NEF`), since the tree row
may be scrolled out of sight, and each arrow's tooltip names where it leads.

The **Go** menu holds the same two moves with their keyboard shortcuts (++cmd+bracket-left++ /
++cmd+bracket-right++ on macOS, ++alt+left++ / ++alt+right++ elsewhere), plus **Enclosing Folder**
(++cmd+up++ / ++alt+up++), which opens the grid of the folder holding the open photo even when you
picked that photo straight from the tree and so have no grid to go back to. Photos you remove from
the list leave the trail with them, so ← never reopens a row that is gone.

### 3. Generate proposals

Pick a **Provider** (Ollama, LM Studio, llama.cpp, or OpenAI) and a **Model** in the header. Press
**Refresh** to query the provider for the models it currently serves and pick from the dropdown
instead of typing; likely vision-capable models are listed first. **Connection...** opens the
settings that rarely change: a custom **Base URL**, a masked **API key** (leave it blank to fall
back to the provider's environment variable), and the **Test Connection** check for ExifTool and the
model.

**Generate Selected** (in the bottom bar) then runs the model on the checked photos on a background
thread, building the same contextual prompt as the CLI (existing keywords, location, GPS, camera).
Results stream in, the tree status updates per photo, and a progress bar in the bottom bar counts
the batch down. Next to it, a clock shows the time **elapsed** and, once the first photo is done,
the estimated time **left** (`2:30 elapsed · 8:10 left`), so a long batch tells you how long it
still needs.

Results are **cached by default** (see [Configuration](#configuration)): re-running a batch after a
crash, or generating a folder you partly processed before, reuses the earlier answers for unchanged
photos instead of calling the model again. Toggle **Settings > Cache AI Results** to turn that off
for the session, or use the arrow on either **Generate** button (or a photo's right-click menu) for
a one-time **Skip Cache** run.

The model writes titles, descriptions, and keywords in English by default. **Settings > Metadata
Language** switches that with one click: pick a language from the menu, or choose **Other...** to
type any language name the model understands (it then joins the menu). The choice applies from the
next generation on (cached results in the old language are not reused) and is saved to the config
file as `output_language` under `[inference]`, so CLI runs pick it up too. It is separate from
**Settings > Language**, which translates the interface itself.

A model asked for one language sometimes answers in two, and a title comes back as
`Decorated陶笛 with Geometric Pattern` where `Ocarina` would have said the same thing. Every generated
proposal is checked for this, and a photo with mixed writing systems gets an amber note over its
editable fields naming the words, a `⚠` in its **Status** column (hover it for the words), and a
count in the run's closing status line. To pull those photos out of a long list, use **Select >
Check Only > Status > Mixed writing systems**, which the closing line names; the folder grid offers
the same entry under **Show**, and marks each flagged thumbnail with an amber `!`. The proposal is
still yours to save: mixed script is right when the photo shows the text (a sign, a book cover, a
printed character), and only you can tell. **Settings > Rewrite Mixed-Language Output** makes the
run ask the model again for those photos, at the cost of one extra call each; it is off by default,
and the note appears either way.

To regenerate a single photo without touching your selection, open it and press **Generate this
photo** in the detail pane, or right-click it in the tree and choose **Generate** (a failed photo
shows **Retry Generation** there instead). If several photos ended up `failed ✗`, **Retry Failed**
in the bottom bar re-runs the model on all of them at once (the button stays disabled while nothing
has failed).

When the model gets a photo wrong (a deer tagged as a boar, say), correct it with the **Hint for the
AI** field next to *Generate This Photo*: type a note such as `The animal is a deer` and press Enter
(or the button) to regenerate. The hint is sent with the photo as a note the model must trust over
its own reading of the image, and a hinted photo always calls the model instead of replaying its
cached result. The corrected answer is not cached either: a one-off correction must not become the
generic answer a later hint-less run of that photo replays. Each photo keeps its own hint (it
survives browsing to other photos and rides along in batch runs too) and it is never written to the
file. The CLI equivalent for a whole run is [`--hint`](cli-reference.md#inference).

To stop a run early, press **Cancel** (next to *Generate Selected*). The photo already in flight
finishes (a model request cannot be interrupted mid-call), then the run stops and the un-started
photos go back to `pending` (or to `ready`, if they already had a proposal) so you can resume them
later with another **Generate Selected**. Anything already generated keeps its proposal. The same
button also cancels a save in progress.

A run you plan to leave alone can write its photos itself: see
[Generate and save in one click](#generate-and-save-in-one-click).

### 4. Review, edit, and save

Click a photo to open it on the right. The detail pane is **side-by-side** for easy comparison: an
**Existing** column (read-only) next to a **New (editable)** column.

- a **preview** (RAW files are decoded just like a real run). Click it to open the photo in a window
    of its own, large enough to judge what the model saw: it starts fitted to the window, then
    **Zoom in** / **Zoom out** (or `+`/`-`, or Ctrl+scroll and the trackpad pinch) magnify up to
    8:1, **Fit** (Ctrl+0) and **100%** (Ctrl+1) jump between the two useful scales, a double-click
    toggles between them, drag the photo to pan it, and **Full screen** (`F`) fills the display.
    `Esc` closes the viewer,
- **Existing** vs **New** Title, Description, and Keywords lined up row by row, with the New side
    editable and seeded from the proposal. Both keyword columns are rendered and sorted the same
    way, so they read line by line: each hierarchy appears once as its deepest chain, in the same
    `<` notation you type, without a second bare copy of a keyword the chain already covers. The
    description boxes grow with their content instead of reserving space,
- a line under each **Existing** Title and Description naming the file it came from. A sidecar wins
    over the photo, so when the photo holds a different value that line says so:
    `shadows "default" in the image file`, some cameras writing exactly that into every photo. That
    value is what a plain `exiftool photo.dng` prints, and only a save to the photo itself (**Write
    To > The Photo Itself**) replaces it,
- a collapsible **Keyword changes** section. Its header always summarizes what a save would do
    (`+3 / -1`, or `no change`); expand it for the colored diff (green added, red struck-through
    removed, grey unchanged) and the resulting keyword **tree**, drawn with `tree`-style branch
    guides (`├─`/`└─`).

Keywords support hierarchy with `<` (specific to general), for example `Eagle<Bird<Animal`; the
summary, diff, and tree update live as you edit. Adjust anything, then press **Save This Photo** to
write just the open one, or **Save Selected** (bottom bar) to write the **checked** photos that have
a proposal. The save scope matches *Generate Selected* (the same checkboxes), so check everything to
save everything.

The arrow on either **Save** button opens the save options, which choose what every save writes.
There is one set of them, shared by everything that writes: both **Save** buttons, and the **Save
Options** submenu on either **Generate** button's arrow, which is the same menu again. Every one of
those buttons spells out the current choice in its tooltip.

- **Write Title**, **Write Description**, **Write Keywords** (all on by default), the GUI's
    equivalent of the CLI's `--no-write-title` / `--no-write-description` / `--no-write-keywords`.
    Uncheck one to leave that field on the photo untouched, for example uncheck **Write Keywords**
    to refresh only the title and description while keeping a curated Lightroom keyword list as is
    (turning a field off also disables its **Overwrite Existing** entry, since there is nothing to
    write).
- **Overwrite Existing** holds one entry per field, deciding what a save replaces on a photo that
    already carries that field: the GUI's `--overwrite-keywords`, `--preserve-title`, and
    `--preserve-description`. **Keywords** is off, so new keywords merge in; check it to replace
    them. **Title** and **Description** are on, since each holds a single value, and that is what
    clears a camera's placeholder. Uncheck one to keep what the photo has and fill in only the
    photos without it, so a caption you wrote by hand survives a re-run.
- **Write To** picks where a save puts the metadata, the GUI's equivalent of the CLI's
    `--sidecar-mode`: **XMP Sidecar** (the default) writes a `.xmp` file next to every photo, **The
    Photo Itself** writes into each image file, **Sidecar for RAW, Photo Otherwise** does one of
    each per file type, for a folder of RAW+JPEG pairs, and **Both, Photo and Sidecar** writes the
    photo *and* a sidecar, so the photo travels tagged and the sidecar stays for your catalog.
- **Keep ExifTool Backup** (on by default) lets ExifTool save the untouched file as `*_original`
    before writing, the GUI's equivalent of the CLI's `--backup-xmp` / `--no-backup-xmp`. Uncheck it
    to write in place: saving a few thousand photos otherwise leaves a full second copy of each one
    next to the original. Only do that if you have your own backup elsewhere.

**Save Selected** writes on a background thread, like a generation run: the same progress bar and
elapsed/remaining clock track the batch, the window keeps responding, and **Cancel** stops it after
the file currently being written (the photos it never reached stay `ready` to save again). Writing a
few thousand photos takes minutes, so the counter is how you tell it is working rather than stuck.

You can edit and save a photo even without generating a proposal first: the editable fields then
start from the existing values.

#### Generate and save in one click

A run you will not sit through can write its own photos. **Save After Generating**, on the arrow of
either **Generate** button, writes every photo a run produced as soon as that run finishes. Both
buttons then read **Generate and Save**, and their tooltips spell out what that save writes.

Before it writes, the run does the two things you would do at the keyboard. First it has another go
at the photos that **failed**, which is what **Retry Failed** does by hand: a model server that
blinks for a minute costs you those photos otherwise, and nobody is there to press the button. Then
it asks again for the photos whose text came back in **two writing systems**, because a second
answer usually comes back in one. That second call reads past the result cache (the answer it is
unhappy with is the one stored there) and stores what it gets, so a later run of the same folder
does not bring the mixed text back. A batch started with **Skip Cache** leaves the cache alone here
too.

Each pass waits a few seconds before it starts, because a failure usually means the server is busy
or restarting, and each happens once per batch, so a provider that is down cannot loop. Photos still
failing keep their `failed ✗` row and are counted in the closing line:
`Saved 48 of 48 checked photos. 2 photos failed (Retry Failed to try again).` Photos that come back
mixed a second time keep their amber `⚠` and are saved as they are.

!!! note "When the foreign word is the right answer"

    Mixed script is correct when the photo shows the text: a sign, a book cover, a printed character.
    Asking again may replace that word with an English one, and with nobody reviewing, that answer is
    what gets written. If your library is full of such photos, leave **Save After Generating** off for
    them and review as usual, or check the `⚠` rows afterwards.

It covers the run you just started, so a proposal left over from an earlier one is still yours to
review. Nothing else changes. The write is the one **Save Selected** makes, down to the
[save options](#4-review-edit-and-save) it makes it with: **Save Options** in the **Generate**
button's own menu is that same menu, so you can set what will be written where you turn the writing
on. **Cancel** stops the writing after the file in flight, and **Undo Writes** puts the batch back.
If the save options would write nothing at all, the window says so instead of finishing an
unattended run in silence.

**Settings > Save Settings as Defaults...** keeps the choice, as `save_after_generating` under
`[gui]`. An overnight import is then one click on the next launch too. The toggle is off by default,
because the window is built around reviewing before writing.

!!! tip "API keys: field or environment"

    The **Connection...** dialog has a masked **API key** field. Leave it blank to use the provider's
    environment variable (`OPENAI_API_KEY`, `LM_STUDIO_API_KEY`, or `OLLAMA_API_KEY`), which keeps the
    secret out of the app entirely; or type a key to use it for this session only. A typed key is held
    in memory for the run and is never written to disk (not even by **Save Settings as Defaults**).
    OpenAI requires a key; local Ollama and LM Studio servers usually do not. To set it from the
    environment instead, launch with, for example, `OPENAI_API_KEY=sk-... photo-tagger gui`.

### 5. When a photo fails

A photo that the model could not process is marked `failed ✗` in the status column. To find out why:

- **Hover** the `failed ✗` cell for a tooltip with the error, or
- **open** the photo: a red banner above the preview shows the reason (for example
    `model unreachable` or a decode error).

Once you have addressed the cause (start the model server, fix the URL, free up memory), click
**Retry Failed** to re-run every failed photo, or right-click one and choose **Retry Generation** to
retry just that one. A successful retry clears the banner and flips the status back to `ready`.

For the full traceback behind a failure, use **Help > Open Logs**. The GUI writes a timestamped,
rotating log file to `~/.photo-tagger/logs/` on every run and the action opens that folder in your
file browser.

### Keyword rules: a vocabulary, and harmonized shoots

**Settings > Keyword Rules...** holds the two settings that decide which keywords a save ends up
writing. Both are kept for the session and persist through *Save Settings as Defaults*, so a CLI run
picks up the same rules.

A **controlled vocabulary** restricts generated keywords to the terms in a keyword file, the GUI's
[`--vocabulary`](cli-reference.md#controlled-vocabulary). Choose a Lightroom keyword export (`.txt`
or `.csv`) or a plain list of terms and `Animal|Bird|Osprey` paths; the label under the picker
reports how many keywords it holds, or why the file was refused. The file does two things at once:
it is listed in the prompt, so the model prefers your catalog's terms in the first place, and every
generated keyword is snapped onto it afterwards, so `ospreys` is written as your `Osprey` under your
`Animal|Bird` hierarchy whatever the model said. Tick **Write only keywords the vocabulary covers**
for [`--vocabulary-strict`](cli-reference.md#controlled-vocabulary): keywords the file does not have
are dropped instead of written as they came, and the status bar names the most frequent rejects so
the vocabulary can grow on purpose rather than by accident.

!!! note "The vocabulary is part of the cache key"

    Swapping vocabulary files starts a fresh cache slice, so a run never replays keywords chosen under
    the old one. Snapping happens *after* the cache lookup, so a vocabulary you choose today also
    applies to answers cached yesterday.

**Split shoots after** N minutes turns on shoot harmonization, the GUI's
[`--session-gap`](cli-reference.md#sessions). Photos are grouped into shoots wherever the capture
time jumps by more than that gap (falling back to the file date), and each shoot's keywords are made
to agree with themselves: the spelling and the hierarchy most of the shoot used win for all of it,
so forty frames of the same bird stop arriving in the catalog as `Osprey`, `Ospreys`, and
`Wildlife|Raptor|Osprey`. The CLI holds a session's writes until every photo in it has been
analyzed; the window writes nothing until you press Save, so the pass runs over the **proposals** at
the end of each generation instead, before you review them. **Tools > Harmonize Shoots Now** runs it
again after you have edited some keywords by hand. Zero (shown as `off`) treats every photo on its
own.

### Your copyright, on every save

**Settings > Copyright...** holds the fields no model can guess: who took the photo, and who owns
it. Fill in **Creator** and a **Copyright notice** and every save writes them, alongside whatever
the model produced. It is the GUI's
[`--artist` and `--copyright`](cli-reference.md#copyright-and-other-fields).

A notice is a template. `{year}` becomes the year of the photo, and `{artist}` becomes the name
above. One notice therefore dates a shoot from 2019 and one from last week correctly, in the same
batch. The year is the photo's own capture date. A photo without one gets its file date instead.

The panel under the fields lists the tags a save writes, and what each one says for a real year. A
template with a typo in it is visible there before anything is written.

The **Insert** row types the awkward parts for you: `©`, `{year}`, and `{artist}`. Each one lands in
the field holding the cursor, and the buttons take no focus, so you can keep typing where you left
off.

**XMP rights** is the same notice for XMP alone, which is UTF-8 and carries a real `©`. EXIF types
its copyright and artist fields as ASCII until Exif 3.0, so `(c)` is the safe spelling in the
notice. The dialog says so in amber when a value needs more than ASCII, and writes it anyway: an
accent in a name is not a mistake. Leave **XMP rights** empty to use the notice as it stands.
**Rights flag** sets `XMP-xmpRights:Marked`, and *Leave as is* writes nothing at all.

The save menu's **Write Copyright** turns the whole block off for a while without clearing the
fields, and is grayed out until there is something to write. A save writes the copyright even when
Title, Description, and Keywords are all unchecked. That is how you stamp photos you do not want
re-tagged. The settings persist through *Save Settings as Defaults* into the config file's
`[copyright]` table, so CLI runs credit your photos the same way.

### Build a vocabulary from your own library

**Tools > Build Vocabulary...** writes the keyword file the strict vocabulary needs, out of the
keywords your photos already carry. It is the
[`photo-tagger vocabulary`](cli-reference.md#building-a-vocabulary) command with its flags as form
fields:

- **Read from** the photos in the list (read through ExifTool, so any application that writes XMP or
    IPTC counts, not only Lightroom), a Lightroom **keyword export**, or both.
- **Keep a keyword when** it is used at least *N* times (the single most useful knob: in a catalog
    an AI has been writing to, most keywords are used once), with a cap on the file's size, an
    option to keep keywords containing digits, and one to write bare keywords without their
    hierarchies.
- **Organize with the model** (off by default) asks the model the two things counting cannot settle:
    which keywords are synonyms of each other, and what hierarchy the list should have. It never
    decides what to keep and never invents a keyword. It needs a reachable provider, and uses the
    one in the header.
- **Write to** a vocabulary file, plus an optional **drop report**: a CSV naming every dropped
    keyword, its count, and the rule that cut it, which is what makes the thresholds tunable.

The build runs in the background (counting a large library takes a while, and organizing takes model
calls), and when it finishes the window offers to put the new file straight to work as the active
vocabulary. Nothing is written to your photos.

### Watch a folder

**Tools > Watch Folder...** is the import-time workflow: point it at the folder your card reader,
tethered capture, or sync client fills, and photos join the list as they arrive. Photos already
there are picked up first. It mirrors [`photo-tagger watch`](cli-reference.md#watching-a-folder),
including **Check every** (how often the folder is listed) and **Settle for** (how long a file must
sit unchanged before it is picked up, so a photo still being copied is left alone until the copy
finishes).

Two toggles decide how far it goes: **Generate each new photo** (on by default) runs the model on
each arrival so a proposal is waiting for you, and **Save it too, without reviewing** (off by
default) writes it straight away. Saving is opt-in because reviewing before writing is the point of
the window; turn it on for an unattended import, where it behaves like the CLI command. A photo that
lands mid-run waits for the run in flight rather than starting a second one. The menu entry becomes
**Stop Watching** while a watch is running, and closing the window stops it.

### Be told when a run ends

A batch of a few hundred photos is something to start and leave, so **Settings > Notify When a Run
Ends** raises a desktop notification the moment a generation or a save finishes. It carries the same
line the status bar shows, so the two never disagree, and **Play a Sound With Notifications** adds
an audible alert. Both are off by default; **Save Settings as Defaults...** writes them to the
config file, where they are the same
[`notify` / `notify_sound`](cli-reference.md#end-of-batch-notification) keys the CLI reads.

A run you cancelled yourself raises no alert: you are already at the keyboard. The notification uses
whatever your system already provides, so nothing appears (and nothing breaks) on a machine without
one, and it pairs naturally with an unattended **Watch Folder** import.

### Rest a long run

A model at full tilt can draw more power than a laptop charger supplies, so an overnight batch
discharges the battery even while plugged in, runs the machine hot, and leaves little of it for
anything else. **Settings > Pacing...** spends part of the time not working instead: the battery
catches up, the fans wind down, and the average draw lands under what the charger gives.

**Work only** is the share of the time a run may work; `70` rests about three minutes of every ten.
**Rest below** rests until the charge comes back whenever the battery falls that low, which is the
one to reach for on a laptop because it needs no wattage numbers and follows your screen brightness
and whatever else is running. It only ever rests while a charger is plugged in, since on battery
power a rest recharges nothing. **Work for** and **Rest at most** bound how often a rest happens and
how long it lasts.

A run rests between photos, never in the middle of one, and the status bar says why it stopped and
for how long. Everything here is off by default, is kept for the session, and **Save Settings as
Defaults...** writes it to the config file's [`[pacing]`](cli-reference.md#pacing) table, which CLI
runs read too.

### Undo what a save wrote

Every save records what it wrote to a small journal (**Settings > Record Saves for Undo**, on by
default), the same journal [`photo-tagger undo`](cli-reference.md#undoing-a-run) reads. **Tools >
Undo Writes...** lists every recorded run, newest first, from this window *and* from the command
line, with the time it ran and how many files it wrote.

Pick one and press **Preview** to see what putting it back would do, file by file, without touching
anything; press **Undo** (after a confirmation) to do it. Sidecars the run created are deleted, and
files it overwrote are restored from their ExifTool `*_original` backup. A file that changed since
the run is left alone, because that change is a later edit and not this run's to undo; **Also revert
files changed since the run** overrides that. A file written with **Keep ExifTool Backup** unchecked
cannot be restored at all, since there is no copy of what it held, and the dialog says so per file.

Photos in the list whose writes were reverted go back to `ready` (their proposal is still there to
save again) and their **Tagged** column is scanned again.

### Export a CSV report

**File > Export CSV Report...** writes a spreadsheet with **one row per photo in the list**, the
same report the CLI's [`--csv-file`](cli-reference.md#csv-report) produces. Each row gathers
everything the GUI knows about a photo: its status, the working title/description/keywords (the
keywords as a save would write them, honoring **Overwrite Existing > Keywords**), the
title/description/keywords already on the file, the camera/location EXIF read when it was generated,
and the per-photo token usage and timing. Photos you have not generated yet still get a row, with
the generated and usage columns left blank.

Any unsaved edits in the open photo are folded into its row first, so the export reflects exactly
what you see. Pick a destination in the save dialog (a `.csv` suffix is added if you omit one); the
status bar confirms how many photos were written. This is handy for reviewing a batch in a
spreadsheet or sharing the results without opening every photo.

## Configuration

The GUI reads the same TOML config file and environment variables as the CLI and pre-fills the
provider, model, URL, extensions, and save options from them. It can also write that file:
**Settings > Save Settings as Defaults...** updates the config file in effect (or creates
`~/.config/photo-tagger/config.toml`) with the current provider, model, URL, file types, recursion,
and save options. The save **merges**: comments, ordering, and every setting the GUI does not manage
are preserved, and the API key is never written. For everything the GUI does not surface (prompt
file, sampling, workers, filters), **Settings > Edit Config File...** opens the file in your editor.
See [Configuration](../getting-started/configuration.md) for the format.

Generated results are cached in the same SQLite format as the CLI's
[`--cache-file`](cli-reference.md): the GUI uses the configured `cache_file` if the config sets one,
and `~/.photo-tagger/cache.sqlite` otherwise. **Settings > Cache AI Results** turns the cache off
for a session; a photo's right-click menu offers **Generate (Skip Cache)** for one-off fresh
results.

The window surfaces the most common options; the rest of the inference settings come from the config
file. One is worth knowing about: `[inference] jpeg_dimensions` is the size the model is shown, and
raising it is what lets a run read small text in a photo. It costs time per photo and starts a fresh
cache, so see [Image size](cli-reference.md#image-size) first. For the full set of flags (custom
prompts, date-range filters, sampling, logging), use the [CLI](cli-reference.md).

## Limitations

!!! warning

    Cancelling (or closing the window mid-run) stops at the next photo boundary: the photo **already in
    flight runs to completion** before the run halts, because a model request cannot be interrupted
    mid-call. With a slow model that one photo can take a while.

- Loading a photo's preview and existing metadata is synchronous, so selecting a large RAW file may
    pause briefly the first time (results are cached afterwards).
- On macOS, `./packaging/build_macos_app.sh` builds a double-clickable (unsigned) `Photo Tagger.app`
    with PyInstaller; elsewhere launch the GUI with `photo-tagger gui`.

For headless machines, scripting, scheduling, or piping results into other tools, use the CLI with
[`--json`](cli-reference.md#display); the GUI is meant for interactive, local use.
