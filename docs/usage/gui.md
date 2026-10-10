---
icon: lucide/app-window
---

# Desktop GUI

photo-tagger includes an optional desktop GUI. It gives a point-and-click, review-before-write
workflow over the same pipeline that the CLI uses. You drag in photos and generate proposals with
the vision model. A proposal is the title, description, and keywords that the model suggests for a
photo. Then you examine and edit each proposal before you save it.

The GUI is only an additional interface: it reads existing metadata, runs the model, and writes with
ExifTool through the same building blocks as the CLI. As a result, what you learn in the
[CLI reference](cli-reference.md) also applies directly to the GUI. See
[Architecture](../architecture/index.md#frontends) for how the frontends share one pipeline.

## Install and launch

The GUI is in the `gui` extra (an optional group of dependencies), so the base CLI does not need Qt:

```bash
uv tool install 'photo-tagger[gui]'
photo-tagger gui
```

On conda-forge, the GUI is a separate package, `photo-tagger-gui`, that bundles PySide6:

```bash
conda install -c conda-forge photo-tagger-gui
photo-tagger gui
```

If the extra (or the `photo-tagger-gui` package) is not installed, the command prints an install
hint instead of a traceback:

```console
$ photo-tagger gui
The desktop GUI needs PySide6, which is not installed.
Install the optional extra with:
    pip install 'photo-tagger[gui]'
```

!!! note

    The GUI needs the same prerequisites as the CLI: ExifTool (on your `PATH` or at the configured path)
    and a reachable model server. Before a run, use **Test Connection** in the window to make sure that
    both are available.

## The workflow

### 1. Add photos

Drag photos or folders anywhere onto the window, or use the **Add Photos...** button. A click on the
button opens the file picker. **File > Add Photos...** and **File > Add Folder...** do the same. The
arrow menu of the button holds **Add Folder...** and two entries for folder scans. The first entry
is the **File types** list, which is comma-separated and case-insensitive. The second entry is
**Include subfolders**. These two entries work exactly like the CLI flags `--ext` and `--recursive`.

Photos appear in a nested tree on the left. The tree groups subfolders under their parent folder, so
a shoot with many folder levels stays organized instead of flat.

!!! note "File types are case-insensitive but variant-aware"

    `jpg` matches `.JPG` (case-insensitive), but it does not match `.jpeg`. These are two different
    extensions, so the default list includes both. Hover over any control to see a tooltip that explains
    it.

### 2. Choose what to process

Every folder and file has a checkbox. To exclude everything under a folder, uncheck the folder. You
can also uncheck individual files. The GUI generates proposals only for the checked photos.

You can select more than one row in the usual way:

- ++shift++-click selects a range.
- ++cmd++/++ctrl++-click adds single rows.
- ++shift++ + arrow keys select rows from the keyboard.

Right-click inside the selection to act on all of it at once. The menu offers **Check**,
**Uncheck**, **Check Only** (checks these rows and unchecks the rest), **Generate** (with or without
the cache), and **Remove From List**. This menu works in the thumbnail grid too.

To take items off the list entirely (instead of only unchecking them), select them first. Then click
**Remove**, press ++delete++ / ++backspace++, or use **Remove From List** in the right-click menu.
The arrow on **Remove** offers **Remove Checked** and **Remove Unchecked**. These two entries act on
the checkboxes, not on the selection. **File > Clear List** empties the whole list.

The tree has four columns, and you can resize and sort each one. To sort by a column, click its
header. To reverse the order, click the header again. In both orders, the folders stay grouped above
their sibling files. The columns are:

- **Photos**: the file name.
- **Type**: the extension. If an XMP sidecar is next to the file, `+xmp` follows the extension.
- **Status**: blank (pending), `working...`, `ready`, `saved ✓` (green), or `failed ✗` (red). If the
    GUI replayed the proposal from the result cache, the status is `ready (cached)`. A `⚠` at the
    end means that the generated text mixes writing systems. Hover over the `⚠` to see the words.
    The sort uses the processing stage, not the label text. As a result, a click on the header
    groups all the failures, or all the ready photos, together.
- **Tagged**: the metadata that the file already carries. After you add photos, a background scan
    reads a few dozen photos at a time and fills in this column as it goes. The letters are `T` for
    title, `D` for description, `K` for keywords, `C` for a copyright notice, and `-` for none
    (hover over a cell for the full list). Some scans take more than a moment, for example a scan of
    a few thousand photos or of a network share. Such a scan shows a progress bar and a **Stop**
    button under the list. If you stop the scan, the column keeps the values that the scan read and
    leaves the rest blank.

A search box is above the list. When you type into it, the list shows only the photos whose name
matches. Plain text matches anywhere in the name (`IMG` finds `IMG_0001.dng`). The characters `*`,
`?`, and `[]` are wildcards (`*_edit.jpg`), with the same rule that **Name Pattern...** in the
**Select** menu uses. The match ignores case. In the folder tree, the folders that hold a match stay
visible, so you can still see where the photos are.

The search only changes what the list shows. A photo that the search hides is still in the list. If
that photo is checked, the GUI still generates and saves it. While a search is active, the status
bar says `Showing 2 of 40 photos`. As a result, you never mistake a filtered list for a list that
lost photos.

The small button at the end of that row sets the layout of the list: **View as Tree** or **View as
List**. **View as Tree** is the default and groups the photos under each folder. In **View as
List**, every photo is at one level. A sort then runs across the whole list instead of inside each
folder. With this layout, a sort shows every failure, or every `dng`, together, no matter which
folder holds it.

If all the photos come from one folder, each row shows only the filename. Otherwise, each row shows
the part of the path that tells the photos apart. As a result, you can tell apart two photos named
`DSC_0042.NEF`. Hover over a row to see the full path.

The list layout removes only the folder rows: every photo keeps its own checkbox, and the rest of
the window does not change. Without folder rows, you cannot check a whole folder in one click. You
also cannot open the grid of a folder with a click on its row. To open the grid while the list is
flat, use **Go > Enclosing Folder** (++cmd+up++ / ++alt+up++). The layout choice lasts for the
session. The GUI does not write it to the configuration file.

The **Select** menu checks and unchecks photos in bulk, so you do not have to search through a large
list by hand:

- **Check All** and **Uncheck All** check or uncheck every photo at once. **Invert Checked** swaps
    checked and unchecked.
- **Check**, **Uncheck**, and **Check Only** each open the same three criteria. With them, you can
    act on part of a list without a click on each photo. **Check Only** checks the photos that match
    and unchecks all the others.
    - **File Type** lists the extensions that are in the list, each with a count (`dng (120)`,
        `jpg (120)`). The GUI builds this list again each time the menu opens.
    - **Status** works on the stage of a photo in its cycle: **Not generated**, **Generated**,
        **Saved**, **Failed**, **Untagged**, **No copyright**, or **Mixed writing systems**. These are
        the same criteria as the filter of the thumbnail grid. As a result, **Uncheck > Status >
        Saved** before a rerun has the same meaning in both places. **No copyright** finds the photos
        that have no copyright notice in EXIF, IPTC, or XMP. A photo that has only a notice (many
        cameras write one) still counts as **Untagged**.
    - **Name Pattern...** asks for text to match against the filename. Plain text means "contains"
        (`IMG` finds `IMG_0001.dng`). Text with `*`, `?`, or `[` is a glob (a wildcard pattern such as
        `*_edit.jpg`). A pattern with a `/` matches against the whole path, so it can name a
        subfolder. The match ignores case. The box remembers your last pattern.
- **Uncheck Already Tagged** opens a menu of criteria for what counts as "already done". It is the
    field-aware GUI version of [`--skip-tagged`](cli-reference.md). It uses the **Tagged** column,
    so it unchecks exactly what the list shows. If the scan for that column is not complete, the
    criterion waits and applies as soon as the scan ends. During the wait, a progress bar with a
    **Stop** button appears under the list, and you can still use the window. The criteria are:
    - **Has any metadata** unchecks any photo that has at least one of these: a title, a description,
        or keywords. This is the broad, original behavior.
    - **Has a title**, **Has a description**, **Has keywords**, and **Has a title and a description**
        target specific fields. A combined criterion requires every one of its fields. As a result, a
        photo that has only keywords stays checked under **Has a title and a description**. To fill in
        the title and description on photos that do not have them, use this criterion.
- **Uncheck From Skip List...** asks for a plain-text file that lists photos to skip, one per line.
    The file names each photo by bare filename or full path, and it can hold `#` comments. The GUI
    then unchecks the photos that the file names. This is the GUI version of
    [`--skip-from`](cli-reference.md), and it pairs with the CLI flag `--append-to-skip-file`. To
    resume the work of a CLI run in the window, choose the list that the CLI run wrote.

Deselecting only unchecks photos: the photos stay in the list, so you can see the skipped photos and
check any of them again. The status bar reports how many photos the criterion deselected and how
many are still selected.

If you select a folder (not a file), a thumbnail grid of its photos appears on the right, like a
contact sheet. Thumbnails load in the background, so the window stays responsive for a large folder
of RAW files. In the grid, you can do these actions:

- Click a thumbnail to open the detail of that photo.
- Check or uncheck the checkbox of a thumbnail. The tree stays in sync with the grid.
- Right-click a thumbnail to get the same menu as its row in the tree (retry, skip cache, reveal,
    remove).

Each thumbnail carries small badges. Hover over a badge to see what it means. The badge at the top
right shows the lifecycle stage: a red ✗ for failed, a green ✓ for saved, and an indigo dot for
generated but not saved. The badges at the top left mark a photo that already has metadata (`M`), a
copyright notice (`C`), or an XMP sidecar (`S`). An amber `!` at the top left marks text that mixes
writing systems. The **Show** box above the grid filters the grid by any of the same states. It also
offers **All** and **Selected** (the checked photos). The **Sort by** box orders the thumbnails by
name, file type, status, or existing metadata. Both choices last for the session.

#### Getting back to where you were

The row above the right-hand pane retraces your steps, like a browser. **←** returns to the place
that you were at before this one. For a photo that you opened from a grid, **←** goes back to that
grid. After you move between photos, **←** goes back to the photo that you looked at just before.
**→** replays a step that you took back.

The label beside the arrows names what is open (`Shoot 1 / DSC_0042.NEF`), because the tree row can
be scrolled out of sight. The tooltip of each arrow names where it leads.

The **Go** menu holds the same two moves with their keyboard shortcuts. The shortcuts are
++cmd+bracket-left++ / ++cmd+bracket-right++ on macOS and ++alt+left++ / ++alt+right++ on other
systems. The menu also holds **Enclosing Folder** (++cmd+up++ / ++alt+up++), which opens the grid of
the folder that holds the open photo. This also works for a photo that you picked directly from the
tree, with no grid to go back to. A photo that you remove from the list also leaves the step
history, so ← never reopens a row that is gone.

### 3. Generate proposals

In the header, pick a **Provider** (Ollama, LM Studio, llama.cpp, or OpenAI) and a **Model**. Press
**Refresh** to get the models that the provider serves now. Then pick a model from the dropdown
instead of typing its name. The dropdown lists the likely vision-capable models first.
**Connection...** opens the configuration that rarely changes:

- A custom **Base URL**.
- A masked **API key**. If you leave it blank, the GUI falls back to the environment variable of the
    provider.
- **Test Connection**, which tests ExifTool and the model.

**Generate Selected** (in the bottom bar) then runs the model on the checked photos on a background
thread. It builds the same contextual prompt as the CLI (existing keywords, location, GPS, camera).

The results appear as they come in, and the status in the tree changes for each photo. A progress
bar in the bottom bar counts the finished photos up to the size of the batch (`12 / 40`). Next to
the bar, a clock shows the elapsed time. After the first photo is done, the clock also shows the
estimated time left (`2:30 elapsed · 8:10 left`). As a result, a long batch tells you how long it
still needs.

By default, the GUI caches the results (see [Configuration](#configuration)). If you run a batch
again after a crash, the GUI reuses the earlier answers for unchanged photos. It does not call the
model again for these photos. The same is true for a folder that you partly processed before. To
turn the cache off for the session, toggle **Settings > Cache AI Results**. For a one-time **Skip
Cache** run, use the arrow on either **Generate** button, or the right-click menu of a photo.

By default, the model writes titles, descriptions, and keywords in English. **Settings > Metadata
Language** changes that language with one click. Pick a language from the menu. To use a language
that is not in the menu, choose **Other...** and type any language name that the model understands.
The new name then joins the menu.

The choice applies from the next generation on. The GUI does not reuse cached results in the old
language. The GUI saves the choice to the configuration file as `output_language` under
`[inference]`, so CLI runs use it too. This choice is separate from **Settings > Language**, which
translates the interface itself.

Sometimes a model that you ask for one language answers in two. For example, a title comes back as
`Decorated陶笛 with Geometric Pattern`, where `Ocarina` can say the same thing. The GUI examines every
generated proposal for this problem. A photo with mixed writing systems gets three marks:

- An amber note over its editable fields, which names the words.
- A `⚠` in its **Status** column. Hover over the `⚠` to see the words.
- A count in the closing status line of the run.

To pull those photos out of a long list, use **Select > Check Only > Status > Mixed writing
systems**. The closing status line names this entry. The folder grid offers the same entry under
**Show** and marks each flagged thumbnail with an amber `!`.

The proposal is still yours to save. If the photo shows the text (a sign, a book cover, a printed
character), mixed script is right. Only you can tell. **Settings > Rewrite Mixed-Language Output**
makes the run ask the model again for those photos. Each of these photos costs one extra call. This
entry is off by default, and the note appears in both cases.

To generate a single photo again without a change to your selection, open the photo. Then press
**Generate This Photo** in the detail pane. You can also right-click the photo in the tree and
choose **Generate**. For a failed photo, this menu shows **Retry Generation** instead.

If several photos ended up `failed ✗`, click **Retry Failed** in the bottom bar. It runs the model
again on all of them at once. The button stays disabled while no photo has the `failed ✗` status.

If the model gets a photo wrong (a deer tagged as a boar), correct it with the **Hint for the AI**
field. This field is next to *Generate This Photo*. Type a note such as `The animal is a deer`. Then
press Enter (or the button) to generate the photo again.

The GUI sends the hint with the photo, as a note that the model must trust over its own reading of
the image. A photo with a hint always calls the model and never replays its cached result. The GUI
does not cache the corrected answer either. A one-off correction must not become the generic answer
that a later run of that photo without a hint replays.

Each photo keeps its own hint. The hint stays after you browse to other photos, and batch runs use
it too. The GUI never writes the hint to the file. The CLI equivalent for a whole run is
[`--hint`](cli-reference.md#inference).

To stop a run early, press **Cancel** (next to *Generate Selected*). The current photo finishes
first, because nothing can interrupt a model request in the middle of a call. Then the run stops.
The same button also cancels a save in progress.

The photos that did not start go back to `pending`. Of these, a photo that already had a proposal
goes back to `ready` instead. To resume these photos later, click **Generate Selected** again. Each
photo that the run already generated keeps its proposal.

A run that you plan to leave alone can write its photos itself: see
[Generate and save in one click](#generate-and-save-in-one-click).

### 4. Review, edit, and save

Click a photo to open it on the right. The detail pane shows two columns side by side: a read-only
**Existing** column and a **New (editable)** column.

- a **preview** (RAW files are decoded just like a real run). Click it to open the photo in its own
    window, large enough to judge what the model saw. Inside that window:
    - It starts fitted to the window.
    - **Zoom in** and **Zoom out** (or `+` / `-`, or Ctrl+scroll and the trackpad pinch) magnify up to
        8:1.
    - **Fit** (Ctrl+0) and **100%** (Ctrl+1) jump between the two scales. A double-click toggles
        between them.
    - Drag the photo to pan it.
    - **Full screen** (`F` or ++f11++) fills the display. In full screen, `Esc` leaves full screen.
        Otherwise, `Esc` closes the viewer.
- **Existing** and **New** Title, Description, and Keywords rows, lined up side by side. The **New**
    side is editable and starts from the proposal. Both keyword columns use the same sort order, so
    you can read them line by line. Each keyword hierarchy appears once, as its deepest chain, in
    the same `<` notation you type. That chain replaces a separate bare copy of a keyword it already
    covers. The description boxes grow to fit their content.
- a line under each **Existing** Title and Description. It names the file that value came from. A
    sidecar wins over the photo, so this line can show a value that differs from what the photo file
    still holds. For example, it can show `shadows "default" in the image file`, because some
    cameras write that placeholder into every photo. That raw value is what the command
    `exiftool photo.dng` prints on its own. Only a save with **Write To > The Photo Itself**
    replaces it.
- a **Copyright** row under Keywords. The **Existing** side shows the notice on the file. The
    **New** side shows the notice that a save writes on this photo, with the year that the photo was
    taken, or `(not written)`. Both sides are read-only. To change the notice, use **Settings >
    Copyright**.
- a collapsible **Keyword changes** section. Its header always summarizes what a save will do, for
    example `+3 / -1` or `no change`. Expand it to see a colored diff: green for added, red and
    struck-through for removed, grey for unchanged. The section also shows the resulting keyword
    **tree**, drawn with `tree`-style branch guides (`├─`/`└─`).

Keywords support hierarchy with `<` (specific to general), for example `Eagle<Bird<Animal`. The
summary, diff, and tree update live as you edit. Adjust anything, then press **Save This Photo** to
write just the open photo. Press **Save Selected** (bottom bar) to write every **checked** photo
that has a proposal. The save scope matches *Generate Selected*: the same checkboxes decide both.
Check every photo to save every photo.

If **Settings > Copyright** has anything to write (a name, a notice, or the rights flag), **Save
Selected** also writes it on the checked photos that you did not generate. On those photos, it does
not change the title, the description, or the keywords. Thus, to add your copyright to photos that
are already tagged, use **Select > Check Only > Status > No copyright**, and then press **Save
Selected**. You do not generate the photos again. To save a hand edit on a photo that you did not
generate, use **Save This Photo**.

The arrow on either **Save** button opens the save options. These options choose what every save
writes, and there is only one set of them. Both **Save** buttons share it, and so does the **Save
Options** submenu on either **Generate** button's arrow: it is the same menu again. Every one of
those buttons spells out the current choice in its tooltip.

- **Write Title**, **Write Description**, and **Write Keywords** (all on by default) are the GUI's
    equivalent of the CLI's `--no-write-title`, `--no-write-description`, and `--no-write-keywords`.
    Uncheck one to leave that field untouched on the photo. For example, uncheck **Write Keywords**
    to refresh only the title and description. A curated Lightroom keyword list then stays as it is.
    If you turn off a field, the GUI grays out its **Overwrite Existing** entry, because there is
    then nothing to write.

- **Overwrite Existing** holds one entry per field. Each entry decides what a save replaces on a
    photo that already has that field. A checked entry is the GUI's `--overwrite-keywords`,
    `--overwrite-title`, or `--overwrite-description`. An unchecked entry is `--preserve-keywords`,
    `--preserve-title`, or `--preserve-description`. **Keywords** is off by default, so new keywords
    merge with the old ones. Check it to replace them instead. **Title** and **Description** are on
    by default, because each field holds only one value. Turning either field on clears a camera's
    placeholder text. Uncheck one to keep the value the photo already has, and fill in only the
    photos that lack it. This way, a caption you wrote by hand survives a rerun.

- **Write To** picks where a save puts the metadata: the GUI's equivalent of the CLI's
    `--sidecar-mode`. The choices are:

    - **XMP Sidecar** (the default) writes a `.xmp` file next to every photo.
    - **The Photo Itself** writes into each image file.
    - **Sidecar for RAW, Photo Otherwise** writes one of each, matched to the file type. This suits a
        folder of RAW+JPEG pairs.
    - **Both, Photo and Sidecar** writes the photo *and* a sidecar. The photo then travels tagged, and
        the sidecar stays for your catalog.

    The menu has no control for `sidecar_overrides`, but a save follows it from the configuration
    file. If one is set, the menu shows it in a grayed-out line, and the tooltip of the Save buttons
    names it. See
    [Tag DNG files for Lightroom Classic](recipes.md#tag-dng-files-for-lightroom-classic).

- **Keep ExifTool Backup** (on by default) tells ExifTool to save the untouched file as `*_original`
    before it writes. This is the GUI's equivalent of the CLI's `--backup-xmp` and
    `--no-backup-xmp`. With this on, saving a few thousand photos leaves a full second copy of each
    one next to the original. If you keep your own backup somewhere else, uncheck it to write in
    place and skip that extra copy. Without such a backup, keep it on.

**Save Selected** writes on a background thread, like a generation run. The same progress bar and
elapsed and remaining clock track the batch, and the window keeps responding. **Cancel** stops the
save after it finishes the file it is currently writing. The photos it never reached stay `ready` to
save again. Writing a few thousand photos takes minutes, so the counter is how you tell that it is
working, not stuck.

You can edit and save a photo without generating a proposal first. The editable fields then start
from the existing values.

#### Generate and save in one click

A run you will not sit through can write its own photos. **Save After Generating**, on the arrow of
either **Generate** button, writes every photo a run produced, as soon as that run finishes. The
buttons then read **Generate and Save Selected** and **Generate and Save This Photo**, and their
tooltips spell out what that save writes.

Before it writes, the run does the same two things you do at the keyboard. First, it tries again on
the photos that **failed**. This is what **Retry Failed** does by hand: a model server that blinks
out for a minute otherwise costs you those photos, because nobody is there to press the button. Then
it asks again for the photos whose text came back in **two writing systems**, because a second
answer usually comes back in one. That second call skips the result cache, because the cached answer
is the one it is unhappy with. It stores the new answer instead, so a later run of the same folder
does not bring the mixed text back. A batch you started with **Skip Cache** leaves the cache alone
here too.

Each pass waits a few seconds before it starts. A failure usually means the server is busy or
restarting. Each pass also happens only once per batch, so a provider that stays down cannot loop
forever. Photos still failing keep their `failed ✗` row and are counted in the closing line:
`Saved 48 of 48 checked photos. 2 photos failed (Retry Failed to try again).` Photos that come back
mixed a second time keep their amber `⚠` and are saved as they are.

!!! note "When the foreign word is the right answer"

    Mixed script is not always wrong. If the photo itself shows text, such as a sign, a book cover, or a
    printed character, that mix is correct. Asking again can replace that word with an English one.
    Because nobody reviews it in this mode, that wrong answer is what gets written. If your library
    holds many such photos, leave **Save After Generating** off for them. Review them as usual instead,
    or look at the `⚠` rows afterward.

**Save After Generating** covers only the run you just started. A proposal left over from an earlier
run is still yours to review, and nothing else changes. The write itself is the same one that **Save
Selected** makes, down to the [save options](#4-review-edit-and-save) it uses. **Save Options** in
the **Generate** button's own menu is that same menu, so you can set what gets written before you
turn writing on. **Cancel** stops the write after the file already in progress, and **Undo Writes**
puts the batch back. If the save options write nothing at all, the window says so instead of
finishing an unattended run in silence.

**Settings > Save Settings as Defaults...** keeps the choice, as `save_after_generating` under
`[gui]`. An overnight import is then one click on the next launch too. The toggle is off by default,
because the window is built around reviewing before writing.

!!! tip "API keys: field or environment"

    The **Connection...** dialog has a masked **API key** field. Leave it blank to use the provider's
    environment variable instead (`OPENAI_API_KEY`, `LM_STUDIO_API_KEY`, `LLAMA_CPP_API_KEY`, or
    `OLLAMA_API_KEY`). This keeps the secret out of the app entirely. Or, type a key to use it for this
    session only. A typed key stays in memory for the run, and the GUI never writes it to disk, not even
    with **Save Settings as Defaults**. OpenAI requires a key. Local Ollama and LM Studio servers
    usually do not. To set it from the environment instead, launch with a command such as
    `OPENAI_API_KEY=sk-... photo-tagger gui`.

### 5. When a photo fails

A photo that the model cannot process is marked `failed ✗` in the status column. To find out why:

- Hover over the `failed ✗` cell to see a tooltip with the error.
- Or, open the photo. A red banner above the preview shows the reason, for example
    `model unreachable` or a decode error.

Once you fix the cause, click **Retry Failed** to re-run every failed photo. Common fixes are
starting the model server, fixing the URL, or freeing up memory. You can also right-click one photo
and choose **Retry Generation** to retry just that photo. A successful retry clears the banner and
flips the status back to `ready`.

For the full traceback behind a failure, use **Help > Open Logs**. The GUI writes a timestamped,
rotating log file to `~/.photo-tagger/logs/` on every run. This action opens that folder in your
file browser.

### Keyword rules: a vocabulary, and harmonized shoots

**Settings > Keyword Rules...** holds two configuration choices that decide which keywords a save
writes. Both stay set for the session, and both persist through *Save Settings as Defaults*, so a
CLI run picks up the same rules.

A **controlled vocabulary** restricts generated keywords to the terms in a keyword file. This is the
GUI's [`--vocabulary`](cli-reference.md#controlled-vocabulary). Choose a Lightroom keyword export
(`.txt` or `.csv`), or a plain list of terms and `Animal|Bird|Osprey` paths. The label under the
picker then reports how many keywords the file holds, or why the GUI refused the file.

The file does two things at once. First, the model sees it in the prompt, so the model prefers your
catalog's terms from the start. Second, the GUI snaps every generated keyword onto it afterward.
`ospreys` then becomes your `Osprey`, under your `Animal|Bird` hierarchy, no matter what the model
said.

Tick **Write only keywords the vocabulary covers** for
[`--vocabulary-strict`](cli-reference.md#controlled-vocabulary). With this on, the GUI drops
keywords that the file does not have, instead of writing them as they came from the model. The
status bar names the most frequent rejects, so you can grow the vocabulary on purpose instead of by
accident.

!!! note "The vocabulary is part of the cache key"

    Swapping vocabulary files starts a fresh cache slice. A run never reuses keywords from the old
    vocabulary file. Snapping happens *after* the cache lookup, so a vocabulary you choose today also
    applies to answers the cache already holds from yesterday.

**Split shoots after** N minutes turns on shoot harmonization. This is the GUI's
[`--session-gap`](cli-reference.md#sessions). Photos are grouped into shoots wherever the capture
time jumps by more than that gap. If a photo has no capture time, the GUI uses its file date
instead. Each shoot's keywords are then made to agree with each other: the spelling and the
hierarchy that most of the shoot used win for all of it. This is why forty frames of the same bird
stop arriving in the catalog as `Osprey`, `Ospreys`, and `Wildlife|Raptor|Osprey`.

The CLI holds a session's writes until every photo in it finishes analysis. The window writes
nothing until you press Save, so this pass runs over the **proposals** at the end of each
generation, before you review them. **Tools > Harmonize Shoots Now** runs the pass again, for
example after you edit some keywords by hand. Zero, shown as `off`, treats every photo on its own.

### Your copyright, on every save

**Settings > Copyright...** holds the fields no model can guess: who took the photo, and who owns
it. Fill in **Creator** and a **Copyright notice** and every save writes them, alongside whatever
the model produced. It is the GUI's
[`--artist` and `--copyright`](cli-reference.md#copyright-and-other-fields).

A notice is a template. `{year}` becomes the year of the photo, and `{artist}` becomes the name
above. One notice therefore dates a shoot from 2019 and one from last week correctly, in the same
batch. The year is the photo's own capture date. On a photo without one, `{year}` drops out. A
placeholder can also read any tag and fall back, as in `{EXIF:Artist|"Unknown"}`. The
[CLI reference](cli-reference.md#copyright-and-other-fields) lists the full syntax.

**Keep the name and notice a photo already has** is on by default. A photo with a creator keeps it,
and a photo with a notice keeps the notice, the XMP rights, and the rights flag. So Save Selected
never dates an old notice again. The Copyright row of the detail pane says *(keeps the notice it
has)* for such a photo. Uncheck the option to replace what the photos carry.

The panel under the fields lists the tags a save writes on a photo with no notice yet, and what each
one says for a real year. A template with a typo in it is visible there before anything is written.

The **Insert** row types the awkward parts for you: `©`, `{year}`, and `{artist}`. Each one lands in
the **Copyright notice** or the **XMP rights** field, whichever holds the cursor. If neither holds
it, the text goes into the notice. The buttons take no focus, so you can keep typing where you left
off.

**XMP rights** is the same notice for XMP alone. XMP uses UTF-8 and can carry a real `©`. EXIF
stores its copyright and artist fields as ASCII until Exif 3.0, so `(c)` is the safe spelling in the
notice. When a value needs more than ASCII, the dialog says so in amber, but writes it anyway: an
accent in a name is not a mistake. Leave **XMP rights** empty to use the notice as it stands.
**Rights flag** sets `XMP-xmpRights:Marked`, and *Leave as is* writes nothing at all.

The save menu's **Write Copyright** turns the whole block off for a while, without clearing the
fields. It is grayed out until there is something to write. When Title, Description, and Keywords
are all unchecked, a save still writes the copyright. That is how you stamp photos that you do not
want re-tagged. These choices persist through *Save Settings as Defaults*, into the configuration
file's `[copyright]` table, so CLI runs credit your photos the same way.

### Build a vocabulary from your own library

**Tools > Build Vocabulary...** writes the keyword file the strict vocabulary needs, out of the
keywords your photos already carry. It is the
[`photo-tagger vocabulary`](cli-reference.md#building-a-vocabulary) command with its flags as form
fields:

- **Read from** the photos in the list (read through ExifTool, so any application that writes XMP or
    IPTC counts, not only Lightroom), a Lightroom **keyword export**, or both.
- **Keep a keyword when** it is used at least *N* times. This is the single most useful knob: in a
    catalog that an AI was writing to, most keywords appear only once. The dialog also has a cap on
    the file's size and an option to keep keywords that contain digits. A separate option writes
    bare keywords without their hierarchies.
- **Organize with the model** (off by default) asks the model to settle two things that counting
    alone cannot: which keywords are synonyms of each other, and what hierarchy the list must have.
    It never decides what to keep, and it never invents a keyword. It needs a reachable provider,
    and uses the one in the header. **Requests at once** (1 to 16) sets how many parts of the list
    go to the model at the same time.
- **Write to** a vocabulary file, plus an optional **drop report**: a CSV naming every dropped
    keyword, its count, and the rule that cut it, which is what makes the thresholds tunable.

The build runs in the background, because counting a large library takes a while and organizing
takes model calls. When it finishes, the window offers to put the new file straight to work as the
active vocabulary. Nothing is written to your photos.

### Watch a folder

**Tools > Watch Folder...** is the import-time workflow. Point it at the folder that your card
reader, tethered capture, or sync client fills, and photos join the list as they arrive. Photos
already there are picked up first. It mirrors
[`photo-tagger watch`](cli-reference.md#watching-a-folder), including **Check every** and **Settle
for**. **Check every** sets how often the GUI lists the folder. **Settle for** sets how long a file
must sit unchanged before the GUI picks it up. This way, a photo still being copied is left alone
until the copy finishes.

Two toggles decide how far this goes. **Generate each new photo** (on by default) runs the model on
each arrival, so a proposal is waiting for you. **Save it too, without reviewing** (off by default)
writes each photo straight away instead. Saving is opt-in, because reviewing before writing is the
point of the window. Turn it on for an unattended import, where it behaves like the CLI command. A
photo that lands mid-run waits for the run already in flight, instead of starting a second one. The
menu entry becomes **Stop Watching** while a watch is running, and closing the window stops it.

### Be told when a run ends

You can start a batch of a few hundred photos and then leave it. **Settings > Notify When a Run
Ends** raises a desktop notification the moment a generation or a save finishes. It carries the same
line that the status bar shows, so the two never disagree. **Play a Sound With Notifications** adds
an audible alert on top of it. Both are off by default. **Save Settings as Defaults...** writes them
to the configuration file, where they are the same
[`notify` / `notify_sound`](cli-reference.md#end-of-batch-notification) keys that the CLI reads.

A run that you cancelled yourself raises no alert, because you are already at the keyboard. The
notification uses whatever your system already provides, so nothing appears, and nothing breaks, on
a machine that has none. It also pairs naturally with an unattended **Watch Folder** import.

### Rest a long run

A model at full tilt can draw more power than a laptop charger supplies. An overnight batch then
discharges the battery even while it stays plugged in, runs the machine hot, and leaves little power
for anything else. **Settings > Pacing...** spends part of the time not working instead: the battery
catches up, the fans wind down, and the average draw lands under what the charger gives.

**Work only** is the share of the time that a run can work. For example, `70` rests the run for
about three minutes of every ten. **Rest below** rests the run until the charge comes back, whenever
the battery falls that low. This is the one to reach for on a laptop, because it needs no wattage
numbers. It follows your screen brightness and whatever else is running on the machine. It only ever
rests while a charger stays plugged in, because on battery power a rest recharges nothing. **Resume
at** is the charge that ends the rest. It must be above **Rest below**, and `0` means 15 points
above it. **Work for** and **Rest at most** bound how often a rest happens and how long each one
lasts.

A run rests between photos, never in the middle of one. The status bar says why it stopped and for
how long. The batches of a **Watch Folder** share one work window, as in `photo-tagger watch`, so a
stream of small imports still rests. The time spent waiting for photos does not count as work.
Everything here is off by default, and each choice is kept for the session. **Save Settings as
Defaults...** writes it to the configuration file's [`[pacing]`](cli-reference.md#pacing) table,
which CLI runs read too.

### Undo what a save wrote

Every save records what it wrote, in a small journal. **Settings > Record Saves for Undo** (on by
default) controls this, and [`photo-tagger undo`](cli-reference.md#undoing-a-run) reads the same
journal. **Tools > Undo Writes...** lists every recorded run, newest first, from this window *and*
from the command line. Each entry shows the time it ran and how many files it wrote.

Pick a run and press **Preview** to see what undoing it changes, file by file, without touching
anything. Press **Undo** to do it. The GUI asks you to approve first. Sidecars that the run created
are deleted. Files that it overwrote are restored from their ExifTool `*_original` backup. If a file
changed since the run, the GUI leaves it alone, because that change is a later edit, and not this
run's to undo. **Also revert files changed since the run** overrides that behavior. A file that was
written with **Keep ExifTool Backup** unchecked cannot be restored at all, because there is no copy
of what it held. The dialog says so, for each such file.

Photos in the list whose writes were reverted go back to `ready`, and their proposal is still there
to save again. A reverted photo without a proposal goes back to `pending`. Their **Tagged** column
is scanned again too.

### Export a CSV report

**File > Export CSV Report...** writes a spreadsheet with **one row per photo in the list**. This is
the same report that the CLI's [`--csv-file`](cli-reference.md#csv-report) produces. Each row
gathers everything the GUI knows about a photo:

- its status.
- the working title, description, and keywords (the keywords as a save writes them, honoring
    **Overwrite Existing > Keywords**).
- the title, description, and keywords already on the file.
- the camera and location EXIF read at generation time.
- the per-photo token usage and timing.

Photos that you have not generated yet still get a row, with the generated and usage columns left
blank.

Any unsaved edits in the open photo are folded into its row first, so the export reflects exactly
what you see. If you omit a `.csv` suffix in the save dialog, the GUI adds one for you. The status
bar then reports how many photos it wrote. This is handy for reviewing a batch in a spreadsheet, or
for sharing the results without opening every photo.

## Configuration

The GUI reads the same TOML configuration file and environment variables as the CLI. It pre-fills
the provider, model, URL, extensions, and save options from them. It can also write that file.
**Settings > Save Settings as Defaults...** updates the configuration file already in effect, or
creates `~/.config/photo-tagger/config.toml`. It writes the current provider, model, URL, file
types, recursion, and save options. It also writes the keyword rules, the copyright fields, **Save
After Generating**, the pacing, and the switches of the **Settings** menu: telemetry, undo
recording, notifications, and the mixed-language rewrite. If **Settings > Copyright** holds a
template with an error, the GUI refuses to save and tells you to fix the template first, because the
CLI would refuse that template at every later run. The save **merges**: it keeps comments, ordering,
and anything in the configuration that the GUI does not manage, and it never writes the API key. For
everything that the GUI does not surface, for example the prompt file, sampling, workers, and
filters, use **Settings > Edit Config File...**. This opens the file in your editor. See
[Configuration](../getting-started/configuration.md) for the format.

Generated results are cached in the same SQLite format as the CLI's
[`--cache-file`](cli-reference.md). If the configuration sets `cache_file`, the GUI uses that path.
Otherwise, it uses `~/.photo-tagger/cache.sqlite`. **Settings > Cache AI Results** turns the cache
off for a session. A photo's right-click menu offers **Generate (Skip Cache)** for one-off fresh
results.

The window surfaces the most common options. From the rest of the `[inference]` table it reads one
setting: `jpeg_dimensions`, the size that the model is shown. Raising it is what lets a run read
small text in a photo. It costs time per photo and starts a fresh cache, so see
[Image size](cli-reference.md#image-size) first. The window always uses the built-in temperature,
frequency penalty, and JPEG quality. For those, and for the full set of flags, for example custom
prompts, date-range filters, and logging, use the [CLI](cli-reference.md).

### Telemetry

The GUI sends the same anonymous usage beacons as the CLI. See [Telemetry](../telemetry.md). The
first time telemetry is active, the GUI shows a dialog that tells what it sends. Press **Turn It
Off** there to stop it. Later, use **Settings > Send Anonymous Telemetry** to turn it on or off. The
GUI keeps the choice for the next launch.

### The Help menu

**Help** holds **Documentation** (opens these docs in your browser), **Test Connection** (the same
check as in **Connection...**), **Open Logs**, and **About Photo Tagger**.

## Limitations

If some photos have proposals that you did not save, closing the window asks you to approve first.

!!! warning

    Cancelling (or closing the window mid-run) stops at the next photo boundary. The photo **already in
    flight runs to completion** before the run halts, because a model request cannot be interrupted
    mid-call. With a slow model, that one photo can take a while.

- Loading a photo's preview and existing metadata is synchronous. Selecting a large RAW file can
    therefore pause briefly the first time. Results are cached afterward.
- On macOS, `./packaging/build_macos_app.sh` builds a double-clickable, unsigned `Photo Tagger.app`
    with PyInstaller. On other systems, launch the GUI with `photo-tagger gui`.

For headless machines, scripting, scheduling, or piping results into other tools, use the CLI with
[`--json`](cli-reference.md#display). The GUI is meant for interactive, local use.
