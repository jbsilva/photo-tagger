"""Generate runs: their options, the extra passes, and the status-bar lines about them."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from photo_tagger.cache import build_cache_namespace
from photo_tagger.config import (
    DEFAULT_DIMENSIONS,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_SETTLE_SECONDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_USER_PROMPT,
)
from photo_tagger.gui_state.files import (
    DEFAULT_GUI_EXTENSIONS,
)
from photo_tagger.gui_state.items import (
    FAILED,
    READY,
    SAVED,
    PhotoItem,
)
from photo_tagger.gui_state.text import (
    format_duration,
)
from photo_tagger.i18n import _, ngettext
from photo_tagger.vocabulary import Vocabulary


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from photo_tagger.pacing import (
        Pacer,
        PacingPlan,
    )
    from photo_tagger.vocabulary import Vocabulary


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    """
    How a Generate run behaves: what it asks the model for, and how it paces and caches.

    The window's counterpart to the pipeline's ProcessingOptions. The worker takes the connection
    (provider, model, server, key, cache file) on its own and everything else as this one value.
    """

    output_language: str = DEFAULT_OUTPUT_LANGUAGE
    # A path (as str) to the photographer's note for that photo; paths without one need no entry.
    hints: Mapping[str, str] = field(default_factory=dict)
    # Listed in the prompt and snapped onto afterwards, exactly as the CLI's --vocabulary does.
    vocabulary: Vocabulary | None = None
    vocabulary_strict: bool = False
    repair_language: bool = False
    # Rests the run between work windows. None works flat out.
    pacing: PacingPlan | None = None
    # A pacer to carry on with instead of a fresh one: a watch hands its batches one, as the CLI
    # does, so a work window spans the batches and short ones still rest.
    pacer: Pacer | None = None
    # The longest edge of the JPEG the model is shown.
    jpeg_dimensions: int = DEFAULT_DIMENSIONS
    # Read past the cache and store what comes back: the cached answer is the one being replaced.
    refresh: bool = False


def gui_cache_namespace(
    model: str,
    output_language: str,
    user_prompt: str = DEFAULT_USER_PROMPT,
    jpeg_dimensions: int = DEFAULT_DIMENSIONS,
) -> str:
    """
    Build the cache namespace for GUI runs: the model plus the settings a run was made with.

    The GUI runs the agent with the library's sampling defaults, so those (not the CLI flags) key
    its entries, alongside *jpeg_dimensions*, which is the size the window sends to the model. The
    metadata language rides along so switching it never replays results generated in the old
    language, and so does the prompt, which carries the controlled vocabulary when there is one.
    """
    return build_cache_namespace(
        model,
        user_prompt=user_prompt,
        temperature=DEFAULT_TEMPERATURE,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        jpeg_dimensions=jpeg_dimensions,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        output_language=output_language,
    )


def review_summary(flagged: int) -> str:
    """
    One line for the status bar naming how many photos in the run mix writing systems.

    Empty when none do. The line names the menu path that isolates them: a bare count sent people
    scrolling a list of hundreds for a row it had never said how to find. :func:`mark_mixed_script`
    marks those rows and :func:`foreign_terms_note` says which words, once you are on one.
    """
    if not flagged:
        return ""
    return ngettext(
        "{n} photo mixes writing systems (Select > Check Only > Status).",
        "{n} photos mix writing systems (Select > Check Only > Status).",
        flagged,
    ).format(n=flagged)


def failure_summary(failed: int) -> str:
    """
    One line for the status bar naming how many photos of a run did not generate.

    Empty when none did. A run that writes by itself ends on its save's tally, which counts only the
    photos it wrote, so without this line a batch left alone comes back reading as a clean success
    it was not. The line names the button that tries again.
    """
    if not failed:
        return ""
    return ngettext(
        "{n} photo failed (Retry Failed to try again).",
        "{n} photos failed (Retry Failed to try again).",
        failed,
    ).format(n=failed)


@dataclass(frozen=True, slots=True)
class WatchSettings:
    """
    What a folder watch was started with.

    ``generate`` runs the model on each photo as it lands (the point of watching at all) while
    ``save`` writes the result without review. Saving is opt-in on purpose: the window's whole
    workflow is review-before-write, and a watch that writes unattended is the CLI's job.
    """

    folders: tuple[Path, ...] = ()
    extensions: str = DEFAULT_GUI_EXTENSIONS
    recursive: bool = True
    interval: float = DEFAULT_INTERVAL_SECONDS
    settle: float = DEFAULT_SETTLE_SECONDS
    generate: bool = True
    save: bool = False


def watch_status_text(settings: WatchSettings, *, added: int) -> str:
    """Status line while a watch runs: where it is looking, and what it has picked up so far."""
    folders = ", ".join(folder.name or str(folder) for folder in settings.folders)
    if not added:
        return _("Watching {folders} for new photos...").format(folders=folders)
    return ngettext(
        "Watching {folders}: {n} photo added so far.",
        "Watching {folders}: {n} photos added so far.",
        added,
    ).format(folders=folders, n=added)


# The extra passes a run that writes by itself makes before it saves, in the order it makes them.
# Both are the window doing by itself what a person at the keyboard would do next: press Retry
# Failed, and generate the odd photo whose text came back in two writing systems.
RERUN_FAILED = "failed"


RERUN_MIXED_SCRIPT = "mixed-script"


EXTRA_PASSES = (RERUN_FAILED, RERUN_MIXED_SCRIPT)


def photos_for_pass(name: str, items: Iterable[PhotoItem]) -> list[PhotoItem]:
    """
    Pick the photos an extra pass re-runs.

    The failed ones, or the ones whose generated text mixes writing systems: a second answer often
    comes back in one script, the way a second attempt often succeeds.
    """
    if name == RERUN_FAILED:
        return [item for item in items if item.status == FAILED]
    return [item for item in items if item.foreign_terms]


def extra_pass_status(name: str, count: int) -> str:
    """Say what a run about to ask the model again for part of its batch is doing."""
    if name == RERUN_FAILED:
        return ngettext(
            "Retrying {n} photo that failed...",
            "Retrying {n} photos that failed...",
            count,
        ).format(n=count)
    return ngettext(
        "Retrying {n} photo that mixed writing systems...",
        "Retrying {n} photos that mixed writing systems...",
        count,
    ).format(n=count)


def photos_to_write(items: Iterable[PhotoItem]) -> list[PhotoItem]:
    """
    Pick the photos a save nobody reviews is to write: generated, and not written yet.

    The caller decides the scope. A run that chains a save passes its own batch, so a proposal left
    over from an earlier run is not swept up; a watch passes the whole list, because it has no
    batch, only whatever has landed.
    """
    return [item for item in items if item.has_proposal and item.status == READY]


def status_summary(items: Iterable[PhotoItem]) -> str:
    """One-line counts for the status bar: selected, generated, saved, failed."""
    items = list(items)
    selected = sum(1 for i in items if i.selected)
    generated = sum(1 for i in items if i.has_proposal)
    saved = sum(1 for i in items if i.status == SAVED)
    failed = sum(1 for i in items if i.status == FAILED)
    files = ngettext("{n} file", "{n} files", len(items)).format(n=len(items))
    return _(
        "{files} · {selected} selected · {generated} generated · {saved} saved · {failed} failed",
    ).format(
        files=files,
        selected=selected,
        generated=generated,
        saved=saved,
        failed=failed,
    )


def search_summary(shown: int, total: int) -> str:
    """
    Lead the status line with what an active search is hiding.

    Without this a filtered list looks like a list that lost photos, and the counts beside it (which
    are of the whole list, since a hidden photo is still processed) would not add up.
    """
    return ngettext(
        "Showing {shown} of {total} photo",
        "Showing {shown} of {total} photos",
        total,
    ).format(shown=shown, total=total)


def estimate_remaining(done: int, total: int, elapsed: float) -> float | None:
    """
    Estimate the seconds left, extrapolating from how long the finished items took.

    Returns None when there is nothing honest to say: before the first item finishes there is no
    rate yet, and once the batch is complete there is nothing left to wait for.
    """
    if done <= 0 or done >= total:
        return None
    return elapsed / done * (total - done)


def progress_timing_text(done: int, total: int, elapsed: float) -> str:
    """
    Build the readout shown next to the progress bar: time spent, plus the estimate of time left.

    The estimate joins in only once the first item has finished, so a run starts out showing bare
    elapsed time. That still tells the user the program is working, which is the point.
    """
    spent = format_duration(elapsed)
    remaining = estimate_remaining(done, total, elapsed)
    if remaining is None:
        return _("{elapsed} elapsed").format(elapsed=spent)
    return _("{elapsed} elapsed · {remaining} left").format(
        elapsed=spent,
        remaining=format_duration(remaining),
    )
