"""
Qt-free helpers and per-photo state for the desktop GUI.

Kept separate from :mod:`photo_tagger.gui` so the GUI's logic (expanding dropped paths, grouping
files for the tree, parsing the editable keyword field, building the keyword set to write) is plain
Python the test suite covers normally, with no display server and no dependency on the optional
PySide6 extra. ``gui.py`` is then just the widget and event-loop shell that wires these helpers to
Qt.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from photo_tagger.discovery import parse_extensions, resolve_image_files
from photo_tagger.keywords import merge_keywords
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


# Per-photo status values, shown as an icon/word in the tree.
PENDING = "pending"  # added, not yet generated
WORKING = "working"  # generation in flight
READY = "ready"  # a proposal is available to review
SAVED = "saved"  # written to the file
FAILED = "failed"  # generation or save failed

# A broad, common default for the GUI's folder-scan extensions. Each distinct extension is listed
# because matching is case-insensitive but not variant-aware (jpg does not cover jpeg).
DEFAULT_GUI_EXTENSIONS = "jpg,jpeg,png,dng,cr3,nef,arw,heic,heif,tif,tiff,webp"

# Display labels for the provider combo box. Maps the internal name to a human spelling.
PROVIDER_LABELS = {"ollama": "Ollama", "lmstudio": "LM Studio", "openai": "OpenAI"}

# Substrings that hint a model is vision-capable, used to surface likely picks first.
_VISION_HINTS = (
    "vl",
    "vision",
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "cogvlm",
    "internvl",
    "pixtral",
    "gemma3",
    "smolvlm",
    "-v-",
)


@dataclass(slots=True)
class PhotoItem:
    """
    Mutable per-photo state shared between the file tree and the detail pane.

    ``existing_*`` holds what was read off the file; ``title``/``description``/ ``keywords`` are the
    editable working copy that :func:`keywords_to_save` and the Save action write. The working copy
    is seeded from a :class:`Proposal` (or from the existing values when the user opens a file
    without generating).
    """

    path: Path
    selected: bool = True
    status: str = PENDING
    error: str = ""
    loaded: bool = False
    existing_title: str | None = None
    existing_description: str | None = None
    existing_keywords: KeywordSet = field(default_factory=KeywordSet)
    existing_sources: list[str] = field(default_factory=list)
    sources_read: bool = False
    has_proposal: bool = False
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Proposal:
    """One file's AI proposal plus the existing metadata read alongside it."""

    path: Path
    existing_title: str | None
    existing_description: str | None
    existing_keywords: KeywordSet
    title: str
    description: str
    keywords: list[str]


def expand_inputs(
    paths: Iterable[Path],
    image_extensions: str,
    *,
    recursive: bool,
) -> list[Path]:
    """
    Expand dropped files and folders into a de-duplicated list of image files.

    Reuses the CLI's discovery: directories are walked and extension-filtered, explicit files are
    kept as is, and order is preserved. An empty extension string yields no files rather than
    raising.
    """
    ext_set = parse_extensions(image_extensions)
    if not ext_set:
        return []
    return resolve_image_files(list(paths), ext_set, recursive=recursive)


def new_paths(existing: Iterable[Path], found: Iterable[Path]) -> list[Path]:
    """Return entries of *found* not already present in *existing* (order preserved)."""
    seen = set(existing)
    out: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def group_by_parent(paths: Iterable[Path]) -> list[tuple[Path, list[Path]]]:
    """Group file paths under their parent directory, first-seen order preserved."""
    groups: dict[Path, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent, []).append(path)
    return list(groups.items())


def parse_keyword_lines(text: str) -> list[str]:
    """Parse the editable keyword field (one keyword per line) into a clean list."""
    return [stripped for line in text.splitlines() if (stripped := line.strip())]


def keywords_to_text(keywords: list[str]) -> str:
    """Render keywords one per line for the editable text field."""
    return "\n".join(keywords)


def keywords_to_save(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
) -> KeywordSet:
    """
    Build the :class:`KeywordSet` to write from the edited keywords.

    Merges with the existing keywords unless *overwrite* is set, in which case the existing keywords
    are dropped first. Hierarchical entries (``Duck<Bird<Animal``) are parsed by
    :func:`merge_keywords` exactly as the CLI does.
    """
    base = KeywordSet() if overwrite else existing
    return merge_keywords(base, edited_keywords)


def apply_proposal(item: PhotoItem, proposal: Proposal) -> None:
    """Fill *item*'s existing metadata and seed its editable copy from *proposal*."""
    item.existing_title = proposal.existing_title
    item.existing_description = proposal.existing_description
    item.existing_keywords = proposal.existing_keywords
    item.loaded = True
    item.title = proposal.title
    item.description = proposal.description
    item.keywords = list(proposal.keywords)
    item.has_proposal = True
    item.status = READY
    item.error = ""


def deselect_paths(items: dict[str, PhotoItem], paths: Iterable[Path]) -> int:
    """
    Uncheck the items whose path is in *paths*; return how many actually changed.

    This is how the GUI "skips" photos: rather than dropping them from the list like the CLI does,
    it just deselects them, so the user still sees what was skipped and can re-check any of it.
    Items already unchecked (or not in the list) are left alone and not counted, so the returned
    tally is the number of newly-skipped photos.
    """
    changed = 0
    for path in paths:
        item = items.get(str(path))
        if item is not None and item.selected:
            item.selected = False
            changed += 1
    return changed


@dataclass(slots=True)
class FolderNode:
    """A folder in the file tree, with its display label, subfolders, and files."""

    path: Path
    label: str
    folders: list[FolderNode]
    files: list[Path]


def _raw_dirs(paths: list[Path]) -> dict[Path, FolderNode]:
    """Build a directory -> node map linking every file's ancestor chain."""
    nodes: dict[Path, FolderNode] = {}
    suborder: dict[Path, list[Path]] = {}

    def ensure(directory: Path) -> None:
        if directory in nodes:
            return
        nodes[directory] = FolderNode(path=directory, label=directory.name, folders=[], files=[])
        suborder[directory] = []
        parent = directory.parent
        if parent != directory:  # stop at the filesystem root, whose parent is itself
            ensure(parent)
            suborder[parent].append(directory)

    for file in paths:
        ensure(file.parent)
        nodes[file.parent].files.append(file)
    for parent, subs in suborder.items():
        nodes[parent].folders = [nodes[s] for s in subs]
    return nodes


def _first_significant(node: FolderNode) -> FolderNode:
    """Descend through single-child, file-less folders to the first meaningful node."""
    while not node.files and len(node.folders) == 1:
        node = node.folders[0]
    return node


def _display_node(node: FolderNode, parent_path: Path | None) -> FolderNode:
    """Re-label *node* relative to its display parent and collapse its child chains."""
    label = str(node.path) if parent_path is None else str(node.path.relative_to(parent_path))
    folders = [_display_node(_first_significant(child), node.path) for child in node.folders]
    return FolderNode(path=node.path, label=label, folders=folders, files=list(node.files))


def build_tree(paths: Iterable[Path]) -> list[FolderNode]:
    """
    Group file paths into a nested folder tree for the GUI.

    Leading single-child directory chains are collapsed (``a/b/c`` shows as one node when only ``c``
    holds files), subfolders nest under their parent, and disjoint roots become separate top-level
    nodes. Each node's ``label`` is its path relative to its display parent (the absolute path for a
    top-level node).
    """
    paths = list(paths)
    if not paths:
        return []
    nodes = _raw_dirs(paths)
    tops: list[FolderNode] = []
    for root in (d for d in nodes if d.parent == d):
        node = _first_significant(nodes[root])
        if node.path.parent == node.path and not node.files and len(node.folders) > 1:
            # The filesystem root is just a container for disjoint trees; promote each branch.
            tops.extend(_first_significant(child) for child in node.folders)
        else:
            tops.append(node)
    return [_display_node(top, None) for top in tops]


def descendant_files(node: FolderNode) -> list[Path]:
    """Return every file under *node*, recursing into subfolders."""
    files = list(node.files)
    for folder in node.folders:
        files.extend(descendant_files(folder))
    return files


def paths_under(paths: Iterable[Path], folder: Path) -> list[Path]:
    """Return the paths that live under *folder* (at any depth), order preserved."""
    return [path for path in paths if path.is_relative_to(folder)]


def rank_vision_models(model_ids: Iterable[str]) -> list[str]:
    """
    Order model ids with likely vision-capable ones first, keeping all of them.

    The provider listings do not reliably flag modality, so this is a name heuristic (``vl``,
    ``vision``, ``llava``, ...) used only to surface probable picks; nothing is hidden, so a model
    the heuristic misses is still selectable.
    """
    model_ids = list(model_ids)
    likely = [m for m in model_ids if any(hint in m.lower() for hint in _VISION_HINTS)]
    likely_set = set(likely)
    others = [m for m in model_ids if m not in likely_set]
    return likely + others


def format_existing_keywords(keywords: KeywordSet) -> str:
    """Render existing keywords for the read-only panel: flat list plus any hierarchy."""
    sections: list[str] = []
    if keywords.subject:
        sections.append(", ".join(keywords.subject))
    if keywords.hierarchical:
        sections.append("Hierarchy:\n" + "\n".join(keywords.hierarchical))
    return "\n".join(sections)


def hierarchy_preview(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
) -> str:
    """Render the Lightroom hierarchy that saving the edited keywords would produce."""
    return "\n".join(keywords_to_save(existing, edited_keywords, overwrite=overwrite).hierarchical)


# Diff states for a keyword when comparing the existing flat subjects to what a save writes.
ADDED = "added"
REMOVED = "removed"
UNCHANGED = "unchanged"


def keyword_diff(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
) -> list[tuple[str, str]]:
    """
    Compare existing flat keywords to the result of saving the edited keywords.

    Returns ``(keyword, state)`` pairs where state is :data:`ADDED`, :data:`REMOVED`, or
    :data:`UNCHANGED`. The keywords that will be written come first in write order, then any that
    would be dropped (only possible with *overwrite*). Comparison is case-insensitive.
    """
    result = keywords_to_save(existing, edited_keywords, overwrite=overwrite).subject
    existing_folds = {kw.casefold() for kw in existing.subject}
    result_folds = {kw.casefold() for kw in result}
    diff = [(kw, UNCHANGED if kw.casefold() in existing_folds else ADDED) for kw in result]
    diff += [(kw, REMOVED) for kw in existing.subject if kw.casefold() not in result_folds]
    return diff


def status_summary(items: Iterable[PhotoItem]) -> str:
    """One-line counts for the status bar: selected, generated, saved, failed."""
    items = list(items)
    selected = sum(1 for i in items if i.selected)
    generated = sum(1 for i in items if i.has_proposal)
    saved = sum(1 for i in items if i.status == SAVED)
    failed = sum(1 for i in items if i.status == FAILED)
    return (
        f"{len(items)} files · {selected} selected · {generated} generated "
        f"· {saved} saved · {failed} failed"
    )
