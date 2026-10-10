"""The folder tree, the place the right pane shows, and the Back and Forward history."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


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


def flat_labels(paths: Iterable[Path]) -> dict[Path, str]:
    """
    Label every path for the flat list: its filename, or its path below the folder they share.

    Flattening drops the folder each photo came from, which matters the moment two of them are
    called DSC_0042.NEF. One folder's worth of photos therefore keeps plain filenames, and a list
    spanning several shows just the part of the path that tells them apart. Paths with nothing in
    common (two drives on Windows) fall back to the full path, which always does.
    """
    paths = list(paths)
    if len({path.parent for path in paths}) <= 1:
        return {path: path.name for path in paths}
    try:
        root = os.path.commonpath([str(path) for path in paths])
    except ValueError:
        return {path: str(path) for path in paths}
    return {path: str(path.relative_to(root)) for path in paths}


def paths_under(paths: Iterable[Path], folder: Path) -> list[Path]:
    """Return the paths that live under *folder* (at any depth), order preserved."""
    return [path for path in paths if path.is_relative_to(folder)]


@dataclass(frozen=True, slots=True)
class Location:
    """One place the right-hand pane can show: a folder's thumbnail grid, or a photo's detail."""

    path: Path
    is_dir: bool


def location_crumb(location: Location) -> str:
    """
    Name a place for the Back/Forward hints and the pane's header.

    A photo carries its folder ("Shoot 1 / DSC_0042.NEF") because the file name alone is rarely
    enough to tell two frames of the same shoot apart.
    """
    if location.is_dir:
        return location.path.name or str(location.path)
    folder = location.path.parent.name
    return f"{folder} / {location.path.name}" if folder else location.path.name


# How many places back the history remembers. Deep enough that reviewing a shoot never runs out of
# trail, shallow enough that the whole thing stays a handful of strings.
HISTORY_LIMIT = 100


class NavigationHistory:
    """
    Where the user has been in the right-hand pane, so Back and Forward can retrace it.

    Browser semantics: visiting a place pushes the one being left onto the back trail and drops the
    forward trail, so what Back offers is always the route actually taken. Re-visiting the place
    already shown is ignored, which is what makes it safe to record every navigation, including the
    ones Back and Forward cause themselves.
    """

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        """Start an empty trail, remembering at most *limit* places to go back to."""
        self._limit = limit
        self._back: list[Location] = []
        self._forward: list[Location] = []
        self._current: Location | None = None

    @property
    def current(self) -> Location | None:
        """The place shown now, or None when the pane is showing nothing."""
        return self._current

    def visit(self, location: Location) -> None:
        """Record *location* as the place now shown, ending the forward trail."""
        if location == self._current:
            return
        if self._current is not None:
            self._back.append(self._current)
            if len(self._back) > self._limit:
                del self._back[0]  # one place at a time, so dropping the oldest is enough
        self._current = location
        self._forward.clear()

    def leave(self) -> None:
        """Note that the pane went empty, so Back returns to the last place rather than past it."""
        if self._current is not None:
            self._back.append(self._current)
            self._current = None

    def peek_back(self) -> Location | None:
        """Return where Back would go (for naming it), or None when nothing is behind."""
        return self._back[-1] if self._back else None

    def peek_forward(self) -> Location | None:
        """Return where Forward would go, or None until a Back has been taken."""
        return self._forward[-1] if self._forward else None

    def back(self) -> Location | None:
        """Step one place back and return it, or None when there is nothing behind."""
        return self._step(self._back, self._forward)

    def forward(self) -> Location | None:
        """Step one place forward and return it, or None when there is nothing ahead."""
        return self._step(self._forward, self._back)

    def _step(self, source: list[Location], sink: list[Location]) -> Location | None:
        if not source:
            return None
        if self._current is not None:
            sink.append(self._current)
        self._current = source.pop()
        return self._current

    def prune(self, keep: Callable[[Location], bool]) -> None:
        """
        Forget every place *keep* rejects, e.g. a photo just removed from the list.

        Dropping the place currently shown leaves the pane on nothing, so Back then lands on the
        last surviving place before it instead of skipping over it.
        """
        self._back = [location for location in self._back if keep(location)]
        self._forward = [location for location in self._forward if keep(location)]
        if self._current is not None and not keep(self._current):
            self._current = None


def navigation_shortcuts(platform_name: str) -> tuple[str, str, str]:
    """
    Return the Back, Forward, and enclosing-folder keys to use on *platform_name*.

    Each platform gets its own idiom: the browser keys on macOS (Qt reads a portable "Ctrl" as Cmd
    there) and the file manager keys elsewhere. Qt's own ``StandardKey.Back`` is Alt+Left on every
    platform, which on macOS is Option+Left, the word-left motion every text field needs, so the
    keys are chosen here rather than taken from Qt.
    """
    if platform_name == "darwin":
        return ("Ctrl+[", "Ctrl+]", "Ctrl+Up")
    return ("Alt+Left", "Alt+Right", "Alt+Up")
