"""The files the window lists: dropped paths, file types, and the file manager's reveal command."""

from typing import TYPE_CHECKING

from photo_tagger.discovery import parse_extensions, resolve_image_files
from photo_tagger.i18n import _
from photo_tagger.programs import find_program


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


# A broad, common default for the GUI's folder-scan extensions. Each distinct extension is listed
# because matching is case-insensitive but not variant-aware (jpg does not cover jpeg).
DEFAULT_GUI_EXTENSIONS = "jpg,jpeg,png,dng,cr3,nef,arw,heic,heif,tif,tiff,webp"


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


def _extension(path: Path) -> str:
    """Return a path's extension, lowercased and without the dot ("" when it has none)."""
    return path.suffix.lstrip(".").lower()


def reveal_label(platform_name: str) -> str:
    """Name the OS file browser for the context-menu action ("Reveal in Finder" on macOS)."""
    if platform_name == "darwin":
        return _("Reveal in Finder")
    if platform_name.startswith("win"):
        return _("Show in Explorer")
    return _("Show in File Manager")


def reveal_command(path: Path, platform_name: str) -> list[str] | None:
    """
    Return the argv that reveals *path* selected in the OS file browser, or None.

    None means the caller should fall back to opening the containing folder instead: the platform
    has no standard "reveal" command (Linux file managers vary), or the one it does have is not on
    PATH. The program is resolved here rather than left to the OS, which on Windows searches the
    current working directory (whichever folder the user last opened photos from) before PATH.
    """
    if platform_name == "darwin":
        argv = ["open", "-R", str(path)]
    elif platform_name.startswith("win"):
        # Explorer's /select switch takes the path in the same argument, comma-separated.
        argv = ["explorer", f"/select,{path}"]
    else:
        return None
    program = find_program(argv[0])
    return [program, *argv[1:]] if program is not None else None


def file_type_label(path: Path) -> str:
    """
    Label a photo's Type column: its extension, plus "+xmp" when an XMP sidecar sits beside it.

    The sidecar check is a plain filesystem stat, so this is cheap enough to run for every file at
    add time (no exiftool involved).
    """
    suffix = _extension(path)
    return f"{suffix}+xmp" if path.with_suffix(".xmp").exists() else suffix


# Image formats the picker always offers as hints, regardless of the user's configured file types.
# Format names are proper nouns, so they are not translated.
KNOWN_IMAGE_FORMATS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("JPEG", ("jpg", "jpeg")),
    ("PNG", ("png",)),
    ("Camera Raw", ("arw", "cr2", "cr3", "dng", "nef", "orf", "raf", "rw2")),
    ("HEIF", ("heic", "heif")),
    ("TIFF", ("tif", "tiff")),
    ("WebP", ("webp",)),
)


def _glob_patterns(extensions_without_dot: Iterable[str]) -> str:
    return " ".join(f"*.{ext}" for ext in extensions_without_dot)


def file_dialog_name_filters(extensions: str) -> list[str]:
    """
    Build Qt open-dialog name filters for the Add Photos picker.

    The first entry (the dialog's default) covers the user's configured file types, so the picker
    highlights what the pipeline will actually accept. The known image formats follow as hints,
    Photoshop-style, and an "All files" escape hatch keeps unusual formats openable.
    """
    filters: list[str] = []
    configured = sorted({ext.lstrip(".").lower() for ext in parse_extensions(extensions)})
    if configured:
        label = _("Your file types ({patterns})")
        filters.append(label.format(patterns=_glob_patterns(configured)))
    known = sorted({ext for _name, exts in KNOWN_IMAGE_FORMATS for ext in exts})
    filters.append(_("All known image formats ({patterns})").format(patterns=_glob_patterns(known)))
    filters.extend(f"{name} ({_glob_patterns(exts)})" for name, exts in KNOWN_IMAGE_FORMATS)
    filters.append(_("All files (*)"))
    return filters
