"""
Which files a save writes for a photo: the photo itself, its XMP sidecar, or both.

The rules behind ``--sidecar-mode`` and ``--sidecar-override``, with what undo needs to know about
each file before the write. Kept apart from the ExifTool reads and writes in
:mod:`photo_tagger.metadata`: callers that only plan (undo, the GUI's save options, the CLI's
checks) need no ExifTool at all.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from photo_tagger.config import NON_RAW_EXTENSIONS, FileTypeTarget, SidecarMode, SidecarOverrides


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def is_raw_file(image_path: Path) -> bool:
    """
    Report whether *image_path* looks like a camera RAW file, by extension alone.

    Decided by exclusion: a suffix Pillow reads natively is not RAW, anything else is. An unknown
    extension counting as RAW is the cautious answer for the caller that matters here, the "raw"
    sidecar mode, which then leaves the file's own bytes alone.
    """
    return image_path.suffix.lower() not in NON_RAW_EXTENSIONS


def use_sidecar_for(image_path: Path, mode: SidecarMode) -> bool:
    """
    Report whether *image_path* gets an XMP sidecar under *mode*.

    True for "both", which also writes the photo: ask :func:`plan_writes` for the full answer.

    Examples:
        >>> use_sidecar_for(Path("/photos/image.dng"), "raw")
        True
        >>> use_sidecar_for(Path("/photos/image.jpg"), "raw")
        False
    """
    match mode:
        case "none":
            return False
        case "raw":
            return is_raw_file(image_path)
        case _:
            return True


def write_target(image_path: Path, *, use_sidecar: bool) -> Path:
    """
    Return the file :func:`write_metadata` will modify for *image_path*.

    Shared so callers that need to inspect the target before the write (the undo journal, which
    must know whether the sidecar already existed) cannot drift from the writer's own choice.

    Examples:
        >>> write_target(Path("/photos/image.cr3"), use_sidecar=True).name
        'image.xmp'
        >>> write_target(Path("/photos/image.cr3"), use_sidecar=False).name
        'image.cr3'
    """
    return image_path.with_suffix(".xmp") if use_sidecar else image_path


@dataclass(slots=True, frozen=True)
class WriteTarget:
    """One file a save writes for a photo."""

    path: Path
    # Whether this target is the sidecar, which decides the payload: a sidecar holds XMP only.
    use_sidecar: bool
    # Whether the file was there before the write. Undo needs it to tell "restore the backup" from
    # "delete what the run created", and it can only be known beforehand.
    existed: bool
    # Whether a *_original backup was there before the write. ExifTool keeps an existing backup and
    # writes the file anyway, so that backup holds an older state, not what this write replaced.
    had_backup: bool = False

    def made_backup(self, *, backup: bool) -> bool:
        """Say whether a write with backups set to *backup* left a backup of its own behind."""
        return backup and not self.had_backup


# What each --sidecar-override target writes, as use_sidecar values: photo before sidecar, the
# order metadata_targets reads them in.
_TARGET_SIDECARS: dict[FileTypeTarget, tuple[bool, ...]] = {
    "photo": (False,),
    "sidecar": (True,),
    "both": (False, True),
}
# The same keys by their spelling, so a parsed string comes out typed without a cast.
_TARGET_NAMES: dict[str, FileTypeTarget] = {name: name for name in _TARGET_SIDECARS}


def parse_sidecar_overrides(entries: Iterable[str] | None) -> dict[str, FileTypeTarget]:
    """
    Parse ``EXT=TARGET`` entries into overrides keyed by lowercase suffix.

    The extension takes any case and an optional dot, so ``DNG=photo`` and ``.dng=photo`` agree. A
    later entry for the same extension wins, like a repeated flag. Raises :class:`ValueError` on an
    entry that is not ``EXT=TARGET``, or that names the sidecar extension itself.

    Examples:
        >>> parse_sidecar_overrides(["DNG=photo", "tif=both"])
        {'.dng': 'photo', '.tif': 'both'}
    """
    overrides: dict[str, FileTypeTarget] = {}
    for entry in entries or ():
        extension, _, name = entry.partition("=")
        suffix = "." + extension.strip().lstrip(".").lower()
        target = _TARGET_NAMES.get(name.strip().lower())
        if target is None or suffix == ".":
            msg = (
                f"--sidecar-override {entry!r} must be EXT=TARGET, where TARGET is sidecar, "
                f"photo, or both. For example: dng=photo"
            )
            raise ValueError(msg)
        if suffix == ".xmp":
            msg = f"--sidecar-override {entry!r}: an XMP sidecar cannot get a sidecar of its own"
            raise ValueError(msg)
        overrides[suffix] = target
    return overrides


def _sidecar_choices(
    image_path: Path,
    mode: SidecarMode,
    overrides: SidecarOverrides | None,
) -> tuple[bool, ...]:
    """Return the use_sidecar value of each file a save writes, photo first."""
    if overrides and (target := overrides.get(image_path.suffix.lower())) is not None:
        return _TARGET_SIDECARS[target]
    return (False, True) if mode == "both" else (use_sidecar_for(image_path, mode),)


def plan_writes(
    image_path: Path,
    mode: SidecarMode,
    overrides: SidecarOverrides | None = None,
) -> list[WriteTarget]:
    """
    List the files a save under *mode* writes for *image_path*, as they are before the write.

    One target for every mode but "both", which writes the photo *and* a sidecar beside it: the
    photo travels with its metadata, and the sidecar is what a catalog edits without touching the
    photo again. An entry in *overrides* for the photo's extension wins over *mode*, so a DNG can
    carry its own metadata (as Lightroom Classic expects) while every other file gets a sidecar.

    Examples:
        >>> [t.path.name for t in plan_writes(Path("/photos/image.cr3"), "both")]
        ['image.cr3', 'image.xmp']
        >>> [t.path.name for t in plan_writes(Path("/photos/image.dng"), "all", {".dng": "photo"})]
        ['image.dng']
    """
    targets: list[WriteTarget] = []
    for use_sidecar in _sidecar_choices(image_path, mode, overrides):
        path = write_target(image_path, use_sidecar=use_sidecar)
        targets.append(
            WriteTarget(
                path=path,
                use_sidecar=use_sidecar,
                existed=path.exists(),
                had_backup=path.with_name(path.name + "_original").exists(),
            ),
        )
    return targets
