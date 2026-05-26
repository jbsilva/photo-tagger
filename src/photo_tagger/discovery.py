"""Resolve user-supplied inputs (files, directories, skip lists) into a final image batch."""

import contextlib
import os
import threading
from datetime import UTC, datetime
from itertools import chain
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.errors import DiscoveryError
from photo_tagger.metadata import find_tagged_images


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]


def parse_extensions(image_extensions: str) -> set[str]:
    """
    Normalize comma-separated extensions into a set like {".cr3", ".jpg"}.

    Examples:
        >>> sorted(parse_extensions("cr3, jpg ,PNG"))
        ['.PNG', '.cr3', '.jpg']

    """
    return {
        f".{ext.strip().lstrip('.')}"
        for ext in image_extensions.split(",")
        if ext.strip().lstrip(".")
    }


def _iter_directory_matches(
    directory: Path,
    casefolded_exts: set[str],
    *,
    recursive: bool,
) -> list[Path]:
    """
    Yield every file under *directory* whose suffix (casefolded) is in *casefolded_exts*.

    pathlib.Path.glob is case-sensitive on Linux and APFS-case-sensitive volumes, so
    matching ``--ext cr3`` against ``IMG_0001.CR3`` used to silently return nothing on
    those filesystems. Walking the directory once and comparing ``suffix.casefold()``
    gives us the case-insensitive behavior the ``--ext`` help text has always advertised.
    """
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return [p for p in iterator if p.is_file() and p.suffix.casefold() in casefolded_exts]


def resolve_image_files(
    inputs: list[Path],
    ext_set: set[str],
    *,
    recursive: bool,
) -> list[Path]:
    """
    Resolve provided inputs into a list of files.

    - Directories are expanded by extension (honouring --recursive).
    - Explicit files are accepted as is (the extension filter does not apply).
    - Extension matching is case-insensitive: ``--ext cr3`` accepts ``.cr3``,
      ``.CR3``, ``.Cr3`` etc. so a Canon-style naming convention works on
      Linux and case-sensitive APFS volumes too.
    - Order is preserved and duplicates are removed.
    """
    casefolded_exts = {ext.casefold() for ext in ext_set}
    files_from_dirs: list[Path] = []
    files_explicit: list[Path] = []

    for path in inputs:
        path_resolved = path
        with contextlib.suppress(OSError):
            path_resolved = path.resolve()
        if path_resolved.is_dir():
            files_from_dirs.extend(
                _iter_directory_matches(path_resolved, casefolded_exts, recursive=recursive),
            )
        elif path_resolved.is_file():
            files_explicit.append(path_resolved)
        else:
            logger.warning("input_not_file_or_dir", path=str(path))

    combined: list[Path] = []
    seen: set[str] = set()
    for f in chain(files_explicit, files_from_dirs):
        key = str(f.resolve()) if f.exists() else str(f)
        if key not in seen:
            combined.append(f)
            seen.add(key)
    return combined


def resolve_image_batch(
    inputs: list[Path] | None,
    image_extensions: str,
    *,
    recursive: bool,
) -> list[Path]:
    """Resolve and validate a batch of images, exiting if the inputs are unusable."""
    ext_set = parse_extensions(image_extensions)
    if not ext_set:
        msg = f"No valid extensions in {image_extensions!r}"
        logger.error("no_valid_extensions_provided", raw_input=image_extensions)
        raise DiscoveryError(msg)
    logger.debug("parsed_extensions", extensions=sorted(ext_set))

    if not inputs:
        msg = "No inputs provided. Pass one or more --input/-i paths (files or directories)"
        logger.error(
            "no_inputs_provided",
            hint="Pass one or more --input/-i paths (files or directories)",
        )
        raise DiscoveryError(msg)

    image_files = resolve_image_files(inputs, ext_set, recursive=recursive)
    if not image_files:
        msg = f"No image files found in {[str(p) for p in inputs]}"
        logger.error(
            "no_image_files_found",
            inputs=[str(p) for p in inputs],
            recursive=recursive,
            extensions=sorted(ext_set),
        )
        raise DiscoveryError(msg)

    logger.info("image_files_discovered", count=len(image_files))
    return image_files


def load_skip_list(skip_file: Path) -> set[str]:
    """Read a newline-delimited list of names or paths to skip."""
    try:
        content = skip_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("skip_file_read_failed", file=str(skip_file), error=str(exc))
        raise DiscoveryError(str(exc)) from exc

    entries = {
        stripped
        for line in content.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    }
    if entries:
        logger.info("skip_entries_loaded", count=len(entries), file=str(skip_file))
    else:
        logger.warning("skip_file_has_no_entries", file=str(skip_file))
    return entries


def _split_skip_keys(skip_entries: set[str]) -> tuple[set[str], set[str]]:
    """Split skip entries into name-only keys and full-path keys (both casefolded)."""
    name_keys = {
        entry.casefold() for entry in skip_entries if os.sep not in entry and "/" not in entry
    }
    path_keys = {entry.casefold() for entry in skip_entries if entry.casefold() not in name_keys}
    return name_keys, path_keys


def filter_skipped_files(
    image_files: list[Path],
    skip_entries: set[str],
) -> tuple[list[Path], int]:
    """Return (kept, skipped_count) after applying the skip list."""
    if not skip_entries:
        return image_files, 0

    name_keys, path_keys = _split_skip_keys(skip_entries)
    filtered: list[Path] = []
    skipped = 0
    for path in image_files:
        if path.name.casefold() in name_keys or str(path).casefold() in path_keys:
            logger.debug("skipping_file_from_list", file=str(path))
            skipped += 1
            continue
        filtered.append(path)
    return filtered, skipped


def apply_skip_file(image_files: list[Path], skip_file: Path | None) -> list[Path]:
    """Apply a skip-list file to a batch of resolved image paths."""
    if not skip_file:
        return image_files

    skip_entries = load_skip_list(skip_file)
    filtered, skipped = filter_skipped_files(image_files, skip_entries)
    if skipped:
        logger.info(
            "skip_list_applied",
            skipped=skipped,
            remaining=len(filtered),
            file=str(skip_file),
        )
    elif skip_entries:
        logger.warning("skip_list_matched_no_files", file=str(skip_file))
    return filtered


def _file_mtime_utc(path: Path) -> datetime:
    """Return the file's mtime as a UTC datetime."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _mtime_within_window(
    mtime: datetime,
    *,
    newer_than: datetime | None,
    older_than: datetime | None,
) -> bool:
    """Return True when *mtime* sits strictly inside the (newer_than, older_than) bounds."""
    if newer_than is not None and mtime <= newer_than:
        return False
    return not (older_than is not None and mtime >= older_than)


def apply_date_filter(
    image_files: list[Path],
    *,
    newer_than: datetime | None = None,
    older_than: datetime | None = None,
) -> list[Path]:
    """
    Keep only files whose mtime falls within the (newer_than, older_than) window.

    Both bounds are optional; either, both, or neither may be set. The comparison
    uses file mtime (which most cameras preserve through copy) because reading
    EXIF DateTimeOriginal off every image would mean another exiftool pass.

    Args:
        image_files: Files resolved by ``resolve_image_files``.
        newer_than: Drop files whose mtime is on or before this instant.
        older_than: Drop files whose mtime is on or after this instant.

    Returns:
        Filtered list preserving input order.

    """
    if newer_than is None and older_than is None:
        return image_files

    kept: list[Path] = []
    for path in image_files:
        try:
            mtime = _file_mtime_utc(path)
        except OSError as exc:
            logger.warning("date_filter_stat_failed", file=str(path), error=str(exc))
            continue
        if _mtime_within_window(mtime, newer_than=newer_than, older_than=older_than):
            kept.append(path)

    skipped = len(image_files) - len(kept)
    if skipped:
        logger.info(
            "date_filter_applied",
            skipped=skipped,
            remaining=len(kept),
            newer_than=newer_than.isoformat() if newer_than else None,
            older_than=older_than.isoformat() if older_than else None,
        )
    return kept


def apply_skip_tagged(
    image_files: list[Path],
    *,
    skip_tagged: bool,
    et: ExifToolHelper | None = None,
) -> list[Path]:
    """
    Drop images that already have keywords, a description, or a title.

    Useful when a folder mixes new shots with photos that have already been processed
    (by photo-tagger, Lightroom, or by hand) and the user does not want to redo the work.
    Pass *et* to reuse an already-open ExifToolHelper across the batch.
    """
    if not skip_tagged or not image_files:
        return image_files

    tagged = find_tagged_images(image_files, et=et)
    if not tagged:
        logger.info("skip_tagged_no_matches", checked=len(image_files))
        return image_files

    kept: list[Path] = []
    for path in image_files:
        if path in tagged:
            logger.debug("skipping_already_tagged_file", file=str(path))
        else:
            kept.append(path)
    logger.info("skip_tagged_applied", skipped=len(tagged), remaining=len(kept))
    return kept


def make_skip_list_appender(skip_file: Path | None) -> Callable[[Path], None] | None:
    """
    Build a callback that appends a filename to *skip_file* on each successful process.

    Existing entries are preserved; duplicates are not appended a second time. The file is
    created if it does not exist yet, so the same path can also be passed to
    ``--skip-from`` on later runs to short-circuit work that already completed.

    The returned callback is safe to invoke from worker threads: a per-callback lock
    serializes the membership check, the file write, and the seen-set update so we
    never get interleaved writes or duplicate lines under ``--workers > 1``.
    """
    if skip_file is None:
        return None

    seen: set[str] = set()
    if skip_file.exists():
        # Preload entries already on disk so re-runs do not duplicate them. Failures are
        # logged but non-fatal: the appender still works on a fresh in-memory set.
        try:
            content = skip_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "append_skip_file_unreadable_starting_fresh",
                file=str(skip_file),
                error=str(exc),
            )
        else:
            seen = {
                stripped
                for line in content.splitlines()
                if (stripped := line.strip()) and not stripped.startswith("#")
            }
    else:
        # Touch the parent directory check; let open() handle creation lazily on first write.
        logger.info("append_skip_file_will_be_created", file=str(skip_file))

    lock = threading.Lock()

    def append(image_path: Path) -> None:
        name = image_path.name
        with lock:
            if name in seen:
                return
            try:
                with skip_file.open("a", encoding="utf-8") as handle:
                    handle.write(name + "\n")
            except OSError as exc:
                logger.warning(
                    "append_skip_file_write_failed",
                    file=str(skip_file),
                    entry=name,
                    error=str(exc),
                )
                return
            seen.add(name)
        logger.debug("appended_to_skip_file", file=str(skip_file), entry=name)

    return append
