"""Resolve user-supplied inputs (files, directories, skip lists) into a final image batch."""

import contextlib
import os
from itertools import chain
from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from pathlib import Path


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
    - Order is preserved and duplicates are removed.
    """
    pattern = "**/*" if recursive else "*"
    files_from_dirs: list[Path] = []
    files_explicit: list[Path] = []

    for path in inputs:
        path_resolved = path
        with contextlib.suppress(Exception):
            path_resolved = path.resolve()
        if path_resolved.is_dir():
            for ext in ext_set:
                files_from_dirs.extend(path_resolved.glob(f"{pattern}{ext}"))
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
        logger.error("no_valid_extensions_provided", raw_input=image_extensions)
        raise SystemExit(1)
    logger.debug("parsed_extensions", extensions=sorted(ext_set))

    if not inputs:
        logger.error(
            "no_inputs_provided",
            hint="Pass one or more --input/-i paths (files or directories)",
        )
        raise SystemExit(1)

    image_files = resolve_image_files(inputs, ext_set, recursive=recursive)
    if not image_files:
        logger.error(
            "no_image_files_found",
            inputs=[str(p) for p in inputs],
            recursive=recursive,
            extensions=sorted(ext_set),
        )
        raise SystemExit(1)

    logger.info("image_files_discovered", count=len(image_files))
    return image_files


def load_skip_list(skip_file: Path) -> set[str]:
    """Read a newline-delimited list of names or paths to skip."""
    try:
        content = skip_file.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("skip_file_read_failed", file=str(skip_file), error=str(exc))
        raise SystemExit(1) from exc

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
