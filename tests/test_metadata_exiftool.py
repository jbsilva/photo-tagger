"""
Integration-ish tests for metadata reads and writes via a real exiftool binary.

The functions here invoke exiftool through pyexiftool, so they need the binary
on PATH. They write to real (small) JPEG files in a tmp directory.
"""

import shutil
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from photo_tagger.metadata import (
    metadata_targets,
    read_existing_keywords,
    read_gps_coordinates,
    read_location_tags,
    write_metadata,
)


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.skipif(
    shutil.which("exiftool") is None,
    reason="exiftool binary not available on PATH",
)


def _write_jpeg(path: Path) -> Path:
    """Save a 4x4 red JPEG so exiftool has a real file to act on."""
    buf = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="JPEG")
    path.write_bytes(buf.getvalue())
    return path


def test_metadata_targets_lists_jpeg_and_sidecar(tmp_path: Path) -> None:
    """Both the primary file and an existing .xmp sidecar appear in the targets list."""
    img = _write_jpeg(tmp_path / "img.jpg")
    sidecar = tmp_path / "img.xmp"
    sidecar.write_text("<?xml version='1.0'?><x:xmpmeta xmlns:x='adobe:ns:meta/'/>")
    assert metadata_targets(img) == [str(img), str(sidecar)]


def test_metadata_targets_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """A missing image returns an empty target list, not a crash."""
    assert metadata_targets(tmp_path / "ghost.jpg") == []


def test_write_and_read_round_trip_subject_and_hierarchy(tmp_path: Path) -> None:
    """Write subjects + hierarchy embedded into the JPEG and read them back."""
    img = _write_jpeg(tmp_path / "img.jpg")
    ok = write_metadata(
        img,
        {
            "subject": ["Beach", "Sunset"],
            "hierarchical": ["Animal|Bird"],
        },
        description="A small description.",
        title="A small title",
        backup=False,
        use_sidecar=False,
    )
    assert ok is True

    keywords = read_existing_keywords(img)
    assert "Beach" in keywords["subject"]
    assert "Sunset" in keywords["subject"]
    assert "Animal|Bird" in keywords["hierarchical"]


def test_write_metadata_returns_false_for_empty_payload(tmp_path: Path) -> None:
    """Nothing to write -> early False, no exiftool call."""
    img = _write_jpeg(tmp_path / "img.jpg")
    assert write_metadata(img, {}, use_sidecar=False) is False


def test_read_location_tags_returns_empty_for_unset_image(tmp_path: Path) -> None:
    """A freshly-written JPEG has no location tags."""
    img = _write_jpeg(tmp_path / "img.jpg")
    assert read_location_tags(img) == {}


def test_read_gps_coordinates_returns_empty_for_unset_image(tmp_path: Path) -> None:
    """A freshly-written JPEG has no GPS coordinates."""
    img = _write_jpeg(tmp_path / "img.jpg")
    assert read_gps_coordinates(img) == {}


def test_read_existing_keywords_handles_missing_file(tmp_path: Path) -> None:
    """A missing file returns the empty buckets, not an exception."""
    out = read_existing_keywords(tmp_path / "ghost.jpg")
    assert out == {"subject": [], "hierarchical": [], "weighted": []}
