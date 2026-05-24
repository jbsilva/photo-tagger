"""Tests for image preparation helpers using synthetic in-memory images."""

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import ExifTags, Image
from pydantic_ai import BinaryContent

from photo_tagger.image_io import (
    _flatten_alpha,
    _open_image,
    prepare_image_for_agent,
)


if TYPE_CHECKING:
    from pathlib import Path


_RESIZE_TARGET = 16


def _save_png(path: Path, *, mode: str = "RGB", size: tuple[int, int] = (16, 16)) -> Path:
    img = Image.new(mode, size, color="red" if "RGB" in mode else 255)
    img.save(path, format="PNG")
    return path


def test_flatten_alpha_returns_rgb_for_opaque_image() -> None:
    """An RGB image is converted (a no-op) but never grows an alpha channel."""
    img = Image.new("RGB", (4, 4), color="blue")
    flat = _flatten_alpha(img)
    assert flat.mode == "RGB"


def test_flatten_alpha_composites_rgba_onto_white() -> None:
    """An RGBA image is composited so transparent pixels become white."""
    img = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    flat = _flatten_alpha(img)
    assert flat.mode == "RGB"
    # Fully transparent pixel should now be white.
    assert flat.getpixel((0, 0)) == (255, 255, 255)


def test_prepare_image_for_agent_returns_jpeg_binary(tmp_path: Path) -> None:
    """A PNG on disk goes in, JPEG bytes wrapped in BinaryContent come out."""
    src = _save_png(tmp_path / "tiny.png")
    out = prepare_image_for_agent(src, jpg_quality=70, max_size=8)
    assert isinstance(out, BinaryContent)
    assert out.media_type == "image/jpeg"
    assert out.data[:3] == b"\xff\xd8\xff"  # JPEG SOI marker


def test_prepare_image_for_agent_resizes_when_max_size_smaller(tmp_path: Path) -> None:
    """thumbnail() shrinks the image when max_size is below either dimension."""
    src = _save_png(tmp_path / "big.png", size=(64, 32))
    out = prepare_image_for_agent(src, max_size=_RESIZE_TARGET)
    decoded = Image.open(BytesIO(out.data))
    assert max(decoded.size) <= _RESIZE_TARGET


def test_prepare_image_for_agent_raises_on_missing_file(tmp_path: Path) -> None:
    """Missing files surface as exceptions, not silent empty buffers."""
    with pytest.raises(Exception):  # noqa: B017, PT011 - any IO error is acceptable here
        prepare_image_for_agent(tmp_path / "nope.png")


def _save_jpeg_with_orientation(path: Path, orientation: int) -> Path:
    """Write a 4x16 JPEG with the EXIF orientation tag set to *orientation*."""
    img = Image.new("RGB", (4, 16), color="red")
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    img.save(path, format="JPEG", exif=exif.tobytes())
    return path


def test_open_image_honors_exif_orientation_rotate_90(tmp_path: Path) -> None:
    """A JPEG flagged orientation=6 (rotate 90 CW) comes back transposed."""
    src = _save_jpeg_with_orientation(tmp_path / "rotated.jpg", orientation=6)
    img = _open_image(src)
    # Source is 4x16; after rotating 90 CW the dimensions swap to 16x4.
    assert img.size == (16, 4)


def test_open_image_passes_through_when_orientation_is_normal(tmp_path: Path) -> None:
    """An image flagged orientation=1 (normal) keeps its original dimensions."""
    src = _save_jpeg_with_orientation(tmp_path / "upright.jpg", orientation=1)
    img = _open_image(src)
    assert img.size == (4, 16)
