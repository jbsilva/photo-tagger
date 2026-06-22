"""Tests for image preparation helpers using synthetic in-memory images."""

from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

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
    """``thumbnail()`` shrinks the image when ``max_size`` is below either dimension."""
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


# ---------------------------------------------------------------------------
# rawpy paths
# ---------------------------------------------------------------------------


def test_open_image_uses_rawpy_for_raw_extension(tmp_path: Path) -> None:
    """A .cr3 suffix is handled by rawpy when rawpy.imread succeeds."""
    import numpy as np  # noqa: PLC0415

    fake_rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    raw_ctx = MagicMock()
    raw_ctx.__enter__ = MagicMock(return_value=raw_ctx)
    raw_ctx.__exit__ = MagicMock(return_value=False)
    raw_ctx.postprocess.return_value = fake_rgb

    src = tmp_path / "photo.cr3"
    src.write_bytes(b"raw-data")
    with patch("photo_tagger.image_io.rawpy") as mock_rawpy:
        mock_rawpy.imread.return_value = raw_ctx
        img = _open_image(src)

    assert img.size == (8, 4)  # numpy shape (h, w, c) -> PIL (w, h)
    mock_rawpy.imread.assert_called_once()


def test_open_image_falls_back_to_pil_when_rawpy_fails(tmp_path: Path) -> None:
    """If rawpy fails on a non-known extension, PIL is used as fallback."""
    src = tmp_path / "photo.dng"
    # Write a real small PNG so PIL can open it (extension doesn't matter for PIL).
    _save_png(src)
    with patch("photo_tagger.image_io.rawpy") as mock_rawpy:
        mock_rawpy.imread.side_effect = Exception("unsupported format")
        img = _open_image(src)
    assert img.mode in {"RGB", "RGBA"}


# ---------------------------------------------------------------------------
# Alpha compositing edge cases
# ---------------------------------------------------------------------------


def test_flatten_alpha_composites_la_mode_onto_white() -> None:
    """An LA (luminance + alpha) image gets flattened to RGB with white background."""
    img = Image.new("LA", (2, 2), (0, 0))
    flat = _flatten_alpha(img)
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)


def test_flatten_alpha_handles_palette_with_transparency() -> None:
    """A P-mode image with transparency info gets composited."""
    img = Image.new("P", (2, 2))
    img.info["transparency"] = 0
    flat = _flatten_alpha(img)
    assert flat.mode == "RGB"


# ---------------------------------------------------------------------------
# prepare_image_for_agent sizing
# ---------------------------------------------------------------------------


def test_prepare_image_for_agent_preserves_small_images(tmp_path: Path) -> None:
    """An image already smaller than max_size is not enlarged."""
    src = _save_png(tmp_path / "small.png", size=(4, 4))
    out = prepare_image_for_agent(src, max_size=100)
    decoded = Image.open(BytesIO(out.data))
    assert decoded.size == (4, 4)
