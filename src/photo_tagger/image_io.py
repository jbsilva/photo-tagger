"""In-memory image preparation: load (with rawpy fallback), resize, encode JPEG."""

from io import BytesIO
from typing import TYPE_CHECKING

import rawpy
from loguru import logger
from PIL import Image, ImageOps
from pydantic_ai import BinaryContent

from photo_tagger.config import DEFAULT_DIMENSIONS, DEFAULT_JPEG_QUALITY, NON_RAW_EXTENSIONS
from photo_tagger.deflate_dng import open_deflate_dng
from photo_tagger.errors import DngDecodeError


if TYPE_CHECKING:
    from pathlib import Path


def _open_image(image_path: Path, max_size: int = DEFAULT_DIMENSIONS) -> Image.Image:
    """
    Open an image with PIL, using rawpy unless the suffix is a format PIL handles natively.

    Returns a fully loaded, upright image, in whatever mode the file has (see :func:`_to_rgb`). The
    underlying file handle is closed before this function returns, so the caller can operate on the
    result without holding the source file open. *max_size* is a hint: a decoder may return a
    smaller image, but never one with a long side below it, unless the photo itself is smaller.
    """
    suffix = image_path.suffix.lower()
    if suffix in NON_RAW_EXTENSIONS:
        logger.info("skipping_rawpy_for_known_format", extension=suffix)
    elif (img := _open_raw(image_path, max_size)) is not None:
        return img

    logger.info("opening_image_with_pil", extension=suffix or "")
    # Use Image.open in a with-block so PIL closes the file handle eagerly even
    # when the worker is part of a thread pool. exif_transpose returns a copy
    # with rotation/mirror EXIF applied, so the model sees the image upright.
    with Image.open(image_path) as img:
        img.load()
        # Typed as returning None, which only happens with in_place=True; the guard is for the
        # type checker.
        transposed = ImageOps.exif_transpose(img)
        return transposed if transposed is not None else img


def _open_raw(image_path: Path, max_size: int) -> Image.Image | None:
    """
    Decode a RAW file with rawpy, or a DNG that LibRaw rejects with ours.

    None if neither can.
    """
    try:
        with rawpy.imread(str(image_path)) as raw:  # type: ignore[no-untyped-call]
            rgb = raw.postprocess()
    except Exception as exc:  # noqa: BLE001 - rawpy raises bare Exception subclasses
        logger.warning("rawpy_failed", error=str(exc))
    else:
        logger.info("image_opened_with_rawpy")
        # rawpy.postprocess() already applies camera orientation; return as-is.
        return Image.fromarray(rgb)

    if image_path.suffix.lower() != ".dng":
        return None
    # LibRaw inflates only floating-point DNG data, so it fails on the Insta360 X6 files.
    try:
        img = open_deflate_dng(image_path, max_size)
    except DngDecodeError as exc:
        logger.warning("deflate_dng_fallback_failed", error=str(exc))
        return None
    logger.info("image_opened_with_deflate_dng_fallback")
    return img


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB for the JPEG, compositing any transparency onto white rather than black."""
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if not has_alpha:
        logger.info("converting_image_to_rgb")
        return img.convert("RGB")

    logger.info("compositing_alpha_to_white")
    alpha = img.convert("RGBA")
    bg = Image.new("RGBA", alpha.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, alpha).convert("RGB")


def prepare_image_for_agent(
    image_path: Path,
    jpg_quality: int = DEFAULT_JPEG_QUALITY,
    max_size: int = DEFAULT_DIMENSIONS,
) -> BinaryContent:
    """
    Open an image, resize it, and encode it to a JPEG buffer ready for the model.

    Args:
        image_path: Path to the input image.
        jpg_quality: JPEG compression quality (1-100; recommended ~80).
        max_size: Maximum dimension in pixels for the resized JPEG.

    Returns:
        BinaryContent wrapping the in-memory JPEG bytes.
    """
    try:
        img = _to_rgb(_open_image(image_path, max_size))
        logger.info("resizing_image", max_size=max_size)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        logger.info("encoding_image_to_jpeg", quality=jpg_quality)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpg_quality)
        jpeg_bytes = buf.getvalue()
    except Exception as exc:
        logger.exception("image_preparation_failed", error=str(exc))
        raise

    logger.debug(
        "image_prepared_for_agent",
        width=img.width,
        height=img.height,
        size_kb=len(jpeg_bytes) // 1024,
    )
    return BinaryContent(data=jpeg_bytes, media_type="image/jpeg")
