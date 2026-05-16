"""In-memory image preparation: load (with rawpy fallback), resize, encode JPEG."""

from io import BytesIO
from typing import TYPE_CHECKING

import rawpy
from loguru import logger
from PIL import Image
from pydantic_ai import BinaryContent

from photo_tagger.config import DEFAULT_DIMENSIONS, DEFAULT_JPEG_QUALITY


if TYPE_CHECKING:
    from pathlib import Path


# File extensions that PIL handles natively. Anything outside this set is tried
# with rawpy first, falling back to PIL when rawpy can't recognize the format.
_NON_RAW_EXTS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".jpe",
        ".jp2",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
        ".psd",
        ".ico",
        ".ppm",
        ".pgm",
        ".pbm",
    },
)


def _open_image(image_path: Path) -> Image.Image:
    """Open an image with PIL, using rawpy unless the suffix is known non-RAW."""
    suffix = image_path.suffix.lower()
    if suffix in _NON_RAW_EXTS:
        logger.info("skipping_rawpy_for_known_format", extension=suffix)
    else:
        try:
            with rawpy.imread(str(image_path)) as raw:  # type: ignore[no-untyped-call]
                rgb = raw.postprocess()
            logger.info("image_opened_with_rawpy")
            return Image.fromarray(rgb)
        except Exception as exc:  # noqa: BLE001 - rawpy raises bare Exception subclasses
            logger.warning("rawpy_failed_falling_back_to_pil", error=str(exc))

    logger.info("opening_image_with_pil", extension=suffix or "")
    return Image.open(image_path)


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """Composite transparent pixels onto white so JPEG encoding is lossless-looking."""
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
        img = _flatten_alpha(_open_image(image_path))
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
