"""
Decode the DNG files that LibRaw rejects: integer samples under Deflate compression.

The Insta360 X6 writes its raw photos as 16-bit integer Bayer data in Deflate (zlib) strips. DNG
allows that, but LibRaw, the decoder inside rawpy, inflates only floating-point data. So rawpy
raises ``LibRawDataError`` on each such file. This module reads the TIFF structure itself, inflates
the strips or tiles with zlib, and renders an sRGB image that is good enough for the model.

The image is an approximation. Each 2x2 Bayer block becomes one RGB pixel (no demosaicing), and the
decoder ignores the linearization table, the opcode lists (the lens shading gain maps), and the
crop. It applies the black and white levels, the as-shot white balance, the color matrix, and the
baseline exposure.

TODO(libraw): delete this module when LibRaw decodes integer Deflate data. The check that refuses it
is in ``deflate_dng_load_raw()`` in ``src/decoders/fp_dng.cpp`` ("Only float deflated supported").
LibRaw 0.22.3 (2026-10-26) still has it. rawpy bundles LibRaw, so a fix reaches us through a rawpy
release: ``test_rawpy_still_rejects_integer_deflate_dng`` fails on the Renovate update that brings
it. Then remove the fallback from ``image_io.py`` with this module.
"""

import struct
import zlib
from enum import IntEnum
from typing import TYPE_CHECKING, BinaryIO, Literal

import numpy as np
from PIL import Image

from photo_tagger.errors import DngDecodeError


if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


type _Ifd = dict[int, tuple[float, ...]]
type _Order = Literal["<", ">"]


class _Tag(IntEnum):
    """The TIFF, TIFF/EP, and DNG tags the decoder reads."""

    NEW_SUBFILE_TYPE = 254
    IMAGE_WIDTH = 256
    IMAGE_LENGTH = 257
    BITS_PER_SAMPLE = 258
    COMPRESSION = 259
    PHOTOMETRIC = 262
    STRIP_OFFSETS = 273
    ORIENTATION = 274
    SAMPLES_PER_PIXEL = 277
    ROWS_PER_STRIP = 278
    STRIP_BYTE_COUNTS = 279
    PREDICTOR = 317
    TILE_WIDTH = 322
    TILE_LENGTH = 323
    TILE_OFFSETS = 324
    TILE_BYTE_COUNTS = 325
    SUB_IFDS = 330
    SAMPLE_FORMAT = 339
    CFA_REPEAT_PATTERN_DIM = 33421
    CFA_PATTERN = 33422
    BLACK_LEVEL = 50714
    WHITE_LEVEL = 50717
    COLOR_MATRIX_1 = 50721
    COLOR_MATRIX_2 = 50722
    AS_SHOT_NEUTRAL = 50728
    BASELINE_EXPOSURE = 50730


_BYTE_ORDERS: dict[bytes, _Order] = {b"II": "<", b"MM": ">"}
_TIFF_MAGIC = 42
# TIFF field type -> (struct code, codes per value). A rational is a numerator and a denominator.
_FORMATS = {
    1: ("B", 1),
    3: ("H", 1),
    4: ("I", 1),
    5: ("I", 2),
    7: ("B", 1),
    8: ("h", 1),
    9: ("i", 1),
    10: ("i", 2),
    11: ("f", 1),
    12: ("d", 1),
    13: ("I", 1),
}
_RATIONALS = frozenset({5, 10})
# A tag value larger than this comes from a damaged count, not from a camera.
_MAX_VALUE_BYTES = 1 << 24
_MAX_PIXELS = 1 << 30
_CFA = 32803
_HORIZONTAL_PREDICTOR = 2
# (tag, default, accepted values, what the decoder needs). Each line is one way to reject a file.
_REQUIREMENTS = (
    (_Tag.COMPRESSION, 1, frozenset({8, 32946}), "Deflate compression"),
    (_Tag.SAMPLES_PER_PIXEL, 1, frozenset({1}), "one sample per pixel"),
    (_Tag.BITS_PER_SAMPLE, 1, frozenset({16}), "16-bit samples"),
    (_Tag.SAMPLE_FORMAT, 1, frozenset({1}), "unsigned integer samples"),
    (_Tag.PREDICTOR, 1, frozenset({1, _HORIZONTAL_PREDICTOR}), "no predictor or a horizontal one"),
    (_Tag.CFA_REPEAT_PATTERN_DIM, 0, frozenset({2}), "a 2x2 color filter pattern"),
)
_RGB = frozenset({0, 1, 2})
_PATTERN_SITES = 4
# Below this linear value, the sRGB curve is a straight line.
_SRGB_KNEE = 0.0031308
_SRGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
)
# Orientation tag -> the transpose that makes the image upright, as in PIL's exif_transpose.
_UPRIGHT = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def open_deflate_dng(path: Path, max_size: int) -> Image.Image:
    """
    Decode an integer Deflate DNG into an upright RGB image.

    The image comes back binned by a whole factor, as small as it can be while its long side stays
    at *max_size* or more. So a 120-megapixel file never exists in memory as floats.

    Raises:
        DngDecodeError: The file is another kind of DNG, or its data is damaged.
    """
    try:
        with path.open("rb") as file:
            return _decode(file, max_size)
    except (struct.error, zlib.error, ValueError) as exc:
        raise DngDecodeError(str(exc)) from exc


def _decode(file: BinaryIO, max_size: int) -> Image.Image:
    order, offset = _read_header(file)
    ifd0 = _read_ifd(file, order, offset)
    raw = _raw_ifd(file, order, ifd0)
    _check_supported(raw)
    mosaic = _read_mosaic(file, order, raw)
    pattern = tuple(int(color) for color in raw[_Tag.CFA_PATTERN])
    planes = _bin_planes(mosaic, pattern, _bin_factor(mosaic.shape, max_size))
    image = Image.fromarray(_render(planes, raw, ifd0))
    upright = _UPRIGHT.get(int(_first(ifd0, _Tag.ORIENTATION, 1)))
    return image.transpose(upright) if upright is not None else image


def _read(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        msg = "the file ends early"
        raise DngDecodeError(msg)
    return data


def _read_header(file: BinaryIO) -> tuple[_Order, int]:
    """Return the struct byte order and the offset of the first IFD."""
    header = _read(file, 8)
    order = _BYTE_ORDERS.get(header[:2])
    if order is None or struct.unpack(f"{order}H", header[2:4])[0] != _TIFF_MAGIC:
        msg = "not a TIFF file"
        raise DngDecodeError(msg)
    return order, struct.unpack(f"{order}I", header[4:])[0]


def _read_ifd(file: BinaryIO, order: _Order, offset: int) -> _Ifd:
    """Read the tags the decoder knows from one IFD, and skip the rest unread."""
    file.seek(offset)
    (count,) = struct.unpack(f"{order}H", _read(file, 2))
    entries = [struct.unpack(f"{order}HHI4s", _read(file, 12)) for _ in range(count)]
    return {
        tag: _read_value(file, order, kind, length, inline)
        for tag, kind, length, inline in entries
        if tag in _Tag and kind in _FORMATS
    }


def _read_value(
    file: BinaryIO,
    order: _Order,
    kind: int,
    length: int,
    inline: bytes,
) -> tuple[float, ...]:
    code, per_value = _FORMATS[kind]
    layout = f"{order}{length * per_value}{code}"
    size = struct.calcsize(layout)
    if size > _MAX_VALUE_BYTES:
        msg = f"a tag value of {size} bytes"
        raise DngDecodeError(msg)
    if size > len(inline):
        file.seek(struct.unpack(f"{order}I", inline)[0])
        values = struct.unpack(layout, _read(file, size))
    else:
        values = struct.unpack(layout, inline[:size])
    if kind in _RATIONALS:
        pairs = zip(values[::2], values[1::2], strict=True)
        return tuple(num / den if den else 0.0 for num, den in pairs)
    return values


def _first(ifd: _Ifd, tag: _Tag, default: float) -> float:
    values = ifd.get(tag)
    return values[0] if values else default


def _raw_ifd(file: BinaryIO, order: _Order, ifd0: _Ifd) -> _Ifd:
    """Find the full-size Bayer image: IFD0 for the Insta360, a SubIFD behind a preview for most."""
    subs = [_read_ifd(file, order, int(offset)) for offset in ifd0.get(_Tag.SUB_IFDS, ())]
    for ifd in (ifd0, *subs):
        full_size = _first(ifd, _Tag.NEW_SUBFILE_TYPE, 0) == 0
        if full_size and _first(ifd, _Tag.PHOTOMETRIC, 0) == _CFA:
            return ifd
    msg = "no full-size color filter array image"
    raise DngDecodeError(msg)


def _check_supported(ifd: _Ifd) -> None:
    for tag, default, accepted, needs in _REQUIREMENTS:
        values = ifd.get(tag, (default,))
        if any(value not in accepted for value in values):
            msg = f"the decoder needs {needs}, not {tag.name} {values}"
            raise DngDecodeError(msg)
    pattern = ifd.get(_Tag.CFA_PATTERN, ())
    if len(pattern) != _PATTERN_SITES or frozenset(pattern) != _RGB:
        msg = f"the decoder needs a red, green, and blue pattern, not {pattern}"
        raise DngDecodeError(msg)


def _layout(
    ifd: _Ifd,
    width: int,
    height: int,
) -> tuple[int, int, tuple[float, ...], tuple[float, ...]]:
    """Return the cell width, cell height, offsets, and byte counts of the tiles or the strips."""
    if _Tag.TILE_OFFSETS in ifd:
        return (
            int(_first(ifd, _Tag.TILE_WIDTH, 0)),
            int(_first(ifd, _Tag.TILE_LENGTH, 0)),
            ifd[_Tag.TILE_OFFSETS],
            ifd.get(_Tag.TILE_BYTE_COUNTS, ()),
        )
    rows = int(_first(ifd, _Tag.ROWS_PER_STRIP, height))
    offsets = ifd.get(_Tag.STRIP_OFFSETS, ())
    return width, min(rows, height), offsets, ifd.get(_Tag.STRIP_BYTE_COUNTS, ())


def _read_mosaic(file: BinaryIO, order: _Order, ifd: _Ifd) -> npt.NDArray[np.uint16]:
    """Inflate every strip or tile into one array of raw sensor values."""
    width = int(_first(ifd, _Tag.IMAGE_WIDTH, 0))
    height = int(_first(ifd, _Tag.IMAGE_LENGTH, 0))
    cell_width, cell_height, offsets, counts = _layout(ifd, width, height)
    if not 0 < width * height <= _MAX_PIXELS or cell_width <= 0 or cell_height <= 0:
        msg = f"an image of {width}x{height} in cells of {cell_width}x{cell_height}"
        raise DngDecodeError(msg)
    across = -(-width // cell_width)
    if len(offsets) != across * -(-height // cell_height) or len(counts) != len(offsets):
        msg = f"{len(offsets)} offsets and {len(counts)} byte counts for a {width}x{height} image"
        raise DngDecodeError(msg)
    samples = np.dtype(np.uint16).newbyteorder(order)
    predicted = _first(ifd, _Tag.PREDICTOR, 1) == _HORIZONTAL_PREDICTOR
    mosaic = np.empty((height, width), dtype=np.uint16)
    for index, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
        top, left = index // across * cell_height, index % across * cell_width
        rows, cols = min(cell_height, height - top), min(cell_width, width - left)
        data = _inflate(file, int(offset), int(count), rows * cell_width * samples.itemsize)
        cell = np.frombuffer(data, dtype=samples).reshape(rows, cell_width)[:, :cols]
        if predicted:
            # Each sample holds the difference to its left neighbor, and the sum wraps like uint16.
            cell = np.cumsum(cell, axis=1, dtype=np.uint16)
        mosaic[top : top + rows, left : left + cols] = cell
    return mosaic


def _inflate(file: BinaryIO, offset: int, count: int, size: int) -> bytes:
    """Inflate one cell, and stop at *size* bytes so a damaged stream cannot grow without end."""
    file.seek(offset)
    data = zlib.decompressobj().decompress(_read(file, count), size)
    if len(data) != size:
        msg = f"a strip or tile holds {len(data)} bytes, not {size}"
        raise DngDecodeError(msg)
    return data


def _bin_factor(shape: tuple[int, ...], max_size: int) -> int:
    """Return the largest bin that keeps the long side of the half-size image at *max_size*."""
    return max(1, max(shape) // 2 // max(max_size, 1))


def _bin_planes(
    mosaic: npt.NDArray[np.uint16],
    pattern: tuple[int, ...],
    factor: int,
) -> npt.NDArray[np.float32]:
    """Average each color of each block of 2x2 Bayer cells (times *factor*) into one pixel."""
    step = 2 * factor
    height, width = mosaic.shape[0] // step, mosaic.shape[1] // step
    if not height or not width:
        msg = f"an image of {mosaic.shape[1]}x{mosaic.shape[0]} is too small"
        raise DngDecodeError(msg)
    planes = np.zeros((height, width, 3), dtype=np.float32)
    for site, color in enumerate(pattern):
        cells = mosaic[site // 2 : height * step : 2, site % 2 : width * step : 2]
        blocks = cells.reshape(height, factor, width, factor)
        planes[..., color] += blocks.mean(axis=(1, 3), dtype=np.float32)
    planes /= np.bincount(pattern, minlength=3).astype(np.float32)
    return planes


def _render(planes: npt.NDArray[np.float32], raw: _Ifd, ifd0: _Ifd) -> npt.NDArray[np.uint8]:
    """Turn camera values into 8-bit sRGB: levels, white balance, color matrix, and exposure."""
    black = float(np.mean(raw.get(_Tag.BLACK_LEVEL, (0,))))
    white = _first(raw, _Tag.WHITE_LEVEL, 65535)
    if white <= black:
        msg = f"the white level {white} is not above the black level {black}"
        raise DngDecodeError(msg)
    camera = (planes - black) / (white - black)
    # Clip after the white balance, so a blown highlight becomes white and not pink.
    camera = np.clip(camera * _white_balance(ifd0), 0, 1)
    rgb = camera @ _camera_to_srgb(ifd0).T.astype(np.float32)
    rgb = np.clip(rgb * 2 ** _first(ifd0, _Tag.BASELINE_EXPOSURE, 0), 0, 1)
    encoded = np.where(rgb <= _SRGB_KNEE, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return (encoded * 255 + 0.5).astype(np.uint8)


def _white_balance(ifd0: _Ifd) -> npt.NDArray[np.float32]:
    """Return the gain of each channel, from the camera's neutral, with the smallest gain at 1."""
    neutral = ifd0.get(_Tag.AS_SHOT_NEUTRAL, ())
    if len(neutral) != len(_RGB) or min(neutral) <= 0:
        return np.ones(len(_RGB), dtype=np.float32)
    gains = 1 / np.asarray(neutral, dtype=np.float32)
    return np.asarray(gains / gains.min(), dtype=np.float32)


def _camera_to_srgb(ifd0: _Ifd) -> npt.NDArray[np.float64]:
    """
    Return the matrix from white-balanced camera values to linear sRGB.

    A DNG writer puts the warmer illuminant (often Standard A) first, so when there are two
    matrices, the second one is the daylight one. Each row of the camera-from-sRGB matrix is scaled
    to sum to 1, as dcraw does, so a white-balanced gray stays gray.
    """
    matrix = ifd0.get(_Tag.COLOR_MATRIX_2) or ifd0.get(_Tag.COLOR_MATRIX_1)
    if matrix is None or len(matrix) != len(_RGB) ** 2:
        return np.eye(len(_RGB))
    camera_from_srgb = np.reshape(matrix, (3, 3)) @ _SRGB_TO_XYZ
    camera_from_srgb /= camera_from_srgb.sum(axis=1, keepdims=True)
    return np.linalg.inv(camera_from_srgb)
