"""Tests for the Deflate DNG fallback, on small DNG files that the tests write themselves."""

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import rawpy
from PIL import Image
from rawpy._rawpy import LibRawDataError  # rawpy re-exports its errors only at runtime.

from photo_tagger.deflate_dng import open_deflate_dng
from photo_tagger.errors import DngDecodeError
from photo_tagger.image_io import _open_image


# TIFF field types.
_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _SRATIONAL = 1, 2, 3, 4, 5, 10
_CODES = {_BYTE: "B", _SHORT: "H", _LONG: "I", _RATIONAL: "I", _SRATIONAL: "i"}
_DEFLATE, _UNCOMPRESSED = 8, 1
_WHITE = 1000
_HALF_WHITE = _WHITE // 2
_RGGB = (0, 1, 1, 2)
# The pattern of the Insta360 X6: green and blue on the first row, red and green on the second.
_GBRG = (1, 2, 0, 1)
# The ColorMatrix1 of the Insta360 X6, and the AsShotNeutral of one of its photos.
_X6_MATRIX = (
    0.71875,
    -0.1484375,
    -0.09375,
    -0.609375,
    1.453125,
    0.109375,
    -0.15625,
    0.296875,
    0.4765625,
)
_X6_NEUTRAL = (0.44921875, 1, 0.5224609375)
_BIG = 10_000  # A max_size that never bins.

type _Order = Literal["<", ">"]
type _Value = tuple[int, Sequence[float] | bytes]
type _Entries = dict[int, _Value | None]


def _pack(order: _Order, kind: int, values: Sequence[float] | bytes) -> tuple[int, bytes]:
    """Return the TIFF count and the bytes of one field."""
    if isinstance(values, bytes):
        return len(values) + 1, values + b"\0"
    if kind in {_RATIONAL, _SRATIONAL}:
        flat = [part for value in values for part in (round(value * 1_000_000), 1_000_000)]
        return len(values), struct.pack(f"{order}{len(flat)}{_CODES[kind]}", *flat)
    return len(values), struct.pack(f"{order}{len(values)}{_CODES[kind]}", *values)


class _Tiff:
    """A TIFF writer that knows just enough for these tests: blobs, then IFDs that point at them."""

    def __init__(self, order: _Order) -> None:
        self.order = order
        self.data = bytearray(b"II*\0" if order == "<" else b"MM\0*") + bytes(4)

    def blob(self, payload: bytes) -> int:
        if len(self.data) % 2:
            self.data.append(0)
        offset = len(self.data)
        self.data += payload
        return offset

    def ifd(self, entries: dict[int, _Value]) -> int:
        fields = []
        for tag in sorted(entries):
            kind, values = entries[tag]
            count, payload = _pack(self.order, kind, values)
            if len(payload) > 4:  # noqa: PLR2004 - a value that fits in 4 bytes sits in the entry.
                payload = struct.pack(f"{self.order}I", self.blob(payload))
            fields.append(
                struct.pack(f"{self.order}HHI", tag, kind, count) + payload.ljust(4, b"\0"),
            )
        header = struct.pack(f"{self.order}H", len(fields))
        return self.blob(header + b"".join(fields) + bytes(4))

    def finish(self, first: int) -> bytes:
        self.data[4:8] = struct.pack(f"{self.order}I", first)
        return bytes(self.data)


@dataclass(frozen=True)
class _Dng:
    """One test DNG: the sensor values, how the file lays them out, and any tag to change."""

    mosaic: npt.NDArray[np.uint16]
    order: _Order = "<"
    tile: int | None = None
    rows_per_strip: int | None = None
    predictor: int = 1
    compression: int = _DEFLATE
    in_sub_ifd: bool = False
    raw_tags: _Entries = field(default_factory=dict)
    ifd0_tags: _Entries = field(default_factory=dict)

    def write(self, path: Path) -> Path:
        tiff = _Tiff(self.order)
        raw = _merge(self._raw_entries(tiff), self.raw_tags)
        ifd0: dict[int, _Value] = {
            50706: (_BYTE, (1, 4, 0, 0)),
            271: (_ASCII, b"Test"),
            272: (_ASCII, b"Camera"),
        }
        if not self.in_sub_ifd:
            ifd0.update(raw)
        else:  # IFD0 holds a 4x4 RGB preview, and the raw image is in its SubIFD.
            preview = bytes(4 * 4 * 3)
            ifd0.update(
                {
                    254: (_LONG, (1,)),
                    256: (_LONG, (4,)),
                    257: (_LONG, (4,)),
                    258: (_SHORT, (8, 8, 8)),
                    259: (_SHORT, (_UNCOMPRESSED,)),
                    262: (_SHORT, (2,)),
                    273: (_LONG, (tiff.blob(preview),)),
                    277: (_SHORT, (3,)),
                    279: (_LONG, (len(preview),)),
                    330: (_LONG, (tiff.ifd(raw),)),
                },
            )
        return _write(path, tiff.finish(tiff.ifd(_merge(ifd0, self.ifd0_tags))))

    def _raw_entries(self, tiff: _Tiff) -> dict[int, _Value]:
        height, width = self.mosaic.shape
        cell_width, cell_height, cells = self._cells()
        offsets = tuple(tiff.blob(cell) for cell in cells)
        counts = tuple(len(cell) for cell in cells)
        entries: dict[int, _Value] = {
            254: (_LONG, (0,)),
            256: (_LONG, (width,)),
            257: (_LONG, (height,)),
            258: (_SHORT, (16,)),
            259: (_SHORT, (self.compression,)),
            262: (_SHORT, (32803,)),
            277: (_SHORT, (1,)),
            317: (_SHORT, (self.predictor,)),
            33421: (_SHORT, (2, 2)),
            33422: (_BYTE, _RGGB),
            50717: (_LONG, (_WHITE,)),
        }
        if self.tile is None:
            entries.update(
                {273: (_LONG, offsets), 278: (_LONG, (cell_height,)), 279: (_LONG, counts)},
            )
        else:
            entries.update(
                {
                    322: (_SHORT, (cell_width,)),
                    323: (_SHORT, (cell_height,)),
                    324: (_LONG, offsets),
                    325: (_LONG, counts),
                },
            )
        return entries

    def _cells(self) -> tuple[int, int, list[bytes]]:
        height, width = self.mosaic.shape
        cell_width = self.tile or width
        cell_height = self.tile or self.rows_per_strip or height
        samples = np.dtype(np.uint16).newbyteorder(self.order)
        cells = []
        for top in range(0, height, cell_height):
            for left in range(0, width, cell_width):
                cell = self.mosaic[top : top + cell_height, left : left + cell_width]
                if self.tile is not None:  # A tile is padded at the edges of the image.
                    padding = ((0, cell_height - cell.shape[0]), (0, cell_width - cell.shape[1]))
                    cell = np.pad(cell, padding)
                if self.predictor == 2:  # noqa: PLR2004 - TIFF's horizontal predictor.
                    cell = np.diff(cell, axis=1, prepend=0).astype(np.uint16)
                data = cell.astype(samples).tobytes()
                cells.append(zlib.compress(data) if self.compression == _DEFLATE else data)
        return cell_width, cell_height, cells


def _merge(entries: dict[int, _Value], changes: _Entries) -> dict[int, _Value]:
    merged = {**entries, **changes}
    return {tag: value for tag, value in merged.items() if value is not None}


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _uniform(
    colors: tuple[int, int, int],
    *,
    pattern: tuple[int, ...] = _RGGB,
    size: tuple[int, int] = (8, 8),
) -> npt.NDArray[np.uint16]:
    """Return a mosaic in which each site of the color filter holds the value of its color."""
    height, width = size
    mosaic = np.empty((height, width), dtype=np.uint16)
    for site, color in enumerate(pattern):
        mosaic[site // 2 :: 2, site % 2 :: 2] = colors[color]
    return mosaic


def _pixels(img: Image.Image) -> set[tuple[int, ...]]:
    return {tuple(int(v) for v in pixel) for pixel in np.asarray(img).reshape(-1, 3)}


def _decode(dng: _Dng, tmp_path: Path, max_size: int = _BIG) -> Image.Image:
    return open_deflate_dng(dng.write(tmp_path / "photo.dng"), max_size)


def test_each_site_of_the_color_filter_goes_to_its_color(tmp_path: Path) -> None:
    """The Insta360 pattern starts with green and blue, so a swapped site would show at once."""
    mosaic = _uniform((_WHITE, 0, _HALF_WHITE), pattern=_GBRG)
    dng = _Dng(mosaic, raw_tags={33422: (_BYTE, _GBRG)})

    img = _decode(dng, tmp_path)

    assert img.size == (4, 4)
    # Linear 0.5 is 188 in sRGB.
    assert _pixels(img) == {(255, 0, 188)}


# Its width and height are no multiple of the tile size, so the tiles at the edges are padded.
_NOISE = np.random.default_rng(7).integers(0, _WHITE, size=(20, 36), dtype=np.uint16)


@pytest.mark.parametrize(
    "dng",
    [
        _Dng(_NOISE, rows_per_strip=3),
        _Dng(_NOISE, tile=16),
        _Dng(_NOISE, tile=16, predictor=2),
        _Dng(_NOISE, rows_per_strip=5, predictor=2, order=">"),
        _Dng(_NOISE, in_sub_ifd=True),
    ],
    ids=["short-last-strip", "padded-tiles", "predictor", "big-endian", "sub-ifd"],
)
def test_every_layout_decodes_to_the_same_image(tmp_path: Path, dng: _Dng) -> None:
    """Strips, tiles, the predictor, and the byte order only change how the values are stored."""
    expected = np.asarray(_decode(_Dng(_NOISE), tmp_path))

    np.testing.assert_array_equal(np.asarray(_decode(dng, tmp_path)), expected)


@pytest.mark.parametrize(
    ("max_size", "size"),
    [(8, (8, 4)), (10, (10, 5)), (_BIG, (32, 16))],
)
def test_the_image_is_binned_toward_max_size(
    tmp_path: Path,
    max_size: int,
    size: tuple[int, int],
) -> None:
    """
    A whole bin keeps the long side at max_size or more.

    With no bin, it is half the sensor.
    """
    dng = _Dng(_uniform((1, 2, 3), size=(32, 64)))

    assert _decode(dng, tmp_path, max_size).size == size


def test_a_blown_highlight_stays_white(tmp_path: Path) -> None:
    """
    The white balance clips in camera space.

    Without it, the color matrix tints a highlight.
    """
    dng = _Dng(
        _uniform((_WHITE, _WHITE, _WHITE)),
        ifd0_tags={50721: (_SRATIONAL, _X6_MATRIX), 50728: (_RATIONAL, _X6_NEUTRAL)},
    )

    assert _pixels(_decode(dng, tmp_path)) == {(255, 255, 255)}


def test_a_neutral_gray_stays_gray(tmp_path: Path) -> None:
    """A surface of the camera's neutral color comes out gray after the white balance and matrix."""
    green = 400
    red, _, blue = (round(green * share) for share in _X6_NEUTRAL)
    dng = _Dng(
        _uniform((red, green, blue)),
        ifd0_tags={50722: (_SRATIONAL, _X6_MATRIX), 50728: (_RATIONAL, _X6_NEUTRAL)},
    )

    (pixel,) = _pixels(_decode(dng, tmp_path))

    assert max(pixel) - min(pixel) <= 1


@pytest.mark.parametrize(("exposure", "value"), [(0, 137), (2, 255)])
def test_the_black_level_and_the_baseline_exposure(
    tmp_path: Path,
    exposure: float,
    value: int,
) -> None:
    """A quarter of the range above black is 137 in sRGB, and two stops brighter it is white."""
    black = 100
    quarter = black + (_WHITE - black) // 4
    dng = _Dng(
        _uniform((quarter, quarter, quarter)),
        raw_tags={50714: (_SHORT, (black,) * 4)},
        ifd0_tags={50730: (_SRATIONAL, (exposure,))},
    )

    assert _pixels(_decode(dng, tmp_path)) == {(value, value, value)}


def test_the_orientation_turns_the_image_upright(tmp_path: Path) -> None:
    """Orientation 6 means the camera was turned a quarter clockwise."""
    dng = _Dng(_uniform((1, 2, 3), size=(4, 8)), ifd0_tags={274: (_SHORT, (6,))})

    assert _decode(dng, tmp_path).size == (2, 4)


def _mosaic() -> npt.NDArray[np.uint16]:
    return _uniform((1, 2, 3))


@pytest.mark.parametrize(
    ("dng", "message"),
    [
        (_Dng(_mosaic(), compression=_UNCOMPRESSED), "Deflate compression"),
        (_Dng(_mosaic(), raw_tags={339: (_SHORT, (3,))}), "unsigned integer"),
        (_Dng(_mosaic(), raw_tags={258: (_SHORT, (12,))}), "16-bit"),
        (_Dng(_mosaic(), raw_tags={317: (_SHORT, (3,))}), "predictor"),
        (_Dng(_mosaic(), raw_tags={33421: (_SHORT, (3, 3))}), "2x2"),
        (_Dng(_mosaic(), raw_tags={33422: (_BYTE, (0, 1, 1, 1))}), "red, green, and blue"),
        (_Dng(_mosaic(), raw_tags={262: (_SHORT, (34892,))}), "color filter array"),
        (_Dng(_mosaic(), raw_tags={273: None, 279: None}), "0 offsets"),
        (_Dng(_mosaic(), raw_tags={256: (_LONG, (0,))}), "0x8"),
        (_Dng(_mosaic(), raw_tags={50714: (_LONG, (_WHITE,))}), "white level"),
        (_Dng(_mosaic(), compression=_UNCOMPRESSED, raw_tags={259: (_SHORT, (8,))}), "header"),
        (
            _Dng(_mosaic(), rows_per_strip=4, raw_tags={257: (_LONG, (16,)), 278: (_LONG, (8,))}),
            "holds",
        ),
        (_Dng(_mosaic(), raw_tags={279: (_LONG, (1_000_000,))}), "ends early"),
        (_Dng(_uniform((1, 2, 3), size=(1, 8))), "too small"),
    ],
    ids=[
        "uncompressed",
        "float",
        "12-bit",
        "float-predictor",
        "3x3-pattern",
        "no-blue",
        "linear-raw",
        "no-strips",
        "no-width",
        "black-at-white",
        "not-zlib",
        "short-strips",
        "strip-past-the-end",
        "one-row",
    ],
)
def test_a_dng_it_cannot_decode_raises(tmp_path: Path, dng: _Dng, message: str) -> None:
    """Each kind of file the decoder does not know is refused by name, never decoded wrong."""
    with pytest.raises(DngDecodeError, match=message):
        _decode(dng, tmp_path)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\x89PNG\r\n\x1a\n" + bytes(16), "not a TIFF"),
        (b"II*", "ends early"),
        # One StripOffsets entry that claims a billion values.
        (b"II*\0\x08\0\0\0\x01\0\x11\x01\x04\0\0\0\0\x40\0\0\0\0", "a tag value of"),
    ],
    ids=["png", "three-bytes", "huge-count"],
)
def test_a_file_that_is_not_a_sound_tiff_raises(tmp_path: Path, data: bytes, message: str) -> None:
    """The header and the tag counts are checked before anything is read on their word."""
    path = _write(tmp_path / "photo.dng", data)

    with pytest.raises(DngDecodeError, match=message):
        open_deflate_dng(path, _BIG)


def test_a_cut_off_file_raises(tmp_path: Path) -> None:
    """
    A copy that stopped early fails cleanly.

    The IFD is last in the file, so the cut is in it.
    """
    path = _Dng(_mosaic()).write(tmp_path / "photo.dng")
    path.write_bytes(path.read_bytes()[:-10])

    with pytest.raises(DngDecodeError, match="ends early"):
        open_deflate_dng(path, _BIG)


def _unpack(path: Path) -> npt.NDArray[np.uint16]:
    with rawpy.imread(str(path)) as raw:  # type: ignore[no-untyped-call]
        return np.array(raw.raw_image)


def test_rawpy_still_rejects_integer_deflate_dng(tmp_path: Path) -> None:
    """
    Fail once LibRaw decodes integer Deflate data, so the fallback in deflate_dng.py can go.

    The same sensor values in an uncompressed DNG show that LibRaw accepts the rest of the file.
    LibRaw takes no image below 22 pixels on a side.
    """
    mosaic = _uniform((1, 2, 3), size=(64, 64))
    plain = _Dng(mosaic, compression=_UNCOMPRESSED).write(tmp_path / "plain.dng")
    deflated = _Dng(mosaic).write(tmp_path / "deflated.dng")

    np.testing.assert_array_equal(_unpack(plain), mosaic)
    with pytest.raises(LibRawDataError):
        _unpack(deflated)


def test_the_image_loader_decodes_a_dng_that_rawpy_rejects(tmp_path: Path) -> None:
    """``_open_image`` tries rawpy first, then this decoder for a DNG."""
    path = _Dng(_uniform((_WHITE, 0, 0))).write(tmp_path / "photo.dng")

    img = _open_image(path)

    assert img.size == (4, 4)
    assert _pixels(img) == {(255, 0, 0)}


def test_the_image_loader_keeps_other_raw_formats_away_from_the_dng_decoder(
    tmp_path: Path,
) -> None:
    """A RAW file that is not a DNG goes from rawpy straight to PIL."""
    path = tmp_path / "photo.cr3"
    Image.new("RGB", (4, 4)).save(path, format="PNG")

    with (
        patch("photo_tagger.image_io.rawpy.imread", side_effect=OSError("no")),
        patch("photo_tagger.image_io.open_deflate_dng") as decoder,
    ):
        img = _open_image(path)

    assert img.size == (4, 4)
    decoder.assert_not_called()
