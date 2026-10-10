"""
Integration-ish tests for metadata reads and writes via a real exiftool binary.

The functions here invoke exiftool through pyexiftool, so they need the binary on PATH. They write
to real (small) JPEG files in a tmp directory.
"""

import shutil
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from photo_tagger.metadata import (
    FIELD_KEYWORDS,
    FIELD_TITLE,
    iter_field_presence,
    metadata_targets,
    read_caption,
    read_image_context,
    write_metadata,
    write_metadata_everywhere,
)
from photo_tagger.models import KeywordSet


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


def test_iter_field_presence_reads_every_batch_through_one_exiftool(tmp_path: Path) -> None:
    """The chunked read reports each batch against a real exiftool, sharing one process."""
    tagged = _write_jpeg(tmp_path / "tagged.jpg")
    plain = _write_jpeg(tmp_path / "plain.jpg")
    last = _write_jpeg(tmp_path / "last.jpg")
    write_metadata(tagged, KeywordSet(subject=["Beach"]), title="A title", use_sidecar=False)

    batches = list(iter_field_presence([tagged, plain, last], batch_size=2))

    assert [batch.scanned for batch in batches] == [2, 1]
    assert batches[0].presence[tagged] >= {FIELD_TITLE, FIELD_KEYWORDS}
    assert batches[0].presence[plain] == set()
    assert batches[1].presence[last] == set()


def test_write_and_read_round_trip_subject_and_hierarchy(tmp_path: Path) -> None:
    """Write subjects + hierarchy embedded into the JPEG and read them back."""
    img = _write_jpeg(tmp_path / "img.jpg")
    ok = write_metadata(
        img,
        KeywordSet(
            subject=["Beach", "Sunset"],
            hierarchical=["Animal|Bird"],
        ),
        description="A small description.",
        title="A small title",
        backup=False,
        use_sidecar=False,
    )
    assert ok is True

    keywords = read_image_context(img).existing_keywords
    assert "Beach" in keywords.subject
    assert "Sunset" in keywords.subject
    assert "Animal|Bird" in keywords.hierarchical


def test_sidecar_write_and_read_round_trip(tmp_path: Path) -> None:
    """
    The default (non-destructive) mode writes a sidecar and reads back through it.

    The embedded-mode round-trips elsewhere never touch this path, yet it is what every default run
    uses; the original file must remain byte-identical.
    """
    img = _write_jpeg(tmp_path / "img.jpg")
    original_bytes = img.read_bytes()

    ok = write_metadata(
        img,
        KeywordSet(subject=["Beach"], hierarchical=["Animal|Bird"]),
        title="Sidecar Title",
        description="Sidecar description.",
        backup=False,
        use_sidecar=True,
    )
    assert ok is True
    assert (tmp_path / "img.xmp").is_file()
    assert img.read_bytes() == original_bytes  # the original was never modified

    keywords = read_image_context(img).existing_keywords
    assert "Beach" in keywords.subject
    assert "Animal|Bird" in keywords.hierarchical
    assert read_caption(img) == ("Sidecar Title", "Sidecar description.")


def test_both_mode_writes_the_photo_and_a_sidecar(tmp_path: Path) -> None:
    """Mode "both" leaves the metadata in two places, each readable on its own."""
    from exiftool import ExifToolHelper  # type: ignore[attr-defined]  # noqa: PLC0415

    img = _write_jpeg(tmp_path / "img.jpg")
    sidecar = tmp_path / "img.xmp"

    ok, targets = write_metadata_everywhere(
        img,
        KeywordSet(subject=["Beach"], hierarchical=["Animal|Bird"]),
        title="A title",
        description="A description.",
        backup=False,
        sidecar_mode="both",
    )

    assert ok is True
    assert [t.path for t in targets] == [img, sidecar]
    assert sidecar.is_file()
    # Each file carries the metadata by itself, so neither depends on the other.
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        for target in (img, sidecar):
            block = et.get_tags(files=[str(target)], tags=["XMP:Subject", "XMP:Description"])[0]
            assert block.get("XMP:Description") == "A description."
            assert "Beach" in block.get("XMP:Subject", [])
    # Only the photo gets the EXIF tag; a sidecar holds XMP.
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        embedded = et.get_tags(files=[str(img)], tags=["EXIF:ImageDescription"])[0]
    assert embedded.get("EXIF:ImageDescription") == "A description."


def test_unicode_metadata_round_trips_unmangled(tmp_path: Path) -> None:
    """
    Accented keywords, titles, and descriptions survive a write/read cycle exactly.

    Encoding regressions here are classic silent corruption (UTF-8 text re-read through a Latin-1
    IPTC path turns every accent into mojibake) that ASCII-only fixtures never catch.
    """
    img = _write_jpeg(tmp_path / "img.jpg")
    ok = write_metadata(
        img,
        KeywordSet(subject=["Pássaro", "München"]),
        title="Café à noite",
        description="Descrição do pôr do sol.",
        backup=False,
        use_sidecar=False,
    )
    assert ok is True

    keywords = read_image_context(img).existing_keywords
    # Exact equality: a garbled IPTC mirror would surface as an extra, mangled entry.
    assert keywords.subject == ["Pássaro", "München"]
    assert read_caption(img) == ("Café à noite", "Descrição do pôr do sol.")


def test_write_metadata_returns_false_for_empty_payload(tmp_path: Path) -> None:
    """Nothing to write -> early False, no exiftool call."""
    img = _write_jpeg(tmp_path / "img.jpg")
    assert write_metadata(img, KeywordSet(), use_sidecar=False) is False


def test_read_image_context_is_empty_for_unset_image(tmp_path: Path) -> None:
    """A freshly-written JPEG has no location tags and no GPS position."""
    img = _write_jpeg(tmp_path / "img.jpg")
    context = read_image_context(img)
    assert context.location_tags == {}
    assert context.gps_position is None


def test_read_image_context_handles_missing_file(tmp_path: Path) -> None:
    """A missing file returns an empty context, not an exception."""
    out = read_image_context(tmp_path / "ghost.jpg")
    assert out.existing_keywords == KeywordSet()


def test_description_mirror_round_trips_as_image_description(tmp_path: Path) -> None:
    """
    The ImageDescription mirror is actually written, not silently rejected.

    Regression test: the payload used XMP-exif:ImageDescription, which exiftool refuses with a
    warning but exit code 0, so the mirror never reached the file and write_metadata still reported
    success.
    """
    from exiftool import ExifToolHelper  # type: ignore[attr-defined]  # noqa: PLC0415

    img = _write_jpeg(tmp_path / "img.jpg")
    ok = write_metadata(
        img,
        KeywordSet(subject=["X"]),
        description="A mirrored description.",
        backup=False,
        use_sidecar=False,
    )
    assert ok is True

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        blocks = et.get_tags(files=[str(img)], tags=["XMP:ImageDescription"])
    assert blocks[0].get("XMP:ImageDescription") == "A mirrored description."


def _set_camera_description(path: Path, value: str) -> None:
    """Put *value* on a photo the way a camera does: the EXIF tag and its XMP copy."""
    from exiftool import ExifToolHelper  # type: ignore[attr-defined]  # noqa: PLC0415

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        et.set_tags(
            tags={"EXIF:ImageDescription": value, "XMP-dc:Description": value},
            files=[str(path)],
            params=["-overwrite_original"],
        )


def test_embedded_write_replaces_a_camera_written_exif_description(tmp_path: Path) -> None:
    """
    Embedding a description replaces the EXIF tag, not only its XMP mirror.

    Some cameras write a placeholder into ImageDescription on every photo. Writing the XMP mirror
    alone left that in IFD0, so plain ``exiftool photo.jpg`` and every EXIF-only reader kept
    reporting it after a save the GUI called successful.
    """
    from exiftool import ExifToolHelper  # type: ignore[attr-defined]  # noqa: PLC0415

    img = _write_jpeg(tmp_path / "img.jpg")
    _set_camera_description(img, "default")

    assert write_metadata(
        img,
        KeywordSet(subject=["X"]),
        description="A real description.",
        backup=False,
        use_sidecar=False,
    )

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        blocks = et.get_tags(files=[str(img)], tags=["EXIF:ImageDescription"])
    assert blocks[0].get("EXIF:ImageDescription") == "A real description."


def test_a_sidecar_caption_wins_over_a_stale_one_in_the_photo(tmp_path: Path) -> None:
    """
    The sidecar is what a save wrote, so it is what reading the photo reports.

    In sidecar mode the photo's own bytes are never touched, which leaves a camera's placeholder
    description in place; reading the image first meant the GUI kept showing it.
    """
    img = _write_jpeg(tmp_path / "img.jpg")
    _set_camera_description(img, "default")

    assert write_metadata(
        img,
        KeywordSet(subject=["X"]),
        description="A real description.",
        backup=False,
        use_sidecar=True,
    )

    assert read_caption(img)[1] == "A real description."
    assert read_image_context(img).existing_description == "A real description."


def test_read_caption_round_trip(tmp_path: Path) -> None:
    """A written title and description are read back by read_caption."""
    img = _write_jpeg(tmp_path / "img.jpg")
    write_metadata(
        img,
        KeywordSet(subject=["X"]),
        description="A small description.",
        title="A small title",
        backup=False,
        use_sidecar=False,
    )
    assert read_caption(img) == ("A small title", "A small description.")
