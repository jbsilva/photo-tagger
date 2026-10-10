"""
Integration-ish tests for metadata reads and writes via a real exiftool binary.

The functions here invoke exiftool through pyexiftool, so they need the binary on PATH. They write
to real (small) JPEG files in a tmp directory.
"""

import os
import shutil
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from PIL import Image

from photo_tagger.config import TAG_EXIF_DATE_TIME_ORIGINAL
from photo_tagger.fields import SHOWN_FIELDS
from photo_tagger.metadata import (
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOFTWARE_AGENT,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    WriteResult,
    capture_year,
    iter_field_presence,
    metadata_targets,
    read_all_tags,
    read_caption,
    read_capture_years,
    read_fields,
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


@pytest.mark.parametrize("use_sidecar", [True, False])
def test_saving_the_same_values_again_leaves_the_file_alone(
    tmp_path: Path,
    *,
    use_sidecar: bool,
) -> None:
    """A second identical save must not move the mtime, or rsync backs the photo up again."""
    img = _write_jpeg(tmp_path / "img.jpg")
    values = {
        "keywords": KeywordSet(subject=["Frog"], hierarchical=["Animal|Frog"]),
        "title": "A frog",
        "description": "A frog at night.",
        "extra": {
            "EXIF:Copyright": "(c) 2019 Ada",
            "IPTC:CopyrightNotice": "(c) 2019 Ada",
            "XMP-dc:Rights": "© 2019 Ada",
            "XMP-dc:Creator": "Ada",
            "XMP-xmpRights:Marked": "True",
        },
    }
    assert write_metadata(img, **values, backup=False, use_sidecar=use_sidecar)  # type: ignore[arg-type]
    target = img.with_suffix(".xmp") if use_sidecar else img
    before = target.stat().st_mtime_ns

    again = write_metadata(img, **values, backup=False, use_sidecar=use_sidecar)  # type: ignore[arg-type]

    assert again == WriteResult.UNCHANGED
    assert target.stat().st_mtime_ns == before


def _iptc(path: Path) -> dict[str, object]:
    """Read the whole IPTC block of *path*, the way photo-tagger reads it."""
    with ExifToolHelper() as helper:
        return dict(helper.get_tags(files=[str(path)], tags=["IPTC:all"])[0])


def test_a_non_latin_name_reaches_iptc_intact_and_a_rerun_leaves_it(tmp_path: Path) -> None:
    """
    IPTC gets UTF-8, cut to its byte limit on a character boundary, and a rerun writes nothing.

    Without the charset marker ExifTool writes Latin-1 ("?ukasz"), and its own cut can split a
    character, so the value never read back as sent and every save rewrote the file.
    """
    img = _write_jpeg(tmp_path / "img.jpg")
    name = "Łukasz Żółwiński Fotografia Studio"  # 38 bytes: over By-line's 32
    extra = {"IPTC:By-line": name, "IPTC:CopyrightNotice": f"© 2024 {name}"}

    assert write_metadata(img, KeywordSet(), extra=extra, backup=False, use_sidecar=False)
    iptc = _iptc(img)
    assert iptc["IPTC:By-line"] == "Łukasz Żółwiński Fotografia"
    assert iptc["IPTC:CopyrightNotice"] == f"© 2024 {name}"

    again = write_metadata(img, KeywordSet(), extra=extra, backup=False, use_sidecar=False)
    assert again == WriteResult.UNCHANGED


def test_switching_iptc_to_utf8_keeps_the_values_already_there(tmp_path: Path) -> None:
    """A Latin-1 caption written by another program reads the same after the block is re-encoded."""
    img = _write_jpeg(tmp_path / "img.jpg")
    with ExifToolHelper() as helper:
        helper.set_tags(
            files=[str(img)],
            tags={"IPTC:Caption-Abstract": "Café crème", "IPTC:City": "Zürich"},
            params=["-overwrite_original"],
        )

    extra = {"IPTC:By-line": "Łukasz"}
    assert write_metadata(img, KeywordSet(), extra=extra, backup=False, use_sidecar=False)

    iptc = _iptc(img)
    assert iptc["IPTC:By-line"] == "Łukasz"
    assert iptc["IPTC:Caption-Abstract"] == "Café crème"
    assert iptc["IPTC:City"] == "Zürich"


def test_a_webp_save_is_left_alone_the_second_time(tmp_path: Path) -> None:
    """WebP holds no IPTC, so the IPTC mirrors are not sent and a rerun finds nothing to change."""
    img = tmp_path / "img.webp"
    Image.new("RGB", (4, 4), color="red").save(img, format="WEBP")
    values = {"keywords": KeywordSet(subject=["Frog"]), "title": "A frog"}

    assert write_metadata(img, **values, backup=False, use_sidecar=False)  # type: ignore[arg-type]
    again = write_metadata(img, **values, backup=False, use_sidecar=False)  # type: ignore[arg-type]

    assert again == WriteResult.UNCHANGED


def test_a_save_stamps_the_software_and_keeps_the_camera_firmware(tmp_path: Path) -> None:
    """
    A write names photo-tagger in ProcessingSoftware and in the history, and leaves Software alone.

    Software holds the camera's firmware. A second identical save writes nothing, so it adds no
    second history entry.
    """
    img = _write_jpeg(tmp_path / "img.jpg")
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        et.set_tags(
            files=[str(img)],
            tags={"EXIF:Software": "Ver.1.00"},
            params=["-overwrite_original"],
        )

    for _ in range(2):
        write_metadata(img, KeywordSet(subject=["Frog"]), backup=False, use_sidecar=False)

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        block = et.get_tags(
            files=[str(img)],
            tags=["EXIF:Software", "EXIF:ProcessingSoftware", "XMP:History"],
            params=["-struct"],
        )[0]
    assert block["EXIF:Software"] == "Ver.1.00"
    assert block["EXIF:ProcessingSoftware"] == SOFTWARE_AGENT
    [event] = block["XMP:History"]
    assert event["SoftwareAgent"] == SOFTWARE_AGENT
    assert event["Changed"] == "/metadata"


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
    assert ok == WriteResult.WRITTEN

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
    assert ok == WriteResult.WRITTEN
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
    assert ok == WriteResult.WRITTEN

    keywords = read_image_context(img).existing_keywords
    # Exact equality: a garbled IPTC mirror would surface as an extra, mangled entry.
    assert keywords.subject == ["Pássaro", "München"]
    assert read_caption(img) == ("Café à noite", "Descrição do pôr do sol.")


def test_write_metadata_returns_false_for_empty_payload(tmp_path: Path) -> None:
    """Nothing to write -> early False, no exiftool call."""
    img = _write_jpeg(tmp_path / "img.jpg")
    assert write_metadata(img, KeywordSet(), use_sidecar=False) == WriteResult.FAILED


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
    assert ok == WriteResult.WRITTEN

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


def _set_capture_date(path: Path, when: str) -> None:
    """Stamp EXIF:DateTimeOriginal so the copyright year has something to come from."""
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        et.set_tags(
            files=[str(path)],
            tags={TAG_EXIF_DATE_TIME_ORIGINAL: when},
            params=["-overwrite_original"],
        )


def test_extra_tags_are_embedded_alongside_the_generated_fields(tmp_path: Path) -> None:
    """A copyright payload rides in the same write as the keywords, in every namespace."""
    img = _write_jpeg(tmp_path / "img.jpg")

    assert write_metadata(
        img,
        KeywordSet(subject=["Beach"]),
        title="A title",
        extra={
            "EXIF:Artist": "Ada Lovelace",
            "XMP-dc:Creator": "Ada Lovelace",
            "IPTC:By-line": "Ada Lovelace",
            "XMP-dc:Rights": "© 2019 Ada Lovelace",
            "XMP-xmpRights:Marked": "True",
        },
        backup=False,
        use_sidecar=False,
    )

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        block = et.get_tags(
            files=[str(img)],
            tags=["EXIF:Artist", "XMP:Creator", "IPTC:By-line", "XMP:Rights", "XMP:Marked"],
        )[0]
    assert block["EXIF:Artist"] == "Ada Lovelace"
    assert block["XMP:Creator"] == "Ada Lovelace"
    assert block["IPTC:By-line"] == "Ada Lovelace"
    assert block["XMP:Rights"] == "© 2019 Ada Lovelace"
    assert block["XMP:Marked"] is True


def test_sidecar_write_keeps_the_xmp_half_of_an_extra_payload(tmp_path: Path) -> None:
    """
    EXIF and IPTC are dropped for a sidecar rather than failing the write.

    ExifTool answers "Nothing to write" when every tag in a payload belongs to a group an XMP file
    cannot hold, which would fail every photo in a run that only asked for a credit.
    """
    img = _write_jpeg(tmp_path / "img.jpg")

    assert write_metadata(
        img,
        KeywordSet(),
        extra={"EXIF:Artist": "Ada Lovelace", "XMP-dc:Creator": "Ada Lovelace"},
        backup=False,
        use_sidecar=True,
    )

    sidecar = tmp_path / "img.xmp"
    assert sidecar.is_file()
    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        block = et.get_tags(files=[str(sidecar)], tags=["XMP:Creator", "EXIF:Artist"])[0]
    assert block["XMP:Creator"] == "Ada Lovelace"
    assert "EXIF:Artist" not in block


def test_extra_tags_alone_are_enough_to_write(tmp_path: Path) -> None:
    """The `set` command writes no keyword, title, or description, and must not be refused."""
    img = _write_jpeg(tmp_path / "img.jpg")

    assert write_metadata(
        img,
        KeywordSet(),
        extra={"EXIF:Artist": "Ada Lovelace"},
        backup=False,
        use_sidecar=False,
    )

    with ExifToolHelper() as et:  # type: ignore[no-untyped-call]
        assert (
            et.get_tags(files=[str(img)], tags=["EXIF:Artist"])[0]["EXIF:Artist"] == "Ada Lovelace"
        )


def test_capture_year_comes_from_the_photo_not_from_today(tmp_path: Path) -> None:
    """A 2019 frame is stamped 2019, whenever the run happens."""
    shot_in = 2019
    img = _write_jpeg(tmp_path / "img.jpg")
    _set_capture_date(img, f"{shot_in}:07:14 10:32:11")

    assert read_capture_years([img]) == {img: shot_in}
    assert capture_year(img, f"{shot_in}:07:14 10:32:11") == shot_in


def test_capture_year_falls_back_to_the_file_mtime(tmp_path: Path) -> None:
    """A photo with no capture date still gets a year: the file's own."""
    img = _write_jpeg(tmp_path / "img.jpg")
    os.utime(img, (1_000_000_000, 1_000_000_000))  # 2001-09-09 UTC
    expected = datetime.fromtimestamp(1_000_000_000, tz=UTC).astimezone().year

    assert read_capture_years([img]) == {img: expected}
    assert capture_year(img, None) == expected
    assert capture_year(img, "not a date") == expected


def test_read_fields_reports_the_tag_behind_every_value(tmp_path: Path) -> None:
    """Two copyright tags that disagree are two lines, each naming where it came from."""
    img = _write_jpeg(tmp_path / "img.jpg")
    write_metadata(
        img,
        KeywordSet(),
        extra={"EXIF:Copyright": "Copyright (c) 2019 Ada", "XMP-dc:Rights": "© 2019 Ada"},
        backup=False,
        use_sidecar=False,
    )

    readings = read_fields(img, SHOWN_FIELDS)
    copyrights = [r for r in readings if r.label == "Copyright"]
    assert [(r.value, r.tag) for r in copyrights] == [
        ("© 2019 Ada", "XMP:Rights"),
        ("Copyright (c) 2019 Ada", "EXIF:Copyright"),
    ]
    assert all(r.source == SOURCE_IMAGE for r in copyrights)


def test_read_fields_collapses_a_value_repeated_across_tags(tmp_path: Path) -> None:
    """One credit written to three namespaces reads back as one line, not three."""
    img = _write_jpeg(tmp_path / "img.jpg")
    write_metadata(
        img,
        KeywordSet(),
        extra={
            "EXIF:Artist": "Ada Lovelace",
            "XMP-dc:Creator": "Ada Lovelace",
            "IPTC:By-line": "Ada Lovelace",
        },
        backup=False,
        use_sidecar=False,
    )

    creators = [r for r in read_fields(img, SHOWN_FIELDS) if r.label == "Creator"]
    assert [(r.value, r.tag) for r in creators] == [("Ada Lovelace", "XMP:Creator")]


def test_read_fields_marks_a_value_that_lives_in_the_sidecar(tmp_path: Path) -> None:
    """Where a value lives is part of the answer: a catalog reads the sidecar, not the photo."""
    img = _write_jpeg(tmp_path / "img.jpg")
    write_metadata(img, KeywordSet(), title="From the sidecar", backup=False, use_sidecar=True)

    titles = [r for r in read_fields(img, SHOWN_FIELDS) if r.label == "Title"]
    assert [(r.value, r.source) for r in titles] == [("From the sidecar", SOURCE_SIDECAR)]


def test_read_fields_is_empty_for_a_missing_photo(tmp_path: Path) -> None:
    """A path with nothing behind it reports nothing rather than raising."""
    assert read_fields(tmp_path / "ghost.jpg", SHOWN_FIELDS) == []
    assert read_all_tags(tmp_path / "ghost.jpg") == {}


def test_read_all_tags_reports_each_file_and_drops_binary(tmp_path: Path) -> None:
    """--all covers the photo and its sidecar, minus the payloads a terminal cannot show."""
    img = _write_jpeg(tmp_path / "img.jpg")
    write_metadata(img, KeywordSet(subject=["Beach"]), backup=False, use_sidecar=True)

    by_file = read_all_tags(img)
    assert set(by_file) == {str(img), str(tmp_path / "img.xmp")}
    assert by_file[str(img)]["File:FileType"] == "JPEG"
    assert by_file[str(tmp_path / "img.xmp")]["XMP:Subject"] == "Beach"
    assert all(not v.startswith("(Binary data") for tags in by_file.values() for v in tags.values())
