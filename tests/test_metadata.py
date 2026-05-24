"""Tests for metadata helpers that don't need a real exiftool binary."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from photo_tagger.metadata import (
    _block_has_indicator,
    _build_write_payload,
    _coerce_to_list,
    _dedup_preserving_first_case,
    _value_is_present,
    build_contextual_prompt,
    find_tagged_images,
    format_metadata_value,
    read_image_context,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_format_metadata_value_passes_strings_through() -> None:
    """Strings are returned unchanged."""
    assert format_metadata_value("hello") == "hello"


def test_format_metadata_value_joins_iterables_and_drops_blanks() -> None:
    """Lists/tuples/sets render as comma-joined strings, blanks removed."""
    assert format_metadata_value(["a", "", "b"]) == "a, b"
    assert format_metadata_value(("a", "b")) == "a, b"


def test_format_metadata_value_falls_back_to_str() -> None:
    """Non-string scalars fall back to str()."""
    assert format_metadata_value(42) == "42"
    assert format_metadata_value(None) == "None"


def test_coerce_to_list_handles_scalar_and_iterable() -> None:
    """Scalar values become single-element lists; lists are filtered."""
    assert _coerce_to_list("solo") == ["solo"]
    assert _coerce_to_list("") == []
    assert _coerce_to_list(["a", "", " "]) == ["a"]
    assert _coerce_to_list(("a", "b")) == ["a", "b"]


def test_dedup_preserving_first_case_collapses_case_duplicates() -> None:
    """Duplicates that differ only by case are folded into the first-seen casing."""
    out = _dedup_preserving_first_case(["Bird", "bird", "BIRD", "Beach"])
    assert out == ["Bird", "Beach"]


def test_dedup_preserving_first_case_preserves_order() -> None:
    """Unique entries appear in their original order."""
    assert _dedup_preserving_first_case(["b", "a", "c"]) == ["b", "a", "c"]


def test_dedup_preserving_first_case_handles_empty() -> None:
    """An empty input returns an empty list, not None."""
    assert _dedup_preserving_first_case([]) == []


def test_build_contextual_prompt_with_no_metadata_returns_base() -> None:
    """An empty metadata block leaves the prompt as the base instruction."""
    out = build_contextual_prompt("Analyze.", [], {}, {})
    assert out == "Analyze."


def test_build_contextual_prompt_includes_only_present_sections() -> None:
    """Only populated metadata sections appear in the prompt."""
    out = build_contextual_prompt(
        "Analyze.",
        ["Beach"],
        {},
        {"position": "0,0"},
    )
    assert "Existing Keywords" in out
    assert "GPS: 0,0" in out
    assert "Location" not in out


def test_build_write_payload_produces_lightroom_compatible_keys() -> None:
    """Subjects mirror to IPTC:Keywords and titles fan out to two tags."""
    payload = _build_write_payload(
        {"subject": ["Beach"], "hierarchical": ["Animal|Bird"]},
        description="A short desc.",
        title="A title",
    )
    assert payload["XMP-dc:Subject"] == ["Beach"]
    assert payload["IPTC:Keywords"] == ["Beach"]
    assert payload["XMP-lr:HierarchicalSubject"] == ["Animal|Bird"]
    assert payload["XMP-dc:Description"] == "A short desc."
    assert payload["XMP-exif:ImageDescription"] == "A short desc."
    assert payload["XMP-dc:Title"] == "A title"
    assert payload["IPTC:ObjectName"] == "A title"


def test_build_write_payload_skips_blank_values() -> None:
    """Empty keyword lists / blank title and description produce no entries."""
    payload = _build_write_payload({}, description=None, title=None)
    assert payload == {}


def test_value_is_present_distinguishes_blanks_from_content() -> None:
    """Blank strings, None, and empty lists count as "no value"."""
    assert _value_is_present("hello") is True
    assert _value_is_present(["", "x"]) is True
    assert _value_is_present(0) is True  # numeric 0 stringifies to "0"
    assert _value_is_present(None) is False
    assert _value_is_present("") is False
    assert _value_is_present("   ") is False
    assert _value_is_present([]) is False
    assert _value_is_present(["", " "]) is False


def test_block_has_indicator_matches_any_indicator_tag() -> None:
    """Any populated indicator tag flips the indicator True; otherwise False."""
    assert _block_has_indicator([{"XMP:Subject": ["Beach"]}]) is True
    assert _block_has_indicator([{"XMP:Description": "A scene."}]) is True
    assert _block_has_indicator([{"IPTC:ObjectName": "A title"}]) is True
    # Non-indicator tag alone (e.g., file size) should not trigger the indicator.
    assert _block_has_indicator([{"File:FileSize": 1234}]) is False
    assert _block_has_indicator([{"XMP:Subject": []}]) is False
    assert _block_has_indicator([]) is False


def test_find_tagged_images_returns_paths_with_indicator(tmp_path: Path) -> None:
    """Images whose exiftool block carries an indicator tag are returned."""
    a = tmp_path / "a.cr3"
    b = tmp_path / "b.cr3"
    a.write_text("x")
    b.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    # First image has keywords; second has nothing.
    fake_helper.get_tags.side_effect = [
        [{"XMP:Subject": ["Beach"]}],
        [{"File:FileSize": 100}],
    ]

    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        tagged = find_tagged_images([a, b])

    assert tagged == {a}


def test_find_tagged_images_returns_empty_for_empty_input() -> None:
    """An empty input list short-circuits without invoking exiftool."""
    with patch("photo_tagger.metadata.ExifToolHelper") as helper:
        assert find_tagged_images([]) == set()
    helper.assert_not_called()


def test_find_tagged_images_skips_paths_with_no_targets(tmp_path: Path) -> None:
    """Paths that have no readable file or sidecar are silently skipped."""
    ghost = tmp_path / "missing.cr3"  # never created

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        assert find_tagged_images([ghost]) == set()
    fake_helper.get_tags.assert_not_called()


def test_read_image_context_batches_keywords_location_camera_and_gps(tmp_path: Path) -> None:
    """A single exiftool call populates every section of the returned ImageContext."""
    img = tmp_path / "img.cr3"
    img.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    fake_helper.get_tags.return_value = [
        {
            "XMP:Subject": ["Beach", "Sunset"],
            "XMP:HierarchicalSubject": ["Animal|Bird"],
            "XMP-photoshop:Country": "Portugal",
            "EXIF:Model": "Canon EOS R5",
            "EXIF:LensModel": "RF24-105mm F4 L IS USM",
            "EXIF:DateTimeOriginal": "2024:01:15 14:32:01",
            "Composite:GPSPosition": "38.7 N, 9.1 W",
        },
    ]

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(img)

    assert context.existing_keywords["subject"] == ["Beach", "Sunset"]
    assert context.existing_keywords["hierarchical"] == ["Animal|Bird"]
    assert context.location_tags == {"XMP-photoshop:Country": "Portugal"}
    assert context.camera_info["EXIF:Model"] == "Canon EOS R5"
    assert context.gps_position == "38.7 N, 9.1 W"
    # The whole point of batching is one IPC call - assert exactly one.
    assert fake_helper.get_tags.call_count == 1


def test_read_image_context_collapses_case_duplicates_across_blocks(tmp_path: Path) -> None:
    """A keyword present in both XMP and IPTC with different casing collapses to one."""
    img = tmp_path / "img.cr3"
    img.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    # First block stands in for the image, second for an XMP sidecar that
    # disagrees on case. Both should not survive the dedup at read time.
    fake_helper.get_tags.return_value = [
        {"XMP:Subject": ["Bird"], "IPTC:Keywords": ["bird"]},
        {"XMP:Subject": ["BIRD", "Beach"]},
    ]

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(img)

    # First-seen casing wins; "Beach" survives once.
    assert context.existing_keywords["subject"] == ["Bird", "Beach"]


def test_read_image_context_returns_empty_when_no_targets(tmp_path: Path) -> None:
    """A missing file returns an empty ImageContext without invoking exiftool."""
    ghost = tmp_path / "ghost.cr3"

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(ghost)

    assert context.existing_keywords == {"subject": [], "hierarchical": [], "weighted": []}
    assert context.gps_position is None
    assert context.camera_info == {}
    fake_helper.get_tags.assert_not_called()


def test_build_contextual_prompt_renders_camera_section() -> None:
    """When camera_info is provided, equipment and capture-date lines appear in the prompt."""
    prompt = build_contextual_prompt(
        "Analyze the scene.",
        [],
        {},
        {},
        camera_info={
            "EXIF:Model": "Canon EOS R5",
            "EXIF:LensModel": "RF100mm F2.8 L Macro IS USM",
            "EXIF:DateTimeOriginal": "2024:01:15 14:32:01",
        },
    )

    assert "- Camera: Canon EOS R5" in prompt
    assert "- Lens: RF100mm F2.8 L Macro IS USM" in prompt
    assert "- Captured: 2024:01:15 14:32:01" in prompt
