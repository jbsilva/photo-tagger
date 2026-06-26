"""Tests for metadata helpers that don't need a real exiftool binary."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    _block_has_indicator,
    _build_write_payload,
    _coerce_to_list,
    _dedup_preserving_first_case,
    _value_is_present,
    build_contextual_prompt,
    find_field_presence,
    find_tagged_images,
    format_metadata_value,
    managed_helper,
    read_caption,
    read_existing_keywords,
    read_gps_coordinates,
    read_image_context,
    read_location_tags,
    read_metadata_sources,
    write_metadata,
)
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from pathlib import Path


def _fake_helper(get_tags_result: object = None) -> MagicMock:
    """Build a MagicMock standing in for an open ExifToolHelper context manager."""
    helper = MagicMock()
    helper.__enter__.return_value = helper
    helper.__exit__.return_value = False
    if get_tags_result is not None:
        helper.get_tags.return_value = get_tags_result
    return helper


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
    """Subjects mirror to IPTC:Keywords, weighted flat subjects, and titles fan out to two tags."""
    payload = _build_write_payload(
        KeywordSet(subject=["Beach"], hierarchical=["Animal|Bird"], weighted=["Beach"]),
        description="A short desc.",
        title="A title",
    )
    assert payload["XMP-dc:Subject"] == ["Beach"]
    assert payload["IPTC:Keywords"] == ["Beach"]
    assert payload["XMP-lr:HierarchicalSubject"] == ["Animal|Bird"]
    assert payload["XMP:WeightedFlatSubject"] == ["Beach"]
    assert payload["XMP-dc:Description"] == "A short desc."
    assert payload["XMP-exif:ImageDescription"] == "A short desc."
    assert payload["XMP-dc:Title"] == "A title"
    assert payload["IPTC:ObjectName"] == "A title"


def test_build_write_payload_skips_blank_values() -> None:
    """Empty keyword lists / blank title and description produce no entries."""
    payload = _build_write_payload(KeywordSet(), description=None, title=None)
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
    # Batched call returns one block per target file. First image has keywords;
    # second has nothing. SourceFile lets the mapper find the owner.
    fake_helper.get_tags.return_value = [
        {"SourceFile": str(a), "XMP:Subject": ["Beach"]},
        {"SourceFile": str(b), "File:FileSize": 100},
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


def test_find_field_presence_classifies_each_field(tmp_path: Path) -> None:
    """Each image reports exactly the fields whose tags are populated, across its targets."""
    a = tmp_path / "a.cr3"  # title + description, no keywords
    b = tmp_path / "b.cr3"  # keywords only
    c = tmp_path / "c.cr3"  # nothing
    for path in (a, b, c):
        path.write_text("x")

    fake_helper = _fake_helper(
        [
            {"SourceFile": str(a), "XMP:Title": "T", "XMP:Description": "D"},
            {"SourceFile": str(b), "IPTC:Keywords": ["Beach"]},
            {"SourceFile": str(c), "File:FileSize": 100},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        presence = find_field_presence([a, b, c])

    assert presence[a] == {FIELD_TITLE, FIELD_DESCRIPTION}
    assert presence[b] == {FIELD_KEYWORDS}
    assert presence[c] == set()


def test_find_field_presence_unions_image_and_sidecar(tmp_path: Path) -> None:
    """A field on the image and another on the sidecar both count for the same photo."""
    image = tmp_path / "a.cr3"
    sidecar = tmp_path / "a.xmp"
    image.write_text("x")
    sidecar.write_text("x")

    fake_helper = _fake_helper(
        [
            {"SourceFile": str(image), "XMP:Title": "T"},
            {"SourceFile": str(sidecar), "XMP:Subject": ["Beach"]},
        ],
    )
    with (
        patch(
            "photo_tagger.metadata.metadata_targets",
            side_effect=lambda _p: [str(image), str(sidecar)],
        ),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        presence = find_field_presence([image])

    assert presence[image] == {FIELD_TITLE, FIELD_KEYWORDS}


def test_find_field_presence_empty_input_skips_exiftool() -> None:
    """An empty input list returns an empty map without invoking exiftool."""
    with patch("photo_tagger.metadata.ExifToolHelper") as helper:
        assert find_field_presence([]) == {}
    helper.assert_not_called()


def test_find_field_presence_degrades_on_exiftool_error(tmp_path: Path) -> None:
    """An exiftool failure yields empty sets for every path rather than raising."""
    img = tmp_path / "a.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_field_presence([img]) == {img: set()}


def test_find_field_presence_skips_paths_with_no_targets(tmp_path: Path) -> None:
    """A path with no readable file or sidecar maps to an empty set and never calls exiftool."""
    ghost = tmp_path / "missing.cr3"  # never created
    helper = _fake_helper()
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_field_presence([ghost]) == {ghost: set()}
    helper.get_tags.assert_not_called()


def test_find_field_presence_ignores_unrecognized_source_file(tmp_path: Path) -> None:
    """A result block whose SourceFile maps to no input path is ignored, not crashed on."""
    img = tmp_path / "a.cr3"
    img.write_text("x")
    helper = _fake_helper([{"SourceFile": "/elsewhere/other.cr3", "XMP:Title": "T"}])
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_field_presence([img]) == {img: set()}


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

    assert context.existing_keywords.subject == ["Beach", "Sunset"]
    assert context.existing_keywords.hierarchical == ["Animal|Bird"]
    assert context.location_tags == {"XMP-photoshop:Country": "Portugal"}
    assert context.camera_info["EXIF:Model"] == "Canon EOS R5"
    assert context.gps_position == "38.7 N, 9.1 W"
    # The whole point of batching is one IPC call - assert exactly one.
    assert fake_helper.get_tags.call_count == 1


def test_read_image_context_includes_content_hash_when_requested(tmp_path: Path) -> None:
    """include_content_hash adds ImageDataHash to the one read and surfaces it on the context."""
    img = tmp_path / "img.cr3"
    img.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    fake_helper.get_tags.return_value = [
        {"SourceFile": str(img), "File:ImageDataHash": "deadbeef"},
    ]

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(img, include_content_hash=True)

    assert context.content_hash == "deadbeef"
    # The hash tag and api option ride along on the single batched call.
    kwargs = fake_helper.get_tags.call_args.kwargs
    assert "ImageDataHash" in kwargs["tags"]
    assert kwargs["params"] == ["-api", "ImageHashType=SHA256"]


def test_read_image_context_omits_content_hash_by_default(tmp_path: Path) -> None:
    """Without the flag, no ImageDataHash is requested and content_hash stays None."""
    img = tmp_path / "img.cr3"
    img.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    fake_helper.get_tags.return_value = [{"XMP:Subject": ["Beach"]}]

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(img)

    assert context.content_hash is None
    kwargs = fake_helper.get_tags.call_args.kwargs
    assert "ImageDataHash" not in kwargs["tags"]
    assert kwargs["params"] == []


def test_read_image_context_content_hash_none_when_unsupported(tmp_path: Path) -> None:
    """A format exiftool cannot hash returns no ImageDataHash, so content_hash stays None."""
    img = tmp_path / "img.cr3"
    img.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    # The hash was requested, but exiftool returned no ImageDataHash for this format.
    fake_helper.get_tags.return_value = [{"SourceFile": str(img), "XMP:Subject": ["Beach"]}]

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(img, include_content_hash=True)

    assert context.content_hash is None


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
    assert context.existing_keywords.subject == ["Bird", "Beach"]


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

    assert context.existing_keywords == KeywordSet()
    assert context.gps_position is None
    assert context.camera_info == {}
    fake_helper.get_tags.assert_not_called()


def test_build_contextual_prompt_joins_city_and_country() -> None:
    """Both city and country survive together as a single 'City, Country' hint."""
    prompt = build_contextual_prompt(
        "Analyze.",
        [],
        {
            "XMP-photoshop:City": "Barcelona",
            "XMP-photoshop:Country": "Spain",
        },
        {},
    )
    assert "- Location: Barcelona, Spain" in prompt


def test_build_contextual_prompt_uses_iptc_location_fallback() -> None:
    """IPTC-only photos still surface their place name through the legacy fields."""
    prompt = build_contextual_prompt(
        "Analyze.",
        [],
        {
            "IPTC:City": "Lisbon",
            "IPTC:Country-PrimaryLocationName": "Portugal",
        },
        {},
    )
    assert "- Location: Lisbon, Portugal" in prompt


def test_build_contextual_prompt_falls_back_to_single_field() -> None:
    """If only one of city/country is set, the line still appears."""
    only_country = build_contextual_prompt(
        "Analyze.",
        [],
        {"XMP-photoshop:Country": "Norway"},
        {},
    )
    assert "- Location: Norway" in only_country
    only_city = build_contextual_prompt(
        "Analyze.",
        [],
        {"IPTC:City": "Tokyo"},
        {},
    )
    assert "- Location: Tokyo" in only_city


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


def test_managed_helper_yields_supplied_helper_without_creating_one() -> None:
    """When an open helper is passed in, managed_helper reuses it and never spins up its own."""
    existing = MagicMock()
    with (
        patch("photo_tagger.metadata.ExifToolHelper") as factory,
        managed_helper(existing) as helper,
    ):
        assert helper is existing
    factory.assert_not_called()


def test_read_existing_keywords_returns_empty_on_exiftool_error(tmp_path: Path) -> None:
    """A failure inside exiftool is logged and yields an empty KeywordSet."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        result = read_existing_keywords(img)

    assert result == KeywordSet()


def test_find_tagged_images_returns_empty_when_no_block_is_tagged(tmp_path: Path) -> None:
    """Blocks without any indicator tag leave the tagged set empty."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper([{"SourceFile": str(img), "File:FileSize": 100}])

    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_tagged_images([img]) == set()


def test_read_location_tags_returns_empty_without_targets(tmp_path: Path) -> None:
    """A path with no readable file or sidecar skips exiftool entirely."""
    ghost = tmp_path / "ghost.cr3"
    with patch("photo_tagger.metadata.metadata_targets", return_value=[]):
        assert read_location_tags(ghost) == {}


def test_read_location_tags_collects_present_values(tmp_path: Path) -> None:
    """Non-empty location tags are formatted and returned."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper([{"XMP-photoshop:City": "Lisbon", "XMP-photoshop:Country": ""}])

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_location_tags(img) == {"XMP-photoshop:City": "Lisbon"}


def test_read_location_tags_returns_empty_on_exiftool_error(tmp_path: Path) -> None:
    """An exiftool failure while reading location tags is swallowed."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = TypeError("boom")

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_location_tags(img) == {}


def test_read_gps_coordinates_returns_empty_without_targets(tmp_path: Path) -> None:
    """A path with no targets skips exiftool and returns no coordinates."""
    ghost = tmp_path / "ghost.cr3"
    with patch("photo_tagger.metadata.metadata_targets", return_value=[]):
        assert read_gps_coordinates(ghost) == {}


def test_read_gps_coordinates_returns_position_when_present(tmp_path: Path) -> None:
    """A non-empty GPS position is formatted and returned under 'position'."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper([{"Composite:GPSPosition": "38.7 N, 9.1 W"}])

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_gps_coordinates(img) == {"position": "38.7 N, 9.1 W"}


def test_read_gps_coordinates_returns_empty_on_exiftool_error(tmp_path: Path) -> None:
    """An exiftool failure while reading GPS is swallowed."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_gps_coordinates(img) == {}


def test_read_image_context_returns_empty_on_exiftool_error(tmp_path: Path) -> None:
    """A failure during the batched read yields a blank ImageContext."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        context = read_image_context(img)

    assert context.existing_keywords == KeywordSet()
    assert context.gps_position is None


def test_write_metadata_with_backup_omits_overwrite_param(tmp_path: Path) -> None:
    """With backup=True the -overwrite_original param is not passed to exiftool."""
    img = tmp_path / "img.cr3"
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok = write_metadata(img, KeywordSet(subject=["Bird"]), backup=True)

    assert ok is True
    _, kwargs = helper.set_tags.call_args
    assert "params" not in kwargs


def test_write_metadata_returns_false_on_exiftool_error(tmp_path: Path) -> None:
    """A write failure is logged and reported as False rather than raising."""
    img = tmp_path / "img.cr3"
    helper = _fake_helper()
    helper.set_tags.side_effect = ValueError("boom")

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert write_metadata(img, KeywordSet(subject=["Bird"])) is False


# ---------------------------------------------------------------------------
# read_caption
# ---------------------------------------------------------------------------


def test_read_caption_reads_title_and_description_with_fallbacks(tmp_path: Path) -> None:
    """The IPTC/EXIF fall-back tags are used when the preferred XMP tags are absent."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper(
        [{"IPTC:ObjectName": "Fallback Title", "EXIF:ImageDescription": "Fallback caption."}],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_caption(img) == ("Fallback Title", "Fallback caption.")


def test_read_caption_returns_none_when_absent(tmp_path: Path) -> None:
    """A file with neither title nor description yields (None, None)."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper([{"SourceFile": str(img)}])
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_caption(img) == (None, None)


def test_read_caption_returns_none_when_no_targets(tmp_path: Path) -> None:
    """A missing file (no metadata targets) returns (None, None) without calling exiftool."""
    with patch("photo_tagger.metadata.metadata_targets", return_value=[]):
        assert read_caption(tmp_path / "ghost.cr3") == (None, None)


def test_read_caption_returns_none_on_exiftool_error(tmp_path: Path) -> None:
    """A failure inside exiftool is logged and yields (None, None)."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_caption(img) == (None, None)


# ---------------------------------------------------------------------------
# _format_location / _camera_lines helpers
# ---------------------------------------------------------------------------


def test_format_location_returns_none_for_empty_tags() -> None:
    """Empty location tags produce None, not a spurious 'None' string."""
    from photo_tagger.metadata import _format_location  # noqa: PLC0415

    assert _format_location({}) is None


def test_camera_lines_renders_partial_info() -> None:
    """Only present camera fields appear; missing ones are skipped."""
    from photo_tagger.metadata import _camera_lines  # noqa: PLC0415

    lines = _camera_lines({"EXIF:Model": "Canon EOS R5"})
    assert lines == ["- Camera: Canon EOS R5"]

    lines_empty = _camera_lines({})
    assert lines_empty == []


def test_select_camera_fields_extracts_model_lens_date() -> None:
    """The camera selector pulls model/lens/date, returning None for any that are absent."""
    from photo_tagger.metadata import select_camera_fields  # noqa: PLC0415

    info = {
        "EXIF:Model": "Canon EOS R5",
        "EXIF:LensModel": "RF 100mm Macro",
        "EXIF:DateTimeOriginal": "2024:05:01 10:00:00",
    }
    assert select_camera_fields(info) == (
        "Canon EOS R5",
        "RF 100mm Macro",
        "2024:05:01 10:00:00",
    )
    assert select_camera_fields({"EXIF:Model": "Canon EOS R5"}) == (
        "Canon EOS R5",
        None,
        None,
    )
    assert select_camera_fields({}) == (None, None, None)


def test_select_location_prefers_xmp_then_falls_back_to_iptc() -> None:
    """City/country come from XMP-photoshop first, then the IPTC variants."""
    from photo_tagger.metadata import select_location  # noqa: PLC0415

    xmp = {"XMP-photoshop:City": "Hamburg", "XMP-photoshop:Country": "Germany"}
    assert select_location(xmp) == ("Hamburg", "Germany")

    iptc = {"IPTC:City": "Berlin", "IPTC:Country-PrimaryLocationName": "Germany"}
    assert select_location(iptc) == ("Berlin", "Germany")

    assert select_location({}) == (None, None)


# ---------------------------------------------------------------------------
# read_metadata_sources
# ---------------------------------------------------------------------------


def test_read_metadata_sources_reports_image_and_sidecar(tmp_path: Path) -> None:
    """Targets carrying indicator tags are reported by source (image vs sidecar)."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    xmp = str(img.with_suffix(".xmp"))
    helper = _fake_helper(
        [
            {"SourceFile": str(img), "XMP:Subject": ["Bird"]},
            {"SourceFile": xmp, "XMP:Title": "A title"},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img), xmp]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_metadata_sources(img) == [SOURCE_IMAGE, SOURCE_SIDECAR]


def test_read_metadata_sources_deduplicates_same_source(tmp_path: Path) -> None:
    """Two blocks from the same source collapse to a single label."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper(
        [
            {"SourceFile": str(img), "XMP:Subject": ["Bird"]},
            {"SourceFile": str(img), "XMP:Title": "A title"},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_metadata_sources(img) == [SOURCE_IMAGE]


def test_read_metadata_sources_skips_targets_without_metadata(tmp_path: Path) -> None:
    """A target present but free of indicator tags is not reported as a source."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper([{"SourceFile": str(img), "EXIF:Make": "Canon"}])
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_metadata_sources(img) == []


def test_read_metadata_sources_empty_when_no_targets(tmp_path: Path) -> None:
    """A missing file yields no sources without calling exiftool."""
    with patch("photo_tagger.metadata.metadata_targets", return_value=[]):
        assert read_metadata_sources(tmp_path / "ghost.cr3") == []


def test_read_metadata_sources_returns_empty_on_exiftool_error(tmp_path: Path) -> None:
    """An exiftool failure is logged and yields no sources."""
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_metadata_sources(img) == []
