"""Tests for metadata helpers that don't need a real exiftool binary."""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from exiftool.exceptions import ExifToolExecuteError

from photo_tagger.config import TAG_EXIF_IMAGE_DESCRIPTION
from photo_tagger.fields import SHOWN_FIELDS
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
    capture_year,
    find_field_presence,
    find_tagged_images,
    format_metadata_value,
    iter_field_presence,
    managed_helper,
    plan_writes,
    prompt_with_hint,
    read_all_tags,
    read_caption,
    read_caption_values,
    read_capture_times,
    read_fields,
    read_image_context,
    read_keyword_sets,
    read_metadata_sources,
    use_sidecar_for,
    write_metadata,
    write_metadata_everywhere,
)
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _fake_helper(get_tags_result: object = None) -> MagicMock:
    """Build a MagicMock standing in for an open ExifToolHelper context manager."""
    helper = MagicMock()
    helper.__enter__.return_value = helper
    helper.__exit__.return_value = False
    if get_tags_result is not None:
        helper.get_tags.return_value = get_tags_result
    return helper


def test_managed_helper_uses_configured_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With PHOTO_TAGGER_EXIFTOOL set, the helper is built with that explicit binary."""
    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "/opt/et/exiftool")
    helper = _fake_helper()
    with (
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper) as ctor,
        managed_helper(None) as opened,
    ):
        assert opened is helper
    ctor.assert_called_once_with(executable="/opt/et/exiftool")


def test_managed_helper_defaults_to_path_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override, the helper is built with no executable, so pyexiftool searches PATH."""
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    helper = _fake_helper()
    with (
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper) as ctor,
        managed_helper(None),
    ):
        pass
    ctor.assert_called_once_with()


def test_managed_helper_yields_supplied_helper_without_opening() -> None:
    """A caller-supplied helper is reused as-is; no new ExifToolHelper is constructed."""
    supplied = MagicMock()
    with (
        patch("photo_tagger.metadata.ExifToolHelper") as ctor,
        managed_helper(supplied) as opened,
    ):
        assert opened is supplied
    ctor.assert_not_called()


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


def test_prompt_with_hint_appends_an_authoritative_note() -> None:
    """A hint becomes a photographer's note the model is told to trust over its own reading."""
    out = prompt_with_hint("Analyze.", "The animal is a deer")
    assert out.startswith("Analyze.")
    assert "Photographer's note about this photo: The animal is a deer" in out
    assert "trust the note" in out


def test_prompt_with_hint_blank_returns_base_unchanged() -> None:
    """No hint (None, empty, or whitespace) leaves the base prompt untouched."""
    assert prompt_with_hint("Analyze.", None) == "Analyze."
    assert prompt_with_hint("Analyze.", "") == "Analyze."
    assert prompt_with_hint("Analyze.", "   ") == "Analyze."


def test_prompt_with_hint_composes_with_the_contextual_prompt() -> None:
    """The note sits between the base instruction and the Existing Metadata block."""
    out = build_contextual_prompt(
        prompt_with_hint("Analyze.", "A deer"),
        ["Garden"],
        {},
        {},
    )
    assert out.index("Analyze.") < out.index("Photographer's note")
    assert out.index("Photographer's note") < out.index("Existing Metadata:")


def test_build_write_payload_produces_lightroom_compatible_keys() -> None:
    """Subjects mirror to IPTC:Keywords, weighted flat subjects, and titles fan out to two tags."""
    payload = _build_write_payload(
        KeywordSet(subject=["Beach"], hierarchical=["Animal|Bird"], weighted=["Beach"]),
        description="A short desc.",
        title="A title",
        use_sidecar=True,
    )
    assert payload["XMP-dc:Subject"] == ["Beach"]
    assert payload["IPTC:Keywords"] == ["Beach"]
    assert payload["XMP-lr:HierarchicalSubject"] == ["Animal|Bird"]
    assert payload["XMP:WeightedFlatSubject"] == ["Beach"]
    assert payload["XMP-dc:Description"] == "A short desc."
    assert payload["XMP-tiff:ImageDescription"] == "A short desc."
    assert payload["XMP-dc:Title"] == "A title"
    assert payload["IPTC:ObjectName"] == "A title"


def test_build_write_payload_skips_blank_values() -> None:
    """Empty keyword lists / blank title and description produce no entries."""
    payload = _build_write_payload(KeywordSet(), description=None, title=None, use_sidecar=True)
    assert payload == {}


def test_plan_writes_lists_one_target_per_mode(tmp_path: Path) -> None:
    """Every mode but "both" writes one file; "both" writes the photo and a sidecar beside it."""
    img = tmp_path / "a.dng"
    img.write_text("x")

    assert [t.path.name for t in plan_writes(img, "all")] == ["a.xmp"]
    assert [t.path.name for t in plan_writes(img, "none")] == ["a.dng"]
    assert [t.path.name for t in plan_writes(img, "raw")] == ["a.xmp"]
    both = plan_writes(img, "both")
    assert [t.path.name for t in both] == ["a.dng", "a.xmp"]
    # The payload differs per target, so each one has to say which it is.
    assert [t.use_sidecar for t in both] == [False, True]
    # Sampled before the write: the photo is there, the sidecar is not.
    assert [t.existed for t in both] == [True, False]


def test_write_metadata_everywhere_writes_both_targets(tmp_path: Path) -> None:
    """A "both" save calls the writer once per target and reports what landed, for undo."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok, written = write_metadata_everywhere(
            img,
            KeywordSet(subject=["Bird"]),
            description="A caption.",
            sidecar_mode="both",
        )

    assert ok is True
    assert [t.path.name for t in written] == ["a.dng", "a.xmp"]
    # The photo gets the real EXIF tag; the sidecar holds XMP only.
    payloads = [call.kwargs["tags"] for call in helper.set_tags.call_args_list]
    assert TAG_EXIF_IMAGE_DESCRIPTION in payloads[0]
    assert TAG_EXIF_IMAGE_DESCRIPTION not in payloads[1]


def test_write_metadata_everywhere_reports_a_half_written_photo(tmp_path: Path) -> None:
    """One target failing is not a clean save, but what did land still has to reach the journal."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    helper = _fake_helper()
    helper.set_tags.side_effect = [None, ValueError("boom")]

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok, written = write_metadata_everywhere(
            img,
            KeywordSet(subject=["Bird"]),
            sidecar_mode="both",
        )

    assert ok is False
    assert [t.path.name for t in written] == ["a.dng"]


def test_build_write_payload_sets_the_exif_description_only_when_embedding() -> None:
    """A sidecar holds XMP only; the photo itself also gets the real IFD0 tag replaced."""
    sidecar = _build_write_payload(
        KeywordSet(),
        description="A short desc.",
        title=None,
        use_sidecar=True,
    )
    assert TAG_EXIF_IMAGE_DESCRIPTION not in sidecar
    embedded = _build_write_payload(
        KeywordSet(),
        description="A short desc.",
        title=None,
        use_sidecar=False,
    )
    assert embedded[TAG_EXIF_IMAGE_DESCRIPTION] == "A short desc."


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


def test_find_tagged_images_credits_every_image_sharing_a_sidecar(tmp_path: Path) -> None:
    """
    A RAW+JPEG pair with the same stem shares one sidecar; both must be reported as tagged.

    IMG_0001.cr3 and IMG_0001.jpg both resolve to IMG_0001.xmp. When that shared sidecar carries an
    indicator tag, both images count as tagged, not just whichever the target index happened to keep
    last.
    """
    raw = tmp_path / "IMG_0001.cr3"
    jpg = tmp_path / "IMG_0001.jpg"
    sidecar = tmp_path / "IMG_0001.xmp"
    raw.write_text("x")
    jpg.write_text("x")
    sidecar.write_text("x")

    fake_helper = _fake_helper([{"SourceFile": str(sidecar), "XMP:Subject": ["Beach"]}])
    with patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper):
        tagged = find_tagged_images([raw, jpg])

    assert tagged == {raw, jpg}


def test_find_tagged_images_returns_empty_for_empty_input() -> None:
    """An empty input list short-circuits without invoking exiftool."""
    with patch("photo_tagger.metadata.ExifToolHelper") as helper:
        assert find_tagged_images([]) == set()
    helper.assert_not_called()


def test_find_tagged_images_degrades_to_empty_on_exiftool_error(tmp_path: Path) -> None:
    """An exiftool failure during the tagged check degrades to 'nothing tagged', never raising."""
    a = tmp_path / "a.cr3"
    a.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    # ValueError is one of the exiftool error types find_tagged_images guards against.
    fake_helper.get_tags.side_effect = ValueError("exiftool unavailable")

    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        assert find_tagged_images([a]) == set()


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


def test_find_field_presence_credits_every_image_sharing_a_sidecar(tmp_path: Path) -> None:
    """The shared-sidecar fix applies to field presence too, not just the tagged/untagged check."""
    raw = tmp_path / "IMG_0001.cr3"
    jpg = tmp_path / "IMG_0001.jpg"
    sidecar = tmp_path / "IMG_0001.xmp"
    raw.write_text("x")
    jpg.write_text("x")
    sidecar.write_text("x")

    fake_helper = _fake_helper([{"SourceFile": str(sidecar), "XMP:Title": "T"}])
    with patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper):
        presence = find_field_presence([raw, jpg])

    assert presence[raw] == {FIELD_TITLE}
    assert presence[jpg] == {FIELD_TITLE}


def test_find_field_presence_empty_input_skips_exiftool() -> None:
    """An empty input list returns an empty map without invoking exiftool."""
    with patch("photo_tagger.metadata.ExifToolHelper") as helper:
        assert find_field_presence([]) == {}
    helper.assert_not_called()


def test_find_field_presence_degrades_on_exiftool_error(tmp_path: Path) -> None:
    """
    An exiftool failure yields an empty dict rather than raising.

    Regression test: the failure path used to return empty sets for every path, which callers read
    as "scanned, nothing found" and the GUI then wrongly showed every photo as untagged. An empty
    dict means "could not read", leaving the per-photo state unknown.
    """
    img = tmp_path / "a.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_field_presence([img]) == {}


def _raise_execute_error_with_payload(blocks: list[dict[str, object]]) -> ExifToolExecuteError:
    """Build the error pyexiftool raises when exiftool exits 1 but still produced JSON."""
    return ExifToolExecuteError(1, json.dumps(blocks), "Error: File format error - bad.jpg\n", [])


def test_find_tagged_images_salvages_batch_with_one_bad_file(tmp_path: Path) -> None:
    """
    One corrupt file in the batch must not wipe out the whole folder's tagged check.

    Regression test: exiftool exits 1 when any file has a format error, pyexiftool raises, and the
    old code returned set() so every already-tagged photo was reported untagged (and re-run). The
    JSON for the healthy files is still on the error's stdout; use it.
    """
    good = tmp_path / "good.cr3"
    bad = tmp_path / "bad.cr3"
    for path in (good, bad):
        path.write_text("x")

    helper = _fake_helper()
    helper.get_tags.side_effect = _raise_execute_error_with_payload(
        [
            {"SourceFile": str(good), "XMP:Subject": ["Beach"]},
            {"SourceFile": str(bad)},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_tagged_images([good, bad]) == {good}


def test_find_field_presence_salvages_batch_with_one_bad_file(tmp_path: Path) -> None:
    """The field-presence scan also survives a single corrupt file in the batch."""
    good = tmp_path / "good.cr3"
    bad = tmp_path / "bad.cr3"
    for path in (good, bad):
        path.write_text("x")

    helper = _fake_helper()
    helper.get_tags.side_effect = _raise_execute_error_with_payload(
        [
            {"SourceFile": str(good), "XMP:Title": "T"},
            {"SourceFile": str(bad)},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        presence = find_field_presence([good, bad])

    assert presence[good] == {FIELD_TITLE}
    assert presence[bad] == set()


def test_batched_get_tags_falls_back_when_nothing_to_salvage(tmp_path: Path) -> None:
    """An execute error with no JSON payload still degrades via the callers' guard."""
    img = tmp_path / "a.cr3"
    img.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ExifToolExecuteError(1, "", "exiftool blew up", [])
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert find_tagged_images([img]) == set()
        assert find_field_presence([img]) == {}


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


def _presence_images(tmp_path: Path, count: int) -> list[Path]:
    """Create *count* stand-in image files for the chunked presence reads."""
    paths = [tmp_path / f"{index:03d}.cr3" for index in range(count)]
    for path in paths:
        path.write_text("x")
    return paths


def test_iter_field_presence_splits_the_read_into_batches(tmp_path: Path) -> None:
    """The chunked read reports a batch at a time, each one counting the images it covered."""
    paths = _presence_images(tmp_path, 5)
    helper = _fake_helper()
    helper.get_tags.side_effect = lambda files, tags: [  # noqa: ARG005 - tags are fixed here
        {"SourceFile": name, "XMP:Title": "T"} for name in files
    ]
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper) as ctor,
    ):
        batches = list(iter_field_presence(paths, batch_size=2))

    assert [batch.scanned for batch in batches] == [2, 2, 1]
    assert all(fields == {FIELD_TITLE} for batch in batches for fields in batch.presence.values())
    # One exiftool process for the whole run, not one per batch: the split must stay cheap.
    ctor.assert_called_once()


def test_iter_field_presence_counts_a_failed_batch_and_keeps_going(tmp_path: Path) -> None:
    """A batch exiftool cannot read reports no photos but still advances the caller's progress."""
    paths = _presence_images(tmp_path, 4)
    helper = _fake_helper()
    # The first batch blows up, the second reads fine: one call per batch, in order.
    helper.get_tags.side_effect = [
        ValueError("boom"),
        [{"SourceFile": str(path), "XMP:Title": "T"} for path in paths[2:]],
    ]
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        batches = list(iter_field_presence(paths, batch_size=2))

    assert [batch.scanned for batch in batches] == [2, 2]
    assert batches[0].presence == {}  # unreadable: unknown, not "carries nothing"
    assert set(batches[1].presence) == set(paths[2:])


def test_iter_field_presence_stops_reading_when_abandoned(tmp_path: Path) -> None:
    """Dropping the iterator early leaves the rest of the library unread; that is the cancel."""
    paths = _presence_images(tmp_path, 6)
    helper = _fake_helper([])
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        batches = iter_field_presence(paths, batch_size=2)
        next(batches)
        batches.close()

    assert helper.get_tags.call_count == 1
    helper.__exit__.assert_called_once()  # the shared process is torn down on the way out


def test_iter_field_presence_pulls_its_input_one_batch_at_a_time(tmp_path: Path) -> None:
    """
    The generator it is fed is consumed batch by batch, not drained up front.

    That is what lets the GUI's scan skip a folder the user removed while the read was running.
    """
    paths = _presence_images(tmp_path, 6)
    pulled: list[Path] = []
    helper = _fake_helper([])

    def feed() -> Iterator[Path]:
        for path in paths:
            pulled.append(path)
            yield path

    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        batches = iter_field_presence(feed(), batch_size=2)
        next(batches)
        assert pulled == paths[:2]  # only the batch that was read, not the whole list
        next(batches)
        assert pulled == paths[:4]
        batches.close()


def test_iter_field_presence_degrades_when_exiftool_cannot_start(tmp_path: Path) -> None:
    """With no exiftool binary the read yields nothing, so every photo stays unknown."""
    paths = _presence_images(tmp_path, 2)
    with patch("photo_tagger.metadata.ExifToolHelper", side_effect=FileNotFoundError("exiftool")):
        assert list(iter_field_presence(paths)) == []
        assert find_field_presence(paths) == {}


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


def test_read_image_context_skips_blank_camera_location_and_gps_tags(tmp_path: Path) -> None:
    """
    A tag that is present but blank is treated as absent, so a later block can still fill it in.

    Collecting the blank value instead put an empty string in camera_info, which the prompt then
    rendered as a dangling "- Camera:" line, and blocked the sidecar's real value behind it.
    """
    img = tmp_path / "img.cr3"
    img.write_text("x")

    fake_helper = MagicMock()
    fake_helper.__enter__.return_value = fake_helper
    fake_helper.__exit__.return_value = False
    fake_helper.get_tags.return_value = [
        {"EXIF:Model": "  ", "XMP-photoshop:City": [], "Composite:GPSPosition": [" "]},
        {"EXIF:Model": "Canon EOS R5", "XMP-photoshop:City": "Lisbon"},
        {"Composite:GPSPosition": "38.7 N, 9.1 W"},
    ]

    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        context = read_image_context(img)

    assert context.camera_info == {"EXIF:Model": "Canon EOS R5"}
    assert context.location_tags == {"XMP-photoshop:City": "Lisbon"}
    assert context.gps_position == "38.7 N, 9.1 W"


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


def test_write_metadata_without_backup_passes_overwrite_original(tmp_path: Path) -> None:
    """Backup=False must reach exiftool as -overwrite_original, or _original files pile up."""
    img = tmp_path / "img.cr3"
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok = write_metadata(img, KeywordSet(subject=["Bird"]), backup=False)

    assert ok is True
    assert helper.set_tags.call_args.kwargs["params"] == ["-overwrite_original"]


def test_write_metadata_targets_the_sidecar_by_default(tmp_path: Path) -> None:
    """
    use_sidecar=True (the production default) writes to img.xmp, never the original.

    This is the non-destructive promise the whole tool is built on: a regression that targeted the
    image would modify originals on every default run.
    """
    img = tmp_path / "img.cr3"
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        write_metadata(img, KeywordSet(subject=["Bird"]), use_sidecar=True)
        write_metadata(img, KeywordSet(subject=["Bird"]), use_sidecar=False)

    sidecar_call, embed_call = helper.set_tags.call_args_list
    assert sidecar_call.kwargs["files"] == [str(img.with_suffix(".xmp"))]
    assert embed_call.kwargs["files"] == [str(img)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shot.dng", True),
        ("shot.CR3", True),
        ("shot.nef", True),
        # A RAW format on no list here (Phase One) still counts as RAW, so its bytes are left
        # alone. Unknown means cautious, not embedded.
        ("shot.iiq", True),
        ("shot.jpg", False),
        ("shot.JPEG", False),
        ("shot.tif", False),
        ("shot.heic", False),
    ],
)
def test_use_sidecar_for_raw_mode_splits_by_file_type(
    tmp_path: Path,
    name: str,
    *,
    expected: bool,
) -> None:
    """The 'raw' mode is the mixed-folder answer: sidecars for RAW, embedded for the rest."""
    assert use_sidecar_for(tmp_path / name, "raw") is expected


@pytest.mark.parametrize("name", ["shot.dng", "shot.jpg"])
def test_use_sidecar_for_all_and_none_ignore_the_file_type(tmp_path: Path, name: str) -> None:
    """Outside 'raw' the mode alone decides, so both modes answer the same for any photo."""
    assert use_sidecar_for(tmp_path / name, "all") is True
    assert use_sidecar_for(tmp_path / name, "none") is False


def test_read_caption_prefers_xmp_over_the_fallback_tags(tmp_path: Path) -> None:
    """
    With both tag families present (even across blocks), the XMP value wins.

    Guards the read priority: swapping the loop nesting in _first_tag_value would let a sidecar's
    IPTC/EXIF values shadow the preferred XMP-dc ones.
    """
    img = tmp_path / "a.cr3"
    img.write_text("x")
    helper = _fake_helper(
        [
            {"IPTC:ObjectName": "Fallback Title", "EXIF:ImageDescription": "Fallback caption."},
            {"XMP:Title": "Preferred Title", "XMP:Description": "Preferred caption."},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_caption(img) == ("Preferred Title", "Preferred caption.")


def test_read_caption_ignores_an_empty_preferred_tag(tmp_path: Path) -> None:
    """
    An empty or blank preferred tag falls through to the fall-back rather than shadowing it.

    exiftool renders an empty ``rdf:Bag`` as ``[]`` and a blank one as ``[" "]``. Comparing the
    value against ``""`` counts both as content, so the photo's real IPTC/EXIF title came back as an
    empty string.
    """
    img = tmp_path / "img.cr3"
    img.write_text("x")
    helper = _fake_helper(
        [
            {
                "XMP:Title": [],
                "XMP:Description": [" "],
                "IPTC:ObjectName": "Fallback Title",
                "EXIF:ImageDescription": "Fallback caption.",
            },
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        assert read_caption(img) == ("Fallback Title", "Fallback caption.")


def test_write_metadata_returns_false_on_exiftool_error(tmp_path: Path) -> None:
    """A write failure is logged and reported as False rather than raising."""
    img = tmp_path / "img.cr3"
    helper = _fake_helper()
    helper.set_tags.side_effect = ValueError("boom")

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert write_metadata(img, KeywordSet(subject=["Bird"])) is False


def _execute_error(stderr: str) -> ExifToolExecuteError:
    """Build the error pyexiftool raises when exiftool exits non-zero, carrying *stderr*."""
    return ExifToolExecuteError(1, "", stderr, [])


def test_write_metadata_retries_a_minor_error_ignoring_it(tmp_path: Path) -> None:
    """
    A write exiftool refuses over a minor error is tried once more with -m.

    Some cameras write maker notes exiftool cannot parse (their offsets already wrong on the file
    straight out of the camera), and it refuses the whole write rather than move a block whose
    offsets it cannot fix. Without the retry, embedding metadata in those photos is impossible.
    """
    img = tmp_path / "img.dng"
    helper = _fake_helper()
    helper.set_tags.side_effect = [
        _execute_error("Error: [minor] Maker notes could not be parsed - img.dng\n"),
        None,
    ]

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        written = write_metadata(
            img,
            KeywordSet(subject=["Bird"]),
            backup=False,
            use_sidecar=False,
        )

    assert written is True
    assert helper.set_tags.call_count == 2  # noqa: PLR2004 - the refused write plus its one retry.
    retry_params = helper.set_tags.call_args.kwargs["params"]
    # The retry keeps whatever the first attempt asked for and only adds the waiver.
    assert retry_params == ["-overwrite_original", "-m"]


def test_write_metadata_does_not_retry_an_error_m_would_not_waive(tmp_path: Path) -> None:
    """Only exiftool's own "[minor]" marker waives a failure; anything else fails once and stays."""
    img = tmp_path / "img.dng"
    helper = _fake_helper()
    helper.set_tags.side_effect = _execute_error("Error: File format error - img.dng\n")

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert write_metadata(img, KeywordSet(subject=["Bird"])) is False

    assert helper.set_tags.call_count == 1


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


def test_read_caption_prefers_the_sidecar_over_the_image(tmp_path: Path) -> None:
    """
    The sidecar's caption wins, whatever order exiftool returns the blocks in.

    The image block comes first (metadata_targets lists the photo before its sidecar), so reading in
    block order left a camera's placeholder description shadowing the one photo-tagger had just
    written into the sidecar next to it.
    """
    img = tmp_path / "a.dng"
    img.write_text("x")
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("x")
    helper = _fake_helper(
        [
            {"SourceFile": str(img), "XMP:Description": "default", "XMP:Title": "Camera Title"},
            {"SourceFile": str(sidecar), "XMP:Description": "A real caption."},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img), str(sidecar)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        title, description = read_caption(img)
    assert description == "A real caption."
    # The sidecar has no title, so the image's is still what the photo carries.
    assert title == "Camera Title"


def test_read_caption_values_reports_what_the_sidecar_shadows(tmp_path: Path) -> None:
    """Both files' captions come back, the one photo-tagger uses first."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("x")
    helper = _fake_helper(
        [
            {"SourceFile": str(img), "XMP:Description": "default"},
            {"SourceFile": str(sidecar), "XMP:Description": "A real caption."},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img), str(sidecar)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        found = read_caption_values(img)

    assert [(v.value, v.source) for v in found[FIELD_DESCRIPTION]] == [
        ("A real caption.", SOURCE_SIDECAR),
        ("default", SOURCE_IMAGE),
    ]
    assert found[FIELD_TITLE] == []


def test_read_caption_values_heads_the_list_with_the_winning_tag(tmp_path: Path) -> None:
    """
    Which file answers is decided per tag, so the breakdown has to agree with read_caption.

    A photo's XMP-dc:Title outranks a sidecar carrying only IPTC:ObjectName, and listing the files
    sidecar-first without that rule would report the loser as the value in use.
    """
    img = tmp_path / "a.dng"
    img.write_text("x")
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text("x")
    helper = _fake_helper(
        [
            {"SourceFile": str(img), "XMP:Title": "Preferred Title"},
            {"SourceFile": str(sidecar), "IPTC:ObjectName": "Fallback Title"},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img), str(sidecar)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        found = read_caption_values(img)

    assert [v.value for v in found[FIELD_TITLE]] == ["Preferred Title", "Fallback Title"]
    assert found[FIELD_TITLE][0].source == SOURCE_IMAGE


def test_read_image_context_reads_the_existing_caption(tmp_path: Path) -> None:
    """The batched read also carries the title and description --preserve-* decides on."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    helper = _fake_helper(
        [{"SourceFile": str(img), "XMP:Title": "A title", "XMP:Description": "A caption."}],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        context = read_image_context(img)
    assert context.existing_title == "A title"
    assert context.existing_description == "A caption."


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


def test_read_capture_times_parses_exif_timestamps(tmp_path: Path) -> None:
    """DateTimeOriginal comes back as a naive datetime in the camera's own local time."""
    photo = tmp_path / "a.cr3"
    photo.write_text("x")
    helper = _fake_helper(
        [{"SourceFile": str(photo), "EXIF:DateTimeOriginal": "2026:05:01 09:15:30"}],
    )

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        times = read_capture_times([photo])

    # Naive by definition: EXIF records the camera's own clock, with no zone.
    assert times == {photo: datetime(2026, 5, 1, 9, 15, 30)}  # noqa: DTZ001


def test_read_capture_times_skips_missing_and_malformed_values(tmp_path: Path) -> None:
    """A blank or malformed tag leaves the path out, so the caller can fall back to mtime."""
    blank = tmp_path / "blank.cr3"
    broken = tmp_path / "broken.cr3"
    for path in (blank, broken):
        path.write_text("x")
    helper = _fake_helper(
        [
            {"SourceFile": str(blank)},
            {"SourceFile": str(broken), "EXIF:DateTimeOriginal": "0000:00:00 00:00:00"},
            {"EXIF:DateTimeOriginal": "2026:05:01 09:15:30"},
        ],
    )

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert read_capture_times([blank, broken]) == {}


def test_read_capture_times_returns_empty_for_no_existing_files(tmp_path: Path) -> None:
    """Paths that are not files never reach exiftool."""
    with patch("photo_tagger.metadata.ExifToolHelper") as ctor:
        assert read_capture_times([tmp_path / "gone.cr3"]) == {}
    ctor.assert_not_called()


def test_read_capture_times_survives_an_exiftool_error(tmp_path: Path) -> None:
    """An exiftool failure degrades to "no timestamps", not an exception."""
    photo = tmp_path / "a.cr3"
    photo.write_text("x")
    helper = _fake_helper()
    helper.get_tags.side_effect = ValueError("boom")

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert read_capture_times([photo]) == {}


def test_read_capture_times_survives_a_missing_exiftool_binary(tmp_path: Path) -> None:
    """No exiftool on PATH degrades to "no timestamps", so --session-gap falls back to mtime."""
    photo = tmp_path / "a.cr3"
    photo.write_text("x")
    helper = _fake_helper()
    # What pyexiftool raises when it cannot find the binary at all.
    helper.get_tags.side_effect = FileNotFoundError('"exiftool" is not found, on path')

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert read_capture_times([photo]) == {}


def test_write_metadata_reports_a_missing_exiftool_binary(tmp_path: Path) -> None:
    """A write with no exiftool on PATH returns False instead of raising into the caller."""
    photo = tmp_path / "a.cr3"
    photo.write_text("x")
    helper = _fake_helper()
    helper.set_tags.side_effect = FileNotFoundError('"exiftool" is not found, on path')

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        written = write_metadata(photo, title="T", description="D", keywords=KeywordSet())

    assert written is False


def test_read_keyword_sets_merges_image_and_sidecar_keywords(tmp_path: Path) -> None:
    """Every keyword view of a photo comes back in one batched read, sidecar included."""
    image = tmp_path / "a.cr3"
    sidecar = tmp_path / "a.xmp"
    bare = tmp_path / "b.cr3"
    for path in (image, sidecar, bare):
        path.write_text("x")

    fake_helper = _fake_helper(
        [
            {"SourceFile": str(image), "IPTC:Keywords": ["Beach"]},
            {
                "SourceFile": str(sidecar),
                "XMP:Subject": ["beach", "Sunset"],
                "XMP:HierarchicalSubject": ["Nature|Sunset"],
            },
            {"SourceFile": str(bare), "File:FileSize": 100},
        ],
    )
    with (
        patch(
            "photo_tagger.metadata.metadata_targets",
            side_effect=lambda p: [str(p), str(p.with_suffix(".xmp"))],
        ),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        keywords = read_keyword_sets([image, bare])

    assert keywords[image].subject == ["Beach", "Sunset"]
    assert keywords[image].hierarchical == ["Nature|Sunset"]
    assert keywords[bare].is_empty()


def test_read_keyword_sets_reports_an_exiftool_failure_as_an_empty_dict(tmp_path: Path) -> None:
    """A read failure must not look like "read and found nothing", or a census would be wrong."""
    image = tmp_path / "a.cr3"
    image.write_text("x")

    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", side_effect=ValueError("no exiftool")),
    ):
        assert read_keyword_sets([image]) == {}


def test_read_keyword_sets_with_no_paths_reads_nothing(tmp_path: Path) -> None:
    """An empty batch short-circuits before exiftool is opened."""
    assert read_keyword_sets([]) == {}
    with patch("photo_tagger.metadata.metadata_targets", return_value=[]):
        assert read_keyword_sets([tmp_path / "gone.cr3"]) == {tmp_path / "gone.cr3": KeywordSet()}


def test_read_fields_degrades_when_exiftool_cannot_run(tmp_path: Path) -> None:
    """A broken ExifTool reports nothing rather than raising into the command."""
    image = tmp_path / "img.jpg"
    image.write_bytes(b"\xff\xd8stub")

    with patch(
        "photo_tagger.metadata._batched_get_tags",
        side_effect=FileNotFoundError("no exiftool"),
    ):
        assert read_fields(image, SHOWN_FIELDS) == []


def test_read_all_tags_degrades_when_exiftool_cannot_run(tmp_path: Path) -> None:
    """Same for the --all dump: an empty answer, and the failure in the log."""
    image = tmp_path / "img.jpg"
    image.write_bytes(b"\xff\xd8stub")

    with patch(
        "photo_tagger.metadata.managed_helper",
        side_effect=FileNotFoundError("no exiftool"),
    ):
        assert read_all_tags(image) == {}


def test_capture_year_falls_back_to_today_for_a_file_it_cannot_stat(tmp_path: Path) -> None:
    """A copyright notice needs a year for every photo, even one that vanished mid-run."""
    assert capture_year(tmp_path / "ghost.jpg", None) == datetime.now(tz=UTC).astimezone().year
