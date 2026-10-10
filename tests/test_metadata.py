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
    FIELD_COPYRIGHT,
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    SOFTWARE_AGENT,
    SOURCE_IMAGE,
    SOURCE_SIDECAR,
    TAG_EXIF_PROCESSING_SOFTWARE,
    WriteResult,
    _block_has_indicator,
    _build_write_payload,
    _coerce_to_list,
    _dedup_preserving_first_case,
    _value_is_present,
    build_contextual_prompt,
    find_field_presence,
    find_tagged_images,
    first_caption,
    format_metadata_value,
    history_event_param,
    iter_field_presence,
    managed_helper,
    parse_sidecar_overrides,
    plan_writes,
    prompt_with_hint,
    read_all_tags,
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

    from photo_tagger.config import FileTypeTarget, SidecarMode


def read_caption(path: Path) -> tuple[str | None, str | None]:
    """Return the title and description in use, as the detail pane's Existing column shows them."""
    found = read_caption_values(path)
    return (first_caption(found[FIELD_TITLE]), first_caption(found[FIELD_DESCRIPTION]))


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


def test_managed_helper_runs_exiftool_by_its_absolute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    With no override, the helper gets ExifTool's absolute path from PATH, never the bare name.

    A bare name would be looked up in the current directory first on Windows.
    """
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    helper = _fake_helper()
    with (
        patch("photo_tagger.metadata.find_program", return_value="/usr/bin/exiftool"),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper) as ctor,
        managed_helper(None),
    ):
        pass
    ctor.assert_called_once_with(executable="/usr/bin/exiftool")


def test_managed_helper_refuses_an_exiftool_found_nowhere_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to the bare name would let Windows run an exiftool.exe from the photo folder."""
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    with (
        patch("photo_tagger.metadata.find_program", return_value=None),
        patch("photo_tagger.metadata.ExifToolHelper") as ctor,
        pytest.raises(FileNotFoundError, match="exiftool is not on PATH"),
        managed_helper(None),
    ):
        pass
    ctor.assert_not_called()


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


@pytest.mark.parametrize(
    ("mode", "target", "expected"),
    [
        # The Lightroom Classic case: a DNG keeps its metadata inside, the rest gets sidecars.
        ("all", "photo", ["a.dng"]),
        ("none", "sidecar", ["a.xmp"]),
        ("raw", "both", ["a.dng", "a.xmp"]),
        ("both", "photo", ["a.dng"]),
    ],
)
def test_plan_writes_lets_an_override_win_over_the_mode(
    tmp_path: Path,
    mode: SidecarMode,
    target: FileTypeTarget,
    expected: list[str],
) -> None:
    """An override for the photo's extension decides alone, whatever the mode says."""
    img = tmp_path / "a.dng"

    assert [t.path.name for t in plan_writes(img, mode, {".dng": target})] == expected


def test_plan_writes_leaves_other_file_types_to_the_mode(tmp_path: Path) -> None:
    """A DNG override must not move the JPEG beside it, and the suffix match ignores case."""
    overrides: dict[str, FileTypeTarget] = {".dng": "photo"}

    assert [t.path.name for t in plan_writes(tmp_path / "b.jpg", "all", overrides)] == ["b.xmp"]
    assert [t.path.name for t in plan_writes(tmp_path / "C.DNG", "all", overrides)] == ["C.DNG"]


def test_parse_sidecar_overrides_normalizes_the_extension() -> None:
    """
    Case, a leading dot, and spaces do not matter.

    A later entry wins, like a repeated flag.
    """
    entries = ["DNG=photo", " .tif = Both ", "dng=sidecar"]

    assert parse_sidecar_overrides(entries) == {".dng": "sidecar", ".tif": "both"}
    assert parse_sidecar_overrides(None) == {}


@pytest.mark.parametrize("entry", ["dng", "dng=embed", "=photo", ".=photo", "dng="])
def test_parse_sidecar_overrides_refuses_a_malformed_entry(entry: str) -> None:
    """A typo must stop the run: the default would silently write sidecars the catalog ignores."""
    with pytest.raises(ValueError, match="EXT=TARGET"):
        parse_sidecar_overrides([entry])


def test_parse_sidecar_overrides_refuses_the_sidecar_extension() -> None:
    """An .xmp is the sidecar itself, so an override for it can only be a mistake."""
    with pytest.raises(ValueError, match="sidecar of its own"):
        parse_sidecar_overrides(["xmp=photo"])


def test_write_metadata_everywhere_follows_an_override(tmp_path: Path) -> None:
    """The override reaches the writer: an embedded DNG gets the real EXIF tags, not XMP only."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok, written = write_metadata_everywhere(
            img,
            KeywordSet(subject=["Bird"]),
            description="A caption.",
            sidecar_mode="all",
            sidecar_overrides={".dng": "photo"},
        )

    assert ok is True
    assert [t.path.name for t in written] == ["a.dng"]
    assert TAG_EXIF_IMAGE_DESCRIPTION in helper.set_tags.call_args.kwargs["tags"]


def _jpeg_file(path: Path) -> Path:
    """Create a placeholder photo; the fake helper never reads its bytes."""
    path.write_text("x")
    return path


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


def test_write_metadata_everywhere_saves_exif_tags_under_both_without_the_sidecar(
    tmp_path: Path,
) -> None:
    """Under "both", a sidecar with nothing it can hold is skipped, not counted as a failure."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok, written = write_metadata_everywhere(
            img,
            KeywordSet(),
            extra={"EXIF:Copyright": "(c) Ana"},
            sidecar_mode="both",
        )

    assert ok is True
    assert [t.path.name for t in written] == ["a.dng"]
    assert helper.set_tags.call_count == 1


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


def test_write_metadata_skips_a_target_that_already_carries_every_value(tmp_path: Path) -> None:
    """ExifTool rewrites a file even for identical values, which makes rsync copy it again."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    (tmp_path / "a.xmp").write_text("x")
    # A read reports a one-item list as a string and the Marked flag as a boolean.
    helper = _fake_helper([{"XMP:Subject": "Bird", "XMP:Rights": "© 2019 Ada", "XMP:Marked": True}])
    extra = {"XMP-dc:Rights": "© 2019 Ada", "XMP-xmpRights:Marked": "True"}

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        result = write_metadata(img, KeywordSet(subject=["Bird"]), extra=extra)

    assert result == WriteResult.UNCHANGED
    helper.set_tags.assert_not_called()


def test_write_metadata_writes_only_the_tags_that_changed(tmp_path: Path) -> None:
    """A new notice goes on; the keywords the file already has are not written again."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    (tmp_path / "a.xmp").write_text("x")
    helper = _fake_helper([{"XMP:Subject": ["Bird"], "XMP:Rights": "© 2018 Ada"}])

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        result = write_metadata(
            img,
            KeywordSet(subject=["Bird"]),
            extra={"XMP-dc:Rights": "© 2019 Ada"},
        )

    assert result == WriteResult.WRITTEN
    assert helper.set_tags.call_args.kwargs["tags"] == {"XMP-dc:Rights": "© 2019 Ada"}


def test_write_metadata_always_writes_a_tag_it_cannot_compare(tmp_path: Path) -> None:
    """A group a read does not report, or two tags that read back as one, is written anyway."""
    img = _jpeg_file(tmp_path / "a.jpg")
    helper = _fake_helper([{"EXIF:Copyright": "X", "XMP:Rights": "X"}])
    extra = {"IFD0:Copyright": "X", "XMP-dc:Rights": "X", "XMP-other:Rights": "X"}

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        write_metadata(img, KeywordSet(), extra=extra, use_sidecar=False)

    assert set(helper.set_tags.call_args.kwargs["tags"]) == {*extra, TAG_EXIF_PROCESSING_SOFTWARE}


def test_an_embedded_write_names_photo_tagger_as_its_software(tmp_path: Path) -> None:
    """The photo gets ProcessingSoftware and a history entry, as digiKam and Lightroom write."""
    img = _jpeg_file(tmp_path / "a.jpg")
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        write_metadata(img, KeywordSet(subject=["Bird"]), use_sidecar=False)

    kwargs = helper.set_tags.call_args.kwargs
    assert kwargs["tags"][TAG_EXIF_PROCESSING_SOFTWARE] == SOFTWARE_AGENT
    assert "EXIF:Software" not in kwargs["tags"]
    assert kwargs["params"][-1].startswith("-XMP-xmpMM:History+={Action=saved,")


def test_a_sidecar_write_logs_the_save_in_xmp_only(tmp_path: Path) -> None:
    """A sidecar holds XMP only, so the history entry is its whole stamp."""
    img = tmp_path / "a.dng"
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        write_metadata(img, KeywordSet(subject=["Bird"]), use_sidecar=True)

    kwargs = helper.set_tags.call_args.kwargs
    assert TAG_EXIF_PROCESSING_SOFTWARE not in kwargs["tags"]
    assert kwargs["params"][-1].startswith("-XMP-xmpMM:History+=")


def test_a_set_processing_software_wins_over_the_stamp(tmp_path: Path) -> None:
    """A value the user named by hand stays, even when the file already carries it."""
    img = _jpeg_file(tmp_path / "a.jpg")
    helper = _fake_helper([{"EXIF:ProcessingSoftware": "Mine"}])

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        write_metadata(
            img,
            KeywordSet(subject=["Bird"]),
            extra={TAG_EXIF_PROCESSING_SOFTWARE: "Mine"},
            use_sidecar=False,
        )

    assert TAG_EXIF_PROCESSING_SOFTWARE not in helper.set_tags.call_args.kwargs["tags"]


def test_history_event_param_names_the_version_and_the_time() -> None:
    """The entry says who saved, when, and that only the metadata changed."""
    param = history_event_param(datetime(2026, 9, 24, 10, 0, tzinfo=UTC))

    assert param == (
        "-XMP-xmpMM:History+={Action=saved,When=2026-09-24T10:00:00+00:00,"
        f"SoftwareAgent={SOFTWARE_AGENT},Changed=/metadata}}"
    )


def test_write_metadata_everywhere_leaves_an_untouched_target_out_of_the_journal(
    tmp_path: Path,
) -> None:
    """Undo would restore an older backup over a file this save never changed."""
    img = tmp_path / "a.dng"
    img.write_text("x")
    (tmp_path / "a.xmp").write_text("x")
    helper = _fake_helper([{"XMP:Subject": "Bird"}])

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok, written = write_metadata_everywhere(
            img,
            KeywordSet(subject=["Bird"]),
            sidecar_mode="all",
        )

    assert ok is True
    assert written == []


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


def test_find_field_presence_reads_the_copyright_notice(tmp_path: Path) -> None:
    """A notice in any one of its three tags counts, so one a camera stamped is found too."""
    exif, iptc, xmp, blank = (tmp_path / f"{name}.jpg" for name in ("exif", "iptc", "xmp", "no"))
    for path in (exif, iptc, xmp, blank):
        path.write_text("x")

    fake_helper = _fake_helper(
        [
            {"SourceFile": str(exif), "EXIF:Copyright": "(c) 2019 Ada"},
            {"SourceFile": str(iptc), "IPTC:CopyrightNotice": "(c) 2019 Ada"},
            {"SourceFile": str(xmp), "XMP:Rights": "© 2019 Ada"},
            {"SourceFile": str(blank), "EXIF:Copyright": "  "},
        ],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", side_effect=lambda p: [str(p)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=fake_helper),
    ):
        presence = find_field_presence([exif, iptc, xmp, blank])

    assert presence == {
        exif: {FIELD_COPYRIGHT},
        iptc: {FIELD_COPYRIGHT},
        xmp: {FIELD_COPYRIGHT},
        blank: set(),
    }


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


def test_a_broken_sidecar_does_not_hide_what_the_photo_carries(tmp_path: Path) -> None:
    """
    The photo's own keywords and title survive a sidecar ExifTool cannot parse.

    Without them, the run would tag the photo as if it held nothing, and the write would replace the
    keywords it should have merged with.
    """
    photo = tmp_path / "a.cr3"
    photo.write_text("x")
    sidecar = tmp_path / "a.xmp"
    helper = _fake_helper()
    helper.get_tags.side_effect = _raise_execute_error_with_payload(
        [
            {"SourceFile": str(photo), "XMP:Subject": ["Beach"], "XMP:Title": "Shore"},
            {"SourceFile": str(sidecar)},
        ],
    )
    with (
        patch(
            "photo_tagger.metadata.metadata_targets",
            side_effect=lambda p: [str(p), str(sidecar)],
        ),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        context = read_image_context(photo)

    assert context.existing_keywords.subject == ["Beach"]
    assert context.existing_title == "Shore"


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

    assert ok == WriteResult.WRITTEN
    assert "-overwrite_original" not in helper.set_tags.call_args.kwargs["params"]


def test_write_metadata_without_backup_passes_overwrite_original(tmp_path: Path) -> None:
    """Backup=False must reach exiftool as -overwrite_original, or _original files pile up."""
    img = tmp_path / "img.cr3"
    helper = _fake_helper()

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        ok = write_metadata(img, KeywordSet(subject=["Bird"]), backup=False)

    assert ok == WriteResult.WRITTEN
    assert helper.set_tags.call_args.kwargs["params"][0] == "-overwrite_original"


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
        assert write_metadata(img, KeywordSet(subject=["Bird"])) == WriteResult.FAILED


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

    assert written == WriteResult.WRITTEN
    assert helper.set_tags.call_count == 2  # noqa: PLR2004 - the refused write plus its one retry.
    retry_params = helper.set_tags.call_args.kwargs["params"]
    # The retry keeps whatever the first attempt asked for and only adds the waiver.
    assert retry_params == [*helper.set_tags.call_args_list[0].kwargs["params"], "-m"]


def test_write_metadata_does_not_retry_an_error_m_would_not_waive(tmp_path: Path) -> None:
    """Only exiftool's own "[minor]" marker waives a failure; anything else fails once and stays."""
    img = tmp_path / "img.dng"
    helper = _fake_helper()
    helper.set_tags.side_effect = _execute_error("Error: File format error - img.dng\n")

    with patch("photo_tagger.metadata.ExifToolHelper", return_value=helper):
        assert write_metadata(img, KeywordSet(subject=["Bird"])) == WriteResult.FAILED

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
    assert context.existing_copyright is None


def test_read_image_context_prefers_the_xmp_notice(tmp_path: Path) -> None:
    """XMP carries the UTF-8 wording (a real ©), so it wins over the EXIF one when both exist."""
    img = tmp_path / "a.jpg"
    img.write_text("x")
    helper = _fake_helper(
        [{"SourceFile": str(img), "EXIF:Copyright": "(c) 2019 Ada", "XMP:Rights": "© 2019 Ada"}],
    )
    with (
        patch("photo_tagger.metadata.metadata_targets", return_value=[str(img)]),
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper),
    ):
        context = read_image_context(img)
    assert context.existing_copyright == "© 2019 Ada"


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

    assert written == WriteResult.FAILED


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


def test_field_values_match_the_read_spelling_and_prefer_the_sidecar() -> None:
    """XMP-dc:Creator comes back as XMP:Creator, a blank value is no value, and a sidecar wins."""
    from photo_tagger.metadata import field_values  # noqa: PLC0415

    blocks = [
        {"SourceFile": "a.jpg", "XMP:Creator": "Ada", "EXIF:Artist": " ", "IPTC:By-line": "Ada"},
        {"SourceFile": "a.xmp", "XMP:Creator": ["Grace", "Ada"]},
    ]
    tags = ("XMP-dc:Creator", "EXIF:Artist", "IFD0:Artist", "iptc:by-line", "EXIF:Copyright")
    found = field_values(blocks, tags)
    assert found.values == {"XMP-dc:Creator": "Grace, Ada", "iptc:by-line": "Ada"}
    assert found.image == {"XMP-dc:Creator": "Ada", "iptc:by-line": "Ada"}
    assert found.sidecar == {"XMP-dc:Creator": "Grace, Ada"}


def test_field_values_find_a_family_one_group_by_the_tag_name() -> None:
    """System:, Canon:, and Keys: come back under File:, MakerNotes:, and QuickTime:."""
    from photo_tagger.metadata import field_values  # noqa: PLC0415

    blocks = [
        {
            "SourceFile": "a.mov",
            "File:FileModifyDate": "2019:01:02 03:04:05",
            "MakerNotes:OwnerName": "Ada",
            "QuickTime:CreationDate": "2018:01:01",
        },
    ]
    tags = ("System:FileModifyDate", "Canon:OwnerName", "Keys:CreationDate", "Keys:Missing")
    assert field_values(blocks, tags).image == {
        "System:FileModifyDate": "2019:01:02 03:04:05",
        "Canon:OwnerName": "Ada",
        "Keys:CreationDate": "2018:01:01",
    }


def test_field_sources_written_reads_only_the_target_files() -> None:
    """Preserve looks at what the save writes: the photo, its sidecar, or both."""
    from pathlib import Path  # noqa: PLC0415

    from photo_tagger.metadata import FieldSources, WriteTarget  # noqa: PLC0415

    found = FieldSources(image={"EXIF:Copyright": "a"}, sidecar={"XMP:Rights": "b"})
    photo = WriteTarget(path=Path("a.jpg"), use_sidecar=False, existed=True)
    sidecar = WriteTarget(path=Path("a.xmp"), use_sidecar=True, existed=True)
    assert found.written([photo]) == {"EXIF:Copyright": "a"}
    assert found.written([sidecar]) == {"XMP:Rights": "b"}
    assert found.written([photo, sidecar]) == found.values


def test_read_field_sources_answers_every_photo_even_when_the_read_fails(tmp_path: Path) -> None:
    """A failed read leaves the photos undated rather than stopping the write."""
    from photo_tagger.metadata import read_field_sources  # noqa: PLC0415

    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    assert read_field_sources([image], ())[image].values == {}
    with patch(
        "photo_tagger.metadata.managed_helper",
        side_effect=FileNotFoundError("no exiftool"),
    ):
        assert read_field_sources([image], ("EXIF:Artist",))[image].values == {}


def test_fit_bytes_never_cuts_inside_a_character() -> None:
    """A cut that lands inside a multi-byte character drops the whole character."""
    from photo_tagger.metadata import _fit_bytes  # noqa: PLC0415

    assert _fit_bytes("a" * 31 + "Ł", 32) == "a" * 31
    assert _fit_bytes("short", 32) == "short"


def test_fit_iptc_cuts_only_iptc_values_with_a_known_limit() -> None:
    """XMP has no such limit, and an IPTC dataset without a known one is left to ExifTool."""
    from photo_tagger.metadata import _fit_iptc  # noqa: PLC0415

    long = "x" * 100
    fitted = _fit_iptc(
        {"IPTC:By-line": long, "IPTC:Keywords": [long], "XMP-dc:Creator": long, "IPTC:Foo": long},
    )
    assert fitted == {
        "IPTC:By-line": "x" * 32,
        "IPTC:Keywords": ["x" * 64],
        "XMP-dc:Creator": long,
        "IPTC:Foo": long,
    }


def test_iptc_recoding_only_when_a_new_value_needs_utf8() -> None:
    """ASCII values keep the block as it is; a non-ASCII one re-encodes the values already there."""
    from photo_tagger.metadata import _iptc_recoding  # noqa: PLC0415

    current = {"IPTC:City": "Zürich", "IPTC:ApplicationRecordVersion": 4, "XMP:Title": "t"}
    assert _iptc_recoding({"IPTC:By-line": "Ada"}, current) == {}
    assert _iptc_recoding({"IPTC:By-line": "Łukasz"}, current) == {
        "IPTC:CodedCharacterSet": "UTF8",
        "IPTC:City": "Zürich",
    }
    utf8 = {**current, "IPTC:CodedCharacterSet": "\x1b%G"}
    assert _iptc_recoding({"IPTC:By-line": "Łukasz"}, utf8) == {}


def test_writable_in_drops_the_groups_a_file_type_cannot_hold() -> None:
    """IPTC goes nowhere in a WebP or a CR3, so it is not sent there at all."""
    from pathlib import Path  # noqa: PLC0415

    from photo_tagger.metadata import _writable_in  # noqa: PLC0415

    tags = {"IPTC:Keywords": ["a"], "XMP-dc:Subject": ["a"], "EXIF:Artist": "Ada"}
    assert _writable_in(Path("x.webp"), tags) == {"XMP-dc:Subject": ["a"], "EXIF:Artist": "Ada"}
    assert _writable_in(Path("x.CR3"), tags) == {"XMP-dc:Subject": ["a"], "EXIF:Artist": "Ada"}
    assert _writable_in(Path("x.jpg"), tags) == tags


def test_changed_payload_matches_tag_names_whatever_their_case() -> None:
    """``xmp-dc:rights`` reads back as XMP:Rights, so the same value is not written again."""
    from photo_tagger.metadata import _changed_payload  # noqa: PLC0415

    payload: dict[str, str | list[str]] = {"xmp-dc:rights": "R"}
    assert _changed_payload(payload, {"XMP:Rights": "R"}, use_sidecar=False) == {}


def test_tag_family_knows_multi_group_and_family_one_names() -> None:
    """EXIF:IFD0:Copyright and IFD0:Copyright are EXIF, so a GIF drops them before comparing."""
    from photo_tagger.metadata import _tag_family  # noqa: PLC0415

    assert _tag_family("EXIF:IFD0:Copyright") == "EXIF"
    assert _tag_family("IFD0:Copyright") == "EXIF"
    assert _tag_family("XMP-dc:Title") == "XMP"
    assert _tag_family("Artist") == ""


def test_iptc_recoding_is_case_insensitive_and_skips_an_unread_block() -> None:
    """A --set in lower case replaces the value, and a failed read re-encodes nothing."""
    from photo_tagger.metadata import _iptc_recoding  # noqa: PLC0415

    current = {"IPTC:City": "Zürich", "IPTC:Headline": "(Binary data 50 bytes, use -b option)"}
    recoded = _iptc_recoding({"iptc:city": "Łódź"}, current)
    assert "IPTC:City" not in recoded
    assert "IPTC:Headline" not in recoded
    assert _iptc_recoding({"IPTC:City": "Łódź"}, None) == {}


def test_a_set_of_processing_software_wins_over_the_stamp_in_any_spelling() -> None:
    """A --set of the tag, with or without a group, is the user's choice and is not overwritten."""
    from photo_tagger.metadata import _software_stamp  # noqa: PLC0415

    for tag in ("EXIF:ProcessingSoftware", "IFD0:processingsoftware", "ProcessingSoftware"):
        assert _software_stamp({tag: "MyTool"}, use_sidecar=False) == {}
    assert _software_stamp({"XMP-dc:Title": "t"}, use_sidecar=False)


def test_plan_writes_notes_a_backup_that_predates_the_write(tmp_path: Path) -> None:
    """
    A *_original already there is an older state, not what this write replaces.

    ExifTool keeps it and writes the file anyway, so undo must not restore it: that would drop every
    run in between.
    """
    from photo_tagger.metadata import plan_writes  # noqa: PLC0415

    photo = tmp_path / "img.jpg"
    photo.write_bytes(b"x")
    (tmp_path / "img.jpg_original").write_bytes(b"old")

    (target,) = plan_writes(photo, "none")

    assert target.had_backup is True
    assert target.made_backup(backup=True) is False
    (fresh,) = plan_writes(tmp_path / "other.jpg", "none")
    assert fresh.made_backup(backup=True) is True
    assert fresh.made_backup(backup=False) is False


def test_managed_helper_searches_path_for_a_configured_bare_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """exiftool_path = "exiftool" must not pick up a copy in the current folder either."""
    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "exiftool")
    helper = _fake_helper()
    with (
        patch("photo_tagger.metadata.find_program", return_value="/usr/bin/exiftool") as found,
        patch("photo_tagger.metadata.ExifToolHelper", return_value=helper) as ctor,
        managed_helper(None),
    ):
        pass
    found.assert_called_once_with("exiftool")
    ctor.assert_called_once_with(executable="/usr/bin/exiftool")


def test_sidecar_can_hold_knows_the_exif_family_one_groups() -> None:
    """IFD0 and ExifIFD are EXIF too, so a sidecar can hold none of them."""
    from photo_tagger.metadata import sidecar_can_hold  # noqa: PLC0415

    assert not sidecar_can_hold({"IFD0:Copyright": "c", "ExifIFD:Artist": "a", "IPTC:City": "x"})
    assert not sidecar_can_hold({"IFD2:Copyright": "c", "MakerNotes:OwnerName": "o"})
    assert sidecar_can_hold({"IFD0:Copyright": "c", "XMP-dc:Rights": "r"})
