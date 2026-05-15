"""Tests for metadata helpers that don't need a real exiftool binary."""

from photo_tagger.metadata import (
    _build_write_payload,
    _coerce_to_list,
    build_contextual_prompt,
    format_metadata_value,
)


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
