"""
The photo's own metadata as text for the model: location, camera, capture date, and keywords.

It is what turns a context read (see :func:`photo_tagger.metadata.read_image_context`) into the part
of the prompt that tells the model what the photo already says, so the answer agrees with it. The
GUI also uses the camera and location helpers for the detail pane. Pure: no ExifTool here.
"""

from photo_tagger.config import CITY_TAGS, COUNTRY_TAGS, TAG_EXIF_DATE_TIME_ORIGINAL


def _first_present(values: dict[str, str], *tags: str) -> str | None:
    """Return the first non-empty value in *values* lookup-ordered by *tags*."""
    for tag in tags:
        if (raw := values.get(tag)) and (stripped := raw.strip()):
            return stripped
    return None


def select_camera_fields(
    camera_info: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Return ``(model, lens, capture_date)`` from a :data:`CAMERA_TAGS` mapping."""
    return (
        camera_info.get("EXIF:Model"),
        camera_info.get("EXIF:LensModel"),
        camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL),
    )


def select_location(location_tags: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Return ``(city, country)`` from a :data:`LOCATION_TAGS` mapping.

    XMP-photoshop and IPTC each define their own city/country fields, so this picks the first
    present in each category, the same fallback order :func:`_format_location` uses for the prompt
    hint and the CSV report uses for its columns.
    """
    return (
        _first_present(location_tags, *CITY_TAGS),
        _first_present(location_tags, *COUNTRY_TAGS),
    )


def _format_location(location_tags: dict[str, str]) -> str | None:
    """
    Build a "City, Country" hint, falling back to whichever single field is set.

    Returning both means the model gets the full place name (e.g. "Barcelona, Spain") instead of
    silently losing the city under the older "first hit wins" logic.
    """
    city, country = select_location(location_tags)
    if city and country:
        return f"{city}, {country}"
    return city or country


def _camera_lines(camera_info: dict[str, str]) -> list[str]:
    """Render ``EXIF:Model`` / ``EXIF:LensModel`` / ``EXIF:DateTimeOriginal`` lines."""
    lines: list[str] = []
    if model := camera_info.get("EXIF:Model"):
        lines.append(f"- Camera: {model}")
    if lens := camera_info.get("EXIF:LensModel"):
        lines.append(f"- Lens: {lens}")
    if captured := camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL):
        lines.append(f"- Captured: {captured}")
    return lines


def prompt_with_hint(base_prompt: str, hint: str | None) -> str:
    """
    Append a photographer's note to *base_prompt*; return it unchanged when *hint* is blank.

    The note is framed as authoritative, unlike the Existing Metadata block (which the system prompt
    tells the model to distrust on conflict): a hint exists precisely because the model misread the
    image ("that animal is a deer, not a boar"), so on conflict the note must win.
    """
    hint = (hint or "").strip()
    if not hint:
        return base_prompt
    return (
        f"{base_prompt.strip()}\n\n"
        f"Photographer's note about this photo: {hint}\n"
        "The photographer knows what is in the photo. If this note conflicts with your own "
        "identification, trust the note."
    )


def build_contextual_prompt(  # noqa: PLR0913 - each kwarg renders an independent section.
    base_prompt: str,
    flat_keywords: list[str],
    location_tags: dict[str, str],
    gps_info: dict[str, str],
    max_prompt_flat_keywords: int = 4,
    *,
    camera_info: dict[str, str] | None = None,
) -> str:
    """
    Create a concise user prompt that surfaces existing photo metadata.

    *camera_info* is a mapping of ExifTool tag names (``EXIF:Model`` etc.) to their string values.
    When present, equipment and capture-date hints are appended so the model can take cues from the
    gear used (macro lens, telephoto, wide angle) and the time of year.
    """
    metadata_lines: list[str] = []

    if unique_keywords := list(dict.fromkeys(flat_keywords)):
        selected = unique_keywords[:max_prompt_flat_keywords]
        line = "- Existing Keywords: " + ", ".join(selected)
        if len(unique_keywords) > max_prompt_flat_keywords:
            line += ", ..."
        metadata_lines.append(line)

    if location_value := _format_location(location_tags):
        metadata_lines.append(f"- Location: {location_value}")

    if gps_position := (gps_info or {}).get("position"):
        metadata_lines.append(f"- GPS: {gps_position}")

    metadata_lines.extend(_camera_lines(camera_info or {}))

    sections = [base_prompt.strip()]
    if metadata_lines:
        sections.append("Existing Metadata:\n" + "\n".join(metadata_lines))
    return "\n\n".join(sections)
