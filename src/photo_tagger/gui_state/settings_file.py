"""Save Settings as Defaults: the window's choices written into the configuration file."""

# Imported at runtime, not under TYPE_CHECKING: _config_table isinstance-checks against it.
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tomlkit
from tomlkit.exceptions import TOMLKitError

from photo_tagger import telemetry
from photo_tagger.config import (
    DEFAULT_BATTERY_FLOOR,
    DEFAULT_BATTERY_RESUME,
    DEFAULT_DUTY_PERCENT,
    DEFAULT_MAX_PAUSE_MINUTES,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_WORK_MINUTES,
    SidecarMode,
)
from photo_tagger.errors import ConfigFileError
from photo_tagger.i18n import AUTO, _


if TYPE_CHECKING:
    from pathlib import Path

    from tomlkit import TOMLDocument


def _toml_str(value: str) -> str:
    """Quote *value* as a TOML basic string."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_bool(value: bool) -> str:  # noqa: FBT001  # the bool is the value being rendered
    """Render a TOML boolean."""
    return "true" if value else "false"


def telemetry_toggle_note(reason: str, *, do_not_track: bool) -> str:
    """Explain the telemetry toggle when an environment variable decides it, else return ""."""
    if reason == telemetry.CONSENT_ENV_OPT_OUT:
        return _(
            "Turned off by PHOTO_TAGGER_TELEMETRY or PHOTO_TAGGER_NO_TELEMETRY in your "
            "environment. Remove it to choose here.",
        )
    if reason == telemetry.CONSENT_DO_NOT_TRACK:
        return _(
            "DO_NOT_TRACK is set in your environment, so nothing is sent. Check this to send "
            "anyway, for Photo Tagger only.",
        )
    if not do_not_track:
        return ""
    if reason == telemetry.CONSENT_GUI_OPT_IN:
        return _("Sending even though DO_NOT_TRACK is set, because you turned this on here.")
    if reason == telemetry.CONSENT_ENV_OPT_IN:
        return _(
            "Sending even though DO_NOT_TRACK is set, because PHOTO_TAGGER_TELEMETRY=1 is set.",
        )
    return ""


@dataclass(frozen=True, slots=True)
class GuiConfigValues:
    """
    The GUI settings that persist to the config file.

    The API key is deliberately not a field: it must never be written to disk.

    The trailing fields have defaults because they were added later and every caller that only cares
    about the provider and the save options should stay readable.
    """

    provider_name: str
    model_name: str
    api_base_url: str | None
    extensions: str
    recursive: bool
    write_title: bool
    write_description: bool
    write_keywords: bool
    preserve_keywords: bool
    sidecar_mode: SidecarMode
    backup_xmp: bool
    telemetry_enabled: bool
    vocabulary: Path | None = None
    vocabulary_strict: bool = False
    session_gap_minutes: float = 0.0
    undo_log: bool = True
    preserve_title: bool = False
    preserve_description: bool = False
    notify: bool = False
    notify_sound: bool = False
    repair_language: bool = False
    # Settings > Copyright. Blank strings and a None flag mean "write nothing", which removes the
    # keys from the file rather than pinning empty ones.
    write_copyright: bool = True
    artist: str = ""
    copyright_notice: str = ""
    rights: str = ""
    marked: bool | None = None
    preserve_copyright: bool = True
    # Window-only, under [gui]: the CLI writes every photo it generates, so it has no flag for it.
    save_after_generating: bool = False
    # Settings > Pacing, under [pacing]. The built-ins pace nothing, so a config the GUI writes
    # without touching the dialog leaves every run working flat out.
    duty: int = DEFAULT_DUTY_PERCENT
    battery_floor: int = DEFAULT_BATTERY_FLOOR
    battery_resume: int = DEFAULT_BATTERY_RESUME
    work_minutes: float = DEFAULT_WORK_MINUTES
    max_pause_minutes: float = DEFAULT_MAX_PAUSE_MINUTES


def _copyright_lines(values: GuiConfigValues) -> list[str]:
    """Render the ``[copyright]`` table, or nothing at all when no field is filled in."""
    entries = [
        (key, value)
        for key, value in (
            ("artist", values.artist),
            ("copyright_notice", values.copyright_notice),
            ("rights", values.rights),
        )
        if value
    ]
    if not entries and values.marked is None and values.preserve_copyright:
        return []
    lines = ["", "[copyright]"]
    lines += [f"{key} = {_toml_str(value)}" for key, value in entries]
    if values.marked is not None:
        lines.append(f"marked = {_toml_bool(values.marked)}")
    if not values.preserve_copyright:
        lines.append("preserve_copyright = false")
    return lines


def config_toml_text(values: GuiConfigValues) -> str:
    """
    Render *values* as a fresh TOML config file the CLI and GUI both load.

    Key names mirror the config tables ``load_defaults`` reads ([provider], [output], [artifacts],
    [telemetry], plus the top-level extensions/recursive). Used only when no config file exists yet;
    an existing file goes through :func:`merged_config_text` instead so nothing is lost.
    """
    lines = [
        "# Written by the Photo Tagger GUI (Settings > Save Settings as Defaults).",
        f"extensions = {_toml_str(values.extensions)}",
        f"recursive = {_toml_bool(values.recursive)}",
        "",
        "[provider]",
        f"provider_name = {_toml_str(values.provider_name)}",
        f"model_name = {_toml_str(values.model_name)}",
    ]
    if values.api_base_url:
        lines.append(f"api_base_url = {_toml_str(values.api_base_url)}")
    lines += [
        "",
        "[output]",
        f"write_title = {_toml_bool(values.write_title)}",
        f"write_description = {_toml_bool(values.write_description)}",
        f"write_keywords = {_toml_bool(values.write_keywords)}",
        f"write_copyright = {_toml_bool(values.write_copyright)}",
        f"preserve_keywords = {_toml_bool(values.preserve_keywords)}",
        f"preserve_title = {_toml_bool(values.preserve_title)}",
        f"preserve_description = {_toml_bool(values.preserve_description)}",
        f"sidecar_mode = {_toml_str(values.sidecar_mode)}",
        f"backup_xmp = {_toml_bool(values.backup_xmp)}",
    ]
    if values.vocabulary is not None:
        lines.append(f"vocabulary = {_toml_str(str(values.vocabulary))}")
    lines += [
        f"vocabulary_strict = {_toml_bool(values.vocabulary_strict)}",
        f"session_gap_minutes = {values.session_gap_minutes}",
    ]
    lines += _copyright_lines(values)
    lines += [
        "",
        "[inference]",
        f"repair_language = {_toml_bool(values.repair_language)}",
        "",
        "[display]",
        f"notify = {_toml_bool(values.notify)}",
        f"notify_sound = {_toml_bool(values.notify_sound)}",
        "",
        "[pacing]",
        f"duty = {values.duty}",
        f"battery_floor = {values.battery_floor}",
        f"battery_resume = {values.battery_resume}",
        f"work_minutes = {values.work_minutes}",
        f"max_pause_minutes = {values.max_pause_minutes}",
        "",
        "[artifacts]",
        f"undo_log = {_toml_bool(values.undo_log)}",
        "",
        "[gui]",
        f"save_after_generating = {_toml_bool(values.save_after_generating)}",
        "",
        "[telemetry]",
        f"enabled = {_toml_bool(values.telemetry_enabled)}",
        "",
    ]
    return "\n".join(lines)


def _parse_config(existing_text: str) -> TOMLDocument:
    """
    Parse a config file the GUI is about to update, or refuse the update.

    Loading a config for *reading* tolerates a broken file (it is logged and ignored), but rewriting
    one cannot: preserving the keys, comments, and ordering the GUI does not manage means starting
    from a document, and a file with a syntax error yields none. Without this the tomlkit ParseError
    escaped a Qt slot and crashed the window on a save meant to be forgiving.
    """
    try:
        return tomlkit.parse(existing_text)
    except TOMLKitError as exc:
        msg = f"The config file is not valid TOML, so it was left unchanged: {exc}"
        raise ConfigFileError(msg) from exc


def _config_table(document: TOMLDocument, name: str) -> MutableMapping[str, object]:
    """
    Return *document*'s ``[name]`` table, creating it when absent.

    A file where the key exists but is not a table (``provider = "lmstudio"``) is refused rather
    than overwritten: assigning into it raised a bare TypeError out of the save slot, and silently
    replacing it would throw away whatever the user meant by it.
    """
    existing = document.get(name)
    if existing is None:
        table = tomlkit.table()
        document[name] = table
        return table
    if not isinstance(existing, MutableMapping):
        msg = f"The config file's '{name}' entry is not a [{name}] table, so it was left unchanged."
        raise ConfigFileError(msg)
    return existing


def _merge_copyright(document: TOMLDocument, values: GuiConfigValues) -> None:
    """
    Write the ``[copyright]`` table, dropping the keys the dialog left empty.

    An emptied field has to remove its key, not write ``""``: the CLI reads the same table, and a
    blank notice there would look like a setting rather than the absence of one. A table left with
    nothing in it goes too.
    """
    if "copyright" not in document and not any(
        (
            values.artist,
            values.copyright_notice,
            values.rights,
            values.marked is not None,
            not values.preserve_copyright,
        ),
    ):
        return
    table = _config_table(document, "copyright")
    for key, value in (
        ("artist", values.artist),
        ("copyright_notice", values.copyright_notice),
        ("rights", values.rights),
    ):
        if value:
            table[key] = value
        else:
            table.pop(key, None)
    if values.marked is None:
        table.pop("marked", None)
    else:
        table["marked"] = values.marked
    # The default is to keep, so only the choice to replace is worth a line.
    if values.preserve_copyright:
        table.pop("preserve_copyright", None)
    else:
        table["preserve_copyright"] = False
    if not table:
        document.pop("copyright", None)


def merged_config_text(existing_text: str, values: GuiConfigValues) -> str:
    """
    Update *existing_text* (a TOML config file) with *values*, preserving everything else.

    tomlkit keeps comments, ordering, and keys the GUI does not manage, so saving from the GUI never
    destroys a hand-written config. Only the GUI-managed keys are set; a blank base URL removes the
    key so the provider default applies again.

    Raises :class:`~photo_tagger.errors.ConfigFileError` when the file cannot be updated in place.
    """
    document = _parse_config(existing_text)
    document["extensions"] = values.extensions
    document["recursive"] = values.recursive

    provider = _config_table(document, "provider")
    provider["provider_name"] = values.provider_name
    provider["model_name"] = values.model_name
    if values.api_base_url:
        provider["api_base_url"] = values.api_base_url
    else:
        provider.pop("api_base_url", None)

    output = _config_table(document, "output")
    output["write_title"] = values.write_title
    output["write_description"] = values.write_description
    output["write_keywords"] = values.write_keywords
    output["write_copyright"] = values.write_copyright
    output["preserve_keywords"] = values.preserve_keywords
    output["preserve_title"] = values.preserve_title
    output["preserve_description"] = values.preserve_description
    output["sidecar_mode"] = values.sidecar_mode
    # The boolean this replaced only knew sidecar-or-not, and the mode wins over it anyway. Drop
    # it so a config the GUI wrote never carries two answers to the same question.
    output.pop("use_sidecar", None)
    output["backup_xmp"] = values.backup_xmp
    output["vocabulary_strict"] = values.vocabulary_strict
    output["session_gap_minutes"] = values.session_gap_minutes
    if values.vocabulary is not None:
        output["vocabulary"] = str(values.vocabulary)
    else:
        # No vocabulary chosen: drop the key rather than write an empty path the CLI would then
        # fail to open on its next run.
        output.pop("vocabulary", None)

    _merge_copyright(document, values)

    inference = _config_table(document, "inference")
    inference["repair_language"] = values.repair_language

    display = _config_table(document, "display")
    display["notify"] = values.notify
    display["notify_sound"] = values.notify_sound

    pacing = _config_table(document, "pacing")
    pacing["duty"] = values.duty
    pacing["battery_floor"] = values.battery_floor
    pacing["battery_resume"] = values.battery_resume
    pacing["work_minutes"] = values.work_minutes
    pacing["max_pause_minutes"] = values.max_pause_minutes

    artifacts = _config_table(document, "artifacts")
    artifacts["undo_log"] = values.undo_log

    window = _config_table(document, "gui")
    window["save_after_generating"] = values.save_after_generating

    telemetry = _config_table(document, "telemetry")
    telemetry["enabled"] = values.telemetry_enabled
    return tomlkit.dumps(document)


def config_text_with_language(existing_text: str, language: str) -> str:
    """
    Set the top-level ``language`` key in a TOML config, preserving everything else.

    Choosing :data:`~photo_tagger.i18n.AUTO` (follow the OS locale, the built-in default) removes
    the key instead of writing it, so a config never pins a language the user did not pick. tomlkit
    keeps comments and ordering intact, like :func:`merged_config_text`.

    Raises :class:`~photo_tagger.errors.ConfigFileError` when the file cannot be updated in place.
    """
    document = _parse_config(existing_text)
    if language == AUTO:
        document.pop("language", None)
    else:
        document["language"] = language
    return tomlkit.dumps(document)


def config_text_with_output_language(existing_text: str, language: str) -> str:
    """
    Set ``[inference] output_language`` in a TOML config, preserving everything else.

    The key the CLI's ``--output-language`` flag reads its default from, so the GUI choice carries
    over to CLI runs too. Choosing the built-in default (English) removes the key instead of pinning
    it, mirroring :func:`config_text_with_language`; an ``[inference]`` table left empty by that
    removal is dropped as well.

    Raises :class:`~photo_tagger.errors.ConfigFileError` when the file cannot be updated in place.
    """
    document = _parse_config(existing_text)
    if language.strip().casefold() == DEFAULT_OUTPUT_LANGUAGE.casefold():
        if "inference" in document:
            inference = _config_table(document, "inference")
            inference.pop("output_language", None)
            if not inference:
                document.pop("inference", None)
    else:
        _config_table(document, "inference")["output_language"] = language.strip()
    return tomlkit.dumps(document)
