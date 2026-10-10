"""
Tests for the privacy-safe facts telemetry derives from URLs, the install, and the environment.

The point of every helper is what it does *not* return: a private hostname, an install path, a key
someone typed into a config file. The tests pin both the categories and those omissions.
"""

import importlib.metadata
import io
import locale
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from photo_tagger import telemetry_facts as facts


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:1234/v1", facts.ENDPOINT_LOCALHOST),
        ("http://127.0.0.1:11434", facts.ENDPOINT_LOCALHOST),
        ("http://[::1]:8080/v1", facts.ENDPOINT_LOCALHOST),
        ("http://studio.localhost:1234", facts.ENDPOINT_LOCALHOST),
        ("http://192.168.1.20:1234/v1", facts.ENDPOINT_LAN),
        ("http://10.0.0.5", facts.ENDPOINT_LAN),
        ("http://nas.local:1234", facts.ENDPOINT_LAN),
        ("http://host.docker.internal:11434", facts.ENDPOINT_LAN),
        ("http://gpubox:8080", facts.ENDPOINT_LAN),
        ("http://100.101.102.103:1234", facts.ENDPOINT_TAILNET),
        ("http://desk.tail1234.ts.net:1234", facts.ENDPOINT_TAILNET),
        ("https://api.openai.com/v1", "api.openai.com"),
        ("https://OpenRouter.ai/api/v1", "openrouter.ai"),  # cspell:disable-line
        ("https://my-team.openai.azure.com/openai", "openai.azure.com"),
        ("https://llm.example.com/v1", facts.ENDPOINT_PUBLIC),
        ("http://8.8.8.8/v1", facts.ENDPOINT_PUBLIC),
        ("", facts.ENDPOINT_INVALID),
        (None, facts.ENDPOINT_INVALID),
        ("not a url", facts.ENDPOINT_INVALID),
        ("http://[abc/v1", facts.ENDPOINT_INVALID),  # urlsplit raises on the lone bracket.
    ],
)
def test_endpoint_kind_classifies_without_revealing_the_host(
    url: str | None,
    expected: str,
) -> None:
    """Known services are named; everything else collapses to a category, never the hostname."""
    assert facts.endpoint_kind(url) == expected


@dataclass
class _FakeDistribution:
    """The two metadata files install detection reads."""

    direct_url: str | None = None
    installer: str | None = None

    def read_text(self, name: str) -> str | None:
        return {"direct_url.json": self.direct_url, "INSTALLER": self.installer}.get(name)


def _install(monkeypatch: pytest.MonkeyPatch, dist: _FakeDistribution | None) -> None:
    monkeypatch.setattr(facts, "_distribution", lambda: dist)


def test_install_method_spots_the_app_bundle_and_an_uninstalled_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen app wins over everything; no distribution at all means running from source."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert facts.install_method() == "app"
    monkeypatch.delattr(sys, "frozen")
    _install(monkeypatch, None)
    assert facts.install_method() == "source"


def test_install_method_reads_editable_and_git_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The install record names a development checkout or a git install."""
    _install(monkeypatch, _FakeDistribution(direct_url='{"dir_info": {"editable": true}}'))
    assert facts.install_method() == "editable"
    _install(monkeypatch, _FakeDistribution(direct_url='{"vcs_info": {"vcs": "git"}}'))
    assert facts.install_method() == "git"


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("/Users/x/.local/share/uv/tools/photo-tagger", "uv-tool"),
        ("/Users/x/.local/pipx/venvs/photo-tagger", "pipx"),
    ],
)
def test_install_method_recognizes_tool_installs(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    expected: str,
) -> None:
    """Uv tool and pipx installs are told apart from a plain venv by where they live."""
    _install(monkeypatch, _FakeDistribution(direct_url="not json"))
    monkeypatch.setattr(sys, "prefix", prefix)
    assert facts.install_method() == expected


def test_install_method_recognizes_conda_venv_and_system(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A conda-meta folder means conda; otherwise a venv differs from the base interpreter."""
    _install(monkeypatch, _FakeDistribution())
    (tmp_path / "conda-meta").mkdir()
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert facts.install_method() == "conda"
    plain = tmp_path / "venv"
    plain.mkdir()
    monkeypatch.setattr(sys, "prefix", str(plain))
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    assert facts.install_method() == "venv"
    monkeypatch.setattr(sys, "base_prefix", str(plain))
    assert facts.install_method() == "system"


def test_installer_names_only_known_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known installer is named, an odd one becomes "other", a missing one stays empty."""
    _install(monkeypatch, _FakeDistribution(installer="uv\n"))
    assert facts.installer() == "uv"
    _install(monkeypatch, _FakeDistribution(installer="my-custom-thing"))
    assert facts.installer() == "other"
    _install(monkeypatch, _FakeDistribution())
    assert facts.installer() == ""
    _install(monkeypatch, None)
    assert facts.installer() == ""


def test_real_distribution_lookup_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A package that is not installed reads as "no distribution", not an error."""

    def missing(_name: str) -> importlib.metadata.Distribution:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    assert facts._distribution() is None  # noqa: SLF001


def test_system_locale_prefers_the_hint_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the language and region survive: an encoding or a modifier is dropped."""
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(locale, "getlocale", lambda: (None, None))
    assert facts.system_locale() == ""
    monkeypatch.setenv("LANG", "de_DE.UTF-8@euro")
    assert facts.system_locale() == "de_DE"
    assert facts.system_locale("pt-br") == "pt_BR"
    assert facts.system_locale("es-419") == "es_419"
    monkeypatch.setenv("LANG", "C.UTF-8")
    assert facts.system_locale() == ""


def test_local_clock_reports_offset_hour_and_weekday() -> None:
    """A fixed instant in a fixed zone gives its offset in minutes, its hour, and its weekday."""
    moment = datetime(2026, 9, 23, 21, 30, tzinfo=timezone(timedelta(hours=2)))
    offset, hour, weekday = facts.local_clock(moment.astimezone(UTC))
    assert isinstance(offset, int)
    assert 0 <= hour <= 23  # noqa: PLR2004 - the hour range
    assert 0 <= weekday <= 6  # noqa: PLR2004 - Monday..Sunday


def test_running_in_ci_reads_the_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CI marker counts unless it says false."""
    for name in facts._CI_VARIABLES:  # noqa: SLF001
        monkeypatch.delenv(name, raising=False)
    assert facts.running_in_ci() is False
    monkeypatch.setenv("CI", "false")
    assert facts.running_in_ci() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert facts.running_in_ci() is True


def test_attached_terminal_survives_missing_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GUI launched from Finder may have no stdin at all; that is no terminal, not a crash."""
    monkeypatch.setattr(sys, "stdin", None)
    assert facts.attached_terminal() is False
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert facts.attached_terminal() is False


def test_env_var_names_lists_names_never_values() -> None:
    """Only the names of photo-tagger variables that are set appear, sorted."""
    environ = {"OPENAI_API_KEY": "sk-secret", "TEMPERATURE": "0.5", "HOME": "/Users/x"}
    assert facts.env_var_names(environ) == "OPENAI_API_KEY,TEMPERATURE"


def test_changed_settings_names_differing_fields() -> None:
    """Fields that differ from the baseline are named, sorted."""

    @dataclass(frozen=True)
    class Knobs:
        alpha: int = 1
        beta: str = "b"
        gamma: bool = False

    assert facts.changed_settings(Knobs(alpha=2, gamma=True), Knobs()) == "alpha,gamma"
    assert facts.changed_settings(Knobs(), Knobs()) == ""


def test_config_key_names_counts_unknown_keys_without_naming_them() -> None:
    """A key photo-tagger does not read could be any text a user typed, so it is only counted."""
    config = {
        "recursive": True,
        "provider": {"model_name": "x", "weird": 1},
        "ada_lovelace": 1,
        "unknown_table": {"a": 1},
    }
    known = {"": frozenset({"recursive"}), "provider": frozenset({"model_name"})}
    assert facts.config_key_names(config, known) == ("provider.model_name,recursive", 3)


def test_extension_list_names_known_formats_and_folds_the_rest() -> None:
    """Case and dots do not matter; an unknown extension becomes a single "other"."""
    assert facts.extension_list(["CR3", ".jpg", " dng ", ""]) == "cr3,dng,jpg"
    assert facts.extension_list(["jpg", "secret", "xyz"]) == "jpg,other"
    assert facts.extension_list([]) == ""


def test_quantile_and_mean() -> None:
    """Linear interpolation between ranks, and 0.0 for an empty list."""
    assert facts.quantile([3.0, 1.0, 2.0, 10.0], 0.5) == 2.5  # noqa: PLR2004 - the midpoint
    assert facts.quantile([5.0], 0.9) == 5.0  # noqa: PLR2004 - a single value
    assert facts.quantile([], 0.5) == 0.0
    assert facts.mean([1.0, 2.0]) == 1.5  # noqa: PLR2004 - the average
    assert facts.mean([]) == 0.0


def test_joined_sorts_and_drops_blanks_and_duplicates() -> None:
    """Joining is order-free and repeat-free, so equal sets always encode the same way."""
    assert facts.joined(["b", "a", "", "b"]) == "a,b"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen/qwen3-vl-30b", "qwen/qwen3-vl-30b"),
        ("llava:13b", "llava:13b"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("/Users/someone/models/qwen3-vl-30b.gguf", "qwen3-vl-30b.gguf"),
        ("~/models/x.gguf", "x.gguf"),
        ("./x.gguf", "x.gguf"),
        ("models/x.gguf", "x.gguf"),
        (r"C:\Users\someone\x.gguf", "x.gguf"),
        ("ft:gpt-4o-mini:acme-corp:photos:abc123", "ft:gpt-4o-mini"),
        ("  gemma3  ", "gemma3"),
        ("/Users/someone/models/qwen-mlx/", "qwen-mlx"),
        ("models/private/thing", "thing"),
        ("m" * 300, "m" * facts.MAX_MODEL_CHARS),
    ],
)
def test_model_label_keeps_the_model_and_drops_what_names_a_person(
    model: str,
    expected: str,
) -> None:
    """A path keeps its file name, a fine-tuned id its base model, and every name is capped."""
    assert facts.model_label(model) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("English", "English"),
        ("brazilian portuguese", "Brazilian Portuguese"),
        (" German ", "German"),
        ("Aragonese", facts.OTHER_LANGUAGE),
        ("my private notes", facts.OTHER_LANGUAGE),
    ],
)
def test_language_label_names_only_known_languages(language: str, expected: str) -> None:
    """A known language is named in its canonical spelling; anything typed is "other"."""
    assert facts.language_label(language) == expected


def test_every_language_the_gui_offers_is_known() -> None:
    """The Metadata Language menu's own choices must never be reported as "other"."""
    from photo_tagger.gui_state import OUTPUT_LANGUAGE_SUGGESTIONS  # noqa: PLC0415

    assert all(facts.language_label(name) == name for name in OUTPUT_LANGUAGE_SUGGESTIONS)


def test_joined_stops_at_the_last_whole_name_that_fits() -> None:
    """A long list is cut between names, never inside one, within the collector's clamp."""
    names = [f"table.key_{index:03}" for index in range(200)]
    result = facts.joined(names)
    assert len(result) <= facts.MAX_JOINED_CHARS
    assert set(result.split(",")) <= set(names)
    assert facts.joined(["b", "a", "", "a"]) == "a,b"


def test_joined_keeps_the_names_that_fit_after_one_that_does_not() -> None:
    """One overlong name must not push every later name out of the list."""
    result = facts.joined(["a" * 2000, "b", "c"])
    assert result == "b,c"
