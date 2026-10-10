"""Shared pytest fixtures for the photo-tagger test suite."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic_ai import BinaryContent


if TYPE_CHECKING:
    from pydantic_ai import Agent


# Make the whole suite independent of any real ~/.config/photo-tagger/config.toml. main.py resolves
# its CLI defaults at import time (a module-level load_defaults()), so this has to be set before any
# test module imports it - here at conftest import, not in a fixture. The target file is empty, so
# config resolution falls through to the built-in defaults. A test asserts this stays in effect.
os.environ["PHOTO_TAGGER_CONFIG"] = str(Path(__file__).parent / "empty-config.toml")


@pytest.fixture(autouse=True)
def _english_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin the UI language to English so string assertions hold on any machine.

    Without this, a developer with a pt_BR locale (or PHOTO_TAGGER_LANG exported) would get
    translated GUI strings and every label assertion in test_gui/test_gui_state would fail.
    activate("en") installs the identity catalog; tests that exercise a real catalog call activate()
    themselves and are reset by the next test's fixture run.
    """
    from photo_tagger import i18n  # noqa: PLC0415 - import here to keep conftest import light.

    monkeypatch.delenv("PHOTO_TAGGER_LANG", raising=False)
    i18n.activate("en")


@pytest.fixture(autouse=True)
def _isolate_telemetry_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Keep telemetry hermetic: redirect its state dir to tmp and clear the opt-out env vars.

    ``main.tag`` and the GUI consult telemetry on every run, so without this the suite would read
    and write the developer's real ``~/.local/state/photo-tagger`` and be swayed by a stray
    ``DO_NOT_TRACK`` in their shell. The ``httpx2.post`` stub guarantees no beacon ever leaves the
    machine (the GUI test fixture closes its window, which would otherwise fire one). Tests that
    assert on the send itself re-patch ``httpx2.post`` locally, overriding this stub.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "pt-state"))
    monkeypatch.delenv("PHOTO_TAGGER_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    # Also clear the exiftool override so a developer's exported path can't sway metadata tests.
    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    monkeypatch.setattr("httpx2.post", lambda *_a, **_k: None)
    # Never shell out to real hardware probes (sysctl, lspci, nvidia-smi) from the suite.
    from photo_tagger.hardware import HardwareInfo  # noqa: PLC0415 - keep conftest import light.

    monkeypatch.setattr("photo_tagger.telemetry.hardware_info", lambda _path: HardwareInfo())


@pytest.fixture(autouse=True)
def _no_stray_http_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fail fast when a test reaches the network without patching it first.

    The provider/diagnostics code GETs http://localhost:1234/v1 and friends by default, so a test
    that forgets its local httpx2.get patch passes on a dev machine running LM Studio and hangs or
    flakes in CI. Tests that need a GET re-patch locally, mirroring the httpx2.post stub above.
    """

    def _refuse(*_args: object, **_kwargs: object) -> None:
        msg = "test attempted a real httpx2.get; patch httpx2.get (or the caller) locally"
        raise AssertionError(msg)

    monkeypatch.setattr("httpx2.get", _refuse)


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the pause before the retry pass so every run_batch test stays instant."""
    monkeypatch.setattr("photo_tagger.pipeline._RETRY_PASS_DELAY_SECONDS", 0.0)


@pytest.fixture
def stub_jpeg_bytes() -> BinaryContent:
    """Return a tiny placeholder JPEG used to short-circuit image preparation in unit tests."""
    return BinaryContent(data=b"\xff\xd8stub", media_type="image/jpeg")


@pytest.fixture
def fake_agent() -> Agent:
    """Return a typed-but-bogus Agent placeholder for tests that mock every IO collaborator."""
    return cast("Agent", object())
