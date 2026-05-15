"""
End-to-end wiring tests for the cyclopts CLI in photo_tagger.main.

The agent (network calls) and pipeline (long-running work) are mocked; the goal is to
prove that flag values reach the right collaborators with the right shape, and that the
short-circuit / skip code paths run end to end without raising. Real work is exercised
elsewhere by the per-module unit tests.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from photo_tagger import main as main_module


if TYPE_CHECKING:
    from pathlib import Path


def _make_jpeg(path: Path) -> Path:
    """Write a small placeholder file so cyclopts' --input validator accepts the path."""
    path.write_bytes(b"\xff\xd8stub")
    return path


def _run_app(args: list[str]) -> None:
    """Invoke the cyclopts app while absorbing the SystemExit it always raises."""
    with contextlib.suppress(SystemExit):
        main_module.app(args)


def _patches(captured: dict[str, Any]) -> Any:  # noqa: ANN401 - context manager juggling.
    """Patch every IO collaborator main.tag invokes; capture the values for assertions."""

    def fake_run_batch(
        image_files: list[Path],
        agent: object,
        options: object,
        **kwargs: object,
    ) -> object:
        captured["image_files"] = list(image_files)
        captured["options"] = options
        captured["on_success"] = kwargs.get("on_success")
        return None

    return (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=fake_run_batch),
    )


def test_cli_passes_inputs_and_dry_run_through_to_pipeline(tmp_path: Path) -> None:
    """A minimal invocation feeds the resolved file list and dry_run flag to run_batch."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--dry-run"])

    assert captured["image_files"] == [image.resolve()]
    assert captured["options"].dry_run is True
    # Default skip_tagged path means no on_success appender created.
    assert captured["on_success"] is None


def test_cli_creates_appender_when_append_to_skip_file_provided(tmp_path: Path) -> None:
    """Passing --append-to-skip-file installs an on_success callback on run_batch."""
    image = _make_jpeg(tmp_path / "img.cr3")
    skip_file = tmp_path / "processed.txt"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--append-to-skip-file", str(skip_file)])

    assert callable(captured["on_success"])

    # The callback writes a single line per call. Exercise it once to be sure.
    captured["on_success"](image)
    assert skip_file.read_text(encoding="utf-8").splitlines() == [image.name]


def test_cli_skip_tagged_filters_before_pipeline(tmp_path: Path) -> None:
    """--skip-tagged removes already-tagged paths before run_batch is even called."""
    keep = _make_jpeg(tmp_path / "keep.cr3")
    drop = _make_jpeg(tmp_path / "drop.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.discovery.find_tagged_images", return_value={drop.resolve()}),
    ):
        _run_app(["--input", str(keep), "--input", str(drop), "--skip-tagged"])

    assert captured["image_files"] == [keep.resolve()]


def test_cli_short_circuits_when_skip_filters_remove_everything(tmp_path: Path) -> None:
    """When all inputs are skipped, run_batch is not called and the CLI exits cleanly."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.discovery.find_tagged_images", return_value={image.resolve()}),
    ):
        _run_app(["--input", str(image), "--skip-tagged"])

    assert "image_files" not in captured  # run_batch never invoked


def test_cli_exits_when_no_inputs_passed() -> None:
    """No --input is a hard error so accidental empty runs surface immediately."""
    setup, create_agent, run_batch = _patches({})
    with setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app([])
