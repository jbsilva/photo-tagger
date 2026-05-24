"""
End-to-end wiring tests for the cyclopts CLI in photo_tagger.main.

The agent (network calls) and pipeline (long-running work) are mocked; the goal is to
prove that flag values reach the right collaborators with the right shape, and that the
short-circuit / skip code paths run end to end without raising. Real work is exercised
elsewhere by the per-module unit tests.
"""

import contextlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from photo_tagger import main as main_module
from photo_tagger.pipeline import ImageOutcome


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
        captured["on_complete"] = kwargs.get("on_complete")
        captured["user_prompt"] = kwargs.get("user_prompt")
        captured["workers"] = kwargs.get("workers")
        captured["on_image_result"] = kwargs.get("on_image_result")
        captured["cache"] = kwargs.get("cache")
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


_EXPECTED_WORKERS = 3


def test_cli_workers_and_prompt_file_reach_run_batch(tmp_path: Path) -> None:
    """--workers and --prompt-file both flow into the run_batch call."""
    image = _make_jpeg(tmp_path / "img.cr3")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Describe like a wildlife photographer.\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            [
                "--input",
                str(image),
                "--workers",
                str(_EXPECTED_WORKERS),
                "--prompt-file",
                str(prompt),
            ],
        )

    assert captured["workers"] == _EXPECTED_WORKERS
    assert captured["user_prompt"] == "Describe like a wildlife photographer."


def _outcome(file: Path, *, success: bool = True, from_cache: bool = False) -> ImageOutcome:
    """Build a representative ImageOutcome for NDJSON-emitter tests."""
    return ImageOutcome(
        file=file,
        success=success,
        from_cache=from_cache,
        retry=False,
        title="A Title",
        description="A description.",
        keywords=["Beach", "Sunset"],
        input_tokens=42,
        output_tokens=7,
        total_tokens=49,
        seconds=1.5,
    )


def test_ndjson_emitter_writes_one_line_per_outcome(tmp_path: Path) -> None:
    """Each call to the emitter writes exactly one JSON line that round-trips through json.loads."""
    buf = io.StringIO()
    emitter = main_module._NDJSONEmitter(buf)  # noqa: SLF001
    emitter(_outcome(tmp_path / "a.cr3", success=True, from_cache=False))
    emitter(_outcome(tmp_path / "b.cr3", success=False, from_cache=True))

    lines = buf.getvalue().splitlines()
    expected_lines = 2
    assert len(lines) == expected_lines
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["status"] == "ok"
    assert first["from_cache"] is False
    assert first["title"] == "A Title"
    assert first["keywords"] == ["Beach", "Sunset"]
    assert second["status"] == "failed"
    assert second["from_cache"] is True


def test_ndjson_emitter_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent emitters never interleave a partial line."""
    buf = io.StringIO()
    emitter = main_module._NDJSONEmitter(buf)  # noqa: SLF001
    paths = [tmp_path / f"img{i:03d}.cr3" for i in range(60)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: emitter(_outcome(p)), paths))

    lines = buf.getvalue().splitlines()
    # Each line parses cleanly: proof that nothing interleaved.
    decoded = [json.loads(line) for line in lines]
    assert sorted(d["file"] for d in decoded) == sorted(str(p) for p in paths)


def test_cli_json_flag_installs_ndjson_emitter(tmp_path: Path) -> None:
    """--json wires an _NDJSONEmitter onto run_batch's on_image_result callback."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--json"])

    emitter = captured["on_image_result"]
    assert isinstance(emitter, main_module._NDJSONEmitter)  # noqa: SLF001


def test_cli_default_does_not_install_ndjson_emitter(tmp_path: Path) -> None:
    """Without --json the on_image_result callback stays None."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["on_image_result"] is None


def test_cli_newer_than_filters_input_batch(tmp_path: Path) -> None:
    """--newer-than parses the timestamp and drops files older than the bound."""
    import os  # noqa: PLC0415 - test-local import.
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}
    boundary = datetime(2024, 1, 1, tzinfo=UTC)
    old_ts = (boundary - timedelta(days=10)).timestamp()
    os.utime(image, (old_ts, old_ts))

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--newer-than", "2024-01-01"])

    # The lone file is older than the bound so run_batch should never be invoked.
    assert "image_files" not in captured


def test_cli_rejects_malformed_newer_than(tmp_path: Path) -> None:
    """--newer-than with a non-ISO string exits before scheduling work."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app(["--input", str(image), "--newer-than", "not-a-date"])


def test_cli_lock_file_blocks_second_run(tmp_path: Path) -> None:
    """A second --lock-file invocation while another holds the lock exits with code 1."""
    from photo_tagger.locking import FileLock  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    lock_path = tmp_path / "run.lock"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    # Hold the lock in this thread, then invoke the CLI which should fail to acquire.
    with FileLock(lock_path), setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app(["--input", str(image), "--lock-file", str(lock_path)])

    # run_batch never ran because the CLI bailed out at lock acquisition.
    assert "image_files" not in captured


_EXPECTED_TOTAL_TOKENS = 49


def test_cli_summary_file_written_on_completion(tmp_path: Path) -> None:
    """--summary-file receives a JSON payload with run totals after the batch finishes."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    summary = tmp_path / "summary.json"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    fake_totals = BatchTotals(
        total_files=1,
        success=1,
        successful_files=[image.name],
        input_tokens=42,
        output_tokens=7,
        total_tokens=_EXPECTED_TOTAL_TOKENS,
        inference_calls=1,
    )

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--summary-file", str(summary)])
        # Simulate run_batch's on_complete callback firing with realistic totals.
        on_complete = captured["on_complete"]
        assert on_complete is not None
        on_complete(fake_totals)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["total_tokens"] == _EXPECTED_TOTAL_TOKENS
    assert payload["successful_files"] == [image.name]
    assert payload["model"]  # provider/model fields are populated.
    assert "started_at" in payload
    assert "finished_at" in payload
