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
from photo_tagger.pipeline import BatchTotals, ImageOutcome


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


def test_parse_filter_date_treats_naive_as_local_time() -> None:
    """A naive ISO date attaches the system local zone, not UTC."""
    from datetime import datetime  # noqa: PLC0415 - test-local import.

    parsed = main_module._parse_filter_date("2024-01-01T00:00:00", flag="--newer-than")  # noqa: SLF001
    assert parsed is not None
    assert parsed.tzinfo is not None
    # Wall-clock fields are preserved verbatim: the user wrote midnight local.
    naive_expected = datetime(2024, 1, 1, 0, 0, 0)  # noqa: DTZ001 - naive on purpose.
    assert parsed.replace(tzinfo=None) == naive_expected
    # Attached offset matches the system's local offset for *that* wall-clock
    # time (which may differ from "now" across DST boundaries). Comparing
    # against datetime.astimezone() of the same naive moment guards against
    # the parser silently falling back to UTC.
    assert parsed.utcoffset() == naive_expected.astimezone().utcoffset()


def test_parse_filter_date_preserves_explicit_timezone() -> None:
    """An ISO string that already carries a timezone is passed through unchanged."""
    parsed = main_module._parse_filter_date("2024-01-01T00:00:00+00:00", flag="--newer-than")  # noqa: SLF001
    assert parsed is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_parse_filter_date_returns_none_for_none() -> None:
    """Passing None short-circuits without raising."""
    assert main_module._parse_filter_date(None, flag="--newer-than") is None  # noqa: SLF001


def test_open_cache_returns_none_when_path_is_none() -> None:
    """No --cache-file means no cache, never an exception."""
    assert main_module._open_cache(None, namespace="m#x") is None  # noqa: SLF001


def test_open_cache_degrades_on_open_failure(tmp_path: Path) -> None:
    """A failing cache open is logged and downgraded to no-cache, not raised."""
    bad_path = tmp_path / "cache.sqlite3"
    with patch.object(main_module, "InferenceCache", side_effect=OSError("denied")):
        result = main_module._open_cache(bad_path, namespace="m#x")  # noqa: SLF001
    assert result is None


def test_cli_skips_cache_when_open_fails(tmp_path: Path) -> None:
    """--cache-file with an unusable target still lets the batch run (no cache)."""
    image = _make_jpeg(tmp_path / "img.cr3")
    cache_path = tmp_path / "cache.sqlite3"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch.object(main_module, "InferenceCache", side_effect=OSError("denied")),
    ):
        _run_app(["--input", str(image), "--cache-file", str(cache_path)])

    # run_batch was invoked, and the cache kwarg fell through as None.
    assert captured.get("image_files") == [image.resolve()]
    assert captured.get("cache") is None


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


def test_cli_lock_file_open_failure_exits(tmp_path: Path) -> None:
    """An OSError while opening the lock file (e.g. unwritable dir) exits with code 1."""
    image = _make_jpeg(tmp_path / "img.cr3")
    lock_path = tmp_path / "run.lock"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch.object(main_module, "FileLock", side_effect=OSError("disk full")),
        pytest.raises(SystemExit),
    ):
        main_module.app(["--input", str(image), "--lock-file", str(lock_path)])

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


def test_cli_summary_file_creates_missing_parent_dir(tmp_path: Path) -> None:
    """--summary-file pointing into a not-yet-existing folder is created transparently."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    nested = tmp_path / "reports" / "today" / "summary.json"
    assert not nested.parent.exists()
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    fake_totals = BatchTotals(total_files=1, success=1, successful_files=[image.name])

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--summary-file", str(nested)])
        captured["on_complete"](fake_totals)

    assert nested.exists()
    payload = json.loads(nested.read_text(encoding="utf-8"))
    assert payload["total_files"] == 1


# ---------------------------------------------------------------------------
# _read_prompt_file edge cases
# ---------------------------------------------------------------------------


def test_read_prompt_file_exits_on_empty_file(tmp_path: Path) -> None:
    """A prompt file that contains only whitespace is rejected with SystemExit."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(SystemExit):
        main_module._read_prompt_file(prompt)  # noqa: SLF001


def test_read_prompt_file_exits_on_read_error(tmp_path: Path) -> None:
    """An unreadable prompt file triggers SystemExit."""
    prompt = tmp_path / "missing.txt"
    with pytest.raises(SystemExit):
        main_module._read_prompt_file(prompt)  # noqa: SLF001


def test_read_prompt_file_returns_default_when_none() -> None:
    """Passing None returns the built-in default prompt."""
    from photo_tagger.config import DEFAULT_USER_PROMPT  # noqa: PLC0415

    assert main_module._read_prompt_file(None) == DEFAULT_USER_PROMPT  # noqa: SLF001


# ---------------------------------------------------------------------------
# _write_summary_file edge cases
# ---------------------------------------------------------------------------


def test_write_summary_file_noop_when_path_is_none(tmp_path: Path) -> None:
    """summary_file=None is the normal no-write path."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Should not raise or create any file.
    main_module._write_summary_file(  # noqa: SLF001
        None,
        BatchTotals(),
        started_at=datetime.now(tz=UTC),
        model_name="m",
        provider_name="p",
        user_prompt_chars=0,
    )


def test_write_summary_file_noop_when_totals_is_none(tmp_path: Path) -> None:
    """totals=None is the early-exit path when the batch never ran."""
    from datetime import UTC, datetime  # noqa: PLC0415

    dest = tmp_path / "out.json"
    main_module._write_summary_file(  # noqa: SLF001
        dest,
        None,
        started_at=datetime.now(tz=UTC),
        model_name="m",
        provider_name="p",
        user_prompt_chars=0,
    )
    assert not dest.exists()


def test_write_summary_file_swallows_write_error(tmp_path: Path) -> None:
    """An OSError during write is logged, not raised."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Point at a directory path so write fails.
    dest = tmp_path / "dir_not_file"
    dest.mkdir()
    dest = dest / "nested" / "summary.json"
    with patch.object(main_module, "_atomic_write_text", side_effect=OSError("boom")):
        # Must not raise.
        main_module._write_summary_file(  # noqa: SLF001
            dest,
            BatchTotals(),
            started_at=datetime.now(tz=UTC),
            model_name="m",
            provider_name="p",
            user_prompt_chars=0,
        )


# ---------------------------------------------------------------------------
# _atomic_write_text edge cases
# ---------------------------------------------------------------------------


def test_atomic_write_text_cleans_up_on_write_failure(tmp_path: Path) -> None:
    """If the write to the temp file fails, the temp file is removed and the error re-raised."""
    target = tmp_path / "output.json"

    # Patch os.fdopen to raise after mkstemp creates the temp file.
    with (
        patch("photo_tagger.main.os.fdopen", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        main_module._atomic_write_text(target, "content")  # noqa: SLF001

    # Target must not exist, and no temp files should be left behind.
    assert not target.exists()
    leftovers = list(tmp_path.glob(f".{target.name}.*"))
    assert leftovers == []
