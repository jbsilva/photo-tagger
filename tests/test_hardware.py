"""Tests for the best-effort hardware probes behind telemetry."""

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from photo_tagger.hardware import (
    HardwareInfo,
    _gpu_from_lspci,
    _gpu_model,
    _parse_cpuinfo,
    _parse_meminfo,
    _run,
    hardware_info,
    probe_hardware,
)


if TYPE_CHECKING:
    from pathlib import Path


_X86_CPUINFO = """\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
cache size\t: 12288 KB
"""

_ARM_CPUINFO = """\
processor\t: 0
BogoMIPS\t: 108.00
Hardware\t: BCM2835
"""


def test_parse_cpuinfo_reads_model_name() -> None:
    """The x86 "model name" line yields the CPU model."""
    assert _parse_cpuinfo(_X86_CPUINFO) == "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz"


def test_parse_cpuinfo_falls_back_to_hardware_line() -> None:
    """ARM boards without "model name" still report via the "Hardware" line."""
    assert _parse_cpuinfo(_ARM_CPUINFO) == "BCM2835"


def test_parse_cpuinfo_empty_for_unknown_content() -> None:
    """Unrecognized content yields "" rather than a guess."""
    assert _parse_cpuinfo("flags: fpu vme\n") == ""


_THIRTY_TWO_GB_BOX_ROUNDED = 31  # 32780488 kB is just under 32 GiB; round() lands on 31.


def test_parse_meminfo_rounds_to_gb() -> None:
    """MemTotal in kB becomes whole gigabytes."""
    assert _parse_meminfo("MemTotal:       32780488 kB\nMemFree: 100 kB\n") == (
        _THIRTY_TWO_GB_BOX_ROUNDED
    )
    assert _parse_meminfo("MemFree: 100 kB\n") == 0


def test_gpu_from_lspci_picks_display_controller() -> None:
    """The VGA/3D controller line yields the model; unrelated lines are ignored."""
    text = (
        "00:1f.3 Audio device: Intel Corporation Device\n"
        "01:00.0 VGA compatible controller: NVIDIA Corporation AD104 [GeForce RTX 4070]\n"
    )
    assert _gpu_from_lspci(text) == "NVIDIA Corporation AD104 [GeForce RTX 4070]"
    assert _gpu_from_lspci("00:1f.3 Audio device: Foo\n") == ""


def test_gpu_model_prefers_nvidia_smi() -> None:
    """When nvidia-smi answers, its name wins on every OS."""
    with patch(
        "photo_tagger.hardware._run",
        side_effect=lambda args: "NVIDIA GeForce RTX 4070" if args[0] == "nvidia-smi" else "",
    ):
        assert _gpu_model("Linux", cpu="whatever") == "NVIDIA GeForce RTX 4070"


def test_gpu_model_apple_silicon_reports_the_soc() -> None:
    """On Apple Silicon the chip name is the GPU name; no subprocess beyond nvidia-smi runs."""
    with patch("photo_tagger.hardware._run", return_value="") as run:
        assert _gpu_model("Darwin", cpu="Apple M3 Pro") == "Apple M3 Pro"
    assert all(call.args[0][0] == "nvidia-smi" for call in run.call_args_list)


def test_run_executes_the_resolved_absolute_path() -> None:
    """
    The probe runs the binary shutil.which found, not the bare name it was asked for.

    Handing subprocess a bare name lets the OS search for it, and on Windows that search reaches the
    current working directory before PATH.
    """
    with (
        patch("photo_tagger.programs.shutil.which", return_value="/usr/bin/lspci"),
        patch("photo_tagger.hardware.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        run.return_value.stdout = "  output  "
        assert _run(["lspci", "-v"]) == "output"
    assert run.call_args.args[0] == ["/usr/bin/lspci", "-v"]


def test_run_skips_the_spawn_when_the_tool_is_absent() -> None:
    """A tool that is not on PATH costs a lookup, not a failed process spawn."""
    with (
        patch("photo_tagger.programs.shutil.which", return_value=None),
        patch("photo_tagger.hardware.subprocess.run") as run,
    ):
        assert _run(["nvidia-smi"]) == ""
    run.assert_not_called()


def test_probe_hardware_survives_total_probe_failure() -> None:
    """A machine where every probe fails still yields a clean, empty HardwareInfo."""
    with (
        patch("photo_tagger.hardware._run", return_value=""),
        patch("photo_tagger.hardware._read_text", return_value=""),
        patch("photo_tagger.hardware.platform.system", return_value="Linux"),
        patch("photo_tagger.hardware.os.cpu_count", return_value=None),
    ):
        info = probe_hardware()
    assert info == HardwareInfo(cpu="", gpu="", cpu_count=0, memory_gb=0)


def test_hardware_info_caches_the_first_probe(tmp_path: Path) -> None:
    """The first call probes and writes the cache; later calls read it without re-probing."""
    cache = tmp_path / "hardware.json"
    probed = HardwareInfo(cpu="Apple M3 Pro", gpu="Apple M3 Pro", cpu_count=12, memory_gb=36)

    with patch("photo_tagger.hardware.probe_hardware", return_value=probed) as probe:
        first = hardware_info(cache)
        second = hardware_info(cache)

    assert first == second == probed
    assert probe.call_count == 1
    assert json.loads(cache.read_text(encoding="utf-8"))["cpu"] == "Apple M3 Pro"


def test_hardware_info_reprobes_on_corrupt_or_stale_cache(tmp_path: Path) -> None:
    """Garbage or a version-mismatched cache is silently replaced by a fresh probe."""
    cache = tmp_path / "hardware.json"
    cache.write_text("not json", encoding="utf-8")
    probed = HardwareInfo(cpu="X", gpu="", cpu_count=4, memory_gb=8)

    with patch("photo_tagger.hardware.probe_hardware", return_value=probed):
        assert hardware_info(cache) == probed

    cache.write_text(json.dumps({"version": -1, "cpu": "old"}), encoding="utf-8")
    with patch("photo_tagger.hardware.probe_hardware", return_value=probed):
        assert hardware_info(cache) == probed


def test_hardware_info_without_cache_path_probes_every_time() -> None:
    """cache_path=None skips caching entirely."""
    probed = HardwareInfo(cpu="X", gpu="", cpu_count=4, memory_gb=8)
    with patch("photo_tagger.hardware.probe_hardware", return_value=probed) as probe:
        assert hardware_info(None) == probed
        assert hardware_info(None) == probed
    assert probe.call_count == 2  # noqa: PLR2004 - once per call, nothing cached
