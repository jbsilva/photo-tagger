"""
Best-effort local hardware facts for telemetry.

Answers "what machines do people run photo-tagger on?" with four coarse values: CPU model, GPU
model, logical core count, and RAM rounded to whole gigabytes. Everything here is:

- **Generic.** A CPU or GPU model name ("Apple M3 Pro", "NVIDIA GeForce RTX 4070") is shared by
  millions of machines; no serial numbers, hostnames, or unique identifiers are ever read.
- **Best-effort.** Every probe is wrapped, every subprocess has a short timeout, and any failure
  yields an empty value. A machine where nothing can be probed still reports cleanly.
- **Cheap after the first run.** Probing shells out (sysctl, nvidia-smi, ...), so the result is
  cached as JSON in the caller-supplied path; hardware rarely changes.

No third-party dependency (psutil etc.) is used; each OS is asked directly.
"""

import contextlib
import json
import os
import platform
import subprocess  # nosec B404 - only used to query fixed, well-known hardware probe tools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from photo_tagger.programs import find_program


# Bumped when probes improve, so stale cached values get re-probed after an upgrade.
_CACHE_VERSION = 1

# Generous per-command budget: probes run on the telemetry thread, never on the tagging path,
# but a hung tool (a broken nvidia-smi is not unheard of) must not pin the thread forever.
_PROBE_TIMEOUT_SECONDS = 3.0

_BYTES_PER_GB = 1024**3


@dataclass(slots=True, frozen=True)
class HardwareInfo:
    """
    The four coarse hardware facts telemetry reports.

    Empty/zero when unknown.
    """

    cpu: str = ""
    gpu: str = ""
    cpu_count: int = 0
    memory_gb: int = 0


def _run(args: list[str]) -> str:
    """
    Run *args* and return stripped stdout, or "" on any failure or non-zero exit.

    The program is resolved with :func:`~photo_tagger.programs.find_program`, and the absolute
    path is what gets executed. Handing a bare name to ``subprocess`` lets the OS do the search,
    and on Windows that search covers the current working directory before PATH: a
    ``nvidia-smi.exe`` sitting in the photo folder the GUI was launched from would run instead of
    the real one. It is also cheaper, since the common case (no ``nvidia-smi`` on a Mac, no
    ``lspci`` on a container) costs a PATH lookup rather than a failed process spawn.
    """
    program = find_program(args[0])
    if program is None:
        return ""
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv from constants below
            [program, *args[1:]],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _parse_cpuinfo(text: str) -> str:
    """Extract the CPU model from ``/proc/cpuinfo`` content ("model name" line, x86 and ARM)."""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("model name", "Model", "Hardware") and value.strip():
            return value.strip()
    return ""


def _parse_meminfo(text: str) -> int:
    """Extract total RAM in GB from ``/proc/meminfo`` content (``MemTotal:  N kB``)."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            with contextlib.suppress(ValueError, IndexError):
                kb = int(line.split()[1])
                return round(kb * 1024 / _BYTES_PER_GB)
    return 0


def _read_text(path: str) -> str:
    """Read *path* or return "" (procfs may be absent in containers or exotic setups)."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _cpu_model(system: str) -> str:
    """Return the CPU model string for *system* ("Darwin" / "Linux" / "Windows")."""
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if system == "Linux":
        return _parse_cpuinfo(_read_text("/proc/cpuinfo"))
    if system == "Windows":
        return os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor()
    return ""


def _memory_gb(system: str) -> int:
    """Return total RAM in whole GB for *system*, 0 when unknown."""
    if system == "Darwin":
        with contextlib.suppress(ValueError):
            return round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / _BYTES_PER_GB)
    if system == "Linux":
        return _parse_meminfo(_read_text("/proc/meminfo"))
    if system == "Windows":
        return _windows_memory_gb()
    return 0


def _windows_memory_gb() -> int:
    """Read total physical RAM via GlobalMemoryStatusEx; 0 off Windows or on failure."""
    try:
        import ctypes  # noqa: PLC0415 - a Windows-only import kept out of the module path.

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        # Typed as Any: ctypes generates the field attributes from _fields_ at runtime, which
        # static analyzers cannot see.
        status: Any = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return round(float(status.ullTotalPhys) / _BYTES_PER_GB)
    except Exception as exc:  # noqa: BLE001 - a hardware probe must never raise.
        logger.debug("hardware_memory_probe_failed", error=str(exc))
    return 0


def _first_line(text: str) -> str:
    """Return the first non-blank line of *text*, stripped."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _gpu_from_lspci(text: str) -> str:
    """Extract the display controller model from ``lspci`` output."""
    for line in text.splitlines():
        if "VGA compatible controller" in line or "3D controller" in line:
            return line.rsplit(": ", 1)[-1].strip()
    return ""


def _gpu_darwin(cpu: str) -> str:
    """
    Return the GPU on macOS.

    Apple Silicon integrates the GPU into the SoC, so the chip name is the GPU name; asking
    system_profiler would only repeat it more slowly. Intel Macs get the (slower) profiler query.
    """
    if cpu.startswith("Apple "):
        return cpu
    raw = _run(["system_profiler", "-json", "SPDisplaysDataType"])
    with contextlib.suppress(ValueError, KeyError, TypeError, IndexError):
        displays = json.loads(raw)["SPDisplaysDataType"]
        names = [str(d["sppci_model"]) for d in displays if d.get("sppci_model")]
        return " + ".join(names)
    return ""


def _gpu_model(system: str, cpu: str) -> str:
    """Return the GPU model for *system*; "" when nothing can be probed."""
    # An NVIDIA card is the one every provider backend can actually use for inference, and
    # nvidia-smi reports it identically on Linux and Windows, so try it first everywhere.
    if nvidia := _first_line(_run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])):
        return nvidia
    if system == "Darwin":
        return _gpu_darwin(cpu)
    if system == "Linux":
        return _gpu_from_lspci(_run(["lspci"]))
    if system == "Windows":
        return _first_line(
            _run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController).Name",
                ],
            ),
        )
    return ""


def probe_hardware() -> HardwareInfo:
    """
    Probe the local machine.

    Each fact degrades to empty/zero independently.
    """
    system = platform.system()
    cpu = _cpu_model(system)
    return HardwareInfo(
        cpu=cpu,
        gpu=_gpu_model(system, cpu),
        cpu_count=os.cpu_count() or 0,
        memory_gb=_memory_gb(system),
    )


def hardware_info(cache_path: Path | None) -> HardwareInfo:
    """
    Return the machine's hardware facts, cached at *cache_path*.

    Probing shells out to OS tools, so the first call per install pays up to a few seconds (on the
    telemetry thread, never the tagging path) and every later call reads the cache. Pass ``None`` to
    skip caching. A corrupt or version-mismatched cache is silently re-probed.
    """
    if cache_path is not None:
        with contextlib.suppress(OSError, ValueError, TypeError, KeyError):
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("version") == _CACHE_VERSION:
                return HardwareInfo(
                    cpu=str(cached["cpu"]),
                    gpu=str(cached["gpu"]),
                    cpu_count=int(cached["cpu_count"]),
                    memory_gb=int(cached["memory_gb"]),
                )

    info = probe_hardware()
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": _CACHE_VERSION, **asdict(info)}
            cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.debug("hardware_cache_persist_failed", error=str(exc))
    return info
