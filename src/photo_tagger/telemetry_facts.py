"""
Privacy-safe facts for telemetry, derived from the machine and a run's settings.

Each helper reduces something that could identify a user (a server URL, an install path, the
environment) to a coarse category, and only the category leaves the machine. They are pure or read
only local state, so :mod:`photo_tagger.telemetry` stays the one place that decides what is sent.
"""

import importlib.metadata
import ipaddress
import json
import locale
import os
import re
import sys
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


# Where a model server runs, when it is not a known public service.
ENDPOINT_LOCALHOST = "localhost"
ENDPOINT_LAN = "lan"
ENDPOINT_TAILNET = "tailnet"
ENDPOINT_PUBLIC = "public"
ENDPOINT_INVALID = "invalid"

# Public services an OpenAI-compatible URL may point at. Their hostnames name a company, not a
# user, so they are reported as they are; any other public host becomes "public".
# cspell:disable - company hostnames, not words.
KNOWN_ENDPOINT_HOSTS = frozenset(
    {
        "api.anthropic.com",
        "api.cerebras.ai",
        "api.deepinfra.com",
        "api.deepseek.com",
        "api.fireworks.ai",
        "api.groq.com",
        "api.hyperbolic.xyz",
        "api.mistral.ai",
        "api.moonshot.ai",
        "api.novita.ai",
        "api.openai.com",
        "api.perplexity.ai",
        "api.sambanova.ai",
        "api.together.xyz",
        "api.x.ai",
        "dashscope-intl.aliyuncs.com",
        "dashscope.aliyuncs.com",
        "generativelanguage.googleapis.com",
        "integrate.api.nvidia.com",
        "models.github.ai",
        "open.bigmodel.cn",
        "openrouter.ai",
        "router.huggingface.co",
    },
)
# Services whose hostname starts with the customer's own resource name: only the suffix is kept.
_KNOWN_ENDPOINT_SUFFIXES = (
    ".openai.azure.com",
    ".services.ai.azure.com",
    ".cognitiveservices.azure.com",
)
# cspell:enable - back to checking.
_LAN_SUFFIXES = (".local", ".lan", ".home", ".internal", ".home.arpa", ".localdomain", ".corp")
# Tailscale and friends hand out addresses from the carrier-grade NAT range.
_CGNAT = ipaddress.ip_network("100.64.0.0/10")  # NOSONAR S1313 - a range to classify, never dialed

# Installers named in a distribution's INSTALLER file that are worth telling apart.
_KNOWN_INSTALLERS = frozenset(
    {"conda", "hatch", "mamba", "micromamba", "pdm", "pip", "pipx", "pixi", "poetry", "rye", "uv"},
)

# CI systems set one of these; a run under one is automation, not a person at a keyboard.
_CI_VARIABLES = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
    "TRAVIS",
)

# The environment variables photo-tagger reads. Only their names are reported, never a value: an
# API key variable says "a key is configured", nothing more.
KNOWN_ENV_VARS = (
    "BATTERY_FLOOR",
    "BATTERY_RESUME",
    "DUTY_PERCENT",
    "FREQUENCY_PENALTY",
    "JPEG_DIMENSIONS",
    "JPEG_QUALITY",
    "LLAMA_CPP_API_KEY",
    "LLAMA_CPP_BASE_URL",
    "LM_STUDIO_API_KEY",
    "LM_STUDIO_BASE_URL",
    "MAX_PAUSE_MINUTES",
    "MAX_TOKENS",
    "MODEL_NAME",
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "PHOTO_TAGGER_CONFIG",
    "PHOTO_TAGGER_EXIFTOOL",
    "PHOTO_TAGGER_LANG",
    "RETRIES",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "WORK_MINUTES",
)

# Image formats worth naming. Any other extension a batch or an --ext value holds is only counted.
KNOWN_IMAGE_EXTENSIONS = frozenset(
    {
        "3fr",
        "arw",
        "avif",
        "bmp",
        "cap",
        "cr2",
        "cr3",
        "crw",
        "dcr",
        "dng",
        "eip",
        "erf",
        "fff",
        "gif",
        "gpr",
        "heic",
        "heif",
        "hif",
        "iiq",
        "jfif",
        "jpe",
        "jpeg",
        "jpg",
        "jxl",
        "k25",
        "kdc",
        "mef",
        "mos",
        "mrw",
        "nef",
        "nrw",
        "orf",
        "ori",
        "pef",
        "png",
        "ptx",
        "raf",
        "raw",
        "rw2",
        "rwl",
        "sr2",
        "srf",
        "srw",
        "tif",
        "tiff",
        "webp",
        "x3f",
    },
)
OTHER_EXTENSION = "other"

# Output languages a beacon may name. The option is free text and can hold anything, so a language
# outside this list is sent as OTHER_LANGUAGE. These are English names, matched without regard to
# case, because that is how the option is spelled (the GUI offers the same names).
KNOWN_OUTPUT_LANGUAGES = (
    "Arabic",
    "Brazilian Portuguese",
    "Catalan",
    "Chinese",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "European Portuguese",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Simplified Chinese",
    "Spanish",
    "Swedish",
    "Thai",
    "Traditional Chinese",
    "Turkish",
    "Ukrainian",
    "Vietnamese",
)
_LANGUAGE_BY_KEY = {name.casefold(): name for name in KNOWN_OUTPUT_LANGUAGES}
OTHER_LANGUAGE = "other"

# A model id is a name the server chose, but some servers choose a file path (llama-server reports
# its -m path unless --alias is set). Only the file name of such an id is sent.
_MODEL_FILE_SUFFIXES = (".gguf", ".ggml", ".bin", ".safetensors", ".pt", ".pth", ".onnx", ".mlx")
_WINDOWS_DRIVE = re.compile(r"^[a-z]:[\\/]", re.IGNORECASE)
MAX_MODEL_CHARS = 100
# The slashes in a model hub id such as "qwen/qwen3-vl-30b".
_HUB_ID_SLASHES = 1
# The collector's clamp for a string inside a beacon (MAX_JSON_STR in telemetry/worker.js).
MAX_JOINED_CHARS = 1000

_LOCALE_PATTERN = re.compile(r"^([a-z]{2,3})(?:[_-]([a-z]{2}|\d{3}))?\b", re.IGNORECASE)


def language_label(language: str) -> str:
    """Return *language* as one of the known language names, or ``other``; never what was typed."""
    return _LANGUAGE_BY_KEY.get(language.strip().casefold(), OTHER_LANGUAGE)


def model_label(model: str) -> str:
    """
    Return the model name a beacon may carry.

    The name identifies the model, which is the point, but two kinds of id carry more. A path keeps
    only its file name, since a folder can name the person. A fine-tuned OpenAI id
    (``ft:base:org:suffix:id``) keeps only its base model, since the rest names the organization.
    """
    name = model.strip()
    if name.startswith("ft:"):
        return "ft:" + name.split(":")[1]
    if _looks_like_path(name):
        name = re.split(r"[\\/]", name.rstrip("/\\"))[-1]
    return name[:MAX_MODEL_CHARS]


def _looks_like_path(name: str) -> bool:
    """Say whether a model id is a file path rather than a name like ``qwen/qwen3-vl-30b``."""
    if name.startswith(("/", "~", ".", "\\")) or "\\" in name or _WINDOWS_DRIVE.match(name):
        return True
    # A hub id has one slash (org/model); a deeper relative path is a folder of the user's.
    return name.count("/") > _HUB_ID_SLASHES or (
        "/" in name and name.lower().endswith(_MODEL_FILE_SUFFIXES)
    )


def endpoint_kind(url: str | None) -> str:
    """
    Classify where a model server runs, from its URL, without revealing the URL.

    Returns a known public service's hostname, or one of ``localhost``, ``lan``, ``tailnet``,
    ``public``, and ``invalid``. A private hostname or address never leaves the machine.
    """
    try:
        host = (urlsplit(url or "").hostname or "").lower().rstrip(".")
    except ValueError:
        # urlsplit raises on a malformed bracket ("http://[abc/v1"). The run reports that URL as
        # it would any other it cannot use, instead of crashing on the way to a beacon.
        return ENDPOINT_INVALID
    if not host:
        return ENDPOINT_INVALID
    if (service := _known_service(host)) is not None:
        return service
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return _hostname_kind(host)
    return _address_kind(address)


def _known_service(host: str) -> str | None:
    """Name the public service *host* belongs to, or None when it is none of them."""
    if host in KNOWN_ENDPOINT_HOSTS:
        return host
    return next(
        (suffix.removeprefix(".") for suffix in _KNOWN_ENDPOINT_SUFFIXES if host.endswith(suffix)),
        None,
    )


def _address_kind(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """Classify an IP address: loopback, a tailnet, a private network, or the internet."""
    if address.is_loopback:
        return ENDPOINT_LOCALHOST
    if address in _CGNAT:
        return ENDPOINT_TAILNET
    if address.is_private or address.is_link_local:
        return ENDPOINT_LAN
    return ENDPOINT_PUBLIC


def _hostname_kind(host: str) -> str:
    """Classify a hostname that is not an IP address."""
    if host == "localhost" or host.endswith(".localhost"):
        return ENDPOINT_LOCALHOST
    if host.endswith(".ts.net"):
        return ENDPOINT_TAILNET
    if "." not in host or host.endswith(_LAN_SUFFIXES):
        return ENDPOINT_LAN
    return ENDPOINT_PUBLIC


def _distribution() -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution("photo-tagger")
    except importlib.metadata.PackageNotFoundError:
        return None


def install_method() -> str:
    """
    Name how photo-tagger was installed: app, editable, git, conda, uv-tool, pipx, venv, or system.

    ``app`` is the bundled desktop app and ``editable`` a development checkout; ``source`` means it
    runs without being installed at all. The install location itself is never reported.
    """
    if getattr(sys, "frozen", False):
        return "app"
    dist = _distribution()
    if dist is None:
        return "source"
    return _direct_url_method(dist) or _prefix_method()


def _direct_url_method(dist: importlib.metadata.Distribution) -> str | None:
    """Recognize a development checkout or a git install from the install record."""
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        return None
    if not isinstance(direct, dict):
        return None
    dir_info = direct.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        return "editable"
    return "git" if "vcs_info" in direct else None


def _prefix_method() -> str:
    """Recognize the environment kind from where the interpreter lives, never reporting where."""
    prefix = Path(sys.prefix)
    posix = prefix.as_posix().lower()
    if (prefix / "conda-meta").is_dir():
        return "conda"
    if "/uv/tools/" in posix:
        return "uv-tool"
    if "/pipx/venvs/" in posix:
        return "pipx"
    return "venv" if sys.prefix != sys.base_prefix else "system"


def installer() -> str:
    """Name the tool that installed photo-tagger (uv, pip, conda, ...), "other", or ""."""
    dist = _distribution()
    name = ((dist.read_text("INSTALLER") if dist else None) or "").strip().lower()
    if not name:
        return ""
    return name if name in _KNOWN_INSTALLERS else "other"


def system_locale(hint: str | None = None) -> str:
    """
    Return the OS locale as a language tag such as ``de_DE``, or "" when there is none.

    *hint* is a locale name from elsewhere (the GUI passes Qt's), used before the environment. Only
    the language and region survive: an encoding or a modifier is dropped.
    """
    candidates = [
        hint,
        os.environ.get("LC_ALL"),
        os.environ.get("LC_MESSAGES"),
        os.environ.get("LANG"),
        locale.getlocale()[0],
    ]
    for candidate in candidates:
        if candidate and (match := _LOCALE_PATTERN.match(candidate)):
            language, region = match.groups()
            return f"{language.lower()}_{region.upper()}" if region else language.lower()
    return ""


def extension_list(extensions: Iterable[str]) -> str:
    """
    Name the known image formats among *extensions*, sorted and comma-joined.

    Case and dots do not matter. Anything else becomes a single "other", so an unusual suffix a user
    typed never leaves the machine.
    """
    cleaned = {ext.strip().lstrip(".").lower() for ext in extensions}
    cleaned.discard("")
    known = sorted(cleaned & KNOWN_IMAGE_EXTENSIONS)
    return ",".join([*known, OTHER_EXTENSION] if cleaned - KNOWN_IMAGE_EXTENSIONS else known)


def local_clock(now: datetime | None = None) -> tuple[int, int, int]:
    """Return ``(utc_offset_minutes, local_hour, local_weekday)``, Monday being weekday 0."""
    local = (now or datetime.now(tz=UTC)).astimezone()
    offset = local.utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    return minutes, local.hour, local.weekday()


def running_in_ci() -> bool:
    """Return True when a CI system's marker variable is set."""
    return any(
        (value := os.environ.get(name)) and value.strip().lower() not in {"0", "false", "no"}
        for name in _CI_VARIABLES
    )


def attached_terminal() -> bool:
    """Return True when a person could be watching: stdin and stderr are both terminals."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except AttributeError, ValueError:
        # A missing or closed stream (a GUI app launched from Finder) is no terminal.
        return False


def env_var_names(environ: Mapping[str, str] | None = None) -> str:
    """Name the photo-tagger variables that are set, sorted and comma-joined, never their values."""
    source = os.environ if environ is None else environ
    return ",".join(sorted(name for name in KNOWN_ENV_VARS if source.get(name)))


def joined(names: Iterable[str]) -> str:
    """
    Sort and comma-join *names*, dropping duplicates and blanks.

    The collector clamps a string at MAX_JOINED_CHARS, which would cut a name in half, so a longer
    list stops at the last whole name that fits.
    """
    kept: list[str] = []
    length = -1
    for name in sorted({name for name in names if name}):
        if length + len(name) + 1 > MAX_JOINED_CHARS:
            continue  # a later, shorter name may still fit
        length += len(name) + 1
        kept.append(name)
    return ",".join(kept)


def changed_settings(current: object, baseline: object) -> str:
    """Name the dataclass fields where *current* differs from *baseline*, sorted and joined."""
    return joined(
        field.name
        for field in fields(current)  # type: ignore[arg-type]
        if getattr(current, field.name) != getattr(baseline, field.name)
    )


def config_key_names(
    config: Mapping[str, object],
    known: Mapping[str, frozenset[str]],
) -> tuple[str, int]:
    """
    Name the keys a config file sets, as ``table.key``, and count the ones photo-tagger ignores.

    *known* maps each table to its keys, with ``""`` for the top level. An unknown key is counted,
    never named: it is text a user typed, and it could say anything.
    """
    found: list[str] = []
    unknown = 0
    for key, value in config.items():
        if isinstance(value, dict):
            table_keys = known.get(key)
            for inner in value:
                if table_keys is not None and inner in table_keys:
                    found.append(f"{key}.{inner}")
                else:
                    unknown += 1
        elif key in known.get("", frozenset()):
            found.append(key)
        else:
            unknown += 1
    return joined(found), unknown


def quantile(values: Sequence[float], q: float) -> float:
    """Return the *q* quantile of *values* by linear interpolation, or 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of *values*, or 0.0 when empty."""
    return sum(values) / len(values) if values else 0.0
