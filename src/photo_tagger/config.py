"""Static configuration: defaults, prompts, and shared constants."""

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]

MIN_HIERARCHICAL_DEPTH = 2

# Image formats Pillow reads natively. Everything else counts as camera RAW, for two callers: the
# image loader tries rawpy first, and the "raw" sidecar mode keeps those files untouched. An
# extension nobody here knows is therefore treated as RAW, which is the cautious answer both times.
NON_RAW_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".jpe",
        ".jp2",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
        ".psd",
        ".ico",
        ".ppm",
        ".pgm",
        ".pbm",
    },
)

# Where a run puts the metadata it writes: "all" into XMP sidecars, "none" into the image files
# themselves, "raw" into a sidecar for RAW files and into the image for everything else, "both"
# into each photo and a sidecar beside it. "raw" is the mixed-folder answer: a RAW keeps its bytes
# untouched while the JPEG beside it carries its own metadata.
SidecarMode = Literal["all", "none", "raw", "both"]
DEFAULT_SIDECAR_MODE: SidecarMode = "all"

# Where one file type goes, whatever the mode says (--sidecar-override). Named by the file, not by
# the mode's "all"/"none", because "dng=photo" reads as what it does and "dng=none" does not. The
# case in point is Lightroom Classic: it keeps a DNG's metadata inside the DNG, not in a sidecar.
FileTypeTarget = Literal["sidecar", "photo", "both"]
# Keyed by the lowercase suffix with its dot (".dng"), the form Path.suffix gives.
type SidecarOverrides = Mapping[str, FileTypeTarget]


def _env_number[N](name: str, default: N, parse: Callable[[str], N], kind: str) -> N:
    """Read *name* from the environment with *parse*, warning and falling back if malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return parse(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not a valid {kind}; using default {default}",
            RuntimeWarning,
            stacklevel=3,
        )
        return default


def _env_int(name: str, default: int) -> int:
    """Read *name* from the environment as an int, warning and falling back if malformed."""
    return _env_number(name, default, int, "integer")


def _env_float(name: str, default: float) -> float:
    """Read *name* from the environment as a float, warning and falling back if malformed."""
    return _env_number(name, default, float, "float")


def _env_in_range[N: (int, float)](name: str, value: N, default: N, low: N, high: N) -> N:
    """
    Return *value* when it lies in ``low..high``, else warn and return *default*.

    An environment variable becomes a default, and a default never passes the flag's validator, so
    ``DUTY_PERCENT=0`` would otherwise reach the pacer as a run that may never work.
    """
    if low <= value <= high:
        return value
    warnings.warn(
        f"{name}={value!r} is outside {low}..{high}; using default {default}",
        RuntimeWarning,
        stacklevel=3,
    )
    return default


# Provider URLs and API keys are read from the environment so the user can override them
# without changing code. Defaults are local-only addresses for Ollama and LM Studio.
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
# Deliberately does not fall back to OPENAI_API_KEY: LM Studio is the default provider, and a
# real OpenAI key sitting in the environment for unrelated tooling must never be sent, as a
# Bearer token, to whatever host --url happens to point at.
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY")
# llama.cpp's llama-server speaks the OpenAI API on port 8080 by default; a key is optional
# (only checked when the server was started with --api-key).
DEFAULT_LLAMACPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
DEFAULT_LLAMACPP_API_KEY = os.getenv("LLAMA_CPP_API_KEY")
# Hosted, OpenAI-compatible endpoints (the real OpenAI API or any drop-in gateway).
# Unlike the local providers this one needs a key, so there is no usable local default.
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-vl-30b")
DEFAULT_JPEG_QUALITY = _env_int("JPEG_QUALITY", 80)
# The longest edge of the JPEG the model is shown. Prefill time scales with the pixel count, and
# it is the largest single lever on how long a photo takes: measured against a local
# Qwen3-VL-30B, one photo costs 0.49s of prefill at 640, 0.83s at 896, 1.11s at 1024 and 1.80s at
# 1280, while the rest of the call (about 1.9s to write 200 tokens) does not move. What the extra
# pixels buy is text inside the frame: 640 is faster, 768 sees more detail, 896 reads large text,
# 1024 reads fine text. Multiples of 128 line up with the vision tower's patch grid.
DEFAULT_DIMENSIONS = _env_int("JPEG_DIMENSIONS", 640)
DEFAULT_TEMPERATURE = _env_float("TEMPERATURE", 0.2)
DEFAULT_MAX_TOKENS = _env_int("MAX_TOKENS", 1200)
DEFAULT_RETRIES = _env_int("RETRIES", 5)
# Hard cap on a single inference HTTP call. A healthy run is a few seconds; this exists
# only to abort pathological cases (token-repetition loops, model hangs) so the retry
# path can step in instead of the call burning ten minutes against the OpenAI client
# default.
DEFAULT_TIMEOUT_SECONDS = _env_float("TIMEOUT_SECONDS", 60.0)
# Discourage the model from chanting the same token over and over. At temperature 0.2 with no
# penalty Qwen3-VL has been observed to repeat a single keyword (e.g. "Human Adult Female") until
# the token budget is exhausted. 0.5 is mild enough to leave legitimate wording intact while
# strongly suppressing pathological loops.
DEFAULT_FREQUENCY_PENALTY = _env_float("FREQUENCY_PENALTY", 0.5)

# Pacing (see photo_tagger.pacing). Nothing here rests a run on its own: a full duty cycle with no
# battery floor builds no pacer at all, so an ordinary run works flat out exactly as it did before
# any of this existed. The two windows below are the shape a rest takes once --duty or
# --battery-floor asks for one, and change nothing until then.
DEFAULT_DUTY_PERCENT = _env_in_range("DUTY_PERCENT", _env_int("DUTY_PERCENT", 100), 100, 1, 100)
DEFAULT_BATTERY_FLOOR = _env_in_range("BATTERY_FLOOR", _env_int("BATTERY_FLOOR", 0), 0, 0, 99)
DEFAULT_BATTERY_RESUME = _env_in_range("BATTERY_RESUME", _env_int("BATTERY_RESUME", 0), 0, 0, 100)
# How long a paced run works between rests. Long enough that a rest is worth taking (a few seconds
# of idling recharges nothing and only gives a model server time to unload), short enough that the
# battery is looked at often.
DEFAULT_WORK_MINUTES = _env_in_range(
    "WORK_MINUTES",
    _env_float("WORK_MINUTES", 10.0),
    10.0,
    0.1,
    24 * 60.0,
)
# The ceiling on one rest in a paced run. LM Studio unloads a JIT-loaded model after an hour idle,
# and a run that sits still any longer looks hung to whoever is watching it.
DEFAULT_MAX_PAUSE_MINUTES = _env_in_range(
    "MAX_PAUSE_MINUTES",
    _env_float("MAX_PAUSE_MINUTES", 15.0),
    15.0,
    0.1,
    24 * 60.0,
)


def state_dir() -> Path:
    """
    Return the directory for photo-tagger's own state, honoring ``XDG_STATE_HOME``.

    Falls back to ``~/.local/state/photo-tagger``. This is state the app maintains (the install id,
    the undo journals), not configuration the user edits, so it lives apart from the TOML file.
    """
    base = os.getenv("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "photo-tagger"


def exiftool_executable() -> str | None:
    """
    Return an explicit ExifTool binary path, or None to find ``exiftool`` on PATH.

    Lets users on non-standard installs point at their exact binary. Read at call time (not import)
    so a config-file ``exiftool_path`` bridged into ``PHOTO_TAGGER_EXIFTOOL`` at startup is honored;
    an explicitly-exported env var always wins over the config file.
    """
    return os.getenv("PHOTO_TAGGER_EXIFTOOL") or None


# XMP-photoshop and IPTC each define their own place fields, so every reader takes both and uses
# the first one that answers.
COUNTRY_TAGS = ("XMP-photoshop:Country", "IPTC:Country-PrimaryLocationName")
CITY_TAGS = ("XMP-photoshop:City", "IPTC:City")
LOCATION_TAGS = (*COUNTRY_TAGS, *CITY_TAGS)

# When the shutter fired. Read as prompt context, and on its own when grouping a batch into shoots.
TAG_EXIF_DATE_TIME_ORIGINAL = "EXIF:DateTimeOriginal"

# Camera/capture tags surfaced to the model as corroborative evidence. The prompt instructs the
# model to use them only to specify or disambiguate what is visible in the image, never to assert
# content the image doesn't show.
CAMERA_TAGS = (
    "EXIF:Model",
    "EXIF:LensModel",
    TAG_EXIF_DATE_TIME_ORIGINAL,
)

# ExifTool tag names referenced from multiple modules. Keeping them centralized avoids the
# typo class of bugs that hits when a literal is duplicated across read/write call sites.
TAG_IPTC_KEYWORDS = "IPTC:Keywords"
TAG_XMP_SUBJECT = "XMP:Subject"
TAG_XMP_HIERARCHICAL_SUBJECT = "XMP:HierarchicalSubject"
TAG_XMP_WEIGHTED_FLAT_SUBJECT = "XMP:WeightedFlatSubject"
# Title and description tags. Reads use the generic family-group tag (e.g. "XMP:Title"); writes
# target the specific namespace tag (e.g. "XMP-dc:Title"). IPTC:ObjectName is shared by both.
TAG_IPTC_OBJECT_NAME = "IPTC:ObjectName"
TAG_XMP_TITLE = "XMP:Title"
TAG_XMP_DESCRIPTION = "XMP:Description"
TAG_EXIF_IMAGE_DESCRIPTION = "EXIF:ImageDescription"
# The family-0 spelling of XMP-tiff:ImageDescription, the mirror our own writes produce.
TAG_XMP_IMAGE_DESCRIPTION = "XMP:ImageDescription"

# The language the model writes titles, descriptions, and keywords in. Any language name the
# model understands works ("German", "Brazilian Portuguese", ...); it is spliced into the system
# prompt as-is.
DEFAULT_OUTPUT_LANGUAGE = "English"

# Plain text only. The OpenAI-compatible chat completion endpoint wraps this in the
# model's chat template (e.g. <|im_start|>system ... <|im_end|> for Qwen), so embedding
# template tokens here would cause them to be applied twice and corrupt the prompt.
_SYSTEM_PROMPT_TEMPLATE = (
    "You generate structured metadata for a single photograph. Output must conform to the schema "
    "provided by the user.\n"
    "\n"
    "**Fields**:\n"
    "- Title: under 10 words, descriptive, Title Case, {language}.\n"
    "- Description: one sentence, 15-25 words, present tense, {language}.\n"
    "- Keywords: up to 15 flat keywords, Title Case, {language}. Cover subject, setting, action, "
    "mood, and style. Every keyword must be supported by something visible in the image.\n"
    "- Hierarchies: a SEPARATE list (the 'hierarchies' field, not 'keywords'). For each keyword "
    "that has a natural taxonomy, add one chain written specific-to-general with '<' as the "
    "separator, max 5 levels, e.g. 'Golden Eagle<Bird of Prey<Animal' or 'Oak<Tree<Plant'. "
    "At most ONE chain per keyword: pick its single most natural taxonomy, never several "
    "alternative parents for the same keyword. Never repeat a chain, never write a term as its "
    "own parent ('Animal<Animal'), and end each chain at a broad category such as 'Animal', "
    "'Plant', 'Object', or 'Place' instead of climbing further. "
    "Populate this field whenever a subject, place, or object has an obvious parent category; "
    "leave it empty only when nothing in the image has one.\n"
    "- Language: {language} only, in every field, including every hierarchy segment. Never mix "
    "languages or scripts within a keyword or append a translation to one (write 'Bird Perching', "
    "never 'Bird栖息'). When a subject has a common {language} name, use it, even if the name it "
    "carries in its own culture is more precise (write 'Ocarina', never '陶笛'). Keep a foreign "
    "term only when the photo shows that text, or when {language} genuinely has no word for it.\n"
    "\n"
    "**Ground truth is the image.** The 'Existing Metadata' block, when present in the user "
    "message, is corroborative evidence only. Use it to disambiguate or specify what you already "
    "see (e.g. GPS in Italy plus a visible cathedral facade allows 'Italian Architecture'). Reuse "
    "the vocabulary of any Existing Keywords for style consistency.\n"
    "\n"
    "Do NOT use the metadata block to write keywords for things you cannot see in the image. A "
    "macro lens in EXIF does not mean the photo is a close-up. A December capture date does not "
    "mean winter. A country tag does not mean its landmarks are present. Do not output the camera "
    "body, lens model, or capture timestamp as keywords; they describe equipment, not subject "
    "content. Do not copy existing keywords as filler if the concept isn't in this image. If the "
    "image and the metadata disagree, trust the image.\n"
    "\n"
    "**When uncertain**: prefer a broader category over guessing a specific name. Write 'Bird' "
    "rather than 'Golden Eagle' if you cannot tell. Do not invent species, person names, or place "
    "names.\n"
)


def build_system_prompt(output_language: str = DEFAULT_OUTPUT_LANGUAGE) -> str:
    """
    Render the system prompt for *output_language*.

    The language name lands in the prompt verbatim, so any name the model understands works. The
    examples stay in English on purpose: they illustrate structure (hierarchy chains, no mixed
    scripts), which models transfer to the target language.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(language=output_language)


# How often to list the watched folders, and how long a file must be unchanged before it counts as
# finished. Two polls plus the age check is a cheap, portable stand-in for "the writer closed it".
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_SETTLE_SECONDS = 2.0


DEFAULT_USER_PROMPT = (
    "Execute your mission: analyze this image and generate the structured metadata."
)

# Sent back to the same agent, with the same image, when script_guard finds that an answer mixed
# writing systems. The previous answer is quoted so the model edits it rather than starting over,
# and the offending words are named so it does not have to find them again. The image goes along
# because the one question that decides the right fix ("is that text in the photo?") needs it.
_REPAIR_PROMPT_TEMPLATE = (
    "Your previous answer for this photo mixes writing systems. Every field must be written in "
    "{language}.\n"
    "\n"
    "Previous answer:\n"
    "- Title: {title}\n"
    "- Description: {description}\n"
    "- Keywords: {keywords}\n"
    "\n"
    "Not in {language}: {terms}\n"
    "\n"
    "Return the same metadata with each of those replaced by its ordinary {language} name (write "
    "'Ocarina' for '陶笛'). Leave everything else exactly as it was. If the photo actually shows "
    "that text (a sign, a book cover, a printed character), keep the term and put the {language} "
    "name beside it instead of dropping it. Never leave one word written half in one script and "
    "half in another."
)


def build_repair_prompt(
    *,
    title: str,
    description: str,
    keywords: Iterable[str],
    terms: Iterable[str],
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
) -> str:
    """Render the follow-up prompt that asks the model to redo an answer in *output_language*."""
    return _REPAIR_PROMPT_TEMPLATE.format(
        language=output_language,
        title=title,
        description=description,
        keywords=", ".join(keywords),
        terms=", ".join(terms),
    )
