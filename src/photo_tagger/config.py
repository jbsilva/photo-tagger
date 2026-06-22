"""Static configuration: defaults, prompts, and shared constants."""

import os
import warnings
from typing import Literal


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]

MIN_HIERARCHICAL_DEPTH = 2


def _env_int(name: str, default: int) -> int:
    """Read *name* from the environment as an int, warning and falling back if malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not a valid integer; using default {default}",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Read *name* from the environment as a float, warning and falling back if malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not a valid float; using default {default}",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


# Provider URLs and API keys are read from the environment so the user can override them
# without changing code. Defaults are local-only addresses for Ollama and LM Studio.
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", os.getenv("OPENAI_API_KEY"))
# Hosted, OpenAI-compatible endpoints (the real OpenAI API or any drop-in gateway).
# Unlike the local providers this one needs a key, so there is no usable local default.
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-vl-30b")
DEFAULT_JPEG_QUALITY = _env_int("JPEG_QUALITY", 80)
DEFAULT_DIMENSIONS = _env_int("JPEG_DIMENSIONS", 1280)
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

LOCATION_TAGS = (
    "XMP-photoshop:Country",
    "IPTC:Country-PrimaryLocationName",
    "XMP-photoshop:City",
    "IPTC:City",
)

# Camera/capture tags surfaced to the model as corroborative evidence. The prompt instructs the
# model to use them only to specify or disambiguate what is visible in the image, never to assert
# content the image doesn't show.
CAMERA_TAGS = (
    "EXIF:Model",
    "EXIF:LensModel",
    "EXIF:DateTimeOriginal",
)

# ExifTool tag names referenced from multiple modules. Keeping them centralized avoids the
# typo class of bugs that hits when a literal is duplicated across read/write call sites.
TAG_IPTC_KEYWORDS = "IPTC:Keywords"
TAG_XMP_SUBJECT = "XMP:Subject"
TAG_XMP_HIERARCHICAL_SUBJECT = "XMP:HierarchicalSubject"
TAG_XMP_WEIGHTED_FLAT_SUBJECT = "XMP:WeightedFlatSubject"

# Plain text only. The OpenAI-compatible chat completion endpoint wraps this in the
# model's chat template (e.g. <|im_start|>system ... <|im_end|> for Qwen), so embedding
# template tokens here would cause them to be applied twice and corrupt the prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You generate structured metadata for a single photograph. Output must conform to the schema "
    "provided by the user.\n"
    "\n"
    "**Fields**:\n"
    "- Title: under 10 words, descriptive, Title Case, English.\n"
    "- Description: one sentence, 15-25 words, present tense, English.\n"
    "- Keywords: up to 15 flat keywords, Title Case, English. Cover subject, setting, action, "
    "mood, and style. Every keyword must be supported by something visible in the image.\n"
    "- Hierarchies: a SEPARATE list (the 'hierarchies' field, not 'keywords'). For each keyword "
    "that has a natural taxonomy, add one chain written specific-to-general with '<' as the "
    "separator, max 5 levels, e.g. 'Golden Eagle<Bird of Prey<Animal' or 'Oak<Tree<Plant'. "
    "Populate this field whenever a subject, place, or object has an obvious parent category; "
    "leave it empty only when nothing in the image has one.\n"
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

DEFAULT_USER_PROMPT = (
    "Execute your mission: analyze this image and generate the structured metadata."
)
