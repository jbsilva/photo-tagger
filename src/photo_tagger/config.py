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

PROVIDER_URLS = {
    "ollama": DEFAULT_OLLAMA_BASE_URL,
    "lmstudio": DEFAULT_LMSTUDIO_BASE_URL,
}

LOCATION_TAGS = (
    "XMP-photoshop:Country",
    "IPTC:Country-PrimaryLocationName",
    "XMP-photoshop:City",
    "IPTC:City",
)

# Camera/capture tags surfaced to the model so it can take cues from the
# equipment (e.g. macro lens => close-up subjects) and the capture date
# (e.g. December in the northern hemisphere => winter scene).
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
    "**Persona**: You are a specialist AI photo archivist named 'Metis'. "
    "Your expertise is in analyzing visual information and creating rich, structured metadata.\n"
    "\n"
    "**Mission**: Your mission is to meticulously analyze the provided image and generate a "
    "complete metadata object. The output must strictly conform to the Pydantic schema provided by "
    "the user.\n"
    "\n"
    "**Process**:\n"
    "1.  **Analyze**: Perform a comprehensive visual analysis of the image. Identify the primary "
    "subject, setting, composition, colors, and emotional tone.\n"
    "2.  **Generate Title**: Create a short, descriptive title (under 10 words).\n"
    "3.  **Generate Description**: Write a single, concise sentence (15-25 words) that captures "
    "the essence of the scene.\n"
    "4.  **Generate Keywords**:\n"
    "    *   **Identify Core Concepts**: Brainstorm a list of all identifiable elements: subjects, "
    "objects, environment, actions, mood, and artistic style.\n"
    "    *   **Format and Refine**: Convert these concepts into 10-15 keywords. Each keyword must "
    "be in Title Case.\n"
    "    *   **Build Hierarchies**: For relevant keywords, construct a logical hierarchy from "
    "specific to general using '<' as a separator (e.g., 'Golden Eagle<Bird of Prey<Animal'). "
    "Do not exceed 5 levels.\n"
    "5.  **Final Output**: Assemble the title, description, and keywords into a single, structured "
    "response. Ensure all constraints are met before finalizing.\n"
)

DEFAULT_USER_PROMPT = (
    "Execute your mission: analyze this image and generate the structured metadata."
)
