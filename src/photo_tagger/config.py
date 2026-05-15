"""Static configuration: defaults, prompts, and shared constants."""

import os
from typing import Literal


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"]

MIN_HIERARCHICAL_DEPTH = 2

# Provider URLs and API keys are read from the environment so the user can override them
# without changing code. Defaults are local-only addresses for Ollama and LM Studio.
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", os.getenv("OPENAI_API_KEY"))

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-vl-30b")
DEFAULT_JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
DEFAULT_DIMENSIONS = int(os.getenv("JPEG_DIMENSIONS", "1280"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
DEFAULT_RETRIES = int(os.getenv("RETRIES", "5"))

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

# ExifTool tag names referenced from multiple modules. Keeping them centralised avoids the
# typo class of bugs that hits when a literal is duplicated across read/write call sites.
TAG_IPTC_KEYWORDS = "IPTC:Keywords"
TAG_XMP_SUBJECT = "XMP:Subject"
TAG_XMP_HIERARCHICAL_SUBJECT = "XMP:HierarchicalSubject"
TAG_XMP_WEIGHTED_FLAT_SUBJECT = "XMP:WeightedFlatSubject"

DEFAULT_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
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
    "<|im_end|>"
)

DEFAULT_USER_PROMPT = (
    "Execute your mission: analyze this image and generate the structured metadata."
)
