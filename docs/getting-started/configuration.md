---
icon: lucide/settings
---

# Configuration

photo-tagger reads settings from three places, plus its built-in defaults. Knowing the order they
apply in saves you from chasing surprises when a value does not take effect.

## Precedence

When the same setting comes from more than one source, the one higher in this list wins:

1. **CLI flag** (for example `--temperature 0.3`)
2. **Config file** (a `[section]` key in TOML)
3. **Environment variable** (for example `TEMPERATURE`)
4. **Built-in default** (the value baked into the code)

So a CLI flag always beats the config file, the config file beats an environment variable, and an
environment variable beats the built-in default. This lets you set a stable baseline in a config
file or your shell profile and still override one value per run on the command line.

## Environment variables

These variables set defaults. Any matching CLI flag still overrides them. See
[CLI reference](../usage/cli-reference.md) for the flags they pair with.

| Variable              | Default                     | Purpose                                                                            |
| --------------------- | --------------------------- | ---------------------------------------------------------------------------------- |
| `PHOTO_TAGGER_CONFIG` | none                        | Explicit path to a config file (highest-priority config source).                   |
| `MODEL_NAME`          | `qwen/qwen3-vl-30b`         | Vision-language model identifier.                                                  |
| `OLLAMA_BASE_URL`     | `http://localhost:11434/v1` | Ollama API base URL.                                                               |
| `OLLAMA_API_KEY`      | none                        | API key for Ollama.                                                                |
| `LM_STUDIO_BASE_URL`  | `http://localhost:1234/v1`  | LM Studio API base URL.                                                            |
| `LM_STUDIO_API_KEY`   | none                        | API key for LM Studio.                                                             |
| `OPENAI_BASE_URL`     | `https://api.openai.com/v1` | Base URL for the `openai` provider (the real OpenAI API or a drop-in gateway).     |
| `OPENAI_API_KEY`      | none                        | API key for the `openai` provider (required); also the fallback key for LM Studio. |
| `TEMPERATURE`         | `0.2`                       | Sampling temperature.                                                              |
| `MAX_TOKENS`          | `1200`                      | Maximum tokens to generate.                                                        |
| `TIMEOUT_SECONDS`     | `60.0`                      | Per-image inference timeout in seconds.                                            |
| `FREQUENCY_PENALTY`   | `0.5`                       | Penalty on repeated tokens.                                                        |
| `JPEG_DIMENSIONS`     | `1280`                      | Max dimension (px) of the JPEG sent to the model.                                  |
| `JPEG_QUALITY`        | `80`                        | JPEG quality (1-100) of the image sent to the model.                               |
| `RETRIES`             | `5`                         | Retries when model output fails schema validation.                                 |

!!! tip

    Prefer the API-key variables over the `-k/--api-key` flag. Keys passed on the command line can leak
    into your shell history and process listings.

## Config file

A TOML config file is the best place for settings you reuse across runs. photo-tagger looks for one
in this order and uses the **first match**:

1. The path in `$PHOTO_TAGGER_CONFIG`
2. `.photo-tagger.toml` in the current directory
3. `~/.config/photo-tagger/config.toml`

If none of these exist, photo-tagger runs with environment-informed and built-in defaults only.

The sections below mirror the internal option groups. The example shows every section with its
default value, so you can copy it and change only what you need.

```toml
# .photo-tagger.toml

# Top-level keys (these have no section header).
extensions = "cr3,jpg"   # Comma-separated extensions used when scanning directories.
recursive = false        # Recurse into subdirectories while scanning input directories.
workers = 1              # Process N photos concurrently with a thread pool.

[provider]
model_name = "qwen/qwen3-vl-30b"          # Vision-language model identifier.
provider_name = "lmstudio"                # Backend: "ollama", "lmstudio", or "openai".
api_base_url = "http://localhost:1234/v1" # Provider API base URL.
# api_key = "..."                         # Prefer the API-key env vars instead.
retries = 5                               # Retries when output fails schema validation.

[inference]
temperature = 0.2        # Sampling temperature.
max_tokens = 1200        # Maximum tokens to generate.
timeout_seconds = 60.0   # Per-image inference timeout in seconds.
frequency_penalty = 0.5  # Penalty on repeated tokens; discourages output loops.
jpeg_dimensions = 1280   # Max dimension (px) of the JPEG sent to the model.
jpeg_quality = 80        # JPEG quality (1-100) of the image sent to the model.

[output]
preserve_keywords = true  # Merge with existing keywords (false replaces them).
write_title = true        # Generate and write a title.
write_description = true  # Generate and write a description.
backup_xmp = true         # Keep ExifTool's *_original backup before writing.
use_sidecar = true        # Write an XMP sidecar (false embeds into the image file).
dry_run = false           # Run the model and log results, but write nothing.
# max_keywords = 10       # Cap AI keywords kept per photo (omit to keep all).

[log]
file_log_level = "DEBUG"     # DEBUG/INFO/WARNING/ERROR/CRITICAL/OFF.
console_log_level = "INFO"   # Same levels; OFF disables.
log_folder = "logs"          # Folder for timestamped log files.

[display]
progress_bar = true   # Live progress bar; auto-disabled when stderr is not a TTY.
json_output = false   # Emit one NDJSON line per processed photo to stdout.

[filter]
skip_tagged = false   # Skip files that already have keywords, a title, or a description.
# newer_than = "2024-01-01"        # Drop files whose mtime is on/before this timestamp.
# older_than = "2024-01-01T14:30"  # Drop files whose mtime is on/after this timestamp.

[artifacts]
# prompt_file = "prompt.txt"   # Replace the default user prompt with this file's contents.
# summary_file = "summary.json" # Write a JSON run summary on completion.
# csv_file = "report.csv"      # Write a per-photo CSV report (one row per photo) as the run streams.
# cache_file = "cache.sqlite"   # SQLite cache of model outputs; reruns skip unchanged calls.
# lock_file = "photo-tagger.lock" # Exclusive lock; refuse to start if another run holds it.
```

!!! note

    Unknown keys are ignored. This keeps older config files working when new options are added, so a
    stray or misspelled key never stops a run. Double-check spelling if a setting seems to have no
    effect.

For the full per-flag detail, including flags that have no config-file or environment equivalent
(such as `-i/--input`, `--skip-from`, and `--append-to-skip-file`), see the
[CLI reference](../usage/cli-reference.md).
