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
| `LLAMA_CPP_BASE_URL`  | `http://localhost:8080/v1`  | llama.cpp `llama-server` API base URL.                                             |
| `LLAMA_CPP_API_KEY`   | none                        | API key for llama.cpp; only needed when `llama-server` runs with `--api-key`.      |
| `OPENAI_BASE_URL`     | `https://api.openai.com/v1` | Base URL for the `openai` provider (the real OpenAI API or a drop-in gateway).     |
| `OPENAI_API_KEY`      | none                        | API key for the `openai` provider (required); also the fallback key for LM Studio. |
| `TEMPERATURE`         | `0.2`                       | Sampling temperature.                                                              |
| `MAX_TOKENS`          | `1200`                      | Maximum tokens to generate.                                                        |
| `TIMEOUT_SECONDS`     | `60.0`                      | Timeout of one model request, in seconds.                                          |
| `FREQUENCY_PENALTY`   | `0.5`                       | Penalty on repeated tokens.                                                        |
| `JPEG_DIMENSIONS`     | `1280`                      | Max dimension (px) of the JPEG sent to the model.                                  |
| `JPEG_QUALITY`        | `80`                        | JPEG quality (1-100) of the image sent to the model.                               |
| `RETRIES`             | `5`                         | Retries when model output fails schema validation.                                 |
| `DUTY_PERCENT`        | `100`                       | Default of `--duty` (1-100).                                                       |
| `BATTERY_FLOOR`       | `0`                         | Default of `--battery-floor` (0-99).                                               |
| `BATTERY_RESUME`      | `0`                         | Default of `--battery-resume` (0-100).                                             |
| `WORK_MINUTES`        | `10`                        | Default of `--work-minutes`.                                                       |
| `MAX_PAUSE_MINUTES`   | `15`                        | Default of `--max-pause-minutes`.                                                  |

A value that is not a number, or is outside its range, is ignored with a warning, and the built-in
default applies.

The standard OpenTelemetry variables (`OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME`) are honored
too, but only once `--otel` is on; see
[tracing model calls](../usage/cli-reference.md#tracing-model-calls).

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
language = "auto"        # GUI interface language: "auto" (OS default), "en", or "pt_BR".

[provider]
model_name = "qwen/qwen3-vl-30b"          # Vision-language model identifier.
provider_name = "lmstudio"                # Backend: "ollama", "lmstudio", "llamacpp", or "openai".
api_base_url = "http://localhost:1234/v1" # Provider API base URL.
# api_key = "..."                         # Prefer the API-key env vars instead.
retries = 5                               # Retries when output fails schema validation.

[inference]
output_language = "English"  # Language of generated titles, descriptions, and keywords.
temperature = 0.2        # Sampling temperature.
max_tokens = 1200        # Maximum tokens to generate.
timeout_seconds = 60.0   # Per-image inference timeout in seconds.
frequency_penalty = 0.5  # Penalty on repeated tokens; discourages output loops.
jpeg_dimensions = 1280   # Max dimension (px) of the JPEG sent to the model.
jpeg_quality = 80        # JPEG quality (1-100) of the image sent to the model.
repair_language = false  # Re-ask the model when its answer mixes writing systems.

[output]
preserve_keywords = true  # Merge with existing keywords (false replaces them).
preserve_title = false    # true keeps the title a photo already has.
preserve_description = false  # true keeps the description a photo already has.
write_title = true        # Generate and write a title.
write_description = true  # Generate and write a description.
write_keywords = true     # Write keywords (false leaves existing keywords untouched).
write_copyright = true    # Write the [copyright] block below (false skips it).
backup_xmp = true         # Keep ExifTool's *_original backup before writing.
sidecar_mode = "all"      # "none" embeds; "raw" is a sidecar for RAW only; "both" writes each.
dry_run = false           # Run the model and log results, but write nothing.
# max_keywords = 10       # Cap AI keywords kept per photo (omit to keep all).
# vocabulary = "~/keywords.txt"  # Restrict keywords to the terms in this file.
# vocabulary_strict = false      # Drop keywords the vocabulary does not cover.
# session_gap_minutes = 60       # Group photos into shoots and harmonize each shoot.

[copyright]
# artist = "Ada Lovelace"  # EXIF:Artist, XMP-dc:Creator, IPTC:By-line.
# copyright_notice = "Copyright (c) {year} {artist}, All rights reserved"
# rights = "© {year} {artist}, All rights reserved"  # XMP-dc:Rights only.
# marked = true            # XMP-xmpRights:Marked; false declares public domain.
# set_tags = ["XMP-xmpRights:WebStatement=https://example.com/terms"]

[log]
file_log_level = "DEBUG"     # DEBUG/INFO/WARNING/ERROR/CRITICAL/OFF.
console_log_level = "INFO"   # Same levels; OFF disables.
log_folder = "logs"          # Folder for timestamped log files.
otel = false                 # Export an OpenTelemetry trace of every model call.
# otel_endpoint = "http://localhost:4318/v1/traces"  # Where to send them. Required by otel.
otel_content = false         # Also export prompts and generated text. Photos never go either way.

[display]
progress_bar = true   # Live progress bar; auto-disabled when stderr is not a TTY.
json_output = false   # Emit one NDJSON line per processed photo to stdout.
notify = false        # Desktop notification when the batch ends.
notify_sound = false  # Play a sound with it (implies notify).

[filter]
skip_tagged = false   # Skip files that already have keywords, a title, or a description.
# newer_than = "2024-01-01"        # Drop files whose mtime is on/before this timestamp.
# older_than = "2024-01-01T14:30"  # Drop files whose mtime is on/after this timestamp.

[pacing]
duty = 100             # Percent of the time the run may work; it rests for the remainder.
battery_floor = 0      # Rest below this charge, while plugged in; 0 never watches the battery.
battery_resume = 0     # Charge that ends the rest, above the floor; 0 means floor plus 15.
work_minutes = 10.0    # Work this long between rests.
max_pause_minutes = 15.0  # Longest a single rest may last.

[artifacts]
# prompt_file = "prompt.txt"   # Replace the default user prompt with this file's contents.
# summary_file = "summary.json" # Write a JSON run summary on completion.
# csv_file = "report.csv"      # Write a per-photo CSV report (one row per photo) as the run streams.
# cache_file = "cache.sqlite"   # SQLite cache of model outputs; reruns skip unchanged calls.
# lock_file = "photo-tagger.lock" # Exclusive lock; refuse to start if another run holds it.
undo_log = true               # Record what the run writes so `photo-tagger undo` can revert it.

[gui]
save_after_generating = false # Desktop window only: write each run's photos when it finishes.
```

!!! note

    Unknown keys are ignored. This keeps older config files working when new options are added, so a
    stray or misspelled key never stops a run. Double-check spelling if a setting seems to have no
    effect.

For the full per-flag detail, including flags that have no config-file or environment equivalent
(such as `-i/--input`, `--skip-from`, and `--append-to-skip-file`), see the
[CLI reference](../usage/cli-reference.md).
