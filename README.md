# Photo Tagger

Photo Tagger is a command-line helper that asks a vision-language model to analyze your photos and
writes Lightroom-compatible metadata.

By default it keeps your originals untouched by creating XMP sidecars, but you can embed the updates
directly into each photo with `--embed-in-photo`.

## Highlights

- Works with RAW and standard image formats (CR3, CR2, NEF, JPG, PNG, and more)
- Generates a title, a concise description, and hierarchical keywords
- Merges with existing metadata unless you opt-in to overwrite
- Supports Ollama and LM Studio compatible OpenAI APIs
- Converts images to compact JPEG bytes to minimize token usage
- Generates detailed log files for easy debugging and auditing
- Highly configurable via CLI flags and environment variables

## Requirements

- Python 3.14+
- [ExifTool](https://exiftool.org/) available on `PATH`
- A running Ollama or LM Studio server exposing a vision-language model (for example Qwen-VL)
- `libraw` support for `rawpy` (install via Homebrew on macOS: `brew install libraw`)

## Installation

For end-users, the recommended installation method is via
[uv](https://docs.astral.sh/uv/guides/tools/):

```bash
uv tool install photo-tagger
```

For development (tests, linting):

```bash
uv sync --group dev --group test
```

## Configuration

Environment variables provide defaults so you can keep the CLI concise:

- `OLLAMA_BASE_URL` – override the Ollama HTTP endpoint (default `http://localhost:11434/v1`)
- `OLLAMA_API_KEY` – optional API key passed to Ollama requests
- `LM_STUDIO_BASE_URL` – override the LM Studio endpoint (default `http://localhost:1234/v1`)
- `LM_STUDIO_API_KEY` / `OPENAI_API_KEY` – API key for LM Studio’s OpenAI-compatible server
- `MODEL_NAME` – default model name (default `qwen3-vl:32b`)
- `JPEG_DIMENSIONS`, `JPEG_QUALITY`, `TEMPERATURE`, `MAX_TOKENS`, `RETRIES` – fine-tune runtime

Any CLI flag takes precedence over the environment.

## Usage

The CLI is exposed as `photo-tagger` once installed, or you can invoke it directly:

```bash
photo-tagger -i ./photos --ext cr3,jpg -r
```

Key options:

- `-i/--input PATH` – repeatable; mix files and directories
- `--ext` – comma-separated extension list used when scanning directories (default `cr3,jpg`)
- `-r/--recursive` – recurse into subdirectories while scanning inputs
- `-m/--model` – model identifier understood by your provider
- `--provider` – `ollama` or `lmstudio` (defaults to `lmstudio`)
- `--url` / `--api-key` – override provider endpoint and credentials
- `--overwrite-keywords` – replace instead of merge existing keyword metadata
- `--no-write-title` / `--no-write-description` – skip writing those fields
- `--no-backup-xmp` – avoid creating `*_original` snapshot before writing
- `--embed-in-photo` – write metadata directly into the image instead of creating an XMP sidecar
- `--jpeg-dimensions`, `--jpeg-quality`, `--temperature`, `--max-tokens`, `--retries` – control
  inference behavior

A successful run creates or updates an `.xmp` sidecar for every processed image (unless you embed
the metadata). Existing metadata is merged so Lightroom keeps hierarchical keywords such as
`Animal|Bird|Osprey` intact.

### Examples

Process a folder of RAW and JPEG files recursively:

```bash
photo-tagger -i ~/Pictures/Portfolio --ext cr3,jpg -r
```

Tag a few explicit files and overwrite existing keywords:

```bash
photo-tagger \
  -i IMG_0001.CR3 \
  -i IMG_0002.CR3 \
  --overwrite-keywords
```

Embed metadata directly into a set of JPEGs:

```bash
photo-tagger -i ./exports --ext jpg --embed-in-photo
```

Send requests to a remote Ollama host with a custom model:

```bash
photo-tagger -i ./shoot --provider ollama --model llava:34b --url http://ollama-box:11434/v1
```

## Logging

Logs are written to stderr and to a timestamped file (for example `20260101...-photo_tagger.log`).
Adjust levels with `--console-log-level` and `--file-log-level`, or disable either by setting the
value to `OFF`.

## Testing

Run the unit tests with:

```bash
pytest
```

If you plan to contribute, also run `ruff check` for linting before opening a PR.
