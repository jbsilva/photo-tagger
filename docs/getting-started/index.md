---
icon: lucide/rocket
---

# Getting started

photo-tagger is a Python command-line tool that sends each photo to a local vision-language model
and writes Lightroom-compatible metadata: a title, a short description, and hierarchical keywords.
It works with RAW and standard formats, and by default it leaves your originals untouched by writing
an XMP sidecar next to each image.

This section walks you from a clean machine to a configured run. First make sure the prerequisites
below are in place, then follow the two child pages: install the tool, then point it at a model
server and tune the defaults that suit your library.

## Prerequisites

photo-tagger leans on a few external pieces that it does not bundle. Make sure each one is available
before you run your first batch.

| Tool                       | Why it is needed                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Python 3.14+               | The runtime. photo-tagger targets modern Python syntax.                                                             |
| ExifTool on `PATH`         | Reads existing metadata and writes the title, description, and keywords. `pyexiftool` drives the `exiftool` binary. |
| `libraw` (rawpy)           | Decodes RAW files (CR3, CR2, NEF, DNG). Linux wheels bundle it; on macOS run `brew install libraw`.                 |
| Ollama or LM Studio server | A running model server exposing a vision-language model (for example Qwen3-VL) over an OpenAI-compatible API.       |

!!! warning

    ExifTool must be discoverable on your `PATH`, and the model server must be running and serving the
    model you ask for. Both are checked at runtime, not at install time, so a missing piece surfaces
    only when you start a batch.

## Next steps

Continue with the two pages in this section:

- [Installation](installation.md): install photo-tagger with `uv`, `pipx`, or conda-forge, or set up
    a source checkout for development.
- [Configuration](configuration.md): choose a provider, set the model URL and API key, and manage
    defaults through environment variables or a TOML config file.

!!! tip

    Prefer a window to a terminal? photo-tagger has an optional [desktop GUI](../usage/gui.md)
    (`photo-tagger gui`) installed via the `gui` extra. It uses the same configuration described here.
