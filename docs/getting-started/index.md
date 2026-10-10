---
icon: lucide/rocket
---

# Getting started

photo-tagger is a Python command-line tool that sends each photo to a vision-language model and
writes Lightroom-compatible metadata: a title, a short description, and hierarchical keywords. The
model can run on your own machine or server, or come from any OpenAI-compatible service, free or
paid. It works with RAW and standard formats. By default, it leaves your originals untouched: it
writes an XMP sidecar next to each image.

This section takes you from a clean machine to a configured run. First, make sure that the
prerequisites below are in place. Then follow the two child pages in order. The first page tells you
how to install the tool. The second page tells you how to point it at a model server and tune the
defaults that suit your library.

## Prerequisites

photo-tagger uses a few external pieces that it does not bundle. Make sure that each one is
available before you run your first batch.

| Tool               | Why it is needed                                                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python 3.14+       | The runtime. photo-tagger targets modern Python syntax.                                                                                                         |
| ExifTool on `PATH` | Reads existing metadata and writes the title, description, and keywords. `pyexiftool` drives the `exiftool` binary.                                             |
| `libraw` (rawpy)   | Decodes RAW files (CR3, CR2, NEF, DNG). Linux wheels bundle it. On macOS, run `brew install libraw`.                                                            |
| Model server       | A vision-language model (for example Qwen3-VL) behind an OpenAI-compatible API: Ollama, LM Studio, or llama.cpp on your machine or server, or a hosted service. |

!!! warning

    ExifTool must be discoverable on your `PATH`. The model server must be running and must serve the
    model that you ask for. photo-tagger looks for both at runtime, not at install time. As a result, a
    missing piece appears only at the start of a batch.

## Next steps

Continue with the two pages in this section:

- [Installation](installation.md): install photo-tagger with `uv`, `pipx`, or conda-forge, or set up
    a source checkout for development.
- [Configuration](configuration.md): choose a provider, set the model URL and API key, and manage
    defaults through environment variables or a TOML configuration file.

!!! tip

    If you prefer a window to a terminal, photo-tagger has an optional [desktop GUI](../usage/gui.md)
    (`photo-tagger gui`). You install it with the `gui` extra. It uses the same configuration that this
    section describes.
