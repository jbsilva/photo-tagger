---
icon: lucide/code
---

# Development

This section is about work on photo-tagger itself. It tells you how to prepare an environment, run
the test suite, keep the code clean, manage dependencies, and build the docs. The project uses
[uv](https://docs.astral.sh/uv/) to manage the virtual environment. The project commits `uv.lock` to
the repository, so everyone builds against the same pins.

## Getting set up

You can get a working development environment in two ways. If you already use VS Code or GitHub
Codespaces, the dev container is the fastest path. The local setup runs the same bootstrap script
directly on your machine.

=== "Dev container"

    Open the repository in VS Code with the Dev Containers extension, or start a GitHub Codespace.
    [`.devcontainer/devcontainer.json`](https://github.com/jbsilva/photo-tagger/blob/main/.devcontainer/devcontainer.json)
    describes the container. This file builds the container from the
    `mcr.microsoft.com/devcontainers/python:3.14-bookworm` image and runs
    [`scripts/dev-setup.sh`](https://github.com/jbsilva/photo-tagger/blob/main/scripts/dev-setup.sh) as
    its `postCreateCommand`.

    That script installs uv and ExifTool, syncs the `dev` and `test` dependency groups from `uv.lock`,
    and installs the git hooks. The container also puts `.venv/bin` on `PATH`, so you can run
    `photo-tagger`, `zuban`, and the other tools without an explicit `uv run` prefix.

=== "Local setup"

    Clone the repository. Then run the same bootstrap script that the dev container uses:

    ```bash
    git clone https://github.com/jbsilva/photo-tagger.git
    cd photo-tagger
    bash scripts/dev-setup.sh
    ```

    The script is idempotent, so it is safe to run it again. If you prefer to do the steps by hand,
    install [ExifTool](https://exiftool.org/) so that the `exiftool` binary is on `PATH`. Then run these
    commands to sync the dependency groups and install the hooks:

    ```bash
    uv sync --group dev --group test
    uv run prek install
    ```

!!! warning

    photo-tagger requires Python 3.14+ and ExifTool on `PATH`. RAW decoding also needs libraw. Linux
    wheels bundle it. On macOS, install it with `brew install libraw`. See
    [Installation](../getting-started/installation.md) for the full list of requirements.

## Everyday commands

The dev commands are [Poe the Poet](https://poethepoet.natn.io) tasks. `pyproject.toml` defines them
under `[tool.poe]`. Each task runs through `uv run` and installs the dependency group that it needs.
These are the commands that you will use most often:

```bash
uv run photo-tagger --help    # See the CLI options
uv run poe                    # List the dev tasks
uv run poe test               # Run the test suite (extra arguments go to pytest)
uv run poe check              # Lint, format, type check, and run all pre-commit hooks
```

!!! tip

    Install Poe once with `uv tool install poethepoet`. Then you can type `poe test` without the
    `uv run` prefix.

## In this section

- [Testing](testing.md): run the suite, work with markers and coverage, and write new tests.
- [Code quality](code-quality.md): the ruff, zuban, pycroscope, and bandit checks, and how to run
    them before you commit.
- [Dependencies](dependencies.md): how uv and Renovate keep `pyproject.toml` and `uv.lock` in sync.
- [Documentation](documentation.md): preview and build this Zensical site.
