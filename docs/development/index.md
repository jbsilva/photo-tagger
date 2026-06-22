---
icon: lucide/code
---

# Development

This section covers working on photo-tagger itself: setting up an environment, running the test
suite, keeping the code clean, managing dependencies, and building the docs. The project uses
[uv](https://docs.astral.sh/uv/) to manage the virtual environment, and `uv.lock` is committed so
everyone builds against the same pins.

## Getting set up

There are two ways to get a working development environment. The dev container is the fastest path
if you already use VS Code or GitHub Codespaces; the local setup runs the same bootstrap script
directly on your machine.

=== "Dev container"

    Open the repository in VS Code with the Dev Containers extension, or launch a GitHub Codespace. The
    container is described by
    [`.devcontainer/devcontainer.json`](https://github.com/jbsilva/photo-tagger/blob/main/.devcontainer/devcontainer.json),
    which builds from the `mcr.microsoft.com/devcontainers/python:3.14-bookworm` image and runs
    [`scripts/dev-setup.sh`](https://github.com/jbsilva/photo-tagger/blob/main/scripts/dev-setup.sh) as
    its `postCreateCommand`.

    That script installs uv and ExifTool, syncs the `dev` and `test` dependency groups from `uv.lock`,
    and installs the git hooks. The container also puts `.venv/bin` on `PATH`, so you can run
    `photo-tagger`, `zuban`, and the other tools without an explicit `uv run` prefix.

=== "Local setup"

    Clone the repository, then run the same bootstrap script that the dev container uses:

    ```bash
    git clone https://github.com/jbsilva/photo-tagger.git
    cd photo-tagger
    bash scripts/dev-setup.sh
    ```

    The script is idempotent, so it is safe to re-run. If you prefer to do the steps by hand, install
    [ExifTool](https://exiftool.org/) so the `exiftool` binary is on `PATH`, then sync the dependency
    groups and install the hooks:

    ```bash
    uv sync --group dev --group test
    uv run prek install
    ```

!!! warning

    photo-tagger requires Python 3.14+ and ExifTool on `PATH`. RAW decoding also needs libraw; on macOS
    install it with `brew install libraw` (Linux wheels bundle it). See
    [Installation](../getting-started/installation.md) for the full list of requirements.

## Everyday commands

Once the environment is set up, these are the commands you will reach for most often:

```bash
uv run photo-tagger --help    # See the CLI options
uv run pytest                 # Run the test suite
prek run -a                   # Run all pre-commit hooks across the tree
```

## In this section

- [Testing](testing.md): run the suite, work with markers and coverage, and write new tests.
- [Code quality](code-quality.md): the ruff, zuban, pycroscope, and bandit checks and how to run
    them before committing.
- [Dependencies](dependencies.md): how uv and Renovate keep `pyproject.toml` and `uv.lock` in sync.
- [Documentation](documentation.md): preview and build this Zensical site.
