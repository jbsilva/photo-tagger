---
icon: lucide/download
---

# Installation

photo-tagger is a Python 3.14+ command-line tool. Install it once with your tool manager of choice,
make sure ExifTool is on your `PATH`, and you are ready to start tagging photos.

## End-user install

For most people the simplest option is [uv](https://docs.astral.sh/uv/). It installs photo-tagger
into an isolated environment and puts the `photo-tagger` command on your `PATH`.

```bash
uv tool install photo-tagger
```

Upgrade to the latest release the same way:

```bash
uv tool upgrade photo-tagger
```

### pipx

If you already use [pipx](https://pipx.pypa.io/), install from PyPI with:

```bash
pipx install photo-tagger
```

### conda-forge

The package is published on conda-forge, so it also installs with conda (or with pixi / mamba):

```bash
conda install -c conda-forge photo-tagger
```

## From source

To work on photo-tagger itself, clone the repository and let uv create the development environment.
The `dev` and `test` dependency groups pull in the linters, type checker, and test runner.

```bash
git clone https://github.com/jbsilva/photo-tagger.git
cd photo-tagger
uv sync --group dev --group test
```

See the [Development](../development/index.md) section for the full workflow, including the local
checks to run before committing.

!!! tip

    The repository ships a dev container (`.devcontainer/devcontainer.json`) for a zero-setup
    environment: open the project in a container and it installs uv, ExifTool, the dependency groups,
    and the git hooks for you. See [Development](../development/index.md) for details.

## System requirements

photo-tagger needs Python 3.14 or newer plus two external libraries that are not Python packages.

[ExifTool](https://exiftool.org/) does the actual metadata reading and writing; photo-tagger drives
the `exiftool` binary through pyexiftool. Install it with your system package manager:

=== "Debian / Ubuntu"

    ```bash
    apt install libimage-exiftool-perl
    ```

=== "macOS"

    ```bash
    brew install exiftool
    ```

=== "Other"

    Download a build for your platform from [exiftool.org](https://exiftool.org/).

On macOS, RAW decoding through rawpy needs libraw as well. Install it with Homebrew:

```bash
brew install libraw
```

On Linux the rawpy wheels bundle libraw, so no extra step is needed there.

!!! warning

    The `exiftool` binary must be on your `PATH`. photo-tagger shells out to it for every photo, so if
    it is not found the run will fail. After installing, confirm it is visible with `exiftool -ver` (see
    [Verify](#verify) below).

## Verify

Check that the command and its ExifTool dependency are both reachable:

```bash
photo-tagger --help
exiftool -ver
```

The first prints the CLI options; the second prints the installed ExifTool version. Once both work,
head to [Configuration](configuration.md) to point photo-tagger at your model server.
