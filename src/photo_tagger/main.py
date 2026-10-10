#!/usr/bin/env python3
"""
Photo Tagger: CLI app to describe photos and add keywords using AI.

Optionally non-destructive: create/update XMP sidecar files with Lightroom-compatible tags.
Alternatively, pass --embed-in-photo to write metadata directly into the original file.
Unfortunately, Lightroom uses XMP sidecar files only for proprietary raw formats (e.g., CR3, NEF).
For JPEG, DNG and other formats, you'll often prefer embedding the metadata directly into the file.
A folder holding both is what --sidecar-mode raw is for: sidecars for the RAW files, embedded
metadata for the rest, in one pass. Mind that DNG counts as RAW there; add
--sidecar-override dng=photo to write into each DNG, as Lightroom Classic does.
This can be done with ExifTool manually as well:
    exiftool -tagsFromFile image.xmp -all:all image.jpg

Requirements:
 - Exiftool installed and available in PATH.
 - Ollama or LM Studio server running and containing a vision-language model.
"""

import importlib
import sys
from typing import Annotated

from cyclopts import App, Parameter
from loguru import logger

from photo_tagger import __version__, telemetry
from photo_tagger.cli_fields import set_fields, show
from photo_tagger.cli_options import (
    CONFIG_SOURCE,
    DEFAULT_PROVIDER_OPTIONS,
)
from photo_tagger.cli_tag import tag
from photo_tagger.cli_telemetry import (
    INTERRUPTED_EXIT,
    CommandReporter,
    exit_status,
)
from photo_tagger.cli_undo import undo
from photo_tagger.cli_vocabulary import vocabulary
from photo_tagger.cli_watch import watch
from photo_tagger.config_file import export_exiftool_path
from photo_tagger.diagnostics import render_report, run_checks

# Runtime import (not type-only): cyclopts evaluates the Annotated[ProviderName, ...] field
# on the doctor command to validate the --provider choices, so it must exist at definition time.
from photo_tagger.providers import ProviderName  # noqa: TC001
from photo_tagger.telemetry_facts import joined


app = App(name="photo-tagger", version=__version__, config=CONFIG_SOURCE)


def doctor(
    *,
    provider: Annotated[
        ProviderName,
        Parameter(name=("--provider",), help="Backend provider to check"),
    ] = DEFAULT_PROVIDER_OPTIONS.provider_name,
    model: Annotated[
        str,
        Parameter(name=("--model", "-m"), help="Model name expected to be served"),
    ] = DEFAULT_PROVIDER_OPTIONS.model_name,
    url: Annotated[
        str | None,
        Parameter(name=("--url", "-u"), help="Provider API base URL"),
    ] = DEFAULT_PROVIDER_OPTIONS.api_base_url,
    api_key: Annotated[
        str | None,
        Parameter(name=("--api-key", "-k"), help="Provider API key (prefer env vars)"),
    ] = DEFAULT_PROVIDER_OPTIONS.api_key,
) -> None:
    """
    Check that ExifTool and the model provider are reachable, then exit.

    Prints a short checklist and exits 0 when everything is in order, 1 if any check fails. Run this
    first when a tagging run cannot start: it isolates a missing ExifTool, an unreachable provider,
    or a model name typo from the rest of the pipeline. Honors the same config file and env vars as
    ``tag``.
    """
    # Resolve the exiftool path (and its own warning, e.g. a CWD config redirecting the binary)
    # before silencing loguru, or that warning never reaches the user in the one command meant
    # to surface exactly this kind of misconfiguration.
    export_exiftool_path()
    # Silence loguru so only the checklist reaches the terminal; failures are
    # captured in the report itself, not the logs.
    logger.remove()
    results = run_checks(provider, model, api_base_url=url, api_key=api_key)
    telemetry.note(
        checks_failed=sum(not result.ok for result in results),
        problems=joined(result.problem for result in results),
    )
    if not render_report(results):
        raise SystemExit(1)


def gui() -> None:
    """
    Launch the desktop GUI (requires the optional ``[gui]`` extra).

    The GUI is a review-before-write frontend over the same pipeline as the ``tag`` command: add
    photos or folders to a checkable list, generate proposals with the model, review and edit each
    photo's title, description, and keywords side by side with the existing metadata, then save.
    Install it with ``pip install 'photo-tagger[gui]'``.

    PySide6 is imported lazily here so the base CLI never depends on Qt; a missing dependency is
    reported with an install hint rather than a traceback.
    """
    try:
        gui_module = importlib.import_module("photo_tagger.gui")
    except ImportError as exc:
        # Only a genuinely missing Qt module earns the install hint. Anything else (a broken
        # shiboken build, an import error inside our own gui code) must surface as itself, or the
        # hint sends the user reinstalling an extra that is not the problem.
        # Without the parentheses the "or" takes the result of the split, and the module name
        # arrives whole. SonarQube reads them as redundant (S1110) and is wrong.
        missing = (exc.name or "").split(".")[0]  # NOSONAR S1110
        if missing not in ("PySide6", "shiboken6"):
            raise
        sys.stderr.write(
            "The desktop GUI needs PySide6, which is not installed.\n"
            "Install the optional extra with:\n"
            "    pip install 'photo-tagger[gui]'\n",
        )
        raise SystemExit(1) from exc
    raise SystemExit(gui_module.launch())


# The commands live in modules of their own; registering them here keeps one list of what the CLI
# offers. The help lists them alphabetically, so the order here is free.
app.command(doctor)
app.command(gui)
app.command(show)
app.command(set_fields, name="set")
app.command(vocabulary)
app.command(undo)
app.default(tag)
app.command(watch)


def main(argv: list[str] | None = None) -> None:
    """
    Console entry point: run the CLI, reporting an unhandled crash before re-raising.

    Expected exits (SystemExit from clean error handling, Ctrl-C) pass through untouched; anything
    else is a genuine crash, so an anonymous beacon (exception type and stack trace as code
    locations, never the message) is sent before the traceback surfaces as usual. An invocation that
    sent no beacon of its own gets a short command beacon on the way out.
    """
    tokens = sys.argv[1:] if argv is None else argv
    reporter = CommandReporter(
        tokens,
        commands=frozenset(name for name in app if not name.startswith("-")),
        source=CONFIG_SOURCE,
    )
    exit_code = 0
    error: BaseException | None = None
    try:
        app(tokens)
    except SystemExit as exc:
        exit_code = exit_status(exc)
        error = exc.__cause__ or exc.__context__
        raise
    except KeyboardInterrupt as exc:
        exit_code, error = INTERRUPTED_EXIT, exc
        raise
    except Exception as exc:
        reporter.crashed(exc)
        raise
    finally:
        reporter.finished(exit_code, error)


if __name__ == "__main__":  # pragma: no cover - module entry point, not exercised by tests
    main()
