"""
Environment diagnostics for the ``photo-tagger doctor`` command.

Runs a short checklist (ExifTool on PATH, provider reachable, requested model served) and reports
each result. The checks are plain functions that return structured :class:`CheckResult` values
instead of printing, so they are easy to unit-test; only :func:`render_report` touches the terminal.
"""

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from photo_tagger import __version__
from photo_tagger.errors import ProviderError
from photo_tagger.providers import get_backend


if TYPE_CHECKING:
    from photo_tagger.providers import ProviderName


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one diagnostic check."""

    name: str
    ok: bool
    detail: str


def check_exiftool() -> CheckResult:
    """Verify the ExifTool binary is on PATH (pyexiftool shells out to it)."""
    path = shutil.which("exiftool")
    if path is None:
        return CheckResult(
            "ExifTool",
            ok=False,
            detail="not found on PATH; install it (https://exiftool.org) and retry",
        )
    return CheckResult("ExifTool", ok=True, detail=path)


def check_provider(
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
) -> CheckResult:
    """
    Verify the provider is reachable and serves *model_name*.

    Returns a single result describing the first problem found (no key, provider unreachable, or
    model absent) or success with the resolved endpoint.
    """
    backend = get_backend(provider_name)
    base_url = api_base_url or backend.default_base_url
    resolved_key = backend.resolve_api_key(api_key)

    label = f"Model '{model_name}' on {provider_name}"
    if backend.requires_api_key and not resolved_key:
        return CheckResult(
            label,
            ok=False,
            detail="provider requires an API key; set OPENAI_API_KEY or pass --api-key",
        )
    try:
        models = backend.list_models(base_url, resolved_key)
    except ProviderError as exc:
        return CheckResult(label, ok=False, detail=f"{base_url} unreachable: {exc}")
    if model_name not in models:
        return CheckResult(label, ok=False, detail=f"not served at {base_url}")
    return CheckResult(label, ok=True, detail=f"available at {base_url}")


def run_checks(
    provider_name: ProviderName,
    model_name: str,
    *,
    api_base_url: str | None,
    api_key: str | None,
) -> list[CheckResult]:
    """Run every diagnostic and return the results in display order."""
    return [
        check_exiftool(),
        check_provider(
            provider_name,
            model_name,
            api_base_url=api_base_url,
            api_key=api_key,
        ),
    ]


def render_report(results: list[CheckResult], *, console: Console | None = None) -> bool:
    """
    Print *results* as a checklist and return True when every check passed.

    Output goes to stdout by default; tests pass a ``Console(file=...)`` to capture it.
    """
    console = console or Console()
    console.print(f"photo-tagger {__version__} environment check\n")
    for result in results:
        mark = "[green]OK  [/green]" if result.ok else "[red]FAIL[/red]"
        console.print(f"  {mark}  {result.name}: {result.detail}")
    all_ok = all(result.ok for result in results)
    summary = "[green]All checks passed.[/green]" if all_ok else "[red]Some checks failed.[/red]"
    console.print(f"\n{summary}")
    return all_ok
