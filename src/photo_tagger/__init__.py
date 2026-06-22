"""
``photo-tagger``: describe photos and add keywords with a vision-language model.

The version is read from the installed distribution metadata so there is a single source of truth
(``[project].version`` in ``pyproject.toml``). No module hardcodes the version string.

The package import stays deliberately light: the heavy collaborators (rawpy, pydantic-ai, exiftool)
are pulled in lazily by the submodules that need them, so ``import photo_tagger`` and ``photo-tagger
--version`` start fast.
"""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("photo-tagger")
except PackageNotFoundError:  # pragma: no cover - only when run from a non-installed tree
    # Source checkout that was never installed (some bare CI shells). The real
    # version is unknowable here, so flag it rather than guess.
    __version__ = "0.0.0+unknown"


__all__ = ["__version__"]
