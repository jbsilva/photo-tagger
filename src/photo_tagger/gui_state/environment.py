"""The PATH a GUI app started from the desktop needs to find ExifTool."""

import os
import subprocess  # nosec B404 - only used to read the user's own login-shell PATH (see below)
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


# How long to wait for the login shell to report its PATH before giving up.
_SHELL_PATH_TIMEOUT = 5.0


def login_shell_path() -> list[str]:
    """
    Return the PATH entries a login shell would set, or ``[]`` if they can't be determined.

    A Finder/Dock launch inherits a minimal PATH, so exiftool installed by any package manager
    (Homebrew, Nix, MacPorts, ...) is missing - those locations are only added by shell startup
    files. Asking the user's own login shell for its PATH recovers exiftool however it was
    installed, instead of hard-coding each manager's bin dir. Best-effort: a missing ``$SHELL``, a
    non-zero exit, or a timeout all yield ``[]``.
    """
    shell = os.environ.get("SHELL")
    if not shell:
        return []
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, trusted $SHELL
            [shell, "-l", "-c", 'printf %s "$PATH"'],
            capture_output=True,
            text=True,
            timeout=_SHELL_PATH_TIMEOUT,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return []
    return [entry for entry in result.stdout.strip().split(os.pathsep) if entry]


def ensure_path_dirs(current_path: str, dirs: Iterable[str]) -> str:
    """
    Prepend any of *dirs* not already in *current_path* to it, order preserved.

    A GUI launched from Finder or the Dock inherits a minimal ``PATH``, so Homebrew's bin
    directories (where exiftool usually lives) are missing and exiftool discovery fails even though
    it works fine from a shell. Prepending them lets a double-clicked app find exiftool the same
    way. Directories that do not exist are harmless on ``PATH``, so no filesystem check is needed.
    """
    entries = current_path.split(os.pathsep) if current_path else []
    additions = [d for d in dirs if d not in entries]
    if not additions:
        return current_path
    return os.pathsep.join([*additions, *entries])
