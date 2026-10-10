"""
Find a helper program the way a double-clicked app should: never in the current directory.

On Windows, a bare program name is looked up in the current directory before PATH, both by the OS
and by ``shutil.which``. photo-tagger runs its helpers (ExifTool, the hardware and battery probes,
the notifier) from wherever it was started, which is often a folder of photos, so a ``exiftool.exe``
or ``powershell.exe`` sitting there would run instead of the real one. Every helper is resolved
here, to an absolute path, from the absolute PATH entries only.
"""

import os
import shutil
import sys
from pathlib import Path


if sys.platform == "win32":
    # The switch Windows itself reads (NeedCurrentDirectoryForExePath) to leave the current
    # directory out of a program search. shutil.which honors it; so do cmd.exe and child processes.
    os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")


def find_program(name: str) -> str | None:
    """
    Return the absolute path of *name* on PATH, or None; a name with a folder is checked as given.

    PATH entries that are not absolute (``.``, an empty entry) stand for the current directory too,
    so they are left out.
    """
    entries = os.environ.get("PATH", "").split(os.pathsep)
    path = os.pathsep.join(entry for entry in entries if entry and Path(entry).is_absolute())
    return shutil.which(name, path=path)
