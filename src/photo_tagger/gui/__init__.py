# mypy: ignore-errors
# The lint/pycroscope environment does not install the [gui] extra, so pycroscope cannot import
# this Qt package (PySide6 is absent) and would also misread shiboken's runtime-generated
# attributes. Every module here carries the same two headers (this line pair and the one below),
# which exclude it from pycroscope and zuban the same way the package is excluded from coverage;
# the Qt-free logic that is worth analyzing lives in gui_state/ and gui_usage.py.
# static analysis: ignore
"""
PySide6 desktop frontend for photo-tagger.

A review-before-write workflow over the same pipeline the CLI uses:

1. Drag photos or folders onto the window (or use *Add files/folder*). They appear in a
   checkable, nested tree on the left; uncheck or remove anything you do not want.
2. *Generate* runs the vision model on the checked photos on a background thread,
   streaming each proposal back through Qt signals (no pipeline code is Qt-aware).
3. Click a photo to see its preview, its existing title/description/keywords, and the
   proposed values in editable fields, then *Save* writes them with ExifTool.

Requires the optional ``[gui]`` extra (``pip install 'photo-tagger[gui]'``). The ``photo-tagger
gui`` command imports this package lazily, so the base CLI never depends on Qt. The Qt-free logic
lives in :mod:`photo_tagger.gui_state` and :mod:`photo_tagger.gui_usage`; this package is the
widget and event-loop shell and is excluded from coverage and the static analyzers.

:mod:`~photo_tagger.gui.app` starts the app and :mod:`~photo_tagger.gui.main_window` holds the
window. :mod:`~photo_tagger.gui.workers` holds the background jobs, and the other modules hold the
pieces the window is built from.
"""

from photo_tagger.gui.app import launch
from photo_tagger.gui.main_window import MainWindow


__all__ = ["MainWindow", "launch"]
