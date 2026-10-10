# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The window's stylesheet and the bundled icons it and the dialogs draw with."""

from pathlib import Path

from PySide6.QtGui import QIcon


# The bundled SVGs live beside the package, in photo_tagger/resources, not inside it.
RESOURCES = Path(__file__).parent.parent / "resources"


# Theme-agnostic polish: only spacing/rounding plus the brand accent on primary actions and
# the preview area. Colors for text and input backgrounds are left to the OS palette, so the
# window stays readable in both light and dark mode (hardcoding a light background would put
# the palette's light text on white in dark mode).
_STYLESHEET = """
QWidget { font-size: 13px; }
QPushButton {
    padding: 6px 12px; border-radius: 6px;
    border: 1px solid rgba(130, 130, 140, 60%);
    background: rgba(130, 130, 140, 14%);
}
QPushButton:hover { background: rgba(130, 130, 140, 26%); }
QPushButton:pressed { background: rgba(130, 130, 140, 36%); }
QPushButton:disabled { color: rgba(130, 130, 140, 70%); border-color: rgba(130, 130, 140, 25%); }
QPushButton#primary {
    background: #6366f1; color: white; border: 1px solid #6366f1; font-weight: 600;
}
QPushButton#primary:hover { background: #4f46e5; border-color: #4f46e5; }
QPushButton#primary:disabled {
    background: #9aa0e8; border-color: #9aa0e8; color: #eaeaff;
}
QLineEdit, QPlainTextEdit, QComboBox { padding: 4px 6px; border-radius: 5px; }
/* Scroll-area edits keep their native square frame unless given an explicit border, which is
   why only the QLineEdit fields looked rounded. */
QPlainTextEdit, QTextEdit {
    border: 1px solid rgba(130, 130, 140, 35%); border-radius: 5px; padding: 4px 6px;
}
QPlainTextEdit#tree { font-family: "SF Mono", Menlo, Consolas, monospace; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 22px; border: none; background: transparent;
}
QComboBox::down-arrow { image: url("@CHEVRON@"); width: 10px; height: 6px; }
QPushButton::menu-indicator {
    image: url("@CHEVRON@"); width: 10px; height: 6px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 10px;
}
QPushButton#menubutton { padding-right: 28px; }
QTreeWidget::item { padding: 2px; }
QToolButton { border: none; background: transparent; padding: 4px; font-weight: 600; }
QToolButton:hover { color: #6366f1; }
QToolButton#sortdir {
    border: 1px solid rgba(130, 130, 140, 60%); border-radius: 6px; padding: 4px 10px;
}
QToolButton#sortdir:hover { background: rgba(130, 130, 140, 26%); }
QToolButton#sortdir:checked { background: rgba(99, 102, 241, 22%); border-color: #6366f1; }
/* The view-mode button: a bare glyph, no frame, so the choice stays quiet next to the labelled
   buttons it shares the row with. */
QToolButton#viewmode { font-size: 15px; padding: 4px 8px; color: rgba(130, 130, 140, 90%); }
QToolButton#viewmode:hover { color: #6366f1; }
QToolButton#navbtn {
    border: 1px solid rgba(130, 130, 140, 60%); border-radius: 6px; padding: 2px 10px;
}
QToolButton#navbtn:hover { background: rgba(130, 130, 140, 26%); }
QToolButton#navbtn:disabled {
    color: rgba(130, 130, 140, 45%); border-color: rgba(130, 130, 140, 25%);
}
QToolButton#add, QToolButton#split {
    padding: 6px 26px 6px 12px; border-radius: 6px; font-weight: 400;
    border: 1px solid rgba(130, 130, 140, 60%);
    background: rgba(130, 130, 140, 14%);
}
QToolButton#add:hover, QToolButton#split:hover { background: rgba(130, 130, 140, 26%); }
QToolButton#split:disabled { color: rgba(130, 130, 140, 70%); }
QToolButton#add::menu-button, QToolButton#split::menu-button {
    border: none; width: 22px;
    border-left: 1px solid rgba(130, 130, 140, 45%);
    margin-top: 6px; margin-bottom: 6px;
}
QToolButton#add::menu-arrow, QToolButton#split::menu-arrow {
    image: url("@CHEVRON@"); width: 10px; height: 6px;
}
QToolButton#primarysplit {
    padding: 6px 26px 6px 12px; border-radius: 6px; font-weight: 600;
    background: #6366f1; color: white; border: 1px solid #6366f1;
}
QToolButton#primarysplit:hover { background: #4f46e5; border-color: #4f46e5; }
QToolButton#primarysplit:disabled {
    background: #9aa0e8; border-color: #9aa0e8; color: #eaeaff;
}
QToolButton#primarysplit::menu-button {
    border: none; width: 22px;
    border-left: 1px solid rgba(255, 255, 255, 40%);
    margin-top: 6px; margin-bottom: 6px;
}
QToolButton#primarysplit::menu-arrow {
    image: url("@CHEVRONLIGHT@"); width: 10px; height: 6px;
}
QProgressBar {
    border: 1px solid rgba(130, 130, 140, 60%); border-radius: 5px; text-align: center;
}
QProgressBar::chunk { background: #6366f1; border-radius: 4px; }
QLabel#preview { background: #1f1f24; border-radius: 8px; color: #9a9aa5; }
/* The full-size viewer: a dark, frameless canvas, so nothing competes with the photo. */
QScrollArea#viewer { background: #1f1f24; border: none; }
QScrollArea#viewer > QWidget > QWidget { background: #1f1f24; }
QLabel#hint, QLabel#status, QLabel#timing { color: #8a8a8a; }
QLabel#crumb { color: #8a8a8a; font-weight: 600; }
QLabel#empty { color: #8a8a8a; font-size: 15px; }
QLabel#section { font-weight: 600; }
QLabel#error {
    background: rgba(248, 81, 73, 18%); color: #f85149;
    border: 1px solid rgba(248, 81, 73, 45%); border-radius: 6px; padding: 8px;
}
/* Amber, not red: the text is usable, it just deserves a read before it is saved. */
QLabel#warning { color: #d29922; }
"""


def stylesheet() -> str:
    """Resolve the stylesheet's image placeholders to the bundled resource files."""
    chevron = (RESOURCES / "chevron-down.svg").as_posix()
    chevron_light = (RESOURCES / "chevron-down-light.svg").as_posix()
    return _STYLESHEET.replace("@CHEVRONLIGHT@", chevron_light).replace("@CHEVRON@", chevron)


def svg_icon(name: str) -> QIcon:
    """
    Load one of the bundled SVG icons, or an empty icon when it is missing.

    Drawn rather than typed: a text glyph renders at whatever weight the platform font feels like
    and cannot be aligned with the rest of a toolbar. The icons share the muted grey the chevron and
    the hint labels use, which reads on both a light and a dark palette.
    """
    path = RESOURCES / name
    return QIcon(str(path)) if path.exists() else QIcon()


def app_icon() -> QIcon:
    """Load the bundled app icon, or an empty icon if it is not present."""
    return svg_icon("icon.svg")
