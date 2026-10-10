"""Short texts the window shows everywhere: tooltips, durations, and the warning mark."""

import textwrap

from photo_tagger.i18n import _


# Where tooltips are broken into lines. Qt only word-wraps a tooltip it takes for rich text, so a
# long plain-text one is drawn as a single line, often wider than the window. Hard-wrapping keeps
# them readable blocks. 72 columns is narrow enough to scan and wide enough to avoid ragged text.
TOOLTIP_WIDTH = 72


def wrap_tooltip(text: str, width: int = TOOLTIP_WIDTH) -> str:
    """
    Break *text* into lines of at most *width* characters for display in a tooltip.

    Line breaks already in the text are kept (each line is wrapped on its own), so a tooltip written
    as two short paragraphs stays two paragraphs, and re-wrapping an already-wrapped tooltip changes
    nothing.
    """
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else line for line in text.splitlines()
    )


def tooltip(message: str, /, **values: object) -> str:
    """
    Translate a tooltip *message*, fill in its ``{placeholders}``, and wrap it for display.

    The wrapping happens after translation on purpose: a translated tooltip is often longer than the
    English one, so hard-coding the breaks in the source strings would leave the catalogs ragged.
    Babel extracts from this function too (see ``scripts/extract_translations.py``), so tooltips
    stay translatable without a separate ``_()`` call.
    """
    text = _(message)
    return wrap_tooltip(text.format(**values) if values else text)


# Appended to a flagged photo's Status cell. A glyph rather than a word: the column is narrow, it
# has to survive translation, and it reads at a glance down a list of hundreds of rows.
MIXED_SCRIPT_MARK = "⚠"  # warning sign


def format_duration(seconds: float) -> str:
    """
    Render a duration as ``m:ss``, growing to ``h:mm:ss`` once it passes an hour.

    Anything below zero clamps to ``0:00``: a long batch is timed against a monotonic clock, but a
    caller doing its own arithmetic should never be able to print a negative countdown.
    """
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
