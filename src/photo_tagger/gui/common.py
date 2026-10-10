# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""
Constants the GUI modules share, and where their settings are saved.

The app's own folder, the item-data roles and tree columns the list and the grid agree on, the
labels more than one dialog uses, and the config file a setting is written into.
"""

from pathlib import Path

from PySide6.QtCore import Qt

from photo_tagger.config_file import find_config_file, user_config_path
from photo_tagger.i18n import gettext_noop


# The folder for everything the window writes for itself: the logs, the result cache, and the
# vocabulary file the builder offers to create.
APP_DIR = Path.home() / ".photo-tagger"
# A stable, cwd-independent place for the GUI's logs. The CLI defaults to ./logs, but a windowed
# app has no meaningful working directory (it may be launched from Finder with cwd "/"), so the
# logs live under the user's home where the "Open logs" button can always find them.
LOG_FOLDER = APP_DIR / "logs"
# The GUI cache lives next to the GUI logs unless the config names a cache_file. Sharing the
# CLI's default would be wrong: the CLI has no default cache, it only caches when asked.
DEFAULT_CACHE_FILE = APP_DIR / "cache.sqlite"
# Where the vocabulary builder offers to write, next to the logs and the cache. It is only the
# pre-filled suggestion; the dialog's Choose button puts the file wherever the user keeps theirs.
DEFAULT_VOCABULARY_FILE = APP_DIR / "vocabulary.txt"

DOCS_URL = "https://jbsilva.github.io/photo-tagger/"

PATH_ROLE = Qt.ItemDataRole.UserRole
IS_DIR_ROLE = Qt.ItemDataRole.UserRole + 1
STATUS_RANK_ROLE = Qt.ItemDataRole.UserRole + 2  # lifecycle rank for sorting the Status column
DIR_MARK = "dir"  # truthy sentinel stored on folder tree items; files leave the role unset

# Tree columns, in display order.
COL_NAME = 0
COL_TYPE = 1
COL_STATUS = 2
COL_TAGGED = 3

# Placeholder shown when a photo has no existing title/description/keywords. Module-level strings
# use gettext_noop (extraction marker); their use sites translate with _() at display time.
NONE_LABEL = gettext_noop("(none)")

# Title of the warning box shown when persisting a setting to the config file fails.
CONFIG_SAVE_ERROR_TITLE = gettext_noop("Could not save the config file")

# Shown when a save is asked for while the save options would write nothing at all.
PICK_A_FIELD = gettext_noop("Pick at least one field to write (Title, Description, or Keywords).")

# Button labels every dialog uses. Named once so the catalogs hold one entry each and the dialogs
# cannot drift apart; the use sites translate them with _() like the other module-level strings.
CHOOSE_LABEL = gettext_noop("Choose...")
CLOSE_LABEL = gettext_noop("Close")

# The note under the settings a dialog keeps for the session only.
SESSION_NOTE = gettext_noop(
    "Kept for this session. Settings > Save Settings as Defaults writes them to the config file, "
    "which CLI runs read too.",
)


def config_target() -> Path:
    """Return the config file a setting is saved into: the one in effect, or the user default."""
    return find_config_file() or user_config_path()
