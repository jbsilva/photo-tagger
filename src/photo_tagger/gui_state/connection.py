"""The connection header: the provider and model, and the Test Connection report."""

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING

from photo_tagger.i18n import gettext_noop


if TYPE_CHECKING:
    from collections.abc import Iterable

    from photo_tagger.diagnostics import CheckResult
    from photo_tagger.providers import ProviderName


# Pre-filled choices for the Metadata Language menu. The values are English names on purpose:
# they are spliced into the (English) system prompt as-is; the GUI translates them for display
# only. The menu's Other... entry accepts any language the model understands, so this list does
# not limit the choice. The first entry must stay DEFAULT_OUTPUT_LANGUAGE (a test asserts it).
OUTPUT_LANGUAGE_SUGGESTIONS = (
    gettext_noop("English"),
    gettext_noop("Brazilian Portuguese"),
    gettext_noop("Dutch"),
    gettext_noop("French"),
    gettext_noop("German"),
    gettext_noop("Italian"),
    gettext_noop("Japanese"),
    gettext_noop("Korean"),
    gettext_noop("Portuguese"),
    gettext_noop("Russian"),
    gettext_noop("Simplified Chinese"),
    gettext_noop("Spanish"),
)


# Substrings that hint a model is vision-capable, used to surface likely picks first.
_VISION_HINTS = (
    "vl",
    "vision",
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "cogvlm",
    "internvl",
    "pixtral",
    "gemma3",
    "smolvlm",
    "-v-",
)


@dataclass(frozen=True, slots=True)
class Connection:
    """
    Where the model a job talks to lives: the backend, the model id, the server, and the key.

    Read off the window's header and Connection dialog when a job starts, so the job never reads a
    widget. A blank server or key is None: the provider's default URL, and its environment variable.
    """

    provider: ProviderName
    model: str
    api_base_url: str | None = None
    api_key: str | None = None


def rank_vision_models(model_ids: Iterable[str]) -> list[str]:
    """
    Order model ids with likely vision-capable ones first, keeping all of them.

    The provider listings do not reliably flag modality, so this is a name heuristic (``vl``,
    ``vision``, ``llava``, ...) used only to surface probable picks; nothing is hidden, so a model
    the heuristic misses is still selectable.
    """
    model_ids = list(model_ids)
    likely = [m for m in model_ids if any(hint in m.lower() for hint in _VISION_HINTS)]
    likely_set = set(likely)
    others = [m for m in model_ids if m not in likely_set]
    return likely + others


def check_results_html(results: Iterable[CheckResult]) -> str:
    """
    Render connection-check results as rich text: a colored mark, the check, then its detail.

    Each check gets its own block with the detail on a second line, so long endpoint URLs and error
    messages stay readable instead of running together on one line.
    """
    blocks: list[str] = []
    for result in results:
        mark = (
            '<span style="color:#3fb950;">&#10004;</span>'
            if result.ok
            else '<span style="color:#f85149;">&#10008;</span>'
        )
        name = html.escape(result.name)
        detail = html.escape(result.detail)
        blocks.append(f'{mark} <b>{name}</b><br><span style="color:#8a8a8a;">{detail}</span>')
    return "<br><br>".join(blocks)
