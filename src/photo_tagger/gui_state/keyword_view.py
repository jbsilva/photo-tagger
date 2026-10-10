"""The editable keyword field: parsing it, showing hierarchies, and the diff against the photo."""

import html
from typing import TYPE_CHECKING

from photo_tagger.i18n import _
from photo_tagger.keywords import dedupe_keywords, merge_keywords
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def parse_keyword_lines(text: str) -> list[str]:
    """
    Parse the editable keyword field (one keyword per line) into a clean list.

    Repeats are collapsed case-insensitively so the change counts stay honest; the model (and a
    pasting user) sometimes repeats a keyword.
    """
    return dedupe_keywords(text.splitlines())


def keywords_to_text(keywords: list[str]) -> str:
    """Render keywords one per line for the editable text field."""
    return "\n".join(keywords)


def keywords_to_save(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> KeywordSet:
    """
    Build the :class:`KeywordSet` to write from the edited keywords.

    Merges with the existing keywords unless *overwrite* is set, in which case the existing keywords
    are dropped first. Hierarchical entries (``Duck<Bird<Animal``) are parsed by
    :func:`merge_keywords` exactly as the CLI does.

    *verbatim* carries the controlled vocabulary's own spelling of each term (its ``exact`` index),
    which the merge step must not touch: a catalog that keeps its keywords in lower case means it,
    and title-casing "gegenlicht" on the way out would write the near-duplicate the vocabulary
    exists to prevent. The CLI's own write path passes the same index.
    """
    base = KeywordSet() if overwrite else existing
    return merge_keywords(base, edited_keywords, verbatim=verbatim)


# A nested name -> children mapping used to fold cumulative hierarchy paths into one tree.
type _Tree = dict[str, "_Tree"]


def chain_to_display(path: str) -> str:
    """
    Convert a Lightroom root-first ``A|B|C`` path to the GUI's leaf-first ``C<B<A`` notation.

    The editable keyword field already speaks the '<' form (it is what the model emits and what the
    CLI documents), so every read-only view uses it too instead of leaking the on-disk '|'.
    """
    return "<".join(reversed(path.split("|")))


def keyword_lines(keywords: KeywordSet) -> list[str]:
    """
    Render a keyword set as the shortest list of lines that reproduces it, sorted.

    Each hierarchy shows once, as its deepest chain (``Duck<Bird<Animal``); the intermediate flat
    copies Lightroom also stores (Animal, Bird) are folded into it. Flat keywords that belong to no
    hierarchy join them. This mirrors what a user would type to reproduce the same metadata.

    Sorting is what lets the Existing and New columns be read side by side: a keyword lands on the
    same line in both. The '<' notation is leaf-first, so sorting the rendered line sorts by leaf.
    """
    chains = [entry for entry in keywords.hierarchical if "|" in entry]
    deepest = [
        entry
        for entry in chains
        if not any(other.startswith(entry + "|") for other in chains if other != entry)
    ]
    covered = {segment.casefold() for entry in chains for segment in entry.split("|")}
    lines = [chain_to_display(entry) for entry in deepest]
    lines += [kw for kw in keywords.subject if kw.casefold() not in covered]
    return sorted(lines, key=str.casefold)


def format_existing_keywords(keywords: KeywordSet) -> str:
    """Render existing keywords one per line, in the same ``<`` notation as the editable field."""
    return "\n".join(keyword_lines(keywords))


def normalize_keyword_lines(
    keywords: list[str],
    *,
    verbatim: Mapping[str, str] | None = None,
) -> list[str]:
    """
    Fold an edited keyword list into the shortest list that writes the same metadata, sorted.

    A model returns its flat keywords and its hierarchies as two lists, so every leaf reached the
    field twice (``Beach`` and ``Beach<Sandy Area<Outdoor Area``). Merging expands every chain into
    its levels, so the bare copies were noise. Run through the save's own merge, so the field shows
    what the file would get. Applied when a proposal arrives, never while the user types.
    """
    return keyword_lines(merge_keywords(KeywordSet(), keywords, verbatim=verbatim))


def _walk_tree(node: _Tree, prefix: str, lines: list[str]) -> None:
    """Append *node*'s children to *lines* with ``tree``-style guide characters."""
    entries = list(node.items())
    for index, (name, child) in enumerate(entries):
        last = index == len(entries) - 1
        lines.append(prefix + ("└─ " if last else "├─ ") + name)
        _walk_tree(child, prefix + ("   " if last else "│  "), lines)


def hierarchy_tree_text(paths: Iterable[str]) -> str:
    """
    Render Lightroom ``A|B|C`` paths as a ``tree``-style diagram with branch guides.

    Cumulative paths ("A|B", "A|B|C") collapse into one branch, so the view shows the taxonomy shape
    rather than repeating every prefix line. Roots sit at column zero and children hang off
    ``├─``/``└─`` connectors like the CLI ``tree`` command, which reads far better than plain two-
    space indentation.
    """
    root: _Tree = {}
    for path in paths:
        node = root
        for segment in path.split("|"):
            node = node.setdefault(segment, {})

    lines: list[str] = []
    for name, child in root.items():
        lines.append(name)
        _walk_tree(child, "", lines)
    return "\n".join(lines)


def hierarchy_preview(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> str:
    """Render the keyword tree that saving the edited keywords would produce."""
    return hierarchy_tree_text(
        keywords_to_save(
            existing,
            edited_keywords,
            overwrite=overwrite,
            verbatim=verbatim,
        ).hierarchical,
    )


# Diff states for a keyword when comparing the existing flat subjects to what a save writes.
ADDED = "added"


REMOVED = "removed"


UNCHANGED = "unchanged"


def keyword_diff(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
    verbatim: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """
    Compare existing flat keywords to the result of saving the edited keywords.

    Returns ``(keyword, state)`` pairs where state is :data:`ADDED`, :data:`REMOVED`, or
    :data:`UNCHANGED`. The keywords that will be written come first in write order, then any that
    would be dropped (only possible with *overwrite*). Comparison is case-insensitive.
    """
    result = keywords_to_save(
        existing,
        edited_keywords,
        overwrite=overwrite,
        verbatim=verbatim,
    ).subject
    existing_folds = {kw.casefold() for kw in existing.subject}
    result_folds = {kw.casefold() for kw in result}
    diff = [(kw, UNCHANGED if kw.casefold() in existing_folds else ADDED) for kw in result]
    diff += [(kw, REMOVED) for kw in existing.subject if kw.casefold() not in result_folds]
    return diff


# How keyword_diff_html draws each state: the inline style, then the marker before the keyword.
_DIFF_STYLE = {
    ADDED: ("color:#3fb950", "+&nbsp;"),
    REMOVED: ("color:#f85149;text-decoration:line-through", "&minus;&nbsp;"),
}


_UNCHANGED_STYLE = ("color:#8a8a8a", "&nbsp;&nbsp;&nbsp;")


def keyword_diff_html(diff: Iterable[tuple[str, str]]) -> str:
    """Render a :func:`keyword_diff` as HTML: green added, red struck-through removed, grey kept."""
    rows: list[str] = []
    for keyword, state in diff:
        safe = html.escape(keyword)
        style, marker = _DIFF_STYLE.get(state, _UNCHANGED_STYLE)
        rows.append(f'<span style="{style}">{marker}{safe}</span>')
    return "<br>".join(rows) or _("(no change)")
