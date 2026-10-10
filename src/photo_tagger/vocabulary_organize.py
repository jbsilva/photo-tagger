"""
The optional model pass over a trimmed keyword list: synonyms and a hierarchy.

:mod:`photo_tagger.vocabulary_build` decides which keywords survive, and counting settles that
better than a model could. What counting cannot settle is meaning. Two questions are left over:

* Are "Golden Light" and "Golden Hour" one keyword or two? Loose matching folds case, punctuation,
  and plurals, so it already unifies "Animal" and "Animals". It cannot know that "Sea Hawk" is an
  osprey.
* Where does a keyword belong? A hierarchy has to come from somewhere, and a catalog a tool has
  been writing to is not a trustworthy source: derived from one, "Beach" ends up under "Sand".

The second question is why the model is asked for a *parent* as well as a category, and that field
earns its place: without it, asked only for synonyms, a model folds "Battery Pack" into "Camera
Accessories" and "Black" into "Color", destroying the narrower keyword. Naming the relation it
actually sees is what stops it reaching for the wrong one. Saying so in the prompt did not.

Both are judgment, so both are asked of the model, and only these two. Nothing here decides whether
a keyword is kept: that has already happened, deterministically, before this module runs.

Three things keep an opt-in model pass from making the file worse:

* **No invented keywords.** Every term the model returns is matched back to a term that was sent;
  anything else is dropped with a warning. The model regroups the list, it does not write it.
* **Nothing is lost.** A term folded into another is written as a ``{synonym}`` rather than deleted,
  so a photo tagged with it still matches.
* **A failed chunk changes nothing.** Terms in a chunk that errors, or that the model omits, keep
  the shape the deterministic pass gave them.

Categories are chosen once, up front, and every chunk assigns against that one fixed list. Asking
each chunk to invent its own would give "Animal" in one and "Animals" in the next, which is exactly
the sprawl this command exists to end.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings

from photo_tagger.ai import build_chat_model
from photo_tagger.vocabulary import loose_key
from photo_tagger.vocabulary_build import DROP_SYNONYM, Dropped, TrimResult


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic_ai.models.openai import OpenAIChatModel

    from photo_tagger.providers import ProviderName


# Keywords per request. Small enough that the reply fits comfortably inside the token budget and a
# weaker local model keeps track of the whole list, large enough that synonyms have a chance of
# landing in the same chunk as each other.
_CHUNK_SIZE = 60

# Most-used keywords shown to the categories pass. Enough to see what the library is about without
# paying for the whole list.
_CATEGORY_SAMPLE = 80

# Bounds on the category list. Too few and every keyword lands under "Nature"; too many and it is
# not a hierarchy, just the keyword list again with extra steps.
_MIN_CATEGORIES = 6
_MAX_CATEGORIES = 20

# Levels a rendered chain may have, category included. Deep hierarchies are unreadable in a
# keyword panel, and a model asked for parents will happily build a ten-level taxonomy.
_MAX_DEPTH = 4

# Most a group may claim before the whole group's synonyms are refused. Real synonym sets are
# small ("Golden Light" for "Golden Hour"); a model listing five is not naming synonyms, it is
# emptying a category into one keyword ("People" swallowing "Person", "Human", "Woman", "Man"),
# and every one of those it takes is a keyword the catalog loses.
_MAX_SYNONYMS = 3

# How far past the category limit a reply may go before it is refused outright. A model that
# ignores "between 6 and 20" tends to echo the keyword list back; truncating that to the first 20
# would look like an answer and behave like noise.
_DEGENERATE_FACTOR = 2

# Deterministic sampling, and room for a reasoning model to think before it answers. The budget is
# a ceiling, not a target: a model that emits its JSON straight away never touches it, while one
# that reasons first needs thousands of tokens before the first brace and otherwise fails the whole
# chunk with nothing to show.
_TEMPERATURE = 0.0
_MAX_TOKENS = 16384
_TIMEOUT_SECONDS = 180.0

_CATEGORY_SYSTEM_PROMPT = (
    "You organize a photographer's keyword catalog. Given the keywords they use most, name the "
    "top-level categories their photo library is about. Use broad, ordinary, singular nouns "
    "('Animal', 'Landscape', 'Architecture', 'Food', 'People'). Do not invent categories for "
    "subjects that are not represented. Answer only with the categories."
)

_ORGANIZE_SYSTEM_PROMPT = (
    "You organize a photographer's keyword catalog. You are given a list of keywords and a list of "
    "categories.\n"
    "Return one group per distinct concept:\n"
    "- 'preferred': the best spelling of that concept, copied EXACTLY from the keyword list.\n"
    "- 'synonyms': other keywords FROM THE LIST that mean the very same thing.\n"
    "- 'parent': a BROADER keyword from the list that this one is a kind of, a part of, or an "
    "example of. Empty when no keyword in the list is broader than it.\n"
    "- 'category': the one category from the category list this concept belongs under, or an "
    "empty string when none of them fits.\n"
    "\n"
    "MOST KEYWORDS HAVE NO SYNONYMS. Two keywords are synonyms only if swapping one for the other "
    "in a photo caption would not change what the caption says: 'Golden Light' and 'Golden Hour', "
    "'Sea Hawk' and 'Osprey', 'Bike' and 'Bicycle'. When in doubt, give a keyword its own group "
    "with no synonyms.\n"
    "A keyword that is a KIND of another, a PART of another, or an EXAMPLE of another is NOT a "
    "synonym of it: it keeps its own group and names the broader keyword as its 'parent'. "
    "'Battery Pack' has parent 'Camera Accessories', 'Black' has parent 'Color', and 'Osprey' has "
    "parent 'Bird'. None of those is a synonym.\n"
    "\n"
    "Never invent a keyword: every string in 'preferred' and 'synonyms' must appear in the "
    "keyword list, spelled the same way. Every keyword in the list must appear exactly once, in "
    "one group."
)


class KeywordGroup(BaseModel):
    """One concept: the spelling to keep, the ones that mean the same, and where it belongs."""

    preferred: str = Field(description="The keyword to keep, copied exactly from the input list")
    synonyms: list[str] = Field(default_factory=list, description="Input keywords meaning the same")
    parent: str = Field(
        default="",
        description="A broader keyword from the list that this one is a kind or part of, or empty",
    )
    category: str = Field(default="", description="A category from the given list, or empty")


class OrganizedChunk(BaseModel):
    """The model's grouping of one chunk of keywords."""

    groups: list[KeywordGroup] = Field(default_factory=list)


class CategoryList(BaseModel):
    """The top-level categories the whole library is organized under."""

    categories: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class OrganizeStats:
    """What the pass changed, for the run summary and the file's own header."""

    model_name: str = ""
    categories: list[str] = field(default_factory=list)
    grouped: int = 0
    categorized: int = 0
    invented: int = 0
    refused_groups: int = 0
    failed_chunks: int = 0


def _chunks(terms: Sequence[str], size: int = _CHUNK_SIZE) -> list[list[str]]:
    """Split *terms* into fixed-size chunks, so the same list always splits the same way."""
    return [list(terms[start : start + size]) for start in range(0, len(terms), size)]


@dataclass(frozen=True, slots=True)
class _Keywords:
    """The keywords a reply may name, by loose key, folded the way the trim folded them."""

    by_key: dict[str, str]
    fold_plurals: bool = True

    @classmethod
    def of(cls, terms: Iterable[str], *, fold_plurals: bool) -> _Keywords:
        """Index *terms* by their loose key."""
        by_key = {loose_key(term, fold_plurals=fold_plurals): term for term in terms}
        return cls(by_key, fold_plurals)

    def resolve(self, candidate: str) -> str | None:
        """
        Map a string the model returned back to the keyword that was sent, or None if invented.

        Matching is loose (case, punctuation, plurals where the language folds them) because a model
        retyping a list will change a capital or drop a hyphen; it is not free-form, because the
        result must be a keyword the library actually uses.
        """
        stripped = candidate.strip()
        if not stripped:
            return None
        return self.by_key.get(loose_key(stripped, fold_plurals=self.fold_plurals))


def _request[T: BaseModel](
    chat_model: OpenAIChatModel,
    output_type: type[T],
    system_prompt: str,
    prompt: str,
) -> T | None:
    """
    Run one text request, returning None instead of raising.

    A provider failure here costs the organizing, never the keywords: every caller falls back to
    what the deterministic pass already decided.
    """
    # pydantic-ai does not propagate `output_type` into the Agent generic, so both static
    # analyzers see Agent[object, str] where the runtime object decodes `output_type`. Same
    # blindness, and same two suppressions, as create_agent in ai.py.
    agent: Agent[None, T] = Agent(  # static analysis: ignore[incompatible_assignment]
        chat_model,
        output_type=output_type,  # type: ignore[arg-type]
        system_prompt=system_prompt,
        retries=2,
    )
    try:
        result = agent.run_sync(
            prompt,
            model_settings=OpenAIChatModelSettings(
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
                timeout=_TIMEOUT_SECONDS,
                # Grouping a list of words is recall, not deduction, and a reasoning model left to
                # its own devices spends thousands of tokens deliberating over sixty of them:
                # minutes per chunk, hours over a catalog. Servers that do not know the setting
                # ignore it, so it costs nothing where it is not needed.
                openai_reasoning_effort="none",
            ),
        )
    except Exception as exc:  # noqa: BLE001 - any provider failure degrades to "leave it alone"
        logger.warning("vocabulary_organize_request_failed", error=str(exc))
        return None
    return result.output


_CATEGORY_CLEAN_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def choose_categories(
    chat_model: OpenAIChatModel,
    terms: Sequence[str],
    *,
    fold_plurals: bool = True,
) -> list[str]:
    """
    Ask once for the library's top-level categories, and sanity-check the answer.

    One call for the whole run: every chunk then assigns against the same fixed list, which is what
    stops chunk one saying "Animal" and chunk two "Animals".
    """
    sample = list(terms[:_CATEGORY_SAMPLE])
    prompt = (
        f"These are the {len(sample)} most-used keywords in the library, most used first:\n"
        + ", ".join(sample)
        + f"\n\nName between {_MIN_CATEGORIES} and {_MAX_CATEGORIES} top-level categories."
    )
    output = _request(chat_model, CategoryList, _CATEGORY_SYSTEM_PROMPT, prompt)
    if output is None:
        return []

    seen: dict[str, str] = {}
    for raw in output.categories:
        name = _CATEGORY_CLEAN_RE.sub("", raw).strip()
        key = loose_key(name, fold_plurals=fold_plurals)
        if name and key not in seen:
            seen[key] = name

    if len(seen) > _MAX_CATEGORIES * _DEGENERATE_FACTOR:
        # Asked for a handful of top-level categories, a weak model hands back the keyword list it
        # was shown. Truncating that to the first 20 would look like an answer and behave like
        # noise, so the hierarchy is skipped and the pass only folds synonyms.
        logger.warning(
            "vocabulary_categories_rejected",
            returned=len(seen),
            limit=_MAX_CATEGORIES,
            reason="model ignored the requested count",
        )
        return []

    categories = list(seen.values())[:_MAX_CATEGORIES]
    logger.info("vocabulary_categories_chosen", categories=categories)
    return categories


def _organize_chunk(
    chat_model: OpenAIChatModel,
    chunk: Sequence[str],
    categories: Sequence[str],
) -> OrganizedChunk | None:
    """Ask the model to group one chunk of keywords under the fixed categories."""
    prompt = (
        "Categories:\n"
        + (", ".join(categories) if categories else "(none)")
        + "\n\nKeywords:\n"
        + "\n".join(f"- {term}" for term in chunk)
    )
    return _request(chat_model, OrganizedChunk, _ORGANIZE_SYSTEM_PROMPT, prompt)


@dataclass(slots=True)
class _Applied:
    """The bookkeeping of applying one chunk's groups: what merged, what moved, what was junk."""

    synonyms: dict[str, list[str]] = field(default_factory=dict)
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    aliased: dict[str, str] = field(default_factory=dict)
    invented: int = 0
    refused_groups: int = 0
    # The chunk's request failed outright, as opposed to succeeding with nothing to group. The two
    # look identical from the outside (no synonyms, no chains), and only this tells them apart.
    failed: bool = False


def _claim_synonyms(
    group: KeywordGroup,
    preferred: str,
    known: _Keywords,
    claimed: set[str],
    applied: _Applied,
) -> None:
    """
    Record *group*'s synonyms of *preferred*, skipping invented and already-claimed keywords.

    A group claiming more than :data:`_MAX_SYNONYMS` is refused whole rather than trimmed: the
    keywords are not a long synonym set that happens to run over, they are a category being poured
    into one keyword, and picking three of them at random would keep the mistake and hide its size.
    """
    if len(group.synonyms) > _MAX_SYNONYMS:
        logger.warning(
            "vocabulary_synonym_group_refused",
            preferred=preferred,
            claimed=len(group.synonyms),
            limit=_MAX_SYNONYMS,
        )
        applied.refused_groups += 1
        return

    aliases = []
    for raw in group.synonyms:
        synonym = known.resolve(raw)
        if synonym is None:
            applied.invented += 1
        elif synonym != preferred and synonym not in claimed:
            claimed.add(synonym)
            aliases.append(synonym)
            applied.aliased[synonym] = preferred
    if aliases:
        applied.synonyms[preferred] = aliases


def _build_chain(
    term: str,
    parent_of: dict[str, str],
    category_of: dict[str, str],
    aliased: dict[str, str],
) -> tuple[str, ...]:
    """
    Walk *term*'s parents up to its root and prepend the root's category.

    Models state parentage both ways round ("A is a part of B" and "B is a part of A" in the same
    reply), so the walk stops the moment it revisits a keyword: a cycle yields the chain built so
    far rather than looping. Depth is capped for the same reason a vocabulary is capped at all, and
    because a Lightroom keyword nobody can read is no better than no hierarchy.
    """
    chain = [term]
    seen = {term}
    while (parent := parent_of.get(chain[0])) is not None and len(chain) < _MAX_DEPTH:
        # A parent folded into a synonym belongs to whichever keyword kept it.
        parent = aliased.get(parent, parent)
        if parent in seen:
            break
        seen.add(parent)
        chain.insert(0, parent)
    if (category := category_of.get(chain[0])) is not None and category != chain[0]:
        # The category is a level like any other, so the broadest parent gives way to it rather
        # than the chain growing past the cap.
        return (category, *chain[: _MAX_DEPTH - 1])
    return tuple(chain)


def _apply_groups(
    groups: Sequence[KeywordGroup],
    chunk: Sequence[str],
    categories: Sequence[str],
    *,
    fold_plurals: bool = True,
) -> _Applied:
    """
    Turn one chunk's reply into synonym and hierarchy decisions, discarding anything invented.

    A keyword may be claimed once. A later group naming a keyword already spoken for is ignored
    rather than allowed to move it, so the result cannot depend on how the model ordered its reply.
    """
    known = _Keywords.of(chunk, fold_plurals=fold_plurals)
    category_by = _Keywords.of(categories, fold_plurals=fold_plurals)
    applied = _Applied()
    claimed: set[str] = set()
    parent_of: dict[str, str] = {}
    category_of: dict[str, str] = {}

    for group in groups:
        preferred = known.resolve(group.preferred)
        if preferred is None:
            applied.invented += 1
            continue
        if preferred in claimed:
            continue
        claimed.add(preferred)
        _claim_synonyms(group, preferred, known, claimed, applied)

        if (parent := known.resolve(group.parent)) is not None and parent != preferred:
            parent_of[preferred] = parent
        if (category := category_by.resolve(group.category)) is not None:
            category_of[preferred] = category

    for term in claimed - set(applied.aliased):
        chain = _build_chain(term, parent_of, category_of, applied.aliased)
        if len(chain) > 1:
            applied.chains[term] = chain
    return applied


def organize(  # noqa: PLR0913 - provider credentials plus the list to organize.
    result: TrimResult,
    *,
    provider_name: ProviderName,
    model_name: str,
    api_base_url: str | None,
    api_key: str | None,
    workers: int = 1,
) -> tuple[TrimResult, OrganizeStats]:
    """
    Fold synonyms together and give the kept keywords a hierarchy, using the model.

    Returns a new :class:`TrimResult` plus what changed. Terms the model folds into another are
    recorded as dropped with the reason ``synonym``, but they are not lost: the renderer writes them
    as ``{braces}`` on the keyword that kept them, so a photo tagged with one still matches.

    Chunks are independent, so *workers* runs several at once; results are reassembled by position,
    which keeps the output the same whatever order they finish in.
    """
    if not result.kept:
        return result, OrganizeStats()

    chat_model = build_chat_model(
        provider_name,
        model_name,
        api_base_url=api_base_url,
        api_key=api_key,
    )
    categories = choose_categories(chat_model, result.kept, fold_plurals=result.fold_plurals)
    chunks = _chunks(result.kept)
    logger.info("vocabulary_organize_started", chunks=len(chunks), terms=len(result.kept))

    def run_chunk(chunk: list[str]) -> _Applied:
        replies = _organize_chunk(chat_model, chunk, categories)
        if replies is None:
            return _Applied(failed=True)
        return _apply_groups(replies.groups, chunk, categories, fold_plurals=result.fold_plurals)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        applied_chunks = list(pool.map(run_chunk, chunks))

    stats = OrganizeStats(model_name=model_name, categories=list(categories))
    synonyms: dict[str, list[str]] = {}
    chains: dict[str, tuple[str, ...]] = {}
    aliased: dict[str, str] = {}
    for applied in applied_chunks:
        synonyms.update(applied.synonyms)
        chains.update(applied.chains)
        aliased.update(applied.aliased)
        stats.invented += applied.invented
        stats.refused_groups += applied.refused_groups

    stats.grouped = len(aliased)
    stats.categorized = sum(1 for chain in chains.values() if len(chain) > 1)
    stats.failed_chunks = sum(1 for applied in applied_chunks if applied.failed)

    kept = [term for term in result.kept if term not in aliased]
    dropped = [
        *result.dropped,
        *(
            Dropped(term, result.census.uses[term], DROP_SYNONYM, f"alias of {preferred}")
            for term, preferred in aliased.items()
        ),
    ]
    logger.info(
        "vocabulary_organized",
        kept=len(kept),
        grouped=stats.grouped,
        categorized=stats.categorized,
        invented=stats.invented,
        refused_groups=stats.refused_groups,
        failed_chunks=stats.failed_chunks,
    )
    return (
        TrimResult(
            kept=kept,
            dropped=dropped,
            census=result.census,
            chains=chains,
            synonyms=synonyms,
            fold_plurals=result.fold_plurals,
        ),
        stats,
    )
