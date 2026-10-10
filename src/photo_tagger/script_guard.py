"""
Spot generated text that mixes writing systems.

A model asked for English output now and then reaches for a word in its own script anyway, so a
title comes back as ``Decorated陶笛 with Geometric Pattern`` where ``Ocarina`` would have said the
same thing. The prompt forbids it (see :mod:`photo_tagger.config`); this module notices when the
model did it regardless, so the photo can be flagged for review or handed back for a repair pass.

The check takes no language argument on purpose. The output language is a free-form name the user
types (``"Brazilian Portuguese"``, ``"Aragonese"``), so a name-to-script table would be a never-
finished list that still guesses wrong. Instead the text is judged against *itself*: whatever script
the bulk of it is written in is the one it meant to use, and a handful of letters in another script
is the anomaly. That holds for every language without knowing any of them.

Everything here is pure, so callers can re-run it on a cached inference rather than store its
verdict alongside one.
"""

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    from photo_tagger.models import InferenceResult


# The scripts recognized here, spelled as the first word of a character's Unicode name
# ("CJK UNIFIED IDEOGRAPH-9676", "CYRILLIC CAPITAL LETTER A"). A letter whose name does not start
# with one of these is ignored rather than counted as a script of its own: some are letters only by
# Unicode category and carry no script at all (MICRO SIGN is an Ll, FEMININE ORDINAL INDICATOR an
# Lo), and treating those as foreign would flag ordinary text. The list covers the scripts a vision
# model plausibly writes in; an omission costs a missed flag, never a false one.
_SCRIPTS = frozenset(
    {
        "ARABIC",
        "ARMENIAN",
        "BENGALI",
        "CHEROKEE",
        "CJK",
        "CYRILLIC",
        "DEVANAGARI",
        "ETHIOPIC",
        "GEORGIAN",
        "GREEK",
        "GUJARATI",
        "GURMUKHI",
        "HANGUL",
        "HEBREW",
        "HIRAGANA",
        "KANNADA",
        "KATAKANA",
        "KHMER",
        "LAO",
        "LATIN",
        "MALAYALAM",
        "MONGOLIAN",
        "MYANMAR",
        "ORIYA",
        "SINHALA",
        "SYRIAC",
        "TAMIL",
        "TELUGU",
        "THAANA",
        "THAI",
        "TIBETAN",
        "YI",
    },
)

# Scripts that belong to one writing system. Japanese prose mixes Han with both kana and Korean
# mixes Hangul with Han, so counting them apart would make perfectly ordinary text look mixed.
_FAMILIES = {"HIRAGANA": "CJK", "KATAKANA": "CJK", "HANGUL": "CJK"}

# Compatibility forms put a width in front of the script word ("FULLWIDTH LATIN CAPITAL LETTER A",
# "HALFWIDTH KATAKANA LETTER A"). They are the same script, so the width is dropped.
_WIDTH_PREFIXES = ("HALFWIDTH ", "FULLWIDTH ")

# How many letters one CJK character (Han, kana, Hangul) counts for when the scripts are weighed.
# It carries about a syllable or a word, where a Latin letter carries a sound: a Japanese word runs
# two characters, an English one five letters. Counted one to one, a few brand names in Latin
# outweigh the Japanese around them ("キヤノン EOS R5 Mark II ボディ"), and the repair would be
# asked to replace the Japanese.
_CJK_WEIGHT = 3

# How many distinct foreign runs one photo reports. A flag is a prompt to go look, so a handful
# makes the point; the cap keeps a badly confused answer from filling a CSV cell or a repair prompt.
_MAX_TERMS = 20

# It takes two scripts to be mixed, and a third-place script changes nothing: the question is only
# ever whether the runner-up is a real minority or half of a bilingual answer.
_MIXED = 2


def script_family(char: str) -> str | None:
    """
    Return the writing system *char* belongs to, or None when it carries none.

    None covers everything that says nothing about language: digits, punctuation, whitespace, emoji,
    and the few letters Unicode names without a script.
    """
    if not unicodedata.category(char).startswith("L"):
        return None
    # Defaulted rather than caught: every letter in the UCD is named, so the fallback is only
    # insurance, and one that reads as "no script" instead of raising out of a whole batch.
    name = unicodedata.name(char, "")
    for prefix in _WIDTH_PREFIXES:
        name = name.removeprefix(prefix)
    script = name.partition(" ")[0]
    if script not in _SCRIPTS:
        return None
    return _FAMILIES.get(script, script)


@dataclass(frozen=True, slots=True)
class ScriptFinding:
    """One photo's generated text written mostly in *dominant* but carrying *terms* in others."""

    dominant: str
    terms: list[str]


def _count_families(texts: list[str]) -> dict[str, int]:
    """Weigh how much of each writing system the pooled *texts* hold (see :data:`_CJK_WEIGHT`)."""
    counts: dict[str, int] = {}
    for text in texts:
        for char in text:
            family = script_family(char)
            if family is not None:
                weight = _CJK_WEIGHT if family == "CJK" else 1
                counts[family] = counts.get(family, 0) + weight
    return counts


def _dominant_family(counts: dict[str, int]) -> str | None:
    """
    Return the writing system the text is written in, or None when that is not a fair reading.

    None when only one script is present (nothing is mixed) and when the top two are tied: an even
    split is bilingual text by construction, and calling either half the anomaly would be a coin
    toss.
    """
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < _MIXED or ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _foreign_runs(texts: list[str], dominant: str) -> list[str]:
    """
    Pull out the stretches of *texts* written in a script other than *dominant*.

    Each run is returned as it appeared, first occurrence first, duplicates collapsed, so a caller
    can show them to the user or name them in a repair prompt. A combining mark extends the run it
    follows instead of cutting it, keeping an accented letter whole.
    """
    found: dict[str, None] = {}
    for text in texts:
        run: list[str] = []
        for char in text:
            family = script_family(char)
            extends = family is not None and family != dominant
            if extends or (run and unicodedata.category(char).startswith("M")):
                run.append(char)
                continue
            if run:
                found.setdefault("".join(run), None)
                run = []
        if run:
            found.setdefault("".join(run), None)
    return list(found)[:_MAX_TERMS]


def find_foreign_script(texts: Iterable[str]) -> ScriptFinding | None:
    """
    Report the minority-script text across *texts*, or None when they read as one language.

    The strings are judged as one pool, so a keyword in another script is caught even when the
    title and description are clean.

    Examples:
        >>> find_foreign_script(["Decorated陶笛 with Geometric Pattern"])
        ScriptFinding(dominant='LATIN', terms=['陶笛'])
        >>> find_foreign_script(["Ocarina on a White Surface"]) is None
        True
    """
    pooled = [text for text in texts if text]
    dominant = _dominant_family(_count_families(pooled))
    if dominant is None:
        return None
    # Never empty: a dominant script means a runner-up was counted, and the two functions start a
    # run on the same characters _count_families counted.
    return ScriptFinding(dominant=dominant, terms=_foreign_runs(pooled, dominant))


def find_in_result(result: InferenceResult) -> ScriptFinding | None:
    """Run :func:`find_foreign_script` over one inference's title, description, and keywords."""
    return find_foreign_script([result.title, result.description, *result.keywords])
