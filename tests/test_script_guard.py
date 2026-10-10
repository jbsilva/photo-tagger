"""Tests for the mixed-writing-system check in photo_tagger.script_guard."""

import pytest

from photo_tagger.models import InferenceResult
from photo_tagger.script_guard import (
    _MAX_TERMS,
    find_foreign_script,
    find_in_result,
    script_family,
)


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("a", "LATIN"),
        ("é", "LATIN"),  # e with acute
        ("ß", "LATIN"),  # sharp s
        ("Ａ", "LATIN"),  # fullwidth A, same script
        ("Ж", "CYRILLIC"),
        ("ω", "GREEK"),
        ("陶", "CJK"),
        ("あ", "CJK"),  # kana and Han are one writing system here
        ("ア", "CJK"),
        ("ｱ", "CJK"),  # halfwidth katakana
        ("가", "CJK"),  # hangul
        ("ا", "ARABIC"),
        ("ก", "THAI"),
        ("5", None),
        (" ", None),
        ("<", None),
        ("\U0001f986", None),  # emoji
        ("µ", None),  # MICRO SIGN is an Ll with no script
        ("ª", None),  # FEMININE ORDINAL INDICATOR likewise
        ("\x00", None),  # unnamed code point
    ],
)
def test_script_family_reads_the_writing_system_off_the_character(
    char: str,
    expected: str | None,
) -> None:
    """Letters report their script (kana folded into CJK); everything else reports none."""
    assert script_family(char) == expected


def test_a_loanword_left_untranslated_is_found() -> None:
    """The case this module exists for: one CJK word inside otherwise English metadata."""
    finding = find_foreign_script(
        [
            "Decorated陶笛 with Geometric Pattern",
            "A陶笛 with a geometric pattern in red, green, and black on a plain surface.",
            "陶笛<Wind Instrument<Music Instrument",
        ],
    )
    assert finding is not None
    assert finding.dominant == "LATIN"
    assert finding.terms == ["陶笛"]


def test_clean_english_is_not_flagged() -> None:
    """Text in one script has no minority to report."""
    assert (
        find_foreign_script(["Ocarina with Geometric Pattern", "Ocarina<Wind Instrument"]) is None
    )


def test_accented_latin_is_not_a_foreign_script() -> None:
    """A language is not another script just because it uses diacritics."""
    assert find_foreign_script(["Café Terrace", "Ausstellung für Straßenfoto"]) is None


def test_japanese_output_is_not_flagged_against_itself() -> None:
    """Japanese mixes Han with kana in ordinary prose; that is one writing system, not a defect."""
    assert find_foreign_script(["白い机の上の陶笛"]) is None


def test_brand_names_in_latin_do_not_outweigh_the_japanese_around_them() -> None:
    """A CJK character carries about a word's worth, so it counts for more than one Latin letter."""
    finding = find_foreign_script(["キヤノン EOS R5 Mark II ボディ"])
    assert finding is not None
    assert finding.dominant == "CJK"


def test_a_keyword_alone_can_raise_the_flag() -> None:
    """The fields are judged as one pool, so a clean title does not hide a mixed keyword."""
    finding = find_foreign_script(
        [
            "Ocarina with Geometric Pattern",
            "An ocarina rests on a plain white surface.",
            "陶笛<Wind Instrument",
        ],
    )
    assert finding is not None
    assert finding.terms == ["陶笛"]


def test_an_even_split_is_read_as_bilingual_rather_than_broken() -> None:
    """With the top two scripts tied there is no minority; calling one of them wrong is a guess."""
    # Nine Latin letters weigh what three CJK characters do (see _CJK_WEIGHT).
    assert find_foreign_script(["Clay Flute", "陶" * 3]) is None


def test_the_minority_is_decided_by_count_not_by_script() -> None:
    """Nothing here privileges Latin: a stray English keyword in Russian output flags too."""
    finding = find_foreign_script(["Изба в лесу", "Log"])
    assert finding is not None
    assert finding.dominant == "CYRILLIC"
    assert finding.terms == ["Log"]


def test_runs_are_reported_once_in_the_order_they_appear() -> None:
    """A term repeated across fields is one thing to look at, not three."""
    sushi, tempura = "寿司", "天ぷら"
    finding = find_foreign_script(
        [
            f"{sushi} and {tempura}",
            f"A plate of {sushi} on a wooden board beside a bowl.",
            f"{sushi}<Food",
        ],
    )
    assert finding is not None
    assert finding.terms == [sushi, tempura]


def test_a_combining_mark_stays_on_the_letter_it_modifies() -> None:
    """A decomposed accent extends its run instead of cutting the term in two."""
    accented = "р́"  # Cyrillic er plus a combining acute
    finding = find_foreign_script(["Moscow Winter Trip Photo Album", f"Reka {accented}"])
    assert finding is not None
    assert finding.terms == [accented]


def test_the_reported_terms_are_capped() -> None:
    """A thoroughly confused answer cannot fill a CSV cell or a repair prompt."""
    keywords = [f"Keyword {n} {'а' * (n + 1)}" for n in range(_MAX_TERMS + 10)]
    finding = find_foreign_script([*keywords, "A" * 500])
    assert finding is not None
    assert len(finding.terms) == _MAX_TERMS


def test_empty_and_blank_text_is_never_flagged() -> None:
    """Nothing generated means nothing to review."""
    assert find_foreign_script([]) is None
    assert find_foreign_script(["", "   ", "123 456"]) is None


def test_find_in_result_reads_all_three_fields() -> None:
    """The InferenceResult wrapper pools the title, the description, and every keyword."""
    result = InferenceResult(
        title="Ocarina on a Table",
        description="An ocarina rests on a plain white surface.",
        keywords=["Ocarina", "陶笛<Wind Instrument"],
    )
    finding = find_in_result(result)
    assert finding is not None
    assert finding.terms == ["陶笛"]
    clean = InferenceResult(title="Ocarina", description="One ocarina.", keywords=["Ocarina"])
    assert find_in_result(clean) is None
