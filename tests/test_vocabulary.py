"""Tests for controlled-vocabulary parsing, matching, and snapping."""

from pathlib import Path

import pytest
from loguru import logger

from photo_tagger.vocabulary import (
    MAX_TRACKED_DROPPED_TERMS,
    Vocabulary,
    VocabularyError,
    _lightroom_csv_keywords,
    _singularize,
    folds_plurals,
    fuzzy_key_match,
    load_vocabulary,
    loose_key,
    parse_keyword_lines,
    record_dropped_terms,
)


LIGHTROOM_EXPORT = """\
Animal
\tBird
\t\tOsprey
\t\t{Sea Hawk}
\t\t{Fish Hawk}
\t\tMallard
\tMammal
\t\t[Pet]
Landscape
"""

# The same keywords as LIGHTROOM_EXPORT, exported the other way: four option columns in front and
# the indentation tucked inside the last field.
LIGHTROOM_CSV_EXPORT = """\
Include On Export,Export Containing Keywords,Export Synonyms,Person Type Keyword,
Y,Y,Y,N,Animal
Y,Y,Y,N,\tBird
Y,Y,Y,N,\t\tOsprey
Y,Y,Y,N,\t\t{Sea Hawk}
Y,Y,Y,N,\t\t{Fish Hawk}
Y,Y,Y,N,\t\tMallard
Y,Y,Y,N,\tMammal
Y,Y,Y,N,\t\t[Pet]
Y,Y,Y,N,Landscape
"""


def test_parse_keyword_lines_reads_an_indented_lightroom_export() -> None:
    """Indentation builds the chain, braces attach synonyms, brackets are stripped."""
    entries = parse_keyword_lines(LIGHTROOM_EXPORT)
    chains = [entry.chain for entry in entries]
    assert chains == [
        ["Animal"],
        ["Animal", "Bird"],
        ["Animal", "Bird", "Osprey"],
        ["Animal", "Bird", "Mallard"],
        ["Animal", "Mammal"],
        ["Animal", "Mammal", "Pet"],
        ["Landscape"],
    ]
    osprey = next(entry for entry in entries if entry.chain[-1] == "Osprey")
    assert osprey.synonyms == ["Sea Hawk", "Fish Hawk"]


def test_parse_keyword_lines_accepts_space_indentation_of_any_width() -> None:
    """Widths are compared, not counted in levels, so four-space indents nest the same."""
    entries = parse_keyword_lines("Animal\n    Bird\n        Osprey\n")
    assert [entry.chain for entry in entries] == [
        ["Animal"],
        ["Animal", "Bird"],
        ["Animal", "Bird", "Osprey"],
    ]


def test_parse_keyword_lines_reads_flat_paths_and_ignores_comments() -> None:
    """Both path spellings work on one line; blanks and '#' lines are skipped."""
    entries = parse_keyword_lines(
        "# my keywords\n\nAnimal|Bird|Osprey\nOak<Tree<Plant\nLandscape\n",
    )
    assert [entry.chain for entry in entries] == [
        ["Animal", "Bird", "Osprey"],
        ["Plant", "Tree", "Oak"],
        ["Landscape"],
    ]


def test_parse_keyword_lines_keeps_hashtag_keywords() -> None:
    """A comment needs a space after the hash, so a hashtag-shaped keyword survives."""
    entries = parse_keyword_lines("# a comment\n#\n#Diversity\n\t#1\nAnimal\n")
    assert [entry.chain for entry in entries] == [
        ["#Diversity"],
        ["#Diversity", "#1"],
        ["Animal"],
    ]


def test_parse_keyword_lines_does_not_let_a_one_line_path_parent_the_next_line() -> None:
    """A full path stands alone; an indented line after it still belongs to the tree above."""
    entries = parse_keyword_lines("Animal\n\tAnimal|Bird|Osprey\n\tMammal\n")
    assert [entry.chain for entry in entries] == [
        ["Animal"],
        ["Animal", "Bird", "Osprey"],
        ["Animal", "Mammal"],
    ]


def test_from_entries_indexes_every_segment_and_keeps_the_first_spelling() -> None:
    """Each level of a path is its own term, and the first casing seen wins."""
    vocab = Vocabulary.from_entries(["Animal|Bird|Osprey", "animal|Mammal"])
    assert vocab.terms == ("Animal", "Bird", "Osprey", "Mammal")
    assert vocab.chain_for("Osprey") == ["Animal", "Bird", "Osprey"]
    assert vocab.match("ANIMAL") == "Animal"


def test_from_entries_keeps_the_longest_chain_for_a_leaf() -> None:
    """A leaf listed twice keeps its deeper hierarchy, whichever order it is listed in."""
    deep_last = Vocabulary.from_entries(["Bird|Osprey", "Animal|Bird|Osprey"])
    assert deep_last.chain_for("Osprey") == ["Animal", "Bird", "Osprey"]

    deep_first = Vocabulary.from_entries(["Animal|Bird|Osprey", "Bird|Osprey"])
    assert deep_first.chain_for("Osprey") == ["Animal", "Bird", "Osprey"]


def test_from_entries_skips_entries_with_no_term() -> None:
    """Blank strings contribute nothing instead of raising."""
    vocab = Vocabulary.from_entries(["", "   ", "[]", "Bird"])
    assert vocab.terms == ("Bird",)


def test_parse_keyword_lines_ignores_a_synonym_with_no_keyword_above_it() -> None:
    """A stray synonym line at the top of the file has nothing to alias."""
    entries = parse_keyword_lines("{Orphan}\nBird\n")
    assert [entry.chain for entry in entries] == [["Bird"]]


def test_empty_vocabulary_is_falsy() -> None:
    """An empty vocabulary reads as False so callers can skip the snap entirely."""
    assert not Vocabulary()
    assert Vocabulary.from_entries(["Bird"])


def test_match_layers_exact_loose_and_fuzzy_lookups() -> None:
    """Case, plurals, punctuation, and small typos all resolve to the catalog spelling."""
    vocab = Vocabulary.from_entries(["Bird of Prey", "Osprey", "NYC"])
    assert vocab.match("bird of prey") == "Bird of Prey"
    assert vocab.match("Ospreys") == "Osprey"
    assert vocab.match("Bird-of-Prey") == "Bird of Prey"
    assert vocab.match("Ospray") == "Osprey"
    assert vocab.match("NYC") == "NYC"
    assert vocab.match("Tractor") is None
    assert vocab.match("   ") is None


def test_match_refuses_a_fuzzy_hit_that_starts_with_another_letter() -> None:
    """A leading extra letter usually means a different word, however close the score."""
    vocab = Vocabulary.from_entries(["Eagle", "Wildlife"])
    assert vocab.match("Beagle") is None
    assert vocab.match("Wild Life") == "Wildlife"


def test_match_resolves_synonyms_to_the_canonical_term() -> None:
    """A synonym is an alias, not a term of its own."""
    vocab = Vocabulary.from_entries(parse_keyword_lines(LIGHTROOM_EXPORT))
    assert vocab.match("Sea Hawk") == "Osprey"
    assert "Sea Hawk" not in vocab.terms


def test_match_skips_the_fuzzy_pass_on_a_huge_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the size limit only the dict lookups run, so a typo no longer resolves."""
    monkeypatch.setattr("photo_tagger.vocabulary._FUZZY_MAX_TERMS", 1)
    vocab = Vocabulary.from_entries(["Osprey", "Mallard"])
    assert vocab.match("Osprey") == "Osprey"
    assert vocab.match("Ospray") is None


def test_chain_for_ignores_single_segment_chains() -> None:
    """A root-level term has no hierarchy to contribute."""
    vocab = Vocabulary.from_entries(["Landscape", "Animal|Bird"])
    assert vocab.chain_for("Landscape") is None
    assert vocab.chain_for("Bird") == ["Animal", "Bird"]


def test_snap_rewrites_to_the_catalog_term_and_hierarchy() -> None:
    """The model's own parents are replaced by the vocabulary's chain, leaf-first."""
    vocab = Vocabulary.from_entries(["Animal|Bird|Osprey"])
    result = vocab.snap(["ospreys", "Osprey<Raptor<Wildlife"])
    assert result.keywords == ["Osprey<Bird<Animal"]
    assert result.mapped == {
        "ospreys": "Osprey<Bird<Animal",
        "Osprey<Raptor<Wildlife": "Osprey<Bird<Animal",
    }
    assert result.dropped == []


def test_snap_passes_unknown_keywords_through_unless_strict() -> None:
    """Non-strict mode is additive; strict mode reports what it discarded."""
    vocab = Vocabulary.from_entries(["Osprey"])
    lenient = vocab.snap(["Osprey", "Golden Hour"])
    assert lenient.keywords == ["Osprey", "Golden Hour"]
    assert lenient.dropped == []

    strict = vocab.snap(["Osprey", "Golden Hour"], strict=True)
    assert strict.keywords == ["Osprey"]
    assert strict.dropped == ["Golden Hour"]


def test_snap_deduplicates_and_skips_blanks() -> None:
    """Two spellings of one term collapse to a single output keyword."""
    vocab = Vocabulary.from_entries(["Osprey"])
    result = vocab.snap(["Osprey", "ospreys", "  ", "OSPREY"])
    assert result.keywords == ["Osprey"]


def test_snap_drops_a_keyword_made_only_of_separators() -> None:
    """A chain that parses to no segments is not a keyword at all."""
    vocab = Vocabulary.from_entries(["Osprey"])
    assert vocab.snap(["<<<"], strict=True).dropped == ["<<<"]


def test_snap_matches_on_the_leaf_of_a_generated_chain() -> None:
    """A chain whose leaf is unknown is unknown, whatever its parents say."""
    vocab = Vocabulary.from_entries(["Animal|Bird"])
    assert vocab.snap(["Tractor<Vehicle"], strict=True).dropped == ["Tractor<Vehicle"]
    assert vocab.snap(["Bird<Something"]).keywords == ["Bird<Animal"]


def test_prompt_section_lists_hierarchies_without_their_prefixes() -> None:
    """Only the deepest path of a branch is listed, parent-first."""
    vocab = Vocabulary.from_entries(parse_keyword_lines(LIGHTROOM_EXPORT))
    section = vocab.prompt_section()
    assert "- Animal > Bird > Osprey" in section
    assert "- Animal > Mammal > Pet" in section
    assert "- Landscape" in section
    assert "- Animal\n" not in section
    assert "- Animal > Bird\n" not in section


def test_prompt_section_truncates_a_large_catalog() -> None:
    """The block is capped and says how much it left out."""
    vocab = Vocabulary.from_entries([f"Term {i}" for i in range(10)])
    section = vocab.prompt_section(max_entries=3)
    assert "- Term 0" in section
    assert "- Term 3" not in section
    assert "... and 7 more" in section


def test_prompt_section_is_empty_for_an_empty_vocabulary() -> None:
    """Nothing to constrain means nothing added to the prompt."""
    assert Vocabulary().prompt_section() == ""


def test_load_vocabulary_reads_a_file(tmp_path: Path) -> None:
    """A real file round-trips into an indexed vocabulary."""
    path = tmp_path / "keywords.txt"
    path.write_text(LIGHTROOM_EXPORT, encoding="utf-8")
    vocab = load_vocabulary(path)
    assert vocab.match("mallard") == "Mallard"
    assert vocab.chain_for("Mallard") == ["Animal", "Bird", "Mallard"]


def test_load_vocabulary_reads_a_lightroom_csv_export(tmp_path: Path) -> None:
    """The CSV export of the same keywords indexes exactly like the txt one."""
    csv_path = tmp_path / "keywords.csv"
    csv_path.write_text(LIGHTROOM_CSV_EXPORT, encoding="utf-8")
    txt_path = tmp_path / "keywords.txt"
    txt_path.write_text(LIGHTROOM_EXPORT, encoding="utf-8")

    from_csv = load_vocabulary(csv_path)
    from_txt = load_vocabulary(txt_path)

    assert from_csv.terms == from_txt.terms
    assert from_csv.chains == from_txt.chains
    assert from_csv.match("sea hawk") == "Osprey"


def test_lightroom_csv_keeps_a_keyword_that_contains_a_comma() -> None:
    """Lightroom leaves such a keyword unquoted, so the row is joined back up, not truncated."""
    text = (
        "Include On Export,Export Containing Keywords,Export Synonyms,Person Type Keyword,\n"
        "Y,Y,Y,N,City\n"
        "Y,Y,Y,N,\tGdansk, Poland\n"
    )
    assert _lightroom_csv_keywords(text) == "City\n\tGdansk, Poland"


def test_lightroom_csv_reads_an_export_without_a_header() -> None:
    """Only a row that names its columns is dropped; a headerless export keeps every keyword."""
    assert _lightroom_csv_keywords("Y,Y,Y,N,Animal\nY,Y,Y,N,\tBird\n") == "Animal\n\tBird"


def test_lightroom_csv_keeps_a_single_letter_keyword() -> None:
    """The last field is never mistaken for one more option column."""
    assert _lightroom_csv_keywords("Y,Y,Y,N,Animal\nY,Y,Y,N,X\n") == "Animal\nX"


@pytest.mark.parametrize(
    "text",
    [
        "Animal\nBird\nOsprey\n",
        "Gdansk, Poland\nLubeck, Germany\n",
        "Animal|Bird|Osprey\n",
        "",
        "Y,Y,Y,N,Animal\n",
        'Y,Y,Y,N,"' + "a" * 200_000,
    ],
    ids=["plain-list", "terms-with-commas", "paths", "empty", "one-row", "unparsable"],
)
def test_lightroom_csv_ignores_anything_that_is_not_one(text: str) -> None:
    """A plain keyword list is left alone, commas in the terms or not, and so is a broken file."""
    assert _lightroom_csv_keywords(text) is None


def test_load_vocabulary_warns_when_the_catalog_outgrows_fuzzy_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the fuzzy limit the vocabulary quietly does less, so say so once at load time."""
    monkeypatch.setattr("photo_tagger.vocabulary._FUZZY_MAX_TERMS", 2)
    path = tmp_path / "keywords.txt"
    path.write_text("Animal\nBird\nOsprey\n", encoding="utf-8")

    events: list[str] = []
    handler = logger.add(lambda message: events.append(message.record["message"]), level="WARNING")
    try:
        load_vocabulary(path)
    finally:
        logger.remove(handler)

    assert "vocabulary_very_large" in events


def test_load_vocabulary_rejects_a_missing_file(tmp_path: Path) -> None:
    """An unreadable path is a clean domain error, not an OSError."""
    with pytest.raises(VocabularyError, match="Could not read"):
        load_vocabulary(tmp_path / "nope.txt")


def test_load_vocabulary_rejects_an_empty_file(tmp_path: Path) -> None:
    """An empty vocabulary would drop every keyword in strict mode, so it fails early."""
    path = tmp_path / "empty.txt"
    path.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(VocabularyError, match="no keywords"):
        load_vocabulary(path)


def test_load_vocabulary_rejects_non_utf8_bytes(tmp_path: Path) -> None:
    """A binary file pointed at by mistake reports the encoding, not a traceback."""
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe\x00\x01Animal")
    with pytest.raises(VocabularyError, match="UTF-8"):
        load_vocabulary(path)


def test_load_vocabulary_rejects_an_oversized_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong (huge) file is refused before it is indexed."""
    monkeypatch.setattr("photo_tagger.vocabulary._MAX_FILE_CHARS", 10)
    path = tmp_path / "big.txt"
    path.write_text("Animal\nBird\nOsprey\nMallard\n", encoding="utf-8")
    with pytest.raises(VocabularyError, match="larger than"):
        load_vocabulary(path)


def test_load_vocabulary_refuses_a_huge_file_without_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit exists so the wrong file never reaches memory, so check the size first."""
    monkeypatch.setattr("photo_tagger.vocabulary._MAX_FILE_CHARS", 10)
    path = tmp_path / "disk-image.bin"
    path.write_bytes(b"x" * 500)

    def refuse_read(*_args: object, **_kwargs: object) -> str:
        message = "the file must be rejected on its size, before any read"
        raise AssertionError(message)

    monkeypatch.setattr(Path, "read_text", refuse_read)

    with pytest.raises(VocabularyError, match="larger than"):
        load_vocabulary(path)


@pytest.mark.parametrize(
    ("plural", "expected"),
    [
        ("berries", "berry"),
        ("churches", "church"),
        ("boxes", "box"),
        ("birds", "bird"),
        ("glass", "glass"),
        ("cactus", "cactus"),
        ("iris", "iris"),
    ],
)
def test_singularize_handles_common_endings(plural: str, expected: str) -> None:
    """The naive plural rules only ever build comparison keys, never output."""
    assert _singularize(plural) == expected


def test_loose_key_normalizes_punctuation_and_case() -> None:
    """Punctuation, spacing, and case never distinguish two terms."""
    assert loose_key("Bird-of-Prey") == loose_key("bird of prey")
    assert loose_key("Bird's Nest") == "bird s nest"


# ---------------------------------------------------------------------------
# Languages other than English
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("English", True),
        ("english", True),
        ("British English", True),
        ("en", True),
        ("German", False),
        ("Brazilian Portuguese", False),
        ("Русский", False),
        ("", False),
    ],
)
def test_folds_plurals_recognizes_english_only(language: str, *, expected: bool) -> None:
    """The plural rules are English morphology, so only English opts into them."""
    assert folds_plurals(language) is expected


def test_german_keeps_words_the_english_plural_rules_would_merge() -> None:
    """German "Alles" is not the plural of "Alle", so the two must stay apart."""
    german = Vocabulary.from_entries(["Alle", "Gras"], fold_plurals=False)

    assert german.match("Alles") is None
    assert german.match("ALLE") == "Alle"
    assert german.match("gras") == "Gras"

    # The same catalog read with the English rules folds a different word onto "Alle".
    english = Vocabulary.from_entries(["Alle", "Gras"])
    assert english.match("Alles") == "Alle"


def test_cyrillic_matches_on_case_and_declared_synonyms() -> None:
    """Case folding is script-neutral; inflections come from the file's own synonyms."""
    russian = Vocabulary.from_entries(
        parse_keyword_lines("Животное\n\tПтица\n\t{птицы}\n\t{птиц}\n"),  # noqa: RUF001
        fold_plurals=False,
    )

    assert russian.match("ПТИЦА") == "Птица"
    assert russian.match("птицы") == "Птица"
    assert russian.match("птиц") == "Птица"
    assert russian.snap(["птицы"]).keywords == ["Птица<Животное"]


def test_loose_key_skips_the_plural_fold_when_asked() -> None:
    """Punctuation and case are still normalized; only the English fold is dropped."""
    assert loose_key("Alles", fold_plurals=False) == "alles"
    assert loose_key("Alles") == "alle"
    # casefold() maps ß to ss, which is Unicode's rule and the useful one here: it unifies the
    # two spellings German itself allows.
    assert loose_key("Straße", fold_plurals=False) == loose_key("Strasse", fold_plurals=False)


def test_load_vocabulary_folds_plurals_only_for_english(tmp_path: Path) -> None:
    """The output language decides how the file is indexed, and is recorded on the result."""
    path = tmp_path / "keywords.txt"
    path.write_text("Alle\nAlles\n", encoding="utf-8")

    german = load_vocabulary(path, output_language="German")
    assert german.fold_plurals is False
    assert german.match("Alles") == "Alles"

    english = load_vocabulary(path)
    assert english.fold_plurals is True


def test_fuzzy_match_measures_the_pair_not_the_query() -> None:
    """Which side is looked up cannot change whether two words are variants."""
    assert fuzzy_key_match("закаты", ["закат"]) == "закат"
    assert fuzzy_key_match("закат", ["закаты"]) == "закаты"
    # Both short: no amount of similarity makes that a safe call.
    assert fuzzy_key_match("alles", ["alle"]) is None
    assert fuzzy_key_match("alle", ["alles"]) is None
    # A different first letter is out regardless of length.
    assert fuzzy_key_match("beagle", ["eagle"]) is None


def test_load_vocabulary_strips_a_byte_order_mark(tmp_path: Path) -> None:
    """A file saved by a Windows editor starts with a BOM; it is not part of the first keyword."""
    listing = tmp_path / "keywords.txt"
    listing.write_bytes(b"\xef\xbb\xbfOsprey\nBird\n")

    vocabulary = load_vocabulary(listing)

    assert vocabulary.terms == ("Osprey", "Bird")
    # Without the strip the match still succeeded but handed back an invisible near-duplicate.
    assert vocabulary.match("ospreys") == "Osprey"


def test_record_dropped_terms_counts_and_stops_at_the_cap() -> None:
    """The tally names the busiest rejects; a run against the wrong file cannot grow it forever."""
    tally: dict[str, int] = {}

    record_dropped_terms(tally, ["Tractor", "Barn", "Tractor"])
    assert tally == {"Tractor": 2, "Barn": 1}

    record_dropped_terms(tally, [f"Term{index}" for index in range(MAX_TRACKED_DROPPED_TERMS)])
    assert len(tally) == MAX_TRACKED_DROPPED_TERMS
    # A term already counted still counts, cap or no cap.
    record_dropped_terms(tally, ["Tractor"])
    assert tally["Tractor"] == 3  # noqa: PLR2004 - two, then one more
