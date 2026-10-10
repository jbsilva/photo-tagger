"""Tests for the optional model pass that folds synonyms and assigns a hierarchy."""

from collections import Counter
from itertools import pairwise
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from photo_tagger.vocabulary import Vocabulary, parse_keyword_lines
from photo_tagger.vocabulary_build import (
    DROP_SYNONYM,
    KeywordCensus,
    TrimResult,
    TrimRules,
    render_drop_report,
    render_vocabulary,
    trim,
)
from photo_tagger.vocabulary_organize import (
    _MAX_CATEGORIES,
    CategoryList,
    KeywordGroup,
    OrganizedChunk,
    _apply_groups,
    _chunks,
    _request,
    choose_categories,
    organize,
)


def _result(*terms: str) -> TrimResult:
    """Build a trimmed result holding *terms*, each used once, with no hierarchy."""
    census = KeywordCensus()
    for term in terms:
        census.add([term])
    return TrimResult(kept=list(terms), census=census)


def _fake_replies(categories: list[str], groups: list[KeywordGroup]) -> Any:  # noqa: ANN401
    """Patch the two model calls: first the categories, then every chunk."""

    def run(chat_model: object, output_type: object, system_prompt: str, prompt: str) -> object:
        del chat_model, output_type, system_prompt, prompt
        run.calls += 1  # type: ignore[attr-defined]
        if run.calls == 1:  # type: ignore[attr-defined]
            return CategoryList(categories=categories)
        return OrganizedChunk(groups=groups)

    run.calls = 0  # type: ignore[attr-defined]
    return patch("photo_tagger.vocabulary_organize._request", side_effect=run)


def _stub_model() -> Any:  # noqa: ANN401
    """Stand in for the chat model; every test patches the request that would use it."""
    return cast("Any", object())


def _no_provider() -> Any:  # noqa: ANN401
    """Stub out the provider handshake; these tests never reach a real model."""
    return patch("photo_tagger.vocabulary_organize.build_chat_model", return_value=object())


def test_chunks_split_the_same_way_every_time() -> None:
    """Fixed-size chunks off a sorted list are what keep the pass reproducible."""
    assert _chunks(["a", "b", "c"], size=2) == [["a", "b"], ["c"]]


def test_apply_groups_folds_synonyms_and_assigns_a_category() -> None:
    """The two judgments the pass exists to make, on a well-behaved reply."""
    applied = _apply_groups(
        [KeywordGroup(preferred="Golden Hour", synonyms=["Golden Light"], category="Lighting")],
        ["Golden Hour", "Golden Light"],
        ["Lighting"],
    )
    assert applied.synonyms == {"Golden Hour": ["Golden Light"]}
    assert applied.aliased == {"Golden Light": "Golden Hour"}
    assert applied.chains == {"Golden Hour": ("Lighting", "Golden Hour")}


def test_apply_groups_folds_plurals_only_where_the_trim_did() -> None:
    """In German, "Alle" and "Alles" are two keywords, so a reply naming one is not the other."""
    applied = _apply_groups(
        [KeywordGroup(preferred="Alle", synonyms=[], category="Lighting")],
        ["Alle", "Alles"],
        ["Lighting"],
        fold_plurals=False,
    )
    assert applied.chains == {"Alle": ("Lighting", "Alle")}


def test_trim_hands_its_plural_rule_to_the_organize_pass() -> None:
    """Organize() reads the rule off the result, so neither frontend has to pass it twice."""
    census = KeywordCensus(uses=Counter({"Alle": 3, "Alles": 3}))

    assert trim(census, TrimRules(min_uses=1, fold_plurals=False)).fold_plurals is False


def test_apply_groups_refuses_a_keyword_the_model_invented() -> None:
    """A model that writes its own keywords must not get them into the catalog."""
    applied = _apply_groups(
        [
            KeywordGroup(preferred="Waterfowl", synonyms=["Duck"], category="Animal"),
            KeywordGroup(preferred="Duck", category="Animal"),
        ],
        ["Duck", "Goose"],
        ["Animal"],
    )
    assert applied.invented == 1
    assert applied.aliased == {}
    assert applied.chains == {"Duck": ("Animal", "Duck")}


def test_apply_groups_matches_loosely_but_writes_the_catalog_spelling() -> None:
    """A retyped 'golden hour' still resolves, and resolves to the library's own spelling."""
    applied = _apply_groups(
        [KeywordGroup(preferred="golden hour", synonyms=["GOLDEN-LIGHT"])],
        ["Golden Hour", "Golden Light"],
        [],
    )
    assert applied.synonyms == {"Golden Hour": ["Golden Light"]}


def test_apply_groups_ignores_a_category_that_was_not_offered() -> None:
    """Categories are fixed up front; a chunk cannot invent one of its own."""
    applied = _apply_groups(
        [KeywordGroup(preferred="Osprey", category="Birds Of Prey")],
        ["Osprey"],
        ["Animal"],
    )
    assert applied.chains == {}


def test_apply_groups_gives_a_keyword_to_the_first_group_that_claims_it() -> None:
    """Whatever order the reply arrives in, one keyword lands in exactly one group."""
    applied = _apply_groups(
        [
            KeywordGroup(preferred="Bird", synonyms=["Osprey"]),
            KeywordGroup(preferred="Osprey", synonyms=["Bird"]),
        ],
        ["Bird", "Osprey"],
        [],
    )
    assert applied.synonyms == {"Bird": ["Osprey"]}
    assert applied.aliased == {"Osprey": "Bird"}


def test_apply_groups_does_not_file_a_category_under_itself() -> None:
    """A keyword that is also a category is a root: it has no hierarchy to write."""
    applied = _apply_groups(
        [KeywordGroup(preferred="Animal", category="Animal")],
        ["Animal"],
        ["Animal"],
    )
    assert applied.chains == {}


def test_choose_categories_dedups_and_cleans_the_reply() -> None:
    """Categories arrive as prose sometimes; they are normalized before anything uses them."""
    with patch(
        "photo_tagger.vocabulary_organize._request",
        return_value=CategoryList(categories=["Animal", "animal!", "  ", "Landscape."]),
    ):
        assert choose_categories(_stub_model(), ["Bird"]) == ["Animal", "Landscape"]


def test_choose_categories_survives_a_failed_call() -> None:
    """No categories is a usable answer: the pass then only folds synonyms."""
    with patch("photo_tagger.vocabulary_organize._request", return_value=None):
        assert choose_categories(_stub_model(), ["Bird"]) == []


def test_organize_folds_synonyms_into_the_written_file() -> None:
    """End to end: the loser is not deleted, it is written as a synonym that still matches."""
    groups = [KeywordGroup(preferred="Golden Hour", synonyms=["Golden Light"], category="Lighting")]
    with _no_provider(), _fake_replies(["Lighting"], groups):
        result, stats = organize(
            _result("Golden Hour", "Golden Light"),
            provider_name="lmstudio",
            model_name="test-model",
            api_base_url=None,
            api_key=None,
        )

    assert result.kept == ["Golden Hour"]
    assert stats.grouped == 1
    assert stats.categorized == 1

    rendered = render_vocabulary(result)
    assert rendered == "Lighting|Golden Hour {Golden Light}\n"
    reloaded = Vocabulary.from_entries(parse_keyword_lines(rendered))
    assert reloaded.match("Golden Light") == "Golden Hour"
    assert reloaded.chain_for("Golden Hour") == ["Lighting", "Golden Hour"]


def test_organize_reports_each_fold_in_the_drop_report() -> None:
    """A folded keyword is accounted for, not quietly missing from the file."""
    groups = [KeywordGroup(preferred="Golden Hour", synonyms=["Golden Light"])]
    with _no_provider(), _fake_replies([], groups):
        result, _ = organize(
            _result("Golden Hour", "Golden Light"),
            provider_name="lmstudio",
            model_name="test-model",
            api_base_url=None,
            api_key=None,
        )

    assert [(d.term, d.reason, d.detail) for d in result.dropped] == [
        ("Golden Light", DROP_SYNONYM, "alias of Golden Hour"),
    ]
    assert "Golden Light,1,synonym,alias of Golden Hour" in render_drop_report(result)


def test_organize_leaves_the_list_alone_when_the_model_fails() -> None:
    """A provider that errors costs the organizing, not the keywords."""
    with _no_provider(), patch("photo_tagger.vocabulary_organize._request", return_value=None):
        result, stats = organize(
            _result("Bird", "Osprey"),
            provider_name="lmstudio",
            model_name="test-model",
            api_base_url=None,
            api_key=None,
        )

    assert result.kept == ["Bird", "Osprey"]
    assert stats.grouped == 0
    assert stats.failed_chunks == 1
    assert render_vocabulary(result) == "Bird\nOsprey\n"


def test_a_chunk_the_model_answered_with_no_groups_is_not_a_failure() -> None:
    """
    Nothing to group is a result, not an error.

    failed_chunks used to be inferred from "produced no synonyms and no chains", which is exactly
    what a chunk the model answered correctly with every keyword standing alone looks like. That
    made the count useless for telling a provider outage from a tidy catalog.
    """
    with _no_provider(), _fake_replies([], []):
        _result_out, stats = organize(
            _result("Bird", "Osprey"),
            provider_name="lmstudio",
            model_name="test-model",
            api_base_url=None,
            api_key=None,
        )

    assert stats.failed_chunks == 0


def test_organize_with_nothing_to_organize_does_not_call_the_model() -> None:
    """An empty list short-circuits before the provider handshake."""
    with patch("photo_tagger.vocabulary_organize.build_chat_model") as build:
        result, stats = organize(
            TrimResult(),
            provider_name="lmstudio",
            model_name="test-model",
            api_base_url=None,
            api_key=None,
        )

    build.assert_not_called()
    assert result.kept == []
    assert stats.categories == []


@pytest.mark.parametrize("workers", [1, 4])
def test_organize_is_unchanged_by_how_many_workers_run_the_chunks(workers: int) -> None:
    """Chunks are reassembled by position, so concurrency cannot reorder the file."""
    groups = [KeywordGroup(preferred="Bird", synonyms=["Birdie"], category="Animal")]
    with _no_provider(), _fake_replies(["Animal"], groups):
        result, _ = organize(
            _result("Bird", "Birdie"),
            provider_name="lmstudio",
            model_name="test-model",
            api_base_url=None,
            api_key=None,
            workers=workers,
        )

    assert render_vocabulary(result) == "Animal|Bird {Birdie}\n"


def test_request_returns_the_models_output() -> None:
    """The happy path: whatever the agent decoded is what the caller gets."""
    agent = MagicMock()
    agent.run_sync.return_value = MagicMock(output=CategoryList(categories=["Animal"]))

    with patch("photo_tagger.vocabulary_organize.Agent", return_value=agent):
        output = _request(_stub_model(), CategoryList, "system", "prompt")

    assert output == CategoryList(categories=["Animal"])
    settings = agent.run_sync.call_args.kwargs["model_settings"]
    assert settings["temperature"] == 0.0


def test_request_swallows_a_provider_failure() -> None:
    """A dead provider must degrade to "leave the keywords alone", not raise through the run."""
    agent = MagicMock()
    agent.run_sync.side_effect = RuntimeError("connection refused")

    with patch("photo_tagger.vocabulary_organize.Agent", return_value=agent):
        assert _request(_stub_model(), CategoryList, "system", "prompt") is None


def test_apply_groups_files_a_narrower_keyword_under_its_parent() -> None:
    """The relation a model reaches for when it has no parent field: kept, not folded away."""
    applied = _apply_groups(
        [
            KeywordGroup(preferred="Camera Accessories", category="Equipment"),
            KeywordGroup(preferred="Battery Pack", parent="Camera Accessories"),
        ],
        ["Camera Accessories", "Battery Pack"],
        ["Equipment"],
    )
    assert applied.aliased == {}
    assert applied.chains["Battery Pack"] == ("Equipment", "Camera Accessories", "Battery Pack")


def test_apply_groups_breaks_a_parent_cycle() -> None:
    """Models state parentage both ways round in one reply; the walk must still terminate."""
    applied = _apply_groups(
        [
            KeywordGroup(preferred="Camera", parent="Camera Body"),
            KeywordGroup(preferred="Camera Body", parent="Camera"),
        ],
        ["Camera", "Camera Body"],
        [],
    )
    assert applied.chains["Camera"] == ("Camera Body", "Camera")
    assert applied.chains["Camera Body"] == ("Camera", "Camera Body")


def test_apply_groups_caps_how_deep_a_hierarchy_goes() -> None:
    """A model asked for parents will build a taxonomy; a keyword panel has to stay readable."""
    terms = ["A", "B", "C", "D", "E", "F"]
    applied = _apply_groups(
        [
            KeywordGroup(preferred="A", category="Root"),
            *(KeywordGroup(preferred=b, parent=a) for a, b in pairwise(terms)),
        ],
        terms,
        ["Root"],
    )
    assert applied.chains["F"] == ("C", "D", "E", "F")


def test_apply_groups_moves_a_parent_onto_the_keyword_that_kept_it() -> None:
    """A parent folded into another keyword resolves to the one that kept it, not to nothing."""
    applied = _apply_groups(
        [
            KeywordGroup(preferred="Camera", synonyms=["Digital Camera"]),
            KeywordGroup(preferred="Strap", parent="Digital Camera"),
        ],
        ["Camera", "Digital Camera", "Strap"],
        [],
    )
    assert applied.chains["Strap"] == ("Camera", "Strap")


def test_apply_groups_ignores_a_parent_the_model_invented() -> None:
    """Same rail as everywhere else: a parent must be a keyword the library really uses."""
    applied = _apply_groups(
        [KeywordGroup(preferred="Osprey", parent="Raptor")],
        ["Osprey"],
        [],
    )
    assert applied.chains == {}


def test_apply_groups_refuses_a_group_that_swallows_a_category() -> None:
    """A group claiming Person, Human, Woman and Man is not a synonym set, it is a category."""
    applied = _apply_groups(
        [KeywordGroup(preferred="People", synonyms=["Person", "Human", "Woman", "Man"])],
        ["People", "Person", "Human", "Woman", "Man"],
        [],
    )
    assert applied.aliased == {}
    assert applied.synonyms == {}
    assert applied.refused_groups == 1


def test_apply_groups_still_takes_a_synonym_set_of_a_believable_size() -> None:
    """The cap refuses the implausible, not the ordinary."""
    applied = _apply_groups(
        [KeywordGroup(preferred="Golden Hour", synonyms=["Golden Light", "Warm Light"])],
        ["Golden Hour", "Golden Light", "Warm Light"],
        [],
    )
    assert applied.synonyms == {"Golden Hour": ["Golden Light", "Warm Light"]}
    assert applied.refused_groups == 0


def test_choose_categories_refuses_a_reply_that_echoes_the_keyword_list() -> None:
    """A model ignoring "6 to 20" echoes the list; 20 of those are noise, not a tree."""
    echoed = CategoryList(categories=[f"Keyword {index}" for index in range(60)])
    with patch("photo_tagger.vocabulary_organize._request", return_value=echoed):
        assert choose_categories(_stub_model(), ["Bird"]) == []


def test_choose_categories_keeps_a_reply_that_merely_runs_over() -> None:
    """A few too many is a long answer, not a broken one: it is trimmed, not thrown away."""
    over = CategoryList(categories=[f"Category {index}" for index in range(25)])
    with patch("photo_tagger.vocabulary_organize._request", return_value=over):
        categories = choose_categories(_stub_model(), ["Bird"])
    assert len(categories) == _MAX_CATEGORIES


def test_request_asks_for_no_reasoning() -> None:
    """Grouping words is recall, not deduction, and deliberation costs minutes per chunk."""
    agent = MagicMock()
    agent.run_sync.return_value = MagicMock(output=CategoryList(categories=["Animal"]))

    with patch("photo_tagger.vocabulary_organize.Agent", return_value=agent):
        _request(_stub_model(), CategoryList, "system", "prompt")

    assert agent.run_sync.call_args.kwargs["model_settings"]["openai_reasoning_effort"] == "none"
