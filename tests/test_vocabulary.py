"""The one vocabulary.

The transformer's token list is the only live token→id mapping. The old
mappings are frozen artefacts and must never produce ids again.
"""

import json

import pytest

from app.services import brain, language_logic, vocabulary


def test_the_canonical_vocabulary_is_the_transformers():
    canonical = vocabulary.canonical(None)

    assert canonical.owner == vocabulary.CANONICAL_OWNER
    assert len(canonical) > 0
    # The ids are exactly the places of the tokens.
    assert len(canonical.index) == len(canonical)
    for position, token in enumerate(canonical.tokens):
        assert canonical.id_of(token) == position
        assert canonical.token_of(position) == token


def test_the_transformer_uses_the_canonical_vocabulary():
    writer = brain.load_writer(None)
    if writer is None:
        pytest.skip("no transformer has been trained")

    assert writer.vocabulary == list(vocabulary.canonical(None).tokens)
    assert writer.index == vocabulary.canonical(None).index


def test_a_retired_vocabulary_cannot_produce_ids():
    for name in vocabulary.retired_names():
        old = vocabulary.retired(name)

        assert old.retired is True
        with pytest.raises(vocabulary.VocabularyError):
            old.encode("the cow is an animal")


def test_the_retired_artefacts_are_frozen_and_marked():
    for name in vocabulary.retired_names():
        stored = json.loads((vocabulary.LEGACY_DIRECTORY / name).read_text(encoding="utf-8"))

        assert stored["retired"] is True
        assert stored["size"] == len(stored["tokens"])
        assert "not used at inference" in stored["note"].lower()


def test_the_guard_refuses_an_id_outside_the_canonical_vocabulary():
    canonical = vocabulary.canonical(None)

    with pytest.raises(vocabulary.VocabularyError):
        vocabulary.guard_tensor_ids([len(canonical)], "a test")
    with pytest.raises(vocabulary.VocabularyError):
        vocabulary.guard_tensor_ids([-1], "a test")
    assert vocabulary.guard_tensor_ids([0], "a test") == [0]


def test_the_startup_guard_passes():
    report = vocabulary.assert_single_vocabulary(None)

    assert report["owner"] == vocabulary.CANONICAL_OWNER
    assert report["retired"] == vocabulary.retired_names()


def test_the_old_word_list_is_not_fully_inside_the_transformers():
    """The gap that has to be closed before the old list can be encoded again.

    The old learner knew some words the transformer was never trained on, so
    those ids cannot be produced. The guard is what keeps them out.
    """
    canonical = vocabulary.canonical(None)
    old = vocabulary.retired("trigram_vocab_1091.json")

    missing = [token for token in old.tokens if token not in canonical]

    # The old list holds words the transformer was never trained on, so those
    # ids cannot be produced. How many there are changes with the corpus, so
    # the gap is checked, not counted.
    assert missing
    # The two mappings are not interchangeable: a word that is in both does
    # not always carry the same id. The ids overlap, so a number on its own
    # cannot be told apart, and the guard is what refuses a retired mapping.
    shared = [token for token in old.tokens if token in canonical]
    assert shared
    different = [token for token in shared if old.id_of(token) != canonical.id_of(token)]
    assert different, "the two mappings would be interchangeable"
    # The retired mapping itself is refused, so a mixed id can never be built.
    with pytest.raises(vocabulary.VocabularyError):
        vocabulary.guard_tensor_ids([0], "a test", old)


def test_the_sidebar_list_is_the_canonical_vocabulary():
    page = language_logic.vocabulary_page(None, 0, 10)
    canonical = vocabulary.canonical(None)

    assert page["total"] == len(canonical)
    assert page["words"] == list(canonical.tokens[:10])
    assert page["has_more"] is (len(canonical) > 10)


def test_the_sidebar_list_is_paged():
    canonical = vocabulary.canonical(None)
    last = language_logic.vocabulary_page(None, len(canonical) - 3, 100)

    assert last["words"] == list(canonical.tokens[-3:])
    assert last["has_more"] is False


def test_the_brain_reports_the_canonical_vocabulary():
    state = brain.state(None)

    assert state["vocabulary"] == len(vocabulary.canonical(None))


def test_the_ladder_reads_the_canonical_vocabulary():
    state = brain.state(None)
    capabilities = brain.model_capabilities(None, state, {"lessons": 1, "passed_grades": 0})
    recalled = next(item for item in capabilities if item["key"] == "recalled")

    assert recalled["detail"] == f"{len(vocabulary.canonical(None))} words it can write with"
