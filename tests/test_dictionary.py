from types import SimpleNamespace

from app.services import dictionary


class FakeLemma:
    def __init__(self, name: str, antonyms: list["FakeLemma"] | None = None) -> None:
        self._name = name
        self._antonyms = antonyms or []

    def name(self) -> str:
        return self._name

    def antonyms(self) -> list["FakeLemma"]:
        return self._antonyms


class FakeSynset:
    def __init__(
        self, pos: str, definition: str, names: list[str], antonym: str | None = None
    ) -> None:
        opposite = [FakeLemma(antonym)] if antonym else []
        self._lemmas = [FakeLemma(name, opposite) for name in names]
        self._pos = pos
        self._definition = definition

    def definition(self) -> str:
        return self._definition

    def pos(self) -> str:
        return self._pos

    def lemmas(self) -> list[FakeLemma]:
        return self._lemmas


def test_first_letter_question_needs_no_wordnet():
    assert dictionary.answer_dictionary_question("What letter does apple start with?") == (
        "Apple starts with the letter A."
    )
    assert dictionary.answer_dictionary_question("Apple starts with which alphabet?") == (
        "Apple starts with the letter A."
    )


def test_next_letter_questions_need_no_wordnet():
    assert dictionary.answer_dictionary_question("What comes after letter D?") == (
        "The letter after D is E."
    )
    assert dictionary.answer_dictionary_question("Which letter comes after Z?") == (
        "There is no letter after Z in the English alphabet."
    )


def test_definition_returns_multiple_senses(monkeypatch):
    monkeypatch.setattr(
        dictionary,
        "_unique_synsets",
        lambda word: [
            FakeSynset("n", "the fruit of an apple tree", ["apple"]),
            FakeSynset("n", "the tree bearing apples", ["apple_tree"]),
        ],
    )
    answer = dictionary.answer_dictionary_question("What does apple mean?")
    assert answer == (
        "Apple has multiple meanings. For example: the fruit of an apple tree; "
        "it can also mean the tree bearing apples"
    )


def test_synonyms_and_antonyms(monkeypatch):
    synset = FakeSynset("a", "having a pleasant taste", ["sweet", "sugary"], "bitter")
    monkeypatch.setattr(dictionary, "_unique_synsets", lambda word: [synset])
    assert (
        dictionary.answer_dictionary_question("Synonyms of sweet") == "Synonyms of sweet: sugary."
    )
    assert (
        dictionary.answer_dictionary_question("Give me some synonyms for sweet")
        == "Synonyms of sweet: sugary."
    )
    assert (
        dictionary.answer_dictionary_question("Antonyms of sweet") == "Antonyms of sweet: bitter."
    )
    assert (
        dictionary.answer_dictionary_question("What is the opposite of sweet")
        == "Antonyms of sweet: bitter."
    )


def test_part_of_speech_and_missing_relations(monkeypatch):
    monkeypatch.setattr(
        dictionary,
        "_unique_synsets",
        lambda word: [
            FakeSynset("n", "a fruit", ["apple"]),
            FakeSynset("v", "to apple-polish", ["apple"]),
        ],
    )
    assert dictionary.answer_dictionary_question("What part of speech is apple?") == (
        '"apple" can be used as a noun or verb.'
    )

    monkeypatch.setattr(
        dictionary, "_unique_synsets", lambda word: [FakeSynset("n", "a fruit", ["apple"])]
    )
    assert dictionary.answer_dictionary_question("Antonyms of apple") == (
        'I couldn\'t find antonyms for "apple" in the offline English dictionary.'
    )


def test_unsupported_question_falls_through(monkeypatch):
    monkeypatch.setattr(dictionary, "_unique_synsets", lambda word: [SimpleNamespace()])
    assert (
        dictionary.answer_dictionary_question("What is the customer name in this voucher?") is None
    )
    assert dictionary.answer_dictionary_question("What is the capital of France?") is None
