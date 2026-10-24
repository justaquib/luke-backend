"""The one answer order every page of the app uses.

The chat, the learner preview, the grade tests and the coaching chat all ask
the same learner the same question. Keeping the order in one place means a
question cannot be answered from the lessons on one page and refused on
another. A page may add its own sources around this answer (documents, the
weather, a coaching writer), but the taught material itself is read here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.services import dictionary
from app.services.language import runtime
from app.services.language.alphabet import _answer_alphabet_question
from app.services.language.basics import (
    _answer_counting_question,
    _answer_number_sequence_question,
    _answer_number_spelling_question,
    answer_assistant_age,
    answer_current_date_time,
)
from app.services.language.calculations import _answer_calculation_question
from app.services.language.conversation import answer_conversation
from app.services.language.curriculum import (
    FOUNDATION_LESSON_CARDS,
    STAGES,
    _allowed_grades,
    _foundation_sentences_for_stage,
    _foundation_words_for_stage,
    _tenant_words_for_stage,
)
from app.services.language.grammar import apply_requested_count
from app.services.language.lessons import all_lesson_cards
from app.services.language.matching import _matched_learned_lesson
from app.services.language.patterns import _NEXT_WORD_REQUEST, _SENTENCE_REQUEST
from app.services.language.writing import _answer_from_sentence_examples

NOT_LEARNED = (
    "I haven't learned an answer to that yet. Teach me the word, sentence, or "
    "question and answer, then ask again."
)

SOURCE_DETAILS = {
    "assistant_profile": "Assistant profile",
    "dictionary": "Offline dictionary",
    "faq": "Trainer-approved answer",
    "learner": "Learner curriculum",
    "lesson_cards": "Learned lesson",
    "letters": "Letter learning",
    "not_learned": "Not learned yet",
    "number_cards": "Number cards",
    "sentence_examples": "Learned sentence examples",
    "sentence_patterns": "Learned sentence patterns",
    "taught": "Taught lesson",
    "trainer": "Trainer-approved answer",
    "word_cards": "Learned word",
    "written": "Written from a lesson",
}

_QUESTION_STARTS = frozenset(
    {
        "ask",
        "can",
        "could",
        "define",
        "describe",
        "did",
        "do",
        "does",
        "explain",
        "give",
        "how",
        "is",
        "list",
        "name",
        "should",
        "tell",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "why",
        "will",
        "would",
        "write",
    }
)


def is_question_or_request(text: str) -> bool:
    """Whether this turn is asking the learner for knowledge."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    first = re.match(r"[A-Za-z']+", stripped)
    return bool(first and first.group(0).lower() in _QUESTION_STARTS)


@dataclass(frozen=True)
class TaughtAnswer:
    """What the learner answered, and where the answer was read from."""

    answer: str
    source: str
    matched_question: str = ""

    @property
    def detail(self) -> str:
        """The plain-language name of the source."""
        return source_detail(self.source)

    def as_dict(self) -> dict[str, str]:
        """The shape the learner routes and their components use."""
        return {
            "answer": self.answer,
            "source": self.source,
            "matched_question": self.matched_question,
        }


def source_detail(source: str) -> str:
    """The plain-language name of an answer source."""
    return SOURCE_DETAILS.get(source, source.replace("_", " ").capitalize())


def current_stage(state: dict) -> str:
    """The first grade the learner has not passed."""
    return next(
        (item["id"] for item in STAGES if item["id"] not in state["passed_stages"]),
        "complete",
    )


def asked_word(question: str) -> str | None:
    """The word a "what does X mean" question asks about."""
    q = question.strip().lower().rstrip("?.! ")
    patterns = (
        r"what\s+does\s+(?:the\s+word\s+)?([a-z][a-z'’-]*)(?:\s+mean)?",
        r"what\s+is\s+(?:the\s+meaning\s+of\s+)?([a-z][a-z'’-]*)",
        r"meaning\s+of\s+([a-z][a-z'’-]*)",
        r"define\s+([a-z][a-z'’-]*)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q)
        if match:
            return match.group(1).casefold()
    return None


def answer_letter_question(question: str, alphabet: str) -> str | None:
    """The letter after or before a letter, or the first letter of a word."""
    normalized = question.strip().lower().rstrip("?.! ")
    match = re.fullmatch(
        r"(?:what\s+)?(?:letter\s+)?comes\s+(after|before)\s+(?:the\s+)?(?:letter\s+)?([a-z])|"
        r"what\s+letter\s+comes\s+(after|before)\s+([a-z])",
        normalized,
    )
    if match:
        direction = match.group(1) or match.group(3)
        letter = match.group(2) or match.group(4)
        index = alphabet.find(letter)
        target = index + (1 if direction == "after" else -1)
        if index >= 0 and 0 <= target < len(alphabet):
            return f"The letter {direction} {letter.upper()} is {alphabet[target].upper()}."
    if dictionary.is_dictionary_question(question) and re.search(
        r"\b(?:start|begin|first\s+letter)\b", question, re.I
    ):
        return dictionary.answer_dictionary_question(question)
    return None


def _quick_answer(question: str, state: dict, tenant_id: int | None = None) -> TaughtAnswer | None:
    """The facts the learner knows without a lesson lookup."""
    from app.services import model_settings

    conversational = answer_conversation(question, model_settings.name_for(tenant_id))
    if conversational:
        return TaughtAnswer(answer=conversational, source="assistant_profile")
    steps: tuple[tuple[Callable[[str], str | None], str], ...] = (
        (_answer_calculation_question, "number_cards"),
        (answer_assistant_age, "assistant_profile"),
        (answer_current_date_time, "lesson_cards"),
        (_answer_number_spelling_question, "number_cards"),
        (_answer_counting_question, "number_cards"),
        (_answer_number_sequence_question, "number_cards"),
        (_answer_alphabet_question, "lesson_cards"),
    )
    for answer_of, source in steps:
        answer = answer_of(question)
        if answer:
            return TaughtAnswer(answer=answer, source=source)
    answer = answer_letter_question(question, state["letters"])
    if answer:
        return TaughtAnswer(answer=answer, source="letters")
    return None


def _word_answer(question: str, state: dict, stage: str) -> TaughtAnswer | None:
    """The meaning of a taught word."""
    word = asked_word(question)
    if not word:
        return None
    for item in [*_tenant_words_for_stage(state, stage), *_foundation_words_for_stage(stage)]:
        if item["word"].casefold() == word:
            return TaughtAnswer(
                answer=f"{item['word'].capitalize()} means {item['meaning']}.",
                source="word_cards",
                matched_question=f"What does {item['word']} mean?",
            )
    return None


def _lesson_answer(
    question: str, state: dict, stage: str, include_lessons: bool
) -> TaughtAnswer | None:
    """The lesson that covers the question, in the grades the learner has reached."""
    if not include_lessons:
        return None
    allowed = _allowed_grades(stage)
    lessons = [
        item
        for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
        if item.get("grade_id", "nursery") in allowed
    ]
    matched = _matched_learned_lesson(question, lessons)
    if not matched:
        return None
    return TaughtAnswer(
        answer=apply_requested_count(question, matched["answer"]),
        source="lesson_cards",
        matched_question=matched.get("question", ""),
    )


def _request_answer(question: str, tenant_id: int | None) -> TaughtAnswer | None:
    """A request for something to be written, answered from a taught lesson."""
    from app.services import brain
    from app.services.language import requests as request_service

    try:
        cards = all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        return None
    result = request_service.answer_request(
        question, tenant_id, cards, writer=brain.load_writer(tenant_id)
    )
    if result is None:
        return None
    return TaughtAnswer(answer=result["answer"], source=result["source"])


def _generated_answer(question: str, state: dict, stage: str) -> TaughtAnswer | None:
    """The dictionary, the word model and the learned sentence examples."""
    dictionary_answer = dictionary.answer_dictionary_question(question)
    if dictionary_answer:
        return TaughtAnswer(answer=dictionary_answer, source="dictionary")
    next_match = _NEXT_WORD_REQUEST.search(question.strip())
    if next_match:
        prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
        options = runtime._model(state, stage).predict_next(prefix)
        if options:
            return TaughtAnswer(
                answer=(
                    f"From my sentence lessons, after “{prefix}” I predict “{options[0]['word']}”."
                ),
                source="sentence_patterns",
            )
    sentence_match = _SENTENCE_REQUEST.search(question.strip())
    if sentence_match:
        requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-")
        answer = (
            runtime._model(state, stage).generate_sentence(seed=requested) if requested else None
        )
        if answer:
            return TaughtAnswer(answer=answer, source="sentence_patterns")
    examples = [
        *(item["text"] for item in state["lessons"]),
        *_foundation_sentences_for_stage(stage),
    ]
    answer = _answer_from_sentence_examples(question, examples)
    if answer:
        return TaughtAnswer(answer=answer, source="sentence_examples")
    return None


def answer_from_state(
    question: str,
    state: dict,
    stage: str | None = None,
    tenant_id: int | None = None,
    include_lessons: bool = True,
) -> TaughtAnswer | None:
    """The taught answer to a question, read in the same order everywhere."""
    grade = stage or current_stage(state)
    steps = (
        lambda: _quick_answer(question, state, tenant_id),
        lambda: _word_answer(question, state, grade),
        lambda: _lesson_answer(question, state, grade, include_lessons),
        lambda: _request_answer(question, tenant_id),
        lambda: _generated_answer(question, state, grade),
    )
    for find_answer in steps:
        answer = find_answer()
        if answer is not None:
            return answer
    return None


def answer_from_curriculum(
    question: str,
    tenant_id: int | None,
    stage: str | None = None,
    include_lessons: bool = True,
) -> TaughtAnswer | None:
    """Read the saved learner and answer from the taught material."""
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        return answer_from_state(question, state, stage, tenant_id, include_lessons)
