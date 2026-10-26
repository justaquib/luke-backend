"""The one answer order every page of the app uses.

The chat, the learner preview, the grade tests and the coaching chat all ask
the same learner the same question. Keeping the order in one place means a
question cannot be answered from the lessons on one page and refused on
another. A page may add its own sources around this answer (documents, the
weather), but the taught material itself is read here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from app.services import dictionary
from app.services.connections import Connection
from app.services.language import fallback as fallback_service
from app.services.language import finishing, runtime
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
from app.services.language.writing import _answer_from_sentence_examples, write_pattern_answer

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
    if first and first.group(0).lower() in _QUESTION_STARTS:
        return True
    from app.services.language import requests as request_service

    return request_service.parse_request(stripped) is not None


@dataclass(frozen=True)
class TaughtAnswer:
    """What the learner answered, and where the answer was read from."""

    answer: str
    source: str
    matched_question: str = ""
    evidence: tuple[Connection, ...] = ()
    confidence: float | None = None

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


def _evidenced(
    answer: str,
    source: str,
    detail: str,
    matched_question: str = "",
    confidence: float | None = None,
) -> TaughtAnswer:
    """A taught answer together with the connection that found it."""
    return TaughtAnswer(
        answer=answer,
        source=source,
        matched_question=matched_question,
        evidence=(Connection(source, "match", detail, matched_question),),
        confidence=confidence,
    )


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
        return _evidenced(
            conversational, "assistant_profile", "The assistant's own profile answers this."
        )
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
            return _evidenced(answer, source, f"{source_detail(source)} answers this.")
    answer = answer_letter_question(question, state["letters"])
    if answer:
        return _evidenced(answer, "letters", "The learner's letter cards answer this.")
    return None


def _word_answer(question: str, state: dict, stage: str) -> TaughtAnswer | None:
    """The meaning of a taught word."""
    word = asked_word(question)
    if not word:
        return None
    for item in [*_tenant_words_for_stage(state, stage), *_foundation_words_for_stage(stage)]:
        if item["word"].casefold() == word:
            return _evidenced(
                f"{item['word'].capitalize()} means {item['meaning']}.",
                "word_cards",
                f'"{item["word"]}" is a taught word.',
                f"What does {item['word']} mean?",
            )
    return None


def _allowed_lessons(state: dict, stage: str, include_lessons: bool = True) -> list[dict]:
    """The lessons the learner has reached, when lessons may answer."""
    if not include_lessons:
        return []
    allowed = _allowed_grades(stage)
    return [
        item
        for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
        if item.get("grade_id", "nursery") in allowed
    ]


def _lesson_answer(
    question: str, state: dict, stage: str, include_lessons: bool
) -> TaughtAnswer | None:
    """The lesson that covers the question, in the grades the learner has reached."""
    lessons = _allowed_lessons(state, stage, include_lessons)
    matched = _matched_learned_lesson(question, lessons)
    if not matched:
        return None
    matched_question = str(matched.get("question") or "")
    stored = matched.get("confidence")
    return _evidenced(
        apply_requested_count(question, matched["answer"]),
        "lesson_cards",
        f"Lesson “{matched_question}” covers the question.",
        matched_question,
        float(stored) if stored is not None else None,
    )


def _request_answer(question: str, tenant_id: int | None) -> TaughtAnswer | None:
    """A request for something to be written, answered from a taught lesson."""
    from app.services.language import requests as request_service

    try:
        cards = all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        return None
    result = request_service.answer_request(question, tenant_id, cards)
    if result is None:
        return None
    kind = str(result.get("kind") or "taught")
    return _evidenced(
        result["answer"],
        result["source"],
        f"The request was answered from the taught {kind} lesson.",
    )


def _generated_answer(
    question: str, state: dict, stage: str, include_lessons: bool = True
) -> TaughtAnswer | None:
    """The dictionary, the word model and the learned sentence examples."""
    dictionary_answer, found = dictionary.answer_dictionary_question_with_status(question)
    if dictionary_answer:
        # A definition question may name something the trainer taught under a
        # different word. The synonym is read from WordNet, and the lesson is
        # served only when it is the one lesson reached and it shares a key
        # word with the question; otherwise the dictionary answer stands.
        attempt = fallback_service.find_fallback(
            question, _allowed_lessons(state, stage, include_lessons)
        )
        if attempt is not None and attempt.answer is not None:
            return TaughtAnswer(
                answer=attempt.answer.answer,
                source="lesson_cards",
                matched_question=attempt.answer.matched_question,
                evidence=(attempt.connection,),
            )
        if found:
            evidence = (
                Connection(
                    "dictionary",
                    "match",
                    "The offline dictionary has an entry for the question word.",
                ),
            )
            if attempt is not None:
                evidence += (attempt.connection,)
            return TaughtAnswer(answer=dictionary_answer, source="dictionary", evidence=evidence)
        return TaughtAnswer(
            answer=dictionary_answer,
            source="dictionary",
            evidence=(attempt.connection,) if attempt is not None else (),
        )
    pattern = write_pattern_answer(question, state, stage)
    if pattern is not None:
        answer, detail = pattern
        return _evidenced(answer, "sentence_patterns", detail)
    examples = [
        *(item["text"] for item in state["lessons"]),
        *_foundation_sentences_for_stage(stage),
    ]
    answer = _answer_from_sentence_examples(question, examples)
    if answer:
        return _evidenced(
            answer,
            "sentence_examples",
            "A taught sentence example answers the question.",
        )
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
    from app.services.language import requests as request_service

    request = request_service.parse_request(question)
    # A request for a poem or a list is not a question a lesson answers: a
    # lesson about poems would otherwise answer a request for one.
    request_first = request is not None and request.kind != "fact"
    steps = [
        lambda: _quick_answer(question, state, tenant_id),
        lambda: _word_answer(question, state, grade),
    ]
    if request_first:
        steps.append(lambda: _request_answer(question, tenant_id))
    steps.append(lambda: _lesson_answer(question, state, grade, include_lessons))
    if not request_first:
        steps.append(lambda: _request_answer(question, tenant_id))
    steps.append(lambda: _generated_answer(question, state, grade, include_lessons))
    for find_answer in steps:
        answer = find_answer()
        if answer is not None:
            return replace(answer, answer=finishing.finish(answer.answer))
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
