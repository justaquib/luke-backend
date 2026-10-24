"""Compatibility surface for the learner's language modules."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from app.config import settings
from app.services import dictionary
from app.services.language import runtime as _runtime
from app.services.language.alphabet import _answer_alphabet_question
from app.services.language.answering import answer_letter_question as _answer_letter_question
from app.services.language.basics import (
    _answer_counting_question,
    _answer_number_sequence_question,
    _answer_number_spelling_question,
    _number_name,
    _number_value,
    answer_assistant_age,
    answer_current_date_time,
)
from app.services.language.calculations import (
    _answer_calculation_question,
    _calculation_result,
    _format_number,
)
from app.services.language.corpus import training_corpus
from app.services.language.curriculum import (
    ALPHABET_WORDS,
    ASSISTANT_BIRTHDAY,
    FOUNDATION_LESSON_CARDS,
    FOUNDATION_SENTENCES_BY_GRADE,
    FOUNDATION_WORDS_BY_GRADE,
    STAGES,
)
from app.services.language.grammar import (
    _is_naming_word,
    _repair_sentence,
    _repair_sentences,
    _repair_with_document,
    _sentence_starts_with,
    apply_requested_count,
)
from app.services.language.helpers import (
    _content_terms,
    _foundation_lesson_id,
    _normalize_phrase,
    _parse_lesson_pairs,
    _parse_word_pairs,
    _phrase_similarity,
    _same_words,
    _stem,
    _token_matches,
    _word_by_word_similarity,
    _words_of,
)
from app.services.language.lessons import (
    _clamp_confidence,
    _existing_lesson_cards,
    all_lesson_cards,
    bump_lesson_confidence,
    check_duplicate_lessons,
    find_known_answer,
    find_lesson_by_answer,
    get_lesson_card,
    match_lesson_by_words,
    remove_lesson,
    tenant_ids_with_lessons,
    update_lesson_answer,
)
from app.services.language.matching import (
    _answer_learned_lesson,
    _matched_learned_lesson,
)
from app.services.language.paraphrase import (
    _fix_article,
    _predicates_of,
    _subject_of,
    _wording_score,
    paraphrase_answer,
)
from app.services.language.patterns import (
    _COUNT_REQUEST,
    _COUNT_WORDS,
    _NEXT_NUMBER_REQUEST,
    _NEXT_WORD_REQUEST,
    _QUESTION_STOP_WORDS,
    _SENTENCE_REQUEST,
)
from app.services.language.preview import (
    answer_from_curriculum,
    answer_from_curriculum_detail,
    preview_answer,
)
from app.services.language.runtime import (
    _model,
    _public_test,
    _read,
    _save,
    _tenant_key,
    get_state,
    parse_subjects,
    subject_of,
    vocabulary_page,
)
from app.services.language.stages import (
    add_dictionary_word,
    ask_test_question,
    finish_test,
    score_test_answer,
    start_test,
)
from app.services.language.teaching import (
    StoryLesson,
    teach,
    teach_story,
)
from app.services.language.writing import (
    _answer_from_sentence_examples,
    _is_describing_word,
    _lesson_statement_for_word,
    _list_statement_for_word,
    _statement_from_lesson,
    _taught_answers,
    _taught_sentences,
    _taught_word_meanings,
    _tenant_lesson_answers,
    answer_next_word_request,
    answer_sentence_request,
    make_sentence,
    make_sentence_from_words,
    predict_next,
)
from app.services.tokenizer import tokenize

_RUNTIME_NAMES = {"_DATA_DIR", "_CACHE", "_LOCK"}

__all__ = [
    "ALPHABET_WORDS",
    "ASSISTANT_BIRTHDAY",
    "FOUNDATION_LESSON_CARDS",
    "FOUNDATION_SENTENCES_BY_GRADE",
    "FOUNDATION_WORDS_BY_GRADE",
    "STAGES",
    "_COUNT_REQUEST",
    "_COUNT_WORDS",
    "_NEXT_NUMBER_REQUEST",
    "_NEXT_WORD_REQUEST",
    "_QUESTION_STOP_WORDS",
    "_SENTENCE_REQUEST",
    "StoryLesson",
    "_answer_alphabet_question",
    "_answer_calculation_question",
    "_answer_counting_question",
    "_answer_from_sentence_examples",
    "_answer_learned_lesson",
    "_answer_letter_question",
    "_answer_number_sequence_question",
    "_answer_number_spelling_question",
    "_calculation_result",
    "_clamp_confidence",
    "_content_terms",
    "_existing_lesson_cards",
    "_fix_article",
    "_format_number",
    "_foundation_lesson_id",
    "_is_describing_word",
    "_is_naming_word",
    "_lesson_statement_for_word",
    "_list_statement_for_word",
    "_matched_learned_lesson",
    "_model",
    "_normalize_phrase",
    "_number_name",
    "_number_value",
    "_parse_lesson_pairs",
    "_parse_word_pairs",
    "_phrase_similarity",
    "_predicates_of",
    "_public_test",
    "_read",
    "_repair_sentence",
    "_repair_sentences",
    "_repair_with_document",
    "_same_words",
    "_save",
    "_sentence_starts_with",
    "_statement_from_lesson",
    "_stem",
    "_subject_of",
    "_taught_answers",
    "_taught_sentences",
    "_taught_word_meanings",
    "_tenant_key",
    "_tenant_lesson_answers",
    "_token_matches",
    "_word_by_word_similarity",
    "_wording_score",
    "_words_of",
    "add_dictionary_word",
    "all_lesson_cards",
    "answer_assistant_age",
    "answer_current_date_time",
    "answer_from_curriculum",
    "answer_from_curriculum_detail",
    "answer_next_word_request",
    "answer_sentence_request",
    "apply_requested_count",
    "ask_test_question",
    "bump_lesson_confidence",
    "check_duplicate_lessons",
    "dictionary",
    "find_known_answer",
    "find_lesson_by_answer",
    "finish_test",
    "get_lesson_card",
    "get_state",
    "make_sentence",
    "make_sentence_from_words",
    "match_lesson_by_words",
    "paraphrase_answer",
    "parse_subjects",
    "predict_next",
    "preview_answer",
    "remove_lesson",
    "score_test_answer",
    "settings",
    "start_test",
    "subject_of",
    "teach",
    "teach_story",
    "tenant_ids_with_lessons",
    "tokenize",
    "training_corpus",
    "update_lesson_answer",
    "vocabulary_page",
]


class _LanguageLogicModule(ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name in _RUNTIME_NAMES:
            return getattr(_runtime, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RUNTIME_NAMES:
            setattr(_runtime, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LanguageLogicModule
