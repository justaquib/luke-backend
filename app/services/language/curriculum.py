"""The static curriculum and the starter lessons built from it."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.services import dictionary
from app.services.language import stage_catalog
from app.services.tokenizer import tokenize

__all__ = ["STAGES", "_GRADE_IDS", "_LEGACY_STAGES"]

_CURRICULUM_PATH = Path(__file__).resolve().parents[2] / "data" / "learner_curriculum.json"
ASSISTANT_BIRTHDAY = date(2026, 10, 19)


def _checked_grade(grade_id: str, grade: object) -> dict:
    """A grade that holds the three lists the lessons are built from."""
    if not isinstance(grade, dict):
        raise ValueError(f"the {grade_id} grade is missing")
    for field in ("sentences", "words", "lessons"):
        if not isinstance(grade.get(field), list):
            raise ValueError(f"{grade_id}.{field} must be a list")
    return grade


def _check_sentences(grade_id: str, sentences: list) -> None:
    for sentence in sentences:
        if not isinstance(sentence, str) or len(tokenize(sentence)) < 2:
            raise ValueError(f"every {grade_id} sentence needs at least two words")


def _check_words(grade_id: str, words: list) -> None:
    for word in words:
        if not isinstance(word, dict) or not word.get("word") or not word.get("meaning"):
            raise ValueError(f"every {grade_id} vocabulary entry needs a word and meaning")


def _check_lessons(grade_id: str, lessons: list) -> None:
    for lesson in lessons:
        if (
            not isinstance(lesson, dict)
            or not lesson.get("question")
            or not lesson.get("answer")
            or not lesson.get("subject")
        ):
            raise ValueError(f"every {grade_id} lesson needs a question, answer, and subject")


def _load_curriculum() -> dict:
    try:
        curriculum = json.loads(_CURRICULUM_PATH.read_text(encoding="utf-8"))
        grades = curriculum.get("grades") if isinstance(curriculum, dict) else None
        if not isinstance(grades, dict):
            raise ValueError("the dataset must contain a grades object")
        for grade_id in ("nursery", "lkg", "ukg"):
            grade = _checked_grade(grade_id, grades.get(grade_id))
            _check_sentences(grade_id, grade["sentences"])
            _check_words(grade_id, grade["words"])
            _check_lessons(grade_id, grade["lessons"])
        return grades
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Unable to load the learner curriculum from {_CURRICULUM_PATH}: {error}"
        ) from error


_CURRICULUM = _load_curriculum()
FOUNDATION_SENTENCES_BY_GRADE = {
    grade_id: list(grade["sentences"]) for grade_id, grade in _CURRICULUM.items()
}
ALPHABET_WORDS = {
    entry["letter"].upper(): entry["word"].strip().lower()
    for entry in _CURRICULUM.get("nursery", {}).get("letter_words", [])
    if isinstance(entry, dict) and entry.get("letter") and entry.get("word")
}


def _curriculum_words() -> dict[str, list[dict]]:
    words_by_grade: dict[str, list[dict]] = {}
    for grade_id, grade in _CURRICULUM.items():
        words_by_grade[grade_id] = []
        for entry in grade["words"]:
            word = entry["word"].strip().lower()
            local_entry = dictionary.lookup_word(word)
            dictionary_meaning = (
                local_entry["definitions"][0]
                if local_entry and local_entry["definitions"]
                else None
            )
            words_by_grade[grade_id].append(
                {
                    "word": word,
                    "meaning": entry["meaning"],
                    "dictionary_meaning": dictionary_meaning,
                    "grade_id": grade_id,
                }
            )
    return words_by_grade


FOUNDATION_WORDS_BY_GRADE = _curriculum_words()
_BASE_FOUNDATION_LESSONS = [
    {**lesson, "grade_id": grade_id, "kind": "starter"}
    for grade_id, grade in _CURRICULUM.items()
    for lesson in grade["lessons"]
] + [
    {
        "question": question,
        "answer": answer,
        "grade_id": "nursery",
        "subject": "Language",
        "kind": "starter",
    }
    for letter, word in ALPHABET_WORDS.items()
    for question, answer in (
        (f"What is {letter} for?", f"{letter} is for {word}."),
        (
            f"What is the spelling of {word}?",
            f"{word.capitalize()} is spelled {'-'.join(word.upper())}.",
        ),
        (f"What letter does {word} start with?", f"{word.capitalize()} starts with {letter}."),
    )
]


def _add(lessons: list[dict], subject: str, question: str, answer: str) -> None:
    lessons.append(
        {
            "question": question,
            "answer": answer,
            "grade_id": "nursery",
            "subject": subject,
            "kind": "starter",
        }
    )


def _alphabet_lessons(lessons: list[dict]) -> None:
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        small = letter.lower()
        answer = f"Capital {letter} is {letter}; small {small} is {small}."
        _add(lessons, "Language", f"What are the capital and small forms of {letter}?", answer)
        _add(lessons, "Language", f"Show capital {letter} and small {small}.", answer)


def _body_lessons(lessons: list[dict], body_parts: list[dict]) -> None:
    if not body_parts:
        return
    names = [item["name"] for item in body_parts]
    _add(lessons, "Body", "Name some body parts.", "Some body parts are " + ", ".join(names) + ".")
    for item in body_parts:
        name = item["name"]
        function = item["function"]
        if function.startswith("It "):
            function_answer = f"The {name} {function[3:]}"
        elif function.startswith("They "):
            function_answer = f"{name.capitalize()} {function[5:]}"
        else:
            function_answer = function
        _add(lessons, "Body", f"What is the {name}?", f"The {name} is a body part. {function}")
        _add(lessons, "Body", f"What does the {name} do?", function_answer)


_CATEGORY_LESSONS = (
    ("animals", "Animals", "Name some animals.", "Some animals are "),
    ("fruits", "Fruits", "Name some fruits.", "Some fruits are "),
    ("vegetables", "Vegetables", "Name some vegetables.", "Some vegetables are "),
)


def _category_lessons(
    lessons: list[dict], category: tuple[str, str, str, str], entries: list[dict]
) -> None:
    if not entries:
        return
    key, subject, question, answer_prefix = category
    names = [item["name"] for item in entries]
    _add(lessons, subject, question, answer_prefix + ", ".join(names) + ".")
    label = {"animals": "animal", "fruits": "fruit", "vegetables": "vegetable"}[key]
    for item in entries:
        name = item["name"]
        article = "an" if name[0].lower() in "aeiou" else "a"
        _add(
            lessons,
            subject,
            f"What is {article} {name}?",
            f"The {name} is {label}. It is {item['description']}.",
        )


def _shape_lessons(lessons: list[dict], shapes: list[dict]) -> None:
    if not shapes:
        return
    _add(
        lessons,
        "Shapes",
        "Name some shapes.",
        "Some shapes are " + ", ".join(item["name"] for item in shapes) + ".",
    )
    for item in shapes:
        name = item["name"]
        _add(lessons, "Shapes", f"Describe a {name}.", f"A {name} is {item['description']}.")
        _add(lessons, "Shapes", f"What is a {name}?", f"A {name} is {item['description']}.")


def _color_lessons(lessons: list[dict], colors: list[dict]) -> None:
    if not colors:
        return
    _add(
        lessons,
        "Colors",
        "Name some colors.",
        "Some colors are " + ", ".join(item["name"] for item in colors) + ".",
    )
    for item in colors:
        color = item["name"]
        example = item["example"]
        _add(lessons, "Colors", f"What color is {example}?", f"{example.capitalize()} is {color}.")
        _add(
            lessons,
            "Colors",
            f"Name an example of something {color}.",
            f"An example of something {color} is {example}.",
        )


def _calendar_lessons(lessons: list[dict], calendar: dict) -> None:
    if not calendar:
        return
    days = calendar.get("days", [])
    months = calendar.get("months", [])
    if days:
        _add(
            lessons,
            "Date & Time",
            "Name the days of the week.",
            "The days of the week are " + ", ".join(days) + ".",
        )
        for index, day in enumerate(days):
            after = days[(index + 1) % len(days)]
            before = days[(index - 1) % len(days)]
            _add(
                lessons,
                "Date & Time",
                f"What day comes after {day}?",
                f"{after} comes after {day}.",
            )
            _add(
                lessons,
                "Date & Time",
                f"What day comes before {day}?",
                f"{before} comes before {day}.",
            )
    if months:
        _add(
            lessons,
            "Date & Time",
            "Name the months of the year.",
            "The months of the year are " + ", ".join(months) + ".",
        )
        for index, month in enumerate(months):
            if index + 1 < len(months):
                _add(
                    lessons,
                    "Date & Time",
                    f"What month comes after {month}?",
                    f"{months[index + 1]} comes after {month}.",
                )
            if index > 0:
                _add(
                    lessons,
                    "Date & Time",
                    f"What month comes before {month}?",
                    f"{months[index - 1]} comes before {month}.",
                )
    for item in calendar.get("facts", []):
        _add(lessons, "Date & Time", item["question"], item["answer"])


def _knowledge_topic_lessons() -> list[dict]:
    topics = _CURRICULUM.get("nursery", {}).get("knowledge_topics", {})
    lessons: list[dict] = []
    _alphabet_lessons(lessons)
    _body_lessons(lessons, topics.get("body_parts", []))
    for category in _CATEGORY_LESSONS:
        _category_lessons(lessons, category, topics.get(category[0], []))
    _shape_lessons(lessons, topics.get("shapes", []))
    _color_lessons(lessons, topics.get("colors", []))
    _calendar_lessons(lessons, topics.get("date_time", {}))
    return lessons


def _unique_lessons(lessons: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for lesson in lessons:
        key = " ".join(lesson["question"].casefold().split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(lesson)
    return unique


FOUNDATION_LESSON_CARDS = _unique_lessons([*_BASE_FOUNDATION_LESSONS, *_knowledge_topic_lessons()])
_FOUNDATION_QUESTIONS = {
    (lesson["question"] or "").strip().lower() for lesson in FOUNDATION_LESSON_CARDS
}
STAGES = stage_catalog.STAGES
_GRADE_IDS = stage_catalog.GRADE_IDS
_LEGACY_STAGES = stage_catalog.LEGACY_STAGES
_DUPLICATE_QUESTION_THRESHOLD = 0.6
_EXAM_PASS_RATE = 0.8


def _allowed_grades(stage_id: str | None) -> set[str]:
    """Every grade up to and including the current one.

    A learner keeps answering from the grades it has already passed, and the
    current grade's own starter lessons are available too.
    """
    order = [item["id"] for item in STAGES]
    if stage_id == "complete":
        return set(order)
    if stage_id not in order:
        stage_id = "nursery"
    current_index = order.index(stage_id)
    return {grade_id for grade_id in order if order.index(grade_id) <= current_index}


def _foundation_sentences_for_stage(stage_id: str | None) -> list[str]:
    allowed = _allowed_grades(stage_id)
    return [
        sentence
        for grade_id, sentences in FOUNDATION_SENTENCES_BY_GRADE.items()
        if grade_id in allowed
        for sentence in sentences
    ]


def _foundation_words_for_stage(stage_id: str | None) -> list[dict]:
    allowed = _allowed_grades(stage_id)
    return [
        word
        for grade_id, words in FOUNDATION_WORDS_BY_GRADE.items()
        if grade_id in allowed
        for word in words
    ]


def _grammar_lessons() -> list[dict]:
    """Every grammar and sentence-making lesson in the curriculum.

    The writing model learns from all of them, so it can write a proper
    sentence even before the learner reaches that grade.
    """
    return [
        lesson
        for lesson in FOUNDATION_LESSON_CARDS
        if lesson.get("subject") in {"English Grammar", "Sentence Making", "Punctuation"}
    ]


def _tenant_words_for_stage(state: dict, stage_id: str | None) -> list[dict]:
    allowed = _allowed_grades(stage_id)
    return [word for word in state["word_cards"] if word.get("grade_id", "nursery") in allowed]


def _foundation_lessons_for_stage(stage_id: str | None) -> list[dict]:
    allowed = _allowed_grades(stage_id)
    return [lesson for lesson in FOUNDATION_LESSON_CARDS if lesson["grade_id"] in allowed]
