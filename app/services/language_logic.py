"""Learning loop and safe, tenant-scoped persistence for the tiny language model."""

from __future__ import annotations

import json
import math
import re
import threading
import hashlib
from uuid import uuid4
from datetime import date, datetime, timezone
from pathlib import Path

from app.config import settings
from app.services import dictionary
from app.services.tiny_language_model import TinyLanguageModel, tokenize


_CURRICULUM_PATH = Path(__file__).resolve().parents[1] / "data" / "learner_curriculum.json"
ASSISTANT_BIRTHDAY = date(2026, 10, 19)


def answer_assistant_age(question: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.casefold())
    normalized = " ".join(normalized.split())
    if not re.search(r"\b(?:your age|how old are you|what age are you|old are you)\b", normalized):
        return None
    today = date.today()
    years = today.year - ASSISTANT_BIRTHDAY.year - ((today.month, today.day) < (ASSISTANT_BIRTHDAY.month, ASSISTANT_BIRTHDAY.day))
    if years == 0:
        return "My birthday is 19 October 2026, so I’m 0 years old today—I was born today!" if today == ASSISTANT_BIRTHDAY else "My birthday is 19 October 2026, so I’m not born yet."
    return f"My birthday is 19 October 2026, so I’m {years} year{'s' if years != 1 else ''} old."


def answer_current_date_time(question: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.casefold().replace("’", "").replace("'", ""))
    normalized = " ".join(normalized.split())
    now = datetime.now().astimezone()
    if re.search(r"\b(?:what time is it|what is the time|whats the time|tell me the time|current time|time right now|time now|what time is it now)\b", normalized):
        time_text = now.strftime("%I:%M %p").lstrip("0")
        timezone_name = now.tzname()
        suffix = f" {timezone_name}" if timezone_name else ""
        return f"It is {time_text}{suffix} on the app’s local clock."
    if re.search(r"\b(?:what month is it|what is the current month|current month)\b", normalized):
        return f"It is {now.strftime('%B')}."
    if re.search(r"\b(?:what year is it|what is the current year|current year)\b", normalized):
        return f"It is {now.year}."
    if re.search(r"\b(?:what is today|what day is it|what day is today|what is the date|what date is it|what date is today|whats todays date|what is todays date|todays date|current date|todays day)\b", normalized):
        date_text = now.strftime("%A, %d %B %Y").replace(" 0", " ")
        return f"Today is {date_text}."
    return None


def _load_curriculum() -> dict:
    try:
        curriculum = json.loads(_CURRICULUM_PATH.read_text(encoding="utf-8"))
        grades = curriculum.get("grades") if isinstance(curriculum, dict) else None
        if not isinstance(grades, dict):
            raise ValueError("the dataset must contain a grades object")
        for grade_id in ("nursery", "lkg", "ukg"):
            grade = grades.get(grade_id)
            if not isinstance(grade, dict):
                raise ValueError(f"the {grade_id} grade is missing")
            for field in ("sentences", "words", "lessons"):
                if not isinstance(grade.get(field), list):
                    raise ValueError(f"{grade_id}.{field} must be a list")
            for sentence in grade["sentences"]:
                if not isinstance(sentence, str) or len(tokenize(sentence)) < 2:
                    raise ValueError(f"every {grade_id} sentence needs at least two words")
            for word in grade["words"]:
                if not isinstance(word, dict) or not word.get("word") or not word.get("meaning"):
                    raise ValueError(f"every {grade_id} vocabulary entry needs a word and meaning")
            for lesson in grade["lessons"]:
                if not isinstance(lesson, dict) or not lesson.get("question") or not lesson.get("answer") or not lesson.get("subject"):
                    raise ValueError(f"every {grade_id} lesson needs a question, answer, and subject")
        return grades
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Unable to load the learner curriculum from {_CURRICULUM_PATH}: {error}") from error


_CURRICULUM = _load_curriculum()
FOUNDATION_SENTENCES_BY_GRADE = {grade_id: list(grade["sentences"]) for grade_id, grade in _CURRICULUM.items()}
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
            dictionary_meaning = (local_entry["definitions"][0] if local_entry and local_entry["definitions"] else None)
            words_by_grade[grade_id].append({
                "word": word,
                "meaning": entry["meaning"],
                "dictionary_meaning": dictionary_meaning,
                "grade_id": grade_id,
            })
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
        (f"What is the spelling of {word}?", f"{word.capitalize()} is spelled {'-'.join(word.upper())}."),
        (f"What letter does {word} start with?", f"{word.capitalize()} starts with {letter}."),
    )
]


def _knowledge_topic_lessons() -> list[dict]:
    topics = _CURRICULUM.get("nursery", {}).get("knowledge_topics", {})
    lessons: list[dict] = []

    def add(subject: str, question: str, answer: str) -> None:
        lessons.append({
            "question": question,
            "answer": answer,
            "grade_id": "nursery",
            "subject": subject,
            "kind": "starter",
        })

    # Uppercase and lowercase forms are taught as explicit question-answer pairs.
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        small = letter.lower()
        answer = f"Capital {letter} is {letter}; small {small} is {small}."
        add("Language", f"What are the capital and small forms of {letter}?", answer)
        add("Language", f"Show capital {letter} and small {small}.", answer)

    body_parts = topics.get("body_parts", [])
    if body_parts:
        names = [item["name"] for item in body_parts]
        add("Body", "Name some body parts.", "Some body parts are " + ", ".join(names) + ".")
        for item in body_parts:
            name = item["name"]
            function = item["function"]
            if function.startswith("It "):
                function_answer = f"The {name} {function[3:]}"
            elif function.startswith("They "):
                function_answer = f"{name.capitalize()} {function[5:]}"
            else:
                function_answer = function
            add("Body", f"What is the {name}?", f"The {name} is a body part. {function}")
            add("Body", f"What does the {name} do?", function_answer)

    for key, subject, category_question, category_answer in (
        ("animals", "Animals", "Name some animals.", "Some animals are "),
        ("fruits", "Fruits", "Name some fruits.", "Some fruits are "),
        ("vegetables", "Vegetables", "Name some vegetables.", "Some vegetables are "),
    ):
        entries = topics.get(key, [])
        names = [item["name"] for item in entries]
        if entries:
            add(subject, category_question, category_answer + ", ".join(names) + ".")
            for item in entries:
                name = item["name"]
                article = "an" if name[0].lower() in "aeiou" else "a"
                label = {"animals": "animal", "fruits": "fruit", "vegetables": "vegetable"}[key]
                add(subject, f"What is {article} {name}?", f"The {name} is {label}. It is {item['description']}.")

    shapes = topics.get("shapes", [])
    if shapes:
        add("Shapes", "Name some shapes.", "Some shapes are " + ", ".join(item["name"] for item in shapes) + ".")
        for item in shapes:
            name = item["name"]
            add("Shapes", f"Describe a {name}.", f"A {name} is {item['description']}.")
            add("Shapes", f"What is a {name}?", f"A {name} is {item['description']}.")

    colors = topics.get("colors", [])
    if colors:
        add("Colors", "Name some colors.", "Some colors are " + ", ".join(item["name"] for item in colors) + ".")
        for item in colors:
            color = item["name"]
            example = item["example"]
            add("Colors", f"What color is {example}?", f"{example.capitalize()} is {color}.")
            add("Colors", f"Name an example of something {color}.", f"An example of something {color} is {example}.")

    calendar = topics.get("date_time", {})
    if calendar:
        days = calendar.get("days", [])
        months = calendar.get("months", [])
        if days:
            add("Date & Time", "Name the days of the week.", "The days of the week are " + ", ".join(days) + ".")
            for index, day in enumerate(days):
                after = days[(index + 1) % len(days)]
                before = days[(index - 1) % len(days)]
                add("Date & Time", f"What day comes after {day}?", f"{after} comes after {day}.")
                add("Date & Time", f"What day comes before {day}?", f"{before} comes before {day}.")
        if months:
            add("Date & Time", "Name the months of the year.", "The months of the year are " + ", ".join(months) + ".")
            for index, month in enumerate(months):
                if index + 1 < len(months):
                    add("Date & Time", f"What month comes after {month}?", f"{months[index + 1]} comes after {month}.")
                if index > 0:
                    add("Date & Time", f"What month comes before {month}?", f"{months[index - 1]} comes before {month}.")
        for item in calendar.get("facts", []):
            add("Date & Time", item["question"], item["answer"])

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
_FOUNDATION_QUESTIONS = {(lesson["question"] or "").strip().lower() for lesson in FOUNDATION_LESSON_CARDS}
STAGES = [
    {"id": "nursery", "title": "Nursery", "description": "First ideas: letters, numbers, colors, shapes, feelings, greetings, and the world around us.", "subjects": ["Language", "Numbers", "Shapes", "Colors", "Feelings & expressions", "My World", "Punctuation"]},
    {"id": "lkg", "title": "Lower Kindergarten (LKG)", "description": "Build early reading, counting, drawing, and everyday knowledge.", "subjects": ["Language", "Math", "Shapes", "Nature", "Daily Life", "English Grammar", "Sentence Making", "Tenses", "Calculation"]},
    {"id": "ukg", "title": "Upper Kindergarten (UKG)", "description": "Practise simple reading, number operations, patterns, and observation.", "subjects": ["Language", "Math", "Patterns", "Science", "Social Skills", "English Grammar", "Sentence Making", "Calculation", "Numbers"]},
    *[{"id": f"class_{grade}", "title": f"Class {grade}", "description": f"Build school knowledge and lessons for Class {grade}.", "subjects": ["Language", "Math", "Science", "Social Studies", "Arts"]} for grade in range(1, 13)],
    {"id": "graduation", "title": "Graduation", "description": "Add undergraduate-level lessons by subject and course.", "subjects": ["Course", "Subject", "Research", "Professional Skills"]},
]
_GRADE_IDS = {item["id"] for item in STAGES}
_LEGACY_STAGES = {"letters", "words", "sentences", "lessons"}
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
    return [sentence for grade_id, sentences in FOUNDATION_SENTENCES_BY_GRADE.items() if grade_id in allowed for sentence in sentences]


def _foundation_words_for_stage(stage_id: str | None) -> list[dict]:
    allowed = _allowed_grades(stage_id)
    return [word for grade_id, words in FOUNDATION_WORDS_BY_GRADE.items() if grade_id in allowed for word in words]


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

_DATA_DIR = Path(__file__).resolve().parents[2] / ".language_model"
_LOCK = threading.RLock()
_CACHE: dict[int, dict] = {}
_SENTENCE_REQUEST = re.compile(
    r"\b(?:make|write|create|give)\s+(?:me\s+)?(?:a|one)\s+sentence\s+(?:using|with|that\s+includes)\s+(.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_NEXT_WORD_REQUEST = re.compile(
    r"\b(?:what|which)\s+(?:word\s+)?(?:comes\s+)?(?:next\s+)?(?:after|follows)\s+(.+?)\s*[?.!]*$|\bpredict\s+the\s+next\s+word\s+after\s+(.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_NUMBER_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS_NAMES = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def _number_value(raw: str) -> int | None:
    token = raw.casefold().strip()
    if token.isdecimal():
        return int(token)
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    if "-" in token:
        parts = token.split("-", 1)
        if len(parts) == 2 and parts[0] in _TENS_NAMES and parts[1] in _NUMBER_WORDS and 1 <= _NUMBER_WORDS[parts[1]] <= 9:
            return _NUMBER_WORDS[parts[1]] + int(next(key for key, name in _TENS_NAMES.items() if name == parts[0]))
    return None


def _number_name(value: int) -> str:
    if value == 100:
        return "one hundred"
    if value < 20:
        return _NUMBER_NAMES[value]
    tens, ones = divmod(value, 10)
    return _TENS_NAMES[tens * 10] + (f"-{_NUMBER_NAMES[ones]}" if ones else "")


_NEXT_NUMBER_REQUEST = re.compile(
    r"\b(?:what|which)\s+(?:number\s+)?(?:comes\s+)?(?:next\s+)?(?:after|follows)\s+([\w-]+)"
    r"|\b(?:what|which)\s+(?:number\s+)?(?:comes\s+)?(?:next\s+)?(?:before|precedes)\s+([\w-]+)",
    re.IGNORECASE,
)


def _answer_number_sequence_question(question: str) -> str | None:
    """Answer "what comes after 20?" for any number, not only the taught ones."""
    match = _NEXT_NUMBER_REQUEST.search(question.strip())
    if not match:
        return None
    after_token, before_token = match.group(1), match.group(2)
    value = _number_value(after_token or before_token or "")
    if value is None or not (0 <= value <= 100):
        return None
    if after_token:
        return f"The number after {value} is {value + 1}."
    return f"The number before {value} is {value - 1}."


def _answer_counting_question(question: str) -> str | None:
    q = question.strip().casefold().rstrip("?.! ")
    match = re.fullmatch(r"count\s+(?:numbers\s+)?(?:from\s+)?([\w-]+)\s*(?:to|[-–])\s*([\w-]+)", q)
    if match:
        start_token, end_token = match.group(1), match.group(2)
        start, end = (_number_value(start_token), _number_value(end_token))
    else:
        match = re.fullmatch(r"count\s+(?:numbers\s+)?(?:up\s+)?to\s+([\w-]+)", q)
        if not match:
            return None
        start, end_token = 1, match.group(1)
        start_token = "one"
        end = _number_value(end_token)
    if start is None or end is None or not (0 <= start <= 100 and 0 <= end <= 100) or abs(end - start) > 100:
        return None
    step = 1 if end >= start else -1
    use_words = start_token.isalpha() and end_token.isalpha()
    render = _number_name if use_words else str
    sequence = ", ".join(render(value) for value in range(start, end + step, step))
    return f"Counting from {render(start)} to {render(end)}: {sequence}."


def _answer_number_spelling_question(question: str) -> str | None:
    q = question.strip().casefold().rstrip("?.! ")
    patterns = (
        r"spell\s+(?:the\s+)?(?:number\s+)?([\w-]+)",
        r"(?:write|say)\s+([\w-]+)\s+in\s+words",
        r"what\s+is\s+([\w-]+)\s+in\s+words",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q)
        if match:
            value = _number_value(match.group(1))
            if value is not None and 0 <= value <= 100:
                return f"{_number_name(value)}."
    return None


def _tenant_key(tenant_id: int | None) -> int:
    return tenant_id if tenant_id is not None else 0


def _path(tenant_id: int | None) -> Path:
    return _DATA_DIR / f"tenant-{_tenant_key(tenant_id)}.json"


def _read(tenant_id: int | None) -> dict:
    key = _tenant_key(tenant_id)
    if key in _CACHE:
        return _CACHE[key]
    path = _path(tenant_id)
    lessons: list[dict] = []
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            lessons = stored.get("lessons", []) if isinstance(stored, dict) else []
        except (OSError, json.JSONDecodeError):
            lessons = []
    state = {
        "lessons": lessons,
        "letters": "abcdefghijklmnopqrstuvwxyz",
        "word_cards": [],
        "lesson_cards": [],
        "passed_stages": [],
        "active_test": None,
        "test_history": [],
    }
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                for name in ("letters", "word_cards", "lesson_cards", "passed_stages", "active_test", "test_history"):
                    if name in stored:
                        state[name] = stored[name]
        except (OSError, json.JSONDecodeError):
            pass
    _CACHE[key] = state
    assigned_lesson_ids = False
    for lesson in state["lesson_cards"]:
        if not lesson.get("id"):
            lesson["id"] = uuid4().hex
            assigned_lesson_ids = True
        topics = lesson.get("subjects")
        if not isinstance(topics, list) or not topics:
            # A lesson taught before topics were split keeps the subject it had.
            lesson["subjects"] = parse_subjects(lesson.get("subject"))
            lesson["subject"] = ", ".join(lesson["subjects"])
            assigned_lesson_ids = True
    if assigned_lesson_ids:
        _save(tenant_id, state)
    active_test = state.get("active_test")
    if active_test and active_test.get("stage") not in _GRADE_IDS:
        active_test["stage"] = "nursery"
    return state


def _save(tenant_id: int | None, state: dict) -> None:
    path = _path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _model(state: dict, stage_id: str | None = None) -> TinyLanguageModel:
    if stage_id is None:
        stage_id = next((stage["id"] for stage in STAGES if stage["id"] not in state["passed_stages"]), "complete")
    foundation_words = _foundation_words_for_stage(stage_id)
    foundation_lessons = _foundation_lessons_for_stage(stage_id)
    allowed_grades = _allowed_grades(stage_id)
    tenant_lessons = [lesson for lesson in state["lesson_cards"] if lesson.get("grade_id", "nursery") in allowed_grades]
    word_cards = [*foundation_words, *_tenant_words_for_stage(state, stage_id)]
    # All the grammar lessons are taught to the writing model, so it knows how
    # a sentence is put together even before the learner reaches that grade.
    grammar_lessons = _grammar_lessons()
    lesson_sentences = [
        f"{item['question']} {item['answer']}"
        for item in [*grammar_lessons, *foundation_lessons, *tenant_lessons]
    ]
    # Every answer is also a sentence on its own, so the model learns how a
    # proper sentence is put together.
    answer_sentences = [
        item["answer"]
        for item in [*grammar_lessons, *foundation_lessons, *tenant_lessons]
        if item.get("answer")
    ]
    sentences = _foundation_sentences_for_stage(stage_id)
    examples = [
        " ".join(state["letters"]),
        *sentences,
        *answer_sentences,
        *(f"{item['word']} means {item['meaning']}." for item in word_cards),
        *(f"{item['word']}: {item['dictionary_meaning']}." for item in foundation_words if item.get("dictionary_meaning") and item["dictionary_meaning"].casefold() != item["meaning"].casefold()),
        *(lesson["text"] for lesson in state["lessons"]),
        *lesson_sentences,
    ]
    return TinyLanguageModel(examples)


def _stage(user_lesson_count: int, model: TinyLanguageModel) -> str:
    if user_lesson_count == 0:
        return "sentence patterns"
    if len(model.vocabulary) < 30:
        return "growing vocabulary"
    if user_lesson_count < 20:
        return "learning sentence patterns"
    return "practising sentence generation"


def get_state(tenant_id: int | None, prefix: str = "") -> dict:
    from app.services import vocabulary as vocabulary_service

    with _LOCK:
        state = _read(tenant_id)
        passed = set(state["passed_stages"])
        current_stage = next((stage["id"] for stage in STAGES if stage["id"] not in passed), "complete")
        model = _model(state, current_stage)
        available_grades = _allowed_grades(current_stage)
        available_sentences = _foundation_sentences_for_stage(current_stage)
        available_words = _foundation_words_for_stage(current_stage)
        available_lessons = _foundation_lessons_for_stage(current_stage)
        lessons = [
            {"text": text, "kind": "foundation"}
            for text in available_sentences
        ] + [
            {"text": lesson["text"], "kind": "trainer", "created_at": lesson.get("created_at")}
            for lesson in reversed(state["lessons"])
        ]
        stage_counts = {
            stage["id"]: sum(
                1 for lesson in state["lesson_cards"]
                if lesson.get("grade_id", "nursery") == stage["id"]
            ) + sum(1 for lesson in FOUNDATION_LESSON_CARDS if lesson["grade_id"] == stage["id"])
            for stage in STAGES
        }
        attempt_counts = {stage["id"]: 0 for stage in STAGES}
        pass_counts = {stage["id"]: 0 for stage in STAGES}
        for result in state["test_history"]:
            grade_id = result.get("stage")
            if grade_id in attempt_counts:
                attempt_counts[grade_id] += 1
                pass_counts[grade_id] += int(bool(result.get("passed")))
        return {
            "model_name": "Tiny Word Trigram",
            "description": "A small from-scratch learner with Nursery, LKG, and UKG starter lessons, sentence examples, and a curated vocabulary seeded from the local dictionary. The trainer decides when it is ready to advance.",
            "stage": next((item["title"] for item in STAGES if item["id"] == current_stage), "All stages passed"),
            "current_stage_id": current_stage,
            "passed_stages": [stage["id"] for stage in STAGES if stage["id"] in passed],
            "stages": [
                {**stage, "lesson_count": stage_counts[stage["id"]], "completed": stage["id"] in passed,
                 "attempt_count": attempt_counts[stage["id"]], "pass_count": pass_counts[stage["id"]]}
                for stage in STAGES
            ],
            "letters": state["letters"],
            "word_cards": [*_tenant_words_for_stage(state, current_stage), *available_words],
            "lesson_cards": [
                *[
                    {
                        "kind": "starter" if (lesson.get("question") or "").strip().lower() in _FOUNDATION_QUESTIONS else "trainer",
                        **lesson,
                        "subject": subject_of(lesson),
                        "subjects": parse_subjects(subject_of(lesson)),
                        "confidence": float(lesson.get("confidence", settings.base_answer_confidence)),
                    }
                    for lesson in reversed(state["lesson_cards"])
                    if lesson.get("grade_id", "nursery") in available_grades
                ],
                *available_lessons,
            ],
            "active_test": _public_test(state.get("active_test")),
            "test_history": list(reversed(state["test_history"][-20:])),
            "foundation_lessons": len(available_sentences),
            "trainer_lessons": len(state["lessons"]),
            "sentence_count": model.sentence_count,
            "token_count": model.token_count,
            "vocabulary_size": len(vocabulary_service.canonical(tenant_id)),
            "transition_count": model.transition_count,
            "next_words": model.predict_next(prefix) if prefix.strip() else [],
            "examples": lessons[:80],
            "generated_example": model.generate_sentence(),
        }


def vocabulary_page(tenant_id: int | None, offset: int = 0, limit: int = 100) -> dict:
    """The words the answering model can write with, one page at a time.

    This is the transformer's vocabulary: the one live token list. A long
    list is not sent all at once, so it can be scrolled and the next words
    arrive when they are needed.
    """
    from app.services import vocabulary as vocabulary_service

    words = list(vocabulary_service.canonical(tenant_id).tokens)
    offset = max(0, offset)
    limit = min(200, max(1, limit))
    page = words[offset:offset + limit]
    return {
        "words": page,
        "total": len(words),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(words),
    }


def _normalize_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _phrase_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_phrase(left).split())
    right_tokens = set(_normalize_phrase(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _words_of(text: str) -> set[str]:
    """The words of a text, one by one.

    Only the punctuation and the case are dropped. Every word is kept, because
    one word is enough to make a different question: the opposite of "in" is
    not the opposite of "loud".
    """
    return set(_normalize_phrase(text).split())


def _same_words(left: str, right: str) -> bool:
    """Two texts that say the same thing, word by word."""
    left_words = _words_of(left)
    right_words = _words_of(right)
    return bool(left_words) and left_words == right_words


def _word_by_word_similarity(left: str, right: str) -> float:
    """How close two questions are, compared word by word.

    The same words mean the same question, whatever order they are in. One
    extra word is still the same question. A word that is different makes it a
    different question, so it is not reported as something already known.
    """
    left_words = _words_of(left)
    right_words = _words_of(right)
    if not left_words or not right_words:
        return 0.0
    smaller, larger = (left_words, right_words) if len(left_words) <= len(right_words) else (right_words, left_words)
    if not smaller <= larger:
        return 0.0
    if len(larger) - len(smaller) > 1:
        return 0.0
    return 1.0 if len(larger) == len(smaller) else 0.9


def _parse_lesson_pairs(raw: str) -> list[tuple[str, str]]:
    pairs = re.findall(r"(?ims)^\s*q\s*:\s*(.+?)\s*\n\s*a\s*:\s*(.+?)(?=\n\s*q\s*:|\Z)", raw)
    return [
        (question.strip(), answer.strip())
        for question, answer in pairs
        if question.strip() and answer.strip()
    ]


def _parse_word_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if ":" not in line:
            continue
        word, meaning = line.split(":", 1)
        if word.strip() and meaning.strip():
            pairs.append((word.strip().lower(), meaning.strip()))
    return pairs


def _existing_lesson_cards(state: dict, stage: str) -> list[dict]:
    cards = [card for card in state.get("lesson_cards", []) if card.get("question")]
    if stage in _GRADE_IDS:
        for lesson in FOUNDATION_LESSON_CARDS:
            if lesson["grade_id"] == stage:
                cards.append({
                    "question": lesson["question"],
                    "answer": lesson["answer"],
                    "subject": lesson.get("subject", ""),
                    "grade_id": stage,
                    "kind": "starter",
                })
    return cards


def check_duplicate_lessons(tenant_id: int | None, text: str, stage: str = "nursery", subject: str = "General") -> dict:
    """Report lessons that already cover the question and answer being taught.

    The report separates an exact duplicate (same question and same answer),
    a repeated question that already has a different answer, and close
    paraphrases that are worth a "did you mean this?" hint.
    """
    raw = (text or "").strip()
    with _LOCK:
        state = _read(tenant_id)

    pairs = _parse_lesson_pairs(raw)
    word_pairs: list[tuple[str, str]] = []
    if not pairs and stage == "words":
        word_pairs = _parse_word_pairs(raw)

    exact: list[dict] = []
    same_question: list[dict] = []
    similar: list[dict] = []
    cards = _existing_lesson_cards(state, stage)

    for question, answer in pairs:
        question_key = _normalize_phrase(question)
        answer_key = _normalize_phrase(answer)
        best: tuple[float, dict] | None = None
        for card in cards:
            card_question = card.get("question", "")
            # The question is the same when the words are the same, word by word.
            if question_key == _normalize_phrase(card_question) or _same_words(question, card_question):
                entry = {
                    "question": card_question,
                    "answer": card.get("answer", ""),
                    "subject": card.get("subject", ""),
                    "grade_id": card.get("grade_id", stage),
                    "kind": card.get("kind", "trainer"),
                }
                # The answer is the same when its words are the same, word by word.
                if answer_key == _normalize_phrase(entry["answer"]) or _same_words(answer, entry["answer"]):
                    exact.append(entry)
                else:
                    same_question.append(entry)
                continue
            score = _word_by_word_similarity(question, card_question)
            if score >= _DUPLICATE_QUESTION_THRESHOLD and (best is None or score > best[0]):
                best = (score, card)
        if best is not None:
            score, card = best
            similar.append({
                "question": card.get("question", ""),
                "answer": card.get("answer", ""),
                "subject": card.get("subject", ""),
                "grade_id": card.get("grade_id", stage),
                "kind": card.get("kind", "trainer"),
                "similarity": round(score, 2),
            })

    duplicate_words: list[dict] = []
    for word, meaning in word_pairs:
        for card in state.get("word_cards", []):
            if card.get("word", "").lower() == word and _normalize_phrase(card.get("meaning", "")) == _normalize_phrase(meaning):
                duplicate_words.append({"word": word, "meaning": meaning})
                break

    return {
        "pairs": [{"question": question, "answer": answer} for question, answer in pairs],
        "exact": exact,
        "same_question": same_question,
        "similar": similar,
        "duplicate_words": duplicate_words,
        "duplicate": bool(exact or duplicate_words),
    }


def parse_subjects(value: str | None) -> list[str]:
    """The topics of a lesson, one by one.

    A trainer can write "Language, Poems" and the lesson then belongs to both
    topics. The same words in a different case are the same topic.
    """
    if not value:
        return ["General"]
    topics: list[str] = []
    for part in str(value).replace("\n", ",").split(","):
        topic = " ".join(part.split())[:80]
        if not topic:
            continue
        if not any(topic.casefold() == existing.casefold() for existing in topics):
            topics.append(topic)
    return topics or ["General"]


def subject_of(card: dict) -> str:
    """The topics of a card as one readable string."""
    topics = card.get("subjects")
    if isinstance(topics, list) and topics:
        return ", ".join(str(topic) for topic in topics)
    return ", ".join(parse_subjects(card.get("subject")))


def teach(tenant_id: int | None, text: str, stage: str = "nursery", subject: str = "General", skip_duplicates: bool = True) -> dict:
    raw = text.strip()
    if not raw:
        raise ValueError("Enter one or more example sentences first.")
    if len(raw) > 3000:
        raise ValueError("Please keep a lesson under 3,000 characters.")
    if stage not in _GRADE_IDS and stage not in _LEGACY_STAGES:
        raise ValueError("Choose a grade from Nursery through Graduation.")
    topics = parse_subjects(subject)
    subject = ", ".join(topics)

    with _LOCK:
        state = _read(tenant_id)
        stage_order = [item["id"] for item in STAGES]
        current_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        if stage in _GRADE_IDS and current_stage != "complete" and stage_order.index(stage) > stage_order.index(current_stage):
            current_title = next(item["title"] for item in STAGES if item["id"] == current_stage)
            stage_title = next(item["title"] for item in STAGES if item["id"] == stage)
            raise ValueError(f"Pass the {current_title} test before teaching {stage_title} lessons.")
        now = datetime.now(timezone.utc).isoformat()
        added: list[str] = []
        skipped: list[dict] = []
        if stage == "letters":
            letters = [char.lower() for char in raw if char.isalpha() and char.isascii()]
            if not letters:
                raise ValueError("Enter alphabet letters such as A B C or a-z.")
            state["letters"] = "".join(dict.fromkeys(state["letters"] + letters))
            added = letters
        elif stage == "words":
            new_cards = []
            known_words = {card.get("word", "").lower() for card in state["word_cards"]}
            for line in raw.splitlines():
                if ":" not in line:
                    raise ValueError("For each word, enter one line in this format: word: meaning")
                word, meaning = (part.strip() for part in line.split(":", 1))
                if not word or not meaning or len(tokenize(word)) != 1:
                    raise ValueError("Each word needs one vocabulary word and a short meaning after a colon.")
                if skip_duplicates and word.lower() in known_words:
                    skipped.append({"word": word.lower(), "meaning": meaning})
                    continue
                known_words.add(word.lower())
                new_cards.append({"word": word.lower(), "meaning": meaning, "created_at": now})
                added.append(word.lower())
            state["word_cards"].extend(new_cards)
            state["word_cards"] = state["word_cards"][-1000:]
        elif stage == "sentences":
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", raw) if part.strip()]
            normalized: list[str] = []
            known_sentences = {_normalize_phrase(item.get("text", "")) for item in state["lessons"]}
            for sentence in sentences:
                words = [word for word in tokenize(sentence) if word not in {".", "!", "?"}]
                if len(words) < 2:
                    raise ValueError("Each sentence needs at least two words so the model can learn a word sequence.")
                if len(words) > 40:
                    raise ValueError("Keep each sentence to 40 words or fewer.")
                if sentence[-1] not in ".!?":
                    sentence += "."
                if skip_duplicates and _normalize_phrase(sentence) in known_sentences:
                    skipped.append({"text": sentence})
                    continue
                known_sentences.add(_normalize_phrase(sentence))
                normalized.append(sentence)
            state["lessons"].extend({"text": sentence, "created_at": now} for sentence in normalized)
            state["lessons"] = state["lessons"][-2000:]
            added = normalized
        elif stage == "lessons" or stage in _GRADE_IDS:
            pairs = _parse_lesson_pairs(raw)
            if not pairs:
                raise ValueError("Enter lessons as question and answer pairs, using Q: on one line and A: on the next.")
            new_cards = []
            known_lessons = {
                (_normalize_phrase(card.get("question", "")), _normalize_phrase(card.get("answer", "")))
                for card in state["lesson_cards"]
            }
            for question, answer in pairs:
                question, answer = question.strip(), answer.strip()
                if not question or not answer:
                    raise ValueError("Each lesson needs both a question and its answer.")
                if skip_duplicates and (_normalize_phrase(question), _normalize_phrase(answer)) in known_lessons:
                    skipped.append({"question": question, "answer": answer})
                    continue
                known_lessons.add((_normalize_phrase(question), _normalize_phrase(answer)))
                new_cards.append({"id": uuid4().hex, "question": question, "answer": answer,
                                  "grade_id": stage if stage in _GRADE_IDS else "nursery",
                                  "subject": subject, "subjects": list(topics),
                                  "confidence": settings.base_answer_confidence,
                                  "created_at": now})
                added.append(question)
            state["lesson_cards"].extend(new_cards)
            state["lesson_cards"] = state["lesson_cards"][-500:]
        _save(tenant_id, state)
        result = get_state(tenant_id)
        result["added_lessons"] = len(added)
        result["last_learned"] = added
        result["skipped_duplicates"] = skipped
        result["added_lesson_cards"] = new_cards if stage in _GRADE_IDS or stage == "lessons" else []
        return result



def get_lesson_card(tenant_id: int | None, lesson_id: str) -> dict | None:
    with _LOCK:
        state = _read(tenant_id)
        return next((item for item in state["lesson_cards"] if item.get("id") == lesson_id), None)


def all_lesson_cards(tenant_id: int | None) -> list[dict]:
    """Every question and answer this tenant can answer with.

    That is the lessons taught in the language learner plus the starter
    lessons of the stages the tenant has reached.
    """
    with _LOCK:
        state = _read(tenant_id)
        current_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        cards = [
            {**item, "subject": subject_of(item), "subjects": parse_subjects(subject_of(item))}
            for item in state["lesson_cards"]
            if item.get("question")
        ]
        seen = {(item.get("question") or "").strip().lower() for item in cards}
        for lesson in FOUNDATION_LESSON_CARDS:
            if lesson["grade_id"] not in _allowed_grades(current_stage):
                continue
            question = lesson["question"].strip()
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            cards.append({
                "id": _foundation_lesson_id(question),
                "question": question,
                "answer": lesson["answer"],
                "subject": subject_of(lesson),
                "subjects": parse_subjects(subject_of(lesson)),
                "grade_id": lesson["grade_id"],
                "kind": "starter",
            })
        return cards


def _foundation_lesson_id(question: str) -> str:
    """A stable id for a starter lesson so it is only embedded once."""
    digest = hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"foundation-{digest}"


def tenant_ids_with_lessons() -> list[int | None]:
    """Tenant ids that have a saved language learner file."""
    keys: list[int | None] = []
    if not _DATA_DIR.is_dir():
        return keys
    for path in sorted(_DATA_DIR.glob("tenant-*.json")):
        try:
            tenant_key = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        keys.append(None if tenant_key == 0 else tenant_key)
    return keys


def update_lesson_answer(
    tenant_id: int | None,
    lesson_id: str,
    answer: str,
    question: str | None = None,
    confidence: float | None = None,
    subject: str | None = None,
) -> dict:
    answer = answer.strip()
    if not answer:
        raise ValueError("An answer is required.")
    cleaned_question = question.strip() if question is not None else None
    if cleaned_question is not None and not cleaned_question:
        raise ValueError("A question is required.")
    topics = parse_subjects(subject) if subject is not None else None
    with _LOCK:
        state = _read(tenant_id)
        lesson = next((item for item in state["lesson_cards"] if item.get("id") == lesson_id), None)
        if not lesson:
            raise LookupError("Learner lesson not found.")
        lesson["answer"] = answer
        if cleaned_question is not None:
            lesson["question"] = cleaned_question
        if confidence is not None:
            lesson["confidence"] = _clamp_confidence(confidence)
        if topics is not None:
            lesson["subjects"] = topics
            lesson["subject"] = ", ".join(topics)
        lesson["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save(tenant_id, state)
        return get_state(tenant_id)


def _clamp_confidence(value: float | None) -> float:
    if value is None:
        return settings.base_answer_confidence
    return max(0.0, min(settings.max_answer_confidence, round(float(value), 3)))


def find_lesson_by_answer(tenant_id: int | None, answer: str) -> dict | None:
    """The lesson that already stores this answer, if the learner has one."""
    target = _normalize_phrase(answer)
    if not target:
        return None
    with _LOCK:
        state = _read(tenant_id)
        for lesson in state["lesson_cards"]:
            if _normalize_phrase(lesson.get("answer", "")) == target:
                return dict(lesson)
    return None


_QUESTION_STOP_WORDS = {
    "what", "who", "whom", "whose", "when", "where", "which", "why", "how", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "done", "the", "a", "an",
    "of", "for", "from", "please", "tell", "give", "show", "me", "about", "us", "it", "its",
    "to", "and", "or", "in", "on", "at", "that", "this", "these", "those", "there", "here",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must", "has", "have",
    "had", "i", "you", "we", "they", "he", "she", "his", "her", "their", "my", "your", "our",
    "some", "any", "all", "many", "much", "more", "most", "not", "no",
}


_STEM_EXCEPTIONS = {"does": "do", "goes": "go", "was": "be", "has": "have", "is": "be"}


def _stem(token: str) -> str:
    """The word without its ending, so "gives" and "give" are the same word."""
    if token in _STEM_EXCEPTIONS:
        return _STEM_EXCEPTIONS[token]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    if token.endswith("ss"):
        return token
    if token.endswith("es") and len(token) > 4 and token[-3] in "sxz":
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def match_lesson_by_words(question: str, tenant_id: int | None) -> dict | None:
    """The lesson that covers the question by its words.

    It is the fallback for a question that is asked in other words, so the
    assistant still finds what it was taught.
    """
    terms = {
        _stem(token)
        for token in _content_terms(question, _QUESTION_STOP_WORDS)
    }
    if not terms:
        return None
    cards = all_lesson_cards(tenant_id)
    # How often each word is used in a lesson. A rare word says much more
    # about the question than a word that is in every lesson.
    frequency: dict[str, int] = {}
    card_words: list[set[str]] = []
    for card in cards:
        words = {
            _stem(token)
            for token in _content_terms(card.get("question", ""), _QUESTION_STOP_WORDS)
        }
        card_words.append(words)
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1

    def weight(word: str) -> float:
        return 1.0 / (1.0 + frequency.get(word, 0))

    best: tuple[float, dict] | None = None
    for card, words in zip(cards, card_words):
        if not words:
            continue
        shared = terms & words
        if not shared:
            continue
        score = sum(weight(word) for word in shared) / max(
            sum(weight(word) for word in terms | words),
            1e-6,
        )
        if best is None or score > best[0]:
            best = (score, card)
    return best[1] if best and best[0] >= 0.3 else None


def find_known_answer(tenant_id: int | None, answer: str) -> dict | None:
    """The lesson that already stores this answer, taught or built in.

    It tells the trainer that a confirmed answer is already known, so it is
    not copied into the Trained answers a second time.
    """
    target = _normalize_phrase(answer)
    if not target:
        return None
    with _LOCK:
        state = _read(tenant_id)
        for lesson in state["lesson_cards"]:
            if _normalize_phrase(lesson.get("answer", "")) == target:
                return {"kind": "trainer", "card": dict(lesson)}
    for lesson in FOUNDATION_LESSON_CARDS:
        if _normalize_phrase(lesson.get("answer", "")) == target:
            return {"kind": "starter", "card": dict(lesson)}
    return None


def bump_lesson_confidence(tenant_id: int | None, answer: str, question: str = "") -> dict | None:
    """Raise the confidence of the lesson that already holds this answer.

    A confirmed answer that the learner already knows is not copied anywhere.
    It only becomes more trusted.
    """
    target = _normalize_phrase(answer)
    if not target:
        return None
    with _LOCK:
        state = _read(tenant_id)
        lesson = next(
            (item for item in state["lesson_cards"] if _normalize_phrase(item.get("answer", "")) == target),
            None,
        )
        if lesson is None:
            return None
        current = float(lesson.get("confidence", settings.base_answer_confidence))
        lesson["confidence"] = round(min(settings.max_answer_confidence, current + settings.confidence_step), 3)
        lesson["confirmed_at"] = datetime.now(timezone.utc).isoformat()
        lesson["confirmations"] = int(lesson.get("confirmations", 0)) + 1
        _save(tenant_id, state)
        return dict(lesson)


def remove_lesson(tenant_id: int | None, lesson_id: str) -> dict:
    with _LOCK:
        state = _read(tenant_id)
        remaining = [item for item in state["lesson_cards"] if item.get("id") != lesson_id]
        if len(remaining) == len(state["lesson_cards"]):
            raise LookupError("Learner lesson not found.")
        state["lesson_cards"] = remaining
        _save(tenant_id, state)
        return get_state(tenant_id)


def _public_test(test: dict | None) -> dict | None:
    if not test:
        return None
    return {key: test.get(key) for key in (
        "stage", "questions_asked", "answers_passed", "needs_learning",
        "awaiting_score", "last_question", "last_answer", "last_source", "last_source_detail",
    )}


def _asked_word(question: str) -> str | None:
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


def _answer_letter_question(question: str, alphabet: str) -> str | None:
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
    if dictionary.is_dictionary_question(question) and re.search(r"\b(?:start|begin|first\s+letter)\b", question, re.I):
        return dictionary.answer_dictionary_question(question)
    return None


def _content_terms(text: str, stop_words: set[str]) -> set[str]:
    """Meaningful tokens of a text.

    Punctuation is dropped on purpose. A question like "who is daraksha?"
    otherwise shares the "?" token with every lesson that ends in a question
    mark, which made unrelated curriculum answers match.
    """
    return {
        token
        for token in tokenize(text)
        if any(character.isalnum() for character in token) and token not in stop_words
    }


def _answer_from_sentence_examples(question: str, examples: list[str]) -> str | None:
    """A taught sentence that covers the question.

    The question has to be mostly covered by the sentence, otherwise a question
    about the sea would be answered with a sentence about the sky just because
    both mention a colour.
    """
    stop_words = {"what", "who", "where", "when", "which", "is", "are", "was", "were", "do", "does", "did", "the", "a", "an", "of", "for", "please", "tell", "me", "in", "on", "from", "about", "to", "with"}
    terms = _content_terms(question, stop_words)
    best: tuple[float, str] | None = None
    for sentence in examples:
        overlap = terms & set(tokenize(sentence))
        if overlap:
            score = len(overlap) / max(len(terms), 1)
            if best is None or score > best[0]:
                best = (score, sentence)
    return best[1] if best and best[0] >= 0.6 else None


def _answer_learned_lesson(question: str, lesson_cards: list[dict]) -> str | None:
    lesson = _matched_learned_lesson(question, lesson_cards)
    return lesson["answer"] if lesson else None


def _matched_learned_lesson(question: str, lesson_cards: list[dict]) -> dict | None:
    """The lesson that best covers the question, with the question it was taught as."""
    stop_words = {"what", "who", "where", "when", "which", "why", "how", "is", "are", "was", "were", "do", "does", "did", "the", "a", "an", "of", "for", "please", "tell", "me", "about"}
    terms = _content_terms(question, stop_words)
    best: tuple[float, float, dict] | None = None
    for lesson in lesson_cards:
        candidate = _content_terms(lesson.get("question", ""), stop_words)
        if terms and candidate:
            score = len(terms & candidate) / max(len(terms | candidate), 1)
            # A trusted answer wins over a weaker match of the same score.
            trust = float(lesson.get("confidence", settings.base_answer_confidence))
            if best is None or (score, trust) > (best[0], best[1]):
                best = (score, trust, lesson)
    return best[2] if best and best[0] >= 0.5 else None


def _answer_alphabet_question(question: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 -]", " ", question.casefold())
    normalized = " ".join(normalized.split())
    letter_to_word = ALPHABET_WORDS
    word_to_letter = {
        variant: letter
        for letter, word in letter_to_word.items()
        for variant in (word, word.replace("-", ""))
    }

    letter_forms = re.search(r"\b(?:capital|uppercase)\s+(?:and\s+small\s+)?(?:forms?\s+of\s+)?([a-z])\b", normalized)
    explicit_forms = re.search(r"\b(?:capital|uppercase)(?:\s+letter)?\s+([a-z])\b.*\b(?:small|lowercase)(?:\s+letter)?\s+([a-z])\b", normalized)
    if explicit_forms:
        capital, small = explicit_forms.group(1).upper(), explicit_forms.group(2).lower()
        return f"Capital {capital} is {capital}; small {small} is {small}."
    if re.search(r"\b(?:capital|uppercase)\s+and\s+(?:small|lowercase)\s+forms?\s+of\s+[a-z]\b", normalized):
        match = re.search(r"\bof\s+([a-z])\b", normalized)
        if match:
            capital = match.group(1).upper()
            return f"Capital {capital} is {capital}; small {capital.lower()} is {capital.lower()}."
    if letter_forms:
        capital = letter_forms.group(1).upper()
        return f"Capital {capital} is the uppercase form of {capital}."
    small_form = re.search(r"\b(?:small|lowercase)\s+(?:letter\s+)?([a-z])\b", normalized)
    if small_form:
        small = small_form.group(1).lower()
        return f"Small {small} is the lowercase form of {small.upper()}."

    match = re.search(r"\b(?:what|which)\s+letter\s+(?:does\s+)?([a-z]+(?:-[a-z]+)?)\s+(?:start|begin)s?\s+(?:with\s*)?\b", normalized)
    if match:
        word = match.group(1)
        letter = word_to_letter.get(word)
        if letter:
            return f"{word.capitalize()} starts with {letter}."

    match = re.search(r"\b(?:what\s+is\s+)?(?:the\s+)?spelling\s+of\s+([a-z-]+)\b|\bspell\s+([a-z-]+)\b", normalized)
    if match:
        word = match.group(1) or match.group(2)
        plain_word = word.replace("-", "")
        if plain_word in word_to_letter:
            return f"{word.capitalize()} is spelled {'-'.join(plain_word.upper())}."

    match = re.search(r"\b(?:what\s+is\s+)?([a-z])\s+for\b|\b([a-z])\s+for\s+what\b", normalized)
    if match:
        letter = (match.group(1) or match.group(2)).upper()
        word = letter_to_word.get(letter)
        if word:
            return f"{letter} is for {word}."

    match = re.search(r"\b(?:what|which)\s+word\s+(?:starts|begins)\s+with\s+([a-z])\b", normalized)
    if match:
        letter = match.group(1).upper()
        word = letter_to_word.get(letter)
        if word:
            return f"{letter} is for {word}."
    return None


def _answer_stage_question(question: str, state: dict, stage: str) -> tuple[str, str]:
    age_answer = answer_assistant_age(question)
    if age_answer:
        return age_answer, "assistant_profile"
    current_date_time = answer_current_date_time(question)
    if current_date_time:
        return current_date_time, "learner"
    spelling_answer = _answer_number_spelling_question(question)
    if spelling_answer:
        return spelling_answer, "learner"
    counting_answer = _answer_counting_question(question)
    if counting_answer:
        return counting_answer, "learner"
    number_sequence = _answer_number_sequence_question(question)
    if number_sequence:
        return number_sequence, "learner"
    alphabet_answer = _answer_alphabet_question(question)
    if alphabet_answer:
        return alphabet_answer, "learner"
    if stage == "letters":
        answer = _answer_letter_question(question, state["letters"])
        return (answer, "learner") if answer else ("I haven't learned an answer to that in the Letters stage yet.", "not_learned")
    if stage == "words":
        word = _asked_word(question)
        if word:
            for item in [*_tenant_words_for_stage(state, "nursery"), *_foundation_words_for_stage("nursery")]:
                if item["word"].casefold() == word:
                    return f"{item['word'].capitalize()} means {item['meaning']}.", "learner"
        return ("I haven't learned that word in the Words stage yet. Look it up in the offline dictionary and teach it to me.", "not_learned")
    if stage == "sentences":
        next_match = _NEXT_WORD_REQUEST.search(question.strip())
        answer = None
        if next_match:
            prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
            options = _model(state, "nursery").predict_next(prefix)
            if options:
                answer = f"From my sentence lessons, after “{prefix}” I predict “{options[0]['word']}”."
        if not answer:
            sentence_match = _SENTENCE_REQUEST.search(question.strip())
            if sentence_match:
                requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-")
                answer = _model(state, "nursery").generate_sentence(seed=requested) if requested else None
        if not answer:
            answer = _answer_from_sentence_examples(question, [*(item["text"] for item in state["lessons"]), *_foundation_sentences_for_stage("nursery")])
        return (answer, "learner") if answer else ("I haven't learned a sentence pattern that answers that yet.", "not_learned")
    if stage in _GRADE_IDS:
        word = _asked_word(question)
        if word:
            for item in [*_tenant_words_for_stage(state, stage), *_foundation_words_for_stage(stage)]:
                if item["word"].casefold() == word:
                    return f"{item['word'].capitalize()} means {item['meaning']}.", "learner"
        sentence_match = _SENTENCE_REQUEST.search(question.strip())
        if sentence_match:
            requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-")
            answer = _model(state, stage).generate_sentence(seed=requested) if requested else None
            if answer:
                return answer, "sentence_patterns"
        next_match = _NEXT_WORD_REQUEST.search(question.strip())
        if next_match:
            prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
            options = _model(state, stage).predict_next(prefix)
            if options:
                return f"From my grade lessons, after “{prefix}” I predict “{options[0]['word']}”.", "sentence_patterns"
        allowed_grades = _allowed_grades(stage)
        grade_lessons = [
            item for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
            if item.get("grade_id", "nursery") in allowed_grades
        ]
        answer = _answer_learned_lesson(question, grade_lessons)
        return (answer, "learner") if answer else ("I haven't learned an answer to that grade's lessons yet.", "not_learned")
    answer = _answer_learned_lesson(question, [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS])
    return (answer, "learner") if answer else ("I haven't learned an answer to the Nursery lessons yet.", "not_learned")


def start_test(tenant_id: int | None, stage_id: str | None = None, kind: str = "trainer") -> dict:
    """Start a test.

    ``kind`` is one of:
      * ``trainer`` - the trainer-led test, which can advance the grade,
      * ``mock``    - a practice run that never advances the grade,
      * ``exam``    - the final exam, which advances the grade when the
                      learner answers enough questions correctly.
    """
    if kind not in {"trainer", "mock", "exam"}:
        raise ValueError("Choose a test type of trainer, mock or exam.")
    with _LOCK:
        state = _read(tenant_id)
        next_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        stage = stage_id or next_stage
        if stage == "complete":
            return {"state": get_state(tenant_id), "message": "All grade tests are passed."}
        if stage not in _GRADE_IDS:
            raise ValueError("Choose a valid grade for this test.")
        order = [item["id"] for item in STAGES]
        if next_stage != "complete" and order.index(stage) > order.index(next_stage):
            raise ValueError("Pass the previous grade before testing this grade.")
        if kind == "exam" and stage in state["passed_stages"]:
            raise ValueError("This grade has already been passed.")
        state["active_test"] = {"stage": stage, "kind": kind, "questions_asked": 0, "answers_passed": 0,
                                "needs_learning": 0, "awaiting_score": False,
                                "last_question": "", "last_answer": "", "last_source": ""}
        _save(tenant_id, state)
        return {"state": get_state(tenant_id), "kind": kind}


def ask_test_question(tenant_id: int | None, question: str, source_resolver=None) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Enter a question for the learner.")
    if len(question) > 1000:
        raise ValueError("Keep a test question under 1,000 characters.")
    with _LOCK:
        state = _read(tenant_id)
        test = state.get("active_test")
        if not test:
            raise ValueError("Start a trainer-led test before asking a question.")
        if test["awaiting_score"]:
            raise ValueError("Score the learner's last answer before asking another question.")
        answer = answer_assistant_age(question)
        source, source_detail = ("assistant_profile", "Assistant birthday") if answer else ("", "")
        if not answer:
            answer = answer_current_date_time(question)
            if answer:
                source, source_detail = "learner", "Date & Time lessons"
        if not answer and source_resolver:
            resolved = source_resolver(question)
            if resolved:
                answer, source, source_detail = resolved
        if not answer:
            answer, source = _answer_stage_question(question, state, test["stage"])
            source_detail = "Learner curriculum" if source in ("learner", "sentence_patterns") else ""
        test["questions_asked"] += 1
        test["last_question"] = question
        test["last_answer"] = answer
        test["last_source"] = source
        test["last_source_detail"] = source_detail
        test["awaiting_score"] = True
        _save(tenant_id, state)
        return {"answer": answer, "source": source, "source_detail": source_detail, "state": get_state(tenant_id)}


def score_test_answer(tenant_id: int | None, passed: bool) -> dict:
    with _LOCK:
        state = _read(tenant_id)
        test = state.get("active_test")
        if not test or not test["awaiting_score"]:
            raise ValueError("Ask a test question before scoring an answer.")
        if passed:
            test["answers_passed"] += 1
        else:
            test["needs_learning"] += 1
        test["awaiting_score"] = False
        _save(tenant_id, state)
        return {"state": get_state(tenant_id)}


def finish_test(tenant_id: int | None, passed: bool) -> dict:
    with _LOCK:
        state = _read(tenant_id)
        test = state.get("active_test")
        if not test:
            raise ValueError("Start a trainer-led test before scoring the stage.")
        if test["awaiting_score"]:
            raise ValueError("Score the learner's last answer first.")
        if test["questions_asked"] == 0:
            raise ValueError("Ask at least one question before scoring this stage.")
        kind = test.get("kind", "trainer")
        asked = max(test["questions_asked"], 1)
        pass_rate = test["answers_passed"] / asked
        # A mock test is practice only, and a final exam needs a strong score.
        if kind == "mock":
            passed = False
        elif kind == "exam":
            passed = passed and pass_rate >= _EXAM_PASS_RATE
        if passed and test["stage"] not in state["passed_stages"]:
            state["passed_stages"].append(test["stage"])
        result = {"stage": test["stage"], "kind": kind, "questions_asked": test["questions_asked"],
                  "answers_passed": test["answers_passed"], "needs_learning": test["needs_learning"],
                  "pass_rate": round(pass_rate, 2),
                  "passed": passed, "created_at": datetime.now(timezone.utc).isoformat()}
        state["test_history"].append(result)
        state["test_history"] = state["test_history"][-200:]
        state["active_test"] = None
        _save(tenant_id, state)
        return {"result": result, "state": get_state(tenant_id)}


def add_dictionary_word(tenant_id: int | None, word: str, meaning: str, stage: str = "nursery") -> dict:
    word, meaning = word.strip().lower(), meaning.strip()
    if not word or not meaning:
        raise ValueError("Choose a dictionary word with a definition first.")
    if stage not in _GRADE_IDS:
        raise ValueError("Choose a curriculum grade for this dictionary word.")
    with _LOCK:
        state = _read(tenant_id)
        current_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        stage_order = [item["id"] for item in STAGES]
        if current_stage != "complete" and stage_order.index(stage) > stage_order.index(current_stage):
            current_title = next(item["title"] for item in STAGES if item["id"] == current_stage)
            stage_title = next(item["title"] for item in STAGES if item["id"] == stage)
            raise ValueError(f"Pass the {current_title} test before teaching {stage_title} vocabulary.")
        now = datetime.now(timezone.utc).isoformat()
        existing = next((item for item in state["word_cards"] if item["word"].casefold() == word.casefold()), None)
        if existing:
            existing["meaning"] = meaning
            existing["grade_id"] = stage
            existing["created_at"] = now
        else:
            state["word_cards"].append({"word": word, "meaning": meaning, "grade_id": stage, "created_at": now})
            state["word_cards"] = state["word_cards"][-1000:]
        _save(tenant_id, state)
        return get_state(tenant_id)


def predict_next(tenant_id: int | None, prefix: str) -> list[dict[str, int | str]]:
    with _LOCK:
        return _model(_read(tenant_id)).predict_next(prefix)


def make_sentence(tenant_id: int | None, seed: str | None = None) -> str | None:
    if seed:
        statement = _lesson_statement_for_word(tenant_id, seed)
        if statement:
            return statement
    with _LOCK:
        sentence = _model(_read(tenant_id)).generate_sentence(seed=seed)
    return _repair_sentence(sentence) if sentence else sentence


def _repair_sentences(texts: list[str]) -> list[str]:
    """Clean a whole list of sentences, with the tagger looking at them all
    at once so it does not have to be loaded for every sentence."""
    if not texts:
        return []
    try:
        from app.services.nlp import get_nlp

        nlp = get_nlp()
        documents = list(nlp.pipe(texts, batch_size=64))
    except Exception:
        documents = [None] * len(texts)
    repaired: list[str] = []
    for text, document in zip(texts, documents):
        if document is None:
            repaired.append(_POSSESSIVE_TYPED.sub(
                lambda match: f"{match.group(1)}s {match.group(2)}",
                text or "",
            ))
            continue
        repaired.append(_repair_with_document(text, document))
    return repaired


def _repair_with_document(sentence: str, document) -> str:
    """Put back the article that a naming word is missing."""
    insertions: list[tuple[int, str]] = []
    for index, token in enumerate(document):
        if token.lower_ not in {"is", "are", "was", "were"}:
            continue
        if index + 1 >= len(document):
            continue
        following = document[index + 1]
        if following.pos_ not in {"NOUN", "PROPN"}:
            continue
        if following.tag_ in {"NNS", "NNPS"}:
            # A plural word does not take an article.
            continue
        article = "an" if following.text[:1].lower() in "aeiou" else "a"
        insertions.append((following.idx, article))
    text = sentence
    for position, article in reversed(insertions):
        text = text[:position] + f"{article} " + text[position:]
    return text


def training_corpus(tenant_id: int | None) -> list[str]:
    """Every sentence the model can learn from.

    It is the trainer's own material: the lessons that were taught, the
    answers that were approved, the sentences that were written and the
    corrections that were kept. The wording is cleaned first, so the model does
    not learn a slip of the pen.
    """
    with _LOCK:
        state = _read(tenant_id)
    raw: list[str] = []
    for card in all_lesson_cards(tenant_id):
        question = (card.get("question") or "").strip()
        answer = (card.get("answer") or "").strip()
        if answer:
            raw.append(answer)
        if question and answer:
            raw.append(f"{question} {answer}")
    for lesson in state.get("lessons", []):
        raw.append((lesson.get("text") or "").strip())
    for lesson in FOUNDATION_LESSON_CARDS:
        raw.append((lesson.get("answer") or "").strip())
    raw = [text for text in raw if text]

    sentences: list[str] = []
    seen: set[str] = set()
    for cleaned in _repair_sentences(raw):
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        sentences.append(cleaned)
    return sentences


_COUNT_REQUEST = re.compile(
    r"\b(\d{1,2})\b|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _is_a_list(answer: str) -> bool:
    """Whether an answer is a list of things, like "Some animals are ant, bear, cat."

    A list is the answer itself, so it is not put into other words.
    """
    head, separator, tail = (answer or "").partition(" are ")
    if not separator:
        return False
    items = [item.strip() for item in tail.rstrip(". ").split(",") if item.strip()]
    return len(items) >= 3


def apply_requested_count(question: str, answer: str) -> str:
    """When the question asks for a number of things, give that many.

    "Name 5 animals" with a lesson that lists every animal gives five of them.
    """
    match = _COUNT_REQUEST.search(question or "")
    if not match:
        return answer
    token = match.group(1) or match.group(2) or ""
    wanted = int(token) if token.isdecimal() else _COUNT_WORDS.get(token.casefold(), 0)
    if wanted < 1 or wanted > 50:
        return answer
    head, separator, tail = (answer or "").partition(" are ")
    if not separator:
        return answer
    items = [item.strip() for item in tail.rstrip(". ").split(",") if item.strip()]
    if len(items) <= wanted:
        return answer
    return f"{head} are {', '.join(items[:wanted])}."


def _subject_of(answer: str) -> str:
    """The subject of an answer, such as "The cow" in "The cow is an animal."

    It is the first few words, up to the first word that tells what is said
    about it.
    """
    words = (answer or "").strip().split()
    if not words:
        return ""
    stop = {"is", "are", "was", "were", "has", "have", "gives", "give", "makes", "make", "can", "will", "does", "do", "said", "says", "means", "mean"}
    for index, word in enumerate(words[:4]):
        if word.strip(".,!?").lower() in stop:
            return " ".join(words[:index]).strip(".,!?")
    return " ".join(words[:2]).strip(".,!?")


def _predicates_of(answer: str, subject: str) -> list[str]:
    """The facts the answer tells about its subject.

    "The cow is animal. It is a farm animal that gives milk." holds the facts
    "is animal", "is a farm animal" and "gives milk".
    """
    facts: list[str] = []
    head_words = re.sub(r"^(?:the|a|an)\s+", "", " ".join(subject.split()), flags=re.IGNORECASE).strip(".,!?")
    if not head_words:
        return facts
    subject_pattern = re.compile(
        r"\b(?:(?:the|a|an)\s+)?" + re.escape(head_words) + r"\b",
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", (answer or "").strip()):
        text = sentence.strip()
        if not text:
            continue
        if text.endswith("?"):
            # A question is not something that can be said about a subject.
            continue
        text = subject_pattern.sub("they", text)
        if re.match(r"^(?:it|they|he|she|this|that)\b", text, re.IGNORECASE):
            text = re.sub(
                r"^(?:it|they|he|she|this|that)\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
        for part in re.split(r"\s+(?:that|which|and)\s+|,\s+", text):
            part = part.strip(" .!?")
            if part:
                facts.append(part)
    return facts


def _article_for(model, word: str) -> str:
    """The article that belongs before a word, such as "an" before "animal".

    The article the model has seen is used, and when it has not seen one the
    usual rule of the language is used.
    """
    lowered = word.lower()
    for article in ("an", "a"):
        counts = model.backoff.get((article, lowered)) or model.transitions.get((article, lowered))
        if counts:
            return article
    if lowered[:1] in "aeiou":
        return "an"
    return "a"


def _fix_article(model, sentence: str) -> str:
    """Put the article back when the taught answer left it out."""
    return _add_missing_articles(sentence)


def _facts_known_about(model, subject: str) -> list[str]:
    """What the model knows about a subject, as facts.

    It is used when an answer is only the subject, so the assistant can still
    say something about it in its own words.
    """
    head = subject.split()[-1].strip(".,!?").lower()
    if not head:
        return []
    facts: list[str] = []
    for sentence in model.sentences:
        if sentence.strip().endswith("?"):
            # A question is not something that can be said about a subject.
            continue
        tokens = tokenize(sentence)
        if not any(
            token == head or token.startswith(head + "'") or token.startswith(head + "s")
            for token in tokens
        ):
            continue
        facts.extend(_predicates_of(sentence, subject))
    return facts


def _wording_score(tenant_id: int | None, model, sentence: str) -> float:
    """How well the model knows a wording.

    The transformer that was trained on the trainer's own words knows it best,
    so its judgement is used when there is one. Without it the count of the
    words is used.
    """
    writer = None
    try:
        from app.services import brain

        writer = brain.load_writer(tenant_id)
    except Exception:
        writer = None
    if writer is not None:
        return writer.score(sentence)
    return model.score_words(sentence.rstrip(".").split())


def paraphrase_answer(tenant_id: int | None, answer: str, subject: str | None = None) -> str | None:
    """Write the answer in the model's own words, from what it has learned.

    The knowledge still comes from the answer the trainer taught, but the
    sentence is put together by the model, and the model chooses the wording
    it knows best. So "The cow is animal. It is a farm animal that gives
    milk." can become "The cow is a farm animal."
    """
    answer = (answer or "").strip()
    if not answer:
        return None
    if _is_a_list(answer):
        # A list of things is the answer itself.
        return None
    subject = subject or _subject_of(answer)
    if not subject:
        return None
    with _LOCK:
        model = _model(_read(tenant_id))
    facts = _predicates_of(answer, subject)
    if not facts:
        # The answer is only the subject. The model does not make something up
        # for it, so the answer is left as it is.
        return None
    scored: list[tuple[float, str]] = []
    for fact in facts:
        sentence = _fix_article(model, f"{subject} {fact}.")
        scored.append((_wording_score(tenant_id, model, sentence), sentence))
    scored.sort(key=lambda item: -item[0])
    # A rewrite may say it in other words, but it must not lose what the
    # answer said. A list of things stays a list.
    original_words = {
        _stem(word)
        for word in _content_terms(answer, _QUESTION_STOP_WORDS)
    }
    for _, sentence in scored:
        if sentence.strip().lower() == answer.strip().lower():
            continue
        sentence_words = {
            _stem(word)
            for word in _content_terms(sentence, _QUESTION_STOP_WORDS)
        }
        if not sentence_words:
            continue
        kept = len(sentence_words & original_words) / max(len(original_words), 1)
        if kept < 0.5:
            continue
        return sentence
    return None


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _calculation_result(operation: str, left: float, right: float) -> str | None:
    try:
        if operation == "plus":
            return f"{_format_number(left)} plus {_format_number(right)} is {_format_number(left + right)}."
        if operation == "minus":
            return f"{_format_number(left)} minus {_format_number(right)} is {_format_number(left - right)}."
        if operation == "times":
            return f"{_format_number(left)} times {_format_number(right)} is {_format_number(left * right)}."
        if operation == "divided":
            if right == 0:
                return "You cannot divide a number by zero."
            return f"{_format_number(left)} divided by {_format_number(right)} is {_format_number(left / right)}."
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _answer_calculation_question(question: str) -> str | None:
    """Answer adding, subtraction, multiplication, division, and square roots."""
    text = question.strip().lower().rstrip("?.! ")
    if not text:
        return None

    root = re.search(r"(?:square\s*root|sqrt)\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if root:
        value = float(root.group(1))
        if value < 0:
            return "A negative number does not have a real square root."
        answer = math.sqrt(value)
        return f"The square root of {_format_number(value)} is {_format_number(answer)}."

    squared = re.search(r"(-?\d+(?:\.\d+)?)\s+(?:squared|to the power of 2)\b", text)
    if squared:
        value = float(squared.group(1))
        return f"{_format_number(value)} squared is {_format_number(value * value)}."

    cubed = re.search(r"(-?\d+(?:\.\d+)?)\s+cubed\b", text)
    if cubed:
        value = float(cubed.group(1))
        return f"{_format_number(value)} cubed is {_format_number(value ** 3)}."

    percent = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s+(-?\d+(?:\.\d+)?)", text)
    if percent:
        value = float(percent.group(1))
        total = float(percent.group(2))
        return f"{_format_number(value)} percent of {_format_number(total)} is {_format_number(total * value / 100)}."

    symbol = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)", text)
    if symbol:
        left = float(symbol.group(1))
        right = float(symbol.group(3))
        operator = symbol.group(2)
        operation = {"+": "plus", "-": "minus", "*": "times", "x": "times", "×": "times", "/": "divided", "÷": "divided"}.get(operator)
        if operation:
            return _calculation_result(operation, left, right)

    words = re.search(
        r"(-?\d+(?:\.\d+)?)\s+(plus|add|added to|addition of|minus|subtract|subtracted from|take away|times|multiplied by|multiply|divided by|divide|divided into)\s+(-?\d+(?:\.\d+)?)",
        text,
    )
    if words:
        left = float(words.group(1))
        right = float(words.group(3))
        operator = words.group(2)
        if operator in {"plus", "add", "added to", "addition of"}:
            return _calculation_result("plus", left, right)
        if operator == "minus":
            return _calculation_result("minus", left, right)
        if operator in {"subtract", "take away"}:
            return _calculation_result("minus", right, left)
        if operator == "subtracted from":
            return _calculation_result("minus", right, left)
        if operator in {"times", "multiplied by", "multiply"}:
            return _calculation_result("times", left, right)
        if operator in {"divided by", "divide"}:
            return _calculation_result("divided", left, right)
        if operator == "divided into":
            return _calculation_result("divided", right, left)

    divided = re.search(r"divide\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)", text)
    if divided:
        return _calculation_result("divided", float(divided.group(1)), float(divided.group(2)))

    added = re.search(r"(?:add|plus)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)", text)
    if added:
        return _calculation_result("plus", float(added.group(1)), float(added.group(2)))

    subtracted = re.search(r"(?:subtract|take)\s+(-?\d+(?:\.\d+)?)\s+from\s+(-?\d+(?:\.\d+)?)", text)
    if subtracted:
        return _calculation_result("minus", float(subtracted.group(2)), float(subtracted.group(1)))

    multiplied = re.search(r"multiply\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)", text)
    if multiplied:
        return _calculation_result("times", float(multiplied.group(1)), float(multiplied.group(2)))

    doubled = re.search(r"double\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", text)
    if doubled:
        value = float(doubled.group(1))
        return f"Double of {_format_number(value)} is {_format_number(value * 2)}."

    tripled = re.search(r"triple\s+(?:of\s+)?(-?\d+(?:\.\d+)?)", text)
    if tripled:
        value = float(tripled.group(1))
        return f"Triple of {_format_number(value)} is {_format_number(value * 3)}."

    halved = re.search(r"half\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if halved:
        value = float(halved.group(1))
        return f"Half of {_format_number(value)} is {_format_number(value / 2)}."

    return None


_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "because", "so", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "is", "are", "was", "were", "am", "be", "been", "it", "this", "that",
    "my", "your", "his", "her", "its", "our", "their", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "how", "can", "do", "does", "did", "make", "makes", "made", "sentence",
}


def _taught_word_meanings(tenant_id: int | None) -> dict[str, str]:
    """The words this tenant has learned and the meaning it was given."""
    with _LOCK:
        state = _read(tenant_id)
        current_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        meanings: dict[str, str] = {}
        for item in [*_tenant_words_for_stage(state, current_stage), *_foundation_words_for_stage(current_stage)]:
            word = (item.get("word") or "").strip().lower()
            meaning = (item.get("meaning") or "").strip()
            if word and meaning:
                meanings[word] = meaning
        for item in state.get("word_cards", []):
            word = (item.get("word") or "").strip().lower()
            meaning = (item.get("meaning") or "").strip()
            if word and meaning:
                meanings[word] = meaning
        return meanings


def _is_describing_word(meaning: str) -> bool:
    """Words like red, blue, or happy describe a noun instead of naming one."""
    text = (meaning or "").strip().lower()
    return text.startswith("the color") or text.startswith("feeling") or text.startswith("a feeling")


def _taught_sentences(tenant_id: int | None) -> list[str]:
    """The sentence examples the learner has been taught."""
    with _LOCK:
        state = _read(tenant_id)
        sentences = [item.get("text", "") for item in state.get("lessons", [])]
    for grade in FOUNDATION_SENTENCES_BY_GRADE.values():
        sentences.extend(grade)
    return [sentence for sentence in sentences if sentence]


def _taught_answers(tenant_id: int | None) -> list[str]:
    """The answers of the starter lessons, which read like sentences."""
    return [lesson["answer"] for lesson in FOUNDATION_LESSON_CARDS if lesson.get("answer")]


def _tenant_lesson_answers(tenant_id: int | None) -> list[str]:
    """The answers the trainer taught, which read like sentences as well."""
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception:
        return []
    return [
        card["answer"]
        for card in cards
        if card.get("kind") != "starter" and card.get("answer")
    ]


_LESSON_QUESTION_STATEMENT = re.compile(
    r"^(?:what|which|who)\s+is\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
# "Which planet has the most moons?" becomes "Saturn has the most moons."
_LESSON_WHICH_SUBJECT = re.compile(
    r"^which\s+(.+?)\s+(has|have|is|are|was|were|does|do|did|can|could|will|would|should)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
# "Name the largest planet" becomes "Saturn is the largest planet."
_LESSON_NAME_REQUEST = re.compile(
    r"^name\s+(the|a|an)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)


def _clean_phrase(text: str) -> str:
    """A phrase without its end mark."""
    return (text or "").strip().rstrip(".!?").strip()


def _statement_from_lesson(question: str, answer: str) -> str | None:
    """Turn a lesson question into a statement that holds the answer.

    "Which is the largest country by area?" with the answer "Russia." becomes
    "Russia is the largest country by area." and "Which planet has the most
    moons?" with the answer "Saturn" becomes "Saturn has the most moons."
    """
    answer = re.sub(r"\s*\([^)]*\)", "", answer or "").strip().rstrip(".!?").strip()
    if not answer or len(answer.split()) > 5:
        return None
    text = (question or "").strip()

    match = _LESSON_QUESTION_STATEMENT.match(text)
    if match:
        rest = _clean_phrase(match.group(1))
        return f"{answer} is {rest}." if rest else None

    match = _LESSON_WHICH_SUBJECT.match(text)
    if match:
        verb = match.group(2).strip()
        rest = _clean_phrase(match.group(3))
        return f"{answer} {verb} {rest}." if rest else None

    match = _LESSON_NAME_REQUEST.match(text)
    if match:
        article = match.group(1).strip()
        rest = _clean_phrase(match.group(2))
        return f"{answer} is {article} {rest}." if rest else None

    return None


# "Name some fruits." is a lesson about a whole list, not about one word.
_LESSON_NAME_LIST = re.compile(
    r"^name\s+(?:some|all|the\s+list\s+of)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)


def _list_statement_for_word(tenant_id: int | None, word: str) -> str | None:
    """A word taken from a list lesson becomes a sentence of its own.

    "Name some fruits." with the word "banana" becomes "Banana is a fruit."
    """
    target = _normalize_phrase(word)
    if not target:
        return None
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception:
        return None
    for card in cards:
        match = _LESSON_NAME_LIST.match((card.get("question") or "").strip())
        if not match:
            continue
        category = _clean_phrase(match.group(1))
        answer = card.get("answer") or ""
        parts = re.split(r"\bare\b", answer, maxsplit=1)
        items_text = parts[1] if len(parts) > 1 else answer
        items = [item.strip(" .!") for item in items_text.split(",")]
        if not any(_normalize_phrase(item) == target for item in items):
            continue
        singular = category[:-1] if category.endswith("s") and not category.endswith("ss") else category
        return f"{word.strip().capitalize()} is a {singular}."
    return None


def _lesson_statement_for_word(tenant_id: int | None, word: str) -> str | None:
    """The sentence the learner's lesson makes for this word.

    A word that was the answer to a lesson becomes the sentence that lesson
    was teaching, so the learner hears it as a sentence and not as a word.
    """
    target = _normalize_phrase(word)
    if not target:
        return None
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception:
        return None
    for card in cards:
        card_answer = card.get("answer", "")
        if _normalize_phrase(card_answer) != target:
            continue
        statement = _statement_from_lesson(card.get("question", ""), card_answer)
        if statement:
            return statement
    return None


def _token_matches(token: str, word: str) -> bool:
    """Whether a token in a sentence is the word that was asked for.

    "bee" is found in "bee's", because that is still the word bee.
    """
    if token == word:
        return True
    return token.startswith(word + "'")


_VERBS_AFTER_POSSESSIVE = {
    "is", "are", "was", "were", "has", "have", "had", "make", "makes", "made",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "say", "says", "said", "eat", "eats", "live", "lives", "run", "runs", "fly", "flies",
    "build", "builds", "give", "gives", "come", "comes", "go", "goes", "like", "likes",
    "love", "loves", "help", "helps", "need", "needs", "want", "wants", "work", "works",
    "play", "plays", "sleep", "sleeps", "swim", "swims", "sing", "sings", "write", "writes",
    "read", "reads", "grow", "grows", "walk", "walks", "jump", "jumps", "drink", "drinks",
    "dance", "dances", "cook", "cooks", "clean", "cleans", "laugh", "laughs", "cry", "cries",
    "smile", "smiles", "bring", "brings", "keep", "keeps", "sell", "sells", "buy", "buys",
    "find", "finds", "know", "knows", "think", "thinks", "see", "sees", "hear", "hears",
}
_POSSESSIVE_TYPED = re.compile(
    r"\b([A-Za-z]+)'s\s+(" + "|".join(sorted(_VERBS_AFTER_POSSESSIVE)) + r")\b",
    re.IGNORECASE,
)


def _is_naming_word(word: str) -> bool:
    """Whether a word names a thing, so an article belongs before it.

    The tagger of the language decides, so "black" the colour stays an
    adjective and "animal" is a naming word.
    """
    try:
        from app.services.nlp import get_nlp

        doc = get_nlp()(word)
    except Exception:
        return False
    if not doc or not doc[0].pos_:
        return False
    return doc[0].pos_ in {"NOUN", "PROPN"}


def _repair_sentence(sentence: str) -> str:
    """Fix the small grammar slips a sentence can carry.

    "Bee's make honey." is the plural "Bees make honey." and not a possessive,
    and "The cow is animal." is "The cow is an animal."
    """
    if not sentence:
        return sentence
    repaired = _POSSESSIVE_TYPED.sub(lambda match: f"{match.group(1)}s {match.group(2)}", sentence)

    return _add_missing_articles(repaired)


def _add_missing_articles(sentence: str) -> str:
    """Put the article back when a naming word stands on its own.

    "The cow is animal." becomes "The cow is an animal."
    """
    try:
        from app.services.nlp import get_nlp

        doc = get_nlp()(sentence)
    except Exception:
        return sentence
    insertions: list[tuple[int, str]] = []
    for index, token in enumerate(doc):
        if token.lower_ not in {"is", "are", "was", "were"}:
            continue
        if index + 1 >= len(doc):
            continue
        following = doc[index + 1]
        if following.pos_ not in {"NOUN", "PROPN"}:
            continue
        if following.tag_ in {"NNS", "NNPS"}:
            # A plural word does not take an article.
            continue
        article = "an" if following.text[:1].lower() in "aeiou" else "a"
        insertions.append((following.idx, article))
    # The words are put back where they were, so the punctuation stays as it was.
    text = sentence
    for position, article in reversed(insertions):
        text = text[:position] + f"{article} " + text[position:]
    return text


def _sentence_starts_with(sentence: str, word: str) -> bool:
    """Whether a sentence is about the word, because it begins with it.

    "A liquid is a state of matter." is about liquid, while "It changes from
    a solid to a liquid." only mentions it.
    """
    tokens = tokenize(sentence)
    if not tokens:
        return False
    if tokens[0] in {"a", "an", "the"} and len(tokens) > 1:
        first = tokens[1]
    else:
        first = tokens[0]
    return first == word or first.startswith(word + "'") or first.startswith(word + "s")


def make_sentence_from_words(tenant_id: int | None, words: list[str], strict: bool = False) -> str | None:
    """Build a simple grammatical sentence from the given words.

    A taught sentence that uses the words is preferred, otherwise the learner
    builds the sentence from the meaning it was taught for each word. In strict
    mode nothing is returned when the learner does not know the words, so the
    caller can fall back to the model's own sentence.
    """
    cleaned = []
    for word in words:
        for part in str(word).replace(",", " ").split():
            cleaned.append(part.strip().lower().strip("?.!"))
    cleaned = [word for word in cleaned if word and word not in {".", "!", "?"}]
    if not cleaned:
        return None

    content = [word for word in cleaned if word not in _FUNCTION_WORDS]
    if not content:
        content = cleaned

    meanings = _taught_word_meanings(tenant_id)
    describing = [word for word in cleaned if word in meanings and _is_describing_word(meanings[word])]
    nouns = [word for word in cleaned if word in meanings and word not in describing]
    others = [word for word in cleaned if word not in meanings and word not in _FUNCTION_WORDS]

    if nouns:
        described = [f"the {noun} is {meanings[noun]}" for noun in nouns]
        modifiers = describing + others
        if modifiers:
            return _repair_sentence(f"The {' '.join(modifiers)} {nouns[0]} is {meanings[nouns[0]]}.")
        if len(described) == 1:
            return _repair_sentence(f"{described[0][0].upper()}{described[0][1:]}.")
        return _repair_sentence(f"{', and '.join(described).capitalize()}.")

    corpus = [
        sentence
        for group in (
            _taught_sentences(tenant_id),
            _taught_answers(tenant_id),
            _tenant_lesson_answers(tenant_id),
        )
        for sentence in group
    ]

    # A sentence that is about the word begins with it, so that is the one
    # that is asked for. A short answer is not a sentence.
    if len(content) == 1:
        about = [
            sentence
            for sentence in corpus
            if _sentence_starts_with(sentence, content[0])
            and len(sentence.split()) >= 5
            and "/" not in sentence
        ]
        if about:
            return _repair_sentence(min(about, key=lambda sentence: len(sentence.split())))

    # A word that was the answer to a lesson becomes the sentence of that
    # lesson, so the learner gets a sentence and not a bare word.
    if len(content) == 1:
        statement = _lesson_statement_for_word(tenant_id, content[0])
        if statement:
            return statement

    # The shortest taught sentence that uses the word is the one that is about
    # that word. A list of many things is not.
    matches = [
        sentence
        for sentence in corpus
        if all(
            any(_token_matches(token, word) for token in tokenize(sentence))
            for word in content
        )
        and sentence.count(",") < 2
    ]
    if matches:
        # The model chooses the wording it knows best, and the shorter one
        # wins when it knows them equally well.
        best = max(
            matches,
            key=lambda sentence: (_wording_score(tenant_id, model, sentence), -len(sentence.split())),
        )
        return _repair_sentence(best)

    # A word that only lives in a list becomes a sentence of its own.
    if len(content) == 1:
        statement = _list_statement_for_word(tenant_id, content[0])
        if statement:
            return statement

    if others:
        if strict:
            return None
        return f"The {' '.join(others)} is a word I am still learning."

    return None


def answer_from_curriculum(question: str, tenant_id: int | None) -> str | None:
    """Recall letter, word, and lesson facts explicitly taught in the curriculum."""
    detail = answer_from_curriculum_detail(question, tenant_id)
    return detail["answer"] if detail else None


def answer_from_curriculum_detail(question: str, tenant_id: int | None, include_lessons: bool = True) -> dict | None:
    """The curriculum answer plus the question it was taught as.

    ``include_lessons`` is off for the normal chat path, where the taught
    lessons are matched semantically from the embedding index instead.
    """
    with _LOCK:
        state = _read(tenant_id)
        current_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        calculation = _answer_calculation_question(question)
        if calculation:
            return {"answer": calculation, "matched_question": ""}
        current_date_time = answer_current_date_time(question)
        if current_date_time:
            return {"answer": current_date_time, "matched_question": ""}
        spelling_answer = _answer_number_spelling_question(question)
        if spelling_answer:
            return {"answer": spelling_answer, "matched_question": ""}
        counting_answer = _answer_counting_question(question)
        if counting_answer:
            return {"answer": counting_answer, "matched_question": ""}
        number_sequence = _answer_number_sequence_question(question)
        if number_sequence:
            return {"answer": number_sequence, "matched_question": ""}
        alphabet_answer = _answer_alphabet_question(question)
        if alphabet_answer:
            return {"answer": alphabet_answer, "matched_question": ""}
        answer = _answer_letter_question(question, state["letters"])
        if answer:
            return {"answer": answer, "matched_question": ""}
        word = _asked_word(question)
        if word:
            for item in [*_tenant_words_for_stage(state, current_stage), *_foundation_words_for_stage(current_stage)]:
                if item["word"].casefold() == word:
                    return {
                        "answer": f"{item['word'].capitalize()} means {item['meaning']}.",
                        "matched_question": f"What does {item['word']} mean?",
                    }
        allowed_grades = _allowed_grades(current_stage)
        lessons = [item for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS] if item.get("grade_id", "nursery") in allowed_grades]
        if include_lessons:
            matched = _matched_learned_lesson(question, lessons)
            if matched:
                return {"answer": matched["answer"], "matched_question": matched.get("question", "")}
    return None


def preview_answer(question: str, tenant_id: int | None) -> dict[str, str]:
    """Answer a Trainer preview using only this learner's local curriculum."""
    with _LOCK:
        state = _read(tenant_id)
        current_stage = next((item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete")
        calculation = _answer_calculation_question(question)
        if calculation:
            return {"answer": calculation, "source": "number_cards"}
        age_answer = answer_assistant_age(question)
        if age_answer:
            return {"answer": age_answer, "source": "assistant_profile"}
        current_date_time = answer_current_date_time(question)
        if current_date_time:
            return {"answer": current_date_time, "source": "lesson_cards"}
        spelling_answer = _answer_number_spelling_question(question)
        if spelling_answer:
            return {"answer": spelling_answer, "source": "number_cards"}
        counting_answer = _answer_counting_question(question)
        if counting_answer:
            return {"answer": counting_answer, "source": "number_cards"}
        number_sequence = _answer_number_sequence_question(question)
        if number_sequence:
            return {"answer": number_sequence, "source": "number_cards"}
        alphabet_answer = _answer_alphabet_question(question)
        if alphabet_answer:
            return {"answer": alphabet_answer, "source": "lesson_cards"}
        answer = _answer_letter_question(question, state["letters"])
        if answer:
            return {"answer": answer, "source": "letters"}

        word = _asked_word(question)
        if word:
            for item in [*_tenant_words_for_stage(state, current_stage), *_foundation_words_for_stage(current_stage)]:
                if item["word"].casefold() == word:
                    return {"answer": f"{item['word'].capitalize()} means {item['meaning']}.", "source": "word_cards"}

        available_grades = _allowed_grades(current_stage)
        available_lessons = [
            item for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
            if item.get("grade_id", "nursery") in available_grades
        ]
        answer = _answer_learned_lesson(question, available_lessons)
        if answer:
            matched = _matched_learned_lesson(question, available_lessons)
            return {
                "answer": apply_requested_count(question, answer),
                "source": "lesson_cards",
                "matched_question": matched.get("question", "") if matched else "",
            }

        # A word the learner has not been taught still has a meaning in the
        # offline dictionary, so answer "what is boy here?" instead of giving up.
        dictionary_answer = dictionary.answer_dictionary_question(question)
        if dictionary_answer:
            return {"answer": dictionary_answer, "source": "dictionary"}

        next_match = _NEXT_WORD_REQUEST.search(question.strip())
        if next_match:
            prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
            options = _model(state, current_stage).predict_next(prefix)
            if options:
                return {
                    "answer": f"From my sentence lessons, after “{prefix}” I predict “{options[0]['word']}”.",
                    "source": "sentence_patterns",
                }

        sentence_match = _SENTENCE_REQUEST.search(question.strip())
        if sentence_match:
            requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-" )
            answer = _model(state, current_stage).generate_sentence(seed=requested) if requested else None
            if answer:
                return {"answer": answer, "source": "sentence_patterns"}

        examples = [*(item["text"] for item in state["lessons"]), *_foundation_sentences_for_stage(current_stage)]
        answer = _answer_from_sentence_examples(question, examples)
        if answer:
            return {"answer": answer, "source": "sentence_examples"}

        return {
            "answer": "I haven't learned an answer to that from my lessons yet. Teach me the word, sentence, or question and answer, then ask again.",
            "source": "not_learned",
        }


def answer_sentence_request(question: str, tenant_id: int | None) -> str | None:
    match = _SENTENCE_REQUEST.search(question.strip())
    if not match:
        return None
    requested = re.sub(r"[^A-Za-z'’-].*$", "", match.group(1)).strip(" '’-")
    if not requested:
        return "Tell me which word you want me to use in the sentence."
    sentence = make_sentence(tenant_id, requested)
    if sentence:
        return sentence
    return f"I haven't learned a sentence pattern for “{requested}” yet. Add a few example sentences with that word in Trainer → Language learner, then ask me again."


def answer_next_word_request(question: str, tenant_id: int | None) -> str | None:
    match = _NEXT_WORD_REQUEST.search(question.strip())
    if not match:
        return None
    prefix = (match.group(1) or match.group(2) or "").strip(" \t\r\n'\".,?!")
    words = predict_next(tenant_id, prefix)
    if not words:
        return f"I haven't learned what usually follows “{prefix}” yet. Add a sentence with those words in Trainer → Language learner."
    options = ", ".join(f"{item['word']} ({item['count']} example{'s' if item['count'] != 1 else ''})" for item in words[:3])
    return f"After “{prefix}”, my learned examples suggest: {options}. This is a word-pattern prediction, not a guaranteed continuation."
