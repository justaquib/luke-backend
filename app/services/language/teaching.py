"""Teaching the tenant's lessons: the pairs, the words and the passages.

A lesson is normally a question and an answer. A story, a lesson or a topic
is taught as one passage instead, and it is filed under the title it was given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from app.config import settings
from app.services.language import runtime
from app.services.language.curriculum import _GRADE_IDS, _LEGACY_STAGES, STAGES
from app.services.language.helpers import _normalize_phrase, _parse_lesson_pairs
from app.services.language.runtime import parse_subjects
from app.services.tiny_language_model import tokenize

# The longest lesson the learner accepts.
MAX_LESSON_CHARACTERS = 3000

# The mark that says a lesson was taught as a passage and not as a pair.
STORY_PREFIX = "Story:"

# The mark on a lesson card that says it was taught as a passage.
STORY_ENTRY = "story"


@dataclass(frozen=True)
class StoryLesson:
    """A story, a lesson or a topic the trainer wants to teach as a passage."""

    text: str
    title: str
    stage: str = "nursery"
    subject: str = "General"
    skip_duplicates: bool = True


def teach(
    tenant_id: int | None,
    text: str,
    stage: str = "nursery",
    subject: str = "General",
    skip_duplicates: bool = True,
) -> dict:
    """Teach the learner one or more question and answer lessons.

    A lesson that repeats one the learner already holds is kept: it is marked
    as a duplicate so the trainer can decide what to do with it.
    """
    raw = _checked_text(text)
    stage = _checked_stage(stage)
    topics = parse_subjects(subject)
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        _ensure_grade_allowed(state, stage)
        result = _teach_by_stage(
            {
                "state": state,
                "raw": raw,
                "stage": stage,
                "subject": ", ".join(topics),
                "topics": topics,
                "skip_duplicates": skip_duplicates,
                "now": datetime.now(UTC).isoformat(),
            }
        )
        return _save_taught(tenant_id, state, result, stage)


def teach_story(tenant_id: int | None, story: StoryLesson) -> dict:
    """Teach a story, a lesson or a topic as one passage.

    A passage is not a question and an answer, so it is filed under the title
    it was given. The learner can ask for it by that title or by its topic,
    and every sentence of it is taught on its own as well.
    """
    raw = _checked_text(story.text)
    stage = _checked_stage(story.stage)
    title = (story.title or "").strip()
    if not title:
        raise ValueError("Give the story, the lesson or the topic a name.")
    topics = parse_subjects(story.subject)
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        _ensure_grade_allowed(state, stage)
        result = _teach_story_card(
            {
                "state": state,
                "raw": raw,
                "title": title,
                "stage": stage,
                "subject": ", ".join(topics),
                "topics": topics,
                "skip_duplicates": story.skip_duplicates,
                "now": datetime.now(UTC).isoformat(),
            }
        )
        return _save_taught(tenant_id, state, result, stage)


def _checked_text(text: str) -> str:
    """The lesson text, checked against the limits it has to meet."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Enter one or more example sentences first.")
    if len(raw) > MAX_LESSON_CHARACTERS:
        raise ValueError("Please keep a lesson under 3,000 characters.")
    return raw


def _checked_stage(stage: str) -> str:
    """The grade the lesson is taught in, checked against the ones there are."""
    if stage not in _GRADE_IDS and stage not in _LEGACY_STAGES:
        raise ValueError("Choose a grade from Nursery through Graduation.")
    return stage


def _ensure_grade_allowed(state: dict, stage: str) -> None:
    stage_order = [item["id"] for item in STAGES]
    current_stage = next(
        (item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete"
    )
    if (
        stage in _GRADE_IDS
        and current_stage != "complete"
        and stage_order.index(stage) > stage_order.index(current_stage)
    ):
        current_title = next(item["title"] for item in STAGES if item["id"] == current_stage)
        stage_title = next(item["title"] for item in STAGES if item["id"] == stage)
        raise ValueError(f"Pass the {current_title} test before teaching {stage_title} lessons.")


def _teach_by_stage(job: dict) -> dict:
    """Teach into the state, in the way the grade asks for."""
    state = job["state"]
    stage = job["stage"]
    if stage == "letters":
        return _teach_letters(state, job["raw"])
    if stage == "words":
        return _teach_words(state, job["raw"], job["skip_duplicates"], job["now"])
    if stage == "sentences":
        return _teach_sentences(state, job["raw"], job["skip_duplicates"], job["now"])
    return _teach_lesson_cards(job)


def _save_taught(tenant_id: int | None, state: dict, result: dict, stage: str) -> dict:
    """Save what was taught and give back the learner's new state."""
    runtime._save(tenant_id, state)
    learner = runtime.get_state(tenant_id)
    learner["added_lessons"] = len(result["added"])
    learner["last_learned"] = result["added"]
    learner["skipped_duplicates"] = result["skipped"]
    learner["added_lesson_cards"] = (
        result["cards"] if stage in _GRADE_IDS or stage == "lessons" else []
    )
    return learner


def _teach_letters(state: dict, raw: str) -> dict:
    letters = [char.lower() for char in raw if char.isalpha() and char.isascii()]
    if not letters:
        raise ValueError("Enter alphabet letters such as A B C or a-z.")
    state["letters"] = "".join(dict.fromkeys(state["letters"] + letters))
    return {"added": letters, "skipped": [], "cards": []}


def _teach_words(state: dict, raw: str, skip_duplicates: bool, now: str) -> dict:
    new_cards: list[dict] = []
    added: list[str] = []
    skipped: list[dict] = []
    known_words = {card.get("word", "").lower() for card in state["word_cards"]}
    for line in raw.splitlines():
        if not _valid_word_line(line):
            raise ValueError(
                "Each word needs one vocabulary word and a short meaning after a colon."
            )
        word, meaning = (part.strip() for part in line.split(":", 1))
        if skip_duplicates and word.lower() in known_words:
            skipped.append({"word": word.lower(), "meaning": meaning})
            continue
        known_words.add(word.lower())
        new_cards.append({"word": word.lower(), "meaning": meaning, "created_at": now})
        added.append(word.lower())
    state["word_cards"].extend(new_cards)
    state["word_cards"] = state["word_cards"][-1000:]
    return {"added": added, "skipped": skipped, "cards": new_cards}


def _valid_word_line(line: str) -> bool:
    """Whether a line holds one vocabulary word and a short meaning."""
    if ":" not in line:
        return False
    word, meaning = (part.strip() for part in line.split(":", 1))
    return bool(word and meaning) and len(tokenize(word)) == 1


def _teach_sentences(state: dict, raw: str, skip_duplicates: bool, now: str) -> dict:
    normalized: list[str] = []
    skipped: list[dict] = []
    known_sentences = {_normalize_phrase(item.get("text", "")) for item in state["lessons"]}
    for sentence in _split_sentences(raw):
        cleaned = _checked_sentence(sentence)
        if skip_duplicates and _normalize_phrase(cleaned) in known_sentences:
            skipped.append({"text": cleaned})
            continue
        known_sentences.add(_normalize_phrase(cleaned))
        normalized.append(cleaned)
    state["lessons"].extend({"text": sentence, "created_at": now} for sentence in normalized)
    state["lessons"] = state["lessons"][-2000:]
    return {"added": normalized, "skipped": skipped, "cards": []}


def _teach_lesson_cards(job: dict) -> dict:
    state = job["state"]
    pairs = _parse_lesson_pairs(job["raw"])
    if not pairs:
        raise ValueError(
            "Enter lessons as question and answer pairs, using Q: on one line and A: on the next."
        )
    added: list[str] = []
    skipped: list[dict] = []
    new_cards: list[dict] = []
    known_lessons = _known_lessons(state)
    for raw_question, raw_answer in pairs:
        cleaned_question, cleaned_answer = raw_question.strip(), raw_answer.strip()
        if not cleaned_question or not cleaned_answer:
            raise ValueError("Each lesson needs both a question and its answer.")
        card = {
            "id": uuid4().hex,
            "question": cleaned_question,
            "answer": cleaned_answer,
            "grade_id": job["stage"] if job["stage"] in _GRADE_IDS else "nursery",
            "subject": job["subject"],
            "subjects": list(job["topics"]),
            "confidence": settings.base_answer_confidence,
            "created_at": job["now"],
        }
        if job["skip_duplicates"] and _lesson_key(card) in known_lessons:
            skipped.append({"question": cleaned_question, "answer": cleaned_answer})
            continue
        known_lessons.add(_lesson_key(card))
        new_cards.append(card)
        added.append(cleaned_question)
    state["lesson_cards"].extend(new_cards)
    state["lesson_cards"] = state["lesson_cards"][-500:]
    return {"added": added, "skipped": skipped, "cards": new_cards}


def _teach_story_card(job: dict) -> dict:
    """File a passage under the title it was given.

    Every sentence of the passage is taught on its own as well, so the writing
    model learns the wording of the passage.
    """
    state = job["state"]
    sentences = _passage_sentences(job["raw"])
    passage = " ".join(sentences)
    question = f"{STORY_PREFIX} {job['title']}"
    card = {
        "id": uuid4().hex,
        "question": question,
        "answer": passage,
        "grade_id": job["stage"] if job["stage"] in _GRADE_IDS else "nursery",
        "subject": job["subject"],
        "subjects": list(job["topics"]),
        "confidence": settings.base_answer_confidence,
        "entry": STORY_ENTRY,
        "created_at": job["now"],
    }
    known_lessons = _known_lessons(state)
    if job["skip_duplicates"] and _lesson_key(card) in known_lessons:
        return {"added": [], "skipped": [{"question": question, "answer": passage}], "cards": []}
    known_lessons.add(_lesson_key(card))
    state["lesson_cards"].append(card)
    state["lesson_cards"] = state["lesson_cards"][-500:]
    state["lessons"].extend({"text": sentence, "created_at": job["now"]} for sentence in sentences)
    state["lessons"] = state["lessons"][-2000:]
    return {"added": [question], "skipped": [], "cards": [card]}


def _passage_sentences(raw: str) -> list[str]:
    """The sentences of a passage, each one checked on its own."""
    sentences = [_checked_sentence(sentence) for sentence in _split_sentences(raw)]
    if not sentences:
        raise ValueError("Write the story, the lesson or the topic as at least one sentence.")
    return sentences


def _split_sentences(raw: str) -> list[str]:
    """The sentences of a text, one by one."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", raw) if part.strip()]


def _checked_sentence(sentence: str) -> str:
    """One sentence, checked on its own and given its full stop."""
    words = [word for word in tokenize(sentence) if word not in {".", "!", "?"}]
    if len(words) < 2:
        raise ValueError(
            "Each sentence needs at least two words so the model can learn a word sequence."
        )
    if len(words) > 40:
        raise ValueError("Keep each sentence to 40 words or fewer.")
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _known_lessons(state: dict) -> set[tuple[str, str]]:
    """The question and answer pairs the learner already holds."""
    return {
        (
            _normalize_phrase(card.get("question", "")),
            _normalize_phrase(card.get("answer", "")),
        )
        for card in state["lesson_cards"]
    }


def _lesson_key(card: dict) -> tuple[str, str]:
    """The key a lesson is filed under, so a repeat is found."""
    return (
        _normalize_phrase(card.get("question", "")),
        _normalize_phrase(card.get("answer", "")),
    )
