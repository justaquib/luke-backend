"""Finding a taught lesson for a definition question, through a synonym.

A trainer may teach a word in the lesson's own words, while the person asking
uses another word for the same thing. The fallback asks the dictionary for
synonyms of the question word and looks each of them up among the lessons the
learner has reached. It answers only when exactly one lesson is reached, and
only when that lesson shares a real word with the question. The caller keeps
the dictionary's own answer when no lesson is reached.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services import dictionary
from app.services.connections import Connection
from app.services.language import matching
from app.services.language.topics import lesson_title

# How many synonyms of the question word are tried.
MAX_SYNONYMS = 6


@dataclass(frozen=True)
class FallbackAnswer:
    """The lesson a synonym reached, and the question it was taught as."""

    answer: str
    matched_question: str


@dataclass(frozen=True)
class FallbackAttempt:
    """What the fallback did: an answer, or nothing and the reason why."""

    answer: FallbackAnswer | None
    connection: Connection


def _lesson_key(lesson: dict) -> str:
    return str(lesson.get("id") or lesson.get("question") or lesson.get("answer") or "")


def _lesson_terms(lesson: dict) -> set[str]:
    title = lesson_title(str(lesson.get("question") or ""))
    return matching._stemmed_terms(f"{title} {lesson.get('answer') or ''}")


def _failed(detail: str, matched_question: str = "") -> FallbackAttempt:
    return FallbackAttempt(None, Connection("fallback", "fail", detail, matched_question))


def find_fallback(question: str, lessons: list[dict]) -> FallbackAttempt | None:
    """The one taught lesson a synonym of the question word reaches.

    ``None`` means the question is not a definition question, so there is
    nothing for this fallback to do.
    """
    parsed = dictionary.lookup_request(question)
    if not parsed or parsed[0] != "definition":
        return None
    word = parsed[1]
    synonyms = [
        synonym
        for synonym in dictionary.search_expansions(word, per_word=3, max_terms=MAX_SYNONYMS)
        # WordNet joins some phrases with an underscore ("looking_at"). A
        # phrase is not a word a lesson can be looked up as, and it matches
        # too loosely, so only single words are tried.
        if " " not in synonym
    ]
    if not synonyms:
        return _failed(f'No known synonym of "{word}" could be looked up in the lessons.')

    reached: dict[str, tuple[dict, str]] = {}
    for synonym in synonyms:
        lesson = matching._matched_learned_lesson(synonym, lessons)
        if lesson is not None and str(lesson.get("answer") or "").strip():
            reached.setdefault(_lesson_key(lesson), (lesson, synonym))
    if not reached:
        return _failed(f'No taught lesson matched a synonym of "{word}".')
    if len(reached) > 1:
        return _failed(
            f'Synonyms of "{word}" reached {len(reached)} different lessons, so none was chosen.'
        )

    lesson, synonym = next(iter(reached.values()))
    shared = sorted(matching._stemmed_terms(question) & _lesson_terms(lesson))
    matched_question = str(lesson.get("question") or "")
    if not shared:
        return _failed(
            f'"{synonym}" reached “{matched_question}”, but the lesson shares no '
            "key word with the question.",
            matched_question,
        )
    connection = Connection(
        "fallback",
        "pass",
        f"“{synonym}” reached “{matched_question}”; they share {', '.join(shared[:3])}.",
        matched_question,
    )
    return FallbackAttempt(
        FallbackAnswer(
            answer=str(lesson.get("answer") or "").strip(),
            matched_question=matched_question,
        ),
        connection,
    )
