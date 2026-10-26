"""HTTP routes for the language learner: the lessons, the tests and the words.

The routes stay thin: they check the caller, read the request, and hand the
work to the service that owns it. The brain's own routes live in
`language_brain.py`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User, get_session
from app.schemas import (
    DictionaryLearnBody,
    DuplicateAcceptBody,
    LessonBody,
    LessonUpdateBody,
    PreviewBody,
    SentenceBody,
    TestQuestionBody,
    TestScoreBody,
    TestStartBody,
)
from app.services import answer_streams, streaming
from app.services import dictionary as dictionary_service
from app.services import reasoning as reasoning_service
from app.services.language import duplicates as duplicates_service
from app.services.language import lessons as lesson_service
from app.services.language import preview as preview_service
from app.services.language import reporting
from app.services.language import runtime as runtime_service
from app.services.language import stages as stage_service
from app.services.language import teaching as teaching_service
from app.services.language import writing as writing_service

router = APIRouter(prefix="/language-model", tags=["language-model"])


@router.get("/state")
def language_model_state(
    prefix: str = "",
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """The learner's saved state, with the approved answers alongside it."""
    tenant_id = tenant_id_of(user)
    state = runtime_service.get_state(tenant_id, prefix)
    state["trainer_answers"] = reporting.trainer_answers(db, tenant_id)
    return state


@router.get("/words")
def language_model_words(
    offset: int = 0,
    limit: int = 100,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """The words the answering model can write with, one page at a time."""
    return runtime_service.vocabulary_page(tenant_id_of(user), offset, limit)


@router.get("/graph")
def language_model_graph(
    subject: str = "",
    limit: int = 90,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """The shape of what this tenant has learned."""
    return reasoning_service.knowledge_graph(tenant_id_of(user), subject, limit)


@router.get("/lessons/duplicates")
def language_model_duplicates(
    stage: str = "",
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """The lessons that repeat each other, so the trainer can tidy them up."""
    tenant_id = tenant_id_of(user)
    groups = duplicates_service.duplicate_groups(tenant_id, stage)
    duplicates = sum(len(group["cards"]) - 1 for group in groups)
    return {
        "groups": groups,
        "total_questions": len(lesson_service.all_lesson_cards(tenant_id)),
        "duplicate_count": duplicates,
    }


@router.post("/lessons/duplicates/accept")
def accept_language_model_duplicates(
    body: DuplicateAcceptBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Keep every lesson of a group: the repeat is on purpose."""
    return duplicates_service.accept_duplicate(tenant_id_of(user), body.key)


@router.post("/lessons/check")
def check_language_lesson(
    body: LessonBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Report what the learner already knows about this lesson."""
    tenant_id = tenant_id_of(user)
    try:
        report = lesson_service.check_duplicate_lessons(
            tenant_id, body.lesson, body.stage, body.subject
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    report["faqs"] = reporting.candidate_answers(body.lesson, db, tenant_id)
    report["intents"] = reporting.matching_intents(body.lesson, db, tenant_id)
    return report


@router.post("/lessons")
def teach_language_model(
    body: LessonBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Teach the learner a lesson and return its new state.

    A lesson is a question and an answer, or a story, a lesson or a topic
    that is taught as one passage. A lesson that repeats one the learner
    already holds is kept: it is marked as a duplicate so the trainer can
    decide what to do with it.
    """
    tenant_id = tenant_id_of(user)
    try:
        if body.kind == "story":
            return teaching_service.teach_story(
                tenant_id,
                teaching_service.StoryLesson(
                    text=body.lesson,
                    title=body.title,
                    stage=body.stage,
                    subject=body.subject,
                    skip_duplicates=body.skip_duplicates,
                ),
            )
        return teaching_service.teach(
            tenant_id,
            body.lesson,
            body.stage,
            body.subject,
            body.skip_duplicates,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.put("/lessons/{lesson_id}")
def update_language_lesson(
    lesson_id: str,
    body: LessonUpdateBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Edit a lesson the learner already holds."""
    try:
        return lesson_service.update_lesson_answer(
            tenant_id_of(user),
            lesson_id,
            body.answer,
            body.question,
            body.confidence,
            body.subject,
        )
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.delete("/lessons/{lesson_id}")
def remove_language_lesson(
    lesson_id: str,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Remove a lesson from the learner."""
    try:
        return lesson_service.remove_lesson(tenant_id_of(user), lesson_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@router.post("/preview")
def preview_language_answer(
    body: PreviewBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Answer a question with the learner's own material only."""
    result = preview_service.preview_payload(
        body.question, tenant_id_of(user), db, body.skip_trainer
    )
    return result.payload


@router.post("/preview/stream")
def preview_language_answer_stream(
    body: PreviewBody,
    user: User = Depends(get_bearer_user),
) -> StreamingResponse:
    """The learner preview as server-sent events, one real stage at a time."""
    return StreamingResponse(
        streaming.frames(
            answer_streams.preview_stream(body.question, tenant_id_of(user), body.skip_trainer)
        ),
        media_type="text/event-stream",
        headers=streaming.SSE_HEADERS,
    )


@router.post("/test/start")
def start_language_test(
    body: TestStartBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Start a mock test, a final exam, or a trainer led test."""
    try:
        return stage_service.start_test(tenant_id_of(user), body.stage, body.kind)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/test/question")
def ask_language_test_question(
    body: TestQuestionBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Ask the learner a question during a grade test."""
    tenant_id = tenant_id_of(user)
    try:
        return stage_service.ask_test_question(
            tenant_id,
            body.question,
            reporting.trained_answer(body.question, db, tenant_id),
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/test/score")
def score_language_test_answer(
    body: TestScoreBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Score the learner's last answer in a grade test."""
    try:
        return stage_service.score_test_answer(tenant_id_of(user), body.passed)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/test/finish")
def finish_language_test(
    body: TestScoreBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Close the grade test and open the next grade when it is passed."""
    try:
        return stage_service.finish_test(tenant_id_of(user), body.passed)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.get("/dictionary")
def lookup_dictionary_word(
    word: str,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Look a word up in the offline dictionary."""
    entry = dictionary_service.lookup_word(word)
    if not entry:
        raise HTTPException(404, f"No dictionary entry for {word}.")
    return entry


@router.post("/dictionary/learn")
def learn_dictionary_word(
    body: DictionaryLearnBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Teach the learner a word and the meaning it should remember."""
    try:
        state = stage_service.add_dictionary_word(
            tenant_id_of(user), body.word, body.meaning, body.stage
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    entry = dictionary_service.lookup_word(body.word) or {
        "word": body.word.lower(),
        "part_of_speech": "",
        "definitions": [body.meaning],
    }
    return {"state": state, "entry": entry}


@router.post("/sentence")
def generate_language_sentence(
    body: SentenceBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Build a sentence from the words the learner knows."""
    words = [word for word in body.words if word.strip()]
    if not words and body.word:
        words = [body.word]
    sentence = writing_service.make_sentence_from_words(tenant_id_of(user), words)
    if not sentence:
        sentence = writing_service.make_sentence(tenant_id_of(user))
    return {"sentence": sentence}
