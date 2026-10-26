"""What the model holds, what it can do with it, and what to teach next.

The capabilities follow the level ladder: a large collection of documents
does not make a model reason, so the checks here read what the brain learned
to do, not how much it was given. The next steps point at the part of the
app that feeds each of them.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services import vector_store
from app.services.brain_rungs import CORRECTION_LIMIT

LOGGER = logging.getLogger(__name__)

# The ladder is measured against the word model's vocabulary, the one live
# token list. The threshold counts the words the model writes with.
CANONICAL_VOCABULARY_FLOOR = 400


def coverage(tenant_id: int | None) -> dict[str, Any]:
    """What the assistant can read today: books, trained answers, lessons."""
    counted = {
        "documents": 0,
        "chunks": 0,
        "trained_answers": 0,
        "lessons": 0,
        "passed_grades": 0,
    }
    try:
        from sqlalchemy import select

        from app.db.database import Document, Faq, SessionLocal

        with SessionLocal() as db:
            document_query = select(Document)
            answer_query = select(Faq).where(Faq.status == "approved")
            if tenant_id is not None:
                document_query = document_query.where(Document.tenant_id == tenant_id)
                answer_query = answer_query.where(Faq.tenant_id == tenant_id)
            counted["documents"] = len(
                {row.source for row in db.execute(document_query).scalars().all()}
            )
            counted["trained_answers"] = len(db.execute(answer_query).scalars().all())
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.warning("The documents and answers could not be counted: %s", error)
    try:
        counted["chunks"] = vector_store.collection_size()
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.warning("The chunks could not be counted: %s", error)
    return counted


def _base_capabilities(
    knowledge: dict[str, Any],
    brain_state: dict[str, Any],
    canonical: Any,
    vocabulary_ready: bool,
) -> list[dict[str, Any]]:
    """The capabilities the ladder starts with: material and recall."""
    return [
        {
            "key": "taught",
            "name": "Taught",
            "ready": knowledge.get("lessons", 0) > 0,
            "detail": f"{knowledge.get('lessons', 0)} lessons taught",
        },
        {
            "key": "recalled",
            "name": "Recalled",
            "ready": (
                knowledge.get("lessons", 0) > 0
                and brain_state.get("trained", False)
                and vocabulary_ready
            ),
            "detail": f"{len(canonical)} words it can write with",
        },
        {
            "key": "searching",
            "name": "Searching",
            "ready": knowledge.get("lessons", 0) > 0,
            "detail": "searches lessons, answers, intents and documents",
        },
        {
            "key": "explaining",
            "name": "Explaining",
            "ready": True,
            "detail": "shows the reasoning and the nodes it visited",
        },
        {
            "key": "writing",
            "name": "Writing",
            "ready": bool(brain_state.get("trained")),
            "detail": "writes its own sentences",
        },
    ]


def _learned_capabilities(
    knowledge: dict[str, Any],
    brain_state: dict[str, Any],
    canonical: Any,
) -> list[dict[str, Any]]:
    """The capabilities training adds: feeling, curiosity, correction, depth."""
    return [
        {
            "key": "empathetic",
            "name": "Empathetic",
            "ready": bool(brain_state.get("emotion_ready")),
            "detail": "reads the feeling in a message",
        },
        {
            "key": "curious",
            "name": "Curious",
            "ready": bool(brain_state.get("clarify_ready")),
            "detail": "ask back when the evidence is thin",
        },
        {
            "key": "self_correcting",
            "name": "Self-correcting",
            "ready": brain_state.get("corrections", 0) > 0,
            "detail": "learn from the trainer's corrections and remember them",
        },
        {
            "key": "examined",
            "name": "Examined",
            "ready": knowledge.get("passed_grades", 0) > 0,
            "detail": f"{knowledge.get('passed_grades', 0)} grade exams passed",
        },
        {
            "key": "autonomous",
            "name": "Autonomous",
            "ready": (
                knowledge.get("passed_grades", 0) > 0
                and brain_state.get("corrections", 0) > 0
                and brain_state.get("clarify_ready")
                and len(canonical) >= CANONICAL_VOCABULARY_FLOOR
            ),
            "detail": "keep its own knowledge in order",
        },
    ]


def model_capabilities(
    tenant_id: int | None,
    brain_state: dict[str, Any],
    knowledge: dict[str, Any],
) -> list[dict[str, Any]]:
    """What the brain can actually do, in the order the levels need it.

    The level follows the capabilities, not the size of the data set: a large
    collection of documents does not make a model reason.

    Every vocabulary check here reads the word model's vocabulary, so the
    words it counts are the words the model actually uses.
    """
    from app.services import vocabulary as vocabulary_service

    canonical = vocabulary_service.canonical(tenant_id)
    vocabulary_ready = len(canonical) > 0 and len(canonical.index) == len(canonical)
    return [
        *_base_capabilities(knowledge, brain_state, canonical, vocabulary_ready),
        *_learned_capabilities(knowledge, brain_state, canonical),
    ]


def _document_steps(counted: dict[str, Any]) -> list[dict[str, Any]]:
    """The first step: give the model something to read."""
    documents = counted.get("documents", 0)
    chunks = counted.get("chunks", 0)
    if documents == 0:
        return [
            {
                "action": "add_document",
                "title": "Give it a book to read",
                "detail": (
                    "Luke has no document to read yet. Add a PDF, a novel or any "
                    "related file in the Trainer's Documents tab: the pages are "
                    "split into chunks, every chunk is embedded for meaning, and "
                    "the passages are read back word by word when a question "
                    "needs them."
                ),
                "link": "/trainer?tab=documents",
                "link_label": "Add a document",
            }
        ]
    if chunks < 40:
        return [
            {
                "action": "add_document",
                "title": "Give it more pages to read",
                "detail": (
                    f"Only {chunks} chunks are indexed. More pages means more "
                    "text chunked, embedded and read word by word."
                ),
                "link": "/trainer?tab=documents",
                "link_label": "Add a document",
            }
        ]
    return []


def _answer_step(counted: dict[str, Any]) -> list[dict[str, Any]]:
    """The step that adds taught answers, when there are few."""
    if counted.get("trained_answers", 0) >= 5:
        return []
    return [
        {
            "action": "add_answer",
            "title": "Store more trained answers",
            "detail": (
                "The Trainer keeps the question and answer pairs. The more it "
                "holds, the more it can answer without guessing."
            ),
            "link": "/trainer?tab=new",
            "link_label": "Add a Q&A",
        }
    ]


def _rung_step(rungs: list[dict[str, Any]], kept: int) -> list[dict[str, Any]]:
    """The step that passes the first rung the model fails."""
    hints = {
        "0.0.1": "Teach Luke a question and answer it cannot already answer.",
        "0.0.2": (
            "Teach the same answers again in different words, so it can find "
            "them when the question is reworded."
        ),
        "0.0.3": "Teach the topic rather than the exact question, and add a few phrasings of it.",
        "0.0.5": "Teach the reason as well as the answer, so it can explain it when asked.",
        "0.0.6": (
            "Mark an answer wrong in the chat and give the right one. "
            f"{max(0, CORRECTION_LIMIT - kept)} more kept corrections and Luke "
            "passes this rung."
        ),
        "0.0.7": "Pass the final exam of the grade it is studying.",
        "0.0.8": (
            "Give it a few corrections so it learns where its knowledge ends "
            "and says so instead of guessing."
        ),
        "0.0.9": "Teach more questions on one topic so it holds a whole subject.",
    }
    for rung in rungs:
        if rung["probes"] and rung["rate"] < 0.6:
            return [
                {
                    "action": "add_lesson",
                    "title": f"Pass the Luke {rung['rung']} test",
                    "detail": hints.get(rung["rung"], "Teach more of what Luke should know."),
                    "link": "/trainer?tab=language",
                    "link_label": "Teach a lesson",
                }
            ]
    return []


def _first_lesson_step(knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    """The step that asks for the first lesson, when none is taught."""
    if knowledge.get("lessons", 0) != 0:
        return []
    return [
        {
            "action": "add_lesson",
            "title": "Add your first lesson",
            "detail": "Luke starts empty. Teach it a question and the answer it should give.",
            "link": "/trainer?tab=language",
            "link_label": "Open the learner",
        }
    ]


def _grade_steps(knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    """The mock test and the final exam of the grade the learner is in."""
    grade = knowledge.get("stage", "nursery")
    if not grade or grade == "complete":
        return []
    from app.services import language_logic

    grade_name = next(
        (stage["title"] for stage in language_logic.STAGES if stage["id"] == grade),
        str(grade).replace("_", " ").title(),
    )
    return [
        {
            "action": "mock_test",
            "title": f"Take a mock test for {grade_name}",
            "detail": (
                "A mock test is practice. It never advances the grade, so you "
                "can see what still needs learning."
            ),
            "link": "/trainer?tab=language",
            "link_label": "Open the learner",
        },
        {
            "action": "final_exam",
            "title": f"Pass the {grade_name} final exam to advance the school grade",
            "detail": (
                "The final exam moves Luke to the next grade once it answers "
                "at least 80% of the questions correctly."
            ),
            "link": "/trainer?tab=language",
            "link_label": "Open the learner",
        },
    ]


def _more_lessons_step(knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    """The step that asks for more lessons, below forty."""
    if knowledge.get("lessons", 0) >= 40:
        return []
    return [
        {
            "action": "add_lesson",
            "title": "Add more knowledge to the school",
            "detail": (
                f"{knowledge.get('lessons', 0)} lessons are taught. Every "
                "lesson is a question and answer the learner can use, and the "
                "school grades are built on them."
            ),
            "link": "/trainer?tab=language",
            "link_label": "Teach a lesson",
        }
    ]


def _lesson_steps(knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    """The lesson and school steps: first lesson, mock test, final exam."""
    return [
        *_first_lesson_step(knowledge),
        *_grade_steps(knowledge),
        *_more_lessons_step(knowledge),
    ]


def _correction_step(brain_state: dict[str, Any]) -> list[dict[str, Any]]:
    """The step that asks for one correction, when none was taught."""
    if brain_state.get("corrections", 0) != 0:
        return []
    return [
        {
            "action": "rate_answers",
            "title": "Mark one answer wrong and correct it",
            "detail": (
                "In the chat, mark an answer wrong and type the right one. Luke "
                "learns it, and the correction becomes part of its memory."
            ),
            "link": "/chat",
            "link_label": "Open the chat",
        }
    ]


def next_steps(
    brain_state: dict[str, Any],
    knowledge: dict[str, Any],
    rungs: list[dict[str, Any]],
    coverage_map: dict[str, Any] | None = None,
    model_name: str = "Luke",
) -> list[dict[str, Any]]:
    """What to do next to advance, in the order it helps most.

    Each step points at the part of the app that feeds the assistant: the
    documents it reads, the answers the Trainer stores, or the lessons it
    learns in the school.
    """
    counted = coverage_map or {}
    kept = brain_state.get("learned_corrections", 0)
    steps = [
        *_document_steps(counted),
        *_answer_step(counted),
        *_rung_step(rungs, kept),
        *_lesson_steps(knowledge),
        *_correction_step(brain_state),
    ]
    steps.append(
        {
            "action": "train_brain",
            "title": "Train the brain",
            "detail": (
                "Training rebuilds the word model and the heads from "
                "everything Luke has been taught, then tests the level again."
            ),
            "link": "",
            "link_label": "",
        }
    )
    if model_name and model_name != "Luke":
        for step in steps:
            step["title"] = step["title"].replace("Luke", model_name)
            step["detail"] = step["detail"].replace("Luke", model_name)
    return steps
