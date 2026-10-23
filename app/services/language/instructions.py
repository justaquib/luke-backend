"""The instructions the tenant's own material can be asked for.

A lesson is a question and an answer. That is enough to look the answer up, but
not enough to answer a request that was never worded the same way: "write a
short poem about the rain" asks for the lesson "Poem: Rain Rain Go Away" even
though the two share almost no words.

This module turns every lesson into a set of instructions, so the model learns
the *request* as well as the answer. The forms are written by hand, and the
ones that are only used to measure the model are kept apart from the ones it
trains on, so the score says something true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.language.topics import card_key, topic_index

# The ways a lesson is asked for. The model trains on these.
QUESTION_FORMS: tuple[str, ...] = (
    "{question}",
    "Q: {question}\nA:",
    "Answer: {question}",
    "Tell me: {question}",
    "Question: {question}\nAnswer:",
    "Please answer: {question}",
    "My question is: {question}",
    "I would like to know: {question}",
)

# The ways a lesson is asked for that the model is never trained on. They are
# the measure: a model that only memorises will fail every one of them.
HELD_OUT_QUESTION_FORMS: tuple[str, ...] = (
    "Could you please answer: {question}",
    "I would like to know the answer to this. {question}",
    "Respond to the following. {question}",
)

# How a topic is asked for, per kind of lesson. These are the shapes the
# request path reads, so the model is trained on exactly what it is asked.
# There are many of them on purpose: a model that sees one wording per
# subject learns that wording, and a model that sees many learns the subject.
TOPIC_FORMS: dict[str, tuple[str, ...]] = {
    "poem": (
        "Write a short poem about the {topic}.",
        "Write a poem about {topic}.",
        "Can you write me a poem about {topic}?",
        "Write me a poem about {topic}.",
        "Say a poem about {topic}.",
        "Recite a poem about {topic}.",
        "Make a poem about {topic}.",
        "I would like a poem about {topic}.",
        "Please write a poem about {topic}.",
        "Write a little poem about {topic}.",
        "Write a new poem about {topic}.",
        "Could you write a poem about {topic}?",
        "Give me a poem about {topic}.",
        "Compose a poem about {topic}.",
        "Write something about {topic} in a poem.",
        "Write a poem on the subject of {topic}.",
        "Write me a short poem about {topic}.",
        "Can you make a poem about {topic}?",
        "Put a poem about {topic} in writing.",
        "I need a poem about {topic}.",
        "Make up a poem about {topic}.",
    ),
    "list": (
        "Name some {topic}.",
        "List {topic} for me.",
        "Give me some {topic}.",
        "Tell me some {topic}.",
        "Name a few {topic}.",
        "What are some {topic}?",
        "Give me a list of {topic}.",
        "I want a list of {topic}.",
        "Please list the {topic}.",
        "Could you name some {topic}?",
        "List out some {topic}.",
        "Give me a few {topic}.",
        "Tell me a few {topic}.",
        "I need some {topic}.",
        "List me some {topic}.",
        "List some {topic} for me.",
    ),
    "fact": (
        "Tell me about {topic}.",
        "What do you know about {topic}?",
        "Give me information about {topic}.",
        "Tell me everything about {topic}.",
        "Give me the details about {topic}.",
        "I would like to know about {topic}.",
        "Explain {topic} to me.",
        "Tell me about the {topic}.",
        "What can you tell me about {topic}?",
        "Give me a few facts about {topic}.",
        "I want to hear about {topic}.",
        "Tell me what you know about {topic}.",
        "Give me a bit about {topic}.",
        "Give me some info about {topic}.",
        "I want to learn about {topic}.",
        "What can you say about {topic}?",
        "Tell me about {topic} please.",
    ),
}

# The topic forms the model is never trained on. They are worded like the
# trained ones but not the same, so answering them shows the model learned
# the shape of the request and not the words of one request.
HELD_OUT_TOPIC_FORMS: dict[str, tuple[str, ...]] = {
    "poem": ("Write a small poem about {topic}.", "Make me a poem about {topic}."),
    "list": ("List some {topic} for me.", "Give me the {topic} please."),
    "fact": ("What is there to know about {topic}?", "Tell me about the {topic} please."),
}

# The words that say a lesson is a joke, a poem, or a general-knowledge fact.
_LEADING_PREFIXES = (
    "poem:",
    "gk:",
    "science:",
    "grammar:",
    "math:",
    "reading:",
    "tiny:",
    "riddle:",
    "story:",
    "joke:",
    "general knowledge:",
)


@dataclass(frozen=True)
class Instruction:
    """One request and the answer that answers it."""

    instruction: str
    response: str
    topic: str = ""
    kind: str = "fact"
    held_out: bool = False


def lesson_kind(card: dict[str, Any]) -> str:
    """What kind of lesson this is: a poem, a list of things, or a fact."""
    subjects = " ".join(card.get("subjects") or [card.get("subject") or ""]).lower()
    question = (card.get("question") or "").lower()
    if "poem" in subjects or question.startswith("poem:"):
        return "poem"
    answer = (card.get("answer") or "").lower()
    if answer and "," in answer and answer.count(",") >= 2:
        return "list"
    return "fact"


def _question_forms(question: str, forms: tuple[str, ...]) -> list[str]:
    """The instructions that ask for a lesson by its question."""
    return [form.format(question=question) for form in forms]


def _topic_forms(kind: str, topic: str, forms: dict[str, tuple[str, ...]]) -> list[str]:
    """The instructions that ask for a lesson by its subject."""
    return [form.format(topic=topic) for form in forms.get(kind, TOPIC_FORMS["fact"])]


def instructions_for(card: dict[str, Any], topics: list[str] | None = None) -> list[Instruction]:
    """Every way this lesson can be asked for, and the answer to each.

    The topics are the ones that name this lesson and few others. Without
    them the lesson is only asked for by its own question.
    """
    question = str(card.get("question") or "").strip()
    answer = str(card.get("answer") or "").strip()
    if not question or not answer:
        return []
    kind = lesson_kind(card)
    result: list[Instruction] = []
    for instruction in _question_forms(question, QUESTION_FORMS):
        result.append(Instruction(instruction=instruction, response=answer, kind=kind))
    for topic in topics or []:
        for instruction in _topic_forms(kind, topic, TOPIC_FORMS):
            result.append(
                Instruction(instruction=instruction, response=answer, topic=topic, kind=kind)
            )
    return result


def held_out_instructions_for(
    card: dict[str, Any], topics: list[str] | None = None
) -> list[Instruction]:
    """The ways this lesson can be asked for that the model never trains on."""
    question = str(card.get("question") or "").strip()
    answer = str(card.get("answer") or "").strip()
    if not question or not answer:
        return []
    kind = lesson_kind(card)
    result: list[Instruction] = []
    for instruction in _question_forms(question, HELD_OUT_QUESTION_FORMS):
        result.append(
            Instruction(instruction=instruction, response=answer, kind=kind, held_out=True)
        )
    for topic in topics or []:
        for instruction in _topic_forms(kind, topic, HELD_OUT_TOPIC_FORMS):
            result.append(
                Instruction(
                    instruction=instruction,
                    response=answer,
                    topic=topic,
                    kind=kind,
                    held_out=True,
                )
            )
    return result


def training_instructions(cards: list[dict[str, Any]]) -> list[Instruction]:
    """Every instruction the model learns from the tenant's lessons."""
    index = topic_index(cards)
    result: list[Instruction] = []
    for card in cards:
        result.extend(instructions_for(card, index.get(card_key(card), [])))
    return result


def held_out_instructions(cards: list[dict[str, Any]], limit: int = 40) -> list[Instruction]:
    """The instructions used to measure the model, kept out of its training."""
    index = topic_index(cards)
    result: list[Instruction] = []
    for card in cards:
        result.extend(held_out_instructions_for(card, index.get(card_key(card), [])))
        if len(result) >= limit:
            break
    return result[:limit]


def held_out_requests(cards: list[dict[str, Any]], limit: int = 12) -> list[Instruction]:
    """The held-out instructions that ask for something to be written.

    The held-out questions are answered by looking them up, which is what the
    paraphrasing rung measures already. These are the ones the request path
    answers, so they are the ones that show what the model can do.
    """
    from app.services.language import requests as request_service

    index = topic_index(cards)
    result: list[Instruction] = []
    for card in cards:
        for item in held_out_instructions_for(card, index.get(card_key(card), [])):
            if request_service.parse_request(item.instruction) is None:
                continue
            result.append(item)
            if len(result) >= limit:
                return result
    return result
