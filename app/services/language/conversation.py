"""The small talk the assistant answers by itself.

A hello, a name, a thank you and a goodbye do not need a lesson: they are part
of how a conversation begins and ends. The answers use the model name from
Settings, so renaming the model renames the voice that greets you. The same
pairs are given to the training, so the model learns the conversation as well
as answering it directly.
"""

from __future__ import annotations

import re

_GREETING = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|good\s+(?:morning|afternoon|evening))\b\s*(.*)$"
)

_FILLERS = frozenset({"there", "again", "everyone", "friend", "my friend", "dear"})

_IDENTITY = frozenset(
    {
        "what is your name",
        "whats your name",
        "what your name",
        "what is your name again",
        "whats your name again",
        "who are you",
        "what are you",
        "what should i call you",
        "what can i call you",
        "what are you called",
        "do you have a name",
        "tell me your name",
        "your name",
    }
)

_STATUS = frozenset(
    {
        "how are you",
        "how are you doing",
        "how is it going",
        "how s it going",
        "how are things",
        "how do you do",
        "how you doing",
    }
)

_THANKS = frozenset(
    {
        "thanks",
        "thank you",
        "thanks a lot",
        "thanks so much",
        "thank you so much",
        "thank you very much",
        "many thanks",
        "thx",
    }
)

_GOODBYE = frozenset(
    {
        "bye",
        "goodbye",
        "good bye",
        "bye bye",
        "see you",
        "see you later",
        "see ya",
        "good night",
        "goodnight",
        "take care",
    }
)

_ABILITY = frozenset(
    {
        "what can you do",
        "what do you do",
        "what are you good at",
        "what can i ask you",
        "how can you help",
    }
)


def greeting_answer(model_name: str) -> str:
    """What the model says when someone greets it."""
    return f"Hello! I am {model_name}. How can I help you today?"


def name_answer(model_name: str) -> str:
    """What the model says when asked for its name."""
    return f"My name is {model_name}. I am a model trained on what you teach me."


def identity_answer(model_name: str) -> str:
    """What the model says when asked who it is."""
    return f"I am {model_name}, a model trained on what you teach me."


def status_answer() -> str:
    """What the model says when asked how it is."""
    return "I am doing well, thank you for asking. What would you like to learn today?"


def thanks_answer() -> str:
    """What the model says when it is thanked."""
    return "You are welcome."


def goodbye_answer() -> str:
    """What the model says when the conversation ends."""
    return "Goodbye! Come back whenever you want to learn something."


def ability_answer() -> str:
    """What the model says it can answer from."""
    return (
        "I answer from what I am taught: lessons, approved answers, corrections, "
        "documents, words and sentences. Ask me something, or teach me something new."
    )


def answer_conversation(question: str, model_name: str) -> str | None:
    """The conversational answer to a greeting or a small remark."""
    text = _normalize(question)
    if not text:
        return None
    greeting = _GREETING.match(text)
    if not greeting:
        return _short_answer(text, model_name)
    rest = _remainder(greeting.group(1), model_name)
    if not rest:
        return greeting_answer(model_name)
    return _short_answer(rest, model_name)


def training_pairs(model_name: str) -> list[tuple[str, str]]:
    """The conversational pairs the model is trained on."""
    pairs: list[tuple[str, str]] = []
    for question in (
        "hi",
        "hello",
        "hey",
        "hiya",
        "good morning",
        "good afternoon",
        "good evening",
        "hey there",
    ):
        pairs.append((question, greeting_answer(model_name)))
    for question in (
        "what is your name",
        "what's your name",
        "whats your name",
        "who are you",
        "what should I call you",
        "what are you called",
        "tell me your name",
    ):
        pairs.append((question, name_answer(model_name)))
    for question in ("how are you", "how are you doing", "how's it going", "how do you do"):
        pairs.append((question, status_answer()))
    for question in ("thanks", "thank you", "thanks a lot", "thank you very much"):
        pairs.append((question, thanks_answer()))
    for question in ("bye", "goodbye", "see you later", "good night"):
        pairs.append((question, goodbye_answer()))
    for question in ("what can you do", "what do you do"):
        pairs.append((question, ability_answer()))
    return pairs


def _remainder(after_greeting: str, model_name: str) -> str:
    """The words after the greeting, with the name and the filler taken out."""
    name = _normalize(model_name)
    words = after_greeting.strip().split()
    while words and (words[0] in _FILLERS or (name and words[0] == name)):
        words.pop(0)
    if name and " ".join(words) == name:
        return ""
    return " ".join(words)


def _short_answer(text: str, model_name: str) -> str | None:
    """The answer to one of the short conversational turns."""
    if text in _IDENTITY:
        if "name" in text or "call" in text:
            return name_answer(model_name)
        return identity_answer(model_name)
    steps = (
        (_STATUS, status_answer),
        (_THANKS, thanks_answer),
        (_GOODBYE, goodbye_answer),
        (_ABILITY, ability_answer),
    )
    for phrases, answer_of in steps:
        if text in phrases:
            return answer_of()
    return None


def _normalize(text: str) -> str:
    """The words of a turn, without case, punctuation or extra spacing."""
    lowered = str(text or "").lower().replace("’", "").replace("'", "")
    cleaned = re.sub(r"[^a-z0-9 ]", " ", lowered)
    return " ".join(cleaned.split())
