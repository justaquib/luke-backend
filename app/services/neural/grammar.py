"""The grammar of a written sentence, read with the language model of the app.

The writer reads the sentence it wrote before it is given away. A text that
is not a sentence at all is a problem, and a sentence whose verb does not
agree with its subject is a note.
"""

from __future__ import annotations

from typing import Any

from app.services.nlp import get_nlp

# The word tags of a subject that is one thing.
SINGULAR = frozenset({"NN", "NNP"})

# The word tags of a subject that is more than one thing.
PLURAL = frozenset({"NNS", "NNPS"})


def _has_verb(document: Any) -> bool:
    """Whether the sentence has a verb of its own.

    The tagger calls "is" and "are" auxiliary verbs, and they are the verb of
    the sentence just the same.
    """
    return any(token.pos_ in {"VERB", "AUX"} for token in document)


def _disagrees(subject: Any, verb: Any) -> bool:
    """Whether the subject and the verb are both there and do not agree."""
    if subject.tag_ in SINGULAR and verb.tag_ == "VBP":
        return True
    return bool(subject.tag_ in PLURAL and verb.tag_ == "VBZ")


def _agreement(document: Any) -> list[str]:
    """Where the verb and its subject do not agree."""
    notes: list[str] = []
    for token in document:
        if token.pos_ != "VERB":
            continue
        for child in token.children:
            if child.dep_ != "nsubj":
                continue
            if _disagrees(child, token):
                notes.append(f"The verb “{token.text}” does not agree with “{child.text}”.")
    return notes


def problems(text: str) -> list[str]:
    """What is wrong with the text, so badly that it cannot be given away."""
    document = get_nlp()(text or "")
    if not document.text.strip():
        return ["The sentence is empty."]
    if not _has_verb(document):
        return ["The sentence has no verb."]
    return []


def notes(text: str) -> list[str]:
    """What could be said better about the sentence."""
    document = get_nlp()(text or "")
    if not document.text.strip():
        return []
    return _agreement(document)
