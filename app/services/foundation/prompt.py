"""The words Luke is given: identity, persona, conversation style and evidence.

The prompt is built here, not in a component, so every surface asks the model
the same way. Evidence is numbered and must be cited; a generated answer is told
that the lab's material did not cover the question, so it never claims to have
been taught it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.foundation.identity import FoundationIdentity

MAX_HISTORY_TURNS = 12
MAX_EVIDENCE_CHARS = 2400
MAX_EVIDENCE_ITEMS = 5

DEFAULT_TONE = "warm, plain and direct"
DEFAULT_VERBOSITY = "concise"


@dataclass(frozen=True)
class Persona:
    """How the model speaks. The name already lives in model settings."""

    tone: str = DEFAULT_TONE
    verbosity: str = DEFAULT_VERBOSITY


@dataclass(frozen=True)
class Evidence:
    """One numbered source the model may ground an answer on."""

    index: int
    source: str
    label: str
    text: str

    def as_line(self) -> str:
        """The numbered line the prompt carries."""
        text = " ".join(self.text.split())[:MAX_EVIDENCE_CHARS]
        return f"[{self.index}] ({self.source}: {self.label}) {text}"


def system_prompt(
    identity: FoundationIdentity,
    model_name: str,
    persona: Persona,
    expressions: Sequence[str] = (),
    evidence_count: int = 0,
) -> str:
    """The system message: who Luke is, how it speaks, and what it may claim."""
    base = identity.base_model.rsplit("/", 1)[-1]
    lines = [
        f"You are {model_name}, the language model of Luke Lab.",
        f"Your base checkpoint is {base} (revision {identity.revision[:7]}, "
        f"license {identity.license}). Your version is {identity.version}.",
        f"Speak in a {persona.tone} voice and keep answers {persona.verbosity}.",
        "You are still learning: say what you know, but never claim that the lab "
        "taught you something it did not.",
    ]
    if expressions:
        lines.append("Conversation style taught by the trainer:")
        lines.extend(f"- {line}" for line in expressions)
    if evidence_count:
        lines.extend(
            [
                f"You are given {evidence_count} numbered pieces of evidence.",
                "Answer only from that evidence and cite every claim with its number, like [1].",
                "If the evidence does not contain the answer, say you have not been "
                "taught it; do not add facts, numbers, names or citations of your own.",
            ]
        )
    else:
        lines.extend(
            [
                "The lab's material does not cover this question, so answer from what you know.",
                "Do not invent a citation, a measurement, a person or a source. If you "
                "are unsure, say so.",
            ]
        )
    lines.append("Refuse unsafe or harmful requests briefly, without moralising.")
    return "\n".join(lines)


def build_messages(
    system: str,
    history: Sequence[dict[str, str]],
    question: str,
    evidence: Sequence[Evidence] = (),
) -> list[dict[str, str]]:
    """The chat messages, with the evidence block before the question."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history[-MAX_HISTORY_TURNS * 2 :]:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant", "tool"} and content:
            messages.append({"role": role, "content": content})
    asked = question.strip()
    if evidence:
        block = "\n".join(item.as_line() for item in evidence[:MAX_EVIDENCE_ITEMS])
        asked = f"Evidence:\n{block}\n\nQuestion: {asked}"
    messages.append({"role": "user", "content": asked})
    return messages


def trim_history(
    messages: Sequence[dict[str, str]],
    count_tokens: Any,
    budget: int,
) -> list[dict[str, str]]:
    """Drop the oldest turns until the prompt fits the token budget."""
    trimmed = list(messages)
    system = trimmed[:1]
    turns = trimmed[1:]
    keep = len(turns)
    while keep > 1 and count_tokens(system + turns[-keep:]) > budget:
        keep -= 1
    return system + turns[-keep:] if keep else system


def fallback_template(messages: Sequence[dict[str, str]]) -> str:
    """A documented chat shape for a checkpoint without its own template."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").capitalize()
        lines.append(f"{role}: {message.get('content') or ''}")
    lines.append("Assistant:")
    return "\n".join(lines)


def clean_answer(text: str) -> str:
    """The written answer, without a role label the template may have added."""
    cleaned = str(text or "").strip()
    for marker in ("Assistant:", "assistant:", "User:", "user:"):
        while cleaned.startswith(marker):
            cleaned = cleaned[len(marker) :].strip()
    return cleaned


def trim_incomplete(text: str) -> str:
    """The answer up to its last finished sentence, when the cap cut it short."""
    cleaned = clean_answer(text)
    if not cleaned or cleaned[-1] in ".!?\"'”’":
        return cleaned
    for index in range(len(cleaned) - 1, -1, -1):
        if cleaned[index] in ".!?" and cleaned[index + 1] == " ":
            return cleaned[: index + 1].strip()
    return cleaned
