"""The model reads its own answer before it is given away.

The critique looks at the answer with the same eyes the model wrote it with:
how sure it was of every word, and whether the answer says what the material
says. When the answer is not good enough, the model is asked again with the
critique in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.transformer import grammar
from app.services.transformer import quality as quality_service

# The average chance below which the model is not sure of its answer.
MIN_CONFIDENCE = 0.35

# The chance below which the model is not sure of one word.
MIN_WORD_CONFIDENCE = 0.25

# The share of the words of the answer that have to come from the material.
MIN_GROUNDED_SHARE = 0.6


@dataclass(frozen=True)
class Critique:
    """What the model thinks of the answer it wrote."""

    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    least_word: str = ""
    least_chance: float = 0.0
    grounded: bool = True

    @property
    def should_retry(self) -> bool:
        """Whether the answer is worth writing again."""
        return bool(self.problems)

    @property
    def unsure_about(self) -> str:
        """What the model is not sure about, in words."""
        if not self.least_word:
            return ""
        return f"I am not sure about the word “{self.least_word}”."

    @property
    def summary(self) -> str:
        """The critique in a form the model can read."""
        return " ".join(self.problems)


@dataclass(frozen=True)
class Answer:
    """An answer the model wrote, and what it thinks of it."""

    text: str
    critique: Critique


def _words_of(text: str) -> set[str]:
    """The words of a text, in lower case."""
    return {word.lower() for word in text.replace(".", " ").split() if len(word) > 2}


def _grounded_share(answer: str, material: str) -> float:
    """The share of the words of the answer that come from the material."""
    words = _words_of(answer)
    if not words:
        return 1.0
    given = _words_of(material)
    return len(words & given) / len(words)


def _least(chances: list[float], words: list[str]) -> tuple[str, float]:
    """The word the model was least sure of."""
    least_word = ""
    least_chance = 1.0
    for chance, word in zip(chances, words, strict=False):
        if chance < least_chance:
            least_chance = chance
            least_word = word
    return least_word, least_chance


def critique(
    answer: str,
    chances: list[float],
    words: list[str],
    material: str = "",
) -> Critique:
    """What the model thinks of the answer it wrote."""
    problems: list[str] = []
    notes: list[str] = []
    confidence = sum(chances) / len(chances) if chances else 0.0
    least_word, least_chance = _least(chances, words)
    if confidence < MIN_CONFIDENCE:
        problems.append("The answer was written with very little certainty.")
    if least_word and least_chance < MIN_WORD_CONFIDENCE:
        problems.append(f"The model was not sure about the word “{least_word}”.")
    grounded = True
    if material:
        share = _grounded_share(answer, material)
        grounded = share >= MIN_GROUNDED_SHARE
        if not grounded:
            problems.append("The answer does not say what the material says.")
    if quality_service.repeats_itself(answer):
        problems.append("The answer says the same words again and again.")
    problems.extend(grammar.problems(answer))
    notes.extend(grammar.notes(answer))
    return Critique(
        problems=problems,
        notes=notes,
        confidence=round(confidence, 4),
        least_word=least_word,
        least_chance=round(least_chance, 4),
        grounded=grounded,
    )
