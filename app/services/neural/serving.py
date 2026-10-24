"""The word backend that serves, and what it measured.

There is one word backend: the neural word model trained from the learner's
own sentences. A tenant that has not trained one is told so rather than
answered from word counts. The gate below no longer decides whether the model
may speak; it only says how well the model measured, in plain words.
"""

from __future__ import annotations

import math
from typing import Any

# The fewest held-out examples a word model should be measured on.
MIN_WORD_VALIDATION = 100

# Below this share of the chance loss, the measured model is clearly learning.
WORD_LOSS_RATIO = 0.75


def word_gate(metrics: dict[str, Any]) -> tuple[bool, str]:
    """Whether a word model measured well, and the reason in plain words."""
    validation = int(metrics.get("validation_examples") or 0)
    if validation < MIN_WORD_VALIDATION:
        return False, f"only {validation} held-out examples; {MIN_WORD_VALIDATION} are needed"
    loss = metrics.get("validation_loss")
    if loss is None:
        return False, "no held-out loss was measured"
    chance = metrics.get("chance_loss")
    if chance is None:
        vocabulary = max(int(metrics.get("vocabulary") or 0), 2)
        chance = math.log(vocabulary)
    if float(loss) <= WORD_LOSS_RATIO * float(chance):
        return True, f"held-out loss {loss} against chance {round(float(chance), 4)}"
    return False, f"held-out loss {loss} is above {WORD_LOSS_RATIO:.0%} of chance {chance}"


class EmptyWordModel:
    """The word backend of a tenant that has not trained one yet."""

    backend = "untrained"
    sentence_count = 0
    token_count = 0
    transition_count = 0
    example_count = 0

    def __init__(self) -> None:
        self.vocabulary: list[str] = []

    def predict_next(
        self,
        prefix: str,
        limit: int = 5,
        allowed: set[str] | None = None,
    ) -> list[dict[str, int | float | str | None]]:
        return []

    def generate_sentence(
        self,
        seed: str | None = None,
        max_words: int = 16,
        use_taught: bool = True,
        allowed: set[str] | None = None,
    ) -> str | None:
        return None

    def taught_sentence(self, seed: str) -> str | None:
        return None

    def score_words(self, words: list[str]) -> float:
        return 0.0


class ServingWord:
    """The word model that answered, and the name and reason of its backend."""

    def __init__(self, model: Any, backend: str, reason: str, allowed: set[str] | None) -> None:
        self._model = model
        self.backend = backend
        self.backend_reason = reason
        self.allowed = allowed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict_next(self, prefix: str, limit: int = 5) -> list[dict[str, Any]]:
        return self._model.predict_next(prefix, limit, self.allowed)

    def generate_sentence(
        self,
        seed: str | None = None,
        max_words: int = 16,
        use_taught: bool = True,
    ) -> str | None:
        return self._model.generate_sentence(seed, max_words, use_taught, self.allowed)


def serving_word(neural: Any, allowed: set[str] | None = None) -> ServingWord:
    """The neural word model, or an empty backend while it is untrained."""
    if neural is None:
        return ServingWord(
            EmptyWordModel(),
            "untrained",
            "the word model has not been trained yet",
            allowed,
        )
    _measured, reason = word_gate(getattr(neural, "report", {}) or {})
    return ServingWord(neural, "neural", reason, allowed)
