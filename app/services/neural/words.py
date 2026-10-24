"""The neural word model: the next word, and a sentence written from it.

The word model learns from everything the learner knows. It is a small
causal transformer trained from scratch to predict the next token, and the
same weights write a sentence one word at a time. The public methods keep the
shape the learner surfaces already read, so switching from counts to the
network did not change them.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from app.services.neural import checkpoint, examples, quality
from app.services.neural.model import WORD_MODEL_CONFIG, TinyTransformer, device_name
from app.services.neural.settings import TrainingSettings
from app.services.neural.tokens import (
    FINISH,
    PAD,
    SPECIAL_TOKENS,
    START,
    decode_text,
    is_a_word,
    tokenize,
)
from app.services.neural.training import train_model
from app.services.vocabulary import CANONICAL_OWNER, Vocabulary

# The punctuation a word model is never asked to predict as a word.
PUNCTUATION = frozenset({".", "!", "?", ","})

# The passes over the learner's sentences the word model asks for by default.
# Its material is small, so one pass would stop at the minimum step count; the
# word model measured below is the eight-pass one.
WORD_PASSES = 8


def _paced(settings: TrainingSettings) -> TrainingSettings:
    """The word model's own horizon when the caller did not ask for one."""
    if settings.steps > 0 or settings.passes is not None:
        return settings
    return replace(settings, passes=WORD_PASSES)


class WordWriter:
    """Write a sentence from a prefix, one word at a time."""

    def generate_sentence(
        self,
        seed: str | None = None,
        max_words: int = 16,
        use_taught: bool = True,
        allowed: set[str] | None = None,
    ) -> str | None:
        """Write a sentence one word at a time, without repeating a naming word."""
        if not self.trained or self.model is None:
            return None
        tokens = self._prefix_ids(seed or "")
        if seed and len(tokens) == 1:
            return None
        allowed_ids = self._allowed_ids(allowed)
        written: list[int] = []
        used = {self.vocabulary[token] for token in tokens[1:]}
        visited: set[tuple[int, ...]] = set()
        for _ in range(max_words):
            token_id = self._next(tokens, used, visited, allowed_ids)
            if token_id is None:
                break
            tokens.append(token_id)
            written.append(token_id)
            used.add(self.vocabulary[token_id])
        if not written:
            return None
        text = decode_text([self.vocabulary[token] for token in written])
        return text[:1].upper() + text[1:]

    def _next(
        self,
        tokens: list[int],
        used: set[str],
        visited: set[tuple[int, ...]],
        allowed_ids: set[int] | None,
    ) -> int | None:
        """The next word of a sentence, or nothing when it has to stop."""
        context = tuple(tokens[-int(self.config["context"]) :])
        if context in visited:
            return None
        visited.add(context)
        token_id = self._choose(context, tokens, used, allowed_ids)
        if token_id == self.index.get(FINISH):
            return None
        word = self.vocabulary[token_id]
        if not is_a_word(word) or word in PUNCTUATION:
            return None
        return token_id

    def _choose(
        self,
        context: tuple[int, ...],
        tokens: list[int],
        used: set[str],
        allowed_ids: set[int] | None,
    ) -> int:
        """The word the model picks, never a mark, a punctuation or a loop."""
        device = next(self.model.parameters()).device
        inputs = torch.tensor([list(context)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.model(inputs)[0, -1]
        for token_id in self._banned(tokens, allowed_ids, allow_finish=True):
            logits[token_id] = -float("inf")
        for token_id, word in enumerate(self.vocabulary):
            if word in used and word not in _REPEATABLE_WORDS:
                logits[token_id] = -float("inf")
        return int(torch.argmax(logits).item())


class WordTask(WordWriter):
    """The neural next-word model, trained on everything the learner knows."""

    task = "words"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**WORD_MODEL_CONFIG, **(config or {})}
        self._vocabulary = Vocabulary(CANONICAL_OWNER, ())
        self.model: TinyTransformer | None = None
        self.trained = False
        self.report: dict[str, Any] = {}
        self.word_counts: dict[str, int] = {}
        self.sentence_count = 0
        self.token_count = 0
        self.transition_count = 0

    @property
    def vocabulary(self) -> list[str]:
        return list(self._vocabulary.tokens)

    @property
    def index(self) -> dict[str, int]:
        return self._vocabulary.index

    def _build_vocabulary(self, tokens: Sequence[str]) -> None:
        self._vocabulary = Vocabulary(CANONICAL_OWNER, sorted(set(tokens) | set(SPECIAL_TOKENS)))

    def fit(
        self,
        sentences: Sequence[str],
        settings: TrainingSettings | None = None,
        progress: Callable[[int, int, float | None], None] | None = None,
    ) -> dict[str, Any]:
        """Learn to predict the next word from the sentences the learner knows."""
        settings = _paced(settings or TrainingSettings())
        kept = quality.keep(sentences)
        tokens = [token for sentence in kept for token in tokenize(sentence)]
        self._build_vocabulary(tokens)
        built = [examples.word_example(self, sentence) for sentence in kept]
        if len(built) < 4 or len(self.vocabulary) < 8:
            return {"trained": False, "reason": "not enough words to train on"}
        config = {**self.config, "pad_id": self.index[PAD]}
        model, report, seconds = train_model(
            len(self.vocabulary), built, config, settings=settings, progress=progress
        )
        self.model = model.to("cpu")
        self.trained = True
        self.word_counts = {
            word: count
            for word, count in Counter(tokens).items()
            if is_a_word(word) and word not in PUNCTUATION
        }
        self.sentence_count = len(kept)
        self.token_count = len(tokens)
        self.transition_count = sum(max(len(example.sequence) - 2, 0) for example in built)
        self.report = {
            **report.as_dict(),
            "schema": checkpoint.SCHEMA,
            "vocabulary": len(self.vocabulary),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "device": device_name(),
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "transition_count": self.transition_count,
            "perplexity": round(math.exp(report.validation_loss), 4)
            if report.validation_loss is not None
            else None,
            "next_word_accuracy": report.validation_accuracy
            if report.validation_accuracy is not None
            else report.train_accuracy,
            "seconds": round(seconds, 1),
        }
        return {"trained": True, **self.report}

    def _allowed_ids(self, allowed: set[str] | None) -> set[int] | None:
        if allowed is None:
            return None
        return {self.index[word] for word in allowed if word in self._vocabulary}

    def _banned(
        self,
        tokens: list[int],
        allowed: set[int] | None,
        allow_finish: bool = False,
    ) -> set[int]:
        """The ids a written word may not be, so a mark or a loop is never written."""
        banned = {self.index[token] for token in SPECIAL_TOKENS if token in self.index}
        banned |= {self.index[token] for token in PUNCTUATION if token in self.index}
        if allow_finish and self.index.get(FINISH) is not None:
            banned.discard(self.index[FINISH])
        if allowed is not None:
            banned |= set(range(len(self.vocabulary))) - allowed
        if tokens:
            banned.add(tokens[-1])
        return banned

    def predict_next(
        self,
        prefix: str,
        limit: int = 5,
        allowed: set[str] | None = None,
    ) -> list[dict[str, int | float | str | None]]:
        """The words the model expects next, with the chance it gives each."""
        if not self.trained or self.model is None:
            return []
        tokens = self._prefix_ids(prefix)
        allowed_ids = self._allowed_ids(allowed)
        if allowed is not None and not allowed_ids:
            return []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            window = tokens[-int(self.config["context"]) :]
            logits = self.model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]
        for token_id in self._banned(tokens, allowed_ids):
            logits[token_id] = -float("inf")
        probabilities = torch.softmax(logits, dim=-1)
        values, positions = torch.topk(probabilities, min(limit * 3, len(self.vocabulary)))
        rows = []
        for value, position in zip(values.tolist(), positions.tolist(), strict=False):
            word = self.vocabulary[position]
            if not is_a_word(word) or word in PUNCTUATION:
                continue
            rows.append(
                {
                    "word": word,
                    "count": int(self.word_counts.get(word, 0)),
                    "probability": round(float(value), 5),
                    "source": "neural",
                }
            )
            if len(rows) == limit:
                break
        return rows

    def _prefix_ids(self, prefix: str) -> list[int]:
        words = [token for token in tokenize(prefix) if is_a_word(token)]
        ids = [self.index[token] for token in words if token in self._vocabulary]
        return [self.index[START], *ids] if ids else [self.index[START]]

    def taught_sentence(self, seed: str) -> str | None:
        """The neural model has no lookup of taught sentences."""
        return None

    def score_words(self, words: list[str]) -> float:
        """How likely the model finds the word order, as a sum of log chances."""
        if not self.trained or self.model is None or len(words) < 2:
            return 0.0
        ids = [self.index[token] for token in words if token in self._vocabulary]
        if len(ids) < 2:
            return 0.0
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(torch.tensor([ids], dtype=torch.long, device=device))[0]
        log_chances = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
        chosen = log_chances[torch.arange(len(ids) - 1), torch.tensor(ids[1:], device=device)]
        return float(chosen.sum().item())

    def checkpoint(self, corpus: dict[str, Any]) -> checkpoint.Checkpoint:
        assert self.model is not None
        return checkpoint.Checkpoint(
            task=self.task,
            model=self.model,
            vocabulary=self.vocabulary,
            config={**self.config, "pad_id": self.index.get(PAD, 0)},
            metrics=self.report,
            corpus=corpus,
            extras={"word_counts": self.word_counts},
        )

    def save(self, path: Path, corpus: dict[str, Any] | None = None) -> None:
        if self.trained and self.model is not None:
            checkpoint.save(path, self.checkpoint(corpus or {}))

    @classmethod
    def load(cls, path: Path) -> WordTask | None:
        stored = checkpoint.load(path)
        if stored is None or stored.task != cls.task:
            return None
        model = stored.model()
        if model is None:
            return None
        task = cls(stored.config)
        task._vocabulary = Vocabulary(CANONICAL_OWNER, stored.vocabulary)
        task.model = model
        task.trained = True
        task.word_counts = {
            str(word): int(count)
            for word, count in dict(stored.extras.get("word_counts", {})).items()
        }
        sentences = stored.metrics.get("sentence_count") or stored.corpus.get("sentences") or 0
        task.sentence_count = int(sentences)
        task.token_count = int(stored.metrics.get("token_count") or 0)
        task.transition_count = int(stored.metrics.get("transition_count") or 0)
        task.report = {
            **stored.metrics,
            "vocabulary": len(stored.vocabulary),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": device_name(),
            "schema": stored.schema,
            "trained_at": stored.created,
            "corpus": stored.corpus,
            "sentence_count": task.sentence_count,
            "token_count": task.token_count,
            "transition_count": task.transition_count,
        }
        return task


# Small words may be said again; a naming word said again is a loop.
_REPEATABLE_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "it",
        "its",
        "that",
        "this",
        "these",
        "those",
        "for",
        "with",
        "as",
        "at",
        "from",
        "by",
        "has",
        "have",
        "had",
        "there",
    }
)
