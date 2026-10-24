"""The two tasks the neural engine serves, and the corpus each one learns.

The answer task reads a request and writes a response. The word task reads a
few words and predicts the next one. They share one tokenizer, one vocabulary
formula and one training loop, and each has its own checkpoint, so a good
answer model never forces a weak word model into service.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from app.services.neural import answering, checkpoint, examples, freewriting, quality
from app.services.neural.model import MODEL_CONFIG, TinyTransformer, device_name
from app.services.neural.settings import ReadingSettings, TrainingSettings, WritingSettings
from app.services.neural.tokens import PAD, SPECIAL_TOKENS, decode_text
from app.services.neural.training import train_model
from app.services.neural.words import WordTask
from app.services.vocabulary import CANONICAL_OWNER, Vocabulary, guard_tensor_ids


@dataclass(frozen=True)
class AnswerCorpus:
    """The material an answer model learns from."""

    sentences: list[str] = field(default_factory=list)
    instructions: list[tuple[str, str]] = field(default_factory=list)
    contexts: list[tuple[str, str, str]] = field(default_factory=list)


class AnswerTask:
    """The trained writer: it learns the tenant's words and answers with them."""

    task = "answer"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**MODEL_CONFIG, **(config or {})}
        self._vocabulary = Vocabulary(CANONICAL_OWNER, ())
        self.model: TinyTransformer | None = None
        self.trained = False
        self.report: dict[str, Any] = {}
        self.last_confidence: float = 0.0
        self.last_least_chance: float = 0.0
        self.last_token_confidences: list[float] = []
        self.last_written_words: list[str] = []
        self.last_uncertain_words: list[str] = []

    @property
    def vocabulary(self) -> list[str]:
        """The tokens the model knows, in id order."""
        return list(self._vocabulary.tokens)

    @property
    def index(self) -> dict[str, int]:
        """The token to id mapping of the canonical vocabulary."""
        return self._vocabulary.index

    def _build_vocabulary(self, tokens: Sequence[str]) -> None:
        keep = set(tokens) | set(SPECIAL_TOKENS)
        self._vocabulary = Vocabulary(CANONICAL_OWNER, sorted(keep))

    def encode(self, text: str) -> list[int]:
        return guard_tensor_ids(self._vocabulary.encode(text), "the answer model", self._vocabulary)

    def decode(self, tokens: Sequence[int]) -> str:
        words = [self.vocabulary[token] for token in tokens if 0 <= token < len(self.vocabulary)]
        return decode_text(words)

    def fit(
        self,
        corpus: AnswerCorpus,
        settings: TrainingSettings | None = None,
        progress: Callable[[int, int, float | None], None] | None = None,
    ) -> dict[str, Any]:
        """Learn the words of the sentences and the answers to the requests."""
        settings = settings or TrainingSettings()
        words = examples.all_words(corpus.sentences, corpus.instructions, corpus.contexts)
        self._build_vocabulary(words)
        kept = quality.keep(corpus.sentences)
        built = examples.build(self, kept, corpus.instructions, corpus.contexts)
        if len(built) < 4 or len(self.vocabulary) < 8:
            return {"trained": False, "reason": "not enough words to train on"}
        config = {**self.config, "pad_id": self.index[PAD]}
        model, report, seconds = train_model(
            len(self.vocabulary), built, config, settings=settings, progress=progress
        )
        self.model = model.to("cpu")
        self.trained = True
        self.report = {
            **report.as_dict(),
            "schema": checkpoint.SCHEMA,
            "vocabulary": len(self.vocabulary),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "device": device_name(),
            "dropped_sentences": len(corpus.sentences) - len(kept),
            "seconds": round(seconds, 1),
        }
        return {"trained": True, **self.report}

    def respond(
        self,
        instruction: str,
        context: str = "",
        max_words: int = 64,
        reading: ReadingSettings | None = None,
    ) -> str | None:
        """Answer a request in the words the model was taught."""
        request = answering.Request(instruction, context, max_words, reading or ReadingSettings())
        return answering.respond(self, request)

    def generate(self, prompt: str = "", settings: WritingSettings | None = None) -> str | None:
        """Write what comes after the prompt, in the words it has learned."""
        return freewriting.generate(self, prompt, settings)

    def score(self, text: str) -> float:
        """How well the model knows a sentence, as a log chance per word."""
        if not self.trained or self.model is None:
            return 0.0
        tokens = self.encode(text)
        if len(tokens) < 2:
            return 0.0
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(torch.tensor([tokens], dtype=torch.long, device=device))[0]
        log_chances = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
        chosen = log_chances[torch.arange(len(tokens) - 1), torch.tensor(tokens[1:], device=device)]
        return float(chosen.mean().item())

    def checkpoint(self, corpus: dict[str, Any]) -> checkpoint.Checkpoint:
        assert self.model is not None
        return checkpoint.Checkpoint(
            task=self.task,
            model=self.model,
            vocabulary=self.vocabulary,
            config={**self.config, "pad_id": self.index.get(PAD, 0)},
            metrics=self.report,
            corpus=corpus,
        )

    def save(self, path: Path, corpus: dict[str, Any] | None = None) -> None:
        if self.trained and self.model is not None:
            checkpoint.save(path, self.checkpoint(corpus or {}))

    @classmethod
    def load(cls, path: Path) -> AnswerTask | None:
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
        task.report = {
            **stored.metrics,
            "vocabulary": len(stored.vocabulary),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": device_name(),
            "schema": stored.schema,
            "trained_at": stored.created,
            "corpus": stored.corpus,
        }
        return task


__all__ = ["AnswerCorpus", "AnswerTask", "WordTask"]
