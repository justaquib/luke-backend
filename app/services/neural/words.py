"""The neural word model: the next token, and a sentence written from it.

The word model learns from everything the learner knows. It is a small causal
transformer trained from scratch to predict the next token, and the same
weights write a sentence one token at a time. Its tokenizer is recorded with
the weights: a checkpoint reads the exact tokenizer it was trained with, and
a missing or mismatched artifact is refused rather than remapped.
"""

from __future__ import annotations

import logging
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
from app.services.neural.tokenizer import artifacts, spec
from app.services.neural.tokenizer import corpus as passage_corpus
from app.services.neural.tokenizer.base import (
    BPE_KIND,
    WORD_KIND,
    Tokenizer,
    TokenizerError,
)
from app.services.neural.training import TrainingReport, train_model
from app.services.neural.writer import PUNCTUATION, WordWriter
from app.services.vocabulary import CANONICAL_OWNER, Vocabulary

logger = logging.getLogger(__name__)

__all__ = ["PUNCTUATION", "WORD_PASSES", "WordTask", "WordWriter"]

# The passes over the learner's sentences the word model asks for by default.
# Its material is small, so one pass would stop at the minimum step count; the
# word model measured below is the eight-pass one.
WORD_PASSES = 8


def _paced(settings: TrainingSettings) -> TrainingSettings:
    """The word model's own horizon when the caller did not ask for one."""
    if settings.steps > 0 or settings.passes is not None:
        return settings
    return replace(settings, passes=WORD_PASSES)


class WordTask(WordWriter):
    """The neural next-token model, trained on everything the learner knows."""

    task = "words"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {
            **WORD_MODEL_CONFIG,
            "tokenizer_kind": spec.default_kind(),
            "vocabulary_size": 0,
            **(config or {}),
        }
        self._vocabulary = Vocabulary(CANONICAL_OWNER, ())
        self._index_map: dict[str, int] = {}
        self.tokenizer: Tokenizer | None = None
        self.tokenizer_record: dict[str, Any] = {}
        self.model: TinyTransformer | None = None
        self.trained = False
        self.report: dict[str, Any] = {}
        self.word_counts: dict[str, int] = {}
        self.sentence_count = 0
        self.token_count = 0
        self.transition_count = 0

    @property
    def vocabulary(self) -> list[str]:
        if self.tokenizer is not None:
            return self.tokenizer.vocabulary()
        return list(self._vocabulary.tokens)

    @property
    def index(self) -> dict[str, int]:
        if self.tokenizer is not None:
            return self._index_map
        return self._vocabulary.index

    def _set_tokenizer(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._index_map = {piece: position for position, piece in enumerate(tokenizer.vocabulary())}

    def fit(
        self,
        sentences: Sequence[str],
        settings: TrainingSettings | None = None,
        progress: Callable[[int, int, float | None], None] | None = None,
    ) -> dict[str, Any]:
        """Learn to predict the next token from the sentences the learner knows.

        A BPE tokenizer is built from the training side of the group split
        only, so a held-out sentence can never shape its vocabulary.
        """
        settings = _paced(settings or TrainingSettings())
        kept = quality.keep(sentences)
        tokenizer, train_sentences, held_sentences = self._tokenizer(kept, settings, progress)
        self._set_tokenizer(tokenizer)
        built = [examples.word_example(self, sentence) for sentence in kept]
        if len(built) < 4 or tokenizer.vocab_size < 8:
            return {"trained": False, "reason": "not enough words to train on"}
        self.config = {
            **self.config,
            "pad_id": tokenizer.pad_id,
            "vocabulary_size": tokenizer.vocab_size,
        }
        model, report, seconds = train_model(
            tokenizer.vocab_size, built, self.config, settings=settings, progress=progress
        )
        self.model = model.to("cpu")
        self.trained = True
        self.word_counts = _piece_counts(tokenizer, kept)
        self.sentence_count = len(kept)
        self.token_count = sum(len(tokenizer.encode(sentence)) for sentence in kept)
        self.transition_count = sum(max(len(example.sequence) - 2, 0) for example in built)
        self.tokenizer_record = artifacts.record(
            tokenizer,
            {
                "sentences": len(kept),
                "tokenizer_sentences": len(train_sentences),
                "held_out_sentences": len(held_sentences),
                "characters": sum(len(sentence) for sentence in train_sentences),
                "tokens": sum(len(tokenizer.encode(sentence)) for sentence in train_sentences),
            },
        )
        self.report = _report(report, seconds, tokenizer, self)
        return {"trained": True, **self.report}

    def _tokenizer(
        self,
        kept: Sequence[str],
        settings: TrainingSettings,
        progress: Callable[[int, int, float | None], None] | None,
    ) -> tuple[Tokenizer, list[str], list[str]]:
        """The tokenizer a new training uses; BPE reads the training side only."""
        kind = str(self.config.get("tokenizer_kind") or spec.default_kind())
        train_sentences, held_sentences = list(kept), []
        if kind == BPE_KIND:
            train_sentences, held_sentences = passage_corpus.split_training(kept, settings)
        tokenizer = spec.build(kind, train_sentences, self.config.get("vocabulary_size"), progress)
        return tokenizer, train_sentences, held_sentences

    def taught_sentence(self, seed: str) -> str | None:
        """The neural model has no lookup of taught sentences."""
        return None

    def score_words(self, words: list[str]) -> float:
        """How likely the model finds the word order, as a sum of log chances."""
        if not self.trained or self.model is None or len(words) < 2:
            return 0.0
        if self.tokenizer is not None and self.tokenizer.kind == WORD_KIND:
            ids = [self.index[token] for token in words if token in self.index]
        elif self.tokenizer is not None:
            ids = self.tokenizer.encode(" ".join(words))
        else:
            ids = [self.index[token] for token in words if token in self.index]
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
        pad = self.tokenizer.pad_id if self.tokenizer is not None else 0
        return checkpoint.Checkpoint(
            task=self.task,
            model=self.model,
            vocabulary=self.vocabulary,
            config={**self.config, "pad_id": pad},
            metrics=self.report,
            corpus=corpus,
            tokenizer=self.tokenizer_record,
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
        try:
            tokenizer = (
                artifacts.load(stored.tokenizer, stored.vocabulary)
                if stored.tokenizer
                else _legacy(stored.vocabulary)
            )
            if stored.tokenizer:
                artifacts.verify(stored.tokenizer, len(stored.vocabulary))
        except TokenizerError as error:
            logger.warning("the word model's tokenizer was refused: %s", error)
            return None
        model = stored.model()
        if model is None:
            return None
        task = cls(stored.config)
        task._set_tokenizer(tokenizer)
        task.tokenizer_record = stored.tokenizer
        if tokenizer.kind == WORD_KIND:
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
            "tokenizer": stored.tokenizer,
            "sentence_count": task.sentence_count,
            "token_count": task.token_count,
            "transition_count": task.transition_count,
        }
        return task


def _report(
    report: TrainingReport,
    seconds: float,
    tokenizer: Tokenizer,
    task: WordTask,
) -> dict[str, Any]:
    """The measured record of a training, as the checkpoint and the page read it."""
    model = task.model
    parameters = sum(parameter.numel() for parameter in model.parameters()) if model else 0
    return {
        **report.as_dict(),
        "schema": checkpoint.SCHEMA,
        "vocabulary": tokenizer.vocab_size,
        "parameters": parameters,
        "device": device_name(),
        "sentence_count": task.sentence_count,
        "token_count": task.token_count,
        "transition_count": task.transition_count,
        "perplexity": round(math.exp(report.validation_loss), 4)
        if report.validation_loss is not None
        else None,
        "next_word_accuracy": report.validation_accuracy
        if report.validation_accuracy is not None
        else report.train_accuracy,
        "seconds": round(seconds, 1),
        "tokenizer": task.tokenizer_record,
    }


def _legacy(vocabulary: list[str]) -> Tokenizer:
    from app.services.neural.tokenizer.word import WordTokenizer

    return WordTokenizer(vocabulary)


def _piece_counts(tokenizer: Tokenizer, sentences: Sequence[str]) -> dict[str, int]:
    """How often each piece was said, so a prediction shows a real count."""
    counts: Counter[str] = Counter()
    for sentence in sentences:
        for token_id in tokenizer.encode(sentence):
            if tokenizer.is_special(token_id):
                continue
            counts[tokenizer.token_text(token_id)] += 1
    return dict(counts)
