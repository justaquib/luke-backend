"""The neural engine: the one place that owns the word model and its numbers.

The engine loads the tenant's checkpoint, trains the word task, measures it on
held-out data and writes the runs the builder reads. Nothing outside this
package touches torch or a checkpoint file.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from app.services.brain_paths import tenant_key
from app.services.model_cache import ByteBoundedCache
from app.services.neural import checkpoint, store
from app.services.neural.model import device_name
from app.services.neural.serving import ServingWord, serving_word, word_gate
from app.services.neural.settings import TrainingSettings
from app.services.neural.tokenizer import artifacts
from app.services.neural.words import WordTask

# The engine of every tenant, kept while the files behind it fit the memory
# cap. A local install serves one tenant, but the trainer visits several, and
# an engine is cheap to keep and kept from disk when its file changes.
_CACHE = ByteBoundedCache()
_LOCK = threading.Lock()


def engine_for(tenant_id: int | None) -> NeuralEngine:
    """The tenant's engine, reloaded when a checkpoint on disk changes."""
    key = tenant_key(tenant_id)
    stamp = store.stamp(tenant_id)
    with _LOCK:
        cached = _CACHE.find(key, stamp)
        if cached is not None:
            return cached
        engine = NeuralEngine(tenant_id)
        _remember(engine, stamp)
        return engine


def _remember(engine: NeuralEngine, stamp: float | None) -> None:
    """Keep the engine while its file fits the cap, letting the oldest go."""
    key = tenant_key(engine.tenant_id)
    _CACHE.keep(key, stamp, engine, store.words_size(engine.tenant_id))


def forget(tenant_id: int | None) -> None:
    """Drop the cached engine, so the next call reads the files again."""
    with _LOCK:
        _CACHE.drop(tenant_key(tenant_id))


def loaded_engines() -> int:
    """How many engines are cached, so a test can hold the memory cap."""
    with _LOCK:
        return len(_CACHE)


@dataclass
class TrainingJob:
    """How a training is run, and where its numbers are kept."""

    corpus: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] | None = None
    settings: TrainingSettings | None = None
    progress: Callable[[int, int, float | None], None] | None = None


class NeuralEngine:
    """The word model of one tenant."""

    def __init__(self, tenant_id: int | None) -> None:
        self.tenant_id = tenant_id
        self._words: WordTask | None = None
        self._words_loaded = False
        self._load_lock = threading.Lock()

    @property
    def word_task(self) -> WordTask | None:
        """The word model, built from its file the first time it is used."""
        with self._load_lock:
            if not self._words_loaded:
                self._words = WordTask.load(store.words_path(self.tenant_id))
                self._words_loaded = True
            return self._words

    @property
    def device(self) -> str:
        return device_name()

    def word_serving(self, allowed: set[str] | None = None) -> ServingWord:
        """The word backend that serves, with the reason it was chosen."""
        return serving_word(self.word_task, allowed)

    def state(self) -> dict[str, Any]:
        """What the engine knows about its model, for the builder.

        The numbers are read from the checkpoint file itself, so a page load
        never builds a model just to show what it measured.
        """
        words = self._read(store.words_path(self.tenant_id)) or {}
        fallback_reason = "the word model has not been trained yet"
        measured, reason = word_gate(words) if words else (False, fallback_reason)
        return {
            "schema": int(words.get("schema") or 0),
            "device": device_name(),
            "torch": torch.__version__,
            "words": {
                "trained": bool(words),
                **words,
                "backend": "neural" if words else "untrained",
                "backend_reason": reason,
                "quality_ok": measured,
            },
            "memory": {
                "words_bytes": int(words.get("weights_bytes") or 0),
                "words_loaded": self._words_loaded,
                "loaded": self._words_loaded,
            },
            "runs": store.load_runs(self.tenant_id),
        }

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        """The numbers a checkpoint file holds, without building its model."""
        stored = checkpoint.load(path)
        if stored is None:
            return None
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return {
            **stored.metrics,
            "schema": stored.schema,
            "vocabulary": len(stored.vocabulary),
            "trained_at": stored.created,
            "corpus": stored.corpus,
            "tokenizer": artifacts.describe(stored.tokenizer),
            "weights_bytes": size,
        }

    def train_words(self, sentences: list[str], job: TrainingJob) -> dict[str, Any]:
        """Train the word model from scratch and keep it if it trained."""
        task = WordTask(job.config)
        report = task.fit(sentences, settings=job.settings, progress=job.progress)
        if report.get("trained"):
            task.save(store.words_path(self.tenant_id), job.corpus)
            self._words = task
            self._words_loaded = True
            self._record("words", report)
            _cache_engine(self)
        return report

    def _record(self, task: str, report: dict[str, Any]) -> None:
        """Note the run in the short history the builder reads."""
        tokenizer = dict(report.get("tokenizer") or {})
        store.append_run(
            self.tenant_id,
            {
                "task": task,
                "loss": report.get("loss"),
                "validation_loss": report.get("validation_loss"),
                "passes": report.get("passes"),
                "stop_reason": report.get("stop_reason"),
                "steps": report.get("steps"),
                "seconds": report.get("seconds"),
                "tokenizer_kind": tokenizer.get("kind"),
                "tokenizer_id": tokenizer.get("id"),
            },
        )


def _cache_engine(engine: NeuralEngine) -> None:
    """Point the cache at the file time that was just written."""
    with _LOCK:
        _remember(engine, store.stamp(engine.tenant_id))
