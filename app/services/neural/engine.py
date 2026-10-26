"""The neural engine: the one place that owns the word model and its numbers.

The engine loads the tenant's checkpoint, trains the word task, measures it on
held-out data and writes the runs the builder reads. Nothing outside this
package touches torch or a checkpoint file.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from app.services.brain_paths import tenant_key
from app.services.neural import checkpoint, store
from app.services.neural.model import device_name
from app.services.neural.serving import ServingWord, serving_word, word_gate
from app.services.neural.settings import TrainingSettings
from app.services.neural.words import WordTask

# The engine of every tenant, watched by the change time of its model.
_CACHE: OrderedDict[int, tuple[float | None, NeuralEngine]] = OrderedDict()
_LOCK = threading.Lock()

# How many engines keep their models built. A local install serves one tenant,
# and the weights are mapped from the files, so a second tenant's engine is
# built from disk when it is asked for and the first one lets go.
MAX_ENGINES = 1


def engine_for(tenant_id: int | None) -> NeuralEngine:
    """The tenant's engine, reloaded when a checkpoint on disk changes."""
    key = tenant_key(tenant_id)
    stamp = store.stamp(tenant_id)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            _CACHE.move_to_end(key)
            return cached[1]
        engine = NeuralEngine(tenant_id)
        _remember(key, stamp, engine)
        return engine


def _remember(key: int, stamp: float | None, engine: NeuralEngine) -> None:
    """Keep the engine and let go of the oldest one beyond the limit."""
    _CACHE[key] = (stamp, engine)
    _CACHE.move_to_end(key)
    while len(_CACHE) > MAX_ENGINES:
        _CACHE.popitem(last=False)


def forget(tenant_id: int | None) -> None:
    """Drop the cached engine, so the next call reads the files again."""
    with _LOCK:
        _CACHE.pop(tenant_key(tenant_id), None)


def loaded_engines() -> int:
    """How many engines are cached, so a test can hold the limit."""
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
            },
        )


def _cache_engine(engine: NeuralEngine) -> None:
    """Point the cache at the file time that was just written."""
    with _LOCK:
        _remember(tenant_key(engine.tenant_id), store.stamp(engine.tenant_id), engine)
