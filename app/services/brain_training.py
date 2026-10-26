"""Training the word model and the heads, and the markers around a run.

The training runs in its own process, so the marker on disk is the truth the
page reads: a flag in memory would not be seen by the process that watches
it. Each stage of the checklist is written as it finishes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.services import brain_paths, training_progress
from app.services.brain_feedback import load_feedback
from app.services.brain_heads import (
    train_emotion_head,
    train_feedback_heads,
    train_intent_head,
)

if TYPE_CHECKING:
    from app.services.language.training_materials import TrainingSet

LOGGER = logging.getLogger(__name__)

_LOCK = threading.RLock()

TRAINING_TIMEOUT_SECONDS = 30 * 60


def training_path(tenant_id: int | None) -> Path:
    """The marker that says this tenant's brain is training."""
    return brain_paths.brain_dir() / f"training-{brain_paths.tenant_key(tenant_id)}.json"


def mark_training(tenant_id: int | None, value: bool) -> None:
    """Note that the brain is training, so the page can say so.

    The note is written next to the model, so a training that runs in another
    process is seen as well.
    """
    path = training_path(tenant_id)
    if value:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"started": time.time()}), encoding="utf-8")
        return
    path.unlink(missing_ok=True)


def clear_training(tenant_id: int | None = None) -> None:
    if tenant_id is None:
        for path in brain_paths.brain_dir().glob("training-*.json"):
            path.unlink(missing_ok=True)
        return
    training_path(tenant_id).unlink(missing_ok=True)


def is_training(tenant_id: int | None) -> bool:
    """Whether this tenant's brain is being trained right now.

    The marker on disk is the truth: the training runs in its own process, so
    the flag cannot be kept in memory alone.
    """
    path = training_path(tenant_id)
    if not path.exists():
        return False
    try:
        started = float(json.loads(path.read_text(encoding="utf-8")).get("started", 0))
    except (OSError, ValueError, TypeError):
        started = 0
    if started and time.time() - started > TRAINING_TIMEOUT_SECONDS:
        # A training that was interrupted leaves its marker behind.
        path.unlink(missing_ok=True)
        return False
    return True


def word_state(tenant_id: int | None) -> dict[str, Any]:
    """What the word model knows about itself, read without building it.

    The report is kept in the checkpoint, so a page that only shows the
    numbers never pays for the weights.
    """
    from app.services import neural

    report = neural.engine_for(tenant_id).state()["words"]
    if not report.get("trained"):
        return {"trained": False}
    return report


def _follow_steps(
    tenant_id: int | None,
    task: str = "words",
) -> Callable[[int, int, float | None], None]:
    """The steps of a model, noted in the checklist as they happen.

    The note is written at most once a second: a training step is much faster
    than a page can read, and the file does not need every one of them.
    """
    started = time.time()
    last = [0.0]

    def follow(done: int, total: int, loss: float | None = None) -> None:
        now = time.time()
        if done < total and now - last[0] < 1.0:
            return
        last[0] = now
        elapsed = max(now - started, 0.001)
        rate = max(done, 1) / elapsed
        remaining = max(total - done, 0)
        training_progress.steps(tenant_id, done, total, eta_seconds=remaining / rate, loss=loss)
        training_progress.stage(tenant_id, task, training_progress.WORKING)

    return follow


def _train_words(
    tenant_id: int | None,
    training_sentences: list[str],
    materials: TrainingSet,
    config: dict[str, Any] | None,
    steps: int,
) -> dict[str, Any]:
    """Train the neural word model on everything the learner knows."""
    from app.services import neural
    from app.services.language import runtime as runtime_service

    corpus = [
        *training_sentences,
        *runtime_service.learning_examples(tenant_id),
        *(response for _, response in materials.instructions),
    ]
    job = neural.TrainingJob(
        corpus={"sentences": len(corpus), "instructions": len(materials.instructions)},
        config=config,
        settings=neural.TrainingSettings(steps=steps),
        progress=_follow_steps(tenant_id, "words"),
    )
    return neural.engine_for(tenant_id).train_words(corpus, job)


def train(
    tenant_id: int | None,
    materials: TrainingSet,
    config: dict[str, Any] | None = None,
    steps: int = 0,
) -> dict[str, Any]:
    """Train the word model and the heads from the tenant's material.

    ``steps`` of 0 means the word model trains as long as its material asks
    for. The word model learns from everything the tenant has taught: every
    lesson, every approved answer, every correction and every document is in
    the material, and the vocabulary grows with it.
    """
    with _LOCK:
        feedback = load_feedback(tenant_id)
    corrections = [entry["correction"] for entry in feedback if entry.get("correction")]
    sentences = [sentence for sentence in materials.sentences if sentence]
    training_progress.start(tenant_id)
    mark_training(tenant_id, True)
    word_report: dict[str, Any] = {}
    try:
        word_report = _train_words(tenant_id, sentences, materials, config, steps)
        training_progress.stage(tenant_id, "words", training_progress.DONE)
        emotion_examples = train_emotion_head(tenant_id, sentences)
        training_progress.stage(tenant_id, "emotion", training_progress.DONE)
        intent_examples = train_intent_head(tenant_id, materials)
        training_progress.stage(tenant_id, "intent", training_progress.DONE)
        if feedback:
            train_feedback_heads(tenant_id, feedback)
        training_progress.stage(tenant_id, "feedback", training_progress.DONE)
    finally:
        clear_training(tenant_id)
        training_progress.finish(tenant_id, word_report.get("stop_reason"))
    return {
        "words": word_report,
        "emotion_examples": emotion_examples,
        "intent_examples": intent_examples,
        "feedback_entries": len(feedback),
        "corrections": len(corrections),
    }
