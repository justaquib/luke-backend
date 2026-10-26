"""A small neural brain for the local assistant.

The assistant already has a neural cortex: the MiniLM transformer that
ChromaDB uses to embed every taught question, answer and document. This
module adds the missing parts of a brain on top of it:

* a **word model** - a transformer trained on the material the tenant taught,
  so the learner can write a sentence word by word,
* an **emotion head** - a trained classifier that reads the feeling in a
  message,
* a **clarify head** - a trained classifier that decides whether the
  assistant has enough to answer or should ask a question back,
* a **confidence head** - a trained classifier that estimates whether the
  answer it is about to give is likely to be right.

The work is split by concern: ``brain_levels`` owns the ladder,
``brain_rungs`` and ``brain_rung_results`` measure it, ``brain_material``
reads what is taught, ``brain_heads`` trains and reads the classifiers,
``brain_training`` owns a training run, ``brain_feedback`` keeps the
trainer's verdicts, and ``brain_state`` assembles the state and the level.
This module is the one place the rest of the app imports them from.
"""

from __future__ import annotations

from app.services import vector_store
from app.services.brain_feedback import (
    Feedback,
    feedback_path,
    learned_corrections,
    load_feedback,
    mark_correction_learned,
    record_feedback,
    save_feedback,
    verify_correction,
)
from app.services.brain_generality import generality_score
from app.services.brain_heads import (
    answer_confidence,
    emotion_of,
    emotion_training_sentences,
    feel,
    should_ask_back,
)
from app.services.brain_levels import (
    BENCHMARK_NOTE,
    BENCHMARK_TITLE,
    LUKE_LEVELS,
    approve_level,
    approved_level,
    level_order,
    level_value,
    training_volume_level,
)
from app.services.brain_material import coverage, model_capabilities, next_steps
from app.services.brain_paths import brain_dir, brain_path
from app.services.brain_rung_results import rung_results
from app.services.brain_rungs import CORRECTION_LIMIT, GENERALITY_PROBES, rung_probes
from app.services.brain_state import model_level, state
from app.services.brain_training import (
    TRAINING_TIMEOUT_SECONDS,
    clear_training,
    is_training,
    mark_training,
    train,
    training_path,
    word_state,
)

__all__ = [
    "BENCHMARK_NOTE",
    "BENCHMARK_TITLE",
    "CORRECTION_LIMIT",
    "GENERALITY_PROBES",
    "LUKE_LEVELS",
    "TRAINING_TIMEOUT_SECONDS",
    "Feedback",
    "answer_confidence",
    "approve_level",
    "approved_level",
    "brain_dir",
    "brain_path",
    "clear_training",
    "coverage",
    "emotion_of",
    "emotion_training_sentences",
    "feedback_path",
    "feel",
    "generality_score",
    "is_training",
    "learned_corrections",
    "level_order",
    "level_value",
    "load_feedback",
    "mark_correction_learned",
    "mark_training",
    "model_capabilities",
    "model_level",
    "next_steps",
    "record_feedback",
    "rung_probes",
    "rung_results",
    "save_feedback",
    "should_ask_back",
    "state",
    "train",
    "training_path",
    "training_volume_level",
    "vector_store",
    "verify_correction",
    "word_state",
]
