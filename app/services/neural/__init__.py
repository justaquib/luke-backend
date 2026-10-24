"""The neural engine of the app.

One package owns the tokenizer, the models, the training, the held-out
measurement, the checkpoints and the routing that decides which word backend
serves. There is one word backend: the neural word model.
"""

from __future__ import annotations

from app.services.neural.engine import NeuralEngine, TrainingJob, engine_for, forget
from app.services.neural.model import MODEL_CONFIG, WORD_MODEL_CONFIG, TinyTransformer, device_name
from app.services.neural.settings import (
    DEFAULT_READING,
    DEFAULT_TRAINING,
    DEFAULT_WRITING,
    ReadingSettings,
    TrainingSettings,
    WritingSettings,
)
from app.services.neural.tasks import AnswerCorpus, AnswerTask, WordTask
from app.services.neural.tokens import decode_text, tokenize

__all__ = [
    "DEFAULT_READING",
    "DEFAULT_TRAINING",
    "DEFAULT_WRITING",
    "MODEL_CONFIG",
    "WORD_MODEL_CONFIG",
    "AnswerCorpus",
    "AnswerTask",
    "NeuralEngine",
    "ReadingSettings",
    "TinyTransformer",
    "TrainingJob",
    "TrainingSettings",
    "WordTask",
    "WritingSettings",
    "decode_text",
    "device_name",
    "engine_for",
    "forget",
    "tokenize",
]
