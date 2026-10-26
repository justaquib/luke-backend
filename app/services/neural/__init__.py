"""The neural engine of the app.

One package owns the tokenizer, the word model, its training, the held-out
measurement, the checkpoints and the routing that decides which word backend
serves. There is one model: the word model.
"""

from __future__ import annotations

from app.services.neural.engine import NeuralEngine, TrainingJob, engine_for, forget
from app.services.neural.model import WORD_MODEL_CONFIG, TinyTransformer, device_name
from app.services.neural.settings import DEFAULT_TRAINING, TrainingSettings
from app.services.neural.tokens import decode_text, tokenize
from app.services.neural.words import WordTask

__all__ = [
    "DEFAULT_TRAINING",
    "WORD_MODEL_CONFIG",
    "NeuralEngine",
    "TinyTransformer",
    "TrainingJob",
    "TrainingSettings",
    "WordTask",
    "decode_text",
    "device_name",
    "engine_for",
    "forget",
    "tokenize",
]
