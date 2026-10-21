"""The transformer the tenant's own words train.

Everything the writer needs is here: the model, the words it reads, the loop
that trains it, and the writing it does.
"""

from __future__ import annotations

from app.services.transformer.model import MODEL_CONFIG, TinyTransformer, device_name
from app.services.transformer.tokens import tokenize
from app.services.transformer.writer import TransformerWriter

__all__ = [
    "MODEL_CONFIG",
    "TinyTransformer",
    "TransformerWriter",
    "device_name",
    "tokenize",
]
