"""An optional cross-encoder pass over the fused passages.

The embedding scores a question and a passage separately, so it can miss a
passage that answers the question word for word. A cross-encoder reads the two
together, which is slower and needs a downloaded model: it is off unless the
deployment turns it on, and a model that cannot load leaves the fused order
unchanged.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.rerank_model, num_labels=1)
    model.eval()
    return tokenizer, model


def _scores(question: str, texts: list[str]) -> list[float]:
    import torch

    tokenizer, model = _load()
    inputs = tokenizer(
        [question] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        return model(**inputs).logits.squeeze(-1).tolist()


def rerank(question: str, hits: list[dict], n_results: int) -> list[dict]:
    """Order the fused hits by reading the question with each passage."""
    if not settings.rerank_enabled or len(hits) < 2:
        return hits[:n_results]
    try:
        scores = _scores(question, [str(hit.get("text", "")) for hit in hits])
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.warning("Cross-encoder rerank skipped: %s", error)
        return hits[:n_results]
    ordered = sorted(zip(scores, hits, strict=True), key=lambda item: -item[0])
    return [hit for _, hit in ordered[:n_results]]
