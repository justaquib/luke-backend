"""What Luke's word model expects next, and the tokens it looked at.

The word model already predicts the next token. This module adds the one thing
the page needs beside that prediction: the attention the last block paid to the
tokens of the prefix, averaged over its heads, so the heatmap is a measurement
and not a drawing. A BPE checkpoint is shown in its own pieces, never relabelled
as words.
"""

from __future__ import annotations

from typing import Any

import torch

from app.services.neural.tokens import START
from app.services.neural.words import WordTask

ATTENTION_TOKENS = 12


def _labels(task: WordTask, ids: list[int]) -> list[str]:
    """The prefix, as the short labels the heatmap shows."""
    labels = [
        "start" if task.token_text(token) == START else task.token_text(token) for token in ids
    ]
    return labels[-ATTENTION_TOKENS:]


def _attention(layers: list[torch.Tensor]) -> list[list[float]] | None:
    """The last layer's attention over the prefix, averaged over its heads."""
    if not layers:
        return None
    matrix = layers[-1][0].mean(dim=0)
    window = matrix[-ATTENTION_TOKENS:, -ATTENTION_TOKENS:]
    return [[round(float(value), 4) for value in row] for row in window.tolist()]


def _next_words(task: WordTask, logits: torch.Tensor, limit: int) -> list[dict[str, Any]]:
    """The tokens the model expects next, with the chance it gives each."""
    return task.rows_from_logits(torch.softmax(logits, dim=-1), limit)


def predict_with_attention(task: WordTask | None, prefix: str, limit: int = 5) -> dict | None:
    """The next words and the attention over the prefix, from the trained model."""
    if task is None or not task.trained or task.model is None:
        return None
    tokens = task._prefix_ids(prefix)
    # The model has exactly `context` position embeddings, so a longer prefix
    # reads its tail: the same window the writing path uses in `writer.py`.
    window = tokens[-int(task.config["context"]) :]
    attention: list[torch.Tensor] = []
    device = next(task.model.parameters()).device
    with torch.no_grad():
        logits = task.model(torch.tensor([window], dtype=torch.long, device=device), attention)[
            0, -1
        ]
    for token_id in task._banned(tokens, None):
        logits[token_id] = -float("inf")
    return {
        "tokens": _labels(task, window),
        "attention": _attention(attention),
        "next": _next_words(task, logits, limit),
    }
