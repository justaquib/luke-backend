"""What Luke's word model expects next, and the words it looked at.

The word model already predicts the next word. This module adds the one thing
the page needs beside that prediction: the attention the last block paid to the
words of the prefix, averaged over its heads, so the heatmap is a measurement
and not a drawing.
"""

from __future__ import annotations

from typing import Any

import torch

from app.services.neural.tokens import FINISH, START, is_a_word
from app.services.neural.words import PUNCTUATION, WordTask

ATTENTION_TOKENS = 12


def _labels(task: WordTask, ids: list[int]) -> list[str]:
    """The prefix, as the short labels the heatmap shows."""
    labels = [
        "start" if task.vocabulary[token] == START else task.vocabulary[token] for token in ids
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
    """The words the model expects next, with the chance it gives each."""
    probabilities = torch.softmax(logits, dim=-1)
    values, positions = torch.topk(probabilities, min(limit * 3, len(task.vocabulary)))
    rows: list[dict[str, Any]] = []
    for value, position in zip(values.tolist(), positions.tolist(), strict=False):
        word = task.vocabulary[position]
        if word == FINISH or not is_a_word(word) or word in PUNCTUATION:
            continue
        rows.append(
            {
                "word": word,
                "probability": round(float(value), 5),
                "count": int(task.word_counts.get(word, 0)),
            }
        )
        if len(rows) == limit:
            break
    return rows


def predict_with_attention(task: WordTask | None, prefix: str, limit: int = 5) -> dict | None:
    """The next words and the attention over the prefix, from the trained model."""
    if task is None or not task.trained or task.model is None:
        return None
    tokens = task._prefix_ids(prefix)
    attention: list[torch.Tensor] = []
    device = next(task.model.parameters()).device
    with torch.no_grad():
        logits = task.model(torch.tensor([tokens], dtype=torch.long, device=device), attention)[
            0, -1
        ]
    for token_id in task._banned(tokens, None):
        logits[token_id] = -float("inf")
    return {
        "tokens": _labels(task, tokens),
        "attention": _attention(attention),
        "next": _next_words(task, logits, limit),
    }
