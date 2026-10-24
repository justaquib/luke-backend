"""The writer reading one question, measured.

A trace is taken on the writer's own weights with dropout off, so every number
it reports is the model's: the tokens it read, the vectors they became, the
attention between them, the words it scored and the word it chose. The answer
a page shows may have been served from a lesson instead of written; the trace
does not claim otherwise, it only shows what the writer itself does with the
same question.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from app.services.neural import answering
from app.services.neural.settings import ReadingSettings
from app.services.neural.tokens import FINISH

# The most words the trace lets the writer produce, and the vector values shown.
TRACE_WORDS = 24
PREVIEW_VALUES = 8

# Six decimal places keep the weights readable and the payload small.
PLACES = 6


def trace(writer: Any, question: str, max_words: int = TRACE_WORDS) -> dict[str, Any]:
    """Trace the writer reading a question, or say why there is no trace."""
    if writer is None or not writer.trained or writer.model is None:
        return {"trained": False, "reason": "The writer has not been trained yet."}
    started = time.perf_counter()
    model = writer.model
    reading = answering.encode_instruction(
        writer,
        answering.Request(question, max_words=max_words, reading=ReadingSettings()),
    )
    choices: list[dict[str, Any]] = []
    attention: list[torch.Tensor] = []
    was_training = model.training
    with torch.inference_mode():
        model.eval()
        try:
            inputs = torch.tensor([reading], dtype=torch.long)
            model(inputs, attention=attention)
            written = answering.write(writer, list(reading), max_words, trace=choices)
        finally:
            model.train(was_training)
    return {
        "trained": True,
        "model": _model_report(model, writer),
        "reading": {
            "tokens": [{"id": token, "token": writer.vocabulary[token]} for token in reading],
            "words": len(question.split()),
        },
        "embeddings": _embeddings(model, writer, reading),
        "attention": _attention(attention, writer, reading),
        "next": choices[0] if choices else None,
        "generation": {
            "steps": choices,
            "text": writer.decode(written) or None,
            "stop": "end" if choices and choices[-1]["token"] == FINISH else "limit",
            "seconds": round(time.perf_counter() - started, 3),
        },
        "measured_at": round(time.time(), 3),
    }


def _model_report(model: torch.nn.Module, writer: Any) -> dict[str, Any]:
    """The shape of the model the trace was taken on."""
    return {
        "layers": len(model.blocks),
        "heads": int(model.config["heads"]),
        "d_model": int(model.config["d_model"]),
        "context": int(model.config["context"]),
        "vocabulary": len(writer.vocabulary),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(next(model.parameters()).device),
    }


def _embeddings(
    model: torch.nn.Module,
    writer: Any,
    tokens: list[int],
) -> dict[str, Any]:
    """The learned vector of every token the writer read."""
    weight = model.token_embedding.weight
    rows = []
    for token in tokens:
        vector = weight[token].detach()
        rows.append(
            {
                "token": writer.vocabulary[token],
                "values": [round(float(value), 3) for value in vector[:PREVIEW_VALUES].tolist()],
                "norm": round(float(vector.norm()), 3),
            }
        )
    return {"dimensions": int(weight.shape[1]), "preview": PREVIEW_VALUES, "tokens": rows}


def _attention(
    captured: list[torch.Tensor],
    writer: Any,
    tokens: list[int],
) -> dict[str, Any]:
    """The attention of every layer and head over the tokens that were read."""
    layers = [
        [
            [[round(float(value), PLACES) for value in row] for row in head.tolist()]
            for head in weights[0]
        ]
        for weights in captured
    ]
    return {
        "tokens": [writer.vocabulary[token] for token in tokens],
        "layers": layers,
    }
