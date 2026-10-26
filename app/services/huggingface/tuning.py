"""Fine-tuning a small Hugging Face base model on the tenant's own material.

The run is a plain causal-language-model loop: every example is read whole,
the model predicts the next token, and the loss is measured as it goes. The
loop reports the step and the loss so the Base model tab can show the run it
is watching, and it writes the fine-tuned weights where the chat reads them.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.services.huggingface.catalog import name_of
from app.services.huggingface.text import training_texts

MAX_LENGTH = 256
BATCH_SIZE = 2
LEARNING_RATE = 5e-4
REPORT_EVERY = 5

Progress = Callable[[int, str, float | None], None]


def _device(torch: Any) -> Any:
    """The device the fine-tuning runs on."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name() -> str:
    """The name of the device a run would use, without loading a model."""
    import torch

    return _device(torch).type


def _load(model_id: str) -> tuple[Any, Any]:
    """The base model and its tokenizer, downloaded once when missing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer


def _batch(texts: Sequence[str], positions: Sequence[int], tokenizer: Any, device: Any) -> dict:
    """One padded batch, with the padding left out of the loss."""
    encoded = tokenizer(
        [texts[position] for position in positions],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels,
    }


def _order(count: int) -> list[int]:
    """The examples of one pass, in a new order."""
    order = list(range(count))
    random.shuffle(order)
    return order


def tune(
    model_id: str,
    folder: Path,
    pairs: Sequence[tuple[str, str]],
    steps: int,
    progress: Progress,
) -> dict[str, Any]:
    """Fine-tune the base model and keep the weights in `folder`."""
    import torch

    texts = [text + "\n" for text in training_texts(pairs)]
    if not texts:
        raise ValueError("There is no taught material to train on yet.")
    started = time.time()
    progress(0, "download", None)
    model, tokenizer = _load(model_id)
    device = _device(torch)
    model.to(device)
    progress(0, "tune", None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    order: list[int] = []
    losses: list[float] = []
    for step in range(1, steps + 1):
        if len(order) < BATCH_SIZE:
            order.extend(_order(len(texts)))
        positions, order = order[:BATCH_SIZE], order[BATCH_SIZE:]
        batch = _batch(texts, positions, tokenizer, device)
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        if step % REPORT_EVERY == 0 or step == steps:
            recent = losses[-REPORT_EVERY:]
            progress(step, "tune", round(sum(recent) / len(recent), 4))
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    recent = losses[-REPORT_EVERY:]
    return {
        "model_id": model_id,
        "name": name_of(model_id),
        "steps": steps,
        "loss": round(sum(recent) / len(recent), 4),
        "seconds": round(time.time() - started, 1),
        "examples": len(texts),
    }
