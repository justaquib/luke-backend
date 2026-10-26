"""Serving a fine-tuned teacher: the answer it writes and the words it expects.

A teacher is loaded once and kept while it is used. It answers from the
material the retriever found, and it reports the next word it would write with
the attention it paid to the words already there, so the page can show what the
model really measured instead of a picture of it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from app.services.huggingface import store

ATTENTION_TOKENS = 12
MAX_ANSWER_TOKENS = 96
MAX_PREDICT_TOKENS = 64
MAX_TEACHERS = 1

_LOCK = threading.Lock()
_CACHE: OrderedDict[str, tuple[float, Teacher]] = OrderedDict()


class Teacher:
    """One fine-tuned model, its tokenizer and where it was read from."""

    def __init__(self, model: Any, tokenizer: Any, folder: Path) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.folder = folder
        self.device = next(model.parameters()).device

    def generate(self, question: str, reference: str = "") -> str | None:
        """The answer the teacher writes, grounded on the reference it was given."""
        import torch

        from app.services.huggingface.text import prompt

        encoded = self.tokenizer(
            prompt(question, reference),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=MAX_ANSWER_TOKENS,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        written = self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return _clean_answer(written)

    def predict(self, text: str, limit: int = 5) -> dict[str, Any]:
        """The next words, and the attention over the words already written."""
        import torch

        encoded = self.tokenizer(
            text or " ",
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PREDICT_TOKENS,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded, output_attentions=True)
        ids = encoded["input_ids"][0].tolist()
        return {
            "tokens": _labels(self.tokenizer, ids),
            "attention": _attention(output.attentions),
            "next": _next_words(output.logits[0, -1], self.tokenizer, limit),
            "heads": int(getattr(self.model.config, "num_attention_heads", 0) or 0),
            "layers": int(getattr(self.model.config, "num_hidden_layers", 0) or 0),
        }


def _clean_answer(text: str) -> str | None:
    """The answer without the next question the model started to write."""
    text = text.split("\n", 1)[0]
    for marker in ("Question:", "Reference:", "Answer:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = " ".join(text.split()).strip()
    return text if len(text) >= 2 else None


def _label(tokenizer: Any, token_id: int) -> str:
    """One token as the heatmap's short label."""
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("Ġ", "").replace("▁", "").strip()
    return (text or "·")[:14]


def _labels(tokenizer: Any, ids: list[int]) -> list[str]:
    labels = [_label(tokenizer, token_id) for token_id in ids]
    return labels[-ATTENTION_TOKENS:]


def _attention(layers: tuple[Any, ...] | None) -> list[list[float]] | None:
    """The last layer's attention, averaged over its heads, for the page."""
    if not layers:
        return None
    matrix = layers[-1][0].mean(dim=0)
    window = matrix[-ATTENTION_TOKENS:, -ATTENTION_TOKENS:]
    return [[round(float(value), 4) for value in row] for row in window.tolist()]


def _next_words(logits: Any, tokenizer: Any, limit: int) -> list[dict[str, Any]]:
    """The words the model would write next, with the chance it gives each."""
    import torch

    probabilities = torch.softmax(logits, dim=-1)
    values, positions = torch.topk(probabilities, min(limit * 8, probabilities.shape[-1]))
    special = set(tokenizer.all_special_ids)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value, position in zip(values.tolist(), positions.tolist(), strict=False):
        if position in special:
            continue
        word = tokenizer.decode([position], skip_special_tokens=True).strip()
        if not word or word in seen or "\n" in word:
            continue
        if not any(character.isalnum() for character in word):
            continue
        seen.add(word)
        rows.append({"word": word, "probability": round(float(value), 5), "count": None})
        if len(rows) == limit:
            break
    return rows


def load(tenant_id: int | None, folder: str) -> Teacher | None:
    """The teacher of a folder, loaded once and reloaded when its file changes."""
    path = store.teacher_path(tenant_id, folder)
    weights = path / "model.safetensors"
    if not weights.is_file():
        return None
    stamp = weights.stat().st_mtime
    with _LOCK:
        cached = _CACHE.get(folder)
        if cached is not None and cached[0] == stamp:
            _CACHE.move_to_end(folder)
            return cached[1]
    teacher = _read(path)
    if teacher is None:
        return None
    with _LOCK:
        _CACHE[folder] = (stamp, teacher)
        _CACHE.move_to_end(folder)
        while len(_CACHE) > MAX_TEACHERS:
            _CACHE.popitem(last=False)
    return teacher


def _read(path: Path) -> Teacher | None:
    """Build a teacher from the files of one folder."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager")
    except (OSError, ValueError):
        return None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return Teacher(model, tokenizer, path)


def forget(folder: str | None = None) -> None:
    """Drop a loaded teacher, or every teacher, so the next call reads disk."""
    with _LOCK:
        if folder is None:
            _CACHE.clear()
        else:
            _CACHE.pop(folder, None)


def loaded() -> int:
    """How many teachers are loaded, so a test can hold the limit."""
    with _LOCK:
        return len(_CACHE)
