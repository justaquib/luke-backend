"""Reading a Hugging Face base model, and naming the device a run uses.

The training loop itself lives in ``app.services.training``: this module only
knows how to read a checkpoint and its tokenizer, so there is one loop and one
place where a run's steps are counted.
"""

from __future__ import annotations

from typing import Any


def _device(torch: Any) -> Any:
    """The device the fine-tuning runs on."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name() -> str:
    """The name of the device a run would use, without loading a model."""
    import torch

    return _device(torch).type


def load_base(model_id: str) -> tuple[Any, Any]:
    """The base model and its tokenizer, downloaded once when missing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer
