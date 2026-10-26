"""The Hugging Face base models the trainer can fine-tune.

Only small models are offered: they load and fine-tune on one machine, which
is where this assistant runs. The list is a catalogue, not a download: a model
is used only after the trainer starts a run.
"""

from __future__ import annotations

from typing import Any

BASE_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "name": "SmolLM2 135M",
        "parameters": "135M",
        "note": "Small enough to fine-tune quickly on this machine.",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "name": "SmolLM2 360M Instruct",
        "parameters": "360M",
        "note": "Answers follow instructions more closely; slower to fine-tune.",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen2.5 0.5B Instruct",
        "parameters": "0.5B",
        "note": "The largest offered; it needs the most time and memory.",
    },
)


def base_models() -> list[dict[str, Any]]:
    """The catalogue the Base model tab shows."""
    return [dict(model) for model in BASE_MODELS]


def known(model_id: str) -> bool:
    """Whether the trainer offers this base model."""
    return any(model["id"] == model_id for model in BASE_MODELS)


def name_of(model_id: str) -> str:
    """The short name of a base model, for a label beside an answer."""
    for model in BASE_MODELS:
        if model["id"] == model_id:
            return str(model["name"])
    return model_id.rsplit("/", 1)[-1]
