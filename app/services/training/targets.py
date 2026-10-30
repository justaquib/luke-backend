"""Which layers an adapter may touch, read from the model it is given.

The usual recipe for a Qwen2 adapter names ``c_attn`` and ``c_proj``, and those
are not the layers this checkpoint has: under ``transformers 5.20.2`` its
attention is ``self_attn.{q,k,v,o}_proj`` and its feed-forward is
``mlp.{gate,up,down}_proj``. Asking for the recipe names is refused by `peft`
before a single step runs, which is measured in ``docs/07-model-measurements.md``.
The names are therefore taken from the loaded model, so a run adapts the layers
it truly has and a checkpoint of another shape does not inherit this one's.
"""

from __future__ import annotations

from typing import Any

# The layer class an adapter wraps. Anything else is left alone.
LINEAR = "Linear"

# Layers that read or write the vocabulary rather than think in the middle of
# the network: adapting them would move the words the tokenizer stores, which
# is a different operation from teaching the model new answers.
UNADAPTED = ("lm_head", "embed_tokens", "embed_out")


def _is_linear(module: Any) -> bool:
    return type(module).__name__ == LINEAR


def _wanted(path: str) -> bool:
    """Whether this layer takes part in a run."""
    return not any(name in path for name in UNADAPTED)


def linear_paths(model: Any) -> tuple[str, ...]:
    """The dotted path of every linear layer the model actually has."""
    return tuple(
        path
        for path, module in model.named_modules()
        if path and _is_linear(module) and _wanted(path)
    )


def target_modules(model: Any) -> tuple[str, ...]:
    """The layer names a config is given, in the model's own words.

    Only the last part of each path is needed by an adapter config, and the same
    name appears in every block, so the answer is the distinct set.
    """
    return tuple(sorted({path.rsplit(".", 1)[-1] for path in linear_paths(model)}))
