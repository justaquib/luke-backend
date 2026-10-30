"""The adapter layers are named by the model that is loaded, not by a recipe."""

from __future__ import annotations

from collections.abc import Iterator

from app.services.training import targets


class Linear:
    """The only class an adapter wraps, named as torch names it."""


class LayerNorm:
    """A normalisation layer, which an adapter leaves alone."""


class Embedding:
    """A vocabulary layer, which is a different operation from this one."""


class FakeModel:
    """A module tree in the shape a checkpoint really has, with no weights."""

    def __init__(self, modules: dict[str, object]) -> None:
        self._modules = modules

    def named_modules(self) -> Iterator[tuple[str, object]]:
        yield "", self
        yield from self._modules.items()


def qwen_like() -> FakeModel:
    """The layer names measured from the pinned checkpoint under transformers 5."""
    block = {
        f"model.layers.0.self_attn.{name}": Linear()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    block.update(
        {f"model.layers.0.mlp.{name}": Linear() for name in ("gate_proj", "up_proj", "down_proj")}
    )
    block["model.layers.0.input_layernorm"] = LayerNorm()
    block["model.embed_tokens"] = Embedding()
    block["lm_head"] = Linear()
    return FakeModel(block)


def other_shape() -> FakeModel:
    """A checkpoint whose attention is fused and whose feed-forward is named apart."""
    return FakeModel(
        {
            "transformer.h.0.attn.c_attn": Linear(),
            "transformer.h.0.attn.c_proj": Linear(),
            "transformer.h.0.mlp.wte": Linear(),
            "transformer.h.0.ln_1": LayerNorm(),
        }
    )


def test_the_targets_are_the_linear_leaves_of_the_loaded_model() -> None:
    assert targets.target_modules(qwen_like()) == (
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    )


def test_the_names_a_recipe_would_have_given_are_absent_from_this_checkpoint() -> None:
    names = set(targets.target_modules(qwen_like()))
    assert "c_attn" not in names
    assert "c_proj" not in names


def test_a_model_of_another_shape_gives_its_own_names() -> None:
    assert targets.target_modules(other_shape()) == ("c_attn", "c_proj", "wte")


def test_a_layer_that_is_not_linear_is_offered_to_nobody() -> None:
    assert "input_layernorm" not in targets.linear_paths(qwen_like())


def test_the_layers_that_read_or_write_the_vocabulary_are_left_alone() -> None:
    paths = targets.linear_paths(qwen_like())
    assert not any("lm_head" in path for path in paths)
    assert not any("embed_tokens" in path for path in paths)


def test_every_path_offered_is_a_layer_the_adapter_can_wrap() -> None:
    model = qwen_like()
    modules = dict(model.named_modules())
    assert all(isinstance(modules[path], Linear) for path in targets.linear_paths(model))
