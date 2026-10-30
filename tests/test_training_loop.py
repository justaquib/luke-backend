"""The Phase 7 step loop: the same steps, the same order, for every operation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.services.training import loop
from app.services.training.formats import IGNORE, Row

VOCAB = 9
SIZE = 4


class PadTokenizer:
    """The only thing the loop reads off a tokenizer: the id it pads with."""

    pad_token_id = 0


class NoPadTokenizer:
    pad_token_id = None


class TinyNet(torch.nn.Module):
    """A model in the shape the loop asks for: ids in, a loss out."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, SIZE)
        self.head = torch.nn.Linear(SIZE, VOCAB)

    def forward(self, input_ids: object, attention_mask: object, labels: object) -> object:
        logits = self.head(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=IGNORE
        )
        return SimpleNamespace(loss=loss)


def row(position: int, supervised: int = 3) -> Row:
    ids = tuple(range(1, 1 + min(position + 2, VOCAB - 1)))
    masked = max(len(ids) - supervised, 0)
    return Row(ids, (IGNORE,) * masked + ids[masked:], f"source {position}", "sft", "train")


def settings_of(
    steps: int = 7, batch_size: int = 2, seed: int = loop.DEFAULT_SEED
) -> loop.Settings:
    return loop.Settings(steps=steps, batch_size=batch_size, learning_rate=0.1, seed=seed)


def reporter() -> tuple[list[tuple[int, str, float | None]], loop.Progress]:
    seen: list[tuple[int, str, float | None]] = []
    return seen, lambda step, stage, loss: seen.append((step, stage, loss))


def test_the_loop_runs_the_steps_it_was_asked_for_and_reports_them() -> None:
    seen, progress = reporter()
    rows = [row(index) for index in range(6)]
    result = loop.train(loop.Run(TinyNet(), PadTokenizer(), rows, settings_of(7), progress))
    assert result.steps == 7
    assert result.stop == loop.FINISHED
    assert [entry[0] for entry in seen] == [5, 7]
    assert all(entry[1] == "tune" for entry in seen)


def test_the_loss_a_run_reports_is_measured_from_the_model() -> None:
    _, progress = reporter()
    rows = [row(index) for index in range(4)]
    result = loop.train(loop.Run(TinyNet(), PadTokenizer(), rows, settings_of(6), progress))
    assert result.loss is not None
    assert result.loss > 0
    assert result.device in {"mps", "cpu"}
    assert result.seconds >= 0


def test_a_run_asked_to_stop_says_that_is_why_it_stopped() -> None:
    _, progress = reporter()
    asked = {"calls": 0}

    def stop_after_three() -> bool:
        asked["calls"] += 1
        return asked["calls"] > 3

    rows = [row(index) for index in range(6)]
    result = loop.train(
        loop.Run(TinyNet(), PadTokenizer(), rows, settings_of(10), progress, stop_after_three)
    )
    assert result.stop == loop.CANCELLED
    assert result.steps == 3
    assert result.rows == 6


def test_the_order_of_the_rows_is_drawn_the_same_way_every_time() -> None:
    assert loop.order(10, 3) == loop.order(10, 3)
    assert sorted(loop.order(10, 3)) == list(range(10))
    assert loop.order(10, 3) != loop.order(10, 4)


def test_a_pass_is_walked_once_before_another_is_started() -> None:
    plan = loop.batches(4, 4, 2, loop.DEFAULT_SEED)
    assert [len(batch) for batch in plan] == [2, 2, 2, 2]
    assert sorted([*plan[0], *plan[1]]) == [0, 1, 2, 3]
    assert sorted([*plan[2], *plan[3]]) == [0, 1, 2, 3]


def test_a_corpus_smaller_than_a_batch_is_not_padded_with_repeats() -> None:
    plan = loop.batches(1, 3, 2, loop.DEFAULT_SEED)
    assert plan == ((0,), (0,), (0,))


def test_a_run_with_nothing_to_train_on_is_refused() -> None:
    _, progress = reporter()
    with pytest.raises(ValueError, match="no examples"):
        loop.train(loop.Run(TinyNet(), PadTokenizer(), [], settings_of(3), progress))


def test_a_run_asked_for_no_steps_is_refused_by_number() -> None:
    _, progress = reporter()
    with pytest.raises(ValueError, match="at least one step, got 0"):
        loop.train(loop.Run(TinyNet(), PadTokenizer(), [row(1)], settings_of(0), progress))


def test_a_tokenizer_that_cannot_pad_is_named_before_the_first_step() -> None:
    _, progress = reporter()
    with pytest.raises(ValueError, match="no padding token"):
        loop.train(loop.Run(TinyNet(), NoPadTokenizer(), [row(1)], settings_of(2), progress))


def test_only_the_free_parameters_are_handed_to_the_optimizer() -> None:
    model = TinyNet()
    for parameter in model.embedding.parameters():
        parameter.requires_grad_(False)
    free = loop.trainable_parameters(model)
    assert len(free) == 2
    assert all(parameter.shape[0] == VOCAB for parameter in free)


def test_a_run_counts_the_rows_it_was_given_and_the_sources_behind_them() -> None:
    _, progress = reporter()
    rows = [row(0), row(1), row(1)]
    result = loop.train(loop.Run(TinyNet(), PadTokenizer(), rows, settings_of(4), progress))
    assert result.rows == 3
    assert result.sources == 2


def test_the_seed_of_a_run_is_installed_into_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = torch.manual_seed

    def record(seed: int) -> torch.Generator:
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(torch, "manual_seed", record)
    _, progress = reporter()
    rows = [row(index) for index in range(4)]
    loop.train(loop.Run(TinyNet(), PadTokenizer(), rows, settings_of(3, seed=7), progress))
    assert calls == [7]
