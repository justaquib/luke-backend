"""The held-out split: a group is never on both sides of the training."""

import math

from app.services.neural.examples import Example
from app.services.neural.measurement import chance_loss, group_split
from app.services.neural.settings import TrainingSettings


def _examples(groups: int, per_group: int) -> list[Example]:
    return [
        Example([1, 2, 3, index], 1, f"group-{index % groups}")
        for index in range(groups * per_group)
    ]


def test_a_group_is_never_on_both_sides():
    settings = TrainingSettings(validation_share=0.2, min_validation=4, min_validation_groups=4)
    train, held = group_split(_examples(20, 2), settings)

    assert held
    assert not {example.group for example in train} & {example.group for example in held}
    assert len(held) < len(train) + len(held)


def test_too_few_groups_means_no_held_out_number():
    settings = TrainingSettings(min_validation=4, min_validation_groups=8)
    examples = _examples(3, 10)

    train, held = group_split(examples, settings)

    assert held == []
    assert train == examples


def test_too_few_held_out_examples_means_no_held_out_number():
    settings = TrainingSettings(min_validation=100, min_validation_groups=2, validation_share=0.1)
    examples = _examples(10, 2)

    train, held = group_split(examples, settings)

    assert held == []
    assert train == examples


def test_the_split_is_the_same_on_every_read():
    settings = TrainingSettings(validation_share=0.2, min_validation=4, min_validation_groups=4)
    examples = _examples(20, 2)

    first = group_split(examples, settings)
    second = group_split(examples, settings)

    assert [example.sequence for example in first[1]] == [example.sequence for example in second[1]]


def test_chance_is_the_log_of_the_vocabulary():
    assert chance_loss(1000) == round(math.log(1000), 4)
