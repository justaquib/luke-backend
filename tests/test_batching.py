"""Fixed batches: the same read every time, and no padding ever scored."""

from app.services.neural.examples import Example
from app.services.neural.measurement import build_batch, fixed_batches
from app.services.neural.settings import TrainingSettings


def _example(sequence: list[int], group: str = "g") -> Example:
    return Example(sequence, group)


def test_a_row_with_no_next_word_to_predict_is_dropped():
    rows = [
        _example([1]),
        _example([2, 3, 4, 5]),
    ]

    batch = build_batch(rows, pad_id=0, context=8)

    assert batch is not None
    assert batch.inputs.shape[0] == 1
    assert batch.inputs[0].tolist() == [2, 3, 4, 5]


def test_a_long_row_keeps_its_last_words_inside_the_window():
    batch = build_batch([_example(list(range(1, 9)))], pad_id=0, context=6)

    assert batch is not None
    assert batch.inputs[0].tolist() == [3, 4, 5, 6, 7, 8]
    assert batch.scored[0].tolist() == [True] * 6


def test_padding_is_never_scored():
    batch = build_batch(
        [_example([1, 2, 3]), _example([1, 2, 3, 4, 5])],
        pad_id=0,
        context=8,
    )

    assert batch is not None
    assert batch.inputs[0].tolist() == [1, 2, 3, 0, 0]
    assert batch.scored[0].tolist() == [True, True, True, False, False]


def test_the_same_seed_reads_the_same_batches():
    examples = [_example([1, 2, 3, i], f"group-{i}") for i in range(10)]
    settings = TrainingSettings(batch_size=2, check_batches=2)
    config = {"pad_id": 0, "context": 8}

    first = fixed_batches(examples, config, settings, seed=7)
    second = fixed_batches(examples, config, settings, seed=7)
    other = fixed_batches(examples, config, settings, seed=11)

    assert [batch.inputs.tolist() for batch in first] == [batch.inputs.tolist() for batch in second]
    assert [batch.inputs.tolist() for batch in first] != [batch.inputs.tolist() for batch in other]


def test_an_empty_read_is_no_batches():
    assert fixed_batches([], {"pad_id": 0, "context": 8}, TrainingSettings(), seed=7) == []
