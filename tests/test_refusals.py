"""The refusal probes are held out of the requests the training refuses."""

from app.services.language import negatives


def _cards() -> list[dict[str, str]]:
    return [
        {"question": "What is wind?", "answer": "The wind blows.", "subject": "Nature"},
        {"question": "What is rain?", "answer": "Rain falls.", "subject": "Nature"},
    ]


def test_the_training_negatives_respect_the_cap() -> None:
    cards = _cards()

    trained = negatives.build_negatives(cards)

    assert len(trained) == negatives.MAX_NEGATIVE_EXAMPLES
    assert all(answer == negatives.REFUSAL for _, answer in trained)


def test_held_out_negatives_are_never_in_the_training_negatives() -> None:
    cards = _cards()

    trained = set(negatives.build_negatives(cards))
    held = negatives.build_held_out_negatives(cards, 8)

    assert len(held) == 8
    assert not (set(held) & trained)


def test_negatives_never_name_a_taught_topic() -> None:
    cards = _cards()

    unknown = negatives.unknown_topics(cards)
    held = negatives.build_held_out_negatives(cards, 8)

    assert "wind" not in unknown
    assert "rain" not in unknown
    assert "accordion" in unknown
    assert all(any(topic in request for topic in unknown) for request, _ in held)


def test_every_unknown_topic_is_asked_in_every_wording() -> None:
    cards = _cards()

    requests = [request for request, _ in negatives._all_negatives(cards)]

    assert any("accordion" in request for request in requests)
    assert any("wind" in request for request in requests) is False
    assert len(requests) == len(negatives.unknown_topics(cards)) * len(negatives.PHRASINGS)
