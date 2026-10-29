"""The cross-encoder is optional: off, on, and unable to load."""

from app.config import settings
from app.services import rerank


def _hits():
    return [
        {"text": "Ferry timetable.", "metadata": {}, "distance": 0.1},
        {"text": "The voucher holds the number.", "metadata": {}, "distance": 0.2},
    ]


def test_it_keeps_the_fused_order_when_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)

    assert rerank.rerank("voucher", _hits(), 1) == _hits()[:1]


def test_it_reorders_by_the_models_score_when_switched_on(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(rerank, "_scores", lambda question, texts: [0.1, 0.9])

    assert rerank.rerank("voucher", _hits(), 1)[0]["text"] == "The voucher holds the number."


def test_a_model_that_cannot_load_leaves_the_order_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)

    def fail(question, texts):
        raise OSError("no model here")

    monkeypatch.setattr(rerank, "_scores", fail)

    assert rerank.rerank("voucher", _hits(), 2) == _hits()
