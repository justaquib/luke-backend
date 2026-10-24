"""Tests for the word backend that serves, and what the learner state says.

There is one word backend: the neural word model. A tenant without one is
told so, and the gate only describes how well a trained model measured. The
state names the backend that answered.
"""

from app.services.language import runtime
from app.services.neural import engine as engine_module
from app.services.neural.serving import WORD_LOSS_RATIO, serving_word, word_gate


class FakeNeural:
    """A neural word model that reports the numbers it is given."""

    def __init__(self, report: dict) -> None:
        self.report = report
        self.sentence_count = 10
        self.token_count = 60
        self.transition_count = 50
        self.example_count = 10
        self.vocabulary = ["the", "cow"]

    def predict_next(self, prefix, limit=5, allowed=None):
        return [{"word": "milk", "count": 3, "probability": 0.4, "source": "neural"}][:limit]

    def generate_sentence(self, seed=None, max_words=16, use_taught=True, allowed=None):
        return "The cow gives milk."


def test_a_model_without_enough_held_out_numbers_is_reported():
    ready, reason = word_gate({"validation_examples": 10, "validation_loss": 1.0})

    assert ready is False
    assert "held-out" in reason


def test_a_model_without_a_held_out_loss_is_reported():
    ready, reason = word_gate({"validation_examples": 200, "validation_loss": None})

    assert ready is False
    assert "no held-out loss" in reason


def test_a_model_that_beats_chance_is_marked_good():
    ready, reason = word_gate(
        {"validation_examples": 200, "validation_loss": 2.0, "chance_loss": 3.0}
    )

    assert ready is True
    assert "2.0" in reason


def test_a_model_above_the_ratio_is_marked_weak():
    loss = WORD_LOSS_RATIO * 3.0 + 0.1

    ready, _ = word_gate({"validation_examples": 200, "validation_loss": loss, "chance_loss": 3.0})

    assert ready is False


def test_a_tenant_without_a_word_model_is_told_so():
    serving = serving_word(None, allowed=None)

    assert serving.backend == "untrained"
    assert "not been trained" in serving.backend_reason
    assert serving.predict_next("the", limit=3) == []
    assert serving.generate_sentence() is None
    assert serving.score_words(["the", "cow"]) == 0.0
    assert serving.sentence_count == 0
    assert serving.example_count == 0


def test_the_neural_model_serves_with_its_measured_reason():
    neural = FakeNeural({"validation_examples": 200, "validation_loss": 1.5, "chance_loss": 3.0})

    serving = serving_word(neural, allowed={"the", "cow", "milk"})

    assert serving.backend == "neural"
    assert "1.5" in serving.backend_reason
    assert serving.predict_next("the cow", limit=2)[0]["source"] == "neural"
    assert serving.generate_sentence(seed="the", max_words=4) == "The cow gives milk."


def test_a_trained_model_serves_even_before_it_measured_well():
    neural = FakeNeural({"validation_examples": 4, "validation_loss": 2.9})

    serving = serving_word(neural)

    assert serving.backend == "neural"
    assert "only 4 held-out examples" in serving.backend_reason


def test_the_learner_state_keeps_the_old_keys_and_names_the_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    engine_module.forget(987001)

    state = runtime.get_state(987001)

    assert state["backend"] == "untrained"
    assert state["model_name"] == "No word model yet"
    assert "not been trained" in state["backend_reason"]
    assert state["example_count"] == 0
    assert state["example_total"] == len(state["examples"])
    assert state["examples"]
    assert all(example["kind"] == "foundation" for example in state["examples"])
    for key in (
        "sentence_count",
        "token_count",
        "transition_count",
        "word_vocabulary_size",
        "vocabulary_size",
        "next_words",
    ):
        assert key in state
    assert state["generated_example"] is None
    assert state["next_words"] == []
