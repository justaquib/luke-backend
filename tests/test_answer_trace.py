"""Tests for the measured trace of the writer reading a question.

The trace is shown beside a chat answer, so its numbers must come from the
model itself: its tokens, its attention, its candidates and its choice.
"""

from app.services import neural
from app.services.neural import tracing
from app.services.neural.settings import TrainingSettings

CORPUS = [
    "The cow is an animal.",
    "The cow is a farm animal that gives milk.",
    "A cow gives milk.",
    "The moon shines at night.",
    "The night sky is black.",
    "Water is the clear liquid people drink.",
    "A liquid takes the shape of its container.",
]


def _writer() -> neural.AnswerTask:
    writer = neural.AnswerTask()
    writer.config = {**writer.config, "d_model": 24, "layers": 2, "heads": 3, "context": 32}
    report = writer.fit(neural.AnswerCorpus(CORPUS), settings=TrainingSettings(steps=30))
    assert report["trained"] is True
    return writer


def test_a_trace_reports_the_real_reading_attention_and_choice():
    writer = _writer()

    trace = tracing.trace(writer, "The cow is an animal.")

    assert trace["trained"] is True
    tokens = [item["token"] for item in trace["reading"]["tokens"]]
    assert tokens[0] == "<start>"
    assert "<instr>" in tokens
    assert tokens[-1] == "<resp>"
    assert trace["model"]["layers"] == 2
    assert trace["model"]["heads"] == 3
    assert trace["model"]["parameters"] > 0
    assert len(trace["embeddings"]["tokens"]) == len(tokens)
    assert trace["attention"]["tokens"] == tokens

    layers = trace["attention"]["layers"]
    assert len(layers) == 2, "one attention map per transformer layer"
    assert len(layers[0]) == 3, "one map per head"
    assert len(layers[0][0]) == len(tokens)
    assert abs(sum(layers[0][0][-1]) - 1.0) < 0.01, "a query row is a distribution"
    assert layers[0][0][0][-1] == 0.0, "a token cannot attend to the future"

    choice = trace["next"]
    assert choice is not None
    candidates = choice["candidates"]
    assert candidates
    assert candidates[0]["probability"] >= candidates[-1]["probability"]
    assert choice["probability"] == candidates[0]["probability"]

    generation = trace["generation"]
    assert generation["steps"]
    assert generation["stop"] in {"end", "limit"}
    assert generation["seconds"] >= 0
    if generation["text"]:
        assert "<end>" not in generation["text"]
        assert "<start>" not in generation["text"]


def test_an_untrained_writer_has_no_trace_to_show():
    trace = tracing.trace(neural.AnswerTask(), "The cow is an animal.")

    assert trace["trained"] is False
    assert trace["reason"]
    assert tracing.trace(None, "The cow is an animal.")["trained"] is False
