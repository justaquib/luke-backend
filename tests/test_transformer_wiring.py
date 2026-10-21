"""Tests for the transformer that writes in the trainer's words and proposes answers."""

from app.services import brain, language_logic, transformer
from app.services import query as query_service
from app.services.language import runtime
from app.services.transformer.settings import TrainingSettings


def test_the_corpus_holds_what_the_trainer_taught(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: What is a kite?\nA: A kite is a light frame with a tail.",
        "nursery",
        "Toys",
    )

    corpus = language_logic.training_corpus(None)

    assert any("kite" in sentence.lower() for sentence in corpus)
    assert len(corpus) > 5


def test_a_question_the_model_was_never_taught_gets_a_proposal(monkeypatch):
    class FakeWriter:
        def generate(self, prompt, settings=None):
            return "A kite is a light frame with a tail."

    monkeypatch.setattr(brain, "load_writer", lambda tenant_id: FakeWriter())

    proposal = query_service.propose_answer(None, "what is a kite?")

    assert proposal == "A kite is a light frame with a tail."


def test_a_proposal_that_says_nothing_is_not_offered(monkeypatch):
    class QuietWriter:
        def generate(self, prompt, settings=None):
            return "No."

    monkeypatch.setattr(brain, "load_writer", lambda tenant_id: QuietWriter())

    assert query_service.propose_answer(None, "what is a kite?") is None


def test_without_a_model_there_is_no_proposal(monkeypatch):
    monkeypatch.setattr(brain, "load_writer", lambda tenant_id: None)

    assert query_service.propose_answer(None, "what is a kite?") is None


def test_the_writer_is_kept_for_the_tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    brain._WRITER_CACHE.clear()
    writer = transformer.TransformerWriter()
    writer.config = {**writer.config, "d_model": 32, "layers": 1, "heads": 2, "context": 32}
    writer.fit(
        [
            "The cow is an animal.",
            "A cow gives milk.",
            "The cow is a farm animal that gives milk.",
            "Milk is the white liquid that comes from a cow.",
            "The night sky is black.",
            "The moon shines at night.",
        ],
        settings=TrainingSettings(steps=20),
    )
    writer.save(brain.writer_path(None))
    brain._WRITER_CACHE.clear()

    loaded = brain.load_writer(None)

    assert loaded is not None
    assert loaded.trained is True
    brain._WRITER_CACHE.clear()
