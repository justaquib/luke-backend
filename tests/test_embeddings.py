"""Embedding every text without asking the model for one enormous batch."""

from app.services import embeddings


class _Recorder:
    """A stand-in embedding function that remembers what it was asked for."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[float(len(text))] for text in texts]


def test_every_text_is_embedded_in_order_across_chunks(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(embeddings, "embedding_function", lambda: recorder)
    texts = [f"text {index}" for index in range(embeddings.CHUNK + 5)]

    vectors = embeddings.embed_texts(texts)

    assert [len(batch) for batch in recorder.batches] == [embeddings.CHUNK, 5]
    assert [text for batch in recorder.batches for text in batch] == texts
    assert vectors == [[float(len(text))] for text in texts]


def test_no_text_asks_the_model_for_nothing(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(embeddings, "embedding_function", lambda: recorder)

    assert embeddings.embed_texts([]) == []
    assert recorder.batches == []
