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


def test_a_text_asked_again_is_not_embedded_again(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(embeddings, "embedding_function", lambda: recorder)
    embeddings.forget()

    first = embeddings.embed_texts(["where is the ferry"])
    second = embeddings.embed_texts(["where is the ferry"])

    assert first == second
    assert recorder.batches == [["where is the ferry"]]


def test_a_long_text_is_not_kept(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(embeddings, "embedding_function", lambda: recorder)
    embeddings.forget()
    passage = "word " * 100

    embeddings.embed_texts([passage])
    embeddings.embed_texts([passage])

    assert recorder.batches == [[passage], [passage]]


def test_a_returned_vector_cannot_change_the_cached_one(monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(embeddings, "embedding_function", lambda: recorder)
    embeddings.forget()
    vector = embeddings.embed_texts(["the moon"])[0]

    vector.append(99.0)

    assert embeddings.embed_texts(["the moon"])[0] == [8.0]


def test_the_model_is_asked_for_the_cpu_and_nothing_else() -> None:
    embeddings.embedding_function.cache_clear()
    try:
        assert embeddings.embedding_function()._preferred_providers == ["CPUExecutionProvider"]
    finally:
        embeddings.embedding_function.cache_clear()
