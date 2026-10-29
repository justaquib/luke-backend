"""The embedding of a text, in the space the vector store keeps its vectors in.

The heads of the brain read the meaning of a text from the same embedding the
collections are stored with, so a head and a search talk about the same words.
A question asked again is answered from the vector it produced the first time;
the cache only holds short texts, so an upload cannot grow it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Any

# The model holds every text of a request at once, so a whole tenant's
# questions would ask it for one enormous batch. Texts are embedded
# independently, and chunks keep the memory the run needs bounded.
CHUNK = 256

# How many vectors of short, repeated texts are kept, and the longest text
# that is kept at all. A question is short; a passage is not, and its vector
# is only needed once while it is written.
CACHE_ENTRIES = 512
CACHE_MAX_CHARS = 256

_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def embedding_function() -> Any:
    """The embedding function the collections are written with.

    The model is Chroma's own ONNX MiniLM, unchanged, and it produces the same
    numbers here as through the default provider list (measured on 2026-10-28:
    the largest difference between the two over a batch of passages was 0.0).
    Only the execution provider is chosen: asking for the CPU stops ONNX from
    handing every batch to CoreML, which cost twice as much on the same work.
    """
    from chromadb.utils import embedding_functions

    return embedding_functions.ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


class CachedEmbeddingFunction:
    """Chroma's default embedding, answering repeated texts from the cache."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_texts(list(input))

    def name(self) -> str:
        return "default"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> CachedEmbeddingFunction:
        return cached_embedding_function()


@lru_cache(maxsize=1)
def cached_embedding_function() -> CachedEmbeddingFunction:
    """The embedding the collections are opened with, so a query is cached."""
    return CachedEmbeddingFunction()


def cached(text: str) -> list[float] | None:
    """The vector of a text that was embedded before, or None."""
    with _CACHE_LOCK:
        vector = _CACHE.get(text)
        if vector is not None:
            _CACHE.move_to_end(text)
        return None if vector is None else list(vector)


def forget() -> None:
    """Drop every remembered vector, so the next call embeds again."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _remember(text: str, vector: list[float]) -> None:
    if len(text) > CACHE_MAX_CHARS:
        return
    with _CACHE_LOCK:
        _CACHE[text] = vector
        _CACHE.move_to_end(text)
        while len(_CACHE) > CACHE_ENTRIES:
            _CACHE.popitem(last=False)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """The embedding of every text, one list of numbers for each."""
    if not texts:
        return []
    vectors: list[list[float] | None] = [cached(text) for text in texts]
    pending: dict[str, list[int]] = {}
    for index, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
        if vector is None:
            pending.setdefault(text, []).append(index)
    embed = embedding_function()
    for start in range(0, len(pending), CHUNK):
        batch = list(pending)[start : start + CHUNK]
        computed = embed(batch)
        for text, vector in zip(batch, computed, strict=True):
            value = [float(number) for number in vector]
            _remember(text, value)
            for position in pending[text]:
                vectors[position] = list(value)
    return [vector if vector is not None else [] for vector in vectors]
