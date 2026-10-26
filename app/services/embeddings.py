"""The embedding of a text, in the space the vector store keeps its vectors in.

The heads of the brain read the meaning of a text from the same embedding the
collections are stored with, so a head and a search talk about the same words.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# The model holds every text of a request at once, so a whole tenant's
# questions would ask it for one enormous batch. Texts are embedded
# independently, and chunks keep the memory the run needs bounded.
CHUNK = 256


@lru_cache(maxsize=1)
def embedding_function() -> Any:
    """The embedding function the collections are written with."""
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """The embedding of every text, one list of numbers for each."""
    if not texts:
        return []
    embed = embedding_function()
    vectors = []
    for start in range(0, len(texts), CHUNK):
        vectors.extend(embed(list(texts[start : start + CHUNK])))
    return [[float(value) for value in vector] for vector in vectors]
