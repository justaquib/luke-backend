"""The scratch tokenizers: the legacy word adapter and byte-level BPE.

A tokenizer belongs to the weights it trained with. The foundation model and
the teachers keep their own tokenizers and are never read or written here.
"""

from __future__ import annotations

from app.services.neural.tokenizer import artifacts, corpus, spec
from app.services.neural.tokenizer.base import (
    BPE_KIND,
    SPECIAL_NAMES,
    WORD_KIND,
    CorruptArtifactError,
    MissingArtifactError,
    Tokenizer,
    TokenizerError,
    VocabularyMismatchError,
)
from app.services.neural.tokenizer.bpe import BpeTokenizer
from app.services.neural.tokenizer.word import WordTokenizer

__all__ = [
    "BPE_KIND",
    "SPECIAL_NAMES",
    "WORD_KIND",
    "BpeTokenizer",
    "CorruptArtifactError",
    "MissingArtifactError",
    "Tokenizer",
    "TokenizerError",
    "VocabularyMismatchError",
    "WordTokenizer",
    "artifacts",
    "corpus",
    "spec",
]
