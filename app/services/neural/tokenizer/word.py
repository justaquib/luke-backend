"""The legacy word tokenizer, wrapped so every old checkpoint keeps its ids.

This is the tokenizer the word model was first trained with: whole words,
their punctuation, and the special marks. It stays the adapter for every
checkpoint that was written before the tokenizer was recorded, so an old
model answers with exactly the ids it learned instead of a remapped guess.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from app.services.neural.tokenizer.base import (
    SPECIAL_NAMES,
    TOKENIZER_FILE,
    WORD_KIND,
    VocabularyMismatchError,
)
from app.services.neural.tokens import (
    PAD,
    SPECIAL_TOKENS,
    decode_text,
    tokenize,
)
from app.services.vocabulary import CANONICAL_OWNER, Vocabulary


class WordTokenizer:
    """Whole words and their marks, with the ids the word model already knows."""

    kind = WORD_KIND

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self._vocabulary = Vocabulary(CANONICAL_OWNER, vocabulary)
        self._index = {token: position for position, token in enumerate(self._vocabulary.tokens)}
        self.vocab_size = len(self._vocabulary)
        missing = [name for name in SPECIAL_NAMES if name not in self._index]
        if missing:
            raise VocabularyMismatchError(f"the word vocabulary is missing {', '.join(missing)}")

    @classmethod
    def build(cls, texts: Iterable[str]) -> WordTokenizer:
        """The vocabulary of a corpus, in the order the word model sorts it."""
        words: set[str] = set(SPECIAL_TOKENS)
        for text in texts:
            words.update(tokenize(text))
        return cls(sorted(words))

    @classmethod
    def load(cls, folder: Path) -> WordTokenizer:
        """Read a saved word vocabulary."""
        stored = json.loads((folder / TOKENIZER_FILE).read_text(encoding="utf-8"))
        return cls([str(token) for token in stored.get("vocabulary", [])])

    @property
    def special_ids(self) -> dict[str, int]:
        return {name: self._index[name] for name in SPECIAL_NAMES}

    @property
    def pad_id(self) -> int:
        return self._index[PAD]

    @property
    def start_id(self) -> int:
        return self._index[SPECIAL_NAMES[1]]

    @property
    def end_id(self) -> int:
        return self._index[SPECIAL_NAMES[2]]

    @property
    def unknown_id(self) -> int:
        return self._index[SPECIAL_NAMES[3]]

    def encode(self, text: str) -> list[int]:
        """The ids of a text, with the unknown mark for a word never seen."""
        return self._vocabulary.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        """The text of the ids, with the structural marks left out."""
        words: list[str] = []
        for token_id in ids:
            piece = self.token_text(int(token_id))
            if piece in SPECIAL_TOKENS and piece != SPECIAL_NAMES[4]:
                continue
            words.append(piece)
        return decode_text(words)

    def token_text(self, token_id: int) -> str:
        piece = self._vocabulary.token_of(int(token_id))
        if piece is None:
            raise IndexError(f"id {token_id} is outside the word vocabulary")
        return piece

    def is_special(self, token_id: int) -> bool:
        return self.token_text(int(token_id)) in SPECIAL_TOKENS

    def vocabulary(self) -> list[str]:
        return list(self._vocabulary.tokens)

    def digest(self) -> str:
        joined = "\0".join(self._vocabulary.tokens)
        return hashlib.sha256(f"{WORD_KIND}\0{joined}".encode()).hexdigest()

    def save(self, folder: Path) -> None:
        """Write the vocabulary in one move."""
        folder.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"kind": WORD_KIND, "vocabulary": list(self._vocabulary.tokens)},
            ensure_ascii=False,
            indent=2,
        )
        temporary = folder / f"{TOKENIZER_FILE}.tmp"
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, folder / TOKENIZER_FILE)

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "special_ids": self.special_ids,
        }
