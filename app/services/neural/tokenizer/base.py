"""The one interface every scratch tokenizer keeps.

A tokenizer belongs to the weights it trained with. Every scratch model reads
its ids through this interface, and nothing outside it may encode, decode or
substitute an id space. A tokenizer that cannot be read together with the
weights it is asked to serve is refused with a clear reason, never remapped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

TOKENIZER_FILE = "tokenizer.json"
META_FILE = "tokenizer-meta.json"

WORD_KIND = "word"
BPE_KIND = "bpe"

# The marks the scratch models use, in the order their ids are fixed in.
SPECIAL_NAMES = ("<pad>", "<start>", "<end>", "<unk>", "<br>")


class TokenizerError(Exception):
    """A tokenizer cannot serve the weights it was asked to serve."""


class MissingArtifactError(TokenizerError):
    """The tokenizer artifact a checkpoint names is not on disk."""


class CorruptArtifactError(TokenizerError):
    """The tokenizer artifact does not match its recorded hash or shape."""


class VocabularyMismatchError(TokenizerError):
    """The weights and the tokenizer do not share one vocabulary size."""


@runtime_checkable
class Tokenizer(Protocol):
    """What a scratch tokenizer must be able to say and do."""

    kind: str
    vocab_size: int

    @property
    def special_ids(self) -> dict[str, int]:
        """The id of every special mark, by name."""

    @property
    def pad_id(self) -> int:
        """The mark that lines examples up in a batch."""

    @property
    def start_id(self) -> int:
        """The mark that opens a sentence."""

    @property
    def end_id(self) -> int:
        """The mark that closes a sentence."""

    @property
    def unknown_id(self) -> int:
        """The mark for a piece the tokenizer cannot write."""

    def encode(self, text: str) -> list[int]:
        """The ids of a text."""

    def decode(self, ids: Sequence[int]) -> str:
        """The text of the ids, without structural marks."""

    def token_text(self, token_id: int) -> str:
        """The piece an id stands for, as it is written in the vocabulary."""

    def is_special(self, token_id: int) -> bool:
        """Whether an id is a structural mark rather than a word or piece."""

    def vocabulary(self) -> list[str]:
        """Every piece, by id."""

    def digest(self) -> str:
        """A hash that changes when the vocabulary or merges change."""

    def save(self, folder: Path) -> None:
        """Write the tokenizer into a folder, in one move."""

    def describe(self) -> dict[str, object]:
        """The recorded shape of the tokenizer, for the checkpoint and the page."""
