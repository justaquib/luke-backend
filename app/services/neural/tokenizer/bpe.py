"""A byte-level BPE tokenizer for scratch models that were trained with it.

The text is normalised to NFC and read as bytes, so any script round-trips and
nothing is ever an unknown character. A line break is split out and written
back as the ``<br>`` mark, so a poem keeps its shape. The special marks are
added first, which fixes their ids, and a raw encode never produces one.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from app.services.neural.tokenizer.base import (
    BPE_KIND,
    SPECIAL_NAMES,
    TOKENIZER_FILE,
    CorruptArtifactError,
    VocabularyMismatchError,
)

# The mark that stands for a line break, so a poem keeps its shape.
LINE_BREAK = SPECIAL_NAMES[4]

Progress = Callable[[int, int, float | None], None]


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _literal_marks(backend: object) -> object:
    """Keep the marks as ids a model may write, not as strings text may match.

    The trainer registers each special mark as a literal string, which would
    turn the text ``<pad>`` into a padding id. Dropping that table keeps the
    fixed ids in the vocabulary and makes a raw encode read the mark as plain
    pieces, so text that looks like a mark stays text.
    """
    from tokenizers import Tokenizer

    payload = json.loads(backend.to_str())
    if not payload.get("added_tokens"):
        return backend
    payload["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(payload))


class BpeTokenizer:
    """Byte-level BPE, trained on a corpus and read back from its artifact."""

    kind = BPE_KIND

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self.vocab_size = int(backend.get_vocab_size())
        self._special_ids = self._read_specials()
        self._special_set = frozenset(self._special_ids.values())
        self._line_break_id = self._special_ids[LINE_BREAK]

    def _read_specials(self) -> dict[str, int]:
        ids: dict[str, int] = {}
        for name in SPECIAL_NAMES:
            token_id = self._backend.token_to_id(name)
            if token_id is None:
                raise CorruptArtifactError(f"the tokenizer artifact has no {name} mark")
            ids[name] = int(token_id)
        return ids

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        progress: Progress | None = None,
    ) -> BpeTokenizer:
        """Learn merges from the corpus, with every special mark already fixed."""
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        lines = [line for text in texts for line in _normalise(text).splitlines() if line.strip()]
        if not lines:
            raise VocabularyMismatchError("the tokenizer corpus is empty")
        if vocab_size <= len(SPECIAL_NAMES):
            raise VocabularyMismatchError(f"a vocabulary of {vocab_size} holds no room for pieces")
        if progress is not None:
            progress(0, len(lines), None)
        backend = Tokenizer(models.BPE(unk_token=None))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            special_tokens=list(SPECIAL_NAMES),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(lines, trainer=trainer, length=len(lines))
        backend = _literal_marks(backend)
        if progress is not None:
            progress(len(lines), len(lines), None)
        return cls(backend)

    @classmethod
    def load(cls, folder: Path) -> BpeTokenizer:
        from tokenizers import Tokenizer

        path = folder / TOKENIZER_FILE
        try:
            backend = Tokenizer.from_str(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CorruptArtifactError(
                f"the tokenizer artifact could not be read: {error}"
            ) from error
        return cls(backend)

    @property
    def special_ids(self) -> dict[str, int]:
        return dict(self._special_ids)

    @property
    def pad_id(self) -> int:
        return self._special_ids[SPECIAL_NAMES[0]]

    @property
    def start_id(self) -> int:
        return self._special_ids[SPECIAL_NAMES[1]]

    @property
    def end_id(self) -> int:
        return self._special_ids[SPECIAL_NAMES[2]]

    @property
    def unknown_id(self) -> int:
        return self._special_ids[SPECIAL_NAMES[3]]

    def encode(self, text: str) -> list[int]:
        """The ids of a text, with a fixed id for every line break."""
        normalised = _normalise(text)
        if not normalised:
            return []
        ids: list[int] = []
        for position, line in enumerate(normalised.split("\n")):
            if position:
                ids.append(self._line_break_id)
            if line:
                ids.extend(self._backend.encode(line, add_special_tokens=False).ids)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """The text of the ids, with the line breaks put back and marks left out."""
        lines: list[str] = []
        current: list[int] = []
        for token_id in ids:
            value = int(token_id)
            if value == self._line_break_id:
                lines.append(self._decode_line(current))
                current = []
            else:
                current.append(value)
        lines.append(self._decode_line(current))
        return "\n".join(lines)

    def _decode_line(self, ids: Sequence[int]) -> str:
        words = [int(token_id) for token_id in ids if int(token_id) not in self._special_set]
        return self._backend.decode(words, skip_special_tokens=False)

    def token_text(self, token_id: int) -> str:
        piece = self._backend.id_to_token(int(token_id))
        if piece is None:
            raise IndexError(f"id {token_id} is outside the BPE vocabulary")
        return str(piece)

    def is_special(self, token_id: int) -> bool:
        return int(token_id) in self._special_set

    def vocabulary(self) -> list[str]:
        return [self.token_text(token_id) for token_id in range(self.vocab_size)]

    def digest(self) -> str:
        return hashlib.sha256(self._backend.to_str().encode("utf-8")).hexdigest()

    def save(self, folder: Path) -> None:
        """Write the artifact in one move."""
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f"{TOKENIZER_FILE}.tmp"
        temporary.write_text(self._backend.to_str(), encoding="utf-8")
        os.replace(temporary, folder / TOKENIZER_FILE)

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "special_ids": self.special_ids,
            "normalization": "NFC",
            "pre_tokenizer": "ByteLevel",
            "decoder": "ByteLevel",
        }
