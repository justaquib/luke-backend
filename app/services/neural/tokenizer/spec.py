"""Which scratch tokenizer a new training uses, and how it is built.

The word tokenizer is the baseline until a measured BPE artifact beats it on
held-out bits per character at an acceptable parameter cost. The constant is
changed only with that measurement written down; an old checkpoint keeps the
tokenizer it was trained with either way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from app.services.neural.tokenizer.base import (
    BPE_KIND,
    WORD_KIND,
    Tokenizer,
    VocabularyMismatchError,
)
from app.services.neural.tokenizer.bpe import BpeTokenizer
from app.services.neural.tokenizer.word import WordTokenizer

DEFAULT_KIND = WORD_KIND

# Measured 2026-10-27 (scripts/measure_tokenizer.py, tenant 0, seed 7, same
# split and 600 steps for every row). Held-out bits per character: word 1.3298
# at 1,683 tokens and 620,416 parameters; BPE 1.4807 at 3,779 tokens and
# 888,704 parameters. The 136,477-character corpus cannot fill 8k or 16k (all
# of 4k/8k/16k collapsed to 3,779 pieces), and BPE even costs more tokens per
# character (0.2505 against 0.2392). The word tokenizer stays the default until
# a measurement reverses that; BPE is an explicit opt-in.
DEFAULT_BPE_VOCAB_SIZE = 4096

# Vocabularies smaller than this cannot hold the special marks and a useful
# number of pieces.
MIN_VOCAB_SIZE = 256


def default_kind() -> str:
    return DEFAULT_KIND


def build(
    kind: str,
    texts: Iterable[str],
    vocab_size: int | None = None,
    progress: Callable[[int, int, float | None], None] | None = None,
) -> Tokenizer:
    """Build the tokenizer a new checkpoint is trained with."""
    if kind == WORD_KIND:
        return WordTokenizer.build(texts)
    if kind == BPE_KIND:
        size = int(vocab_size or DEFAULT_BPE_VOCAB_SIZE)
        if size < MIN_VOCAB_SIZE:
            raise VocabularyMismatchError(f"a BPE vocabulary of {size} is below {MIN_VOCAB_SIZE}")
        return BpeTokenizer.train(texts, size, progress)
    raise VocabularyMismatchError(f"unknown tokenizer kind '{kind}'")
