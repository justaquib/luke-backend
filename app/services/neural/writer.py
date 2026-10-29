"""Writing a sentence from the word model, one token at a time.

The writer keeps the word model's generation rules in one place: a mark is
never written, a naming word is not said twice, and the sentence stops at the
end mark or at the context limit. A byte-level tokenizer writes pieces rather
than words, so its rules are token-aware and its text is decoded and trimmed
to a sentence end instead of being joined word by word.
"""

from __future__ import annotations

import torch

from app.services.neural.tokenizer.base import BPE_KIND, Tokenizer
from app.services.neural.tokens import decode_text, is_a_word, tokenize

# The punctuation a word model is never asked to predict as a word.
PUNCTUATION = frozenset({".", "!", "?", ","})

# Small words may be said again; a naming word said again is a loop.
_REPEATABLE_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "it",
        "its",
        "that",
        "this",
        "these",
        "those",
        "for",
        "with",
        "as",
        "at",
        "from",
        "by",
        "has",
        "have",
        "had",
        "there",
    }
)


class WordWriter:
    """Write a sentence from a prefix, one token at a time."""

    tokenizer: Tokenizer | None
    model: torch.nn.Module | None
    trained: bool
    config: dict
    vocabulary: list[str]
    index: dict[str, int]
    word_counts: dict[str, int]

    @property
    def _pieces(self) -> bool:
        return self.tokenizer is not None and self.tokenizer.kind == BPE_KIND

    def predict_next(
        self,
        prefix: str,
        limit: int = 5,
        allowed: set[str] | None = None,
    ) -> list[dict[str, int | float | str | bool | None]]:
        """The tokens the model expects next, with the chance it gives each."""
        if not self.trained or self.model is None:
            return []
        tokens = self._prefix_ids(prefix)
        allowed_ids = self._allowed_ids(allowed)
        if allowed is not None and not allowed_ids:
            return []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            window = tokens[-int(self.config["context"]) :]
            logits = self.model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]
        for token_id in self._banned(tokens, allowed_ids):
            logits[token_id] = -float("inf")
        return self.rows_from_logits(torch.softmax(logits, dim=-1), limit)

    def rows_from_logits(self, probabilities: torch.Tensor, limit: int = 5) -> list[dict]:
        """The tokens the model expects next, ranked, with the chance it gives each."""
        values, positions = torch.topk(probabilities, min(limit * 3, len(self.vocabulary)))
        rows = []
        for value, position in zip(values.tolist(), positions.tolist(), strict=False):
            token_id = int(position)
            if self.tokenizer is not None and self.tokenizer.is_special(token_id):
                continue
            piece = self.token_text(token_id)
            if not self._pieces and (not is_a_word(piece) or piece in PUNCTUATION):
                continue
            rows.append(
                {
                    "word": piece,
                    "piece": self._pieces,
                    "count": int(self.word_counts.get(piece, 0)),
                    "probability": round(float(value), 5),
                    "source": "neural",
                }
            )
            if len(rows) == limit:
                break
        return rows

    def token_text(self, token_id: int) -> str:
        """The text of an id, in this tokenizer's own vocabulary."""
        if self.tokenizer is not None:
            return self.tokenizer.token_text(token_id)
        return self.vocabulary[token_id]

    def _prefix_ids(self, prefix: str) -> list[int]:
        if self.tokenizer is not None:
            start = self.tokenizer.start_id
            ids = self.tokenizer.encode(prefix) if prefix else []
            return [start, *ids] if ids else [start]
        words = [token for token in tokenize(prefix) if is_a_word(token)]
        ids = [self.index[token] for token in words if token in self.index]
        start = self.index.get("<start>", 0)
        return [start, *ids] if ids else [start]

    def _allowed_ids(self, allowed: set[str] | None) -> set[int] | None:
        if allowed is None or self._pieces:
            return None
        return {self.index[word] for word in allowed if word in self.index}

    def _banned(
        self,
        tokens: list[int],
        allowed: set[int] | None,
        allow_finish: bool = False,
    ) -> set[int]:
        """The ids a written token may not be, so a mark or a loop is never written."""
        banned: set[int] = set()
        end_id = self.tokenizer.end_id if self.tokenizer is not None else None
        if self.tokenizer is not None:
            banned |= set(self.tokenizer.special_ids.values())
        if not self._pieces:
            banned |= {self.index[token] for token in PUNCTUATION if token in self.index}
        if allow_finish and end_id is not None:
            banned.discard(end_id)
        if allowed is not None:
            banned |= set(range(len(self.vocabulary))) - allowed
        if tokens:
            banned.add(tokens[-1])
        return banned

    def generate_sentence(
        self,
        seed: str | None = None,
        max_words: int = 16,
        use_taught: bool = True,
        allowed: set[str] | None = None,
    ) -> str | None:
        """Write a sentence one token at a time, without repeating a naming word."""
        if not self.trained or self.model is None:
            return None
        tokens = self._prefix_ids(seed or "")
        if seed and len(tokens) == 1:
            return None
        allowed_ids = self._allowed_ids(allowed)
        written: list[int] = []
        used = {self.vocabulary[token] for token in tokens[1:]}
        visited: set[tuple[int, ...]] = set()
        for _ in range(max_words):
            token_id = self._next(tokens, used, visited, allowed_ids)
            if token_id is None:
                break
            tokens.append(token_id)
            written.append(token_id)
            used.add(self.vocabulary[token_id])
        if not written:
            return None
        text = self._read(written)
        return text[:1].upper() + text[1:] if text else None

    def _read(self, written: list[int]) -> str:
        if self._pieces:
            text = self.tokenizer.decode(written) if self.tokenizer is not None else ""
            return _finished(text)
        return decode_text([self.vocabulary[token] for token in written])

    def _next(
        self,
        tokens: list[int],
        used: set[str],
        visited: set[tuple[int, ...]],
        allowed_ids: set[int] | None,
    ) -> int | None:
        """The next token of a sentence, or nothing when it has to stop."""
        context = tuple(tokens[-int(self.config["context"]) :])
        if context in visited:
            return None
        visited.add(context)
        token_id = self._choose(context, tokens, used, allowed_ids)
        if self.tokenizer is not None and token_id == self.tokenizer.end_id:
            return None
        if self._pieces:
            return token_id
        piece = self.vocabulary[token_id]
        if not is_a_word(piece) or piece in PUNCTUATION:
            return None
        return token_id

    def _choose(
        self,
        context: tuple[int, ...],
        tokens: list[int],
        used: set[str],
        allowed_ids: set[int] | None,
    ) -> int:
        """The token the model picks, never a mark, a punctuation or a loop."""
        assert self.model is not None
        device = next(self.model.parameters()).device
        inputs = torch.tensor([list(context)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.model(inputs)[0, -1]
        for token_id in self._banned(tokens, allowed_ids, allow_finish=True):
            logits[token_id] = -float("inf")
        if not self._pieces:
            for token_id, word in enumerate(self.vocabulary):
                if word in used and word not in _REPEATABLE_WORDS:
                    logits[token_id] = -float("inf")
        return int(torch.argmax(logits).item())


def _finished(text: str) -> str:
    """The written text, trimmed to its last sentence end."""
    clean = " ".join((text or "").replace("\ufffd", "").split())
    if not clean:
        return ""
    last = max(clean.rfind("."), clean.rfind("!"), clean.rfind("?"))
    if last >= 0:
        clean = clean[: last + 1]
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean
