"""A tiny word-level trigram model trained from local example sentences.

This deliberately has no pretrained parameters: its next-word probabilities are
counts learned from the sentences passed to its constructor.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable


_TOKEN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?|[.!?]")
_START = "<s>"
_END = "</s>"
_QUESTION_WORDS = {"what", "who", "whom", "whose", "when", "where", "why", "how", "which", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should"}
# Small words may show up again in a sentence; a naming word showing up again
# is the sign of a loop.
_REPEATABLE_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "in", "on", "of", "to", "and", "or", "it", "its", "that", "this", "these",
    "those", "for", "with", "as", "at", "from", "by", "has", "have", "had", "there",
}


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in _TOKEN.finditer(text)]


class TinyLanguageModel:
    """An in-memory trigram next-token model with backoff to shorter contexts.

    The model looks at the last three words to choose the next one. When it
    has never seen that pair of words together before, it falls back to the
    last two words, and then to the last one, so a young model still writes
    something instead of nothing.
    """

    order = 3

    def __init__(self, sentences: Iterable[str]):
        # The trigram counts: the last three words before the next one.
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        # The shorter counts used when the trigram is unknown.
        self.backoff: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()
        self.token_count = 0
        self.sentence_count = 0
        # The sentences the model was taught, kept as they were written.
        self.sentences: Counter[str] = Counter()
        for sentence in sentences:
            self.learn_sentence(sentence)

    def learn_sentence(self, sentence: str) -> None:
        tokens = tokenize(sentence)
        tokens = [token for token in tokens if token not in {".", "!", "?"}]
        if not tokens:
            return
        stream = [_START, _START, _START, *tokens, _END]
        for index in range(3, len(stream)):
            context = tuple(stream[index - 3:index])
            self.transitions[context][stream[index]] += 1
        for index in range(2, len(stream)):
            context = tuple(stream[index - 2:index])
            self.backoff[context][stream[index]] += 1
        self.vocabulary.update(tokens)
        self.token_count += len(tokens)
        self.sentence_count += 1
        self.sentences[sentence.strip()] += 1

    @property
    def transition_count(self) -> int:
        return sum(sum(next_words.values()) for next_words in self.transitions.values())

    def _context(self, tokens: list[str]) -> tuple[str, ...]:
        """The last three words, padded at the start of a sentence."""
        padded = [_START] * (self.order - 1) + list(tokens)
        return tuple(padded[-self.order:])

    def _counts(self, tokens: list[str]) -> Counter[str]:
        """The counts for the longest context the model has seen."""
        context = self._context(tokens)
        counts = self.transitions.get(context)
        if counts:
            return counts
        for size in (2, 1):
            if len(tokens) >= size:
                shorter = tuple(tokens[-size:])
                counts = self.backoff.get(shorter)
                if counts:
                    return counts
        return self.backoff.get((_START, _START), Counter())

    def predict_next(self, prefix: str, limit: int = 5) -> list[dict[str, int | str]]:
        tokens = [token for token in tokenize(prefix) if token not in {".", "!", "?"}]
        counts = self._counts(tokens)
        return [
            {"word": word, "count": count}
            for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
            if word != _END
        ]

    def _has_word(self, word: str) -> bool:
        """Whether the model knows the word, ending or not.

        "bee" is known when it has seen "bee's".
        """
        return any(
            token == word or token.startswith(word + "'")
            for token in self.vocabulary
        )

    def taught_sentence(self, seed: str) -> str | None:
        """A sentence the model was taught that uses the given word.

        A sentence that was taught is a proper one, so it is a better answer
        than a sentence the model stitches together on its own.
        """
        wanted = [token for token in tokenize(seed) if token not in {".", "!", "?"}]
        if not wanted:
            return None
        best: str | None = None
        best_count = 0
        for sentence, count in self.sentences.items():
            tokens = tokenize(sentence)
            if not all(
                any(token == word or token.startswith(word + "'") for token in tokens)
                for word in wanted
            ):
                continue
            if count > best_count:
                best = sentence
                best_count = count
        return best

    def score_words(self, words: list[str]) -> float:
        """How well the model knows a word order.

        The words it has seen together often score higher, so the model can
        choose the wording it knows best.
        """
        if len(words) < 2:
            return 0.0
        total = 0.0
        for index in range(len(words) - 1):
            pair = (words[index], words[index + 1])
            counts = self.backoff.get(pair) or self.transitions.get(pair)
            total += math.log1p(sum(counts.values())) if counts else 0.0
        return total

    def generate_sentence(self, seed: str | None = None, max_words: int = 16, use_taught: bool = True) -> str | None:
        if seed:
            first = [token for token in tokenize(seed) if token not in {".", "!", "?"}]
            if not first:
                return None
            word = first[-1]
            if not self._has_word(word):
                return None
            if use_taught:
                taught = self.taught_sentence(word)
                if taught:
                    return taught
            # Start with the requested word and use the trigram the model
            # actually observed after that word, the most frequent one.
            contexts = [(context, counter) for context, counter in self.transitions.items() if context[-1] == word]
            if not contexts:
                return None
            context = max(contexts, key=lambda item: sum(item[1].values()))[0]
            output = [word]
        else:
            context = (_START, _START, _START)
            first_counts = self.transitions.get(context)
            if not first_counts:
                return None
            # A sentence that tells something is better than a question, so a
            # question word is only used when there is nothing else.
            starts = [word for word in first_counts if word != _END]
            telling = [word for word in starts if word not in _QUESTION_WORDS]
            word = max(telling or starts, key=lambda item: (first_counts[item], item))
            if word == _END:
                return None
            output = [word]
            context = (_START, _START, word)

        used = set(output)
        visited: set[tuple[str, ...]] = set()
        for _ in range(max_words - len(output)):
            # Walking into the same words again means the model is looping, so
            # the sentence ends there.
            if context in visited:
                break
            visited.add(context)
            options = self.transitions.get(context)
            if not options:
                # The trigram is unknown, so the shorter context is used.
                shorter = tuple(context[-2:])
                options = self.backoff.get(shorter) or self.backoff.get((context[-1],))
                if not options:
                    break
            best_word, best_count = max(options.items(), key=lambda item: (item[1], item[0]))
            # The sentence ends when nothing else can follow it, or when the end
            # is clearly the likeliest next word.
            others = [item for item in options.items() if item[0] != _END]
            if not others or options.get(_END, 0) > best_count:
                break
            # A naming word that is already in the sentence is the sign of a
            # loop, so it is not used again. Small words may be used again.
            candidates = [
                (word, count)
                for word, count in options.items()
                if word != _END and (word not in used or word in _REPEATABLE_WORDS)
            ]
            if not candidates:
                break
            next_word = max(candidates, key=lambda item: (item[1], item[0]))[0]
            output.append(next_word)
            used.add(next_word)
            context = (context[-2], context[-1], next_word)

        if not output:
            return None
        sentence = " ".join(output)
        sentence = re.sub(r"\s+([.!?,])", r"\1", sentence)
        return sentence[:1].upper() + sentence[1:] + "."
