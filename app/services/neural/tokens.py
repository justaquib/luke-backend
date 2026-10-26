"""The words the neural models read, and the marks that hold a text's shape.

A sentence is scored from its first word to its last, so the marks only say
where it starts, where it ends, what an unknown word is, and where a line
break falls.
"""

from __future__ import annotations

import re

# The word, the punctuation around it, a number, and the line break between two lines.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?|\d+(?:\.\d+)?|[.!?]|,|\n")

# The punctuation that ends a sentence.
END_TOKENS = {".", "?", "!"}
# The marks the model uses to know where a sentence starts and ends.
START = "<start>"
FINISH = "<end>"
UNKNOWN = "<unk>"
# The mark used to line the sentences up in a batch. It is never a word.
PAD = "<pad>"
# The mark that stands for a line break inside a poem.
LINE_BREAK = "<br>"

SPECIAL_TOKENS = (
    START,
    FINISH,
    UNKNOWN,
    PAD,
    LINE_BREAK,
)

# The marks that are structure, not words, so a prediction must never be one.
STRUCTURE_TOKENS = frozenset(SPECIAL_TOKENS)


def tokenize(text: str) -> list[str]:
    """The words of a text, in lower case, with their punctuation.

    A line break becomes a mark of its own, so a poem keeps its shape.
    """
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text or ""):
        word = match.group(0)
        tokens.append(LINE_BREAK if word == "\n" else word.lower().replace("’", "'"))
    return tokens


def is_a_word(token: str) -> bool:
    """Whether a token can be written as a word of an answer."""
    return bool(token) and token not in STRUCTURE_TOKENS


def decode_text(words: list[str]) -> str:
    """The words of a written answer, put back into a readable text."""
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if word == LINE_BREAK:
            lines.append(" ".join(current))
            current = []
            continue
        current.append(word)
    lines.append(" ".join(current))
    text = "\n".join(line for line in lines if line.strip())
    text = re.sub(r"\s+([.!?,])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text
