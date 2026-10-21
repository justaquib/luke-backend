"""The words the transformer reads, and the marks that separate the parts.

A training example has three parts: the request, the material it may use, and
the answer. The marks tell the model where each part starts, so the loss can
be taken over the answer alone and the model learns to answer a request
instead of continuing a text.
"""

from __future__ import annotations

import re

# The word, the punctuation around it, and the line break between two lines.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?|[.!?]|,|\n")

# The punctuation that ends a sentence.
END_TOKENS = {".", "?", "!"}
# The marks the model uses to know where a sentence starts and ends.
START = "<start>"
FINISH = "<end>"
UNKNOWN = "<unk>"
# The mark used to line the sentences up in a batch. It is never a word.
PAD = "<pad>"
# The mark that opens a request.
INSTRUCTION = "<instr>"
# The mark that opens the material the answer is written from.
CONTEXT = "<ctx>"
# The mark that opens the answer.
RESPONSE = "<resp>"
# The mark that stands for a line break inside a poem.
LINE_BREAK = "<br>"

SPECIAL_TOKENS = (
    START,
    FINISH,
    UNKNOWN,
    PAD,
    INSTRUCTION,
    CONTEXT,
    RESPONSE,
    LINE_BREAK,
)


def tokenize(text: str) -> list[str]:
    """The words of a text, in lower case, with their punctuation.

    A line break becomes a mark of its own, so a poem keeps its shape.
    """
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text or ""):
        word = match.group(0)
        tokens.append(LINE_BREAK if word == "\n" else word.lower().replace("’", "'"))
    return tokens


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
