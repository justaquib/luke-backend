"""The answer model writing a response to a request.

The model reads the request and the material it was given, writes an answer,
and reads its own answer before the answer is given away. When the answer is
not good enough it is written again with the critique in front of the model,
and the better of the two is kept.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch

from app.services.neural import critique as critique_service
from app.services.neural.critique import Answer, Critique
from app.services.neural.decoding import block_loops
from app.services.neural.settings import ReadingSettings, WritingSettings
from app.services.neural.tokens import (
    CONTEXT,
    FINISH,
    INSTRUCTION,
    LINE_BREAK,
    PAD,
    RESPONSE,
    START,
    UNKNOWN,
    tokenize,
)

# A written answer is at least this long before it may say it is finished.
MIN_ANSWER_WORDS = 4

# A content word that was already written is less likely to be written again.
REPEAT_PENALTY = 1.5

# Small words and the marks of a sentence may come back without penalty.
_REPEATABLE = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
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
        "for",
        "with",
        "as",
        "at",
        "from",
        "by",
        "has",
        "have",
        "had",
        ".",
        ",",
        "!",
        "?",
        LINE_BREAK,
    }
)


@dataclass(frozen=True)
class Request:
    """A request to answer, and how much of it the model may read."""

    instruction: str
    context: str = ""
    max_words: int = 64
    reading: ReadingSettings = field(default_factory=ReadingSettings)


def encode_instruction(writer: Any, request: Request) -> list[int]:
    """The words the model reads before it starts answering.

    The marks are the same ones the model was trained with. The request is
    never cut short: it is the part that says what is being asked for.
    """
    reading = request.reading
    tokens = [writer.index[START], writer.index[INSTRUCTION]]
    tokens += [
        writer.index.get(token, writer.index[UNKNOWN])
        for token in tokenize(request.instruction)[-reading.instruction_words :]
    ]
    if request.context:
        tokens += [writer.index[CONTEXT]]
        tokens += [
            writer.index.get(token, writer.index[UNKNOWN])
            for token in tokenize(request.context)[-reading.material_words :]
        ]
    tokens += [writer.index[RESPONSE]]
    return tokens


def _penalize_repeats(logits: torch.Tensor, written: list[int], writer: Any) -> None:
    """Make a content word that was already said less likely to be said again."""
    words = writer.vocabulary
    for token_id, count in Counter(written).items():
        word = words[token_id] if 0 <= token_id < len(words) else ""
        if word in _REPEATABLE:
            continue
        logits[token_id] -= REPEAT_PENALTY * count


def write(writer: Any, tokens: list[int], max_words: int) -> list[int]:
    """The words the model writes after reading what it has read.

    The answer ends when the model says it is done, not at the first full
    stop: a poem has a full stop in the middle of it. How sure the model was
    about every word is kept, so the answer can be read again before it is
    given away.
    """
    device = next(writer.model.parameters()).device
    written: list[int] = []
    chances: list[float] = []
    finish = writer.index.get(FINISH)
    # Inference mode keeps no autograd bookkeeping at all; every change to the
    # logits happens inside it, so nothing is carried out of the block.
    with torch.inference_mode():
        for _ in range(max_words):
            window = tokens[-int(writer.config["context"]) :]
            inputs = torch.tensor([window], dtype=torch.long, device=device)
            logits = writer.model(inputs)[0, -1]
            if writer.index.get(UNKNOWN) is not None:
                logits[writer.index[UNKNOWN]] = -float("inf")
            block_loops(logits, tokens)
            _penalize_repeats(logits, written, writer)
            if finish is not None and len(written) < MIN_ANSWER_WORDS:
                logits[finish] = -float("inf")
            token_id = int(torch.argmax(logits).item())
            chance = float(torch.softmax(logits, dim=-1)[token_id].clamp_min(1e-9).item())
            chances.append(chance)
            tokens.append(token_id)
            written.append(token_id)
            if token_id == finish:
                break
    skip = {writer.index.get(START), writer.index.get(FINISH), writer.index.get(PAD)}
    _remember(writer, written, chances)
    return [token for token in written if token not in skip]


def _remember(writer: Any, written: list[int], chances: list[float]) -> None:
    """Keep how sure the model was, so the critique can read the answer."""
    words = [writer.vocabulary[token] for token in written if token < len(writer.vocabulary)]
    writer.last_confidence = round(sum(chances) / max(len(written), 1), 3)
    writer.last_least_chance = round(min(chances, default=1.0), 4)
    writer.last_token_confidences = [round(chance, 4) for chance in chances]
    writer.last_written_words = words
    writer.last_uncertain_words = [
        word
        for word, chance in zip(words, writer.last_token_confidences, strict=False)
        if chance < critique_service.MIN_WORD_CONFIDENCE
    ]


def critique(writer: Any, answer: str, context: str) -> Critique:
    """What the model thinks of the answer it wrote."""
    return critique_service.critique(
        answer,
        writer.last_token_confidences,
        writer.last_written_words,
        material=context,
    )


def write_again(writer: Any, request: Request, first: Answer) -> str:
    """Write the answer again, with the critique in front of the model."""
    saved = (writer.last_token_confidences, writer.last_written_words, writer.last_uncertain_words)
    retry = Request(
        request.instruction,
        f"{request.context} {first.critique.summary}",
        request.max_words,
        request.reading,
    )
    written = write(writer, encode_instruction(writer, retry), request.max_words)
    if not written:
        _restore(writer, saved, first.critique)
        return first.text
    answer = writer.decode(written)
    second = critique(writer, answer, request.context)
    if second.confidence < first.critique.confidence:
        _restore(writer, saved, first.critique)
        return first.text
    writer.last_critique = second
    return answer


def _restore(
    writer: Any,
    saved: tuple[list[float], list[str], list[str]],
    result: Critique,
) -> None:
    """Put back what the model was sure about before it wrote again."""
    writer.last_token_confidences, writer.last_written_words, writer.last_uncertain_words = saved
    writer.last_critique = result


def respond(writer: Any, request: Request) -> str | None:
    """Answer a request in the words the model was taught."""
    if not writer.trained or writer.model is None:
        return None
    tokens = encode_instruction(writer, request)
    written = write(writer, tokens, request.max_words)
    if not written:
        return None
    answer = writer.decode(written)
    first = Answer(answer, critique(writer, answer, request.context))
    writer.last_critique = first.critique
    if first.critique.should_retry:
        answer = write_again(writer, request, first)
    return answer


def generate(writer: Any, prompt: str = "", settings: WritingSettings | None = None) -> str | None:
    """Write what comes after the prompt, in the words it has learned."""
    from app.services.neural import freewriting

    return freewriting.generate(writer, prompt, settings)
