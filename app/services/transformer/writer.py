"""A small transformer language model, trained from scratch on the tenant's own words.

There is no pretrained model anywhere in here. The weights are learned from the
lessons, the answers and the corrections the trainer gave, on the machine that
runs the app, and they are kept per tenant next to the other brain files.

The model is a decoder-only transformer: it reads the words before it and
writes the next one. It is the writer of the assistant, and the one that can
answer a request the trainer never worded that way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from app.services.transformer import critique as critique_service
from app.services.transformer import examples as example_service
from app.services.transformer import freewriting, quality
from app.services.transformer.critique import Answer, Critique
from app.services.transformer.model import MODEL_CONFIG, TinyTransformer, device_name
from app.services.transformer.settings import ReadingSettings, TrainingSettings, WritingSettings
from app.services.transformer.tokens import (
    CONTEXT,
    FINISH,
    INSTRUCTION,
    PAD,
    RESPONSE,
    SPECIAL_TOKENS,
    START,
    UNKNOWN,
    decode_text,
    tokenize,
)
from app.services.transformer.training import train_model
from app.services.vocabulary import CANONICAL_OWNER, Vocabulary, guard_tensor_ids


class TransformerWriter:
    """The trained writer: it learns the tenant's words and writes with them."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**MODEL_CONFIG, **(config or {})}
        # The one live token list. The transformer's own mapping is the
        # canonical one, so the ids and the words always line up.
        self._vocabulary = Vocabulary(CANONICAL_OWNER, ())
        self.model: TinyTransformer | None = None
        self.trained = False
        self.loss: float | None = None
        self.steps: int = 0
        # How sure the model was about the words it wrote last.
        self.last_confidence: float = 0.0
        # The least sure word of the last sentence it wrote.
        self.last_least_chance: float = 0.0
        # How sure it was about every single word it wrote.
        self.last_token_confidences: list[float] = []
        # The words it wrote.
        self.last_written_words: list[str] = []
        # The words it was not sure about.
        self.last_uncertain_words: list[str] = []
        # What it thought of the last answer it wrote.
        self.last_critique: Critique | None = None

    # -- the vocabulary ---------------------------------------------------

    @property
    def vocabulary(self) -> list[str]:
        """The tokens the model knows, in id order."""
        return list(self._vocabulary.tokens)

    @property
    def index(self) -> dict[str, int]:
        """The token to id mapping of the canonical vocabulary."""
        return self._vocabulary.index

    def _build_vocabulary(self, tokens: Sequence[str]) -> None:
        # Every word of the corpus is kept: there are not many of them, and a
        # word that is dropped would come back as the unknown mark.
        keep = set(tokens) | set(SPECIAL_TOKENS)
        self._vocabulary = Vocabulary(CANONICAL_OWNER, sorted(keep))

    def encode(self, text: str) -> list[int]:
        # The ids come from the canonical vocabulary and are checked before
        # they are used, so an id from an old mapping can never reach a tensor.
        return guard_tensor_ids(self._vocabulary.encode(text), "the writer", self._vocabulary)

    def decode(self, tokens: Iterable[int]) -> str:
        words = [self.vocabulary[token] for token in tokens if 0 <= token < len(self.vocabulary)]
        return decode_text(words)

    # -- training ---------------------------------------------------------

    def fit(
        self,
        sentences: Sequence[str],
        instructions: Sequence[tuple[str, str]] = (),
        contexts: Sequence[tuple[str, str, str]] = (),
        settings: TrainingSettings | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Learn the words of the sentences and the answers to the requests.

        The sentences that are not worth the training steps are left out, and
        the rest are learned from the easy ones to the hard ones. ``progress``
        is told how far the training has come, at every check the loss is
        read at.
        """
        settings = settings or TrainingSettings()
        words = example_service.all_words(sentences, instructions, contexts)
        self._build_vocabulary(words)
        kept = quality.keep(sentences)
        examples = example_service.build(self, kept, instructions, contexts)
        if len(examples) < 4 or len(self.vocabulary) < 8:
            return {"trained": False, "reason": "not enough words to train on"}
        model, report, seconds = train_model(
            len(self.vocabulary),
            examples,
            {**self.config, "pad_id": self.index[PAD]},
            settings=settings,
            progress=progress,
        )
        # The answers are written on the processor, so the training in the
        # background never meets another use of the GPU.
        self.model = model.to("cpu")
        self.trained = True
        self.loss = report.loss
        self.steps = report.steps
        return {
            "trained": True,
            "vocabulary": len(self.vocabulary),
            "sentences": len(examples),
            "tokens": len(self.vocabulary),
            "steps": report.steps,
            "loss": report.loss,
            "best_loss": report.best_loss,
            "stopped_early": report.stopped_early,
            "curve": report.curve,
            "dropped_sentences": len(sentences) - len(kept),
            "seconds": round(seconds, 1),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": device_name(),
        }

    # -- answering a request ----------------------------------------------

    def respond(
        self,
        instruction: str,
        context: str = "",
        max_words: int = 64,
        reading: ReadingSettings | None = None,
    ) -> str | None:
        """Answer a request in the words the model was taught.

        The model reads its own answer before it is given away. When the
        answer is not good enough, the model is asked again with the critique
        in front of it, and the better of the two answers is the one kept.
        """
        if not self.trained or self.model is None:
            return None
        reading = reading or ReadingSettings()
        tokens = self._encode_instruction(instruction, context, reading)
        written = self._write(tokens, max_words)
        if not written:
            return None
        answer = self.decode(written)
        first = Answer(answer, self._critique(answer, context))
        if first.critique.should_retry:
            answer = self._write_again(instruction, context, reading, max_words, first)
        self.last_critique = first.critique if answer == first.text else self.last_critique
        return answer

    def _write_again(
        self,
        instruction: str,
        context: str,
        reading: ReadingSettings,
        max_words: int,
        first: Answer,
    ) -> str:
        """Write the answer again, with the critique in front of the model.

        The first answer is kept when the second one is not better, and the
        words the model was sure about stay the words of the answer that is
        given away.
        """
        saved = (self.last_token_confidences, self.last_written_words, self.last_uncertain_words)
        tokens = self._encode_instruction(
            instruction, f"{context} {first.critique.summary}", reading
        )
        written = self._write(tokens, max_words)
        if not written:
            self._restore(saved)
            self.last_critique = first.critique
            return first.text
        answer = self.decode(written)
        critique = self._critique(answer, context)
        if critique.confidence < first.critique.confidence:
            self._restore(saved)
            self.last_critique = first.critique
            return first.text
        self.last_critique = critique
        return answer

    def _restore(self, saved: tuple[list[float], list[str], list[str]]) -> None:
        """Put back what the model was sure about before it wrote again."""
        self.last_token_confidences, self.last_written_words, self.last_uncertain_words = saved

    def _critique(self, answer: str, context: str) -> Critique:
        """What the model thinks of the answer it wrote."""
        return critique_service.critique(
            answer,
            self.last_token_confidences,
            self.last_written_words,
            material=context,
        )

    def _encode_instruction(
        self, instruction: str, context: str, reading: ReadingSettings
    ) -> list[int]:
        """The words the model reads before it starts answering.

        The marks are the same ones the model was trained with, so the
        request and the material are read the way it learned to read them.
        The request is never cut short: it is the part that says what is
        being asked for.
        """
        tokens = [self.index[START], self.index[INSTRUCTION]]
        tokens += [
            self.index.get(token, self.index[UNKNOWN])
            for token in tokenize(instruction)[-reading.instruction_words :]
        ]
        if context:
            tokens += [self.index[CONTEXT]]
            tokens += [
                self.index.get(token, self.index[UNKNOWN])
                for token in tokenize(context)[-reading.material_words :]
            ]
        tokens += [self.index[RESPONSE]]
        return tokens

    def _write(self, tokens: list[int], max_words: int) -> list[int]:
        """The words the model writes after reading what it has read.

        The answer ends when the model says it is done, not at the first full
        stop: a poem has a full stop in the middle of it. How sure the model
        was about every word is kept, so the answer can be read again before
        it is given away.
        """
        device = next(self.model.parameters()).device
        written: list[int] = []
        chances: list[float] = []
        confidence = 0.0
        least_chance = 1.0
        finish = self.index.get(FINISH)
        with torch.no_grad():
            for _ in range(max_words):
                window = tokens[-(int(self.config["context"])) :]
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                logits = self.model(input_ids)[0, -1]
                if self.index.get(UNKNOWN) is not None:
                    # The unknown mark is never written.
                    logits[self.index[UNKNOWN]] = -float("inf")
                if tokens:
                    # The same word twice in a row is a stutter, not an answer.
                    logits[tokens[-1]] = min(float(logits[tokens[-1]]), -1e4)
                token_id = int(torch.argmax(logits).item())
                chances_of_words = torch.softmax(logits, dim=-1)
                chance = float(chances_of_words[token_id].clamp_min(1e-9).item())
                confidence += chance
                least_chance = min(least_chance, chance)
                chances.append(chance)
                tokens.append(token_id)
                written.append(token_id)
                if token_id == finish:
                    break
        skip = {self.index.get(START), self.index.get(FINISH), self.index.get(PAD)}
        self.last_confidence = round(confidence / max(len(written), 1), 3)
        self.last_least_chance = round(least_chance, 4)
        self.last_token_confidences = [round(chance, 4) for chance in chances]
        self.last_written_words = [
            self.vocabulary[token] for token in written if token < len(self.vocabulary)
        ]
        self.last_uncertain_words = [
            word
            for word, chance in zip(
                self.last_written_words, self.last_token_confidences, strict=False
            )
            if chance < critique_service.MIN_WORD_CONFIDENCE
        ]
        return [token for token in written if token not in skip]

    def generate(
        self,
        prompt: str = "",
        settings: WritingSettings | None = None,
    ) -> str | None:
        """Write what comes after the prompt, in the words it has learned."""
        return freewriting.generate(self, prompt, settings)

    def score(self, text: str) -> float:
        """How well the model knows a sentence, as a log chance per word.

        A higher score means the model has seen that wording more often.
        """
        if not self.trained or self.model is None:
            return 0.0
        tokens = self.encode(text)
        if len(tokens) < 2:
            return 0.0
        device = next(self.model.parameters()).device
        with torch.no_grad():
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.model(input_ids)[0]
        log_chances = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
        chosen = log_chances[torch.arange(len(tokens) - 1), torch.tensor(tokens[1:], device=device)]
        return float(chosen.mean().item())

    # -- keeping the model -----------------------------------------------

    def save(self, path: Path) -> None:
        if not self.trained or self.model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "vocabulary": self.vocabulary,
                "state": self.model.state_dict(),
                "loss": self.loss,
                "steps": self.steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TransformerWriter | None:
        if not path.is_file():
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            writer = cls(config=data.get("config"))
            writer._vocabulary = Vocabulary(
                CANONICAL_OWNER, [str(token) for token in data.get("vocabulary", [])]
            )
            model = TinyTransformer(max(len(writer.vocabulary), 1), writer.config)
            model.load_state_dict(data["state"])
            model.eval()
            writer.model = model
            writer.trained = True
            writer.loss = data.get("loss")
            writer.steps = int(data.get("steps") or 0)
            return writer
        except Exception:
            return None
