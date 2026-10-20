"""A small transformer language model, trained from scratch on the tenant's own words.

There is no pretrained model anywhere in here. The weights are learned from the
lessons, the answers and the corrections the trainer gave, on the machine that
runs the app, and they are kept per tenant next to the other brain files.

The model is a decoder-only transformer: it reads the words before it and
writes the next one. It is the writer of the assistant, and the one that can
propose an answer for a question it was never taught.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.vocabulary import CANONICAL_OWNER, Vocabulary, guard_tensor_ids


# The word and the punctuation around it.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?|[.!?]")

# A small model, because the data is small: a few hundred sentences.
MODEL_CONFIG: dict[str, Any] = {
    "d_model": 128,
    "layers": 3,
    "heads": 4,
    "context": 64,
    "dropout": 0.1,
}

# The punctuation that ends a sentence.
_END_TOKENS = {".", "?", "!"}
# The marks the model uses to know where a sentence starts and ends.
_START = "<start>"
_FINISH = "<end>"
_UNKNOWN = "<unk>"
# The mark used to line the sentences up in a batch. It is never a word.
_PAD = "<pad>"


def tokenize(text: str) -> list[str]:
    """The words of a text, in lower case, with their punctuation."""
    return [match.group(0).lower().replace("’", "'") for match in _TOKEN_RE.finditer(text or "")]


# The GPU is used for the training. The answers are written on the processor,
# so the training in the background never meets another use of the GPU.
_GPU_LOCK = threading.Lock()


def device_name() -> str:
    """The device the model trains and writes on."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():  # pragma: no cover - not used on this machine
        return "cuda"
    return "cpu"


class Block(nn.Module):
    """One transformer block: attention over the words before, then a feed forward."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(x)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask, is_causal=True)
        x = x + attended
        return x + self.feed_forward(self.norm_feed_forward(x))


class TinyTransformer(nn.Module):
    """A decoder-only transformer over the words of the tenant's own material."""

    def __init__(self, vocabulary_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        settings = {**MODEL_CONFIG, **(config or {})}
        self.config = settings
        d_model = int(settings["d_model"])
        self.vocabulary_size = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        self.position_embedding = nn.Embedding(int(settings["context"]), d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, int(settings["heads"]), float(settings["dropout"]))
            for _ in range(int(settings["layers"]))
        )
        self.norm = nn.LayerNorm(d_model)
        # The output shares the weights of the input, which helps a small model.
        self.output = nn.Linear(d_model, vocabulary_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(self.norm(hidden))


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
        keep = set(tokens) | _END_TOKENS | {_START, _FINISH, _UNKNOWN, _PAD}
        self._vocabulary = Vocabulary(CANONICAL_OWNER, sorted(keep))

    def encode(self, text: str) -> list[int]:
        # The ids come from the canonical vocabulary and are checked before
        # they are used, so an id from an old mapping can never reach a tensor.
        return guard_tensor_ids(self._vocabulary.encode(text), "the writer", self._vocabulary)

    def decode(self, tokens: Iterable[int]) -> str:
        words = [self.vocabulary[token] for token in tokens if 0 <= token < len(self.vocabulary)]
        text = " ".join(words)
        text = re.sub(r"\s+([.!?,])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    # -- training ---------------------------------------------------------

    def fit(
        self,
        sentences: Sequence[str],
        steps: int = 300,
        batch_size: int = 16,
        learning_rate: float = 3e-3,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Learn the words of the sentences, from scratch."""
        tokens: list[str] = []
        for sentence in sentences:
            tokens.extend(tokenize(sentence))
        if len(tokens) < 40:
            return {"trained": False, "reason": "not enough words to train on"}

        self._build_vocabulary(tokens)
        if len(self.vocabulary) < 8:
            return {"trained": False, "reason": "not enough words to train on"}

        generator = torch.Generator().manual_seed(seed)
        # Every sentence is its own sequence, so the model learns where a
        # sentence starts and where it ends.
        sequences = [
            guard_tensor_ids(
                [self.index[_START], *[self.index.get(token, self.index[_UNKNOWN]) for token in tokenize(sentence)], self.index[_FINISH]],
                "the training",
                self._vocabulary,
            )
            for sentence in sentences
            if len(tokenize(sentence)) > 1
        ]
        context = int(self.config["context"])

        device = torch.device(device_name())
        with _GPU_LOCK:
            model = TinyTransformer(len(self.vocabulary), self.config).to(device)
            optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(steps, 1))
            model.train()

            started = time.time()
            last_loss = 0.0
            for _ in range(steps):
                windows = []
                for _ in range(batch_size):
                    sequence = sequences[int(torch.randint(0, len(sequences), (1,), generator=generator).item())]
                    if len(sequence) >= context:
                        start = int(torch.randint(0, len(sequence) - context + 1, (1,), generator=generator).item())
                    else:
                        start = 0
                    window = sequence[start:start + context]
                    if len(window) < 2:
                        continue
                    windows.append(window)
                if not windows:
                    break
                width = max(len(window) for window in windows)
                padded = []
                for window in windows:
                    padded.append(window + [self.index[_PAD]] * (width - len(window)))
                batch = torch.tensor(padded, dtype=torch.long).to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, len(self.vocabulary)),
                    targets.reshape(-1),
                    ignore_index=self.index[_PAD],
                )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                last_loss = float(loss.item())

        model.eval()
        # The answers are written on the processor, so the training in the
        # background never meets another use of the GPU.
        self.model = model.to("cpu")
        self.trained = True
        self.loss = round(last_loss, 4)
        self.steps = steps
        return {
            "trained": True,
            "vocabulary": len(self.vocabulary),
            "sentences": len(sequences),
            "tokens": len(tokens),
            "steps": steps,
            "loss": self.loss,
            "seconds": round(time.time() - started, 1),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": device.type,
        }

    # -- writing ----------------------------------------------------------

    def generate(
        self,
        prompt: str = "",
        max_tokens: int = 24,
        temperature: float = 0.3,
        top_k: int = 5,
        seed: int | None = None,
        include_prompt: bool = False,
    ) -> str | None:
        """Write what comes after the prompt, in the words it has learned."""
        if not self.trained or self.model is None:
            return None
        device = next(self.model.parameters()).device
        generator = torch.Generator(device="cpu").manual_seed(seed if seed is not None else 11)
        context = int(self.config["context"])

        tokens = self.encode(prompt) if prompt else []
        if not tokens:
            tokens = [self.index.get(_START, 0)]
        if len(tokens) >= context:
            tokens = tokens[-context:]

        written: list[int] = []
        confidence = 0.0
        least_chance = 1.0
        with torch.no_grad():
            for _ in range(max_tokens):
                window = tokens[-context:]
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                logits = self.model(input_ids)[0, -1]
                logits = logits / max(temperature, 0.05)
                if self.index.get(_UNKNOWN) is not None:
                    # The unknown mark is never written.
                    logits[self.index[_UNKNOWN]] = -float("inf")
                # A pair of words that came before is a loop, not a sentence.
                if len(tokens) >= 2:
                    repeated = (tokens[-2], tokens[-1])
                    for position, token_id in enumerate(tokens[:-1]):
                        if token_id == repeated[1] and position and tokens[position - 1] == repeated[0]:
                            logits[token_id] = -float("inf")
                # The same word twice in a row is a stutter, not a sentence.
                if tokens:
                    logits[tokens[-1]] = min(float(logits[tokens[-1]]), -1e4)
                values, positions = torch.topk(logits, min(top_k, logits.shape[-1]))
                probabilities = torch.softmax(values, dim=-1)
                choice = int(torch.multinomial(probabilities.cpu(), 1, generator=generator).item())
                token_id = int(positions[choice].item())
                token = self.vocabulary[token_id]
                # How sure the model is about the word it wrote. The least
                # sure word is the one that gives the sentence away.
                chance = float(probabilities[choice].clamp_min(1e-9).item())
                confidence += chance
                if chance < least_chance:
                    least_chance = chance
                tokens.append(token_id)
                written.append(token_id)
                if token in _END_TOKENS or token == _FINISH:
                    break
        # The start and the end marks are not part of the sentence.
        skip = {self.index.get(_START), self.index.get(_FINISH), self.index.get(_PAD)}
        if include_prompt:
            return self.decode([token for token in tokens if token not in skip]) or None
        text = self.decode([token for token in written if token not in skip])
        if not text:
            return None
        self.last_confidence = round(confidence / max(len(written), 1), 3)
        self.last_least_chance = round(least_chance, 4)
        return text

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
        log_chances = F.log_softmax(logits[:-1], dim=-1)
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
    def load(cls, path: Path) -> "TransformerWriter | None":
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
