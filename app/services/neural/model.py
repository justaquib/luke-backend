"""The transformer blocks and their device placement."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

# The word model reads a few words of prefix and predicts the next one.
WORD_MODEL_CONFIG: dict[str, Any] = {
    "d_model": 128,
    "layers": 2,
    "heads": 4,
    "context": 64,
    "dropout": 0.1,
}


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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attention: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        normed = self.norm_attention(x)
        attended, weights = self.attention(
            normed,
            normed,
            normed,
            attn_mask=mask,
            is_causal=True,
            need_weights=attention is not None,
            average_attn_weights=False,
        )
        if attention is not None and weights is not None:
            attention.append(weights)
        x = x + attended
        return x + self.feed_forward(self.norm_feed_forward(x))


class TinyTransformer(nn.Module):
    """A decoder-only transformer over the words of the tenant's own material."""

    def __init__(self, vocabulary_size: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        settings = {**WORD_MODEL_CONFIG, **(config or {})}
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

    def forward(
        self,
        tokens: torch.Tensor,
        attention: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.full((length, length), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, mask, attention)
        return self.output(self.norm(hidden))
