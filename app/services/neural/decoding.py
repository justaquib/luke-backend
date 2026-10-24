"""Guards that stop a model from repeating itself while it writes.

A small model trained on a few sentences finds loops very easy: the same pair
of words can follow itself forever. The guards here are applied to the logits
before a token is chosen, so a loop is never written in the first place.
"""

from __future__ import annotations

import torch


def block_loops(logits: torch.Tensor, tokens: list[int]) -> None:
    """Stop the model from repeating itself."""
    if len(tokens) >= 2:
        # A pair of words that came before is a loop, not a sentence.
        repeated = (tokens[-2], tokens[-1])
        for position, token_id in enumerate(tokens[:-1]):
            if token_id == repeated[1] and position and tokens[position - 1] == repeated[0]:
                logits[token_id] = -float("inf")
    # The same word twice in a row is a stutter, not a sentence.
    if tokens:
        logits[tokens[-1]] = min(float(logits[tokens[-1]]), -1e4)
