"""The model writing freely, without a request to answer.

This is the older way of using the model: give it the beginning of a text and
it writes what comes next. The instruction path in the writer is the newer
one, and the two live side by side.
"""

from __future__ import annotations

import torch

from app.services.neural.decoding import block_loops
from app.services.neural.settings import WritingSettings
from app.services.neural.tokens import END_TOKENS, FINISH, PAD, START

# The number of words the model reads before it writes.
MAX_PROMPT_WORDS = 96


def _next_token(
    model: torch.nn.Module,
    tokens: list[int],
    context: int,
    settings: WritingSettings,
    generator: torch.Generator,
) -> tuple[int, float]:
    """The next word, and how sure the model was about it."""
    device = next(model.parameters()).device
    window = tokens[-context:]
    input_ids = torch.tensor([window], dtype=torch.long, device=device)
    logits = model(input_ids)[0, -1]
    logits = logits / max(settings.temperature, 0.05)
    block_loops(logits, tokens)
    values, positions = torch.topk(logits, min(settings.top_k, logits.shape[-1]))
    probabilities = torch.softmax(values, dim=-1)
    choice = int(torch.multinomial(probabilities.cpu(), 1, generator=generator).item())
    token_id = int(positions[choice].item())
    return token_id, float(probabilities[choice].clamp_min(1e-9).item())


def generate(
    writer: object,
    prompt: str = "",
    settings: WritingSettings | None = None,
) -> str | None:
    """Write what comes after the prompt, in the words the model has learned."""
    settings = settings or WritingSettings()
    if not writer.trained or writer.model is None:
        return None
    generator = torch.Generator(device="cpu").manual_seed(
        settings.seed if settings.seed is not None else 11
    )
    context = int(writer.config["context"])

    tokens = writer.encode(prompt) if prompt else []
    if not tokens:
        tokens = [writer.index.get(START, 0)]
    if len(tokens) >= context:
        tokens = tokens[-context:]

    written: list[int] = []
    confidence = 0.0
    least_chance = 1.0
    # Inference mode keeps no autograd bookkeeping at all; the logits are only
    # changed inside it.
    with torch.inference_mode():
        for _ in range(settings.max_words):
            token_id, chance = _next_token(writer.model, tokens, context, settings, generator)
            confidence += chance
            least_chance = min(least_chance, chance)
            tokens.append(token_id)
            written.append(token_id)
            if writer.vocabulary[token_id] in END_TOKENS or token_id == writer.index.get(FINISH):
                break
    skip = {writer.index.get(START), writer.index.get(FINISH), writer.index.get(PAD)}
    if settings.include_prompt:
        return writer.decode([token for token in tokens if token not in skip]) or None
    text = writer.decode([token for token in written if token not in skip])
    if not text:
        return None
    writer.last_confidence = round(confidence / max(len(written), 1), 3)
    writer.last_least_chance = round(least_chance, 4)
    return text
