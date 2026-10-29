"""Measure the installed foundation checkpoint on this machine.

The script reads the pinned checkpoint through the same loader the app uses,
then reports what it measured: the verified identity, the load time, the
writing speed on one fixed prompt, the process's peak memory and perplexity on
a small general-text sample that is not part of the taught corpus. It never
saves, trains or changes a weight.

Run with:
    .venv/bin/python scripts/measure_foundation.py
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = (
    "The water cycle moves water through the air, the land and the sea. Heat "
    "from the sun turns liquid water into vapour, which rises and cools. The "
    "cooling vapour condenses into tiny droplets that gather as clouds. When "
    "the droplets grow heavy enough, they fall back as rain or snow. Some of "
    "the water sinks into the soil, some runs into streams and rivers, and "
    "some returns to the ocean. The cycle then begins again, and it has no "
    "starting point."
)

PROMPT = "Explain in three sentences why the sky looks blue during the day."

STREAM_PROMPT = "Name one colour and describe it in one short sentence."


@cache
def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "identity": importlib.import_module("app.services.foundation.identity"),
        "runtime": importlib.import_module("app.services.foundation.runtime"),
    }


def _perplexity(model: Any, text: str) -> float:
    """The checkpoint's own perplexity on a sample it was not trained on."""
    import torch

    encoded = model.tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.model(**encoded, labels=encoded["input_ids"])
    return round(math.exp(float(output.loss)), 4)


def _streamed(model: Any, runtime: Any) -> dict[str, Any]:
    """Read one answer through the streaming path and time its first piece."""
    started = time.perf_counter()
    first: float | None = None
    chunks: list[str] = []
    params = runtime.GenerationParams(max_new_tokens=48, temperature=0)
    for chunk in model.stream([{"role": "user", "content": STREAM_PROMPT}], params):
        if first is None:
            first = time.perf_counter() - started
        chunks.append(chunk)
    text = "".join(chunks)
    return {
        "prompt": STREAM_PROMPT,
        "chunks": len(chunks),
        "first_seconds": round(first, 3) if first is not None else None,
        "seconds": round(time.perf_counter() - started, 3),
        "echoes_prompt": text.startswith(STREAM_PROMPT[:24]),
        "text": text,
    }


def _measure() -> dict[str, Any]:
    """Load the checkpoint once and read every number from the run."""
    modules = _modules()
    started = time.perf_counter()
    model = modules["runtime"].load()
    load_seconds = round(time.perf_counter() - started, 2)
    result = model.generate(
        [{"role": "user", "content": PROMPT}],
        modules["runtime"].GenerationParams(max_new_tokens=80),
    )
    identity = modules["identity"].load_identity()
    return {
        "identity": identity.as_dict(),
        "device": model.device,
        "load_seconds": load_seconds,
        "prompt": PROMPT,
        "new_tokens": result.new_tokens,
        "generation_seconds": result.seconds,
        "tokens_per_second": result.tokens_per_second,
        "stop_reason": result.stop_reason,
        "peak_memory_bytes": modules["runtime"].memory_bytes(),
        "sample_tokens": len(model.tokenizer(SAMPLE)["input_ids"]),
        "sample_perplexity": _perplexity(model, SAMPLE),
        "answer": result.text,
        "stream": _streamed(model, modules["runtime"]),
    }


def main() -> int:
    print(json.dumps(_measure(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
