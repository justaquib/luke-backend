"""What a run measures under: checkpoint, tokenizer, decoding and prompts.

Kept apart from the job loop so the provenance of a number can be read without
reading the loop that produced it, and so every run records the same shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

MAX_EVIDENCE_ITEMS = 4
MAX_EVIDENCE_CHARS = 600
MAX_NEW_TOKENS = 220

SAFE_FAILURE = "The assessment stopped unexpectedly."

SYSTEM = (
    "You are taking a skills assessment. Read the task carefully, follow its "
    "instructions exactly, and answer with the shortest correct response it allows."
)


def messages(item: dict[str, Any], evidence: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    """The one shape every item is asked in, with evidence only in assisted mode."""
    task = str(item["prompt"])
    if evidence:
        lines = [
            f"[{index}] {' '.join(hit['text'].split())[:MAX_EVIDENCE_CHARS]}"
            for index, hit in enumerate(evidence, start=1)
        ]
        task = "Evidence:\n" + "\n".join(lines) + "\n\nTask: " + task
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def evidence(prompt: str, tenant_id: int | None) -> tuple[dict[str, str], ...]:
    """What the corpus says about the task, read only in assisted mode."""
    from app.services import vector_store

    hits = vector_store.query(prompt, n_results=MAX_EVIDENCE_ITEMS, tenant_id=tenant_id)
    return tuple(
        {
            "text": str(hit.get("text") or ""),
            "source": str((hit.get("metadata") or {}).get("source") or "corpus"),
        }
        for hit in hits
    )


def load_model() -> Any:
    from app.services.foundation import runtime

    return runtime.load()


def generate(model: Any, messages: list[dict[str, str]], cancel: Any) -> tuple[str, float]:
    """One generation, with the settings this run recorded."""
    result = model.generate(messages, generation_params(), cancel)
    return result.text, float(result.seconds)


def generation_params() -> Any:
    from app.services.foundation.runtime import GenerationParams

    # Greedy decoding: an assessment number should not move because of a
    # sampling seed. The settings are recorded in the run row either way.
    return GenerationParams(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)


def provenance() -> dict[str, Any]:
    """The exact thing being measured, read from the installed checkpoint.

    When an adapter is loaded, its folder and weight hash are named in the
    artifact string, so a score with an adapter can never be mistaken for one
    of the base checkpoint.
    """
    from app.services.foundation import runtime
    from app.services.foundation.identity import load_identity

    identity = load_identity()
    artifact = f"{identity.base_model}@{identity.revision}#{identity.weight_sha256[:16]}"
    label = runtime.adapter()
    if label:
        artifact = f"{artifact}+adapter:{label}"
    return {
        "model_id": f"{identity.name} {identity.version}",
        "model_artifact": artifact,
        "tokenizer_id": tokenizer_id(str(identity.folder)),
        "generation": json.dumps(asdict(generation_params())),
        "bank_version": bank_version(),
        "bank_salt": bank_salt(),
    }


def tokenizer_id(folder: str) -> str:
    """The tokenizer file's own hash, so a swapped tokenizer is visible."""
    from app.services.foundation.identity import PINNED_REVISION, sha256_file

    path = Path(folder) / "tokenizer.json"
    prefix = f"Qwen2.5-0.5B-Instruct@{PINNED_REVISION[:7]}"
    if not path.exists():
        return prefix
    return f"{prefix}:tokenizer.json:{sha256_file(path)[:16]}"


def bank_version() -> int:
    from app.services.assessment import bank

    return bank.version()


def bank_salt() -> str:
    from app.services.assessment import bank

    return bank.salt()


def failure_message(error: Exception) -> str:
    from app.services.foundation.identity import FoundationError

    if isinstance(error, FoundationError):
        return f"Luke's foundation could not answer: {error}"
    return SAFE_FAILURE


def generation(run: Any) -> dict[str, Any]:
    try:
        payload = json.loads(run.generation or "{}")
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}
