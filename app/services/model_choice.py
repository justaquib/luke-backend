"""Which model answers the chat, and what that model expects next.

Luke is the taught assistant: it answers from the material it was trained on
and predicts the next word from its own word model. A fine-tuned Hugging Face
teacher answers the same question from the same retrieved material, and it
predicts the next word from its own weights. The two are chosen per message,
and one is never presented as the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.schemas import QueryResponse
from app.services.connections import Connection
from app.services.huggingface import service as hf_service
from app.services.huggingface import store as hf_store
from app.services.huggingface import text as hf_text
from app.services.neural import engine_for
from app.services.neural import prediction as word_prediction
from app.services.neural import store as neural_store

LUKE = "luke"
LUKE_NAME = "Luke"

# A teacher's answer must share at least this much of its content with the
# reference it was given, or the matched answer is shown instead.
_MIN_MATERIAL_SHARE = 1 / 3


@dataclass(frozen=True)
class ModelChoice:
    """The model the chat asked, and the teacher folder when one was chosen."""

    id: str
    name: str
    kind: str
    folder: str | None = None


def catalog(tenant_id: int | None) -> dict[str, Any]:
    """The models this tenant can answer with, Luke first."""
    trained = neural_store.words_path(tenant_id).is_file()
    models = [
        {
            "id": LUKE,
            "name": LUKE_NAME,
            "kind": "luke",
            "detail": "Answers from the taught material and predicts with its own word model.",
            "ready": trained,
        }
    ]
    for row in hf_store.list_teachers(tenant_id):
        models.append(
            {
                "id": f"hf:{row['folder']}",
                "name": str(row.get("name") or row["folder"]),
                "kind": "huggingface",
                "detail": (
                    f"Fine-tuned for {row.get('steps')} steps on this tenant's material."
                    if row.get("steps")
                    else "Fine-tuned on this tenant's material."
                ),
                "ready": True,
            }
        )
    return {"models": models, "default": LUKE}


def resolve(tenant_id: int | None, model_id: str | None) -> ModelChoice | None:
    """The model the request named, or Luke when it named none."""
    wanted = (model_id or "").strip()
    if not wanted or wanted == LUKE:
        return ModelChoice(LUKE, LUKE_NAME, "luke")
    if not wanted.startswith("hf:"):
        return None
    row = hf_service.teacher(tenant_id, wanted)
    if row is None:
        return None
    return ModelChoice(
        wanted,
        str(row.get("name") or wanted.removeprefix("hf:")),
        "huggingface",
        str(row["folder"]),
    )


def model_of(tenant_id: int | None, model_id: str | None) -> tuple[ModelChoice, bool]:
    """The chosen model, and whether the request's own choice was available."""
    choice = resolve(tenant_id, model_id)
    if choice is not None:
        return choice, True
    return ModelChoice(LUKE, LUKE_NAME, "luke"), False


def grounded(
    tenant_id: int | None, choice: ModelChoice, response: QueryResponse
) -> tuple[QueryResponse, Connection | None]:
    """The teacher's answer from the retrieved material, and what it did."""
    if choice.kind != "huggingface" or choice.folder is None:
        return response, None
    if response.source in {"pending", "weather", "wiki"}:
        return response, None
    passages = [hit.text for hit in response.hits]
    reference = hf_text.grounding(response.answer, passages)
    written = hf_service.answer(tenant_id, choice.folder, response.question, reference)
    if not written:
        return response, Connection(
            "model",
            "fail",
            f"{choice.name} did not write an answer, so the matched answer is shown.",
        )
    if not _uses_material(written, reference):
        return response, Connection(
            "model",
            "fail",
            f"{choice.name} wrote an answer that did not use the material, so the matched "
            "answer is shown.",
        )
    connection = Connection(
        "model",
        "match",
        f"{choice.name} wrote the answer with the retrieved material as its reference.",
        "",
    )
    return response.model_copy(update={"answer": written}), connection


def _uses_material(written: str, reference: str) -> bool:
    """Whether the written answer actually rewrites what it was given.

    At least two of its content words must come from the reference and make up
    a third of it. A small teacher often continues with facts of its own; those
    answers are refused so the matched answer is shown instead.
    """
    words = _content_words(written)
    material = _content_words(reference)
    if not material:
        return True
    if not words:
        return False
    shared = len(words & material)
    return shared >= 2 and shared / len(words) >= _MIN_MATERIAL_SHARE


def _content_words(text: str) -> set[str]:
    """The words of a text long enough to name something."""
    return set(re.findall(r"[a-z]{4,}", text.casefold()))


def predict(
    tenant_id: int | None, model_id: str | None, text: str, limit: int = 5
) -> dict[str, Any] | None:
    """The next words and the attention of the selected model."""
    choice, _available = model_of(tenant_id, model_id)
    if choice.kind == "huggingface" and choice.folder:
        result = hf_service.predict(tenant_id, choice.folder, text, limit)
        if result is None:
            return None
        result.update(
            {
                "model": choice.id,
                "model_name": choice.name,
                "kind": choice.kind,
            }
        )
        return result
    task = engine_for(tenant_id).word_task
    result = word_prediction.predict_with_attention(task, text, limit)
    if result is None:
        return None
    result.update(
        {
            "model": choice.id,
            "model_name": choice.name,
            "kind": choice.kind,
            "heads": int(task.config["heads"]) if task else None,
            "layers": int(task.config["layers"]) if task else None,
        }
    )
    return result


def name_of(choice: ModelChoice) -> str:
    """The label beside an answer: Luke's name or the teacher's own."""
    return choice.name
