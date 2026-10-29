"""Luke writing an answer when the taught material has nothing.

The taught answer is always tried first, and it always wins. This module runs
only after every taught source missed: it asks the loaded checkpoint to write
an answer, and it never stores that answer as knowledge. A written answer is
labelled `generated` and carries no citation. When the caller asks for a
composed answer, the passages are numbered, the model must cite them, and the
verifier refuses an answer that cites nothing or invents a number or a name;
the passages themselves are then served instead.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Sequence

from app.config import settings
from app.schemas import CitationOut, QueryHit, QueryRequest, QueryResponse
from app.services import model_settings
from app.services.connections import Connection
from app.services.foundation import identity as foundation_identity
from app.services.foundation import prompt as foundation_prompt
from app.services.foundation import runtime as foundation_runtime
from app.services.foundation import verify as foundation_verify
from app.services.language import expressions

logger = logging.getLogger(__name__)

GENERATED_SOURCE = "generated"
COMPOSED_SOURCE = "composed"


def installed() -> bool:
    """Whether the checkpoint is installed; a missing one is reported, not faked."""
    return foundation_identity.status()["installed"]


def missing_connection() -> Connection:
    """The dot that says why no answer was written."""
    reason = foundation_identity.status()["error"] or "Luke's foundation is not installed."
    return Connection("foundation", "miss", str(reason))


def evidence_from_hits(hits: Sequence[dict]) -> list[foundation_prompt.Evidence]:
    """The numbered evidence the prompt carries, one entry per passage."""
    evidence: list[foundation_prompt.Evidence] = []
    for index, hit in enumerate(hits, start=1):
        text = " ".join(str(hit.get("text") or "").split())
        if not text:
            continue
        metadata = hit.get("metadata") or {}
        evidence.append(
            foundation_prompt.Evidence(
                index=index,
                source=str(metadata.get("source") or "document"),
                label=str(metadata.get("source_name") or metadata.get("title") or "passage"),
                text=text,
            )
        )
    return evidence


def citations_for(evidence: Sequence[foundation_prompt.Evidence]) -> list[CitationOut]:
    """The evidence as the page lists it under the answer."""
    return [
        CitationOut(
            index=item.index,
            source=item.source,
            label=item.label,
            excerpt=" ".join(item.text.split())[:240],
        )
        for item in evidence
    ]


def write(
    req: QueryRequest,
    tenant_id: int | None,
    entities: list,
    model_name: str,
) -> tuple[QueryResponse, Connection]:
    """A generated answer, from the checkpoint's own knowledge only."""
    result, model = _generate(req, tenant_id, [])
    response = QueryResponse(
        question=req.question,
        source=GENERATED_SOURCE,
        answer=_written_text(result),
        hits=[],
        entities=entities,
        mode="generated",
        model=model.identity.base_model,
        model_name=model_name,
        model_version=model.identity.version,
        tokens_per_second=result.tokens_per_second,
    )
    detail = (
        f"Luke wrote this answer with {model.identity.display} in "
        f"{result.seconds:.2f} s ({result.tokens_per_second:.1f} tokens/s). "
        "No taught material covered the question, so it is not verified."
    )
    return response, Connection("foundation", "match", detail)


def compose(
    req: QueryRequest,
    tenant_id: int | None,
    entities: list,
    model_name: str,
    hits: Sequence[dict],
) -> tuple[QueryResponse, Connection] | None:
    """A written answer that must cite the numbered passages, or the passages."""
    evidence = evidence_from_hits(hits)
    if not evidence:
        return None
    result, model = _generate(req, tenant_id, evidence)
    written = _written_text(result)
    verified, reason = foundation_verify.verify_grounded(written, [item.text for item in evidence])
    if not verified:
        logger.info("The composed answer did not pass verification: %s", reason)
        return _extractive(req, entities, hits, evidence, reason)
    response = QueryResponse(
        question=req.question,
        source=COMPOSED_SOURCE,
        answer=written,
        hits=[QueryHit(**hit) for hit in hits],
        entities=entities,
        mode="hybrid",
        citations=citations_for(evidence),
        model=model.identity.base_model,
        model_name=model_name,
        model_version=model.identity.version,
        tokens_per_second=result.tokens_per_second,
    )
    detail = (
        f"Luke wrote this answer from {len(evidence)} numbered passages and cited "
        f"{len(foundation_verify.cited(result.text))} of them."
    )
    return response, Connection("foundation", "match", detail)


def stream(
    req: QueryRequest,
    tenant_id: int | None,
    entities: list,
    model_name: str,
    cancel: threading.Event | None = None,
) -> Iterator[tuple[str, object]]:
    """A generated answer as it is written: typing events, then the answer."""
    model = foundation_runtime.load()
    messages = _messages(req, tenant_id, [], model)
    params = _params()
    parts: list[str] = []
    started = threading.Event()
    for chunk in model.stream(messages, params, cancel):
        if not started.is_set():
            started.set()
            yield "writing", {"model": model_name, "mode": "generated"}
        parts.append(chunk)
        yield "token", {"text": chunk}
    if not started.is_set():
        yield "writing", {"model": model_name, "mode": "generated"}
    text = foundation_prompt.clean_answer("".join(parts))
    response = QueryResponse(
        question=req.question,
        source=GENERATED_SOURCE,
        answer=text,
        hits=[],
        entities=entities,
        mode="generated",
        model=model.identity.base_model,
        model_name=model_name,
        model_version=model.identity.version,
    )
    yield "answer", response


def _extractive(
    req: QueryRequest,
    entities: list,
    hits: Sequence[dict],
    evidence: Sequence[foundation_prompt.Evidence],
    reason: str,
) -> tuple[QueryResponse, Connection]:
    """The closest passage, when the written answer could not be verified."""
    response = QueryResponse(
        question=req.question,
        source="docs",
        answer=evidence[0].text,
        hits=[QueryHit(**hit) for hit in hits],
        entities=entities,
        mode="retrieved",
    )
    detail = (
        f"The written answer did not pass verification ({reason}); the closest passage is served."
    )
    return response, Connection("foundation", "miss", detail)


def _generate(
    req: QueryRequest,
    tenant_id: int | None,
    evidence: Sequence[foundation_prompt.Evidence],
) -> tuple[foundation_runtime.GenerationResult, foundation_runtime.FoundationModel]:
    """One full generation, with the prompt the model actually reads."""
    model = foundation_runtime.load()
    messages = _messages(req, tenant_id, evidence, model)
    return model.generate(messages, _params()), model


def _messages(
    req: QueryRequest,
    tenant_id: int | None,
    evidence: Sequence[foundation_prompt.Evidence],
    model: foundation_runtime.FoundationModel,
) -> list[dict[str, str]]:
    """The system prompt, the recent turns and the question, within the context."""
    name = model_settings.name_for(tenant_id)
    system = foundation_prompt.system_prompt(
        model.identity,
        name,
        foundation_prompt.Persona(),
        expressions.style_lines(tenant_id),
        len(evidence),
    )
    messages = foundation_prompt.build_messages(system, req.history, req.question, evidence)
    budget = model.identity.context_limit - settings.foundation_max_new_tokens - 128
    return foundation_prompt.trim_history(messages, model.count_tokens, budget)


def _written_text(result: foundation_runtime.GenerationResult) -> str:
    """The answer, cut back to its last finished sentence when the cap stopped it."""
    if result.stop_reason == "max_new_tokens":
        return foundation_prompt.trim_incomplete(result.text)
    return foundation_prompt.clean_answer(result.text)


def _params() -> foundation_runtime.GenerationParams:
    """The configured generation settings, held to the token limit."""
    return foundation_runtime.GenerationParams.from_settings()
