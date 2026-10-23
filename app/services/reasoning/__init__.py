"""A reasoning pass over everything the assistant knows.

The retriever already answers "what is close to this?". This package answers
"how did we get here?" by reading the question, recalling from every memory
store in two passes, relating the candidates to each other, and explaining the
choice. The result is a trace of steps plus a small graph of the nodes that
were visited, which the UI can draw.
"""

from __future__ import annotations

from typing import Any

from app.services.reasoning import intent as intent_service
from app.services.reasoning import recall as recall_service
from app.services.reasoning import trace as trace_service
from app.services.reasoning.entities import entities_of, relate
from app.services.reasoning.graph import knowledge_graph

__all__ = ["explain", "knowledge_graph", "tokens", "understand"]


def _read(question: str, tenant_id: int | None) -> tuple[str, list[str], str]:
    """The kind of the question, the words that matter, and who said so."""
    kind, source = intent_service.classify(tenant_id, question)
    return kind, entities_of(question), source


def understand(question: str, tenant_id: int | None = None) -> tuple[str, list[str]]:
    """What kind of question this is, and which words matter."""
    kind, entities, _ = _read(question, tenant_id)
    return kind, entities


def tokens(question: str) -> list[str]:
    """The words of the question that carry the meaning.

    These are the tokens the lookup was made with, so the trainer can see
    which words of the question carried the answer.
    """
    _, entities = understand(question)
    return entities


def _closeness(hit: dict[str, Any]) -> float:
    """How close a candidate is, as a number between nothing and one."""
    distance = hit.get("score")
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def _candidate_nodes(
    recalls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """The nodes of the candidates, the edges between them, and the best one."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for recall in recalls:
        for hit in recall["hits"]:
            node = {
                "id": hit["id"],
                "label": hit["label"],
                "type": hit["type"],
                "score": round(_closeness(hit), 2),
                "answer": hit.get("answer", ""),
                "entities": entities_of(hit["label"]),
            }
            nodes.append(node)
            if best is None or node["score"] > best["score"]:
                best = node
        if recall["hits"]:
            edges.append(
                {
                    "source": "question",
                    "target": recall["hits"][0]["id"],
                    "label": f"closest in {recall['store'].lower()}",
                }
            )
    return nodes, edges, best


def explain(
    question: str,
    tenant_id: int | None,
    answer: str,
    source: str,
    matched_question: str | None = None,
) -> dict[str, Any]:
    """Build the reasoning trace and the knowledge graph for one question."""
    kind, entities, kind_source = _read(question, tenant_id)
    recalls = recall_service.recall(question, tenant_id)

    steps: list[dict[str, Any]] = [trace_service.understood_step(kind, entities, kind_source)]
    nodes: list[dict[str, Any]] = [
        {
            "id": "question",
            "label": question.strip(),
            "type": "question",
            "score": 1.0,
            "entities": entities,
        }
    ]

    candidates, edges, best = _candidate_nodes(recalls)
    nodes.extend(candidates)
    for recall in recalls:
        steps.append(trace_service.search_step(recall))

    relate(nodes, edges, question)

    steps.append(trace_service.why_step(question, best))
    steps.append(trace_service.decide_step(source, matched_question))

    if best is not None:
        edges.append({"source": best["id"], "target": "result", "label": "used for the answer"})
    confidence = float(best["score"]) if best else 0.0
    nodes.append(
        {
            "id": "result",
            "label": answer[:120] if answer else "No answer",
            "type": "result",
            "score": round(confidence, 2),
            "entities": [],
        }
    )

    return {
        "intent": kind,
        "entities": entities,
        "steps": steps,
        "nodes": nodes,
        "edges": edges,
        "confidence": round(confidence, 2),
    }
