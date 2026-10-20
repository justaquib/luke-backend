"""A reasoning pass over everything the assistant knows.

The retriever already answers "what is close to this?". This module answers
"how did we get here?" by planning the question, recalling from every memory
store, relating the candidates to each other, and explaining the choice. The
result is a trace of steps plus a small graph of the nodes that were visited,
which the UI can draw.
"""

from __future__ import annotations

import re
from typing import Any

from app.services import dictionary, vector_store


_QUESTION_KINDS = (
    ("calculation", re.compile(r"(\d+\s*[-+*/x×÷]\s*\d+)|(\b(?:plus|minus|times|divided|multiplied|square root|percent)\b)")),
    ("definition", re.compile(r"\bwhat (?:is|are|does|do)\b|\bdefine\b|\bmeaning of\b", re.IGNORECASE)),
    ("letter", re.compile(r"\bletter\b|\balphabet\b", re.IGNORECASE)),
    ("counting", re.compile(r"\bhow many\b|\bcount\b", re.IGNORECASE)),
    ("spelling", re.compile(r"\bspell\b|\bwritten in words\b", re.IGNORECASE)),
    ("time", re.compile(r"\bwhat time\b|\bwhat day\b|\bwhat is the date\b", re.IGNORECASE)),
)

_STOP_WORDS = {
    "what", "who", "where", "when", "which", "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "of", "for", "please", "tell", "me", "in", "on", "at", "to", "with", "and",
    "or", "but", "if", "then", "this", "that", "these", "those", "it", "its", "my", "your", "his",
    "her", "our", "their", "i", "you", "he", "she", "we", "they", "can", "could", "would",
    "should", "will", "shall", "here", "there", "from", "about", "how", "why", "s", "t",
}


def understand(question: str) -> tuple[str, list[str]]:
    """Work out what kind of question this is and which words matter."""
    kind = "lookup"
    for name, pattern in _QUESTION_KINDS:
        if pattern.search(question):
            kind = name
            break
    entities = [
        word for word in re.findall(r"[A-Za-z][A-Za-z'’-]*", question.lower())
        if word not in _STOP_WORDS and len(word) > 1
    ]
    return kind, entities[:8]


def tokens(question: str) -> list[str]:
    """The words of the question that carry the meaning.

    These are the tokens the lookup was made with, so the trainer can see
    which words of the question carried the answer.
    """
    _, entities = understand(question)
    return entities


def _recall(question: str, tenant_id: int | None) -> list[dict]:
    """Pull the closest candidates from every memory store."""
    recalls: list[dict] = []

    lessons = vector_store.search_lessons(question, n_results=3, tenant_id=tenant_id)
    recalls.append({
        "store": "Learner lessons",
        "source": "language_model",
        "hits": [
            {"id": f"lesson:{hit.get('lesson_id')}", "label": hit["question"], "answer": hit.get("answer", ""), "score": hit.get("distance"), "type": "lesson"}
            for hit in lessons
        ],
    })

    answers = vector_store.search_faq(question, n_results=3, tenant_id=tenant_id)
    recalls.append({
        "store": "Trained answers",
        "source": "faq",
        "hits": [
            {"id": f"answer:{hit.get('faq_id')}", "label": hit["question"], "answer": hit.get("answer", ""), "score": hit.get("distance"), "type": "trained_answer"}
            for hit in answers
        ],
    })

    intents = vector_store.search_intents(question, n_results=3, tenant_id=tenant_id)
    recalls.append({
        "store": "Saved intents",
        "source": "intent",
        "hits": [
            {"id": f"intent:{index}", "label": hit.get("name", ""), "answer": hit.get("response_template", ""), "score": hit.get("distance"), "type": "intent"}
            for index, hit in enumerate(intents)
        ],
    })

    documents = vector_store.query(question, n_results=3, tenant_id=tenant_id)
    recalls.append({
        "store": "Documents",
        "source": "docs",
        "hits": [
            {
                "id": f"document:{hit.get('metadata', {}).get('source', index)}:{index}",
                "label": _document_label(hit.get("metadata", {}), index),
                "answer": hit.get("text", "")[:160],
                "score": hit.get("distance"),
                "type": "document",
            }
            for index, hit in enumerate(documents)
        ],
    })

    return recalls


def _document_label(metadata: dict, index: int) -> str:
    source = metadata.get("source") or "Passage"
    page = metadata.get("page_number")
    return f"{source} · page {page}" if page else f"{source}"


def _relate(nodes: list[dict], edges: list[dict], question: str) -> None:
    """Connect the visited nodes to the words of the question."""
    for node in nodes:
        if node["type"] == "question":
            continue
        words = set(re.findall(r"[A-Za-z][A-Za-z'’-]*", (node.get("label") or "").lower()))
        shared = words & set(node.get("entities", []))
        if shared:
            edges.append({
                "source": node["id"],
                "target": "question",
                "label": ", ".join(sorted(shared)[:3]),
            })


def explain(
    question: str,
    tenant_id: int | None,
    answer: str,
    source: str,
    matched_question: str | None = None,
) -> dict[str, Any]:
    """Build the reasoning trace and the knowledge graph for one question."""
    kind, entities = understand(question)
    recalls = _recall(question, tenant_id)

    steps: list[dict[str, Any]] = [{
        "label": "Understand",
        "detail": f"Treated as a {kind} question. Key words: {', '.join(entities) if entities else 'none'}.",
        "outcome": "planned",
    }]

    nodes: list[dict[str, Any]] = [{
        "id": "question",
        "label": question.strip(),
        "type": "question",
        "score": 1.0,
        "entities": entities,
    }]
    edges: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for recall in recalls:
        hits = recall["hits"]
        if not hits:
            steps.append({
                "label": recall["store"],
                "detail": "Nothing found in this memory.",
                "outcome": "empty",
            })
            continue
        closest = hits[0]
        distance = closest.get("score")
        closeness = 0.0 if distance is None else max(0.0, 1.0 - float(distance))
        steps.append({
            "label": recall["store"],
            "detail": f"{len(hits)} candidate(s). Closest: {closest['label']}" + (f" ({float(distance):.2f})" if distance is not None else "") + ".",
            "outcome": "searched",
        })
        for hit in hits:
            node = {
                "id": hit["id"],
                "label": hit["label"],
                "type": hit["type"],
                "score": round(closeness if hit is closest else max(0.0, 1.0 - float(hit.get("score") or 1.0)), 2),
                "answer": hit.get("answer", ""),
                "entities": entities,
            }
            nodes.append(node)
            if best is None or node["score"] > best["score"]:
                best = node
        if hits:
            edges.append({
                "source": "question",
                "target": hits[0]["id"],
                "label": f"closest in {recall['store'].lower()}",
            })

    _relate(nodes, edges, question)

    if best is not None:
        edges.append({"source": best["id"], "target": "result", "label": "used for the answer"})

    chosen = next((node for node in nodes if node["type"] == "trained_answer" and node.get("id") == best["id"]), None)
    confidence = best["score"] if best else 0.0
    if matched_question:
        steps.append({
            "label": "Decide",
            "detail": f"The reply came from {source.replace('_', ' ')} using the taught question \"{matched_question}\".",
            "outcome": "selected",
        })
    else:
        steps.append({
            "label": "Decide",
            "detail": f"The reply came from {source.replace('_', ' ')}.",
            "outcome": "selected",
        })

    nodes.append({
        "id": "result",
        "label": answer[:120] if answer else "No answer",
        "type": "result",
        "score": round(confidence, 2),
        "entities": [],
    })

    return {
        "intent": kind,
        "entities": entities,
        "steps": steps,
        "nodes": nodes,
        "edges": edges,
        "confidence": round(confidence, 2),
    }


def knowledge_graph(tenant_id: int | None, subject: str = "", limit: int = 90) -> dict[str, Any]:
    """The shape of what the tenant has learned: subjects, lessons, answers, intents."""
    from app.services import language_logic

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except Exception:
        cards = []

    wanted = (subject or "").strip().lower()
    if wanted:
        cards = [
            card for card in cards
            if (card.get("subject") or "General").strip().lower() == wanted
        ]

    by_subject: dict[str, int] = {}
    for card in cards:
        subject_name = (card.get("subject") or "General").strip() or "General"
        by_subject[subject_name] = by_subject.get(subject_name, 0) + 1

    for subject_name, count in sorted(by_subject.items(), key=lambda item: -item[1]):
        nodes.append({
            "id": f"subject:{subject_name}",
            "label": subject_name,
            "type": "subject",
            "count": count,
        })

    for card in cards[: max(0, limit)]:
        subject_name = (card.get("subject") or "General").strip() or "General"
        node_id = f"lesson:{card.get('id')}"
        nodes.append({
            "id": node_id,
            "label": card.get("question", ""),
            "type": "lesson",
            "grade": card.get("grade_id", "nursery"),
        })
        edges.append({"source": node_id, "target": f"subject:{subject_name}", "label": "taught in"})

    return {
        "nodes": nodes,
        "edges": edges,
        "subjects": sorted(by_subject, key=lambda item: -by_subject[item]),
        "total": len(cards),
    }
