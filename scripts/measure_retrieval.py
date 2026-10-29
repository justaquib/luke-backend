"""Measure retrieval on the indexed chunks against chunks built by the new splitter.

The old chunks are read from the vector store, the new chunks are built from the
same uploaded text, and both are embedded in memory. Nothing is written: the
live index and the uploads are only read.

A question counts as found when the retrieved chunk holds the start of its
answer. The report also says how many questions survive whole in a chunk.

Run with:
    .venv/bin/python scripts/measure_retrieval.py --source <uploaded file>
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_LEAD_WORDS = 24


@cache
def _modules() -> tuple[Any, Any, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (
        importlib.import_module("app.services.nlp"),
        importlib.import_module("app.services.embeddings"),
        importlib.import_module("app.services.vector_store"),
    )


def _upload_text(source: str) -> str:
    from app.config import settings
    from app.core.files import extract_text_from_file

    root = Path(settings.upload_dir)
    path = next(
        (candidate for candidate in (root / source, root / "0" / source) if candidate.is_file()),
        root / source,
    )
    return extract_text_from_file(str(path), None)


def _stored_chunks(source: str, tenant_id: int | None) -> list[str]:
    store = _modules()[2]
    where: dict[str, Any] = {"source": source}
    if tenant_id is not None:
        where = {"$and": [{"source": source}, {"tenant_id": str(tenant_id)}]}
    result = store.get_collection().get(where=where)
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    ordered = sorted(zip(ids, documents, strict=True), key=_chunk_index)
    return [str(document) for _, document in ordered]


def _chunk_index(chunk: tuple[str, str]) -> int:
    try:
        return int(chunk[0].rsplit("-", 1)[-1])
    except ValueError:
        return 0


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _pairs(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"(?:^|\s)Q[.:]\s*(.*?)\s*Answer:\s*(.*?)(?=\sQ[.:]\s|\Z)", re.DOTALL)
    pairs = []
    for question, answer in pattern.findall(_normal(text)):
        lead = " ".join(_normal(answer).split()[:_LEAD_WORDS])
        if len(lead.split()) < _LEAD_WORDS:
            continue
        pairs.append({"question": _normal(question), "answer_lead": lead})
    return pairs


def _vectors(texts: list[str]) -> list[list[float]]:
    return _modules()[1].embed_texts(texts)


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def _semantic_ranking(question_vector: list[float], vectors: list[list[float]]) -> list[int]:
    distances = [_cosine_distance(question_vector, vector) for vector in vectors]
    return sorted(range(len(vectors)), key=lambda index: distances[index])


def _hybrid_ranking(
    question: str, question_vector: list[float], chunks: list[str], vectors: list[list[float]]
) -> list[int]:
    from app.services.retrieval import bm25_ranking, fuse_rankings

    semantic = _semantic_ranking(question_vector, vectors)
    return list(fuse_rankings([semantic, bm25_ranking(question, chunks)]))


def _contains(chunk: str, phrase: str) -> bool:
    return phrase.lower() in _normal(chunk).lower()


def _scores(
    chunks: list[str],
    vectors: list[list[float]],
    pairs: list[dict[str, str]],
    question_vectors: list[list[float]],
    hybrid: bool,
) -> dict[str, Any]:
    top1 = 0
    top4 = 0
    answerable = 0
    for pair, question_vector in zip(pairs, question_vectors, strict=True):
        ranking = (
            _hybrid_ranking(pair["question"], question_vector, chunks, vectors)
            if hybrid
            else _semantic_ranking(question_vector, vectors)
        )
        if any(_contains(chunks[index], pair["answer_lead"]) for index in ranking):
            answerable += 1
        if ranking and _contains(chunks[ranking[0]], pair["answer_lead"]):
            top1 += 1
        if any(_contains(chunks[index], pair["answer_lead"]) for index in ranking[:4]):
            top4 += 1
    total = len(pairs)
    return {
        "top1": top1,
        "top4": top4,
        "total": total,
        "answerable": answerable,
        "top1_share": round(top1 / total, 4) if total else None,
        "top4_share": round(top4 / total, 4) if total else None,
    }


def _whole_questions(chunks: list[str], pairs: list[dict[str, str]]) -> dict[str, Any]:
    whole = sum(1 for pair in pairs if any(_contains(chunk, pair["question"]) for chunk in chunks))
    return {
        "whole": whole,
        "total": len(pairs),
        "share": round(whole / len(pairs), 4) if pairs else None,
    }


def _open_ends(chunks: list[str]) -> dict[str, Any]:
    opens = [chunk for chunk in chunks[:-1] if (chunk.rstrip()[-1:] or "") not in '.!?:;"']
    return {"open": len(opens), "boundaries": max(len(chunks) - 1, 0)}


def _report(args: argparse.Namespace) -> dict[str, Any]:
    nlp, _, _ = _modules()
    text = _upload_text(args.source)
    new_chunks = nlp.chunk_text(text)
    old_chunks = _stored_chunks(args.source, None if args.tenant < 0 else args.tenant)
    pairs = _pairs(text)
    old_vectors = _vectors(old_chunks)
    new_vectors = _vectors(new_chunks)
    question_vectors = _vectors([pair["question"] for pair in pairs])
    return {
        "source": args.source,
        "tenant": None if args.tenant < 0 else args.tenant,
        "questions": len(pairs),
        "chunks": {"old": len(old_chunks), "new": len(new_chunks)},
        "whole_questions": {
            "old": _whole_questions(old_chunks, pairs),
            "new": _whole_questions(new_chunks, pairs),
        },
        "chunk_ends": {"old": _open_ends(old_chunks), "new": _open_ends(new_chunks)},
        "retrieval": {
            "old_semantic": _scores(old_chunks, old_vectors, pairs, question_vectors, hybrid=False),
            "old_hybrid": _scores(old_chunks, old_vectors, pairs, question_vectors, hybrid=True),
            "new_semantic": _scores(new_chunks, new_vectors, pairs, question_vectors, hybrid=False),
            "new_hybrid": _scores(new_chunks, new_vectors, pairs, question_vectors, hybrid=True),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="React_JavaScript_NextJS_Interview_QA_Guide.pdf")
    parser.add_argument(
        "--tenant", type=int, default=-1, help="-1 reads chunks stored without a tenant"
    )
    args = parser.parse_args()
    print(json.dumps(_report(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
