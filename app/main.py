from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import init_db, get_session, SessionLocal, Faq, PendingQuestion, Intent
from app.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QueryHit,
    StatsResponse,
    FaqCreate,
    FaqUpdate,
    FaqStatusUpdate,
    FaqOut,
    PendingOut,
    PendingLog,
    IntentCreate,
    IntentOut,
    IntentUtterance,
    MergePreview,
    MergeMatch,
    MergeRequest,
    MergeResult,
    FaqSuggestion,
    PendingSuggestion,
    IntentSuggestions,
    FaqMergePreview,
    FaqMergeRequest,
    FaqMergeResult,
)
from app.services.ingestion import ingest_text
from app.services import vector_store, nlp, apps


app = FastAPI(title="Assistance RAG API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    _resync_approved_faqs_to_chroma()


def _resync_approved_faqs_to_chroma() -> None:
    """On startup, make sure every 'approved' FAQ is in the Chroma faq collection.

    Postgres is the source of truth; Chroma is rebuilt from it so that we recover
    after wiping .chroma or moving to a new machine.
    """
    with SessionLocal() as db:
        approved = db.execute(
            select(Faq).where(Faq.status == "approved")
        ).scalars().all()
        for faq in approved:
            vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)

        pending = db.execute(
            select(PendingQuestion).where(PendingQuestion.resolved == False)  # noqa: E712
        ).scalars().all()
        for p in pending:
            vector_store.index_pending(p.id, p.question)

        intents = db.execute(select(Intent)).scalars().all()
        for intent in intents:
            vector_store.upsert_intent(
                intent.id, intent.name, intent.response_template, intent.faq_id
            )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, db: Session = Depends(get_session)):
    return ingest_text(db, req.source, req.text)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_session)):
    ents = nlp.extract_entities(req.question)
    apps_cfg = req.apps or {}
    location = req.location or {}
    detected = apps.detect_intents(req.question)
    disabled_external_sources: list[str] = []
    if "weather" in detected and not apps_cfg.get("weather"):
        disabled_external_sources.append("weather")
    if "wiki" in detected and not apps_cfg.get("wiki"):
        disabled_external_sources.append("wiki")

    faq_hits = vector_store.search_faq(req.question, n_results=1)
    if faq_hits:
        best = faq_hits[0]
        if best["distance"] is not None and best["distance"] <= settings.faq_match_threshold:
            return QueryResponse(
                question=req.question,
                source="faq",
                answer=best["answer"],
                hits=[QueryHit(text=best["question"], metadata={"source": "faq", "faq_id": best["faq_id"]}, distance=best["distance"])],
                entities=ents,
                disabled_external_sources=disabled_external_sources or None,
            )

    loose = vector_store.search_faq_loose(req.question, n_results=3)
    intent_hit: dict | None = None
    if loose:
        candidate = loose[0]
        d = candidate.get("distance")
        if (
            d is not None
            and settings.faq_match_threshold < d <= settings.intent_match_threshold
        ):
            intent_hit = {
                "matched_question": candidate.get("question"),
                "matched_faq_id": candidate.get("faq_id"),
                "distance": d,
                "answer": candidate.get("answer"),
            }

    if intent_hit is not None:
        return QueryResponse(
                question=req.question,
                source="intent",
                answer=intent_hit["answer"],
                hits=[
                    QueryHit(
                        text=intent_hit["matched_question"],
                        metadata={
                            "source": "faq",
                            "faq_id": intent_hit["matched_faq_id"],
                            "via": "intent",
                        },
                        distance=intent_hit["distance"],
                    )
                ],
                entities=ents,
                intent=intent_hit,
                disabled_external_sources=disabled_external_sources or None,
            )

    doc_hits = vector_store.query(req.question, n_results=req.n_results)
    if doc_hits:
        return QueryResponse(
                question=req.question,
                source="docs",
                answer=doc_hits[0]["text"],
                hits=[QueryHit(**h) for h in doc_hits],
                entities=ents,
                disabled_external_sources=disabled_external_sources or None,
            )

    if apps_cfg.get("weather") and "weather" in detected:
        lat = location.get("lat")
        lon = location.get("lon")
        result = apps.weather_for(req.question, lat=lat, lon=lon)
        if result:
            return QueryResponse(
                question=req.question,
                source="weather",
                answer=result["answer"],
                hits=[],
                entities=ents,
                app=result["payload"],
                disabled_external_sources=disabled_external_sources or None,
            )
    if apps_cfg.get("wiki") and "wiki" in detected:
        result = apps.wiki_summary(req.question)
        if result:
            return QueryResponse(
                question=req.question,
                source="wiki",
                answer=result["answer"],
                hits=[
                    QueryHit(
                        text=result["payload"].get("title", req.question),
                        metadata={
                            "source": "wiki",
                            "url": result["payload"].get("url"),
                        },
                        distance=None,
                    )
                ],
                entities=ents,
                app=result["payload"],
                disabled_external_sources=disabled_external_sources or None,
            )

    if disabled_external_sources:
        parts = []
        if "weather" in disabled_external_sources:
            parts.append("Weather")
        if "wiki" in disabled_external_sources:
            parts.append("Wikipedia")
        joined = " and ".join(parts)
        return QueryResponse(
            question=req.question,
            source="pending",
            answer=f"I don't have a trained answer for that yet. Enable {joined} in Settings to get live answers from your location or the web.",
            hits=[],
            entities=ents,
            pending_id=None,
            disabled_external_sources=disabled_external_sources or None,
        )

    pending = _record_pending(db, req.question)
    return QueryResponse(
        question=req.question,
        source="pending",
        answer="I don't have an answer for that yet. A trainer has been notified.",
        hits=[],
        entities=ents,
        pending_id=pending.id,
        disabled_external_sources=disabled_external_sources or None,
    )


@app.post("/faq", response_model=FaqOut)
def create_faq(req: FaqCreate, db: Session = Depends(get_session)):
    faq = Faq(
        question=req.question.strip(),
        answer=req.answer.strip(),
        status="pending",
    )
    db.add(faq)
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)
    return faq


@app.get("/faq", response_model=list[FaqOut])
def list_faqs(status: str | None = None, db: Session = Depends(get_session)):
    stmt = select(Faq).order_by(Faq.updated_at.desc())
    if status:
        stmt = stmt.where(Faq.status == status)
    return db.execute(stmt).scalars().all()


@app.patch("/faq/{faq_id}", response_model=FaqOut)
def update_faq(faq_id: int, req: FaqUpdate, db: Session = Depends(get_session)):
    faq = db.get(Faq, faq_id)
    if not faq:
        raise HTTPException(404, "FAQ not found")
    faq.answer = req.answer.strip()
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)
    return faq


@app.patch("/faq/{faq_id}/status", response_model=FaqOut)
def update_faq_status(faq_id: int, req: FaqStatusUpdate, db: Session = Depends(get_session)):
    allowed = {"pending", "approved", "declined", "duplicate"}
    if req.status not in allowed:
        raise HTTPException(400, f"status must be one of {sorted(allowed)}")
    faq = db.get(Faq, faq_id)
    if not faq:
        raise HTTPException(404, "FAQ not found")
    faq.status = req.status
    db.commit()
    db.refresh(faq)
    if faq.status == "approved":
        vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)
    else:
        vector_store.delete_faq(faq.id)
    return faq


@app.delete("/faq/{faq_id}")
def remove_faq(faq_id: int, db: Session = Depends(get_session)):
    faq = db.get(Faq, faq_id)
    if not faq:
        raise HTTPException(404, "FAQ not found")
    db.delete(faq)
    db.commit()
    vector_store.delete_faq(faq_id)
    return {"deleted": faq_id}


@app.get("/faq/inbox", response_model=list[PendingOut])
def list_inbox(resolved: bool = False, db: Session = Depends(get_session)):
    return db.execute(
        select(PendingQuestion)
        .where(PendingQuestion.resolved == resolved)
        .order_by(PendingQuestion.occurrences.desc(), PendingQuestion.last_seen_at.desc())
    ).scalars().all()


@app.post("/faq/inbox/{pending_id}/answer", response_model=FaqOut)
def answer_pending(pending_id: int, req: FaqUpdate, db: Session = Depends(get_session)):
    pending = db.get(PendingQuestion, pending_id)
    if not pending:
        raise HTTPException(404, "Pending question not found")
    faq = Faq(question=pending.question, answer=req.answer.strip(), status="pending")
    db.add(faq)
    pending.resolved = True
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)
    vector_store.delete_pending_index(pending_id)
    return faq


@app.post("/faq/log", response_model=PendingOut)
def log_question(req: PendingLog, db: Session = Depends(get_session)):
    return _record_pending(db, req.question)


@app.delete("/faq/inbox/{pending_id}")
def delete_pending(pending_id: int, db: Session = Depends(get_session)):
    pending = db.get(PendingQuestion, pending_id)
    if not pending:
        raise HTTPException(404, "Pending question not found")
    db.delete(pending)
    db.commit()
    vector_store.delete_pending_index(pending_id)
    return {"deleted": pending_id}


@app.get("/faq/pending/{pending_id}/merge-preview", response_model=MergePreview)
def merge_preview(pending_id: int, db: Session = Depends(get_session)):
    canonical = db.get(PendingQuestion, pending_id)
    if not canonical:
        raise HTTPException(404, "Pending question not found")
    candidates = db.execute(
        select(PendingQuestion).where(
            PendingQuestion.resolved == False,  # noqa: E712
            PendingQuestion.id != canonical.id,
        )
    ).scalars().all()
    candidate_texts = [c.question for c in candidates]
    if not candidate_texts:
        return MergePreview(
            canonical_id=canonical.id,
            canonical_question=canonical.question,
            answer="",
            matches=[],
        )
    hits = vector_store.search_pending(
        canonical.question, candidate_texts, n_results=max(50, len(candidate_texts))
    )
    by_question = {h["question"]: h for h in hits}
    matches: list[MergeMatch] = []
    for c in candidates:
        h = by_question.get(c.question)
        if not h or h["distance"] is None:
            continue
        if h["distance"] <= settings.intent_merge_threshold:
            matches.append(
                MergeMatch(
                    pending_id=c.id,
                    question=c.question,
                    occurrences=c.occurrences,
                    distance=h["distance"],
                )
            )
    matches.sort(key=lambda m: m.distance)
    return MergePreview(
        canonical_id=canonical.id,
        canonical_question=canonical.question,
        answer="",
        matches=matches,
    )


@app.post("/faq/pending/{pending_id}/merge", response_model=MergeResult)
def merge_pending(
    pending_id: int,
    req: MergeRequest,
    db: Session = Depends(get_session),
):
    canonical = db.get(PendingQuestion, pending_id)
    if not canonical:
        raise HTTPException(404, "Pending question not found")
    answer = req.answer.strip()
    if not answer:
        raise HTTPException(400, "answer is required")

    targets = []
    if req.pending_ids:
        for tid in req.pending_ids:
            t = db.get(PendingQuestion, tid)
            if t and not t.resolved and t.id != canonical.id:
                targets.append(t)

    faq = Faq(
        question=canonical.question.strip(),
        answer=answer,
        status="pending",
    )
    db.add(faq)
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)

    created = 1
    for t in targets:
        alias = Faq(
            question=t.question.strip(),
            answer=answer,
            status="pending",
        )
        db.add(alias)
        db.flush()
        vector_store.upsert_faq(alias.id, alias.question, alias.answer, alias.status)
        t.resolved = True
        vector_store.delete_pending_index(t.id)
        created += 1

    canonical.resolved = True
    vector_store.delete_pending_index(canonical.id)
    db.commit()
    return MergeResult(
        canonical_faq_id=faq.id,
        created=created,
        resolved=len(targets) + 1,
    )


@app.get("/faq/intent-suggestions", response_model=IntentSuggestions)
def intent_suggestions(question: str, db: Session = Depends(get_session)):
    q = question.strip()
    faq_hits = vector_store.search_faq_loose(q, n_results=10)
    pending_hits = vector_store.search_pending_all(q, n_results=10)
    return IntentSuggestions(
        query=q,
        faqs=[
            FaqSuggestion(
                faq_id=h["faq_id"],
                question=h["question"],
                answer=h.get("answer", ""),
                status=h.get("status") or "unknown",
                distance=h.get("distance") if h.get("distance") is not None else 1.0,
            )
            for h in faq_hits
        ],
        pending=[
            PendingSuggestion(
                pending_id=h["pending_id"],
                question=h["question"],
                occurrences=next(
                    (p.occurrences for p in db.execute(
                        select(PendingQuestion).where(PendingQuestion.id == h["pending_id"])
                    ).scalars() if p),
                    1,
                ),
                distance=h.get("distance") if h.get("distance") is not None else 1.0,
            )
            for h in pending_hits
        ],
    )


@app.get("/faq/{faq_id}/merge-preview", response_model=FaqMergePreview)
def faq_merge_preview(faq_id: int, db: Session = Depends(get_session)):
    canonical = db.get(Faq, faq_id)
    if not canonical:
        raise HTTPException(404, "FAQ not found")
    others = db.execute(
        select(Faq).where(Faq.id != canonical.id)
    ).scalars().all()
    other_ids = [o.id for o in others]
    if not other_ids:
        return FaqMergePreview(
            canonical_id=canonical.id,
            canonical_question=canonical.question,
            canonical_answer=canonical.answer,
            matches=[],
        )
    hits = vector_store.search_faq_in(canonical.question, other_ids, n_results=max(50, len(other_ids)))
    matches: list[FaqSuggestion] = []
    for h in hits:
        if h.get("distance") is None or h["distance"] > settings.intent_merge_threshold:
            continue
        matches.append(
            FaqSuggestion(
                faq_id=h["faq_id"],
                question=h["question"],
                answer=h.get("answer", ""),
                status=h.get("status") or "unknown",
                distance=h["distance"],
            )
        )
    matches.sort(key=lambda m: m.distance)
    return FaqMergePreview(
        canonical_id=canonical.id,
        canonical_question=canonical.question,
        canonical_answer=canonical.answer,
        matches=matches,
    )


@app.post("/faq/{faq_id}/merge", response_model=FaqMergeResult)
def faq_merge(
    faq_id: int,
    req: FaqMergeRequest,
    db: Session = Depends(get_session),
):
    canonical = db.get(Faq, faq_id)
    if not canonical:
        raise HTTPException(404, "FAQ not found")
    absorbed = 0
    recreated = 0

    target_id = req.target_id if req.target_id else canonical.id
    target = db.get(Faq, target_id)
    if not target:
        raise HTTPException(404, "Target FAQ not found")

    if req.answer is not None:
        new_answer = req.answer.strip()
        if not new_answer:
            raise HTTPException(400, "answer cannot be empty")
        if target.id == canonical.id:
            target.answer = new_answer
        else:
            canonical.answer = new_answer
            db.flush()

    db.flush()
    target_id_final = target.id
    target_obj = db.get(Faq, target_id_final)
    final_answer = target_obj.answer

    absorbed_questions: list[str] = []
    for fid in req.faq_ids:
        other = db.get(Faq, fid)
        if not other or other.id == target_id_final:
            continue
        absorbed_questions.append(other.question)
        db.delete(other)
        vector_store.delete_faq(other.id)
        absorbed += 1

    db.commit()
    db.refresh(target_obj)
    vector_store.upsert_faq(target_obj.id, target_obj.question, target_obj.answer, target_obj.status)

    for q in absorbed_questions:
        alias = Faq(
            question=q.strip(),
            answer=final_answer,
            status="pending",
        )
        db.add(alias)
        db.flush()
        vector_store.upsert_faq(alias.id, alias.question, alias.answer, alias.status)
        recreated += 1
    db.commit()

    return FaqMergeResult(
        canonical_faq_id=target_id_final,
        absorbed=absorbed,
        recreated=recreated,
    )


@app.post("/intents", response_model=IntentOut)
def create_intent(req: IntentCreate, db: Session = Depends(get_session)):
    intent = Intent(
        name=req.name.strip(),
        response_template=req.response_template.strip(),
        faq_id=req.faq_id,
    )
    db.add(intent)
    db.commit()
    db.refresh(intent)
    vector_store.upsert_intent(intent.id, intent.name, intent.response_template, intent.faq_id)
    return intent


@app.get("/intents", response_model=list[IntentOut])
def list_intents(db: Session = Depends(get_session)):
    return db.execute(select(Intent).order_by(Intent.updated_at.desc())).scalars().all()


@app.patch("/intents/{intent_id}", response_model=IntentOut)
def update_intent(intent_id: int, req: IntentCreate, db: Session = Depends(get_session)):
    intent = db.get(Intent, intent_id)
    if not intent:
        raise HTTPException(404, "Intent not found")
    intent.name = req.name.strip()
    intent.response_template = req.response_template.strip()
    intent.faq_id = req.faq_id
    db.commit()
    db.refresh(intent)
    vector_store.upsert_intent(intent.id, intent.name, intent.response_template, intent.faq_id)
    return intent


@app.delete("/intents/{intent_id}")
def remove_intent(intent_id: int, db: Session = Depends(get_session)):
    intent = db.get(Intent, intent_id)
    if not intent:
        raise HTTPException(404, "Intent not found")
    db.delete(intent)
    db.commit()
    vector_store.delete_intent(intent_id)
    return {"deleted": intent_id}


@app.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_session)):
    faq_count = vector_store.faq_size()
    pending_count = db.execute(
        select(func.count(PendingQuestion.id)).where(PendingQuestion.resolved == False)  # noqa: E712
    ).scalar_one()
    return StatsResponse(
        chunks=vector_store.collection_size(),
        faqs=faq_count,
        pending=pending_count,
    )


def _record_pending(db: Session, question: str) -> PendingQuestion:
    q = question.strip()
    existing = db.execute(
        select(PendingQuestion)
        .where(PendingQuestion.question == q, PendingQuestion.resolved == False)  # noqa: E712
    ).scalar_one_or_none()
    if existing:
        existing.occurrences += 1
        existing.last_seen_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        vector_store.index_pending(existing.id, existing.question)
        return existing
    pending = PendingQuestion(question=q)
    db.add(pending)
    db.commit()
    db.refresh(pending)
    vector_store.index_pending(pending.id, pending.question)
    return pending