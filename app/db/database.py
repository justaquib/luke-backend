from collections.abc import Iterator
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="tenant")
    invites = relationship("Invite", back_populates="tenant")


class Invite(Base):
    __tablename__ = "invites"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    email = Column(String, nullable=False)
    token = Column(String, unique=True, nullable=False)
    role = Column(String, nullable=False, default="user")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="invites")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Where the chunk was found: the upload it came from, the characters it
    # covers in the extracted text, the page, the hash of its own text, the
    # fingerprint of its wording, the language it is written in and how much
    # prose it reads as.
    document_id = Column(Integer, nullable=True)
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    page = Column(Integer, nullable=True)
    content_hash = Column(String, nullable=True)
    simhash = Column(BigInteger, nullable=True)
    language = Column(String, nullable=True)
    quality = Column(Float, nullable=True)


class UploadedFile(Base):
    """One file kept on disk, with the hash that recognises the same content.

    The hash makes a repeated upload a no-op and keeps two different files
    from silently replacing each other under one name.
    """

    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class IngestJob(Base):
    """One upload being read, chunked and indexed in the background.

    The row holds what survives the run: the status, the counts and the error.
    The live step and the passages stored so far are published in memory while
    the worker runs, so reporting progress never opens a second write
    transaction.
    """

    __tablename__ = "ingest_jobs"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    step = Column(String, nullable=False, default="")
    progress = Column(Float, nullable=False, default=0.0)
    done = Column(Integer, nullable=False, default=0)
    total = Column(Integer, nullable=False, default=0)
    chunks = Column(Integer, nullable=False, default=0)
    exact_duplicates = Column(Integer, nullable=False, default=0)
    near_duplicates = Column(Integer, nullable=False, default=0)
    error = Column(String, nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)


class WebSource(Base):
    """One allowlisted page the engine may read and re-read.

    The domain is what the policy checks; the URL is what refresh reads.
    """

    __tablename__ = "web_sources"

    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class WebSnapshot(Base):
    """One page as it was read, so a citation can be reproduced later.

    ``status`` is ``indexed`` for an allowlisted page in the unverified web
    collection, ``candidate`` for a general page waiting for the trainer, and
    ``superseded`` for an older version of a page that has since changed.
    """

    __tablename__ = "web_snapshots"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("web_sources.id"), nullable=True)
    url = Column(String, nullable=False)
    title = Column(String, nullable=False, default="")
    text = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)
    trust = Column(String, nullable=False, default="general")
    status = Column(String, nullable=False, default="candidate")
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    checked_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AssessmentRun(Base):
    """One pass over the item bank with one checkpoint in one mode.

    A score means nothing without the thing that produced it, so the row keeps
    the checkpoint, tokenizer and generation settings, the bank version and
    salt, and whether retrieval was disabled (``unaided``) or available
    (``assisted``).
    """

    __tablename__ = "assessment_runs"

    id = Column(Integer, primary_key=True)
    mode = Column(String, nullable=False, default="unaided")
    status = Column(String, nullable=False, default="queued")
    step = Column(String, nullable=False, default="")
    progress = Column(Float, nullable=False, default=0.0)
    done = Column(Integer, nullable=False, default=0)
    total = Column(Integer, nullable=False, default=0)
    model_id = Column(String, nullable=False, default="")
    model_artifact = Column(String, nullable=False, default="")
    tokenizer_id = Column(String, nullable=False, default="")
    generation = Column(Text, nullable=False, default="{}")
    bank_version = Column(Integer, nullable=False, default=0)
    bank_salt = Column(String, nullable=False, default="")
    error = Column(String, nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    results = relationship("AssessmentResult", back_populates="run", cascade="all, delete-orphan")


class AssessmentResult(Base):
    """One item answered, scored by rule or held for a trainer's rubric.

    ``score`` stays null while a rubric item waits for review: a null score is
    unassessed, never zero. A deterministic item stores its score and whether
    the answer was correct; a rubric item stores the trainer's score once.
    """

    __tablename__ = "assessment_results"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("assessment_runs.id"), nullable=False, index=True)
    item_id = Column(String, nullable=False, index=True)
    skill = Column(String, nullable=False, index=True)
    split = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    stage = Column(Integer, nullable=False, default=1)
    topic = Column(String, nullable=False, default="")
    response = Column(Text, nullable=True)
    method = Column(String, nullable=False, default="")
    score = Column(Float, nullable=True)
    correct = Column(Boolean, nullable=True)
    latency_ms = Column(Float, nullable=True)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_note = Column(Text, nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("AssessmentRun", back_populates="results")


class Faq(Base):
    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="pending")
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    confidence = Column(Float, nullable=False, default=settings.base_answer_confidence)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PendingQuestion(Base):
    __tablename__ = "pending_questions"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    note = Column(Text, nullable=True)
    occurrences = Column(Integer, default=1, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Intent(Base):
    __tablename__ = "intents"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    response_template = Column(Text, nullable=False)
    faq_id = Column(Integer, nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    rating = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    google_id = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    picture = Column(String, nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)
    role = Column(String, nullable=False, default="user")
    is_systemadmin = Column(Boolean, default=False, nullable=False)
    pin_hash = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="users")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
