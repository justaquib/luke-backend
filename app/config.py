import os

from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "True")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5433/assistance"
    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "docs"
    chroma_faq_collection: str = "faq"
    chroma_intents_collection: str = "intents"
    # Official web pages, stored unverified; never mixed with the taught corpus.
    chroma_web_collection: str = "web"
    # Cosine distances, the space the rebuilt collections use. The saved
    # feedback has one positive, so these translate the old L2 limits and stay
    # below its lowest negative (0.4455); recalibrate when more labels exist.
    faq_match_threshold: float = 0.28
    intent_match_threshold: float = 0.36
    intent_merge_threshold: float = 0.35
    # An extra cross-encoder pass over the fused passages. Off by default: it
    # downloads a model and is slower, and its score is only used for order.
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    base_answer_confidence: float = 0.5
    max_answer_confidence: float = 0.95
    confidence_step: float = 0.05
    # The confidence above which a correction is kept for review instead of
    # being used at once.
    correction_review_confidence: float = 0.9
    spacy_model: str = "en_core_web_sm"
    # The local, read-only copy of Luke's base checkpoint. It is never packaged,
    # merged or fine-tuned in place; `scripts/install_foundation.py` installs it.
    foundation_model_dir: str = "./.models/foundation"
    foundation_verify_weights: bool = True
    foundation_max_new_tokens: int = 180
    foundation_temperature: float = 0.7
    foundation_top_p: float = 0.9
    foundation_repetition_penalty: float = 1.1
    # How much model memory the process keeps, measured by the files of the
    # loaded models. The least recently used model is let go once it is
    # passed, so visiting several workspaces neither reloads each one nor
    # grows without limit.
    weights_cache_bytes: int = 1536 * 1024 * 1024
    cors_origins: str = "http://localhost:4200"
    upload_dir: str = "./uploads"
    # One uploaded file may not exceed this many bytes. The limit is enforced
    # while reading the upload, before anything is written to disk.
    max_upload_bytes: int = 25 * 1024 * 1024
    # Off by default: when on, emails, phone numbers and long identifiers are
    # masked in uploaded and pasted text before it is chunked and indexed.
    pii_scrub_enabled: bool = False
    # How many uploads one process reads at once. The queue is bounded, so a
    # burst of uploads waits instead of exhausting the memory.
    ingest_workers: int = 2
    # A package carries the model and material, so it is allowed to be larger
    # than a document. The same rule applies: the limit stops the read.
    max_package_bytes: int = 4 * 1024 * 1024 * 1024
    # Reading public pages as evidence. The size limit stops the read while it
    # streams, the redirect cap keeps every hop public, and the timeout bounds
    # one request. The search key comes from the environment, never settings.
    web_max_bytes: int = 2 * 1024 * 1024
    web_timeout_seconds: float = 10.0
    web_max_redirects: int = 4
    web_search_api_key: str = ""
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    systemadmin_email: str = "systemadmin@example.com"
    systemadmin_pin: str = "123456"
    systemadmin_pin_hash: str = ""
    invite_expire_hours: int = 72

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
