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
    faq_match_threshold: float = 0.55
    intent_match_threshold: float = 0.75
    intent_merge_threshold: float = 0.35
    base_answer_confidence: float = 0.5
    max_answer_confidence: float = 0.95
    confidence_step: float = 0.05
    # The confidence above which a correction is kept for review instead of
    # being used at once.
    correction_review_confidence: float = 0.9
    spacy_model: str = "en_core_web_sm"
    cors_origins: str = "http://localhost:4200"
    upload_dir: str = "./uploads"
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
