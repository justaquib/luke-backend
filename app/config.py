from pydantic_settings import BaseSettings, SettingsConfigDict
import os

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
    spacy_model: str = "en_core_web_sm"
    cors_origins: str = "http://localhost:4200"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()