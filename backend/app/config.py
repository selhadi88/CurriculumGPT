from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root regardless of where the process is launched from
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+psycopg2://gurriculum:password@127.0.0.1:5433/gurriculum_db"

    # Model
    model_type: str = "bge"
    model_cache_dir: str = ".model_cache"
    model_checkpoint: Optional[str] = None
    device: Optional[str] = None

    # API security
    secret_key: str = "change-me-to-a-long-random-string"
    # Stored as plain str; parsed into a list by the property below.
    # Accepts both JSON array and comma-separated values in .env.
    cors_origins_raw: str = "http://localhost:5173,http://localhost:3000"
    # Optional regex alternative — e.g. r"https://.*\.onrender\.com" lets any
    # Render subdomain call the API without hardcoding the frontend URL.
    cors_origin_regex: str = ""

    # External APIs
    onet_api_key: str = ""
    usajobs_api_key: str = ""
    usajobs_email: str = "research@example.com"

    # App
    log_level: str = "INFO"
    workers: int = 1
    api_v1_prefix: str = "/api/v1"

    @property
    def sqlalchemy_url(self) -> str:
        """Normalize managed-host DSNs (Render/Railway/Heroku hand out
        ``postgres://`` or ``postgresql://``) to the psycopg2 driver SQLAlchemy
        2.0 expects. A DSN that already names a driver is left untouched."""
        url = self.database_url.strip()
        if url.startswith("postgres://"):
            url = "postgresql+psycopg2://" + url[len("postgres://") :]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://") :]
        return url

    @property
    def cors_origins(self) -> list[str]:
        val = self.cors_origins_raw.strip()
        if val.startswith("["):
            return json.loads(val)
        return [o.strip() for o in val.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
