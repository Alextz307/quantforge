"""Webapp runtime configuration loaded from the environment."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebappEnv(StrEnum):
    DEVELOPMENT = "development"
    LOCAL = "local"


DEFAULT_SESSION_TTL_MINUTES = 12 * 60


class WebappSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEBAPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    env: WebappEnv = WebappEnv.LOCAL
    store_root: Path = Path("experiment_results")
    config_root: Path = Path("config")
    db_path: Path = Path("webapp/data/webapp.sqlite")
    secret_key: str = ""
    session_ttl_minutes: int = DEFAULT_SESSION_TTL_MINUTES
    job_temp_dir: Path = Path("webapp/data/jobs")
    study_spec_uploads_dir: Path = Path("webapp/data/study_specs")
    universe_spec_uploads_dir: Path = Path("webapp/data/universe_specs")


@lru_cache
def get_settings() -> WebappSettings:
    return WebappSettings()
