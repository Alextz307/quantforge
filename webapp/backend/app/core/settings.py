"""Webapp runtime configuration loaded from the environment."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebappEnv(StrEnum):
    DEVELOPMENT = "development"
    LOCAL = "local"


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


@lru_cache
def get_settings() -> WebappSettings:
    return WebappSettings()
