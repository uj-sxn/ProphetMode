from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM (via OpenRouter)
    openrouter_api_key: str = ""
    anthropic_model: str = "anthropic/claude-sonnet-4-5"

    # Web search
    tavily_api_key: str = ""

    # Prophet Arena
    pa_server_api_key: str = ""
    pa_server_url: str = "https://api.aiprophet.dev"

    # Server
    port: int = 8000

    # Pipeline tuning
    max_concurrent_events: int = 5
    search_timeout: float = 12.0
    llm_timeout: float = 90.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
