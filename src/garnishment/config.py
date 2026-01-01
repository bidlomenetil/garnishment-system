from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "fake"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-nano-2025-08-07"
    openai_reasoning_effort: str | None = "minimal"
    database_url: str = "sqlite+aiosqlite:///./garnishment.db"
    log_level: str = "INFO"
    """Applied to this package's logger only, so raising it to DEBUG never turns on
    a third-party library's debug stream as well."""


def get_settings() -> Settings:
    return Settings()
