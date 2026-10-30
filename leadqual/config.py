from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # don't blow up on unexpected vars
    )

    # ── Required ────────────────────────────────────────────────
    MODEL_API_KEY: str            # no default → must be set in .env or env vars

    # ── Optional (sensible Groq defaults) ───────────────────────
    MODEL_NAME: str = "llama-3.1-70b-versatile"
    MODEL_TIMEOUT_SECONDS: float = 10.0
    MODEL_MAX_RETRIES: int = 1


# Singleton — import this everywhere you need config
settings = Settings()