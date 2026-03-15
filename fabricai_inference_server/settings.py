"""
Environment configuration with type safety via pydantic-settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = Field(
        "http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_default_model: str = Field(
        "llama3.2", alias="OLLAMA_DEFAULT_MODEL"
    )

    # Redis
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")

    # Cloud providers (optional — backend activates when key is set)
    anthropic_api_key: str | None = Field(
        None, alias="ANTHROPIC_API_KEY"
    )
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")

    # Routing
    routing_config_path: str = Field(
        "config/routing.yaml", alias="ROUTING_CONFIG_PATH"
    )

    # Auth (disabled by default for local development)
    auth_enabled: bool = Field(False, alias="AUTH_ENABLED")
    api_keys: str = Field("", alias="FABRICAI_API_KEYS")

    # Rate limiting
    rate_limit_rpm: int = Field(60, alias="RATE_LIMIT_RPM")

    # Logging
    log_format: str = Field("text", alias="LOG_FORMAT")  # "text" or "json"
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Concurrency
    queue_max_size: int = Field(10, alias="QUEUE_MAX_SIZE")
    max_concurrent_requests: int = Field(4, alias="MAX_CONCURRENT_REQUESTS")

    @property
    def api_key_set(self) -> set[str]:
        """Parse comma-separated API keys into a set."""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


settings = Settings()
