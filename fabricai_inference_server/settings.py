"""
settings.py

Env settings with type guard for /fabricai_inference_server.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    local_model_dir: str = Field(..., env="LOCAL_MODEL_DIR")
    llm_model: str = Field(..., env="LLM_MODEL")
    queue_max_size: int = Field(10, env="QUEUE_MAX_SIZE")
    max_concurrent_requests: int = Field(2, env="MAX_CONCURRENT_REQUESTS")
    llama_threads: int = Field(8, env="LLAMA_THREADS")
    llama_ctx: int = Field(2048, env="LLAMA_CTX")
    llama_gpu_layers: int = Field(0, env="LLAMA_GPU_LAYERS")
    redis_host: str = Field("redis", alias="REDIS_HOST")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
