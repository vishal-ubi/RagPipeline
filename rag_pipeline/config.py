"""Validated, provider-neutral application configuration."""

import os
from dataclasses import dataclass
from typing import Mapping


_APPLICATION_ENVIRONMENTS = {"development", "test", "staging", "production"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _optional_env_value(values: Mapping[str, str], name: str) -> str | None:
    value = values.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    app_env: str = "development"
    log_level: str = "INFO"
    llm_provider: str | None = None
    llm_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    vector_store_provider: str | None = None
    vector_store_location: str | None = None

    def __post_init__(self) -> None:
        if self.app_env not in _APPLICATION_ENVIRONMENTS:
            allowed = ", ".join(sorted(_APPLICATION_ENVIRONMENTS))
            raise ValueError(f"APP_ENV must be one of: {allowed}")
        if self.log_level not in _LOG_LEVELS:
            allowed = ", ".join(sorted(_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")

        optional_fields = (
            ("llm_provider", self.llm_provider),
            ("llm_model", self.llm_model),
            ("embedding_provider", self.embedding_provider),
            ("embedding_model", self.embedding_model),
            ("vector_store_provider", self.vector_store_provider),
            ("vector_store_location", self.vector_store_location),
        )
        for name, value in optional_fields:
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "PipelineConfig":
        values = os.environ if environ is None else environ
        app_env = values.get("APP_ENV", "development").strip()
        log_level = values.get("LOG_LEVEL", "INFO").strip().upper()

        return cls(
            app_env=app_env,
            log_level=log_level,
            llm_provider=_optional_env_value(values, "LLM_PROVIDER"),
            llm_model=_optional_env_value(values, "LLM_MODEL"),
            embedding_provider=_optional_env_value(values, "EMBEDDING_PROVIDER"),
            embedding_model=_optional_env_value(values, "EMBEDDING_MODEL"),
            vector_store_provider=_optional_env_value(values, "VECTOR_STORE_PROVIDER"),
            vector_store_location=_optional_env_value(values, "VECTOR_STORE_LOCATION"),
        )
