import pytest

from rag_pipeline.config import PipelineConfig


def test_configuration_defaults():
    config = PipelineConfig.from_env({})

    assert config.app_env == "development"
    assert config.log_level == "INFO"
    assert config.llm_provider is None
    assert config.embedding_model is None
    assert config.vector_store_provider is None


def test_configuration_reads_optional_provider_neutral_settings():
    config = PipelineConfig.from_env(
        {
            "APP_ENV": "test",
            "LOG_LEVEL": "debug",
            "LLM_PROVIDER": "example-provider",
            "LLM_MODEL": "example-model",
            "EMBEDDING_PROVIDER": "embedding-provider",
            "EMBEDDING_MODEL": "embedding-model",
            "VECTOR_STORE_PROVIDER": "vector-store",
            "VECTOR_STORE_LOCATION": "local-data",
        }
    )

    assert config.app_env == "test"
    assert config.log_level == "DEBUG"
    assert config.llm_provider == "example-provider"
    assert config.llm_model == "example-model"
    assert config.embedding_provider == "embedding-provider"
    assert config.embedding_model == "embedding-model"
    assert config.vector_store_provider == "vector-store"
    assert config.vector_store_location == "local-data"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"APP_ENV": "unknown"}, "APP_ENV"),
        ({"LOG_LEVEL": "TRACE"}, "LOG_LEVEL"),
    ],
)
def test_configuration_rejects_invalid_values(values, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_env(values)


def test_configuration_rejects_blank_optional_values():
    with pytest.raises(ValueError, match="llm_model"):
        PipelineConfig(llm_model=" ")


def test_configuration_uses_process_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    config = PipelineConfig.from_env()

    assert config.app_env == "production"
    assert config.log_level == "WARNING"
