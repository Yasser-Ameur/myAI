import pytest

from companion.core.errors import ConfigurationError
from companion.infrastructure.models.embeddings.onnx import MockEmbeddingProvider
from companion.infrastructure.models.factory import build_provider
from companion.infrastructure.models.llm.mock import MockLLMProvider
from companion.infrastructure.models.registry import ModelRegistry
from companion.infrastructure.models.router import TaskRouter


def test_registry_builds_mock_slots():
    registry = ModelRegistry({
        "llm": {"default": {"provider": "mock", "model_id": "qwen3-1.7b",
                            "script": [{"match": "", "response": "ok"}]}},
        "embeddings": {"default": {"provider": "mock", "model_id": "bge-mock"}},
    })
    registry.build_from_config(build_provider)
    assert registry.has("llm.default")
    assert isinstance(registry.get("llm.default"), MockLLMProvider)
    assert isinstance(registry.get_optional("embeddings.default"), MockEmbeddingProvider)


def test_unknown_provider_raises():
    registry = ModelRegistry({"llm": {"default": {"provider": "nope"}}})
    with pytest.raises(ConfigurationError):
        registry.build_from_config(build_provider)


def test_load_fallback_for_missing_dependency(monkeypatch):
    """When a provider's dependency is unavailable, build must fall back to mock.

    Simulates the missing dependency regardless of what is installed on the
    machine, so the fallback machinery stays testable in any environment.
    """
    from companion.core.errors import ProviderNotAvailableError
    from companion.infrastructure.models.llm.llama_cpp import LlamaCppProvider

    monkeypatch.setattr(
        LlamaCppProvider,
        "__init__",
        lambda self, config, model_id="": (_ for _ in ()).throw(
            ProviderNotAvailableError("llama-cpp-python is not installed")
        ),
    )
    registry = ModelRegistry({
        "llm": {"default": {"provider": "llama_cpp", "model_id": "x"}},
    })
    registry.build_from_config(build_provider)
    provider = registry.get("llm.default")
    assert isinstance(provider, MockLLMProvider)
    assert registry.meta("llm.default").get("fallback")


def test_ensure_loaded_works_on_fallback():
    registry = ModelRegistry({
        "llm": {"default": {"provider": "mock", "model_id": "m"}},
    })
    registry.build_from_config(build_provider)
    provider = registry.ensure_loaded("llm.default")
    assert provider.is_loaded()


def test_router_routes_to_fast_when_available():
    registry = ModelRegistry({
        "llm": {"default": {"provider": "mock"}, "fast": {"provider": "mock"}},
    })
    registry.build_from_config(build_provider)
    router = TaskRouter(registry)
    assert router.llm_for_task("classification") == "llm.fast"
    assert router.llm_for_task("chat") == "llm.default"


def test_router_routes_reasoning_keywords():
    registry = ModelRegistry({
        "llm": {"default": {"provider": "mock"}, "reasoning": {"provider": "mock"}},
    })
    registry.build_from_config(build_provider)
    router = TaskRouter(registry)
    assert router.llm_for_task("reasoning", "compare these two options") == "llm.reasoning"
    assert router.llm_for_task("reasoning", "hello") == "llm.default"
