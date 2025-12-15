import logging
from typing import Dict, Type, List
from .base import Engine
from .llama import LlamaCppEngine
from .huggingface import HuggingFaceEngine
from .vllm import VLLMEngine
from .vllm_dp import VLLMDataParallelEngine

logger = logging.getLogger(__name__)

# 1. Define a mapping of engine-name → Engine subclass
_engine_registry: Dict[str, Type[Engine]] = {
    "llama_cpp": LlamaCppEngine,
    "huggingface": HuggingFaceEngine,
    "vllm": VLLMEngine,
    "vllm_dp": VLLMDataParallelEngine,
}


def get_engine(engine_name: str, model_name: str, **kwargs) -> Engine:
    """
    Factory function to instantiate an inference engine.
    Raises ValueError for unknown/unavailable engines, RuntimeError on instantiation failure.
    """
    logger.info(f"Attempting to get engine: {engine_name}")

    engine_cls = _engine_registry.get(engine_name)
    if engine_cls is None:
        raise ValueError(
            f"Unknown engine '{engine_name}'. "
            f"Available choices: {list(_engine_registry.keys())}"
        )

    if not engine_cls.is_available():
        raise ValueError(f"Engine '{engine_name}' is not available on this system.")

    try:
        instance = engine_cls(model_name=model_name, **kwargs)
        logger.info(f"Successfully instantiated engine: {engine_name}")
        return instance
    except Exception as e:
        logger.error(
            f"Failed to instantiate engine '{engine_name}': {e}", exc_info=True
        )
        raise RuntimeError(
            f"Failed to create instance of engine '{engine_name}'"
        ) from e


def list_available_engines() -> List[str]:
    """Returns a list of names of currently available engines."""
    return [name for name, cls in _engine_registry.items() if cls.is_available()]


def is_engine_available(engine_name: str) -> bool:
    """Checks if a named engine is available."""
    cls = _engine_registry.get(engine_name)
    return bool(cls and cls.is_available())
