"""Public config-first adapter API."""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    AdapterConfigError,
    AdapterEvidenceError,
    AdapterResult,
    ConfigFirstAdapter,
    DefinitionSpec,
    InstanceSpec,
    LayerStripEntry,
    SemanticNodeSpec,
)
from .generic_config import GenericConfigAdapter
from .glm_moe_dsa import GlmMoeDsaAdapter
from .qwen3_5 import Qwen35Adapter
from .registry import AdapterRegistry
from .tiny_dense import TinyDenseAdapter

DEFAULT_REGISTRY = AdapterRegistry(
    (
        TinyDenseAdapter(),
        Qwen35Adapter(),
        GlmMoeDsaAdapter(),
    ),
    fallback=GenericConfigAdapter(),
)


def build_from_config(
    config: Mapping[str, Any],
    *,
    model_id: str = "local",
    revision: str = "local",
    registry: AdapterRegistry = DEFAULT_REGISTRY,
) -> AdapterResult:
    """Build a config-first architecture with the matching registered adapter."""

    return registry.build(config, model_id=model_id, revision=revision)


def build_model_map_from_config(
    config: Mapping[str, Any],
    *,
    model_id: str = "local",
    revision: str = "local",
    registry: AdapterRegistry = DEFAULT_REGISTRY,
):
    """Build and convert a config-first result to strict Model Map IR."""

    return build_from_config(
        config,
        model_id=model_id,
        revision=revision,
        registry=registry,
    ).to_model_map()


__all__ = [
    "AdapterConfigError",
    "AdapterEvidenceError",
    "AdapterRegistry",
    "AdapterResult",
    "ConfigFirstAdapter",
    "DEFAULT_REGISTRY",
    "DefinitionSpec",
    "GenericConfigAdapter",
    "GlmMoeDsaAdapter",
    "InstanceSpec",
    "LayerStripEntry",
    "Qwen35Adapter",
    "SemanticNodeSpec",
    "TinyDenseAdapter",
    "build_from_config",
    "build_model_map_from_config",
]
