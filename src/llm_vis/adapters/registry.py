"""Adapter registration and deterministic model-type selection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

from .base import AdapterConfigError, AdapterResult, ConfigFirstAdapter


class AdapterRegistry:
    """Registry keyed by Hugging Face ``model_type`` values."""

    def __init__(self, adapters: Iterable[ConfigFirstAdapter] = ()) -> None:
        self._by_model_type: Dict[str, ConfigFirstAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    @property
    def model_types(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_model_type))

    @property
    def adapters(self) -> Tuple[ConfigFirstAdapter, ...]:
        unique = {id(adapter): adapter for adapter in self._by_model_type.values()}
        return tuple(sorted(unique.values(), key=lambda adapter: adapter.name))

    def register(self, adapter: ConfigFirstAdapter, *, replace: bool = False) -> None:
        if not adapter.model_types:
            raise ValueError(f"Adapter {adapter.name!r} does not declare any model types")
        for model_type in adapter.model_types:
            if not isinstance(model_type, str) or not model_type:
                raise ValueError(f"Adapter {adapter.name!r} declares an invalid model type")
            existing = self._by_model_type.get(model_type)
            if existing is not None and existing is not adapter and not replace:
                raise ValueError(
                    f"model_type {model_type!r} is already registered by {existing.name!r}"
                )
        for model_type in adapter.model_types:
            self._by_model_type[model_type] = adapter

    def resolve(self, config: Mapping[str, Any]) -> ConfigFirstAdapter:
        model_type = config.get("model_type")
        if isinstance(model_type, str) and model_type in self._by_model_type:
            return self._by_model_type[model_type]

        text_config = config.get("text_config")
        nested_model_type: str | None = None
        if isinstance(text_config, Mapping):
            candidate = text_config.get("model_type")
            if isinstance(candidate, str):
                nested_model_type = candidate
                adapter = self._by_model_type.get(candidate)
                if adapter is not None:
                    return adapter

        observed = nested_model_type or model_type or "<missing>"
        raise AdapterConfigError(
            f"No config-first adapter is registered for model_type {observed!r}; "
            f"supported types: {', '.join(self.model_types)}"
        )

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str = "local",
        revision: str = "local",
    ) -> AdapterResult:
        return self.resolve(config).build(config, model_id=model_id, revision=revision)
