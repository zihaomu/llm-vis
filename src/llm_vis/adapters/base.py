"""Framework-neutral contracts used by config-first architecture adapters.

Adapters intentionally emit lightweight specifications first.  The IR bridge turns
these specifications into versioned Model Map objects, so model-family parsing is
kept independent from the serialization layer.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple


class AdapterConfigError(ValueError):
    """Raised when a supported config is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class DefinitionSpec:
    """A reusable architecture definition, separate from concrete layer instances."""

    key: str
    kind: str
    label: str
    class_name: str | None = None
    children: Tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceSpec:
    """A concrete occurrence of a reusable definition."""

    path: str
    definition_key: str
    parent_path: str | None = None
    layer_index: int | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticNodeSpec:
    """A semantic view node associated with a concrete module instance."""

    path: str
    kind: str
    label: str
    instance_path: str
    state_kind: str = "none"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigTensorSpec:
    """One virtual tensor contract justified by config-only evidence.

    ``semantic_name`` is deliberately not claimed to be the runtime framework's
    exact ``named_parameters`` key.  A missing shape means the category is known
    to exist while its concrete layout remains Unknown at the zero-weight
    boundary.
    """

    key: str
    semantic_name: str
    owner_instance_path: str
    role: str
    shape: Tuple[int | str, ...] | None
    dtype: str = "unknown"
    storage_dtype: str = "unknown"
    alias_group: str | None = None
    edge_kind: str = "weight"
    evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigInventoryUnknownSpec:
    """A structured inventory fact that config alone cannot resolve."""

    code: str
    message: str
    owner_instance_path: str | None = None
    tensor_key: str | None = None
    evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigInventory:
    """Config-provable tensor contracts plus explicit Unknown coverage."""

    tensors: Tuple[ConfigTensorSpec, ...] = ()
    unknowns: Tuple[ConfigInventoryUnknownSpec, ...] = ()


@dataclass(frozen=True)
class LayerStripEntry:
    """Compact, ordered description used by the UI layer-type strip."""

    layer_index: int
    instance_path: str
    definition_key: str
    attention_kind: str
    mlp_kind: str
    state_kind: str
    label: str
    expected_pattern: str
    anomaly: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class AdapterResult:
    """Normalized architecture produced without constructing model modules or weights."""

    adapter_name: str
    model_type: str
    model_id: str
    revision: str
    definitions: Tuple[DefinitionSpec, ...]
    instances: Tuple[InstanceSpec, ...]
    semantic_nodes: Tuple[SemanticNodeSpec, ...]
    layer_strip: Tuple[LayerStripEntry, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        definition_keys = [definition.key for definition in self.definitions]
        if len(definition_keys) != len(set(definition_keys)):
            raise AdapterConfigError("Adapter emitted duplicate definition keys")

        instance_paths = [instance.path for instance in self.instances]
        if len(instance_paths) != len(set(instance_paths)):
            raise AdapterConfigError("Adapter emitted duplicate instance paths")

        known_definitions = set(definition_keys)
        known_instances = set(instance_paths)
        for definition in self.definitions:
            missing_children = set(definition.children) - known_definitions
            if missing_children:
                raise AdapterConfigError(
                    f"Definition {definition.key!r} references unknown children: "
                    f"{sorted(missing_children)!r}"
                )
        for instance in self.instances:
            if instance.definition_key not in known_definitions:
                raise AdapterConfigError(
                    f"Instance {instance.path!r} references unknown definition "
                    f"{instance.definition_key!r}"
                )
            if instance.parent_path is not None and instance.parent_path not in known_instances:
                raise AdapterConfigError(
                    f"Instance {instance.path!r} references unknown parent {instance.parent_path!r}"
                )
        for node in self.semantic_nodes:
            if node.instance_path not in known_instances:
                raise AdapterConfigError(
                    f"Semantic node {node.path!r} references unknown instance "
                    f"{node.instance_path!r}"
                )

        strip_indices = [entry.layer_index for entry in self.layer_strip]
        if strip_indices != list(range(len(strip_indices))):
            raise AdapterConfigError("Layer strip must be contiguous and start at layer zero")
        for entry in self.layer_strip:
            if not entry.expected_pattern:
                raise AdapterConfigError("Layer strip expected_pattern must be non-empty")
            if entry.anomaly and not entry.reason:
                raise AdapterConfigError("An anomalous layer strip entry must include a reason")
            if not entry.anomaly and entry.reason is not None:
                raise AdapterConfigError(
                    "A non-anomalous layer strip entry cannot include a reason"
                )

        inventory_config = self.metadata.get("inventory_config")
        if inventory_config is not None and not isinstance(inventory_config, Mapping):
            raise AdapterConfigError("metadata.inventory_config must be a JSON object")

    def definition(self, key: str) -> DefinitionSpec:
        """Return one definition by its adapter-stable key."""

        for definition in self.definitions:
            if definition.key == key:
                return definition
        raise KeyError(key)

    def instance(self, path: str) -> InstanceSpec:
        """Return one concrete instance by semantic module path."""

        for instance in self.instances:
            if instance.path == path:
                return instance
        raise KeyError(path)

    @property
    def inventory_config(self) -> Mapping[str, Any]:
        """Return the normalized config evidence retained for M1 inventory."""

        value = self.metadata.get("inventory_config")
        return value if isinstance(value, Mapping) else {}

    def to_model_map(self):
        """Convert this normalized result into the strict versioned Model Map IR."""

        from .ir_bridge import result_to_model_map

        return result_to_model_map(self)


class ConfigFirstAdapter(ABC):
    """Base class for deterministic, non-executing config adapters."""

    name: str
    model_types: Tuple[str, ...]

    def can_handle(self, config: Mapping[str, Any]) -> bool:
        model_type = config.get("model_type")
        if model_type in self.model_types:
            return True
        text_config = config.get("text_config")
        return (
            isinstance(text_config, Mapping) and text_config.get("model_type") in self.model_types
        )

    @abstractmethod
    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str = "local",
        revision: str = "local",
    ) -> AdapterResult:
        """Build a normalized architecture without importing model code."""


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise AdapterConfigError(f"{key!r} must be a JSON object")
    return value


def require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterConfigError(f"{key!r} must be a positive integer")
    return value


def optional_positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    if key not in config:
        return default
    return require_positive_int(config, key)


def require_string_sequence(
    config: Mapping[str, Any], key: str, *, expected_length: int
) -> Tuple[str, ...]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AdapterConfigError(f"{key!r} must be a list of strings")
    if len(value) != expected_length:
        raise AdapterConfigError(f"{key!r} has {len(value)} entries; expected {expected_length}")
    if any(not isinstance(item, str) for item in value):
        raise AdapterConfigError(f"{key!r} must contain only strings")
    return tuple(value)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a config using the same deterministic JSON conventions as fixtures."""

    payload = json.dumps(
        config,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
