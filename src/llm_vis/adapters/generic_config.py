"""Conservative fallback adapter for otherwise unsupported JSON configs.

The fallback deliberately emits only a display skeleton.  It does not infer an
attention implementation, feed-forward layout, executable module tree, or cost
model from naming conventions shared by unrelated Hugging Face families.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from .base import (
    AdapterResult,
    ConfigFirstAdapter,
    DefinitionSpec,
    InstanceSpec,
    SemanticNodeSpec,
    config_sha256,
)

_COMMON_INT_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "num_hidden_layers": ("num_hidden_layers", "num_layers", "n_layer"),
    "hidden_size": ("hidden_size", "d_model", "n_embd"),
    "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
    "vocab_size": ("vocab_size",),
}


def _nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _first_positive_int(
    config: Mapping[str, Any], aliases: Sequence[str]
) -> tuple[int | None, str | None]:
    for key in aliases:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        return value, key
    return None, None


def _declared_class_name(config: Mapping[str, Any]) -> str | None:
    architectures = config.get("architectures")
    if not isinstance(architectures, Sequence) or isinstance(architectures, (str, bytes)):
        return None
    for value in architectures:
        candidate = _nonempty_string(value)
        if candidate is not None:
            return candidate
    return None


def _declares_causal_lm(config: Mapping[str, Any]) -> bool:
    architectures = config.get("architectures")
    if not isinstance(architectures, Sequence) or isinstance(architectures, (str, bytes)):
        return False
    return any(
        isinstance(value, str) and "causallm" in value.replace("_", "").lower()
        for value in architectures
    )


class GenericConfigAdapter(ConfigFirstAdapter):
    """Render unsupported configs as an explicitly partial, opaque L0 skeleton."""

    name = "generic-config"
    model_types: Tuple[str, ...] = ()

    def can_handle(self, config: Mapping[str, Any]) -> bool:
        """The registry uses this adapter only after all specific adapters miss."""

        return True

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str = "local",
        revision: str = "local",
    ) -> AdapterResult:
        outer_model_type = _nonempty_string(config.get("model_type"))
        nested = config.get("text_config")
        text_config = nested if isinstance(nested, Mapping) else None
        nested_model_type = (
            _nonempty_string(text_config.get("model_type"))
            if text_config is not None
            else None
        )
        architecture_config = text_config if text_config is not None else config
        config_scope = "text_config" if text_config is not None else "config"

        normalized: dict[str, Any] = {}
        field_sources: dict[str, str] = {}
        for canonical_name, aliases in _COMMON_INT_FIELDS.items():
            value, source_key = _first_positive_int(architecture_config, aliases)
            normalized[canonical_name] = value
            if source_key is not None:
                field_sources[canonical_name] = f"{config_scope}.{source_key}"

        dtype = architecture_config.get("torch_dtype", architecture_config.get("dtype"))
        normalized["dtype"] = dtype if isinstance(dtype, str) else None
        model_type = outer_model_type or nested_model_type or "unknown"
        layer_count = normalized["num_hidden_layers"]
        architecture_label = (
            f"Opaque architecture × {layer_count}"
            if layer_count is not None
            else "Opaque architecture"
        )
        class_name = _declared_class_name(config) or _declared_class_name(architecture_config)
        recognized_text_dimensions = {
            name
            for name in ("num_hidden_layers", "hidden_size", "vocab_size")
            if normalized[name] is not None
        }
        text_model_evidence = []
        if recognized_text_dimensions == {"num_hidden_layers", "hidden_size", "vocab_size"}:
            text_model_evidence.append("common_fields:vocab+hidden+layers")
        if text_config:
            text_model_evidence.append("config:text_config")
        if _declares_causal_lm(config) or _declares_causal_lm(architecture_config):
            text_model_evidence.append("config:architectures=CausalLM")
        detailed_display_skeleton = bool(text_model_evidence)
        common_attributes = {
            "config_only": True,
            "coverage_status": "opaque",
            "executable_module_tree_proven": False,
        }

        model_definition = DefinitionSpec(
            "model",
            "generic_model",
            "Config-only model skeleton",
            class_name,
            (
                ("token_embedding", "architecture", "final_norm", "lm_head")
                if detailed_display_skeleton
                else ("architecture",)
            ),
            {
                **common_attributes,
                "model_type": model_type,
                "config_scope": config_scope,
                "detailed_display_skeleton": detailed_display_skeleton,
            },
        )
        architecture_definition = DefinitionSpec(
            "architecture",
            "opaque",
            architecture_label,
            attributes={
                **common_attributes,
                "instance_count": layer_count,
                "attention_kind": "unknown",
                "ffn_kind": "unknown",
            },
        )
        detailed_definitions = (
            DefinitionSpec(
                "token_embedding",
                "opaque",
                "Token embedding (unverified)",
                attributes=common_attributes,
            ),
            DefinitionSpec(
                "final_norm",
                "opaque",
                "Final norm (unverified)",
                attributes=common_attributes,
            ),
            DefinitionSpec(
                "lm_head",
                "opaque",
                "LM head (unverified)",
                attributes=common_attributes,
            ),
        )
        if detailed_display_skeleton:
            definitions = (
                model_definition,
                detailed_definitions[0],
                architecture_definition,
                *detailed_definitions[1:],
            )
        else:
            definitions = (model_definition, architecture_definition)
        detailed_instances = (
            InstanceSpec("model.embed_tokens", "token_embedding", "model"),
            InstanceSpec("model.norm", "final_norm", "model"),
            InstanceSpec("lm_head", "lm_head", "model"),
        )
        if detailed_display_skeleton:
            instances = (
                InstanceSpec("model", "model"),
                detailed_instances[0],
                InstanceSpec("model.architecture", "architecture", "model"),
                *detailed_instances[1:],
            )
        else:
            instances = (
                InstanceSpec("model", "model"),
                InstanceSpec("model.architecture", "architecture", "model"),
            )
        semantic_nodes = (
            SemanticNodeSpec(
                "model.architecture",
                "opaque_architecture",
                architecture_label,
                "model.architecture",
                attributes={
                    "confidence": 0.0,
                    "coverage_status": "opaque",
                    "config_scope": config_scope,
                },
            ),
        )

        nested_sha256 = config_sha256(text_config) if text_config is not None else None
        return AdapterResult(
            adapter_name=self.name,
            model_type=model_type,
            model_id=model_id,
            revision=revision,
            definitions=definitions,
            instances=instances,
            semantic_nodes=semantic_nodes,
            layer_strip=(),
            metadata={
                "config_first": True,
                "config_sha256": config_sha256(config),
                "weights_loaded": False,
                "remote_code_executed": False,
                "fallback": True,
                "fallback_reason": "no_registered_adapter",
                "missing_adapter_for_model_type": (
                    nested_model_type or outer_model_type or "<missing>"
                ),
                "outer_model_type": outer_model_type,
                "nested_model_type": nested_model_type,
                "nested_text_config_present": text_config is not None,
                "nested_text_config_sha256": nested_sha256,
                "config_scope": config_scope,
                "evidence_scope": "config-only",
                "coverage_status": "partial",
                "architecture_coverage": "opaque",
                "operator_decomposition_available": False,
                "cost_model_available": False,
                "detailed_display_skeleton": detailed_display_skeleton,
                "text_model_evidence": tuple(text_model_evidence),
                "num_hidden_layers": normalized["num_hidden_layers"],
                "hidden_size": normalized["hidden_size"],
                "vocab_size": normalized["vocab_size"],
                "recognized_field_sources": field_sources,
                "inventory_config": normalized,
            },
        )


__all__ = ["GenericConfigAdapter"]
