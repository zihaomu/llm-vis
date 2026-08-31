"""Conversion from adapter-neutral specs to the strict Model Map IR."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from llm_vis.ir.ids import artifact_local_id, canonical_json, canonical_semantic_key
from llm_vis.ir.models import (
    Definition,
    DefinitionKind,
    Diagnostic,
    DiagnosticSeverity,
    DType,
    Edge,
    EdgeKind,
    Framework,
    Instance,
    Model,
    ModelMap,
    Port,
    PortDirection,
    SemanticKind,
    SemanticNode,
    SourceArtifact,
    SourceArtifactKind,
    Symbol,
    SymbolicExpression,
    TensorLayout,
    TensorOrigin,
    TensorRole,
    TensorSpec,
)

from .base import (
    AdapterResult,
    ConfigInventory,
    ConfigInventoryUnknownSpec,
    ConfigTensorSpec,
    DefinitionSpec,
    SemanticNodeSpec,
)

_DEFINITION_KINDS = {
    "causal_lm": DefinitionKind.MODEL,
    "generic_model": DefinitionKind.MODEL,
    "multimodal_causal_lm": DefinitionKind.MODEL,
    "vision_tower": DefinitionKind.TOWER,
    "decoder_stack": DefinitionKind.STAGE,
    "decoder_block": DefinitionKind.BLOCK,
    "expert_pool": DefinitionKind.EXPERT_POOL,
    "conditional_head": DefinitionKind.CONDITIONAL,
    "opaque": DefinitionKind.OPAQUE,
}

_SEMANTIC_KINDS = {
    "dense_decoder_block": SemanticKind.FULL_ATTENTION,
    "linear_attention_block": SemanticKind.LINEAR_ATTENTION,
    "full_attention_block": SemanticKind.FULL_ATTENTION,
    "dense_dsa_block": SemanticKind.DSA,
    "sparse_dsa_block": SemanticKind.DSA,
    "expert_pool": SemanticKind.EXPERT_POOL,
    "conditional_mtp": SemanticKind.MTP,
    "vision_tower": SemanticKind.VISION_TOWER,
    "opaque_architecture": SemanticKind.OPAQUE,
}

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
}


def _artifact_id(result: AdapterResult, module_path: str, variant: str) -> str:
    return artifact_local_id(
        model_revision=result.revision,
        framework_domain=Framework.TRANSFORMERS.value,
        module_path=module_path,
        capture_variant=variant,
    )


def _definition_signature(spec: DefinitionSpec) -> Dict[str, object]:
    return {
        "key": spec.key,
        "kind": spec.kind,
        "class_name": spec.class_name,
        "children": list(spec.children),
        "attributes": dict(spec.attributes),
    }


def _config_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _config_dtype(value: Any) -> str:
    if not isinstance(value, str):
        return DType.UNKNOWN.value
    normalized = _DTYPE_ALIASES.get(value.lower(), value.lower())
    try:
        return DType(normalized).value
    except ValueError:
        return DType.UNKNOWN.value


def _alias_group(result: AdapterResult, semantic_path: str) -> str:
    return _artifact_id(result, f"alias/{semantic_path}", "config-first-alias")


def _build_config_inventory(result: AdapterResult) -> ConfigInventory:
    """Derive virtual inventory contracts from the adapter-retained config subset.

    These records describe semantic parameter/state categories.  They do not claim
    exact runtime ``named_parameters`` keys, object identity, layout, or buffers.
    Unknown facts remain explicit instead of being replaced by zero-sized tensors.
    """

    config = result.inventory_config
    tensors: List[ConfigTensorSpec] = []
    unknowns: List[ConfigInventoryUnknownSpec] = []
    if result.adapter_name == "generic-config":
        return ConfigInventory(
            unknowns=(
                ConfigInventoryUnknownSpec(
                    code="CONFIG_PARAMETER_INVENTORY_OPAQUE",
                    message=(
                        "No family adapter matched this config; parameter categories and "
                        "layouts remain Unknown rather than being inferred from field names"
                    ),
                    owner_instance_path="model.architecture",
                    evidence=(
                        "source:config",
                        "coverage:opaque",
                        "weights_loaded:false",
                        "unknown-is-not-zero",
                    ),
                ),
            )
        )
    if not config:
        return ConfigInventory(
            unknowns=(
                ConfigInventoryUnknownSpec(
                    code="CONFIG_INVENTORY_UNAVAILABLE",
                    message="Adapter result did not retain config evidence for tensor inventory",
                    evidence=("weights_loaded:false", "full_meta_tree:false"),
                ),
            )
        )

    parameter_dtype = _config_dtype(config.get("dtype"))
    hidden_size = _config_int(config, "hidden_size")
    intermediate_size = _config_int(config, "intermediate_size")
    vocab_size = _config_int(config, "vocab_size")
    tie_word_embeddings = config.get("tie_word_embeddings")
    common_evidence = (
        f"adapter:{result.adapter_name}",
        "source:config",
        "inventory:semantic_contract_not_runtime_parameter_name",
        "weights_loaded:false",
    )

    def add_tensor(
        key: str,
        semantic_name: str,
        owner: str,
        role: str,
        shape: Sequence[int | str] | None,
        *,
        dtype: str | None = None,
        alias_group: str | None = None,
        edge_kind: str | None = None,
        evidence: Sequence[str] = (),
        unknown_message: str | None = None,
    ) -> None:
        normalized_shape = tuple(shape) if shape is not None else None
        tensors.append(
            ConfigTensorSpec(
                key=key,
                semantic_name=semantic_name,
                owner_instance_path=owner,
                role=role,
                shape=normalized_shape,
                dtype=dtype or parameter_dtype,
                storage_dtype=dtype or parameter_dtype,
                alias_group=alias_group,
                edge_kind=edge_kind
                or ("weight" if role == TensorRole.PARAMETER.value else "state_write"),
                evidence=(*common_evidence, *tuple(evidence)),
            )
        )
        if normalized_shape is None:
            unknowns.append(
                ConfigInventoryUnknownSpec(
                    code="CONFIG_TENSOR_SHAPE_UNKNOWN",
                    message=unknown_message
                    or f"Config proves {semantic_name!r} exists but not its tensor layout",
                    owner_instance_path=owner,
                    tensor_key=key,
                    evidence=(*common_evidence, *tuple(evidence)),
                )
            )

    def add_ffn(owner: str, *, expert_prefix: str = "mlp") -> None:
        if hidden_size is None or intermediate_size is None:
            for name in (
                "gate_projection.weight",
                "up_projection.weight",
                "down_projection.weight",
            ):
                add_tensor(
                    f"{owner}.{expert_prefix}.{name}",
                    f"{expert_prefix}.{name}",
                    owner,
                    TensorRole.PARAMETER.value,
                    None,
                    evidence=("config:hidden_size/intermediate_size:missing",),
                )
            return
        add_tensor(
            f"{owner}.{expert_prefix}.gate_projection.weight",
            f"{expert_prefix}.gate_projection.weight",
            owner,
            TensorRole.PARAMETER.value,
            (intermediate_size, hidden_size),
            evidence=("shape:linear[out,in]",),
        )
        add_tensor(
            f"{owner}.{expert_prefix}.up_projection.weight",
            f"{expert_prefix}.up_projection.weight",
            owner,
            TensorRole.PARAMETER.value,
            (intermediate_size, hidden_size),
            evidence=("shape:linear[out,in]",),
        )
        add_tensor(
            f"{owner}.{expert_prefix}.down_projection.weight",
            f"{expert_prefix}.down_projection.weight",
            owner,
            TensorRole.PARAMETER.value,
            (hidden_size, intermediate_size),
            evidence=("shape:linear[out,in]",),
        )

    if result.adapter_name == "qwen3-5":
        embedding_owner = "model.text.embed_tokens"
        norm_owner = "model.text.norm"
    else:
        embedding_owner = "model.embed_tokens"
        norm_owner = "model.norm"

    tied_alias = (
        _alias_group(result, "token_embedding_lm_head") if tie_word_embeddings is True else None
    )
    tying_evidence = (
        f"config:tie_word_embeddings={str(tie_word_embeddings).lower()}"
        if isinstance(tie_word_embeddings, bool)
        else "config:tie_word_embeddings=unknown"
    )
    embedding_shape = (
        (vocab_size, hidden_size) if vocab_size is not None and hidden_size is not None else None
    )
    add_tensor(
        f"{embedding_owner}.token_embedding.weight",
        "token_embedding.weight",
        embedding_owner,
        TensorRole.PARAMETER.value,
        embedding_shape,
        alias_group=tied_alias,
        evidence=(tying_evidence, "shape:embedding[vocab,hidden]"),
    )
    add_tensor(
        "lm_head.output_projection.weight",
        "output_projection.weight",
        "lm_head",
        TensorRole.PARAMETER.value,
        embedding_shape,
        alias_group=tied_alias,
        evidence=(tying_evidence, "shape:linear[out,in]"),
    )
    add_tensor(
        f"{norm_owner}.final_norm.weight",
        "final_norm.weight",
        norm_owner,
        TensorRole.PARAMETER.value,
        (hidden_size,) if hidden_size is not None else None,
        evidence=("config:hidden_size",),
    )

    if tie_word_embeddings is None:
        unknowns.append(
            ConfigInventoryUnknownSpec(
                code="CONFIG_SHARED_WEIGHT_UNKNOWN",
                message="tie_word_embeddings is absent; embedding/head aliasing remains Unknown",
                evidence=common_evidence,
            )
        )
    elif not isinstance(tie_word_embeddings, bool):
        unknowns.append(
            ConfigInventoryUnknownSpec(
                code="CONFIG_SHARED_WEIGHT_UNKNOWN",
                message=(
                    "tie_word_embeddings is not boolean; embedding/head aliasing remains Unknown"
                ),
                evidence=(*common_evidence, f"value:{tie_word_embeddings!r}"),
            )
        )

    attention_bias = config.get("attention_bias")
    if attention_bias is None:
        unknowns.append(
            ConfigInventoryUnknownSpec(
                code="CONFIG_ATTENTION_BIAS_UNKNOWN",
                message="attention_bias is absent; attention bias inventory remains Unknown",
                evidence=common_evidence,
            )
        )
    elif not isinstance(attention_bias, bool):
        unknowns.append(
            ConfigInventoryUnknownSpec(
                code="CONFIG_ATTENTION_BIAS_UNKNOWN",
                message="attention_bias is not boolean; bias inventory remains Unknown",
                evidence=(*common_evidence, f"value:{attention_bias!r}"),
            )
        )
        attention_bias = None
    if parameter_dtype == DType.UNKNOWN.value:
        unknowns.append(
            ConfigInventoryUnknownSpec(
                code="CONFIG_PARAMETER_DTYPE_UNKNOWN",
                message="Config does not declare a supported parameter dtype",
                evidence=(*common_evidence, f"value:{config.get('dtype')!r}"),
            )
        )

    if result.adapter_name == "tiny-dense":
        query_heads = _config_int(config, "num_attention_heads")
        kv_heads = _config_int(config, "num_key_value_heads")
        head_dim = _config_int(config, "head_dim")
        for entry in result.layer_strip:
            owner = entry.instance_path
            for norm_name in ("input_norm.weight", "post_attention_norm.weight"):
                add_tensor(
                    f"{owner}.{norm_name}",
                    norm_name,
                    owner,
                    TensorRole.PARAMETER.value,
                    (hidden_size,) if hidden_size is not None else None,
                    evidence=("config:hidden_size",),
                )
            if None not in (hidden_size, query_heads, kv_heads, head_dim):
                query_width = int(query_heads) * int(head_dim)
                kv_width = int(kv_heads) * int(head_dim)
                projection_shapes: Mapping[str, Tuple[int, int]] = {
                    "self_attention.query_projection.weight": (query_width, int(hidden_size)),
                    "self_attention.key_projection.weight": (kv_width, int(hidden_size)),
                    "self_attention.value_projection.weight": (kv_width, int(hidden_size)),
                    "self_attention.output_projection.weight": (int(hidden_size), query_width),
                }
            else:
                projection_shapes = {}
            for name in (
                "self_attention.query_projection.weight",
                "self_attention.key_projection.weight",
                "self_attention.value_projection.weight",
                "self_attention.output_projection.weight",
            ):
                add_tensor(
                    f"{owner}.{name}",
                    name,
                    owner,
                    TensorRole.PARAMETER.value,
                    projection_shapes.get(name),
                    evidence=("shape:linear[out,in]",),
                )
            if attention_bias is True:
                bias_widths = (
                    ("query", query_heads * head_dim if query_heads and head_dim else None),
                    ("key", kv_heads * head_dim if kv_heads and head_dim else None),
                    ("value", kv_heads * head_dim if kv_heads and head_dim else None),
                    ("output", hidden_size),
                )
                for name, width in bias_widths:
                    add_tensor(
                        f"{owner}.self_attention.{name}_projection.bias",
                        f"self_attention.{name}_projection.bias",
                        owner,
                        TensorRole.PARAMETER.value,
                        (width,) if width is not None else None,
                        evidence=("config:attention_bias=true",),
                    )
            add_ffn(owner)
            state_shape = (
                ("B", int(kv_heads), "S_KV", int(head_dim))
                if kv_heads is not None and head_dim is not None
                else None
            )
            for state_name in ("kv_cache.key", "kv_cache.value"):
                add_tensor(
                    f"{owner}.{state_name}",
                    state_name,
                    owner,
                    TensorRole.CACHE.value,
                    state_shape,
                    dtype=DType.UNKNOWN.value,
                    edge_kind=EdgeKind.STATE_WRITE.value,
                    evidence=(f"config:use_cache={config.get('use_cache')!r}",),
                    unknown_message=(
                        "KV cache category is known but its shape is not config-provable"
                    ),
                )

    elif result.adapter_name == "qwen3-5":
        query_heads = _config_int(config, "num_attention_heads")
        kv_heads = _config_int(config, "num_key_value_heads")
        head_dim = _config_int(config, "head_dim")
        key_heads = _config_int(config, "linear_num_key_heads")
        value_heads = _config_int(config, "linear_num_value_heads")
        key_head_dim = _config_int(config, "linear_key_head_dim")
        value_head_dim = _config_int(config, "linear_value_head_dim")
        conv_kernel = _config_int(config, "linear_conv_kernel_dim")
        output_gate = config.get("attn_output_gate")
        state_dtype = _config_dtype(config.get("mamba_ssm_dtype"))
        for entry in result.layer_strip:
            owner = entry.instance_path
            for norm_name in ("input_norm.weight", "post_attention_norm.weight"):
                add_tensor(
                    f"{owner}.{norm_name}",
                    norm_name,
                    owner,
                    TensorRole.PARAMETER.value,
                    (hidden_size,) if hidden_size is not None else None,
                    evidence=("config:hidden_size",),
                )
            if entry.attention_kind == "full_attention":
                q_multiplier = 2 if output_gate is True else 1 if output_gate is False else None
                query_width = (
                    int(query_heads) * int(head_dim)
                    if query_heads is not None and head_dim is not None
                    else None
                )
                kv_width = (
                    int(kv_heads) * int(head_dim)
                    if kv_heads is not None and head_dim is not None
                    else None
                )
                shapes = {
                    "self_attention.query_projection.weight": (
                        (q_multiplier * query_width, hidden_size)
                        if q_multiplier is not None
                        and query_width is not None
                        and hidden_size is not None
                        else None
                    ),
                    "self_attention.key_projection.weight": (
                        (kv_width, hidden_size)
                        if kv_width is not None and hidden_size is not None
                        else None
                    ),
                    "self_attention.value_projection.weight": (
                        (kv_width, hidden_size)
                        if kv_width is not None and hidden_size is not None
                        else None
                    ),
                    "self_attention.output_projection.weight": (
                        (hidden_size, query_width)
                        if query_width is not None and hidden_size is not None
                        else None
                    ),
                }
                for name, shape in shapes.items():
                    add_tensor(
                        f"{owner}.{name}",
                        name,
                        owner,
                        TensorRole.PARAMETER.value,
                        shape,
                        evidence=(
                            "shape:linear[out,in]",
                            f"config:attn_output_gate={output_gate!r}",
                        ),
                    )
                for name in ("self_attention.query_norm.weight", "self_attention.key_norm.weight"):
                    add_tensor(
                        f"{owner}.{name}",
                        name,
                        owner,
                        TensorRole.PARAMETER.value,
                        (head_dim,) if head_dim is not None else None,
                        evidence=("config:head_dim",),
                    )
                if attention_bias is True:
                    bias_widths = {
                        "query": q_multiplier * query_width
                        if q_multiplier is not None and query_width is not None
                        else None,
                        "key": kv_width,
                        "value": kv_width,
                        "output": hidden_size,
                    }
                    for name, width in bias_widths.items():
                        add_tensor(
                            f"{owner}.self_attention.{name}_projection.bias",
                            f"self_attention.{name}_projection.bias",
                            owner,
                            TensorRole.PARAMETER.value,
                            (width,) if width is not None else None,
                            evidence=("config:attention_bias=true",),
                        )
                state_shape = (
                    ("B", int(kv_heads), "S_KV", int(head_dim))
                    if kv_heads is not None and head_dim is not None
                    else None
                )
                for state_name in ("kv_cache.key", "kv_cache.value"):
                    add_tensor(
                        f"{owner}.{state_name}",
                        state_name,
                        owner,
                        TensorRole.CACHE.value,
                        state_shape,
                        dtype=DType.UNKNOWN.value,
                        edge_kind=EdgeKind.STATE_WRITE.value,
                        evidence=(f"config:use_cache={config.get('use_cache')!r}",),
                    )
            else:
                dimensions = (
                    hidden_size,
                    key_heads,
                    value_heads,
                    key_head_dim,
                    value_head_dim,
                    conv_kernel,
                )
                if all(value is not None for value in dimensions):
                    key_width = int(key_heads) * int(key_head_dim)
                    value_width = int(value_heads) * int(value_head_dim)
                    conv_width = 2 * key_width + value_width
                    linear_shapes: Mapping[str, Tuple[int, ...] | None] = {
                        "linear_attention.qkv_projection.weight": (conv_width, int(hidden_size)),
                        "linear_attention.gate_projection.weight": (
                            value_width,
                            int(hidden_size),
                        ),
                        "linear_attention.beta_projection.weight": (
                            int(value_heads),
                            int(hidden_size),
                        ),
                        "linear_attention.alpha_projection.weight": (
                            int(value_heads),
                            int(hidden_size),
                        ),
                        "linear_attention.output_projection.weight": (
                            int(hidden_size),
                            value_width,
                        ),
                        "linear_attention.depthwise_conv.weight": (conv_width, int(conv_kernel)),
                        "linear_attention.side_parameters": (
                            2 * int(value_heads) + int(value_head_dim),
                        ),
                    }
                else:
                    conv_width = None
                    linear_shapes = {
                        "linear_attention.parameters": None,
                    }
                for name, shape in linear_shapes.items():
                    add_tensor(
                        f"{owner}.{name}",
                        name,
                        owner,
                        TensorRole.PARAMETER.value,
                        shape,
                        evidence=("config:gated_deltanet_parameter_contract",),
                    )
                recurrent_shape = (
                    ("B", int(value_heads), int(key_head_dim), int(value_head_dim))
                    if None not in (value_heads, key_head_dim, value_head_dim)
                    else None
                )
                conv_state_shape = (
                    ("B", int(conv_width), int(conv_kernel))
                    if conv_width is not None and conv_kernel is not None
                    else None
                )
                add_tensor(
                    f"{owner}.recurrent_state.delta",
                    "recurrent_state.delta",
                    owner,
                    TensorRole.STATE.value,
                    recurrent_shape,
                    dtype=state_dtype,
                    edge_kind=EdgeKind.STATE_WRITE.value,
                    evidence=("config:mamba_ssm_dtype",),
                )
                add_tensor(
                    f"{owner}.recurrent_state.conv",
                    "recurrent_state.conv",
                    owner,
                    TensorRole.STATE.value,
                    conv_state_shape,
                    dtype=state_dtype,
                    edge_kind=EdgeKind.STATE_WRITE.value,
                    evidence=("config:linear_conv_kernel_dim",),
                )
            add_ffn(owner)

        for instance in result.instances:
            if instance.path in {"model.vision", "model.projector"}:
                add_tensor(
                    f"{instance.path}.parameter_inventory",
                    "parameter_inventory",
                    instance.path,
                    TensorRole.PARAMETER.value,
                    None,
                    evidence=(
                        "config:component_present",
                        "source:model_implementation_unavailable",
                    ),
                    unknown_message=(
                        f"{instance.path} parameter layout requires model implementation evidence"
                    ),
                )
            elif instance.path.startswith("model.mtp."):
                add_tensor(
                    f"{instance.path}.parameter_inventory",
                    "parameter_inventory",
                    instance.path,
                    TensorRole.PARAMETER.value,
                    None,
                    evidence=("config:mtp_num_hidden_layers",),
                    unknown_message="MTP parameter layout requires model implementation evidence",
                )
        if config.get("mtp_use_dedicated_embeddings") is False:
            unknowns.append(
                ConfigInventoryUnknownSpec(
                    code="CONFIG_SHARED_USAGE_DECLARED",
                    message=(
                        "MTP declares no dedicated embedding; exact shared object relationship "
                        "requires module evidence"
                    ),
                    evidence=(*common_evidence, "config:mtp_use_dedicated_embeddings=false"),
                )
            )

    elif result.adapter_name == "glm-moe-dsa":
        routed_experts = _config_int(config, "n_routed_experts")
        shared_experts = config.get("n_shared_experts")
        expert_intermediate = _config_int(config, "moe_intermediate_size")
        if isinstance(shared_experts, bool) or not isinstance(shared_experts, int):
            shared_experts = None
        for entry in result.layer_strip:
            owner = entry.instance_path
            for norm_name in ("input_norm.weight", "post_attention_norm.weight"):
                add_tensor(
                    f"{owner}.{norm_name}",
                    norm_name,
                    owner,
                    TensorRole.PARAMETER.value,
                    (hidden_size,) if hidden_size is not None else None,
                    evidence=("config:hidden_size",),
                )
            add_tensor(
                f"{owner}.dsa_attention.parameter_inventory",
                "dsa_attention.parameter_inventory",
                owner,
                TensorRole.PARAMETER.value,
                None,
                evidence=(
                    "config:dsa_component_present",
                    "coverage:parameter_layout_unknown",
                ),
                unknown_message="GLM DSA parameter layout is not proven by config alone",
            )
            if entry.mlp_kind == "dense":
                add_ffn(owner)
            else:
                if None not in (hidden_size, routed_experts, expert_intermediate):
                    add_tensor(
                        f"{owner}.moe.router.weight",
                        "moe.router.weight",
                        owner,
                        TensorRole.PARAMETER.value,
                        (int(routed_experts), int(hidden_size)),
                        evidence=("shape:linear[out,in]",),
                    )
                    add_tensor(
                        f"{owner}.moe.routed_experts.gate_up.weight",
                        "moe.routed_experts.gate_up.weight",
                        owner,
                        TensorRole.PARAMETER.value,
                        (int(routed_experts), 2 * int(expert_intermediate), int(hidden_size)),
                        evidence=("config:virtual_expert_pool", "experts_expanded:false"),
                    )
                    add_tensor(
                        f"{owner}.moe.routed_experts.down.weight",
                        "moe.routed_experts.down.weight",
                        owner,
                        TensorRole.PARAMETER.value,
                        (int(routed_experts), int(hidden_size), int(expert_intermediate)),
                        evidence=("config:virtual_expert_pool", "experts_expanded:false"),
                    )
                else:
                    add_tensor(
                        f"{owner}.moe.routed_experts.parameter_inventory",
                        "moe.routed_experts.parameter_inventory",
                        owner,
                        TensorRole.PARAMETER.value,
                        None,
                        evidence=("config:virtual_expert_pool",),
                    )
                if shared_experts is not None and shared_experts > 0:
                    shared_shape_known = hidden_size is not None and expert_intermediate is not None
                    add_tensor(
                        f"{owner}.moe.shared_experts.gate_up.weight",
                        "moe.shared_experts.gate_up.weight",
                        owner,
                        TensorRole.PARAMETER.value,
                        (
                            (shared_experts, 2 * int(expert_intermediate), int(hidden_size))
                            if shared_shape_known
                            else None
                        ),
                        evidence=("config:n_shared_experts",),
                    )
                    add_tensor(
                        f"{owner}.moe.shared_experts.down.weight",
                        "moe.shared_experts.down.weight",
                        owner,
                        TensorRole.PARAMETER.value,
                        (
                            (shared_experts, int(hidden_size), int(expert_intermediate))
                            if shared_shape_known
                            else None
                        ),
                        evidence=("config:n_shared_experts",),
                    )
            add_tensor(
                f"{owner}.kv_cache",
                "kv_cache",
                owner,
                TensorRole.CACHE.value,
                None,
                dtype=DType.UNKNOWN.value,
                edge_kind=EdgeKind.STATE_WRITE.value,
                evidence=("config:use_cache", "coverage:dsa_state_layout_unknown"),
                unknown_message=(
                    "GLM DSA KV/state tensor layout is Unknown at config-only capability"
                ),
            )

        for instance in result.instances:
            if instance.path.startswith("model.mtp."):
                add_tensor(
                    f"{instance.path}.parameter_inventory",
                    "parameter_inventory",
                    instance.path,
                    TensorRole.PARAMETER.value,
                    None,
                    evidence=("config:num_nextn_predict_layers",),
                    unknown_message="GLM MTP parameter layout requires module evidence",
                )
        if config.get("index_share_for_mtp_iteration") is True:
            unknowns.append(
                ConfigInventoryUnknownSpec(
                    code="CONFIG_SHARED_USAGE_DECLARED",
                    message=(
                        "GLM config declares MTP index sharing; exact alias object remains "
                        "Unknown without module evidence"
                    ),
                    evidence=(*common_evidence, "config:index_share_for_mtp_iteration=true"),
                )
            )
    else:
        unknowns.append(
            ConfigInventoryUnknownSpec(
                code="CONFIG_INVENTORY_ADAPTER_UNSUPPORTED",
                message=f"No config inventory rule for adapter {result.adapter_name!r}",
                evidence=common_evidence,
            )
        )

    unknowns.append(
        ConfigInventoryUnknownSpec(
            code="CONFIG_BUFFER_INVENTORY_UNKNOWN",
            message=(
                "Config does not enumerate runtime buffers; zero-weight analysis does not "
                "construct or inspect named_buffers"
            ),
            evidence=(*common_evidence, "full_meta_tree:false"),
        )
    )
    return ConfigInventory(tensors=tuple(tensors), unknowns=tuple(unknowns))


def _semantic_kind(spec: SemanticNodeSpec) -> SemanticKind:
    direct = _SEMANTIC_KINDS.get(spec.kind)
    if direct is not None:
        return direct
    attention_kind = spec.attributes.get("attention_kind")
    if attention_kind == "linear_attention":
        return SemanticKind.LINEAR_ATTENTION
    if attention_kind == "full_attention":
        return SemanticKind.FULL_ATTENTION
    if attention_kind == "dsa":
        return SemanticKind.DSA
    return SemanticKind.GENERIC


def result_to_model_map(result: AdapterResult) -> ModelMap:
    """Create a reference-complete, deterministically identified Model Map."""

    source_id = _artifact_id(result, "source/config.json", "config-first-source")
    model_id = _artifact_id(result, "model", "config-first-model")
    is_huggingface = "/" in result.model_id and result.model_id != "local"
    source_uri = (
        f"hf://{result.model_id}@{result.revision}/config.json"
        if is_huggingface
        else f"config://{result.model_id}@{result.revision}"
    )
    source_artifact = SourceArtifact(
        id=source_id,
        kind=(
            SourceArtifactKind.HUGGINGFACE_CONFIG
            if is_huggingface
            else SourceArtifactKind.LOCAL_CONFIG
        ),
        uri=source_uri,
        revision=result.revision,
        sha256=str(result.metadata["config_sha256"]),
        license=result.metadata.get("license"),
        trusted=False,
    )
    model = Model(
        id=model_id,
        name=result.model_id,
        family=result.model_type,
        revision=result.revision,
        framework=Framework.TRANSFORMERS,
        source_artifact_ids=[source_id],
    )
    definition_ids = {
        spec.key: _artifact_id(result, f"definition/{spec.key}", "config-first-definition")
        for spec in result.definitions
    }
    instance_ids = {
        spec.path: _artifact_id(result, f"instance/{spec.path}", "config-first-instance")
        for spec in result.instances
    }
    instance_specs = {spec.path: spec for spec in result.instances}
    inventory = _build_config_inventory(result)

    ports_by_definition: Dict[str, Dict[Tuple[str, str], Port]] = {}
    for tensor_spec in inventory.tensors:
        owner_spec = instance_specs[tensor_spec.owner_instance_path]
        role = TensorRole(tensor_spec.role)
        direction = PortDirection.INPUT if role == TensorRole.PARAMETER else PortDirection.INOUT
        ports_by_definition.setdefault(owner_spec.definition_key, {})[
            (tensor_spec.semantic_name, role.value)
        ] = Port(
            name=tensor_spec.semantic_name,
            direction=direction,
            tensor_role=role,
        )

    definitions = []
    for spec in result.definitions:
        semantic_pattern = None
        if spec.attributes:
            semantic_pattern = canonical_json(dict(spec.attributes))
        definitions.append(
            Definition(
                id=definition_ids[spec.key],
                kind=_DEFINITION_KINDS.get(spec.kind, DefinitionKind.MODULE),
                label=spec.label,
                class_name=spec.class_name,
                ports=[
                    port
                    for _, port in sorted(
                        ports_by_definition.get(spec.key, {}).items(),
                        key=lambda item: item[0],
                    )
                ],
                children=[definition_ids[key] for key in spec.children],
                semantic_pattern=semantic_pattern,
            )
        )

    instances = []
    definition_specs = {spec.key: spec for spec in result.definitions}
    for spec in result.instances:
        definition_spec = definition_specs[spec.definition_key]
        overrides = dict(spec.overrides)
        if definition_spec.attributes:
            overrides.setdefault("definition_attributes", dict(definition_spec.attributes))
        overrides["canonical_semantic_key"] = canonical_semantic_key(
            semantic_path=spec.path,
            definition_signature=_definition_signature(definition_spec),
            symbolic_contract={},
            structural_signature={"parent_path": spec.parent_path},
        )
        instances.append(
            Instance(
                id=instance_ids[spec.path],
                definition_id=definition_ids[spec.definition_key],
                parent_id=instance_ids[spec.parent_path] if spec.parent_path is not None else None,
                module_path=spec.path,
                layer_index=spec.layer_index,
                overrides=overrides,
            )
        )

    semantic_nodes: List[SemanticNode] = []
    main_semantic_indices: Dict[str, int] = {}
    state_semantic_indices: Dict[str, int] = {}
    for spec in result.semantic_nodes:
        instance_spec = result.instance(spec.instance_path)
        semantic_id = _artifact_id(result, f"semantic/{spec.path}", "config-first-semantic")
        state_id = None
        if spec.state_kind in {"recurrent_state", "kv_cache"}:
            state_id = _artifact_id(result, f"semantic/{spec.path}/state", "config-first-semantic")
        evidence = [f"adapter:{result.adapter_name}", "source:config"]
        if spec.attributes:
            evidence.append(f"attributes:{canonical_json(dict(spec.attributes))}")
        main_semantic_indices[spec.instance_path] = len(semantic_nodes)
        raw_confidence = spec.attributes.get("confidence", 1.0)
        confidence = (
            float(raw_confidence)
            if not isinstance(raw_confidence, bool)
            and isinstance(raw_confidence, (int, float))
            and 0.0 <= raw_confidence <= 1.0
            else 1.0
        )
        semantic_nodes.append(
            SemanticNode(
                id=semantic_id,
                kind=_semantic_kind(spec),
                label=spec.label,
                definition_id=definition_ids[instance_spec.definition_key],
                instance_ids=[instance_ids[spec.instance_path]],
                child_ids=[state_id] if state_id is not None else [],
                confidence=confidence,
                evidence=evidence,
            )
        )
        if state_id is not None:
            state_kind = (
                SemanticKind.RECURRENT_STATE
                if spec.state_kind == "recurrent_state"
                else SemanticKind.KV_CACHE
            )
            state_semantic_indices[spec.instance_path] = len(semantic_nodes)
            semantic_nodes.append(
                SemanticNode(
                    id=state_id,
                    kind=state_kind,
                    label=(
                        f"{spec.label} recurrent state"
                        if state_kind == SemanticKind.RECURRENT_STATE
                        else f"{spec.label} KV cache"
                    ),
                    definition_id=definition_ids[instance_spec.definition_key],
                    instance_ids=[instance_ids[spec.instance_path]],
                    confidence=1.0,
                    evidence=[f"adapter:{result.adapter_name}", f"state:{spec.state_kind}"],
                )
            )

    tensors: List[TensorSpec] = []
    tensor_ids: Dict[str, str] = {}
    symbol_names = set()
    for spec in inventory.tensors:
        tensor_id = _artifact_id(
            result,
            f"tensor/inventory/{spec.key}",
            "config-first-inventory",
        )
        tensor_ids[spec.key] = tensor_id
        symbolic_shape = []
        concrete_shape = None
        if spec.shape is not None:
            for dimension in spec.shape:
                if isinstance(dimension, int):
                    symbolic_shape.append(SymbolicExpression.literal(dimension))
                else:
                    symbol_names.add(dimension)
                    symbolic_shape.append(SymbolicExpression.symbol_ref(dimension))
            if all(isinstance(dimension, int) for dimension in spec.shape):
                concrete_shape = [int(dimension) for dimension in spec.shape]
        tensors.append(
            TensorSpec(
                id=tensor_id,
                semantic_name=spec.semantic_name,
                owner_id=instance_ids[spec.owner_instance_path],
                symbolic_shape=symbolic_shape,
                concrete_shape=concrete_shape,
                shape_known=spec.shape is not None,
                dtype=DType(spec.dtype),
                storage_dtype=DType(spec.storage_dtype),
                layout=TensorLayout.UNKNOWN,
                role=TensorRole(spec.role),
                alias_group=spec.alias_group,
                origin=TensorOrigin.CONFIG,
                materialized=False,
                source_artifact_ids=[source_id],
                evidence=list(spec.evidence),
            )
        )

        if spec.role in {TensorRole.CACHE.value, TensorRole.STATE.value}:
            main_index = main_semantic_indices.get(spec.owner_instance_path)
            state_index = state_semantic_indices.get(spec.owner_instance_path)
            if main_index is not None:
                semantic_nodes[main_index].input_tensor_ids.append(tensor_id)
                semantic_nodes[main_index].output_tensor_ids.append(tensor_id)
            if state_index is not None:
                semantic_nodes[state_index].input_tensor_ids.append(tensor_id)
                semantic_nodes[state_index].output_tensor_ids.append(tensor_id)

    symbols = [
        Symbol(
            id=_artifact_id(result, f"symbol/{name}", "config-first-inventory-symbol"),
            name=name,
            description=(
                "Batch size; bound by a Scenario at cost-analysis time"
                if name == "B"
                else "Visible KV sequence length; bound by a Scenario at cost-analysis time"
            ),
            lower_bound=0 if name == "S_KV" else 1,
        )
        for name in sorted(symbol_names)
    ]

    edges: List[Edge] = []
    for spec in inventory.tensors:
        owner_spec = instance_specs[spec.owner_instance_path]
        tensor_id = tensor_ids[spec.key]
        main_index = main_semantic_indices.get(spec.owner_instance_path)
        state_index = state_semantic_indices.get(spec.owner_instance_path)
        if (
            spec.role in {TensorRole.CACHE.value, TensorRole.STATE.value}
            and main_index is not None
            and state_index is not None
        ):
            main_id = semantic_nodes[main_index].id
            state_id = semantic_nodes[state_index].id
            edges.extend(
                (
                    Edge(
                        id=_artifact_id(
                            result,
                            f"edge/inventory/{spec.key}/read",
                            "config-first-inventory-edge",
                        ),
                        source_id=state_id,
                        target_id=main_id,
                        tensor_id=tensor_id,
                        kind=EdgeKind.STATE_READ,
                    ),
                    Edge(
                        id=_artifact_id(
                            result,
                            f"edge/inventory/{spec.key}/write",
                            "config-first-inventory-edge",
                        ),
                        source_id=main_id,
                        target_id=state_id,
                        tensor_id=tensor_id,
                        kind=EdgeKind.STATE_WRITE,
                    ),
                )
            )
        else:
            edges.append(
                Edge(
                    id=_artifact_id(
                        result,
                        f"edge/inventory/{spec.key}",
                        "config-first-inventory-edge",
                    ),
                    source_id=definition_ids[owner_spec.definition_key],
                    target_id=instance_ids[spec.owner_instance_path],
                    tensor_id=tensor_id,
                    kind=EdgeKind(spec.edge_kind),
                )
            )

    diagnostics = []
    for ordinal, unknown in enumerate(inventory.unknowns):
        subject_id = model_id
        if unknown.tensor_key is not None and unknown.tensor_key in tensor_ids:
            subject_id = tensor_ids[unknown.tensor_key]
        elif unknown.owner_instance_path is not None:
            subject_id = instance_ids[unknown.owner_instance_path]
        severity = (
            DiagnosticSeverity.INFO
            if unknown.code
            in {
                "CONFIG_BUFFER_INVENTORY_UNKNOWN",
                "CONFIG_SHARED_USAGE_DECLARED",
            }
            else DiagnosticSeverity.WARNING
        )
        diagnostics.append(
            Diagnostic(
                id=_artifact_id(
                    result,
                    f"diagnostic/inventory/{ordinal}/{unknown.code}",
                    "config-first-inventory-diagnostic",
                ),
                severity=severity,
                code=unknown.code,
                subject_id=subject_id,
                message=unknown.message,
                evidence=list(unknown.evidence),
            )
        )

    if result.adapter_name == "generic-config":
        fallback_reason = str(result.metadata.get("fallback_reason", "no_registered_adapter"))
        rejected_adapter = result.metadata.get("rejected_adapter")
        adapter_failure = result.metadata.get("adapter_failure")
        if rejected_adapter:
            code = "CONFIG_ADAPTER_FALLBACK"
            message = (
                f"Adapter {rejected_adapter!r} could not safely interpret this config; "
                "a generic opaque skeleton was emitted instead."
            )
        else:
            code = "CONFIG_ADAPTER_UNAVAILABLE"
            message = (
                "No registered family adapter matches this config; a generic opaque "
                "skeleton was emitted instead."
            )
        fallback_evidence = [
            f"fallback_reason:{fallback_reason}",
            f"model_type:{result.metadata.get('missing_adapter_for_model_type', '<missing>')}",
            "coverage:config-only-partial",
            "operator_decomposition:unavailable",
        ]
        if rejected_adapter:
            fallback_evidence.append(f"rejected_adapter:{rejected_adapter}")
        if adapter_failure:
            fallback_evidence.append(f"adapter_failure:{adapter_failure}")
        diagnostics.extend(
            (
                Diagnostic(
                    id=_artifact_id(
                        result,
                        f"diagnostic/generic/{code}",
                        "config-first-generic-diagnostic",
                    ),
                    severity=DiagnosticSeverity.WARNING,
                    code=code,
                    subject_id=model_id,
                    message=message,
                    evidence=fallback_evidence,
                ),
                Diagnostic(
                    id=_artifact_id(
                        result,
                        "diagnostic/generic/ARCHITECTURE_INTERNALS_OPAQUE",
                        "config-first-generic-diagnostic",
                    ),
                    severity=DiagnosticSeverity.WARNING,
                    code="ARCHITECTURE_INTERNALS_OPAQUE",
                    subject_id=instance_ids["model.architecture"],
                    message=(
                        "Attention, FFN, state, operator, and cost internals are Unknown; "
                        "the visible L0 path is a conservative navigation skeleton only."
                    ),
                    evidence=[
                        "source:config",
                        "coverage:opaque",
                        "runtime_structure:not_observed",
                        "unknown-is-not-zero",
                    ],
                ),
            )
        )

    return ModelMap(
        model=model,
        source_artifacts=[source_artifact],
        symbols=symbols,
        definitions=definitions,
        instances=instances,
        tensors=tensors,
        edges=edges,
        semantic_nodes=semantic_nodes,
        diagnostics=diagnostics,
    )
