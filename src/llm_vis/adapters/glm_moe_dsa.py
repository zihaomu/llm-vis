"""Config-first adapter for GLM-5.3's DSA/MoE decoder."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from .base import (
    AdapterConfigError,
    AdapterResult,
    ConfigFirstAdapter,
    DefinitionSpec,
    InstanceSpec,
    LayerStripEntry,
    SemanticNodeSpec,
    config_sha256,
    optional_positive_int,
    require_positive_int,
    require_string_sequence,
)

_DENSE = "dense"
_SPARSE = "sparse"


class GlmMoeDsaAdapter(ConfigFirstAdapter):
    """Build GLM DSA blocks and virtual expert pools from config only."""

    name = "glm-moe-dsa"
    model_types = ("glm_moe_dsa",)

    def _declared_mlp_pattern(self, config: Mapping[str, Any], num_layers: int) -> Tuple[str, ...]:
        first_dense = int(config.get("first_k_dense_replace", 0))
        if first_dense < 0 or first_dense > num_layers:
            raise AdapterConfigError(
                "'first_k_dense_replace' must be between zero and num_hidden_layers"
            )
        frequency = optional_positive_int(config, "moe_layer_freq", 1)
        return tuple(
            _DENSE
            if layer_index < first_dense or (layer_index - first_dense) % frequency != 0
            else _SPARSE
            for layer_index in range(num_layers)
        )

    def _mlp_layer_types(self, config: Mapping[str, Any], num_layers: int) -> Tuple[str, ...]:
        if "mlp_layer_types" in config:
            layer_types = require_string_sequence(
                config, "mlp_layer_types", expected_length=num_layers
            )
        else:
            layer_types = self._declared_mlp_pattern(config, num_layers)
        unsupported = sorted(set(layer_types) - {_DENSE, _SPARSE})
        if unsupported:
            raise AdapterConfigError(f"Unsupported GLM mlp_layer_types: {unsupported!r}")
        return layer_types

    def _expected_mlp_layer_types(
        self,
        config: Mapping[str, Any],
        num_layers: int,
        layer_types: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        if "mlp_layer_types" in config and not {
            "first_k_dense_replace",
            "moe_layer_freq",
        }.intersection(config):
            return layer_types
        return self._declared_mlp_pattern(config, num_layers)

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str = "local",
        revision: str = "local",
    ) -> AdapterResult:
        if config.get("model_type") != "glm_moe_dsa":
            raise AdapterConfigError(
                f"{self.name} cannot handle model_type {config.get('model_type')!r}"
            )
        num_layers = require_positive_int(config, "num_hidden_layers")
        hidden_size = require_positive_int(config, "hidden_size")
        intermediate_size = require_positive_int(config, "intermediate_size")
        moe_intermediate_size = require_positive_int(config, "moe_intermediate_size")
        routed_experts = require_positive_int(config, "n_routed_experts")
        top_k = require_positive_int(config, "num_experts_per_tok")
        shared_experts = int(config.get("n_shared_experts", 0))
        if shared_experts < 0:
            raise AdapterConfigError("'n_shared_experts' cannot be negative")
        if top_k > routed_experts:
            raise AdapterConfigError("num_experts_per_tok cannot exceed n_routed_experts")
        layer_types = self._mlp_layer_types(config, num_layers)
        expected_layer_types = self._expected_mlp_layer_types(config, num_layers, layer_types)
        mtp_layers = int(config.get("num_nextn_predict_layers", 0))
        if mtp_layers < 0:
            raise AdapterConfigError("'num_nextn_predict_layers' cannot be negative")

        definitions = (
            DefinitionSpec(
                "model",
                "causal_lm",
                "GLM-5.3 causal language model",
                "GlmMoeDsaForCausalLM",
                ("token_embedding", "decoder", "lm_head", "mtp_block"),
            ),
            DefinitionSpec("token_embedding", "embedding", "Token embedding", "Embedding"),
            DefinitionSpec(
                "decoder",
                "decoder_stack",
                f"DSA decoder × {num_layers}",
                "GlmMoeDsaModel",
                ("dense_dsa_block", "sparse_dsa_moe_block", "final_norm"),
                {"instance_count": num_layers},
            ),
            DefinitionSpec(
                "dense_dsa_block",
                "decoder_block",
                "Dense DSA decoder block",
                "GlmMoeDsaDecoderLayer",
                ("rms_norm", "dsa_attention", "rms_norm", "dense_ffn"),
                {"mlp_kind": _DENSE},
            ),
            DefinitionSpec(
                "sparse_dsa_moe_block",
                "decoder_block",
                "Sparse DSA + MoE decoder block",
                "GlmMoeDsaDecoderLayer",
                ("rms_norm", "dsa_attention", "rms_norm", "moe", "expert_pool"),
                {"mlp_kind": _SPARSE},
            ),
            DefinitionSpec("rms_norm", "normalization", "RMSNorm", "GlmMoeDsaRMSNorm"),
            DefinitionSpec(
                "dsa_attention",
                "attention",
                "DeepSeek sparse attention",
                "GlmMoeDsaAttention",
                attributes={
                    "state_kind": "kv_cache",
                    "index_topk": config.get("index_topk"),
                    "kv_lora_rank": config.get("kv_lora_rank"),
                },
            ),
            DefinitionSpec(
                "dense_ffn",
                "feed_forward",
                "Dense gated FFN",
                "GlmMoeDsaMLP",
                attributes={"intermediate_size": intermediate_size},
            ),
            DefinitionSpec(
                "moe",
                "mixture_of_experts",
                "Routed MoE",
                "GlmMoeDsaMoE",
                ("expert_pool",),
            ),
            DefinitionSpec(
                "expert_pool",
                "expert_pool",
                f"Expert Pool [{routed_experts} total, top-{top_k} active]",
                "VirtualExpertPool",
                attributes={
                    "virtual": True,
                    "experts_expanded": False,
                    "routed_experts": routed_experts,
                    "top_k": top_k,
                    "shared_experts": shared_experts,
                    "expert_intermediate_size": moe_intermediate_size,
                },
            ),
            DefinitionSpec("final_norm", "normalization", "Final RMSNorm", "GlmMoeDsaRMSNorm"),
            DefinitionSpec("lm_head", "output_head", "LM head", "Linear"),
            DefinitionSpec(
                "mtp_block",
                "conditional_head",
                "Next-token prediction block",
                "GlmMoeDsaNextNPredictLayer",
                attributes={"instance_count": mtp_layers, "conditional": True},
            ),
        )

        instances = [
            InstanceSpec("model", "model"),
            InstanceSpec("model.embed_tokens", "token_embedding", "model"),
            InstanceSpec("model.layers", "decoder", "model"),
        ]
        semantic_nodes = []
        layer_strip = []
        dense_count = 0
        sparse_count = 0
        for layer_index, (mlp_kind, expected_pattern) in enumerate(
            zip(layer_types, expected_layer_types)
        ):
            path = f"model.layers.{layer_index}"
            if mlp_kind == _DENSE:
                definition_key = "dense_dsa_block"
                label = "D"
                dense_count += 1
            else:
                definition_key = "sparse_dsa_moe_block"
                label = "M"
                sparse_count += 1
            instances.append(InstanceSpec(path, definition_key, "model.layers", layer_index))
            semantic_nodes.append(
                SemanticNodeSpec(
                    path,
                    f"{mlp_kind}_dsa_block",
                    f"{'Dense' if mlp_kind == _DENSE else 'Sparse MoE'} layer {layer_index}",
                    path,
                    "kv_cache",
                    {"attention_kind": "dsa", "mlp_kind": mlp_kind},
                )
            )
            if mlp_kind == _SPARSE:
                pool_path = f"{path}.mlp.expert_pool"
                instances.append(
                    InstanceSpec(
                        pool_path,
                        "expert_pool",
                        path,
                        overrides={
                            "virtual": True,
                            "experts_expanded": False,
                            "routed_experts": routed_experts,
                            "top_k": top_k,
                            "shared_experts": shared_experts,
                            "expert_intermediate_size": moe_intermediate_size,
                        },
                    )
                )
                semantic_nodes.append(
                    SemanticNodeSpec(
                        pool_path,
                        "expert_pool",
                        f"Layer {layer_index} Expert Pool",
                        pool_path,
                        attributes={
                            "virtual": True,
                            "experts_expanded": False,
                            "routed_experts": routed_experts,
                            "top_k": top_k,
                            "shared_experts": shared_experts,
                        },
                    )
                )
            layer_strip.append(
                LayerStripEntry(
                    layer_index=layer_index,
                    instance_path=path,
                    definition_key=definition_key,
                    attention_kind="dsa",
                    mlp_kind=mlp_kind,
                    state_kind="kv_cache",
                    label=label,
                    expected_pattern=expected_pattern,
                    anomaly=mlp_kind != expected_pattern,
                    reason=(
                        "mlp_kind_deviates_from_declared_cycle"
                        if mlp_kind != expected_pattern
                        else None
                    ),
                )
            )

        instances.extend(
            (
                InstanceSpec("model.norm", "final_norm", "model"),
                InstanceSpec("lm_head", "lm_head", "model"),
            )
        )
        for mtp_index in range(mtp_layers):
            path = f"model.mtp.{mtp_index}"
            instances.append(InstanceSpec(path, "mtp_block", "model", mtp_index))
            semantic_nodes.append(
                SemanticNodeSpec(
                    path,
                    "conditional_mtp",
                    f"MTP layer {mtp_index}",
                    path,
                    attributes={"conditional": True},
                )
            )

        return AdapterResult(
            adapter_name=self.name,
            model_type="glm_moe_dsa",
            model_id=model_id,
            revision=revision,
            definitions=definitions,
            instances=tuple(instances),
            semantic_nodes=tuple(semantic_nodes),
            layer_strip=tuple(layer_strip),
            metadata={
                "config_first": True,
                "config_sha256": config_sha256(config),
                "weights_loaded": False,
                "remote_code_executed": False,
                "num_hidden_layers": num_layers,
                "hidden_size": hidden_size,
                "dense_layers": dense_count,
                "sparse_moe_layers": sparse_count,
                "routed_experts": routed_experts,
                "top_k": top_k,
                "shared_experts": shared_experts,
                "experts_materialized": 0,
                "layer_pattern": f"dense × {dense_count} + sparse MoE × {sparse_count}",
                "inventory_config": {
                    "attention_bias": config.get("attention_bias")
                    if "attention_bias" in config
                    else None,
                    "dtype": config.get("dtype")
                    if "dtype" in config
                    else config.get("torch_dtype"),
                    "head_dim": config.get("head_dim"),
                    "hidden_size": hidden_size,
                    "index_head_dim": config.get("index_head_dim"),
                    "index_n_heads": config.get("index_n_heads"),
                    "index_share_for_mtp_iteration": config.get("index_share_for_mtp_iteration")
                    if "index_share_for_mtp_iteration" in config
                    else None,
                    "intermediate_size": intermediate_size,
                    "kv_lora_rank": config.get("kv_lora_rank"),
                    "moe_intermediate_size": moe_intermediate_size,
                    "n_routed_experts": routed_experts,
                    "n_shared_experts": shared_experts,
                    "num_attention_heads": config.get("num_attention_heads"),
                    "num_experts_per_tok": top_k,
                    "num_key_value_heads": config.get("num_key_value_heads"),
                    "num_nextn_predict_layers": mtp_layers,
                    "q_lora_rank": config.get("q_lora_rank"),
                    "qk_head_dim": config.get("qk_head_dim"),
                    "qk_nope_head_dim": config.get("qk_nope_head_dim"),
                    "qk_rope_head_dim": config.get("qk_rope_head_dim"),
                    "tie_word_embeddings": config.get("tie_word_embeddings")
                    if "tie_word_embeddings" in config
                    else None,
                    "use_cache": config.get("use_cache") if "use_cache" in config else None,
                    "v_head_dim": config.get("v_head_dim"),
                    "vocab_size": config.get("vocab_size"),
                },
            },
        )
