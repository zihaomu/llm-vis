"""Config-first adapter for the Qwen3.8/Qwen3.5 Transformers family."""

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
    require_mapping,
    require_positive_int,
    require_string_sequence,
)

_LINEAR = "linear_attention"
_FULL = "full_attention"


class Qwen35Adapter(ConfigFirstAdapter):
    """Build Qwen hybrid-attention structure directly from config metadata."""

    name = "qwen3-5"
    model_types = ("qwen3_5", "qwen3_5_text")

    def _text_config(self, config: Mapping[str, Any]) -> Tuple[Mapping[str, Any], bool]:
        model_type = config.get("model_type")
        if model_type == "qwen3_5":
            text_config = require_mapping(config, "text_config")
            nested_type = text_config.get("model_type")
            if nested_type not in (None, "qwen3_5_text"):
                raise AdapterConfigError(f"Unexpected Qwen text_config.model_type {nested_type!r}")
            return text_config, True
        if model_type == "qwen3_5_text":
            return config, False
        raise AdapterConfigError(f"{self.name} cannot handle model_type {model_type!r}")

    def _layer_types(self, text_config: Mapping[str, Any], num_layers: int) -> Tuple[str, ...]:
        if "layer_types" in text_config:
            layer_types = require_string_sequence(
                text_config, "layer_types", expected_length=num_layers
            )
        else:
            interval = optional_positive_int(text_config, "full_attention_interval", 4)
            layer_types = tuple(
                _FULL if (layer_index + 1) % interval == 0 else _LINEAR
                for layer_index in range(num_layers)
            )
        unsupported = sorted(set(layer_types) - {_LINEAR, _FULL})
        if unsupported:
            raise AdapterConfigError(f"Unsupported Qwen layer_types: {unsupported!r}")
        return layer_types

    def _expected_layer_types(
        self,
        text_config: Mapping[str, Any],
        num_layers: int,
        layer_types: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        """Return the declared cycle expectation without treating expected Full layers as odd."""

        if "full_attention_interval" not in text_config:
            return layer_types
        interval = optional_positive_int(text_config, "full_attention_interval", 4)
        return tuple(
            _FULL if (layer_index + 1) % interval == 0 else _LINEAR
            for layer_index in range(num_layers)
        )

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str = "local",
        revision: str = "local",
    ) -> AdapterResult:
        text_config, is_multimodal = self._text_config(config)
        num_layers = require_positive_int(text_config, "num_hidden_layers")
        hidden_size = require_positive_int(text_config, "hidden_size")
        intermediate_size = require_positive_int(text_config, "intermediate_size")
        num_heads = require_positive_int(text_config, "num_attention_heads")
        num_kv_heads = require_positive_int(text_config, "num_key_value_heads")
        vocab_size = require_positive_int(text_config, "vocab_size")
        layer_types = self._layer_types(text_config, num_layers)
        expected_layer_types = self._expected_layer_types(text_config, num_layers, layer_types)
        mtp_layers = int(text_config.get("mtp_num_hidden_layers", 0))
        if mtp_layers < 0:
            raise AdapterConfigError("'mtp_num_hidden_layers' cannot be negative")

        root_kind = "multimodal_causal_lm" if is_multimodal else "causal_lm"
        root_children = ["text_decoder", "lm_head"]
        if is_multimodal:
            root_children[0:0] = ["vision_tower", "multimodal_projector"]
        if mtp_layers:
            root_children.append("mtp_block")

        definitions = [
            DefinitionSpec(
                "model",
                root_kind,
                "Qwen3.8 model",
                "Qwen3_5ForConditionalGeneration" if is_multimodal else "Qwen3_5ForCausalLM",
                tuple(root_children),
            ),
            DefinitionSpec("token_embedding", "embedding", "Token embedding", "Embedding"),
            DefinitionSpec(
                "text_decoder",
                "decoder_stack",
                f"Text decoder × {num_layers}",
                "Qwen3_5TextModel",
                ("token_embedding", "linear_decoder_block", "full_decoder_block", "final_norm"),
                {"instance_count": num_layers},
            ),
            DefinitionSpec(
                "linear_decoder_block",
                "decoder_block",
                "Linear-attention decoder block",
                "Qwen3_5TextDecoderLayer",
                ("rms_norm", "linear_attention", "rms_norm", "dense_ffn"),
                {"attention_kind": _LINEAR, "state_kind": "recurrent_state"},
            ),
            DefinitionSpec(
                "full_decoder_block",
                "decoder_block",
                "Full-attention decoder block",
                "Qwen3_5TextDecoderLayer",
                ("rms_norm", "full_attention", "rms_norm", "dense_ffn"),
                {"attention_kind": _FULL, "state_kind": "kv_cache"},
            ),
            DefinitionSpec("rms_norm", "normalization", "RMSNorm", "Qwen3_5RMSNorm"),
            DefinitionSpec(
                "linear_attention",
                "linear_attention",
                "Gated DeltaNet linear attention",
                "Qwen3_5GatedDeltaNet",
                attributes={
                    "state_kind": "recurrent_state",
                    "state_components": ("conv_state", "recurrent_state"),
                    "key_heads": text_config.get("linear_num_key_heads"),
                    "value_heads": text_config.get("linear_num_value_heads"),
                },
            ),
            DefinitionSpec(
                "full_attention",
                "attention",
                "Grouped-query full attention",
                "Qwen3_5Attention",
                attributes={
                    "state_kind": "kv_cache",
                    "num_query_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                },
            ),
            DefinitionSpec(
                "dense_ffn",
                "feed_forward",
                "Dense gated FFN",
                "Qwen3_5MLP",
                attributes={"intermediate_size": intermediate_size},
            ),
            DefinitionSpec("final_norm", "normalization", "Final RMSNorm", "Qwen3_5RMSNorm"),
            DefinitionSpec("lm_head", "output_head", "LM head", "Linear"),
        ]
        if is_multimodal:
            vision_config = require_mapping(config, "vision_config")
            definitions.extend(
                (
                    DefinitionSpec(
                        "vision_tower",
                        "vision_tower",
                        "Vision encoder",
                        "Qwen3_5VisionModel",
                        attributes={"depth": vision_config.get("depth")},
                    ),
                    DefinitionSpec(
                        "multimodal_projector",
                        "projector",
                        "Vision-to-text projector",
                        "Qwen3_5VisionPatchMerger",
                    ),
                )
            )
        if mtp_layers:
            definitions.append(
                DefinitionSpec(
                    "mtp_block",
                    "conditional_head",
                    "Multi-token prediction block",
                    "Qwen3_5MTPDecoderLayer",
                    attributes={"instance_count": mtp_layers, "conditional": True},
                )
            )

        instances = [InstanceSpec("model", "model")]
        semantic_nodes = []
        if is_multimodal:
            instances.extend(
                (
                    InstanceSpec("model.vision", "vision_tower", "model"),
                    InstanceSpec("model.projector", "multimodal_projector", "model"),
                )
            )
            semantic_nodes.append(
                SemanticNodeSpec("model.vision", "vision_tower", "Vision tower", "model.vision")
            )

        instances.extend(
            (
                InstanceSpec("model.text", "text_decoder", "model"),
                InstanceSpec("model.text.embed_tokens", "token_embedding", "model.text"),
            )
        )
        layer_strip = []
        linear_count = 0
        full_count = 0
        for layer_index, (attention_kind, expected_pattern) in enumerate(
            zip(layer_types, expected_layer_types)
        ):
            path = f"model.text.layers.{layer_index}"
            if attention_kind == _LINEAR:
                definition_key = "linear_decoder_block"
                state_kind = "recurrent_state"
                label = "L"
                linear_count += 1
            else:
                definition_key = "full_decoder_block"
                state_kind = "kv_cache"
                label = "A"
                full_count += 1
            instances.append(InstanceSpec(path, definition_key, "model.text", layer_index))
            semantic_nodes.append(
                SemanticNodeSpec(
                    path,
                    f"{attention_kind}_block",
                    f"{'Linear' if label == 'L' else 'Full attention'} layer {layer_index}",
                    path,
                    state_kind,
                    {
                        "attention_kind": attention_kind,
                        "mlp_kind": "dense",
                        "hidden_size": hidden_size,
                    },
                )
            )
            layer_strip.append(
                LayerStripEntry(
                    layer_index=layer_index,
                    instance_path=path,
                    definition_key=definition_key,
                    attention_kind=attention_kind,
                    mlp_kind="dense",
                    state_kind=state_kind,
                    label=label,
                    expected_pattern=expected_pattern,
                    anomaly=attention_kind != expected_pattern,
                    reason=(
                        "attention_kind_deviates_from_declared_cycle"
                        if attention_kind != expected_pattern
                        else None
                    ),
                )
            )

        instances.extend(
            (
                InstanceSpec("model.text.norm", "final_norm", "model.text"),
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
            model_type=str(config.get("model_type")),
            model_id=model_id,
            revision=revision,
            definitions=tuple(definitions),
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
                "vocab_size": vocab_size,
                "linear_attention_layers": linear_count,
                "full_attention_layers": full_count,
                "mtp_num_hidden_layers": mtp_layers,
                "is_multimodal": is_multimodal,
                "layer_pattern": "linear_attention × 3 + full_attention × 1"
                if num_layers == 64 and tuple(layer_types[:4]) == (_LINEAR, _LINEAR, _LINEAR, _FULL)
                else "explicit",
                "inventory_config": {
                    "attention_bias": text_config.get("attention_bias")
                    if "attention_bias" in text_config
                    else None,
                    "attn_output_gate": text_config.get("attn_output_gate")
                    if "attn_output_gate" in text_config
                    else None,
                    "dtype": text_config.get("dtype")
                    if "dtype" in text_config
                    else text_config.get("torch_dtype"),
                    "head_dim": text_config.get("head_dim"),
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
                    "linear_key_head_dim": text_config.get("linear_key_head_dim"),
                    "linear_num_key_heads": text_config.get("linear_num_key_heads"),
                    "linear_num_value_heads": text_config.get("linear_num_value_heads"),
                    "linear_value_head_dim": text_config.get("linear_value_head_dim"),
                    "mamba_ssm_dtype": text_config.get("mamba_ssm_dtype"),
                    "mtp_num_hidden_layers": mtp_layers,
                    "mtp_use_dedicated_embeddings": text_config.get("mtp_use_dedicated_embeddings")
                    if "mtp_use_dedicated_embeddings" in text_config
                    else None,
                    "num_attention_heads": num_heads,
                    "num_key_value_heads": num_kv_heads,
                    "tie_word_embeddings": text_config.get("tie_word_embeddings")
                    if "tie_word_embeddings" in text_config
                    else (
                        config.get("tie_word_embeddings")
                        if "tie_word_embeddings" in config
                        else None
                    ),
                    "use_cache": text_config.get("use_cache")
                    if "use_cache" in text_config
                    else None,
                    "vocab_size": vocab_size,
                },
            },
        )
