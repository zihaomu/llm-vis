"""Config-first adapter for small Llama/Qwen2-style dense decoders."""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    AdapterResult,
    ConfigFirstAdapter,
    DefinitionSpec,
    InstanceSpec,
    LayerStripEntry,
    SemanticNodeSpec,
    config_sha256,
    require_positive_int,
)


class TinyDenseAdapter(ConfigFirstAdapter):
    """Represent dense decoder-only configs without loading parameter storage."""

    name = "tiny-dense"
    model_types = ("llama", "qwen2")

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_id: str = "local",
        revision: str = "local",
    ) -> AdapterResult:
        model_type = str(config.get("model_type"))
        if model_type not in self.model_types:
            raise ValueError(f"{self.name} cannot handle model_type {model_type!r}")

        num_layers = require_positive_int(config, "num_hidden_layers")
        hidden_size = require_positive_int(config, "hidden_size")
        intermediate_size = require_positive_int(config, "intermediate_size")
        num_heads = require_positive_int(config, "num_attention_heads")
        num_kv_heads = int(config.get("num_key_value_heads", num_heads))
        vocab_size = require_positive_int(config, "vocab_size")
        if num_kv_heads <= 0:
            raise ValueError("'num_key_value_heads' must be a positive integer")

        prefix = "Qwen2" if model_type == "qwen2" else "Llama"
        definitions = (
            DefinitionSpec(
                "model",
                "causal_lm",
                f"{prefix} causal language model",
                f"{prefix}ForCausalLM",
                ("token_embedding", "decoder", "lm_head"),
            ),
            DefinitionSpec("token_embedding", "embedding", "Token embedding", "Embedding"),
            DefinitionSpec(
                "decoder",
                "decoder_stack",
                f"Decoder × {num_layers}",
                f"{prefix}Model",
                ("dense_decoder_block", "final_norm"),
                {"instance_count": num_layers},
            ),
            DefinitionSpec(
                "dense_decoder_block",
                "decoder_block",
                "Dense decoder block",
                f"{prefix}DecoderLayer",
                ("rms_norm", "full_attention", "rms_norm", "dense_ffn"),
            ),
            DefinitionSpec("rms_norm", "normalization", "RMSNorm", f"{prefix}RMSNorm"),
            DefinitionSpec(
                "full_attention",
                "attention",
                "Causal self-attention",
                f"{prefix}Attention",
                attributes={
                    "num_query_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "state_kind": "kv_cache",
                },
            ),
            DefinitionSpec(
                "dense_ffn",
                "feed_forward",
                "Dense gated FFN",
                f"{prefix}MLP",
                attributes={"intermediate_size": intermediate_size},
            ),
            DefinitionSpec("final_norm", "normalization", "Final RMSNorm", f"{prefix}RMSNorm"),
            DefinitionSpec("lm_head", "output_head", "LM head", "Linear"),
        )

        instances = [
            InstanceSpec("model", "model"),
            InstanceSpec("model.embed_tokens", "token_embedding", "model"),
            InstanceSpec("model.layers", "decoder", "model"),
        ]
        semantic_nodes = []
        layer_strip = []
        for layer_index in range(num_layers):
            path = f"model.layers.{layer_index}"
            instances.append(InstanceSpec(path, "dense_decoder_block", "model.layers", layer_index))
            semantic_nodes.append(
                SemanticNodeSpec(
                    path,
                    "dense_decoder_block",
                    f"Dense layer {layer_index}",
                    path,
                    "kv_cache",
                    {
                        "attention_kind": "full_attention",
                        "mlp_kind": "dense",
                        "hidden_size": hidden_size,
                    },
                )
            )
            layer_strip.append(
                LayerStripEntry(
                    layer_index=layer_index,
                    instance_path=path,
                    definition_key="dense_decoder_block",
                    attention_kind="full_attention",
                    mlp_kind="dense",
                    state_kind="kv_cache",
                    label="D",
                    expected_pattern="dense",
                )
            )
        instances.extend(
            (
                InstanceSpec("model.norm", "final_norm", "model"),
                InstanceSpec("lm_head", "lm_head", "model"),
            )
        )

        return AdapterResult(
            adapter_name=self.name,
            model_type=model_type,
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
                "vocab_size": vocab_size,
                "layer_pattern": "dense",
                "inventory_config": {
                    "attention_bias": config.get("attention_bias")
                    if "attention_bias" in config
                    else None,
                    "dtype": config.get("torch_dtype")
                    if "torch_dtype" in config
                    else config.get("dtype"),
                    "head_dim": config.get("head_dim")
                    if "head_dim" in config
                    else (hidden_size // num_heads if hidden_size % num_heads == 0 else None),
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "num_attention_heads": num_heads,
                    "num_key_value_heads": num_kv_heads,
                    "tie_word_embeddings": config.get("tie_word_embeddings")
                    if "tie_word_embeddings" in config
                    else None,
                    "use_cache": config.get("use_cache") if "use_cache" in config else None,
                    "vocab_size": vocab_size,
                },
            },
        )
