from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

from llm_vis.adapters import build_from_config
from llm_vis.ir import EdgeKind, ModelMap, TensorOrigin, TensorRole

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _fixture(name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    provenance = json.loads((FIXTURE_DIR / f"{name}.provenance.json").read_text(encoding="utf-8"))
    return config, provenance


def _model_map(name: str) -> ModelMap:
    config, provenance = _fixture(name)
    return build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    ).to_model_map()


def _owner_id(model_map: ModelMap, module_path: str) -> str:
    return next(
        instance.id for instance in model_map.instances if instance.module_path == module_path
    )


def _tensor(model_map: ModelMap, module_path: str, semantic_name: str):
    owner_id = _owner_id(model_map, module_path)
    return next(
        tensor
        for tensor in model_map.tensors
        if tensor.owner_id == owner_id and tensor.semantic_name == semantic_name
    )


def test_tiny_inventory_has_owned_parameters_state_edges_and_explicit_buffer_unknown() -> None:
    model_map = _model_map("tiny_dense")

    assert Counter(tensor.role for tensor in model_map.tensors) == {
        TensorRole.PARAMETER: 39,
        TensorRole.CACHE: 8,
    }
    assert all(tensor.origin == TensorOrigin.CONFIG for tensor in model_map.tensors)
    assert all(tensor.materialized is False for tensor in model_map.tensors)
    assert all(tensor.alias_group is None for tensor in model_map.tensors)
    assert all(
        tensor.source_artifact_ids == model_map.model.source_artifact_ids
        for tensor in model_map.tensors
    )
    assert _tensor(model_map, "model.embed_tokens", "token_embedding.weight").concrete_shape == [
        32000,
        512,
    ]
    assert _tensor(
        model_map, "model.layers.0", "self_attention.key_projection.weight"
    ).concrete_shape == [128, 512]
    assert _tensor(model_map, "model.layers.0", "mlp.gate_projection.weight").concrete_shape == [
        1376,
        512,
    ]

    kv_key = _tensor(model_map, "model.layers.0", "kv_cache.key")
    assert [dimension.symbol or dimension.value for dimension in kv_key.symbolic_shape] == [
        "B",
        2,
        "S_KV",
        64,
    ]
    assert kv_key.concrete_shape is None
    assert Counter(edge.kind for edge in model_map.edges) == {
        EdgeKind.WEIGHT: 39,
        EdgeKind.STATE_READ: 8,
        EdgeKind.STATE_WRITE: 8,
    }
    assert {diagnostic.code for diagnostic in model_map.diagnostics} == {
        "CONFIG_BUFFER_INVENTORY_UNKNOWN"
    }
    layer_definition_id = _owner_id(model_map, "model.layers.0")
    layer_definition_id = next(
        instance.definition_id
        for instance in model_map.instances
        if instance.id == layer_definition_id
    )
    layer_definition = next(
        definition for definition in model_map.definitions if definition.id == layer_definition_id
    )
    assert "self_attention.query_projection.weight" in {
        port.name for port in layer_definition.ports
    }


def test_config_declared_embedding_head_tying_uses_one_stable_alias_group() -> None:
    config, _ = _fixture("tiny_dense")
    config["tie_word_embeddings"] = True

    first = build_from_config(config, revision="tie-fixture").to_model_map()
    second = build_from_config(
        json.loads(json.dumps(config, sort_keys=True)), revision="tie-fixture"
    ).to_model_map()
    embedding = _tensor(first, "model.embed_tokens", "token_embedding.weight")
    lm_head = _tensor(first, "lm_head", "output_projection.weight")
    second_embedding = _tensor(second, "model.embed_tokens", "token_embedding.weight")

    assert embedding.id != lm_head.id
    assert embedding.alias_group == lm_head.alias_group
    assert embedding.alias_group is not None
    assert embedding.alias_group == second_embedding.alias_group


def test_missing_tying_and_dtype_are_unknown_not_guessed() -> None:
    config, _ = _fixture("tiny_dense")
    config.pop("tie_word_embeddings")
    config.pop("torch_dtype")

    model_map = build_from_config(config).to_model_map()
    embedding = _tensor(model_map, "model.embed_tokens", "token_embedding.weight")
    lm_head = _tensor(model_map, "lm_head", "output_projection.weight")
    codes = {diagnostic.code for diagnostic in model_map.diagnostics}

    assert embedding.alias_group is None
    assert lm_head.alias_group is None
    assert embedding.dtype.value == "unknown"
    assert "CONFIG_SHARED_WEIGHT_UNKNOWN" in codes
    assert "CONFIG_PARAMETER_DTYPE_UNKNOWN" in codes


def test_qwen_inventory_covers_hybrid_parameters_kv_and_recurrent_state() -> None:
    model_map = _model_map("qwen3_8_27b")

    assert Counter(tensor.role for tensor in model_map.tensors) == {
        TensorRole.PARAMETER: 758,
        TensorRole.STATE: 96,
        TensorRole.CACHE: 32,
    }
    assert _tensor(
        model_map, "model.text.layers.3", "self_attention.query_projection.weight"
    ).concrete_shape == [12288, 5120]
    recurrent = _tensor(model_map, "model.text.layers.0", "recurrent_state.delta")
    conv = _tensor(model_map, "model.text.layers.0", "recurrent_state.conv")
    assert [dimension.symbol or dimension.value for dimension in recurrent.symbolic_shape] == [
        "B",
        48,
        128,
        128,
    ]
    assert [dimension.symbol or dimension.value for dimension in conv.symbolic_shape] == [
        "B",
        10240,
        4,
    ]
    assert recurrent.dtype.value == "float32"
    assert sum(not tensor.shape_known for tensor in model_map.tensors) == 3
    assert Counter(diagnostic.code for diagnostic in model_map.diagnostics) == {
        "CONFIG_TENSOR_SHAPE_UNKNOWN": 3,
        "CONFIG_SHARED_USAGE_DECLARED": 1,
        "CONFIG_BUFFER_INVENTORY_UNKNOWN": 1,
    }


def test_glm_inventory_keeps_dsa_and_state_unknown_and_experts_virtual() -> None:
    model_map = _model_map("glm_5_3_bf16")

    assert Counter(tensor.role for tensor in model_map.tensors) == {
        TensorRole.PARAMETER: 622,
        TensorRole.CACHE: 78,
    }
    routed = _tensor(model_map, "model.layers.3", "moe.routed_experts.gate_up.weight")
    shared = _tensor(model_map, "model.layers.3", "moe.shared_experts.gate_up.weight")
    assert routed.concrete_shape == [256, 4096, 6144]
    assert shared.concrete_shape == [1, 4096, 6144]
    assert "experts_expanded:false" in routed.evidence
    assert routed.materialized is False

    dsa = [
        tensor
        for tensor in model_map.tensors
        if tensor.semantic_name == "dsa_attention.parameter_inventory"
    ]
    kv = [tensor for tensor in model_map.tensors if tensor.semantic_name == "kv_cache"]
    assert len(dsa) == 78
    assert len(kv) == 78
    assert all(not tensor.shape_known and tensor.concrete_shape is None for tensor in (*dsa, *kv))
    assert Counter(diagnostic.code for diagnostic in model_map.diagnostics) == {
        "CONFIG_TENSOR_SHAPE_UNKNOWN": 157,
        "CONFIG_SHARED_USAGE_DECLARED": 1,
        "CONFIG_BUFFER_INVENTORY_UNKNOWN": 1,
    }
