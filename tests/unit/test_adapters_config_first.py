from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from llm_vis.adapters import (
    DEFAULT_REGISTRY,
    AdapterConfigError,
    build_from_config,
    build_model_map_from_config,
)
from llm_vis.ir.models import DefinitionKind, SemanticKind

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _fixture(name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _load(name), _load(f"{name}.provenance")


def test_default_registry_selects_top_level_and_nested_model_types() -> None:
    assert DEFAULT_REGISTRY.model_types == (
        "glm_moe_dsa",
        "llama",
        "qwen2",
        "qwen3_5",
        "qwen3_5_text",
    )
    qwen, _ = _fixture("qwen3_8_27b")
    assert DEFAULT_REGISTRY.resolve(qwen).name == "qwen3-5"
    assert DEFAULT_REGISTRY.resolve(qwen["text_config"]).name == "qwen3-5"

    with pytest.raises(AdapterConfigError, match="No config-first adapter"):
        DEFAULT_REGISTRY.resolve({"model_type": "unknown_model"})


def test_tiny_dense_uses_one_definition_for_all_layer_instances() -> None:
    config, provenance = _fixture("tiny_dense")
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )

    assert len(result.layer_strip) == 4
    assert {entry.definition_key for entry in result.layer_strip} == {"dense_decoder_block"}
    assert [entry.layer_index for entry in result.layer_strip] == [0, 1, 2, 3]
    assert all(entry.state_kind == "kv_cache" for entry in result.layer_strip)
    assert all(entry.expected_pattern == "dense" for entry in result.layer_strip)
    assert not any(entry.anomaly for entry in result.layer_strip)
    assert all(entry.reason is None for entry in result.layer_strip)
    assert result.metadata["weights_loaded"] is False
    assert result.metadata["remote_code_executed"] is False

    llama_config = dict(config, model_type="llama", architectures=["LlamaForCausalLM"])
    llama = build_from_config(llama_config)
    assert llama.definition("dense_decoder_block").class_name == "LlamaDecoderLayer"


def test_build_never_imports_torch_or_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = _fixture("tiny_dense")
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("config-first build imported torch")
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("config-first build imported transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = build_from_config(config)
    assert len(result.instances) == 9


def test_qwen_official_fixture_has_64_layer_3l_1a_pattern_and_state_split() -> None:
    config, provenance = _fixture("qwen3_8_27b")
    assert provenance["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )

    assert len(result.layer_strip) == 64
    assert [entry.label for entry in result.layer_strip[:8]] == [
        "L",
        "L",
        "L",
        "A",
        "L",
        "L",
        "L",
        "A",
    ]
    assert sum(entry.attention_kind == "linear_attention" for entry in result.layer_strip) == 48
    assert sum(entry.attention_kind == "full_attention" for entry in result.layer_strip) == 16
    assert {
        entry.state_kind
        for entry in result.layer_strip
        if entry.attention_kind == "linear_attention"
    } == {"recurrent_state"}
    assert {
        entry.state_kind for entry in result.layer_strip if entry.attention_kind == "full_attention"
    } == {"kv_cache"}
    assert {entry.definition_key for entry in result.layer_strip} == {
        "linear_decoder_block",
        "full_decoder_block",
    }
    assert all(entry.expected_pattern == entry.attention_kind for entry in result.layer_strip)
    assert not any(entry.anomaly for entry in result.layer_strip)
    assert all(entry.reason is None for entry in result.layer_strip)
    assert all(
        entry.expected_pattern == "full_attention" and not entry.anomaly
        for entry in result.layer_strip[3::4]
    )
    assert result.metadata["is_multimodal"] is True
    assert result.instance("model.mtp.0").definition_key == "mtp_block"

    model_map = result.to_model_map()
    assert sum(node.kind == SemanticKind.RECURRENT_STATE for node in model_map.semantic_nodes) == 48
    assert sum(node.kind == SemanticKind.KV_CACHE for node in model_map.semantic_nodes) == 16
    assert sum(node.kind == SemanticKind.VISION_TOWER for node in model_map.semantic_nodes) == 1
    assert sum(node.kind == SemanticKind.MTP for node in model_map.semantic_nodes) == 1
    assert model_map.tensors
    assert all(tensor.materialized is False for tensor in model_map.tensors)


def test_qwen_text_only_config_uses_same_hybrid_pattern() -> None:
    config, _ = _fixture("qwen3_8_27b")
    result = build_from_config(config["text_config"])

    assert result.model_type == "qwen3_5_text"
    assert result.metadata["is_multimodal"] is False
    assert all(instance.path != "model.vision" for instance in result.instances)
    assert len(result.layer_strip) == 64


def test_qwen_explicit_layer_type_deviation_is_marked_as_anomaly() -> None:
    config, _ = _fixture("qwen3_8_27b")
    config["text_config"]["layer_types"][3] = "linear_attention"

    result = build_from_config(config)

    entry = result.layer_strip[3]
    assert entry.attention_kind == "linear_attention"
    assert entry.expected_pattern == "full_attention"
    assert entry.anomaly is True
    assert entry.reason == "attention_kind_deviates_from_declared_cycle"
    assert not any(item.anomaly for item in result.layer_strip[:3])


def test_glm_official_fixture_is_3_dense_plus_75_sparse_virtual_moe() -> None:
    config, provenance = _fixture("glm_5_3_bf16")
    assert provenance["revision"] == "304b8051cfb2b260b61ce0cbe330e02a98e73639"
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )

    assert len(result.layer_strip) == 78
    assert [entry.mlp_kind for entry in result.layer_strip[:4]] == [
        "dense",
        "dense",
        "dense",
        "sparse",
    ]
    assert sum(entry.mlp_kind == "dense" for entry in result.layer_strip) == 3
    assert sum(entry.mlp_kind == "sparse" for entry in result.layer_strip) == 75
    assert all(entry.expected_pattern == entry.mlp_kind for entry in result.layer_strip)
    assert not any(entry.anomaly for entry in result.layer_strip)
    assert all(entry.reason is None for entry in result.layer_strip)
    assert result.metadata["experts_materialized"] == 0

    pool_definition = result.definition("expert_pool")
    assert pool_definition.attributes == {
        "virtual": True,
        "experts_expanded": False,
        "routed_experts": 256,
        "top_k": 8,
        "shared_experts": 1,
        "expert_intermediate_size": 2048,
    }
    pool_instances = [
        instance for instance in result.instances if instance.definition_key == "expert_pool"
    ]
    assert len(pool_instances) == 75
    assert all(instance.overrides["experts_expanded"] is False for instance in pool_instances)

    model_map = result.to_model_map()
    assert (
        sum(definition.kind == DefinitionKind.EXPERT_POOL for definition in model_map.definitions)
        == 1
    )
    assert (
        sum(definition.kind == DefinitionKind.EXPERT for definition in model_map.definitions) == 0
    )
    assert sum(node.kind == SemanticKind.EXPERT_POOL for node in model_map.semantic_nodes) == 75
    assert sum(node.kind == SemanticKind.DSA for node in model_map.semantic_nodes) == 78


def test_glm_explicit_layer_type_deviation_is_marked_as_anomaly() -> None:
    config, _ = _fixture("glm_5_3_bf16")
    config["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * 75
    config["mlp_layer_types"][3] = "dense"

    result = build_from_config(config)

    entry = result.layer_strip[3]
    assert entry.mlp_kind == "dense"
    assert entry.expected_pattern == "sparse"
    assert entry.anomaly is True
    assert entry.reason == "mlp_kind_deviates_from_declared_cycle"
    assert not any(item.anomaly for item in result.layer_strip[:3])


def test_model_map_ids_are_stable_for_fixed_config_and_revision() -> None:
    config, provenance = _fixture("qwen3_8_27b")
    first = build_model_map_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    second = build_model_map_from_config(
        json.loads(json.dumps(config, sort_keys=True)),
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert len({item.id for item in first.definitions}) == len(first.definitions)
    assert len({item.id for item in first.instances}) == len(first.instances)
    assert all(
        instance.overrides["canonical_semantic_key"].startswith("sem_")
        for instance in first.instances
    )


def test_rejects_inconsistent_explicit_layer_pattern() -> None:
    config, _ = _fixture("qwen3_8_27b")
    config["text_config"]["layer_types"] = config["text_config"]["layer_types"][:-1]

    with pytest.raises(AdapterConfigError, match="expected 64"):
        build_from_config(config)


def test_fixture_provenance_never_claims_weights() -> None:
    for fixture_name in ("tiny_dense", "qwen3_8_27b", "glm_5_3_bf16"):
        _, provenance = _fixture(fixture_name)
        assert provenance["weights_included"] is False
