from __future__ import annotations

import json

import pytest

from llm_vis.adapters import (
    AdapterConfigError,
    AdapterRegistry,
    TinyDenseAdapter,
    build_from_config,
)
from llm_vis.graph_view import (
    GraphDecompositionStatus,
    GraphEdgeKind,
    build_graph_view_document,
)
from llm_vis.ir import SemanticKind


def _unknown_config() -> dict[str, object]:
    return {
        "model_type": "future_multimodal_wrapper",
        "architectures": ["FutureForConditionalGeneration"],
        "text_config": {
            "model_type": "future_text",
            "num_layers": 3,
            "d_model": 96,
            "ffn_dim": 256,
            "vocab_size": 1024,
            "torch_dtype": "bfloat16",
        },
    }


def test_unknown_and_empty_configs_build_stable_generic_model_maps() -> None:
    config = _unknown_config()
    first = build_from_config(config, model_id="org/future", revision="deadbeef")
    second = build_from_config(
        json.loads(json.dumps(config, sort_keys=True)),
        model_id="org/future",
        revision="deadbeef",
    )

    assert first.adapter_name == "generic-config"
    assert first.model_type == "future_multimodal_wrapper"
    assert first.layer_strip == ()
    assert first.metadata["config_scope"] == "text_config"
    assert first.metadata["nested_text_config_present"] is True
    assert first.metadata["nested_text_config_sha256"]
    assert first.metadata["detailed_display_skeleton"] is True
    assert first.metadata["recognized_field_sources"] == {
        "num_hidden_layers": "text_config.num_layers",
        "hidden_size": "text_config.d_model",
        "intermediate_size": "text_config.ffn_dim",
        "vocab_size": "text_config.vocab_size",
    }
    assert first.to_model_map().model_dump(mode="json") == second.to_model_map().model_dump(
        mode="json"
    )

    model_map = first.to_model_map()
    assert model_map.tensors == []
    assert model_map.semantic_nodes[0].kind == SemanticKind.OPAQUE
    assert model_map.semantic_nodes[0].confidence == 0.0
    assert {item.code for item in model_map.diagnostics} >= {
        "CONFIG_ADAPTER_UNAVAILABLE",
        "CONFIG_PARAMETER_INVENTORY_OPAQUE",
        "ARCHITECTURE_INTERNALS_OPAQUE",
    }

    empty = build_from_config({})
    assert empty.adapter_name == "generic-config"
    assert empty.model_type == "unknown"
    assert empty.instance("model.architecture").definition_key == "architecture"
    assert empty.metadata["detailed_display_skeleton"] is False
    empty_view = build_graph_view_document(empty, empty.to_model_map()).views[0]
    assert [node.key for node in empty_view.nodes] == ["architecture"]
    assert empty_view.nodes[0].opaque is True
    assert empty_view.nodes[0].coverage == 0.0
    assert empty_view.ports == []
    assert empty_view.edges == []

    non_text = build_from_config({"model_type": "future_vision", "image_size": 224})
    non_text_view = build_graph_view_document(non_text, non_text.to_model_map()).views[0]
    assert [node.key for node in non_text_view.nodes] == ["architecture"]


def test_incomplete_known_config_falls_back_but_explicit_invalid_pattern_still_raises() -> None:
    result = build_from_config({"model_type": "llama", "hidden_size": 64})
    assert result.adapter_name == "generic-config"
    assert result.metadata["rejected_adapter"] == "tiny-dense"
    assert "num_hidden_layers" in result.metadata["adapter_failure"]
    diagnostic = next(
        item for item in result.to_model_map().diagnostics if item.code == "CONFIG_ADAPTER_FALLBACK"
    )
    assert any("rejected_adapter:tiny-dense" == item for item in diagnostic.evidence)

    malformed = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 256,
        "layer_types": ["full_attention"],
    }
    with pytest.raises(AdapterConfigError, match="expected 2"):
        build_from_config(malformed)

    unsupported = dict(malformed, layer_types=["linear_attention", "typo"])
    with pytest.raises(AdapterConfigError, match="Unsupported Qwen layer_types"):
        build_from_config(unsupported)

    explicit_invalid = dict(malformed, hidden_size=0, layer_types=["linear_attention"] * 2)
    with pytest.raises(AdapterConfigError, match="hidden_size"):
        build_from_config(explicit_invalid)

    nested_known_under_unknown_wrapper = {
        "model_type": "future_wrapper",
        "text_config": {
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "vocab_size": 256,
        },
    }
    wrapper_result = build_from_config(nested_known_under_unknown_wrapper)
    assert wrapper_result.adapter_name == "generic-config"
    assert wrapper_result.metadata["nested_model_type"] == "llama"

    strict_registry = AdapterRegistry((TinyDenseAdapter(),))
    with pytest.raises(AdapterConfigError, match="No config-first adapter"):
        strict_registry.build({"model_type": "future"})


def test_generic_graph_is_one_opaque_l0_path_with_unknown_costs() -> None:
    result = build_from_config(_unknown_config(), model_id="org/future", revision="deadbeef")
    model_map = result.to_model_map()
    first = build_graph_view_document(result, model_map)
    view_document = build_graph_view_document(result, model_map)
    assert first.model_dump(mode="json") == view_document.model_dump(mode="json")
    assert [view.key for view in view_document.views] == ["l0"]
    view = view_document.views[0]
    assert view.metadata["coverage_status"] == "partial"
    assert view.metadata["architecture_status"] == "opaque"
    assert [node.key for node in view.nodes] == [
        "tokens",
        "embedding",
        "architecture",
        "final_norm",
        "lm_head",
        "output",
    ]
    nodes = {node.key: node for node in view.nodes}
    assert nodes["architecture"].opaque is True
    assert nodes["architecture"].coverage == 0.0
    assert nodes["architecture"].decomposition_status == GraphDecompositionStatus.OPAQUE
    assert {binding.status.value for binding in nodes["architecture"].metric_bindings} == {
        "unknown"
    }
    assert all(
        node.opaque or node.attributes.get("coverage_status") == "unverified"
        for node in view.nodes
    )

    ports = {port.id: port for port in view.ports}
    node_keys = {node.id: node.key for node in view.nodes}
    pairs = [
        (
            node_keys[ports[edge.source_port_id].node_id],
            node_keys[ports[edge.target_port_id].node_id],
            edge.kind,
        )
        for edge in view.edges
    ]
    assert pairs == [
        ("tokens", "embedding", GraphEdgeKind.DATA),
        ("embedding", "architecture", GraphEdgeKind.DATA),
        ("architecture", "final_norm", GraphEdgeKind.DATA),
        ("final_norm", "lm_head", GraphEdgeKind.DATA),
        ("lm_head", "output", GraphEdgeKind.DATA),
    ]
    assert all(edge.coverage == 0.0 for edge in view.edges)
