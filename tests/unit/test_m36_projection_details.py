from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from llm_vis.adapters import build_from_config
from llm_vis.cost import CostModelError, analyze_costs
from llm_vis.graph_view import (
    GraphEdgeKind,
    GraphPrimitiveKind,
    GraphView,
    build_graph_view_document,
)
from llm_vis.ir import DType, Scenario

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _document(config: Dict[str, Any]):
    result = build_from_config(config, model_id="m3.6-test", revision="fixture")
    return result, build_graph_view_document(result, result.to_model_map())


def _view(document: Any, key: str) -> GraphView:
    return next(view for view in document.views if view.key == key)


def _edge_pairs(view: GraphView) -> set[tuple[str, str, GraphEdgeKind]]:
    ports = {port.id: port for port in view.ports}
    node_keys = {node.id: node.key for node in view.nodes}
    return {
        (
            node_keys[ports[edge.source_port_id].node_id],
            node_keys[ports[edge.target_port_id].node_id],
            edge.kind,
        )
        for edge in view.edges
    }


def test_qwen_full_attention_exposes_layout_and_gqa_shape_transitions() -> None:
    _, document = _document(_load("qwen3_8_27b"))
    view = _view(document, "op_full_attention")
    nodes = {node.key: node for node in view.nodes}
    ports = {port.key: port for port in view.ports}

    assert (len(view.nodes), len(view.edges), len(view.cost_frontier_node_ids)) == (32, 36, 28)
    assert nodes["q_reshape"].primitive_kind == GraphPrimitiveKind.RESHAPE
    assert nodes["q_transpose"].primitive_kind == GraphPrimitiveKind.TRANSPOSE
    assert nodes["k_gqa_expand"].primitive_kind == GraphPrimitiveKind.BROADCAST
    assert nodes["v_gqa_expand"].primitive_kind == GraphPrimitiveKind.BROADCAST
    assert nodes["context_transpose"].primitive_kind == GraphPrimitiveKind.TRANSPOSE

    assert ports["q_proj.projected"].shape == ["B", "T", 12288]
    assert ports["q_split.q"].shape == ["B", "T", 6144]
    assert ports["q_reshape.heads"].shape == ["B", "T", 24, 256]
    assert ports["q_transpose.heads_first"].shape == ["B", 24, "T", 256]
    assert ports["k_proj.flat"].shape == ["B", "T", 1024]
    assert ports["k_gqa_expand.kv_heads"].shape == ["B", 4, "S_KV", 256]
    assert ports["k_gqa_expand.query_heads"].shape == ["B", 24, "S_KV", 256]
    assert ports["context_transpose.tokens_first"].shape == ["B", "T", 24, 256]

    assert {
        ("q_proj", "q_split", GraphEdgeKind.DATA),
        ("q_split", "q_reshape", GraphEdgeKind.DATA),
        ("q_reshape", "q_transpose", GraphEdgeKind.DATA),
        ("k_append", "k_gqa_expand", GraphEdgeKind.STATE_READ),
        ("k_gqa_expand", "qk_matmul", GraphEdgeKind.DATA),
        ("v_append", "v_gqa_expand", GraphEdgeKind.STATE_READ),
        ("v_gqa_expand", "pv_matmul", GraphEdgeKind.DATA),
        ("pv_matmul", "context_transpose", GraphEdgeKind.DATA),
        ("context_transpose", "merge_heads", GraphEdgeKind.DATA),
    } <= _edge_pairs(view)


def test_qwen_output_gate_false_changes_graph_and_q_projection_cost_without_coercion() -> None:
    gated_config = _load("qwen3_8_27b")
    ungated_config = copy.deepcopy(gated_config)
    ungated_config["text_config"]["attn_output_gate"] = False

    gated_result, gated_document = _document(gated_config)
    ungated_result, ungated_document = _document(ungated_config)
    ungated_view = _view(ungated_document, "op_full_attention")
    ungated_nodes = {node.key for node in ungated_view.nodes}
    ungated_ports = {port.key: port for port in ungated_view.ports}

    assert {"q_split", "gate_silu", "gate_multiply"}.isdisjoint(ungated_nodes)
    assert ungated_ports["q_proj.projected"].shape == ["B", "T", 6144]
    assert ("q_proj", "q_reshape", GraphEdgeKind.DATA) in _edge_pairs(ungated_view)
    assert (len(ungated_view.nodes), len(ungated_view.edges)) == (29, 32)

    scenario = Scenario(
        phase="decode",
        batch=1,
        new_tokens=1,
        past_tokens=128,
        activation_dtype="bfloat16",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
        backend="formula",
        hardware="generic",
    )
    gated_cost = analyze_costs(
        gated_config, gated_result, gated_result.to_model_map(), [scenario]
    ).for_scenario(scenario.id)
    ungated_cost = analyze_costs(
        ungated_config, ungated_result, ungated_result.to_model_map(), [scenario]
    ).for_scenario(scenario.id)
    gated_full = next(subject for subject in gated_cost.subjects if subject.layer_index == 3)
    ungated_full = next(subject for subject in ungated_cost.subjects if subject.layer_index == 3)
    assert gated_full.metric("flops.attention.q_proj").value == (
        2 * ungated_full.metric("flops.attention.q_proj").value
    )

    invalid_config = copy.deepcopy(gated_config)
    invalid_config["text_config"]["attn_output_gate"] = "false"
    invalid_result = build_from_config(invalid_config)
    with pytest.raises(ValueError, match="boolean attn_output_gate"):
        build_graph_view_document(invalid_result, invalid_result.to_model_map())
    with pytest.raises(CostModelError, match="boolean attn_output_gate"):
        analyze_costs(invalid_config, invalid_result, invalid_result.to_model_map(), [scenario])


def test_glm_topk_routes_indices_and_float32_weights_without_runtime_values() -> None:
    result, document = _document(_load("glm_5_3_bf16"))
    view = _view(document, "l1_sparse_dsa_moe")
    nodes = {node.key: node for node in view.nodes}
    ports = {port.key: port for port in view.ports}

    assert result.inventory_config["moe_router_dtype"] == "float32"
    assert ports["router.scores"].dtype == DType.FLOAT32
    assert ports["topk.scores"].dtype == DType.FLOAT32
    assert ports["topk.routes"].dtype == DType.INT64
    assert ports["topk.weights"].dtype == DType.FLOAT32
    assert ports["topk.weights"].shape == ["B", "T", 8]
    assert ports["combine.routing_weights"].dtype == DType.FLOAT32
    assert ("topk", "combine", GraphEdgeKind.ROUTE) in _edge_pairs(view)
    assert nodes["topk"].attributes["runtime_route_known"] is False
    assert nodes["topk"].attributes["selected_expert_ids"] is None
    assert nodes["topk"].attributes["routing_weight_values_known"] is False
