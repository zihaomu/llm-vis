from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import pytest
from pydantic import ValidationError

from llm_vis.adapters import build_from_config
from llm_vis.graph_view import (
    GraphBoundaryKind,
    GraphCostReconciliationStatus,
    GraphDecompositionStatus,
    GraphEdgeKind,
    GraphNode,
    GraphPrimitiveKind,
    GraphView,
    GraphViewDocument,
    build_graph_view_document,
)
from llm_vis.ir import Scenario
from llm_vis.ir.ids import canonical_json, deterministic_id

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _build(name: str):
    config = _load(name)
    provenance = _load(f"{name}.provenance")
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    model_map = result.to_model_map()
    return result, model_map, build_graph_view_document(result, model_map)


def _view(document: GraphViewDocument, key: str) -> GraphView:
    return next(view for view in document.views if view.key == key)


def _node_edges(view: GraphView) -> Set[Tuple[str, str, GraphEdgeKind]]:
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


@pytest.mark.parametrize("fixture_name", ["tiny_dense", "qwen3_8_27b", "glm_5_3_bf16"])
def test_graph_view_is_deterministic_json_and_reference_complete(fixture_name: str) -> None:
    result, model_map, first = _build(fixture_name)
    second = build_graph_view_document(result, model_map)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert canonical_json(first) == canonical_json(second)
    assert GraphViewDocument.model_validate_json(first.model_dump_json()) == first
    assert GraphViewDocument.model_json_schema()["title"] == "GraphViewDocument"
    assert first.views[0].key == "l0"
    assert first.source_model_map_id == deterministic_id("mmap", model_map)
    assert f"model_map:{first.source_model_map_id}" in first.provenance.evidence
    assert f"model:{model_map.model.id}" in first.provenance.evidence

    model_map_subject_ids = {
        model_map.model.id,
        *[item.id for item in model_map.definitions],
        *[item.id for item in model_map.instances],
        *[item.id for item in model_map.semantic_nodes],
        *[item.id for item in model_map.logical_ops],
    }
    tensor_ids = {tensor.id for tensor in model_map.tensors}
    view_ids = {view.id for view in first.views}
    for view in first.views:
        assert view.metadata["primary_flow"] == "acyclic"
        if view.parent_view_id is not None:
            assert view.parent_view_id in view_ids
        for node in view.nodes:
            assert set(node.subject_ids) <= model_map_subject_ids
            if node.drilldown_view_id is not None:
                assert node.drilldown_view_id in view_ids
        for port in view.ports:
            if port.tensor_spec_id is not None:
                assert port.tensor_spec_id in tensor_ids
        for edge in view.edges:
            assert edge.origin.value == "config"
            assert "source:config" in edge.evidence
            if edge.tensor_spec_id is not None:
                assert edge.tensor_spec_id in tensor_ids


def test_qwen_l0_and_representative_l1_paths() -> None:
    result, _, document = _build("qwen3_8_27b")
    assert result.metadata["weights_loaded"] is False
    assert [view.key for view in document.views] == [
        "l0",
        "l1_linear_attention",
        "op_linear_attention",
        "op_linear_ffn",
        "l1_full_attention",
        "op_full_attention",
        "op_full_ffn",
    ]

    l0 = _view(document, "l0")
    pairs = _node_edges(l0)
    required = {
        ("tokens", "embedding", GraphEdgeKind.DATA),
        ("embedding", "decoder_pattern", GraphEdgeKind.DATA),
        ("decoder_pattern", "final_norm", GraphEdgeKind.DATA),
        ("final_norm", "lm_head", GraphEdgeKind.DATA),
        ("lm_head", "logits", GraphEdgeKind.DATA),
        ("vision_input", "vision_tower", GraphEdgeKind.DATA),
        ("vision_tower", "projector", GraphEdgeKind.DATA),
        ("projector", "decoder_pattern", GraphEdgeKind.DATA),
        ("decoder_pattern", "mtp", GraphEdgeKind.DATA),
        ("mtp", "mtp_logits", GraphEdgeKind.DATA),
    }
    assert required <= pairs
    decoder = next(node for node in l0.nodes if node.key == "decoder_pattern")
    assert decoder.attributes["linear_attention_layers"] == 48
    assert decoder.attributes["full_attention_layers"] == 16
    assert decoder.attributes["repeat_count"] == 16
    assert decoder.drilldown_view_id == _view(document, "l1_linear_attention").id
    assert decoder.attributes["drilldown_view_ids"] == {
        "linear_attention": _view(document, "l1_linear_attention").id,
        "full_attention": _view(document, "l1_full_attention").id,
    }
    data_pairs = {(source, target) for source, target, kind in pairs if kind == GraphEdgeKind.DATA}
    reachable = {"tokens", "vision_input"}
    while True:
        expanded = reachable | {target for source, target in data_pairs if source in reachable}
        if expanded == reachable:
            break
        reachable = expanded
    assert {"logits", "mtp_logits"} <= reachable

    linear = _view(document, "l1_linear_attention")
    full = _view(document, "l1_full_attention")
    assert linear.parent_view_id == l0.id and linear.layer_index == 0
    assert full.parent_view_id == l0.id and full.layer_index == 3
    assert linear.metadata["layer_strip_target"] == 0
    assert full.metadata["layer_strip_target"] == 3
    assert next(node for node in linear.nodes if node.key == "attention").kind == "linear_attention"
    assert next(node for node in full.nodes if node.key == "attention").kind == "full_attention"
    assert sum(edge.kind == GraphEdgeKind.STATE_READ for edge in linear.edges) == 2
    assert sum(edge.kind == GraphEdgeKind.STATE_WRITE for edge in linear.edges) == 2
    assert sum(edge.kind == GraphEdgeKind.STATE_READ for edge in full.edges) == 2
    assert sum(edge.kind == GraphEdgeKind.STATE_WRITE for edge in full.edges) == 2


def test_glm_l0_dense_sparse_and_opaque_dsa_moe_route() -> None:
    _, _, document = _build("glm_5_3_bf16")
    assert [view.key for view in document.views] == [
        "l0",
        "l1_dense_dsa",
        "op_glm_dense_ffn",
        "l1_sparse_dsa_moe",
        "op_glm_routed_expert_ffn",
        "op_glm_shared_expert_ffn",
    ]
    l0 = _view(document, "l0")
    pairs = _node_edges(l0)
    assert {
        ("tokens", "embedding", GraphEdgeKind.DATA),
        ("embedding", "dense_decoder_stage", GraphEdgeKind.DATA),
        ("dense_decoder_stage", "sparse_decoder_stage", GraphEdgeKind.DATA),
        ("sparse_decoder_stage", "final_norm", GraphEdgeKind.DATA),
        ("final_norm", "lm_head", GraphEdgeKind.DATA),
        ("lm_head", "logits", GraphEdgeKind.DATA),
    } <= pairs
    dense_stage = next(node for node in l0.nodes if node.key == "dense_decoder_stage")
    sparse_stage = next(node for node in l0.nodes if node.key == "sparse_decoder_stage")
    assert dense_stage.attributes["layer_count"] == 3
    assert sparse_stage.attributes["layer_count"] == 75
    assert sparse_stage.attributes["experts_materialized"] == 0

    for view_key in ("l1_dense_dsa", "l1_sparse_dsa_moe"):
        dsa = next(node for node in _view(document, view_key).nodes if node.key == "attention")
        assert dsa.kind == "dsa"
        assert dsa.opaque is True
        assert dsa.coverage == 0.0
        state_edges = [
            edge
            for edge in _view(document, view_key).edges
            if edge.kind in {GraphEdgeKind.STATE_READ, GraphEdgeKind.STATE_WRITE}
        ]
        assert len(state_edges) == 2
        assert all(edge.coverage == 0.0 for edge in state_edges)

    sparse = _view(document, "l1_sparse_dsa_moe")
    sparse_pairs = _node_edges(sparse)
    assert ("router", "topk", GraphEdgeKind.DATA) in sparse_pairs
    assert ("topk", "expert_pool", GraphEdgeKind.ROUTE) in sparse_pairs
    assert ("topk", "combine", GraphEdgeKind.CONTROL) in sparse_pairs
    assert ("expert_pool", "combine", GraphEdgeKind.DATA) in sparse_pairs
    assert ("shared_expert", "combine", GraphEdgeKind.DATA) in sparse_pairs
    expert_pool = next(node for node in sparse.nodes if node.key == "expert_pool")
    assert expert_pool.attributes == {
        "virtual": True,
        "experts_expanded": False,
        "experts_materialized": 0,
        "routed_experts": 256,
        "top_k": 8,
        "runtime_route_known": False,
    }


def test_op01_qwen_full_attention_decomposes_to_semantic_primitives() -> None:
    _, _, document = _build("qwen3_8_27b")
    parent = _view(document, "l1_full_attention")
    attention = next(node for node in parent.nodes if node.key == "attention")
    operators = _view(document, "op_full_attention")
    assert attention.decomposition_status == GraphDecompositionStatus.AVAILABLE
    assert attention.drilldown_view_id == operators.id
    assert operators.parent_view_id == parent.id
    assert operators.decomposes_node_id == attention.id

    nodes = {node.key: node for node in operators.nodes}
    required = {
        "q_proj": GraphPrimitiveKind.GEMM,
        "k_proj": GraphPrimitiveKind.GEMM,
        "v_proj": GraphPrimitiveKind.GEMM,
        "qk_matmul": GraphPrimitiveKind.MATMUL,
        "softmax": GraphPrimitiveKind.SOFTMAX,
        "pv_matmul": GraphPrimitiveKind.MATMUL,
        "o_proj": GraphPrimitiveKind.GEMM,
        "q_norm": GraphPrimitiveKind.RMS_NORM,
        "k_norm": GraphPrimitiveKind.RMS_NORM,
        "q_reshape": GraphPrimitiveKind.RESHAPE,
        "q_transpose": GraphPrimitiveKind.TRANSPOSE,
        "k_reshape": GraphPrimitiveKind.RESHAPE,
        "k_transpose": GraphPrimitiveKind.TRANSPOSE,
        "v_reshape": GraphPrimitiveKind.RESHAPE,
        "v_transpose": GraphPrimitiveKind.TRANSPOSE,
        "q_rope": GraphPrimitiveKind.ROPE,
        "k_rope": GraphPrimitiveKind.ROPE,
        "k_gqa_expand": GraphPrimitiveKind.BROADCAST,
        "v_gqa_expand": GraphPrimitiveKind.BROADCAST,
        "context_transpose": GraphPrimitiveKind.TRANSPOSE,
        "gate_multiply": GraphPrimitiveKind.MULTIPLY,
    }
    for key, primitive in required.items():
        assert nodes[key].primitive_kind == primitive
        assert nodes[key].decomposition_status == GraphDecompositionStatus.PRIMITIVE
    pairs = _node_edges(operators)
    assert {
        ("q_rope", "qk_matmul", GraphEdgeKind.DATA),
        ("qk_matmul", "scale", GraphEdgeKind.DATA),
        ("scale", "causal_mask", GraphEdgeKind.DATA),
        ("causal_mask", "softmax", GraphEdgeKind.DATA),
        ("softmax", "pv_matmul", GraphEdgeKind.DATA),
        ("pv_matmul", "context_transpose", GraphEdgeKind.DATA),
        ("context_transpose", "merge_heads", GraphEdgeKind.DATA),
        ("gate_multiply", "o_proj", GraphEdgeKind.DATA),
    } <= pairs


@pytest.mark.parametrize("view_key", ["op_linear_ffn", "op_full_ffn"])
def test_op02_qwen_ffn_has_three_gemms_activation_and_multiply(view_key: str) -> None:
    _, _, document = _build("qwen3_8_27b")
    view = _view(document, view_key)
    primitives = {node.key: node.primitive_kind for node in view.nodes}
    assert {key for key, value in primitives.items() if value == GraphPrimitiveKind.GEMM} == {
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert primitives["activation"] == GraphPrimitiveKind.SILU
    assert primitives["multiply"] == GraphPrimitiveKind.MULTIPLY
    assert {
        ("gate_proj", "activation", GraphEdgeKind.DATA),
        ("activation", "multiply", GraphEdgeKind.DATA),
        ("up_proj", "multiply", GraphEdgeKind.DATA),
        ("multiply", "down_proj", GraphEdgeKind.DATA),
    } <= _node_edges(view)


def test_op03_decomposition_boundary_preserves_all_ports_shapes_and_reachability() -> None:
    _, _, document = _build("qwen3_8_27b")
    parent = _view(document, "l1_full_attention")
    child = _view(document, "op_full_attention")
    parent_node = next(node for node in parent.nodes if node.key == "attention")
    parent_ports = {port.id: port for port in parent.ports}
    child_ports = {port.id: port for port in child.ports}
    assert {binding.parent_port_id for binding in child.boundary_bindings} == set(
        parent_node.input_port_ids + parent_node.output_port_ids
    )
    assert {binding.kind for binding in child.boundary_bindings} == {
        GraphBoundaryKind.INPUT,
        GraphBoundaryKind.OUTPUT,
        GraphBoundaryKind.STATE_READ,
        GraphBoundaryKind.STATE_WRITE,
    }
    for binding in child.boundary_bindings:
        before = parent_ports[binding.parent_port_id]
        after = child_ports[binding.child_port_id]
        assert (
            before.role,
            before.dtype,
            before.shape_known,
            before.shape,
            before.tensor_spec_id,
        ) == (
            after.role,
            after.dtype,
            after.shape_known,
            after.shape,
            after.tensor_spec_id,
        )

    payload = document.model_dump(mode="json")
    child_payload = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    boundary_port_id = child_payload["boundary_bindings"][0]["child_port_id"]
    next(port for port in child_payload["ports"] if port["id"] == boundary_port_id)["shape"] = [
        "broken"
    ]
    with pytest.raises(ValidationError, match="preserve Tensor shape"):
        GraphViewDocument.model_validate(payload)


def test_op04_static_cost_reconciliation_never_encodes_unknown_as_zero() -> None:
    _, _, document = _build("qwen3_8_27b")
    attention = _view(document, "op_full_attention")
    by_dimension = {item.dimension: item for item in attention.cost_reconciliations}
    assert by_dimension["flops"].status == GraphCostReconciliationStatus.COMPLETE
    assert len(by_dimension["flops"].known_child_metric_names) == 6
    assert by_dimension["logical_bytes"].status == GraphCostReconciliationStatus.PARTIAL
    assert by_dimension["logical_bytes"].unattributed_node_ids
    unknown_bindings = [
        binding
        for node in attention.nodes
        for binding in node.metric_bindings
        if binding.status.value == "unknown"
    ]
    assert unknown_bindings
    assert all(binding.metric_name is None and binding.reason for binding in unknown_bindings)


@pytest.mark.parametrize("fixture_name", ["qwen3_8_27b", "glm_5_3_bf16"])
def test_op05_visible_cost_frontier_never_contains_parent_and_child_together(
    fixture_name: str,
) -> None:
    _, _, document = _build(fixture_name)
    views = {view.id: view for view in document.views}
    for view in document.views:
        frontier = set(view.cost_frontier_node_ids)
        assert len(frontier) == len(view.cost_frontier_node_ids)
        assert frontier <= {node.id for node in view.nodes}
        assert not frontier & {node.id for node in view.nodes if node.kind == "boundary"}
        for node in view.nodes:
            if node.drilldown_view_id is None:
                continue
            child = views[node.drilldown_view_id]
            assert node.id not in {child_node.id for child_node in child.nodes}
            assert node.id not in set(child.cost_frontier_node_ids)


def test_op06_qwen_kv_and_recurrent_state_connect_to_their_operator_cores() -> None:
    _, _, document = _build("qwen3_8_27b")
    full_pairs = _node_edges(_view(document, "op_full_attention"))
    assert {
        ("state_in", "k_append", GraphEdgeKind.STATE_READ),
        ("state_in", "v_append", GraphEdgeKind.STATE_READ),
        ("k_append", "state_out", GraphEdgeKind.STATE_WRITE),
        ("v_append", "state_out", GraphEdgeKind.STATE_WRITE),
        ("k_append", "k_gqa_expand", GraphEdgeKind.STATE_READ),
        ("v_append", "v_gqa_expand", GraphEdgeKind.STATE_READ),
        ("k_gqa_expand", "qk_matmul", GraphEdgeKind.DATA),
        ("v_gqa_expand", "pv_matmul", GraphEdgeKind.DATA),
    } <= full_pairs
    linear_pairs = _node_edges(_view(document, "op_linear_attention"))
    assert {
        ("state_in", "depthwise_conv", GraphEdgeKind.STATE_READ),
        ("depthwise_conv", "state_out", GraphEdgeKind.STATE_WRITE),
        ("state_in", "delta_core", GraphEdgeKind.STATE_READ),
        ("delta_core", "state_out", GraphEdgeKind.STATE_WRITE),
    } <= linear_pairs


def test_op07_glm_moe_is_static_topk_structure_without_fabricated_route() -> None:
    _, _, document = _build("glm_5_3_bf16")
    sparse = _view(document, "l1_sparse_dsa_moe")
    nodes = {node.key: node for node in sparse.nodes}
    assert nodes["router"].primitive_kind == GraphPrimitiveKind.GEMM
    assert nodes["topk"].primitive_kind == GraphPrimitiveKind.TOP_K
    assert nodes["topk"].attributes["top_k"] == 8
    assert nodes["topk"].attributes["routed_experts"] == 256
    assert nodes["topk"].attributes["runtime_route_known"] is False
    assert nodes["topk"].attributes["selected_expert_ids"] is None
    assert nodes["expert_pool"].attributes["experts_materialized"] == 0
    routed = _view(document, "op_glm_routed_expert_ffn")
    assert (
        next(node for node in routed.nodes if node.key == "gather").attributes[
            "runtime_route_known"
        ]
        is False
    )


def test_op08_glm_dsa_remains_opaque_without_decomposition_target() -> None:
    _, _, document = _build("glm_5_3_bf16")
    for view_key in ("l1_dense_dsa", "l1_sparse_dsa_moe"):
        node = next(node for node in _view(document, view_key).nodes if node.key == "attention")
        assert node.opaque is True
        assert node.coverage == 0
        assert node.decomposition_status == GraphDecompositionStatus.OPAQUE
        assert node.drilldown_view_id is None
        assert node.primitive_kind is None


def test_op09_scenario_attachment_does_not_change_stable_graph_identity() -> None:
    result, model_map, before = _build("qwen3_8_27b")
    scenario = Scenario(
        phase="decode",
        batch=1,
        new_tokens=1,
        past_tokens=128,
        activation_dtype="bfloat16",
        weight_format="int4",
        kv_dtype="bfloat16",
        backend="formula-only",
        hardware="unknown",
    )
    with_scenario = model_map.model_copy(update={"scenarios": [scenario]})
    after = build_graph_view_document(result, with_scenario)

    def identity(document: GraphViewDocument) -> list[tuple[object, ...]]:
        return [
            (
                view.key,
                view.id,
                [node.id for node in view.nodes],
                [port.id for port in view.ports],
                [edge.id for edge in view.edges],
            )
            for view in document.views
        ]

    assert identity(after) == identity(before)


@pytest.mark.parametrize("fixture_name", ["qwen3_8_27b", "glm_5_3_bf16"])
def test_op10_target_projection_imports_no_runtime_and_records_zero_execution(
    fixture_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("M3.6 target projection imported torch")
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("M3.6 target projection imported transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _, _, document = _build(fixture_name)
    assert document.provenance.weights_loaded is False
    assert document.provenance.target_model_constructed is False
    assert document.provenance.target_model_forward is False
    assert document.provenance.remote_code_executed is False


def test_tiny_l0_drills_into_dense_l1_and_carries_zero_execution_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load("tiny_dense")
    provenance = _load("tiny_dense.provenance")
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    model_map = result.to_model_map()
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("GraphView projection imported torch")
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("GraphView projection imported transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    document = build_graph_view_document(result, model_map)

    assert [view.key for view in document.views] == ["l0", "l1_dense"]
    decoder = next(node for node in document.views[0].nodes if node.key == "decoder_pattern")
    assert decoder.drilldown_view_id == document.views[1].id
    assert document.views[1].parent_view_id == document.views[0].id
    assert document.views[1].layer_index == 0
    assert document.provenance.weights_loaded is False
    assert document.provenance.target_model_constructed is False
    assert document.provenance.target_model_forward is False
    assert document.provenance.remote_code_executed is False


def test_graph_view_contract_rejects_primary_flow_cycle() -> None:
    _, _, document = _build("tiny_dense")
    payload = _view(document, "l1_dense").model_dump(mode="json")
    ports = {port["key"]: port["id"] for port in payload["ports"]}
    input_norm = next(node for node in payload["nodes"] if node["key"] == "input_norm")
    cycle_port = next(port for port in payload["ports"] if port["key"] == "input_norm.in").copy()
    cycle_port.update({"id": "gport_cycle_input", "key": "input_norm.cycle_in"})
    payload["ports"].append(cycle_port)
    input_norm["input_port_ids"].append(cycle_port["id"])
    payload["edges"].append(
        {
            "id": "gedge_cycle",
            "key": "cycle",
            "source_port_id": ports["residual_1.hidden"],
            "target_port_id": cycle_port["id"],
            "kind": "data",
            "tensor_spec_id": None,
            "origin": "config",
            "evidence": ["source:config", "test:cycle"],
            "coverage": 1.0,
        }
    )
    with pytest.raises(ValidationError, match="must be acyclic"):
        GraphView.model_validate(payload)


def test_graph_view_contract_rejects_ambiguous_multi_producer_input() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = _view(document, "l0").model_dump(mode="json")
    ports = {port["key"]: port["id"] for port in payload["ports"]}
    payload["edges"].append(
        {
            "id": "gedge_ambiguous",
            "key": "ambiguous_mtp_merge",
            "source_port_id": ports["mtp.logits"],
            "target_port_id": ports["logits.logits"],
            "kind": "data",
            "tensor_spec_id": None,
            "origin": "config",
            "evidence": ["source:config", "test:ambiguous"],
            "coverage": 1.0,
        }
    )
    with pytest.raises(ValidationError, match="at most one producer"):
        GraphView.model_validate(payload)


def test_graph_view_contract_rejects_edge_endpoint_shape_mismatch() -> None:
    _, _, document = _build("tiny_dense")
    payload = _view(document, "l1_dense").model_dump(mode="json")
    edge = payload["edges"][0]
    target = next(port for port in payload["ports"] if port["id"] == edge["target_port_id"])
    target["shape"] = ["broken"]
    with pytest.raises(ValidationError, match="edge endpoints must preserve Tensor shape"):
        GraphView.model_validate(payload)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"opaque": True}, "none decomposition status requires opaque=false"),
        (
            {"primitive_kind": "gemm"},
            "none decomposition status cannot declare primitive_kind",
        ),
        (
            {"drilldown_view_id": "gvw_child"},
            "none decomposition status cannot have a drilldown view",
        ),
        (
            {
                "decomposition_status": "available",
                "drilldown_view_id": "gvw_child",
                "opaque": True,
            },
            "available decomposition requires opaque=false",
        ),
        (
            {
                "decomposition_status": "available",
                "drilldown_view_id": "gvw_child",
                "primitive_kind": "gemm",
            },
            "expandable compound node cannot declare primitive_kind",
        ),
        (
            {
                "decomposition_status": "primitive",
                "primitive_kind": "gemm",
                "opaque": True,
            },
            "primitive decomposition status requires opaque=false",
        ),
        (
            {
                "decomposition_status": "primitive",
                "primitive_kind": "gemm",
                "drilldown_view_id": "gvw_child",
            },
            "primitive node cannot have a drilldown view",
        ),
        (
            {
                "decomposition_status": "opaque",
                "opaque": True,
                "primitive_kind": "gemm",
            },
            "opaque node cannot declare primitive_kind",
        ),
        (
            {
                "decomposition_status": "opaque",
                "opaque": True,
                "drilldown_view_id": "gvw_child",
            },
            "opaque node cannot have a drilldown view",
        ),
    ],
)
def test_graph_node_decomposition_states_are_mutually_exclusive(
    fields: Dict[str, Any], message: str
) -> None:
    payload: Dict[str, Any] = {
        "id": "gnode_test",
        "key": "test",
        "label": "Test",
        "kind": "operator",
    }
    payload.update(fields)
    with pytest.raises(ValidationError, match=message):
        GraphNode.model_validate(payload)


def test_graph_view_document_rejects_non_child_drilldown_target() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    l0 = next(view for view in payload["views"] if view["key"] == "l0")
    source = next(view for view in payload["views"] if view["key"] == "l1_full_attention")
    attention = next(node for node in source["nodes"] if node["key"] == "attention")
    target = next(view for view in payload["views"] if view["id"] == attention["drilldown_view_id"])
    target["parent_view_id"] = l0["id"]
    with pytest.raises(ValidationError, match="must be a direct child view"):
        GraphViewDocument.model_validate(payload)


def test_operator_drilldown_target_must_identify_its_source_node() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    source = next(view for view in payload["views"] if view["key"] == "l1_full_attention")
    attention = next(node for node in source["nodes"] if node["key"] == "attention")
    ffn = next(node for node in source["nodes"] if node["key"] == "ffn")
    target = next(view for view in payload["views"] if view["id"] == attention["drilldown_view_id"])
    target["decomposes_node_id"] = ffn["id"]
    with pytest.raises(ValidationError, match="must identify its source node"):
        GraphViewDocument.model_validate(payload)


def test_graph_view_document_rejects_parent_hierarchy_cycle() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    l0 = next(view for view in payload["views"] if view["key"] == "l0")
    operator = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    l0["parent_view_id"] = operator["id"]
    with pytest.raises(ValidationError, match="hierarchy must be acyclic"):
        GraphViewDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("source_kind", "replacement_kind", "message"),
    [
        (
            "input",
            "state_read",
            "STATE_READ/STATE_WRITE boundary requires cache or state role",
        ),
        (
            "state_read",
            "input",
            "INPUT/OUTPUT boundary cannot bind cache or state role",
        ),
    ],
)
def test_boundary_kind_must_match_tensor_role(
    source_kind: str, replacement_kind: str, message: str
) -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    child = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    binding = next(item for item in child["boundary_bindings"] if item["kind"] == source_kind)
    binding["kind"] = replacement_kind
    with pytest.raises(ValidationError, match=message):
        GraphViewDocument.model_validate(payload)


def test_state_boundaries_must_pair_by_tensor_spec_id() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    parent = next(view for view in payload["views"] if view["key"] == "l1_full_attention")
    child = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    fake_tensor_id = "tensor_test_unpaired_state"

    for port in parent["ports"]:
        if port["key"] in {
            "attention.kv_cache_key_out",
            "state_out.kv_cache_key",
        }:
            port["tensor_spec_id"] = fake_tensor_id
    next(edge for edge in parent["edges"] if edge["key"] == "state_write_0")["tensor_spec_id"] = (
        fake_tensor_id
    )

    for port in child["ports"]:
        if port["key"] in {
            "state_out.key_cache",
            "k_append.cache_out",
            "k_gqa_expand.kv_heads",
        }:
            port["tensor_spec_id"] = fake_tensor_id
    for edge in child["edges"]:
        if edge["key"] in {"key_state_write", "key_cache_to_gqa"}:
            edge["tensor_spec_id"] = fake_tensor_id

    with pytest.raises(ValidationError, match="must pair by tensor_spec_id"):
        GraphViewDocument.model_validate(payload)


def test_state_boundary_path_must_traverse_compute() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    child = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    child["edges"] = [edge for edge in child["edges"] if edge["key"] != "key_state_read"]
    with pytest.raises(ValidationError, match="must traverse a non-boundary compute node"):
        GraphViewDocument.model_validate(payload)


def test_data_boundary_path_must_traverse_compute() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    child = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    child["edges"] = [edge for edge in child["edges"] if edge["key"] != "output"]
    with pytest.raises(ValidationError, match="preserve data boundary reachability"):
        GraphViewDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("fixture_name", "view_key", "dimension", "field", "value", "message"),
    [
        (
            "qwen3_8_27b",
            "op_full_attention",
            "flops",
            "known_child_metric_names",
            [],
            "complete reconciliation requires known child metrics",
        ),
        (
            "qwen3_8_27b",
            "op_full_attention",
            "logical_bytes",
            "unattributed_node_ids",
            [],
            "partial reconciliation requires unattributed nodes",
        ),
        (
            "glm_5_3_bf16",
            "op_glm_dense_ffn",
            "flops",
            "known_child_metric_names",
            ["metric_test_fabricated"],
            "unknown reconciliation cannot retain known child metrics",
        ),
    ],
)
def test_reconciliation_status_requires_coherent_accounting_sets(
    fixture_name: str,
    view_key: str,
    dimension: str,
    field: str,
    value: list[str],
    message: str,
) -> None:
    _, _, document = _build(fixture_name)
    payload = document.model_dump(mode="json")
    operator = next(view for view in payload["views"] if view["key"] == view_key)
    reconciliation = next(
        item for item in operator["cost_reconciliations"] if item["dimension"] == dimension
    )
    reconciliation[field] = value
    with pytest.raises(ValidationError, match=message):
        GraphViewDocument.model_validate(payload)


def test_reconciliation_dimensions_must_be_unique() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    operator = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    operator["cost_reconciliations"].append(dict(operator["cost_reconciliations"][0]))
    with pytest.raises(ValidationError, match="must use unique dimensions"):
        GraphViewDocument.model_validate(payload)


def test_frontier_binding_parent_metric_must_match_reconciliation() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    operator = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    q_proj = next(node for node in operator["nodes"] if node["key"] == "q_proj")
    binding = next(item for item in q_proj["metric_bindings"] if item["dimension"] == "flops")
    binding["parent_metric_name"] = "flops.wrong_parent"
    with pytest.raises(ValidationError, match="parent metric must match"):
        GraphViewDocument.model_validate(payload)


def test_reconciliation_known_metrics_must_exactly_match_frontier_bindings() -> None:
    _, _, document = _build("qwen3_8_27b")
    payload = document.model_dump(mode="json")
    operator = next(view for view in payload["views"] if view["key"] == "op_full_attention")
    reconciliation = next(
        item for item in operator["cost_reconciliations"] if item["dimension"] == "flops"
    )
    reconciliation["known_child_metric_names"] = reconciliation["known_child_metric_names"][:-1]
    with pytest.raises(ValidationError, match="known metrics must match frontier bindings"):
        GraphViewDocument.model_validate(payload)
