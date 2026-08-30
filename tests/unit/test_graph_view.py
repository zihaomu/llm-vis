from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import pytest
from pydantic import ValidationError

from llm_vis.adapters import build_from_config
from llm_vis.graph_view import (
    GraphEdgeKind,
    GraphView,
    GraphViewDocument,
    build_graph_view_document,
)
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
        "l1_full_attention",
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
        "l1_sparse_dsa_moe",
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
    assert ("router", "expert_pool", GraphEdgeKind.ROUTE) in sparse_pairs
    assert ("router", "combine", GraphEdgeKind.CONTROL) in sparse_pairs
    assert ("expert_pool", "combine", GraphEdgeKind.DATA) in sparse_pairs
    assert ("shared_expert", "combine", GraphEdgeKind.DATA) in sparse_pairs
    expert_pool = next(node for node in sparse.nodes if node.key == "expert_pool")
    assert expert_pool.attributes == {
        "virtual": True,
        "experts_expanded": False,
        "experts_materialized": 0,
        "routed_experts": 256,
        "top_k": 8,
    }


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
