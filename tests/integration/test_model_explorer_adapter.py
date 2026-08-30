from __future__ import annotations

import json
from pathlib import Path

from llm_vis.adapters import build_from_config
from llm_vis.ir import LogicalOp, Model, ModelMap, SymbolicExpression, TensorSpec
from llm_vis.report.model_explorer import model_explorer_graphs

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _graph_payload(name: str):
    config = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    provenance = json.loads((FIXTURE_DIR / f"{name}.provenance.json").read_text(encoding="utf-8"))
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    return model_explorer_graphs(result.to_model_map())


def test_model_explorer_output_is_deterministic_and_bounded() -> None:
    for fixture in ("tiny_dense", "qwen3_8_27b", "glm_5_3_bf16"):
        first = _graph_payload(fixture)
        second = _graph_payload(fixture)
        assert first == second
        assert [graph["id"] for graph in first["graphs"]] == [
            "architecture-definitions",
            "semantic-structure",
        ]
        assert len(first["graphs"][0]["nodes"]) <= 50
        assert all(len(graph["nodes"]) <= 200 for graph in first["graphs"])


def test_model_explorer_nodes_expose_stable_ids_state_and_provenance() -> None:
    payload = _graph_payload("qwen3_8_27b")
    semantic_nodes = payload["graphs"][1]["nodes"]
    assert all(node["id"].startswith("art_") for node in semantic_nodes)
    attribute_keys = {attribute["key"] for node in semantic_nodes for attribute in node["attrs"]}
    assert {
        "kind",
        "representative_module_path",
        "instance_count",
        "confidence",
        "evidence",
        "source_semantic_ids",
    } <= attribute_keys
    assert any(
        attribute["value"] == "recurrent_state"
        for node in semantic_nodes
        for attribute in node["attrs"]
        if attribute["key"] == "kind"
    )


def test_logical_op_graph_uses_tensor_edges_and_enforces_node_bound() -> None:
    tensor_count = 206
    tensors = [
        TensorSpec(
            id=f"tensor.{index:03d}",
            symbolic_shape=[SymbolicExpression.literal(1)],
            dtype="float32",
            storage_dtype="float32",
        )
        for index in range(tensor_count)
    ]
    ops = [
        LogicalOp(
            id=f"op.{index:03d}",
            domain="aten",
            op_type="aten.add.Tensor",
            inputs=[f"tensor.{index:03d}"],
            outputs=[f"tensor.{index + 1:03d}"],
        )
        for index in range(tensor_count - 1)
    ]
    model_map = ModelMap(
        model=Model(
            id="model.logical-bound",
            name="Logical bound fixture",
            family="tiny",
            revision="fixture-r1",
            framework="torch_export",
        ),
        tensors=tensors,
        logical_ops=ops,
    )

    payload = model_explorer_graphs(model_map)
    logical = payload["graphs"][2]
    assert [graph["id"] for graph in payload["graphs"]] == [
        "architecture-definitions",
        "semantic-structure",
        "logical-ops",
    ]
    assert len(logical["nodes"]) == 200
    selected_ids = {node["id"] for node in logical["nodes"]}
    assert all(
        edge["sourceNodeId"] in selected_ids
        for node in logical["nodes"]
        for edge in node["incomingEdges"]
    )
    second_edge = logical["nodes"][1]["incomingEdges"][0]
    assert second_edge == {
        "sourceNodeId": "op.000",
        "sourceNodeOutputId": "tensor.001",
        "targetNodeInputId": "tensor.001",
    }
