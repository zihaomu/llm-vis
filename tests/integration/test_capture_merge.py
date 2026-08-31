from __future__ import annotations

import json
from pathlib import Path

from llm_vis.analysis import capture_representatives, inspect_model
from llm_vis.capture import CaptureStatus, RepresentativeKind, capture_tiny_representative
from llm_vis.graph_view import build_graph_view_document
from llm_vis.ir import ModelMap, TensorOrigin

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"
CAPTURE_GOLDEN = Path(__file__).parents[1] / "golden" / "capture" / "qwen_representatives.json"


def _model(name: str):
    config_path = FIXTURE_DIR / f"{name}.json"
    provenance = json.loads((FIXTURE_DIR / f"{name}.provenance.json").read_text(encoding="utf-8"))
    revision = (
        provenance["revision"]
        if provenance["kind"] == "huggingface_config_subset"
        else None
    )
    return inspect_model(str(config_path), revision=revision)


def test_tiny_capture_merges_two_representatives_into_valid_ir() -> None:
    captured = capture_representatives(_model("tiny_dense"))

    assert {item["kind"] for item in captured.captures} == {"dense", "full_attention"}
    assert all(item["status"] == "captured" for item in captured.captures)
    assert captured.model_map.logical_ops
    assert captured.model_map.tensors
    assert len(captured.model_map.lowerings) == 2
    assert captured.manifest()["capability"]["level"] == "C2"
    assert captured.manifest()["safety"]["weights_loaded"] is False
    assert all(
        item["architecture_equivalence"] == "semantic-kind-only"
        and item["valid_for_full_model_performance"] is False
        and item["capture_backend"] == "torch.export.strict"
        and item["tensor_mode"] == "meta-parameters-and-inputs/faketensor-trace"
        and item["torch_version"] != "unavailable"
        for item in captured.captures
    )
    assert "CAPTURE_SCOPE_REPRESENTATIVE_ONLY" in {
        item.code for item in captured.model_map.diagnostics
    }
    assert "LOGICAL_OPS_NOT_CAPTURED" not in {item.code for item in captured.model_map.diagnostics}
    ModelMap.model_validate(captured.model_map.model_dump(mode="json"))


def test_qwen_capture_selects_linear_layer_zero_and_full_layer_three() -> None:
    captured = capture_representatives(_model("qwen3_8_27b"))
    summaries = {item["kind"]: item for item in captured.captures}
    golden = json.loads(CAPTURE_GOLDEN.read_text(encoding="utf-8"))

    assert set(summaries) == {item["kind"] for item in golden["representatives"]}
    for expected in golden["representatives"]:
        summary = summaries[expected["kind"]]
        assert summary["representative_layer_index"] == expected["representative_layer_index"]
        assert summary["representative_module_path"] == expected["module_path"]
        assert summary["semantic_kind"] == expected["semantic_kind"]
        assert summary["logical_op_count"] >= expected["minimum_logical_ops"]
        assert summary["tensor_count"] >= expected["minimum_tensors"]
        for key, value in golden["policy"].items():
            assert summary[key] == value
        assert summary["source_artifact_id"].startswith("source_")
    pattern_kinds = {
        node.kind.value
        for node in captured.model_map.semantic_nodes
        if any(evidence.startswith("op_pattern=") for evidence in node.evidence)
    }
    assert pattern_kinds == set(golden["component_patterns"])
    assert all(item["coverage"] == 1.0 for item in summaries.values())
    assert len(captured.model_map.lowerings) == 3
    capture_sources = {
        item["capture_id"]: item["source_artifact_id"] for item in summaries.values()
    }
    source_artifacts = {item.id: item for item in captured.model_map.source_artifacts}
    assert len(set(capture_sources.values())) == len(summaries)
    assert all(
        len(source_artifacts[source_id].sha256 or "") == 64
        for source_id in capture_sources.values()
    )
    capture_tensors = [
        tensor for tensor in captured.model_map.tensors if tensor.origin == TensorOrigin.CAPTURE
    ]
    assert len(capture_tensors) == sum(item["tensor_count"] for item in summaries.values())
    for tensor in capture_tensors:
        capture_ids = [
            item.removeprefix("capture_id:")
            for item in tensor.evidence
            if item.startswith("capture_id:")
        ]
        assert len(capture_ids) == 1
        assert tensor.materialized is False
        assert tensor.source_artifact_ids == [capture_sources[capture_ids[0]]]
    assert all(
        op.attrs["source_artifact_id"] == capture_sources[op.attrs["capture_id"]]
        and op.attrs["capture_backend"] == "torch.export.strict"
        and op.scope_instance_id is not None
        for op in captured.model_map.logical_ops
    )
    semantics_by_id = {item.id: item for item in captured.model_map.semantic_nodes}
    dense_lowering = next(
        lowering
        for lowering in captured.model_map.lowerings
        if any(
            "representative_kind=dense" in semantics_by_id[source_id].evidence
            for source_id in lowering.source_ids
        )
    )
    dense_config_sources = [
        semantics_by_id[source_id]
        for source_id in dense_lowering.source_ids
        if "source:config" in semantics_by_id[source_id].evidence
    ]
    assert len(dense_config_sources) == 1
    assert dense_config_sources[0].kind.value == "ffn"
    assert dense_config_sources[0].label == "Dense FFN layer 0"
    assert all(
        semantics_by_id[source_id].kind.value not in {"full_attention", "linear_attention"}
        for source_id in dense_lowering.source_ids
    )
    coverage_contract = golden["coverage_contract"]
    assert coverage_contract["model_scope_diagnostic"] in {
        item.code for item in captured.model_map.diagnostics
    }
    uncovered = [
        item
        for item in captured.model_map.diagnostics
        if item.code == coverage_contract["unselected_region_diagnostic"]
    ]
    assert len(uncovered) == 62
    assert all(item.subject_id is not None for item in uncovered)
    assert all(
        coverage_contract["unselected_region_evidence"] in item.evidence for item in uncovered
    )
    assert all(coverage_contract["scope_evidence"] in item.evidence for item in uncovered)

    graph_view = build_graph_view_document(captured.adapter_result, captured.model_map)
    linear_view = next(view for view in graph_view.views if view.key == "l1_linear_attention")
    graph_nodes = {node.key: node for node in linear_view.nodes}
    semantic_kinds = {item.id: item.kind.value for item in captured.model_map.semantic_nodes}
    attention_kinds = {
        semantic_kinds[subject_id]
        for subject_id in graph_nodes["attention"].subject_ids
        if subject_id in semantic_kinds
    }
    ffn_kinds = {
        semantic_kinds[subject_id]
        for subject_id in graph_nodes["ffn"].subject_ids
        if subject_id in semantic_kinds
    }
    assert attention_kinds == {"linear_attention"}
    assert ffn_kinds == {"ffn"}


def test_glm_capture_is_explicitly_deferred_without_executing_anything() -> None:
    called = False

    def forbidden(kind: RepresentativeKind):
        nonlocal called
        called = True
        raise AssertionError(kind)

    captured = capture_representatives(_model("glm_5_3_bf16"), capture_fn=forbidden)
    assert called is False
    assert not captured.model_map.logical_ops
    assert "CAPTURE_DEFERRED_TO_M5" in {item.code for item in captured.model_map.diagnostics}


def test_inapplicable_tiny_capture_kind_is_not_misreported_as_glm_deferred() -> None:
    called = False

    def forbidden(kind: RepresentativeKind):
        nonlocal called
        called = True
        raise AssertionError(kind)

    captured = capture_representatives(
        _model("tiny_dense"),
        kinds=[RepresentativeKind.LINEAR_STATE],
        capture_fn=forbidden,
    )
    codes = {item.code for item in captured.model_map.diagnostics}
    assert called is False
    assert "CAPTURE_KIND_NOT_APPLICABLE" in codes
    assert "CAPTURE_DEFERRED_TO_M5" not in codes


def test_opaque_capture_is_preserved_and_never_falls_back() -> None:
    calls = 0

    def opaque(kind: RepresentativeKind):
        nonlocal calls
        calls += 1

        def fail(module: object, args: tuple[object, ...]):
            raise RuntimeError("fault injection")

        return capture_tiny_representative(kind, export_fn=fail)

    captured = capture_representatives(
        _model("tiny_dense"), kinds=[RepresentativeKind.FULL_ATTENTION], capture_fn=opaque
    )
    assert calls == 1
    assert captured.captures[0]["status"] == CaptureStatus.OPAQUE.value
    assert captured.captures[0]["coverage"] == 0.0
    assert not captured.model_map.logical_ops
    assert any(node.opaque for node in captured.model_map.semantic_nodes)
    assert "CAPTURE_EXPORT_FAILED" in {item.code for item in captured.model_map.diagnostics}
