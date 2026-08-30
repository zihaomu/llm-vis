from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_vis.capture import (  # noqa: E402
    CaptureStatus,
    RepresentativeCandidate,
    RepresentativeKind,
    UnsupportedCaptureError,
    available_tiny_representatives,
    build_tiny_representative,
    capture_tiny_representative,
    fixtures,  # noqa: E402
    select_representatives,
    symbolic_expression_from_dimension,
)
from llm_vis.capture.patterns import detect_semantic_components  # noqa: E402

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is an optional dependency")


def test_tiny_fixture_registry_is_closed_and_complete() -> None:
    specs = available_tiny_representatives()
    assert tuple(spec.kind for spec in specs) == tuple(RepresentativeKind)
    assert {spec.kind for spec in specs} == {
        RepresentativeKind.DENSE,
        RepresentativeKind.FULL_ATTENTION,
        RepresentativeKind.LINEAR_STATE,
    }
    assert all(spec.module_path.startswith("llm_vis.fixtures.") for spec in specs)


def test_representative_selector_is_deterministic_and_skips_unsupported() -> None:
    candidates = [
        RepresentativeCandidate(
            kind=RepresentativeKind.FULL_ATTENTION,
            instance_id="full.late",
            module_path="layers.7.attn",
            layer_index=7,
        ),
        RepresentativeCandidate(
            kind=RepresentativeKind.DENSE,
            instance_id="dense.unsupported",
            module_path="layers.0.mlp",
            layer_index=0,
            supported=False,
        ),
        RepresentativeCandidate(
            kind=RepresentativeKind.LINEAR_STATE,
            instance_id="linear.first",
            module_path="layers.1.linear_attn",
            layer_index=1,
        ),
        RepresentativeCandidate(
            kind=RepresentativeKind.FULL_ATTENTION,
            instance_id="full.first",
            module_path="layers.3.attn",
            layer_index=3,
        ),
        RepresentativeCandidate(
            kind=RepresentativeKind.DENSE,
            instance_id="dense.good",
            module_path="layers.2.mlp",
            layer_index=2,
        ),
    ]

    selected = select_representatives(reversed(candidates))
    assert [(item.kind, item.instance_id) for item in selected] == [
        (RepresentativeKind.DENSE, "dense.good"),
        (RepresentativeKind.FULL_ATTENTION, "full.first"),
        (RepresentativeKind.LINEAR_STATE, "linear.first"),
    ]


def test_missing_torch_returns_structured_unavailable_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> object:
        raise ModuleNotFoundError("torch intentionally absent")

    monkeypatch.setattr(fixtures, "_load_torch", unavailable)
    result = capture_tiny_representative(RepresentativeKind.DENSE)

    assert result.status == CaptureStatus.UNAVAILABLE
    assert result.coverage == 0.0
    assert result.exported_program is None
    assert len(result.semantic_nodes) == 1
    assert result.semantic_nodes[0].opaque is True
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["CAPTURE_TORCH_UNAVAILABLE"]


@requires_torch
@pytest.mark.parametrize(
    ("kind", "semantic_kind", "component_kinds", "input_count", "output_count"),
    [
        (RepresentativeKind.DENSE, "ffn", {"norm", "residual"}, 1, 1),
        (RepresentativeKind.FULL_ATTENTION, "full_attention", set(), 1, 1),
        (
            RepresentativeKind.LINEAR_STATE,
            "linear_attention",
            {"recurrent_state"},
            2,
            2,
        ),
    ],
)
def test_torch_export_captures_all_three_tiny_fixtures(
    kind: RepresentativeKind,
    semantic_kind: str,
    component_kinds: set[str],
    input_count: int,
    output_count: int,
) -> None:
    result = capture_tiny_representative(kind)

    assert result.status == CaptureStatus.CAPTURED
    assert result.coverage == 1.0
    assert result.exported_program is not None
    assert result.logical_ops
    assert result.tensors
    assert not result.diagnostics
    semantic, *components = result.semantic_nodes
    assert semantic.kind.value == semantic_kind
    assert semantic.opaque is False
    assert {item.kind.value for item in components} == component_kinds
    assert semantic.child_ids == [item.id for item in components]
    assert len(semantic.input_tensor_ids) == input_count
    assert len(semantic.output_tensor_ids) == output_count

    metadata_values = [
        node.meta.get("val")
        for node in result.exported_program.graph_module.graph.nodes
        if node.meta.get("val") is not None
    ]
    flattened_types = []
    for value in metadata_values:
        values = value if isinstance(value, (tuple, list)) else (value,)
        flattened_types.extend(type(item).__name__ for item in values if hasattr(item, "shape"))
    assert "FakeTensor" in flattened_types


@requires_torch
def test_semantic_component_catalog_uses_ops_not_only_fixture_hints() -> None:
    dense = capture_tiny_representative(RepresentativeKind.DENSE)
    full = capture_tiny_representative(RepresentativeKind.FULL_ATTENTION)
    linear = capture_tiny_representative(RepresentativeKind.LINEAR_STATE)

    assert {kind.value for kind, _ in detect_semantic_components(dense.logical_ops)} == {
        "norm",
        "ffn",
        "residual",
    }
    assert {kind.value for kind, _ in detect_semantic_components(full.logical_ops)} == {
        "full_attention"
    }
    assert {kind.value for kind, _ in detect_semantic_components(linear.logical_ops)} == {
        "linear_attention",
        "recurrent_state",
    }


@requires_torch
@pytest.mark.parametrize("kind", list(RepresentativeKind))
def test_capture_ids_and_normalized_ir_are_deterministic(kind: RepresentativeKind) -> None:
    first = capture_tiny_representative(kind)
    second = capture_tiny_representative(kind)

    assert first.capture_id == second.capture_id
    assert [tensor.id for tensor in first.tensors] == [tensor.id for tensor in second.tensors]
    assert [op.id for op in first.logical_ops] == [op.id for op in second.logical_ops]
    assert [node.id for node in first.semantic_nodes] == [node.id for node in second.semantic_nodes]


@requires_torch
def test_unsupported_export_becomes_opaque_without_fallback() -> None:
    calls = 0

    def unsupported(module: object, args: tuple[object, ...]) -> object:
        nonlocal calls
        calls += 1
        assert module._llm_vis_tiny_fixture is True
        assert args
        raise UnsupportedCaptureError("fault injection")

    result = capture_tiny_representative(
        RepresentativeKind.FULL_ATTENTION,
        export_fn=unsupported,
    )

    assert calls == 1
    assert result.status == CaptureStatus.OPAQUE
    assert result.coverage == 0.0
    assert not result.logical_ops
    assert result.semantic_nodes[0].opaque is True
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["CAPTURE_UNSUPPORTED"]


@requires_torch
@pytest.mark.parametrize("kind", list(RepresentativeKind))
def test_fixtures_never_load_or_materialize_weights(
    kind: RepresentativeKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    def forbidden_weight_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("weight loading is forbidden in representative capture")

    monkeypatch.setattr(torch, "load", forbidden_weight_load)
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden_weight_load)
    transformers_before = {name for name in sys.modules if name.startswith("transformers")}

    built = build_tiny_representative(kind, torch_module=torch)
    assert built.module._llm_vis_tiny_fixture is True
    assert all(parameter.is_meta for parameter in built.module.parameters())
    assert all(argument.device.type == "meta" for argument in built.args)

    result = capture_tiny_representative(kind, torch_module=torch)
    assert result.status == CaptureStatus.CAPTURED
    transformers_after = {name for name in sys.modules if name.startswith("transformers")}
    assert transformers_after == transformers_before


def test_symbolic_dimension_parser_keeps_expression_structure() -> None:
    expression = symbolic_expression_from_dimension("2 * S + 1")
    assert expression.op.value == "add"
    assert expression.args[0].op.value == "multiply"
    assert expression.args[0].args[1].symbol == "S"
