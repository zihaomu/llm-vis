from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_vis.ir import (  # noqa: E402
    Definition,
    Diagnostic,
    Edge,
    Instance,
    LogicalOp,
    Lowering,
    Metric,
    Model,
    ModelMap,
    Scenario,
    SemanticNode,
    SourceArtifact,
    Symbol,
    SymbolicExpression,
    TensorSpec,
    TraceDocument,
    TraceEvent,
    scenario_id,
    schema_documents,
)


def decode_scenario(**overrides: object) -> Scenario:
    values = {
        "phase": "decode",
        "batch": 1,
        "new_tokens": 1,
        "past_tokens": 128,
        "output_length": 32,
        "activation_dtype": "bfloat16",
        "weight_format": "bfloat16",
        "kv_dtype": "bfloat16",
        "backend": "eager",
        "hardware": "gfx942",
    }
    values.update(overrides)
    return Scenario(**values)


def test_scenario_derives_visible_length_and_deterministic_id() -> None:
    left = decode_scenario()
    right = decode_scenario(total_visible_length=129)

    assert left.total_visible_length == 129
    assert left.id == right.id
    assert left.id == scenario_id(left)
    assert left.id.startswith("scn_")


def test_scenario_rejects_inconsistent_phase_length_and_id() -> None:
    with pytest.raises(ValidationError, match="total_visible_length"):
        decode_scenario(total_visible_length=130)
    with pytest.raises(ValidationError, match="decode requires"):
        decode_scenario(new_tokens=2)
    with pytest.raises(ValidationError, match="canonical payload"):
        decode_scenario(id="scn_not_the_payload")


def test_models_forbid_extra_fields_and_unknown_enum_values() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Model(
            id="model",
            name="Tiny",
            family="tiny",
            revision="r1",
            framework="pytorch",
            unexpected=True,
        )
    with pytest.raises(ValidationError, match="Input should be"):
        Definition(id="d", kind="not-a-kind", label="bad")


def test_symbolic_shape_is_a_validated_ast() -> None:
    batch = SymbolicExpression.symbol_ref("B")
    tokens = SymbolicExpression.symbol_ref("T")
    hidden = SymbolicExpression.literal(4096)
    product = SymbolicExpression(
        op="multiply",
        args=[batch, tokens, hidden],
    )
    tensor = TensorSpec(
        id="tensor.hidden",
        symbolic_shape=[batch, tokens, hidden],
        concrete_shape=[2, 8, 4096],
        dtype="bfloat16",
        storage_dtype="bfloat16",
        layout="contiguous",
        stride=[32768, 4096, 1],
        role="activation",
    )

    assert product.op.value == "multiply"
    assert tensor.symbolic_shape[0].symbol == "B"
    with pytest.raises(ValidationError, match="exactly two"):
        SymbolicExpression(op="divide", args=[batch])
    with pytest.raises(ValidationError, match="rank"):
        TensorSpec(
            id="bad",
            symbolic_shape=[batch, tokens],
            concrete_shape=[8],
            dtype="float16",
            storage_dtype="float16",
        )


def test_unknown_metric_requires_none_and_never_accepts_zero() -> None:
    unknown = Metric(
        id="metric.unknown",
        subject_id="model",
        name="flops",
        value=None,
        unit="FLOP",
        origin="unknown",
        coverage=0.0,
        coverage_status="unknown",
    )
    assert unknown.value is None

    with pytest.raises(ValidationError, match="unknown metrics must have value=None"):
        Metric(
            id="metric.bad",
            subject_id="model",
            name="flops",
            value=0,
            unit="FLOP",
            origin="unknown",
        )
    with pytest.raises(ValidationError, match="require a numeric value"):
        Metric(
            id="metric.bad2",
            subject_id="model",
            name="flops",
            value=None,
            unit="FLOP",
            origin="formula",
        )


def build_model_map() -> ModelMap:
    scenario = decode_scenario()
    return ModelMap(
        model=Model(
            id="model.tiny",
            name="Tiny Dense",
            family="tiny",
            revision="fixture-r1",
            framework="pytorch",
            source_artifact_ids=["source.config"],
        ),
        source_artifacts=[
            SourceArtifact(
                id="source.config",
                kind="local_config",
                path="fixtures/tiny/config.json",
                revision="fixture-r1",
                trusted=True,
            )
        ],
        symbols=[Symbol(id="symbol.B", name="B", lower_bound=1, default_value=1)],
        scenarios=[scenario],
        definitions=[
            Definition(
                id="def.block",
                kind="block",
                label="Decoder Block",
                children=["def.attn"],
            ),
            Definition(id="def.attn", kind="module", label="Attention"),
        ],
        instances=[
            Instance(
                id="inst.block.0",
                definition_id="def.block",
                module_path="model.layers.0",
                layer_index=0,
            ),
            Instance(
                id="inst.attn.0",
                definition_id="def.attn",
                parent_id="inst.block.0",
                module_path="model.layers.0.self_attn",
            ),
        ],
        tensors=[
            TensorSpec(
                id="tensor.hidden",
                symbolic_shape=[
                    SymbolicExpression.symbol_ref("B"),
                    SymbolicExpression.symbol_ref("T"),
                    SymbolicExpression.literal(128),
                ],
                dtype="bfloat16",
                storage_dtype="bfloat16",
                role="activation",
            )
        ],
        semantic_nodes=[
            SemanticNode(
                id="semantic.attn",
                kind="attention",
                label="Attention",
                definition_id="def.attn",
                instance_ids=["inst.attn.0"],
                input_tensor_ids=["tensor.hidden"],
                output_tensor_ids=["tensor.hidden"],
                evidence=["config.num_attention_heads"],
            )
        ],
        logical_ops=[
            LogicalOp(
                id="op.mm",
                domain="aten",
                op_type="aten.mm.default",
                inputs=["tensor.hidden"],
                outputs=["tensor.hidden"],
                scope_instance_id="inst.attn.0",
            )
        ],
        edges=[
            Edge(
                id="edge.lowering-input",
                source_id="semantic.attn",
                target_id="op.mm",
                tensor_id="tensor.hidden",
                kind="data",
            )
        ],
        lowerings=[
            Lowering(
                id="lowering.attn",
                source_ids=["semantic.attn"],
                target_ids=["op.mm"],
                kind="decompose",
            )
        ],
        metrics=[
            Metric(
                id="metric.params",
                subject_id="semantic.attn",
                scenario_id=scenario.id,
                name="parameters",
                value=65536,
                unit="element",
                origin="exact",
            )
        ],
        trace_events=[
            TraceEvent(
                id="trace.kernel.0",
                category="kernel",
                start=10.0,
                duration=2.5,
                device="gpu:0",
                kernel_name="tiny_gemm",
            )
        ],
        diagnostics=[
            Diagnostic(
                id="diag.coverage",
                severity="info",
                code="IR_COMPLETE",
                subject_id="model.tiny",
                message="Fixture fully covered",
            )
        ],
    )


def test_model_map_round_trip_and_reference_validation() -> None:
    model_map = build_model_map()
    encoded = model_map.model_dump_json()
    decoded = ModelMap.model_validate_json(encoded)

    assert decoded == model_map
    assert decoded.schema_version == "0.1"

    payload = json.loads(encoded)
    payload["instances"][0]["definition_id"] = "def.missing"
    with pytest.raises(ValidationError, match="does not exist"):
        ModelMap.model_validate(payload)


def test_trace_document_checks_parent_references() -> None:
    with pytest.raises(ValidationError, match="trace parent"):
        TraceDocument(
            events=[
                TraceEvent(
                    id="child",
                    parent_id="missing",
                    category="operator",
                    start=0,
                    duration=1,
                )
            ]
        )


def test_checked_in_schemas_match_export_function() -> None:
    expected = schema_documents()
    assert set(expected) == {
        "graph-view.schema.json",
        "model-map.schema.json",
        "scenario.schema.json",
        "trace.schema.json",
    }
    for filename, generated in expected.items():
        checked_in = json.loads((PROJECT_ROOT / "schemas" / filename).read_text())
        assert checked_in == generated
        assert checked_in["$schema"].endswith("2020-12/schema")


def test_schemas_are_standalone_draft_2020_12_and_validate_artifacts() -> None:
    model_map = build_model_map()
    scenario = model_map.scenarios[0]
    trace = TraceDocument(
        events=[
            TraceEvent(
                id="range",
                category="module_range",
                start=0.0,
                duration=3.0,
            ),
            TraceEvent(
                id="kernel",
                parent_id="range",
                category="kernel",
                start=0.5,
                duration=1.0,
            ),
        ]
    )
    instances = {
        "model-map.schema.json": model_map.model_dump(mode="json"),
        "scenario.schema.json": scenario.model_dump(mode="json"),
        "trace.schema.json": trace.model_dump(mode="json"),
    }

    for filename, instance in instances.items():
        schema = json.loads((PROJECT_ROOT / "schemas" / filename).read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
