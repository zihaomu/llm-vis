from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from llm_vis.adapters import AdapterResult, build_from_config
from llm_vis.cost import (
    CostAnalysis,
    CostModelError,
    analyze_costs,
    evaluate_expression,
    required_symbols,
)
from llm_vis.ir import CoverageStatus, ExpressionOp, MetricOrigin, ModelMap, Scenario

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"
GOLDEN_DIR = Path(__file__).parents[1] / "golden" / "cost"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture(name: str) -> Tuple[Dict[str, Any], Dict[str, Any], AdapterResult, ModelMap]:
    config = _load(FIXTURE_DIR / f"{name}.json")
    provenance = _load(FIXTURE_DIR / f"{name}.provenance.json")
    adapter = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    return config, provenance, adapter, adapter.to_model_map()


def _scenario(
    phase: str,
    *,
    batch: int,
    new_tokens: int,
    past_tokens: int,
    weight_format: str = "bfloat16",
) -> Scenario:
    return Scenario(
        phase=phase,
        batch=batch,
        new_tokens=new_tokens,
        past_tokens=past_tokens,
        activation_dtype="bfloat16",
        weight_format=weight_format,
        kv_dtype="bfloat16",
        backend="formula",
        hardware="generic",
    )


def _analyze(name: str, scenario: Scenario) -> CostAnalysis:
    config, _, adapter, model_map = _fixture(name)
    return analyze_costs(config, adapter, model_map, [scenario])


def test_tiny_dense_per_layer_reference_formulas() -> None:
    scenario = _scenario("prefill", batch=2, new_tokens=8, past_tokens=0)
    analysis = _analyze("tiny_dense", scenario)
    subject = analysis.for_scenario(scenario.id).subjects[0]

    hidden = 512
    intermediate = 1376
    query_width = 8 * 64
    kv_width = 2 * 64
    attention_parameters = hidden * query_width + 2 * hidden * kv_width
    attention_parameters += query_width * hidden + hidden
    ffn_parameters = 3 * hidden * intermediate + hidden
    expected_parameters = attention_parameters + ffn_parameters
    expected_core_flops = 4 * 2 * 8 * 64 * (8 * 9 // 2)
    expected_projection_flops = (
        2 * (2 * 8) * (hidden * query_width + 2 * hidden * kv_width + query_width * hidden)
    )
    expected_ffn_flops = 6 * 2 * 8 * hidden * intermediate

    assert subject.metric("parameters.resident").value == expected_parameters
    assert subject.metric("parameters.active").value == expected_parameters
    assert subject.metric("weights.storage_bytes").value == expected_parameters * 2
    assert subject.metric("flops.attention").value == (
        expected_projection_flops + expected_core_flops
    )
    assert subject.metric("flops.ffn").value == expected_ffn_flops
    assert subject.metric("flops").value == (
        expected_projection_flops + expected_core_flops + expected_ffn_flops
    )
    assert subject.metric("kv_cache.bytes").value == 2 * 8 * 2 * 2 * 64 * 2
    assert subject.metric("state.bytes").value == 0

    for metric in subject.metrics:
        if metric.value is not None and metric.formula is not None:
            assert evaluate_expression(metric.formula, subject.bindings) == metric.value


@pytest.mark.parametrize(
    ("fixture_name", "phase", "new_tokens", "past_tokens"),
    [
        ("tiny_dense", "prefill", 16, 0),
        ("qwen3_8_27b", "decode", 1, 1024),
    ],
)
def test_op04_parent_cost_reconciles_with_known_primitive_children(
    fixture_name: str, phase: str, new_tokens: int, past_tokens: int
) -> None:
    scenario = _scenario(phase, batch=1, new_tokens=new_tokens, past_tokens=past_tokens)
    subjects = _analyze(fixture_name, scenario).for_scenario(scenario.id).subjects
    subject = next(
        item
        for item in subjects
        if any(metric.name == "flops.attention.q_proj" for metric in item.metrics)
    )
    attention_children = (
        "q_proj",
        "k_proj",
        "v_proj",
        "qk_matmul",
        "pv_matmul",
        "o_proj",
    )
    attention_values = [
        subject.metric(f"flops.attention.{name}").value for name in attention_children
    ]
    assert all(value is not None for value in attention_values)
    assert subject.metric("flops.attention").value == sum(
        int(value) for value in attention_values if value is not None
    )
    ffn_children = ("gate_proj", "up_proj", "down_proj")
    ffn_flops = [subject.metric(f"flops.ffn.{name}").value for name in ffn_children]
    ffn_logical = [subject.metric(f"logical_bytes.ffn.{name}").value for name in ffn_children]
    assert all(value is not None for value in (*ffn_flops, *ffn_logical))
    assert subject.metric("flops.ffn").value == sum(
        int(value) for value in ffn_flops if value is not None
    )
    assert subject.metric("logical_bytes.ffn").value == sum(
        int(value) for value in ffn_logical if value is not None
    )
    projection_values = [
        subject.metric(f"logical_bytes.attention.{name}").value
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    assert all(value is not None for value in projection_values)
    projection_traffic = sum(int(value) for value in projection_values if value is not None)
    assert subject.metric("logical_bytes.attention").value is not None
    assert projection_traffic < int(subject.metric("logical_bytes.attention").value)


def test_tiny_int4_storage_includes_scales_not_just_half_a_byte() -> None:
    scenario = _scenario("decode", batch=1, new_tokens=1, past_tokens=128, weight_format="int4")
    subject = _analyze("tiny_dense", scenario).for_scenario(scenario.id).subjects[0]
    parameters = subject.metric("parameters.resident").value
    storage = subject.metric("weights.storage_bytes")
    assert parameters is not None
    assert storage.value is not None
    assert storage.value > parameters / 2
    assert any(item.startswith("scale_metadata_bytes:") for item in storage.assumptions)
    assert any("int4-groupwise-symmetric" in item for item in storage.assumptions)


@pytest.mark.parametrize(
    ("phase", "new_tokens", "past_tokens"),
    [("prefill", 16, 0), ("chunked_prefill", 8, 128), ("decode", 1, 256)],
)
def test_prefill_chunked_and_decode_keep_decoder_metrics_complete_but_model_partial(
    phase: str, new_tokens: int, past_tokens: int
) -> None:
    scenario = _scenario(
        phase,
        batch=2,
        new_tokens=new_tokens,
        past_tokens=past_tokens,
        weight_format="float16",
    )
    scenario_cost = _analyze("tiny_dense", scenario).for_scenario(scenario.id)
    assert len(scenario_cost.subjects) == 4
    assert scenario_cost.aggregate_scope.name == "decoder_layers"
    assert scenario_cost.aggregate_scope.included_instance_count == 4
    assert scenario_cost.aggregate_scope.excluded_instance_paths == (
        "lm_head",
        "model.embed_tokens",
        "model.norm",
    )
    assert scenario_cost.aggregate_scope.structural_coverage == pytest.approx(4 / 7)
    for subject in scenario_cost.subjects:
        assert subject.metric("flops").coverage == 1
        assert subject.metric("flops").coverage_status == CoverageStatus.COMPLETE
    for name in (
        "parameters.resident",
        "parameters.active",
        "weights.storage_bytes",
        "flops",
        "logical_bytes",
        "kv_cache.bytes",
        "state.bytes",
        "arithmetic_intensity",
    ):
        metric = scenario_cost.aggregate_metric(name)
        assert metric.value is not None
        assert metric.coverage == pytest.approx(4 / 7)
        assert metric.coverage_status == CoverageStatus.PARTIAL
        assert "aggregate_scope:decoder_layers" in metric.assumptions
        assert "aggregate_value_is_decoder_only_not_whole_model" in metric.assumptions


def test_qwen_hybrid_state_and_full_attention_kv_formulas() -> None:
    scenario = _scenario("decode", batch=1, new_tokens=1, past_tokens=1024, weight_format="int4")
    scenario_cost = _analyze("qwen3_8_27b", scenario).for_scenario(scenario.id)
    linear = scenario_cost.subjects[0]
    full = scenario_cost.subjects[3]

    recurrent = 1 * 48 * 128 * 128 * 4
    conv = 1 * (2 * 16 * 128 + 48 * 128) * 4 * 4
    assert linear.metric("recurrent_state.bytes").value == recurrent
    assert linear.metric("conv_state.bytes").value == conv
    assert linear.metric("state.bytes").value == recurrent + conv
    assert linear.metric("kv_cache.bytes").value == 0
    assert linear.metric("flops").origin == MetricOrigin.ESTIMATED

    expected_full_kv = 1 * (1024 + 1) * 2 * 4 * 256 * 2
    assert full.metric("kv_cache.bytes").value == expected_full_kv
    assert full.metric("state.bytes").value == 0
    assert full.metric("flops").origin == MetricOrigin.FORMULA

    aggregate = scenario_cost.aggregate_metric("state.bytes")
    assert aggregate.value == 48 * (recurrent + conv)
    assert scenario_cost.aggregate_metric("kv_cache.bytes").value == 16 * expected_full_kv
    assert scenario_cost.aggregate_scope.structural_coverage == pytest.approx(64 / 70)
    assert set(scenario_cost.aggregate_scope.excluded_instance_paths) == {
        "lm_head",
        "model.mtp.0",
        "model.projector",
        "model.text.embed_tokens",
        "model.text.norm",
        "model.vision",
    }
    assert aggregate.coverage == pytest.approx(64 / 70)
    assert aggregate.coverage_status == CoverageStatus.PARTIAL
    hotspots = scenario_cost.hotspots("flops", limit=5)
    assert [entry.layer_index for entry in hotspots] == [3, 7, 11, 15, 19]
    assert [entry.value for entry in hotspots] == sorted(
        (entry.value for entry in hotspots), reverse=True
    )


def test_glm_emits_partial_config_proven_parameters_and_unknown_execution_costs() -> None:
    scenario = _scenario("decode", batch=1, new_tokens=1, past_tokens=1024)
    scenario_cost = _analyze("glm_5_3_bf16", scenario).for_scenario(scenario.id)
    dense = scenario_cost.subjects[0]
    sparse = scenario_cost.subjects[3]

    assert dense.metric("parameters.resident").value is not None
    assert sparse.metric("parameters.resident").value is not None
    assert sparse.metric("parameters.active").value < sparse.metric("parameters.resident").value
    assert sparse.metric("parameters.resident").coverage == 0.5
    assert sparse.metric("parameters.resident").coverage_status == CoverageStatus.PARTIAL
    assert scenario_cost.aggregate_scope.excluded_instance_paths == (
        "lm_head",
        "model.embed_tokens",
        "model.mtp.0",
        "model.norm",
    )
    assert not any(
        "expert_pool" in path for path in scenario_cost.aggregate_scope.excluded_instance_paths
    )
    assert scenario_cost.aggregate_metric("parameters.resident").coverage == pytest.approx(
        0.5 * 78 / 82
    )
    for name in (
        "flops",
        "logical_bytes",
        "kv_cache.bytes",
        "state.bytes",
        "arithmetic_intensity",
    ):
        metric = sparse.metric(name)
        assert metric.value is None
        assert metric.origin == MetricOrigin.UNKNOWN
        assert metric.coverage == 0
        assert scenario_cost.aggregate_metric(name).value is None
    assert scenario_cost.hotspots("flops") == ()
    assert scenario_cost.hotspots("parameters.resident", limit=1)[0].layer_index == 3


def test_analysis_attaches_reference_complete_metrics_to_model_map() -> None:
    config, _, adapter, model_map = _fixture("tiny_dense")
    scenarios = [
        _scenario("prefill", batch=1, new_tokens=4, past_tokens=0),
        _scenario("decode", batch=2, new_tokens=1, past_tokens=64),
    ]
    analysis = analyze_costs(config, adapter, model_map, scenarios)
    enriched = analysis.attach_to_model_map(model_map)
    repeated = analysis.attach_to_model_map(model_map)

    assert len(enriched.scenarios) == 2
    assert len(enriched.metrics) == len(analysis.metrics)
    assert {metric.scenario_id for metric in enriched.metrics} == {
        scenario.id for scenario in scenarios
    }
    assert enriched.model_dump(mode="json") == repeated.model_dump(mode="json")

    serialized = enriched.model_dump_json()
    reloaded = ModelMap.model_validate_json(serialized)
    bindings = {
        symbol.name: symbol.default_value
        for symbol in reloaded.symbols
        if symbol.default_value is not None
    }
    catalog_names = {symbol.name for symbol in reloaded.symbols}
    referenced_original_names = set()
    for metric in reloaded.metrics:
        if metric.formula is None:
            continue
        references = required_symbols(metric.formula)
        assert references <= catalog_names
        referenced_original_names.update(name.rsplit(".", 1)[-1] for name in references)
        assert evaluate_expression(
            metric.formula,
            bindings,
            symbols=reloaded.symbols,
        ) == pytest.approx(metric.value)

    assert {"B", "T", "L", "H", "I", "N_Q", "N_KV", "D_H"} <= (referenced_original_names)
    catalog_original_names = {symbol.name.rsplit(".", 1)[-1] for symbol in reloaded.symbols}
    assert {"B", "T", "L", "S_KV", "H", "I", "N_Q", "N_KV", "D_H"} <= (catalog_original_names)
    scenario_batch_symbols = [
        symbol
        for symbol in reloaded.symbols
        if symbol.name.startswith("scenario.") and symbol.name.endswith(".B")
    ]
    assert {symbol.default_value for symbol in scenario_batch_symbols} == {1, 2}
    assert all(
        "source=Scenario.batch" in (symbol.description or "") for symbol in scenario_batch_symbols
    )

    aggregate_flops = next(
        metric
        for metric in reloaded.metrics
        if metric.subject_id == reloaded.model.id
        and metric.scenario_id == scenarios[0].id
        and metric.name == "flops"
    )
    aggregate_intensity = next(
        metric
        for metric in reloaded.metrics
        if metric.subject_id == reloaded.model.id
        and metric.scenario_id == scenarios[0].id
        and metric.name == "arithmetic_intensity"
    )
    assert aggregate_flops.formula is not None
    assert aggregate_flops.formula.op == ExpressionOp.ADD
    assert aggregate_intensity.formula is not None
    assert aggregate_intensity.formula.op == ExpressionOp.DIVIDE
    assert "formula_provenance:composed_decoder_layer_formulas" in (aggregate_flops.assumptions)


def test_rejects_unsupported_scenario_dtype() -> None:
    config, _, adapter, model_map = _fixture("tiny_dense")
    scenario = Scenario(
        phase="decode",
        batch=1,
        new_tokens=1,
        past_tokens=1,
        activation_dtype="float32",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
    )
    with pytest.raises(CostModelError, match="BF16/FP16 activations"):
        analyze_costs(config, adapter, model_map, [scenario])


@pytest.mark.parametrize("golden_name", ["tiny_prefill_bf16", "qwen_decode_int4"])
def test_fixed_scenario_golden(golden_name: str) -> None:
    golden = _load(GOLDEN_DIR / f"{golden_name}.json")
    scenario = Scenario(**golden["scenario"])
    scenario_cost = _analyze(golden["fixture"], scenario).for_scenario(scenario.id)
    first = scenario_cost.subjects[golden["expected"]["first_layer_index"]]

    actual = {
        "scenario_id": scenario.id,
        "layer_count": len(scenario_cost.subjects),
        "first_layer_index": first.layer_index,
        "first_layer_path": first.module_path,
        "first_layer_metrics": {
            name: first.metric(name).value for name in golden["expected"]["first_layer_metrics"]
        },
        "aggregate_metrics": {
            name: scenario_cost.aggregate_metric(name).value
            for name in golden["expected"]["aggregate_metrics"]
        },
        "aggregate_coverage": {
            name: scenario_cost.aggregate_metric(name).coverage
            for name in golden["expected"]["aggregate_coverage"]
        },
        "aggregate_scope": {
            "name": scenario_cost.aggregate_scope.name,
            "included_instance_count": scenario_cost.aggregate_scope.included_instance_count,
            "excluded_instance_paths": list(scenario_cost.aggregate_scope.excluded_instance_paths),
            "structural_coverage": scenario_cost.aggregate_scope.structural_coverage,
        },
        "top_flops_layer": scenario_cost.hotspots("flops", limit=1)[0].layer_index,
    }
    assert actual == golden["expected"]
