from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from llm_vis.analysis import inspect_model
from llm_vis.ir import Metric, MetricOrigin, Scenario
from llm_vis.performance import HardwareProfile, HardwareProvenance

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _inspect(
    fixture: str,
    scenario: Scenario,
    *,
    hardware_profile: HardwareProfile | None = None,
):
    provenance = json.loads(
        (FIXTURE_DIR / f"{fixture}.provenance.json").read_text(encoding="utf-8")
    )
    return inspect_model(
        str(FIXTURE_DIR / f"{fixture}.json"),
        revision=provenance["revision"],
        scenarios=[scenario],
        hardware_profile=hardware_profile,
    )


def _scenario(
    phase: str = "decode",
    *,
    batch: int = 1,
    new_tokens: int = 1,
    past_tokens: int = 127,
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
        backend="formula-only",
        hardware="unknown",
    )


def _profile(*, bandwidth: float = 1_000_000_000_000.0) -> HardwareProfile:
    return HardwareProfile(
        id="synthetic-bf16-pipeline",
        name="Synthetic BF16 pipeline profile",
        peak_flops_per_second=1_000_000_000_000_000.0,
        memory_bandwidth_bytes_per_second=bandwidth,
        dtype="bfloat16",
        provenance=HardwareProvenance(
            kind="synthetic",
            source="integration-test-only fictional values",
        ),
    )


def _model_metric(bundle: Any, scenario_id: str, name: str) -> Metric:
    matches = [
        metric
        for metric in bundle.model_map.metrics
        if metric.subject_id == bundle.model_map.model.id
        and metric.scenario_id == scenario_id
        and metric.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _summaries_by_metric(bundle: Any) -> Mapping[str, Mapping[str, Any]]:
    return {item["metric_name"]: item for item in bundle.hotspot_summaries}


def test_tiny_scenario_automatically_attaches_costs_and_stable_hotspots() -> None:
    scenario = _scenario("prefill", batch=2, new_tokens=8, past_tokens=0)
    bundle = _inspect("tiny_dense", scenario)

    assert bundle.cost_analysis is not None
    assert bundle.hardware_profile is None
    assert bundle.roofline_summaries == ()
    assert _model_metric(bundle, scenario.id, "flops").value is not None
    assert _model_metric(bundle, scenario.id, "logical_bytes").value is not None
    assert _model_metric(bundle, scenario.id, "flops").coverage == pytest.approx(4 / 7)
    assert _model_metric(bundle, scenario.id, "runtime.measured_latency").value is None

    summaries = _summaries_by_metric(bundle)
    assert tuple(summaries) == ("flops", "logical_bytes")
    for metric_name, summary in summaries.items():
        assert summary["scenario_id"] == scenario.id
        assert summary["aggregate"]["name"] == metric_name
        assert summary["aggregate"]["value"] is not None
        assert [entry["rank"] for entry in summary["hotspots"]] == [1, 2, 3, 4]
    json.dumps(bundle.hotspot_summaries, sort_keys=True)

    manifest = bundle.manifest()
    assert manifest["capability"]["cost_analyzed"] is True
    assert manifest["capability"]["roofline_lower_bounds"] is False
    assert manifest["cost"]["unknown_values_preserved"] is True
    assert manifest["cost"]["aggregate_scopes"] == [
        {
            "scenario_id": scenario.id,
            "name": "decoder_layers",
            "subject_id": bundle.model_map.model.id,
            "included_instance_count": 4,
            "excluded_instance_count": 3,
            "excluded_instance_paths": ["lm_head", "model.embed_tokens", "model.norm"],
            "structural_coverage": pytest.approx(4 / 7),
            "coverage_basis": "structural_instances_not_cost_fraction",
        }
    ]
    assert manifest["hardware_profile"] is None
    assert manifest["roofline"]["is_latency_estimate"] is False


def test_qwen_roofline_uses_model_aggregate_metrics_and_is_deterministic() -> None:
    scenario = _scenario(past_tokens=1024, weight_format="int4")
    profile = _profile()
    first = _inspect("qwen3_8_27b", scenario, hardware_profile=profile)
    second = _inspect("qwen3_8_27b", scenario, hardware_profile=profile)

    assert first.analysis_id == second.analysis_id
    assert first.hotspot_summaries == second.hotspot_summaries
    assert first.roofline_summaries == second.roofline_summaries
    assert first.model_map.model_dump(mode="json") == second.model_map.model_dump(mode="json")

    flops = _model_metric(first, scenario.id, "flops")
    logical_bytes = _model_metric(first, scenario.id, "logical_bytes")
    roofline = first.roofline_summaries[0]
    assert roofline["source_subject_id"] == first.model_map.model.id
    assert roofline["source_scope"]["name"] == "decoder_layers"
    assert roofline["source_scope"]["structural_coverage"] == pytest.approx(64 / 70)
    assert set(roofline["source_scope"]["excluded_instance_paths"]) >= {
        "model.vision",
        "model.projector",
        "model.text.embed_tokens",
        "lm_head",
    }
    assert roofline["source_metrics"]["flops"]["value"] == flops.value
    assert roofline["source_metrics"]["logical_bytes"]["value"] == logical_bytes.value
    assert roofline["compute_lower_bound_seconds"] == pytest.approx(
        float(flops.value) / float(profile.peak_flops_per_second)
    )
    assert roofline["bandwidth_lower_bound_seconds"] == pytest.approx(
        float(logical_bytes.value) / float(profile.memory_bandwidth_bytes_per_second)
    )
    assert roofline["is_latency_estimate"] is False
    assert roofline["interpretation"] == "theoretical_lower_bounds_not_latency_estimate"

    flops_hotspots = _summaries_by_metric(first)["flops"]["hotspots"]
    assert flops_hotspots[0]["layer_index"] == 3
    different_peaks = _inspect(
        "qwen3_8_27b",
        scenario,
        hardware_profile=_profile(bandwidth=2_000_000_000_000.0),
    )
    assert different_peaks.analysis_id != first.analysis_id
    assert first.manifest()["hardware_profile"] == profile.model_dump(mode="json")
    json.dumps(first.manifest(), sort_keys=True)


def test_glm_unknown_execution_costs_remain_null_through_hotspots_and_roofline() -> None:
    scenario = _scenario(past_tokens=1024)
    bundle = _inspect("glm_5_3_bf16", scenario, hardware_profile=_profile())

    assert _model_metric(bundle, scenario.id, "parameters.resident").value is not None
    assert _model_metric(bundle, scenario.id, "parameters.active").value is not None
    for name in ("flops", "logical_bytes"):
        metric = _model_metric(bundle, scenario.id, name)
        assert metric.origin == MetricOrigin.UNKNOWN
        assert metric.value is None
        summary = _summaries_by_metric(bundle)[name]
        assert summary["aggregate"]["origin"] == "unknown"
        assert summary["aggregate"]["value"] is None
        assert summary["hotspots"] == []

    roofline = bundle.roofline_summaries[0]
    assert roofline["source_metrics"]["flops"]["value"] is None
    assert roofline["source_metrics"]["logical_bytes"]["value"] is None
    assert roofline["compute_lower_bound_seconds"] is None
    assert roofline["bandwidth_lower_bound_seconds"] is None
    assert roofline["max_lower_bound_seconds"] is None
    assert roofline["bottleneck_class"] == "unknown"
    assert {"flops_unknown", "logical_bytes_unknown"} <= set(roofline["unknown_reasons"])
    assert '"value": null' in json.dumps(bundle.manifest(), sort_keys=True)


def test_analysis_id_distinguishes_capture_outcomes_with_the_same_capture_id() -> None:
    bundle = inspect_model(str(FIXTURE_DIR / "tiny_dense.json"))
    common = {"capture_id": "capture.same", "kind": "dense"}
    captured = replace(
        bundle,
        captures=(
            {
                **common,
                "status": "captured",
                "coverage": 1.0,
                "logical_op_count": 7,
            },
        ),
    )
    opaque = replace(
        bundle,
        captures=(
            {
                **common,
                "status": "opaque",
                "coverage": 0.0,
                "logical_op_count": 0,
            },
        ),
    )

    assert captured.analysis_id != opaque.analysis_id
