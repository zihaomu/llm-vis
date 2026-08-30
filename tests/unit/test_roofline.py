from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from llm_vis.ir import Metric
from llm_vis.performance import (
    HardwareProfile,
    HardwareProvenance,
    analyze_roofline,
    hardware_profile_schema,
    roofline_from_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def synthetic_profile(**overrides: object) -> HardwareProfile:
    values = {
        "id": "synthetic-bf16",
        "name": "Synthetic BF16 test profile",
        "peak_flops_per_second": 1_000_000_000_000_000.0,
        "memory_bandwidth_bytes_per_second": 1_000_000_000_000.0,
        "dtype": "bfloat16",
        "provenance": HardwareProvenance(
            kind="synthetic",
            source="unit-test-only fictional values",
        ),
    }
    values.update(overrides)
    return HardwareProfile(**values)


def test_hardware_profile_is_explicit_strict_and_schema_valid() -> None:
    profile = synthetic_profile()
    assert profile.dtype.value == "bfloat16"
    assert profile.provenance.kind.value == "synthetic"

    with pytest.raises(ValidationError, match="Field required"):
        HardwareProfile(id="missing", name="Missing required fields")
    with pytest.raises(ValidationError, match="greater than 0"):
        synthetic_profile(peak_flops_per_second=0)
    with pytest.raises(ValidationError, match="concrete"):
        synthetic_profile(dtype="unknown")
    with pytest.raises(ValidationError, match="Extra inputs"):
        synthetic_profile(device_defaults=True)

    schema = hardware_profile_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(profile.model_dump(mode="json"))


def test_checked_in_synthetic_profile_is_fictional_and_loadable() -> None:
    path = PROJECT_ROOT / "examples" / "hardware" / "synthetic-bf16.json"
    profile = HardwareProfile.model_validate_json(path.read_text())
    assert profile.provenance.kind.value == "synthetic"
    assert "fictional" in profile.provenance.source.lower()


def test_roofline_reports_only_independent_lower_bounds() -> None:
    result = analyze_roofline(
        flops=2_000_000_000_000,
        logical_bytes=4_000_000_000,
        hardware_profile=synthetic_profile(),
        workload_dtype="bfloat16",
    )

    assert result.compute_lower_bound_seconds == pytest.approx(0.002)
    assert result.bandwidth_lower_bound_seconds == pytest.approx(0.004)
    assert result.max_lower_bound_seconds == pytest.approx(0.004)
    assert result.arithmetic_intensity_flops_per_byte == pytest.approx(500.0)
    assert result.bottleneck_class.value == "bandwidth"
    assert result.is_latency_estimate is False
    assert result.interpretation == "theoretical_lower_bounds_not_latency_estimate"


def test_compute_and_balanced_bottleneck_classification() -> None:
    compute = analyze_roofline(
        flops=10_000,
        logical_bytes=1,
        hardware_profile=synthetic_profile(
            peak_flops_per_second=1_000.0,
            memory_bandwidth_bytes_per_second=1_000.0,
        ),
        workload_dtype="bfloat16",
    )
    assert compute.bottleneck_class.value == "compute"
    assert compute.max_lower_bound_seconds == pytest.approx(10.0)

    balanced = analyze_roofline(
        flops=1_000,
        logical_bytes=100,
        hardware_profile=synthetic_profile(
            peak_flops_per_second=1_000.0,
            memory_bandwidth_bytes_per_second=100.0,
        ),
        workload_dtype="bfloat16",
    )
    assert balanced.bottleneck_class.value == "balanced"


def test_missing_work_or_hardware_peak_remains_unknown() -> None:
    result = analyze_roofline(
        flops=None,
        logical_bytes=400,
        hardware_profile=synthetic_profile(peak_flops_per_second=None),
        workload_dtype="bfloat16",
    )

    assert result.compute_lower_bound_seconds is None
    assert result.bandwidth_lower_bound_seconds == pytest.approx(4e-10)
    assert result.max_lower_bound_seconds is None
    assert result.arithmetic_intensity_flops_per_byte is None
    assert result.bottleneck_class.value == "unknown"
    assert "flops_unknown" in result.unknown_reasons
    assert "peak_flops_per_second_unknown" in result.unknown_reasons


def test_dtype_mismatch_does_not_apply_the_wrong_peak() -> None:
    result = analyze_roofline(
        flops=1_000,
        logical_bytes=100,
        hardware_profile=synthetic_profile(),
        workload_dtype="float16",
    )

    assert result.arithmetic_intensity_flops_per_byte == 10.0
    assert result.compute_lower_bound_seconds is None
    assert result.bandwidth_lower_bound_seconds is None
    assert result.max_lower_bound_seconds is None
    assert result.bottleneck_class.value == "unknown"
    assert "hardware_profile_dtype_mismatch" in result.unknown_reasons


def test_zero_bytes_keeps_intensity_unknown_without_breaking_bounds() -> None:
    result = analyze_roofline(
        flops=100,
        logical_bytes=0,
        hardware_profile=synthetic_profile(
            peak_flops_per_second=100.0,
            memory_bandwidth_bytes_per_second=100.0,
        ),
        workload_dtype="bfloat16",
    )
    assert result.arithmetic_intensity_flops_per_byte is None
    assert result.compute_lower_bound_seconds == 1.0
    assert result.bandwidth_lower_bound_seconds == 0.0
    assert result.max_lower_bound_seconds == 1.0
    assert result.bottleneck_class.value == "compute"


def test_unknown_metric_is_not_coerced_to_zero() -> None:
    unknown_flops = Metric(
        id="metric.flops",
        subject_id="semantic.attn",
        scenario_id="scenario.a",
        name="flops",
        value=None,
        unit="FLOP",
        origin="unknown",
        coverage=0.0,
        coverage_status="unknown",
    )
    known_bytes = Metric(
        id="metric.bytes",
        subject_id="semantic.attn",
        scenario_id="scenario.a",
        name="logical_bytes",
        value=1000,
        unit="byte",
        origin="formula",
    )
    result = roofline_from_metrics(
        flops_metric=unknown_flops,
        logical_bytes_metric=known_bytes,
        hardware_profile=synthetic_profile(),
        workload_dtype="bfloat16",
    )

    assert result.compute_lower_bound_seconds is None
    assert result.max_lower_bound_seconds is None
    assert result.bottleneck_class.value == "unknown"


def test_roofline_metric_contract_rejects_mismatched_subject_scenario_and_units() -> None:
    def metric(metric_id: str, subject: str, scenario: str, unit: str) -> Metric:
        return Metric(
            id=metric_id,
            subject_id=subject,
            scenario_id=scenario,
            name=metric_id,
            value=1,
            unit=unit,
            origin="formula",
        )

    with pytest.raises(ValueError, match="same subject"):
        roofline_from_metrics(
            flops_metric=metric("flops", "a", "s", "FLOP"),
            logical_bytes_metric=metric("bytes", "b", "s", "byte"),
            hardware_profile=synthetic_profile(),
            workload_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="same scenario"):
        roofline_from_metrics(
            flops_metric=metric("flops", "a", "left", "FLOP"),
            logical_bytes_metric=metric("bytes", "a", "right", "byte"),
            hardware_profile=synthetic_profile(),
            workload_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="unit"):
        roofline_from_metrics(
            flops_metric=metric("flops", "a", "s", "ms"),
            logical_bytes_metric=None,
            hardware_profile=synthetic_profile(),
            workload_dtype="bfloat16",
        )
