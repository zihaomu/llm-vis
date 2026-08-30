from __future__ import annotations

import pytest

from llm_vis.diff import (
    CrossArtifactDiffError,
    CrossModelDiffError,
    DuplicateScenarioMetricError,
    ScenarioMetricDiffError,
    ScenarioNotFoundError,
    diff_model_maps,
    diff_scenario_metrics,
)
from llm_vis.ir import Definition, Metric, Model, ModelMap, Scenario


def scenarios() -> tuple[Scenario, Scenario]:
    prefill = Scenario(
        phase="prefill",
        batch=1,
        new_tokens=8,
        past_tokens=0,
        activation_dtype="bfloat16",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
        backend="static",
        hardware="unknown",
    )
    decode = Scenario(
        phase="decode",
        batch=1,
        new_tokens=1,
        past_tokens=8,
        activation_dtype="bfloat16",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
        backend="static",
        hardware="unknown",
    )
    return prefill, decode


def make_metric(
    metric_id: str,
    scenario_id: str | None,
    name: str,
    value: int | float | None,
    *,
    origin: str = "formula",
    unit: str = "FLOP",
    coverage: float = 1.0,
    coverage_status: str = "complete",
) -> Metric:
    return Metric(
        id=metric_id,
        subject_id="semantic.block",
        scenario_id=scenario_id,
        name=name,
        value=value,
        unit=unit,
        origin=origin,
        coverage=coverage,
        coverage_status=coverage_status,
    )


def workload_map(*, model_id: str = "model.tiny") -> tuple[ModelMap, Scenario, Scenario]:
    left, right = scenarios()
    metrics = [
        make_metric("left.flops", left.id, "flops", 100),
        make_metric(
            "left.bytes",
            left.id,
            "logical_bytes",
            None,
            origin="unknown",
            unit="byte",
            coverage=0.0,
            coverage_status="unknown",
        ),
        make_metric("left.removed", left.id, "left_only", 7),
        make_metric("left.same", left.id, "same", 3),
        make_metric("right.flops", right.id, "flops", 150),
        make_metric("right.bytes", right.id, "logical_bytes", 200, unit="byte"),
        make_metric("right.added", right.id, "right_only", 9),
        make_metric("right.same", right.id, "same", 3),
        make_metric("global.params", None, "parameters", 1024, unit="element"),
    ]
    return (
        ModelMap(
            model=Model(
                id=model_id,
                name="Tiny",
                family="tiny",
                revision="fixture-r1",
                framework="pytorch",
            ),
            definitions=[Definition(id="semantic.block", kind="block", label="Block")],
            scenarios=[left, right],
            metrics=metrics,
        ),
        left,
        right,
    )


def test_diff_outputs_added_removed_changed_and_unknown_without_zero_fill() -> None:
    model_map, left, right = workload_map()
    result = diff_scenario_metrics(model_map, left.id, right.id)
    entries = {entry.name: entry for entry in result.entries}

    assert entries["flops"].status.value == "changed"
    assert entries["flops"].value_delta == 50.0
    assert entries["flops"].relative_delta == pytest.approx(0.5)

    assert entries["logical_bytes"].status.value == "unknown"
    assert entries["logical_bytes"].left is not None
    assert entries["logical_bytes"].left.value is None
    assert entries["logical_bytes"].value_delta is None
    assert entries["logical_bytes"].relative_delta is None

    assert entries["left_only"].status.value == "removed"
    assert entries["left_only"].value_delta is None
    assert entries["right_only"].status.value == "added"
    assert entries["right_only"].value_delta is None
    assert "parameters" not in entries
    assert "same" not in entries
    assert result.unchanged_count == 1
    assert result.status_counts[next(s for s in result.status_counts if s.value == "changed")] == 1


def test_diff_is_deterministic_and_matches_by_subject_and_name_not_metric_id() -> None:
    model_map, left, right = workload_map()
    first = diff_scenario_metrics(model_map, left.id, right.id)
    second = diff_scenario_metrics(model_map, left.id, right.id)

    assert first == second
    assert first.id.startswith("wdiff_")
    flops = next(entry for entry in first.entries if entry.name == "flops")
    assert flops.left is not None and flops.left.metric_id == "left.flops"
    assert flops.right is not None and flops.right.metric_id == "right.flops"


def test_equal_value_with_changed_metric_provenance_is_changed() -> None:
    left, right = scenarios()
    model_map = ModelMap(
        model=Model(
            id="model.tiny",
            name="Tiny",
            family="tiny",
            revision="r1",
            framework="pytorch",
        ),
        definitions=[Definition(id="semantic.block", kind="block", label="Block")],
        scenarios=[left, right],
        metrics=[
            make_metric("left", left.id, "flops", 100, origin="formula"),
            make_metric("right", right.id, "flops", 100, origin="estimated"),
        ],
    )
    entry = diff_scenario_metrics(model_map, left.id, right.id).entries[0]
    assert entry.status.value == "changed"
    assert entry.value_delta == 0.0
    assert "origin_changed" in entry.reasons


def test_units_that_cannot_be_compared_produce_unknown() -> None:
    left, right = scenarios()
    model_map = ModelMap(
        model=Model(
            id="model.tiny",
            name="Tiny",
            family="tiny",
            revision="r1",
            framework="pytorch",
        ),
        definitions=[Definition(id="semantic.block", kind="block", label="Block")],
        scenarios=[left, right],
        metrics=[
            make_metric("left", left.id, "traffic", 100, unit="byte"),
            make_metric("right", right.id, "traffic", 1, unit="GiB"),
        ],
    )
    entry = diff_scenario_metrics(model_map, left.id, right.id).entries[0]
    assert entry.status.value == "unknown"
    assert entry.value_delta is None
    assert "units_not_comparable" in entry.reasons


def test_missing_or_identical_scenarios_are_rejected() -> None:
    model_map, left, right = workload_map()
    with pytest.raises(ScenarioNotFoundError, match="does not exist"):
        diff_scenario_metrics(model_map, "missing", right.id)
    with pytest.raises(ScenarioMetricDiffError, match="distinct"):
        diff_scenario_metrics(model_map, left.id, left.id)

    empty = ModelMap(
        model=Model(
            id="model.empty",
            name="Empty",
            family="empty",
            revision="r1",
            framework="generic",
        )
    )
    with pytest.raises(ScenarioNotFoundError, match="no scenarios"):
        diff_scenario_metrics(empty, "left", "right")


def test_ambiguous_duplicate_subject_name_metric_is_rejected() -> None:
    left, right = scenarios()
    model_map = ModelMap(
        model=Model(
            id="model.tiny",
            name="Tiny",
            family="tiny",
            revision="r1",
            framework="pytorch",
        ),
        definitions=[Definition(id="semantic.block", kind="block", label="Block")],
        scenarios=[left, right],
        metrics=[
            make_metric("left.one", left.id, "flops", 1),
            make_metric("left.two", left.id, "flops", 2),
            make_metric("right", right.id, "flops", 3),
        ],
    )
    with pytest.raises(DuplicateScenarioMetricError, match="duplicate metric"):
        diff_scenario_metrics(model_map, left.id, right.id)


def test_cross_model_and_cross_artifact_comparisons_are_rejected() -> None:
    left_map, left, right = workload_map()
    other_model, other_left, other_right = workload_map(model_id="model.other")
    with pytest.raises(CrossModelDiffError, match="cross-model"):
        diff_model_maps(left_map, other_model, left.id, other_right.id)

    same_identity_copy = left_map.model_copy(deep=True)
    with pytest.raises(CrossArtifactDiffError, match="same ModelMap"):
        diff_model_maps(left_map, same_identity_copy, left.id, right.id)

    result = diff_model_maps(left_map, left_map, left.id, right.id)
    assert result.left_scenario_id == left.id
    assert other_left.id != ""
