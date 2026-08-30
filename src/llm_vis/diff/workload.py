# ruff: noqa: UP045
"""Metric diffs between two scenarios inside one Model Map artifact."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from llm_vis.ir import (
    CoverageStatus,
    Metric,
    MetricOrigin,
    ModelMap,
    SymbolicExpression,
    deterministic_id,
)

Number = Union[int, float]
MetricKey = Tuple[str, str]


class ScenarioMetricDiffError(ValueError):
    """Base error for unsupported or ambiguous workload comparisons."""


class ScenarioNotFoundError(ScenarioMetricDiffError):
    """A requested scenario is not present in the Model Map."""


class CrossModelDiffError(ScenarioMetricDiffError):
    """Cross-model comparison is an M5 feature and is rejected by this module."""


class CrossArtifactDiffError(ScenarioMetricDiffError):
    """The M3 API only compares scenarios in one Model Map object."""


class DuplicateScenarioMetricError(ScenarioMetricDiffError):
    """A scenario has more than one metric for the stable subject/name key."""


class MetricDiffStatus(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNKNOWN = "unknown"


class DiffBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class MetricSnapshot(DiffBaseModel):
    """Scenario-specific Metric fields retained without treating absence as zero."""

    metric_id: str
    scenario_id: str
    value: Optional[Number]
    unit: str
    origin: MetricOrigin
    formula: Optional[SymbolicExpression] = None
    assumptions: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    coverage: float
    coverage_status: CoverageStatus

    @classmethod
    def from_metric(cls, metric: Metric) -> MetricSnapshot:
        if metric.scenario_id is None:
            raise ValueError("scenario-bound snapshot requires metric.scenario_id")
        return cls(
            metric_id=metric.id,
            scenario_id=metric.scenario_id,
            value=metric.value,
            unit=metric.unit,
            origin=metric.origin,
            formula=metric.formula,
            assumptions=list(metric.assumptions),
            confidence=metric.confidence,
            coverage=metric.coverage,
            coverage_status=metric.coverage_status,
        )


class MetricDiffEntry(DiffBaseModel):
    subject_id: str
    name: str
    status: MetricDiffStatus
    left: Optional[MetricSnapshot] = None
    right: Optional[MetricSnapshot] = None
    value_delta: Optional[float] = None
    relative_delta: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)


class WorkloadMetricDiff(DiffBaseModel):
    id: str
    model_id: str
    model_revision: str
    left_scenario_id: str
    right_scenario_id: str
    entries: List[MetricDiffEntry] = Field(default_factory=list)
    unchanged_count: int = Field(default=0, ge=0)

    @property
    def status_counts(self) -> Dict[MetricDiffStatus, int]:
        counts = {status: 0 for status in MetricDiffStatus}
        for entry in self.entries:
            counts[entry.status] += 1
        return counts


def _scenario_ids(model_map: ModelMap) -> set[str]:
    return {scenario.id for scenario in model_map.scenarios}


def _scenario_metrics(model_map: ModelMap, scenario_id: str) -> Dict[MetricKey, Metric]:
    indexed: Dict[MetricKey, Metric] = {}
    for metric in model_map.metrics:
        if metric.scenario_id != scenario_id:
            continue
        key = (metric.subject_id, metric.name)
        if key in indexed:
            raise DuplicateScenarioMetricError(
                "scenario contains duplicate metric for "
                f"subject={metric.subject_id!r}, name={metric.name!r}"
            )
        indexed[key] = metric
    return indexed


def _is_unknown(snapshot: MetricSnapshot) -> bool:
    return (
        snapshot.origin == MetricOrigin.UNKNOWN
        or snapshot.value is None
        or snapshot.coverage_status == CoverageStatus.UNKNOWN
    )


def _change_reasons(left: MetricSnapshot, right: MetricSnapshot) -> List[str]:
    reasons: List[str] = []
    if left.value != right.value:
        reasons.append("value_changed")
    if left.unit != right.unit:
        reasons.append("unit_changed")
    if left.origin != right.origin:
        reasons.append("origin_changed")
    if left.formula != right.formula:
        reasons.append("formula_changed")
    if left.assumptions != right.assumptions:
        reasons.append("assumptions_changed")
    if left.confidence != right.confidence:
        reasons.append("confidence_changed")
    if left.coverage != right.coverage:
        reasons.append("coverage_changed")
    if left.coverage_status != right.coverage_status:
        reasons.append("coverage_status_changed")
    return reasons


def _matched_entry(
    key: MetricKey,
    left_metric: Metric,
    right_metric: Metric,
) -> Optional[MetricDiffEntry]:
    left = MetricSnapshot.from_metric(left_metric)
    right = MetricSnapshot.from_metric(right_metric)
    reasons = _change_reasons(left, right)

    unknown_reasons: List[str] = []
    if _is_unknown(left):
        unknown_reasons.append("left_metric_unknown")
    if _is_unknown(right):
        unknown_reasons.append("right_metric_unknown")
    if left.unit != right.unit:
        unknown_reasons.append("units_not_comparable")
    if unknown_reasons:
        return MetricDiffEntry(
            subject_id=key[0],
            name=key[1],
            status=MetricDiffStatus.UNKNOWN,
            left=left,
            right=right,
            reasons=unknown_reasons,
        )

    if not reasons:
        return None

    assert left.value is not None
    assert right.value is not None
    value_delta = float(right.value) - float(left.value)
    relative_delta: Optional[float] = None
    if float(left.value) == 0.0:
        reasons.append("relative_delta_undefined_for_zero_left")
    else:
        relative_delta = value_delta / abs(float(left.value))

    return MetricDiffEntry(
        subject_id=key[0],
        name=key[1],
        status=MetricDiffStatus.CHANGED,
        left=left,
        right=right,
        value_delta=value_delta,
        relative_delta=relative_delta,
        reasons=reasons,
    )


def diff_scenario_metrics(
    model_map: ModelMap,
    left_scenario_id: str,
    right_scenario_id: str,
) -> WorkloadMetricDiff:
    """Diff two scenario-bound Metric sets within one ``ModelMap``.

    Matching uses only the stable ``(subject_id, name)`` key.  Scenario-free
    metrics are intentionally ignored.  Equal metrics are counted but omitted
    from the change list; every emitted entry is added, removed, changed, or
    unknown.
    """

    known_scenarios = _scenario_ids(model_map)
    if not known_scenarios:
        raise ScenarioNotFoundError("model map contains no scenarios")
    if not left_scenario_id or left_scenario_id not in known_scenarios:
        raise ScenarioNotFoundError(f"left scenario {left_scenario_id!r} does not exist")
    if not right_scenario_id or right_scenario_id not in known_scenarios:
        raise ScenarioNotFoundError(f"right scenario {right_scenario_id!r} does not exist")
    if left_scenario_id == right_scenario_id:
        raise ScenarioMetricDiffError("workload diff requires two distinct scenarios")

    left_metrics = _scenario_metrics(model_map, left_scenario_id)
    right_metrics = _scenario_metrics(model_map, right_scenario_id)
    entries: List[MetricDiffEntry] = []
    unchanged_count = 0

    for key in sorted(set(left_metrics) | set(right_metrics)):
        left_metric = left_metrics.get(key)
        right_metric = right_metrics.get(key)
        if left_metric is None:
            assert right_metric is not None
            entries.append(
                MetricDiffEntry(
                    subject_id=key[0],
                    name=key[1],
                    status=MetricDiffStatus.ADDED,
                    right=MetricSnapshot.from_metric(right_metric),
                    reasons=["metric_missing_on_left"],
                )
            )
            continue
        if right_metric is None:
            entries.append(
                MetricDiffEntry(
                    subject_id=key[0],
                    name=key[1],
                    status=MetricDiffStatus.REMOVED,
                    left=MetricSnapshot.from_metric(left_metric),
                    reasons=["metric_missing_on_right"],
                )
            )
            continue

        entry = _matched_entry(key, left_metric, right_metric)
        if entry is None:
            unchanged_count += 1
        else:
            entries.append(entry)

    result_id = deterministic_id(
        "wdiff",
        {
            "model_id": model_map.model.id,
            "model_revision": model_map.model.revision,
            "left_scenario_id": left_scenario_id,
            "right_scenario_id": right_scenario_id,
        },
    )
    return WorkloadMetricDiff(
        id=result_id,
        model_id=model_map.model.id,
        model_revision=model_map.model.revision,
        left_scenario_id=left_scenario_id,
        right_scenario_id=right_scenario_id,
        entries=entries,
        unchanged_count=unchanged_count,
    )


def diff_model_maps(
    left_model_map: ModelMap,
    right_model_map: ModelMap,
    left_scenario_id: str,
    right_scenario_id: str,
) -> WorkloadMetricDiff:
    """Guarded entry point that explicitly rejects cross-model/artifact diff."""

    left_identity = (
        left_model_map.model.id,
        left_model_map.model.family,
        left_model_map.model.revision,
    )
    right_identity = (
        right_model_map.model.id,
        right_model_map.model.family,
        right_model_map.model.revision,
    )
    if left_identity != right_identity:
        raise CrossModelDiffError("cross-model or cross-revision diff is deferred to M5")
    if left_model_map is not right_model_map:
        raise CrossArtifactDiffError("M3 only compares scenarios in the same ModelMap")
    return diff_scenario_metrics(left_model_map, left_scenario_id, right_scenario_id)


compare_workloads = diff_scenario_metrics
