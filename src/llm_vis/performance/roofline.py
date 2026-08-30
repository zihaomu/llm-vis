# ruff: noqa: UP045
"""Independent roofline lower-bound analysis.

The results in this module are theoretical lower bounds.  They are never a
latency estimate and do not claim to model launch overhead, cache behavior,
fusion, occupancy, synchronization, scheduling, or physical HBM traffic.
"""

from __future__ import annotations

import math
from enum import Enum
from numbers import Real
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from llm_vis.ir import DType, Metric, MetricOrigin

from .hardware import HardwareProfile

Number = Union[int, float]


class BottleneckClass(str, Enum):
    COMPUTE = "compute"
    BANDWIDTH = "bandwidth"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


class RooflineResult(BaseModel):
    """Theoretical roofline bounds for one workload subject."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    hardware_profile_id: str
    hardware_dtype: DType
    workload_dtype: DType
    arithmetic_intensity_flops_per_byte: Optional[float] = Field(default=None, ge=0.0)
    compute_lower_bound_seconds: Optional[float] = Field(default=None, ge=0.0)
    bandwidth_lower_bound_seconds: Optional[float] = Field(default=None, ge=0.0)
    max_lower_bound_seconds: Optional[float] = Field(default=None, ge=0.0)
    bottleneck_class: BottleneckClass
    is_latency_estimate: Literal[False] = False
    interpretation: Literal["theoretical_lower_bounds_not_latency_estimate"] = (
        "theoretical_lower_bounds_not_latency_estimate"
    )
    unknown_reasons: List[str] = Field(default_factory=list)


def _optional_non_negative(name: str, value: Optional[Number]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _append_once(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def analyze_roofline(
    *,
    flops: Optional[Number],
    logical_bytes: Optional[Number],
    hardware_profile: HardwareProfile,
    workload_dtype: Union[DType, str],
) -> RooflineResult:
    """Calculate independent compute/bandwidth lower bounds.

    ``flops`` and ``logical_bytes`` are useful work and logical traffic, not
    measured device counters.  A missing input or peak remains unknown.  The
    max bound and bottleneck are emitted only when both independent bounds are
    known.
    """

    flops_value = _optional_non_negative("flops", flops)
    bytes_value = _optional_non_negative("logical_bytes", logical_bytes)
    dtype = DType(workload_dtype)
    reasons: List[str] = []

    if flops_value is None:
        _append_once(reasons, "flops_unknown")
    if bytes_value is None:
        _append_once(reasons, "logical_bytes_unknown")

    intensity: Optional[float] = None
    if flops_value is not None and bytes_value is not None:
        if bytes_value == 0:
            _append_once(reasons, "arithmetic_intensity_undefined_for_zero_bytes")
        else:
            intensity = flops_value / bytes_value

    dtype_compatible = True
    if dtype == DType.UNKNOWN:
        dtype_compatible = False
        _append_once(reasons, "workload_dtype_unknown")
    elif dtype != hardware_profile.dtype:
        dtype_compatible = False
        _append_once(reasons, "hardware_profile_dtype_mismatch")

    compute_bound: Optional[float] = None
    if hardware_profile.peak_flops_per_second is None:
        _append_once(reasons, "peak_flops_per_second_unknown")
    elif flops_value is not None and dtype_compatible:
        compute_bound = flops_value / float(hardware_profile.peak_flops_per_second)

    bandwidth_bound: Optional[float] = None
    if hardware_profile.memory_bandwidth_bytes_per_second is None:
        _append_once(reasons, "memory_bandwidth_bytes_per_second_unknown")
    elif bytes_value is not None and dtype_compatible:
        bandwidth_bound = bytes_value / float(hardware_profile.memory_bandwidth_bytes_per_second)

    max_bound: Optional[float] = None
    bottleneck = BottleneckClass.UNKNOWN
    if compute_bound is not None and bandwidth_bound is not None:
        max_bound = max(compute_bound, bandwidth_bound)
        if math.isclose(compute_bound, bandwidth_bound, rel_tol=1e-12, abs_tol=0.0):
            bottleneck = BottleneckClass.BALANCED
        elif compute_bound > bandwidth_bound:
            bottleneck = BottleneckClass.COMPUTE
        else:
            bottleneck = BottleneckClass.BANDWIDTH

    return RooflineResult(
        hardware_profile_id=hardware_profile.id,
        hardware_dtype=hardware_profile.dtype,
        workload_dtype=dtype,
        arithmetic_intensity_flops_per_byte=intensity,
        compute_lower_bound_seconds=compute_bound,
        bandwidth_lower_bound_seconds=bandwidth_bound,
        max_lower_bound_seconds=max_bound,
        bottleneck_class=bottleneck,
        unknown_reasons=reasons,
    )


def _metric_value(metric: Optional[Metric]) -> Optional[Number]:
    if metric is None or metric.origin == MetricOrigin.UNKNOWN:
        return None
    return metric.value


def roofline_from_metrics(
    *,
    flops_metric: Optional[Metric],
    logical_bytes_metric: Optional[Metric],
    hardware_profile: HardwareProfile,
    workload_dtype: Union[DType, str],
) -> RooflineResult:
    """Calculate bounds from two IR Metrics without converting Unknown to zero."""

    if flops_metric is not None and flops_metric.unit.lower() not in {"flop", "flops"}:
        raise ValueError("flops_metric unit must be FLOP or FLOPs")
    if logical_bytes_metric is not None and logical_bytes_metric.unit.lower() not in {
        "b",
        "byte",
        "bytes",
    }:
        raise ValueError("logical_bytes_metric unit must be byte, bytes, or B")
    if flops_metric is not None and logical_bytes_metric is not None:
        if flops_metric.subject_id != logical_bytes_metric.subject_id:
            raise ValueError("roofline metrics must describe the same subject")
        if flops_metric.scenario_id != logical_bytes_metric.scenario_id:
            raise ValueError("roofline metrics must describe the same scenario")

    return analyze_roofline(
        flops=_metric_value(flops_metric),
        logical_bytes=_metric_value(logical_bytes_metric),
        hardware_profile=hardware_profile,
        workload_dtype=workload_dtype,
    )


roofline_lower_bounds = analyze_roofline
