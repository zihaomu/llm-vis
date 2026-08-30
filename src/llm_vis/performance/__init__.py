"""Public API for explicit-hardware theoretical roofline analysis."""

from .hardware import (
    HardwareProfile,
    HardwareProvenance,
    HardwareProvenanceKind,
    hardware_profile_schema,
)
from .roofline import (
    BottleneckClass,
    RooflineResult,
    analyze_roofline,
    roofline_from_metrics,
    roofline_lower_bounds,
)

__all__ = [
    "BottleneckClass",
    "HardwareProfile",
    "HardwareProvenance",
    "HardwareProvenanceKind",
    "RooflineResult",
    "analyze_roofline",
    "hardware_profile_schema",
    "roofline_from_metrics",
    "roofline_lower_bounds",
]
