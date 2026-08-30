"""Deterministic presentation summaries derived from Model Map facts."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from llm_vis.diff import diff_scenario_metrics
from llm_vis.ir import ModelMap


def workload_diff_summaries(model_map: ModelMap) -> Tuple[Dict[str, Any], ...]:
    """Compare the first Scenario with each later Scenario in one artifact.

    M3 intentionally supports only same-model, same-artifact workload comparison.
    The first Scenario is a deterministic baseline; no comparison is emitted when
    fewer than two Scenarios are present.
    """

    if len(model_map.scenarios) < 2:
        return ()
    baseline = model_map.scenarios[0]
    return tuple(
        diff_scenario_metrics(model_map, baseline.id, scenario.id).model_dump(mode="json")
        for scenario in model_map.scenarios[1:]
    )


__all__ = ["workload_diff_summaries"]
