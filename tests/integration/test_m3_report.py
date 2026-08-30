from __future__ import annotations

import json
from pathlib import Path

from llm_vis.analysis import inspect_model
from llm_vis.ir import Scenario
from llm_vis.performance import HardwareProfile, HardwareProvenance
from llm_vis.report import write_analysis

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _profile() -> HardwareProfile:
    return HardwareProfile(
        id="synthetic-bf16-report",
        name="Synthetic report profile",
        peak_flops_per_second=1_000_000_000_000_000.0,
        memory_bandwidth_bytes_per_second=1_000_000_000_000.0,
        dtype="bfloat16",
        provenance=HardwareProvenance(
            kind="synthetic",
            source="integration-test-only fictional values",
        ),
    )


def _scenarios() -> tuple[Scenario, Scenario]:
    return (
        Scenario(
            phase="prefill",
            batch=1,
            new_tokens=16,
            past_tokens=0,
            activation_dtype="bfloat16",
            weight_format="bfloat16",
            kv_dtype="bfloat16",
            backend="formula-only",
            hardware="unknown",
        ),
        Scenario(
            phase="decode",
            batch=1,
            new_tokens=1,
            past_tokens=1024,
            activation_dtype="bfloat16",
            weight_format="int4",
            kv_dtype="bfloat16",
            backend="formula-only",
            hardware="unknown",
        ),
    )


def test_m3_artifact_contains_hotspots_roofline_diff_and_offline_controls(
    tmp_path: Path,
) -> None:
    scenarios = _scenarios()
    bundle = inspect_model(
        str(FIXTURE_DIR / "tiny_dense.json"),
        scenarios=scenarios,
        hardware_profile=_profile(),
    )
    written = write_analysis(bundle, tmp_path / "artifact")

    hotspots = json.loads(written["hotspots.json"].read_text(encoding="utf-8"))
    roofline = json.loads(written["roofline.json"].read_text(encoding="utf-8"))
    diffs = json.loads(written["workload-diffs.json"].read_text(encoding="utf-8"))
    markdown = written["reports/report.md"].read_text(encoding="utf-8")
    html = written["reports/report.html"].read_text(encoding="utf-8")

    assert len(hotspots) == 4
    assert all(item["hotspots"] for item in hotspots)
    assert len(roofline) == 2
    assert all(item["is_latency_estimate"] is False for item in roofline)
    assert len(diffs) == 1
    assert diffs[0]["left_scenario_id"] == scenarios[0].id
    assert diffs[0]["right_scenario_id"] == scenarios[1].id
    assert "## Theoretical hotspots" in markdown
    assert "## Workload diff" in markdown
    assert "not a latency estimate" in markdown
    assert 'id="scenario-select"' in html
    assert 'id="hotspot-table"' in html
    assert 'id="diff-table"' in html
    assert 'id="runtime-table"' in html
    assert 'id="coverage-badge"' in html
    assert 'id="model-dag"' in html
    assert 'id="heatmap-mode"' in html
    assert '<option value="pressure">Pressure</option>' in html
    assert '<option value="compute">Compute</option>' in html
    assert '<option value="memory">Memory</option>' in html
    assert "Nodes marked “N ops ›” expand in this canvas" in html
    assert "data-initial-view-id=" in html
    assert 'data-dag-action="fit"' in html
    assert 'class="llm-dag-minimap"' in html
    assert "window.llmVisInspect" in html
    assert "window.LLMVisDAG?.search('model-dag',query)" in html
    assert "graphMetricsFor" in html
    assert "scopedGraphMetrics" in html
    assert "metricNameForGraphNode" in html
    assert "theoreticalNodeHeat" in html
    assert "heatmapForView" in html
    assert "window.LLMVisDAG?.setHeatmap('model-dag',config)" in html
    assert "renderHeatmap(scenario)" in html
    assert "Unknown is not zero" in html
    assert "not measured latency" in html
    assert '"hardwareProfile": {' in html
    assert '"id": "synthetic-bf16-report"' in html
    assert '"kind": "synthetic"' in html
    assert "theoretical_bottleneck" in html
    assert "is_latency_estimate:false" in html
    assert "metric_unknown_reason" in html
    assert "lastGraphDetail" in html
    assert "bounded capture remains separate evidence" in html
    assert "capture_scope:'not-applicable'" in html
    assert "locateGraphSubject" in html
    assert "focusNode('model-dag',target.node.id)" in html
    assert "graphViewForLayer" in html
    assert 'id="graph-heading">Model graph' in html
    assert 'class="dag-panel" aria-labelledby="graph-heading"' in html
    assert 'id="navigator-drawer" aria-hidden="true"' in html
    assert 'id="inspector-drawer" aria-hidden="true"' in html
    assert 'data-drawer-target="navigator-drawer"' in html
    assert 'aria-controls="navigator-drawer" aria-expanded="false"' in html
    assert 'data-drawer-target="inspector-drawer"' in html
    assert 'aria-controls="inspector-drawer" aria-expanded="false"' in html
    assert 'id="supporting-analysis"' in html
    assert 'id="layer-count"' in html
    assert "setInspector(graphSelectionContext(detail),{reveal:false})" in html
    assert "Structure index" in html
    for inspector_tab in (
        "explain",
        "tensors",
        "cost",
        "runtime",
        "provenance",
        "coverage",
    ):
        assert f'data-inspector-tab="{inspector_tab}"' in html
    assert "symbols:map.symbols" in html
    assert "current visible cost frontier only" in html
    assert "no parent/child double count" in html
    assert "filter(node=>frontierIds.has(node.id))" in html
    assert "!frontierIds.size||frontierIds.has(node.id)" not in html
    assert "decompositionCostSummary" in html
    assert "decomposition_cost_reconciliation" in html
    assert "filter(item=>childIds.has(item.id))" in html
    assert "!childIds.size||childIds.has(item.id)" not in html
    assert "signed_remainder:signedRemainder" in html
    assert "unexpected_known_child_ids" in html
    assert "inconsistent:inconsistencyReasons.length>0" in html
    assert "Math.max(0,Number(parent.value)-knownSubtotal)" not in html
    assert "Math.min(1,knownSubtotal/Number(parent.value))" not in html
    assert "graphDetailByView" in html
    assert "graphDetailByView.get(view.id)" in html
    assert "selection_coverage" in html
    assert "scope.included_instance_count+scope.excluded_instance_count" in html
    assert "cost ${known}/${metrics.length} known" in html
    assert "Mean metric coverage" in html
    html_without_svg_namespace = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in html_without_svg_namespace
    assert "https://" not in html_without_svg_namespace


def test_glm_unknown_cost_is_visible_and_never_serialized_as_zero(tmp_path: Path) -> None:
    scenario = _scenarios()[1]
    bundle = inspect_model(
        str(FIXTURE_DIR / "glm_5_3_bf16.json"),
        scenarios=[scenario],
        hardware_profile=_profile(),
    )
    written = write_analysis(bundle, tmp_path / "glm")

    hotspots = json.loads(written["hotspots.json"].read_text(encoding="utf-8"))
    roofline = json.loads(written["roofline.json"].read_text(encoding="utf-8"))[0]
    execution_summaries = {
        item["metric_name"]: item for item in hotspots if item["scenario_id"] == scenario.id
    }

    assert execution_summaries["flops"]["aggregate"]["value"] is None
    assert execution_summaries["flops"]["hotspots"] == []
    assert execution_summaries["logical_bytes"]["aggregate"]["value"] is None
    assert roofline["compute_lower_bound_seconds"] is None
    assert roofline["bandwidth_lower_bound_seconds"] is None
    assert roofline["max_lower_bound_seconds"] is None
    assert roofline["bottleneck_class"] == "unknown"
    markdown = written["reports/report.md"].read_text(encoding="utf-8")
    html = written["reports/report.html"].read_text(encoding="utf-8")
    assert "Unknown aggregates are not ranked" in markdown
    assert "| decode | flops | Unknown | unknown | 0.0% |" in markdown
    assert "The config inventory does not provide a trustworthy shape" in html
    assert "GLM DSA and executed MoE costs are intentionally Unknown" in html
    assert "Unknown is not zero" in html
    assert "metricNameForGraphNode(node,view,'parameters.active')" not in html
