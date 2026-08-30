"""Deterministic Markdown report renderer."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, List

from llm_vis.analysis.service import AnalysisBundle
from llm_vis.report.summaries import workload_diff_summaries

_COST_METRICS = (
    "parameters.resident",
    "parameters.active",
    "weights.storage_bytes",
    "flops",
    "logical_bytes",
    "kv_cache.bytes",
    "state.bytes",
    "arithmetic_intensity",
)


def _table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    header_list = list(headers)
    lines = [
        "| " + " | ".join(header_list) + " |",
        "|" + "|".join("---" for _ in header_list) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _value(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _model_metrics(bundle: AnalysisBundle) -> List[Any]:
    return [
        metric
        for metric in bundle.model_map.metrics
        if metric.subject_id == bundle.model_map.model.id
    ]


def render_markdown(bundle: AnalysisBundle) -> str:
    model_map = bundle.model_map
    manifest = bundle.manifest()
    capability = manifest["capability"]
    definition_counts = Counter(item.kind.value for item in model_map.definitions)
    semantic_counts = Counter(item.kind.value for item in model_map.semantic_nodes)
    strip = " ".join(str(item["label"]) for item in bundle.layer_strip)
    source_description = (
        "config plus project-owned Tiny meta/FakeTensor semantic representatives"
        if bundle.captures
        else "config only"
    )

    lines = [
        f"# LLM-Vis report: {model_map.model.name}",
        "",
        f"> Generated from {source_description}: no weights were loaded, no full model was "
        "constructed, and no full-model forward pass ran.",
        "",
        "> Formula costs and roofline values are theoretical. They are not measured runtime "
        "latency or HBM traffic.",
        "",
        "## Reproducibility",
        "",
        _table(
            ("Field", "Value"),
            (
                ("Analysis ID", bundle.analysis_id),
                ("Revision", model_map.model.revision),
                ("Config SHA-256", bundle.resolved.sha256),
                ("Adapter", bundle.adapter_result.adapter_name),
                ("Schema", model_map.schema_version),
                ("Capability", capability["level"]),
                ("Cost analyzed", capability["cost_analyzed"]),
                ("Roofline lower bounds", capability["roofline_lower_bounds"]),
            ),
        ),
        "",
        "## Capture summary",
        "",
    ]
    if bundle.captures:
        lines.append(
            _table(
                (
                    "Kind",
                    "Status",
                    "Coverage",
                    "Logical ops",
                    "Tensors",
                    "Selected target",
                    "Equivalence",
                    "Full-model perf evidence",
                ),
                (
                    (
                        capture.get("kind", "unknown"),
                        capture.get("status", "unknown"),
                        f"{float(capture.get('coverage', 0.0)):.1%}",
                        capture.get("logical_op_count", 0),
                        capture.get("tensor_count", 0),
                        capture.get("representative_module_path", "unknown"),
                        capture.get("architecture_equivalence", "unknown"),
                        capture.get("valid_for_full_model_performance", False),
                    )
                    for capture in bundle.captures
                ),
            )
        )
    else:
        lines.append("No representative block capture was requested for this artifact.")

    lines.extend(
        [
            "",
            "## Architecture summary",
            "",
            _table(("Definition kind", "Count"), sorted(definition_counts.items())),
            "",
            _table(("Semantic kind", "Count"), sorted(semantic_counts.items())),
            "",
            "## Layer type strip",
            "",
            strip or "No repeated decoder layers were identified.",
            "",
            "## Scenarios",
            "",
        ]
    )
    if model_map.scenarios:
        lines.append(
            _table(
                ("ID", "Phase", "B", "T", "L", "dtype", "weights", "backend", "hardware"),
                (
                    (
                        scenario.id,
                        scenario.phase.value,
                        scenario.batch,
                        scenario.new_tokens,
                        scenario.past_tokens,
                        scenario.activation_dtype.value,
                        scenario.weight_format.value,
                        scenario.backend,
                        scenario.hardware,
                    )
                    for scenario in model_map.scenarios
                ),
            )
        )
    else:
        lines.append("No workload was supplied; workload-dependent costs remain unavailable.")

    lines.extend(["", "## Cost overview", ""])
    cost_rows = []
    model_metrics = _model_metrics(bundle)
    for scenario in model_map.scenarios:
        by_name = {
            metric.name: metric for metric in model_metrics if metric.scenario_id == scenario.id
        }
        for name in _COST_METRICS:
            metric = by_name.get(name)
            if metric is None:
                continue
            cost_rows.append(
                (
                    scenario.phase.value,
                    name,
                    _value(metric.value),
                    metric.unit,
                    metric.origin.value,
                    f"{metric.coverage:.1%}",
                )
            )
    if cost_rows:
        lines.append(
            _table(
                ("Scenario", "Metric", "Value", "Unit", "Origin", "Coverage"),
                cost_rows,
            )
        )
        lines.extend(
            [
                "",
                "Aggregates cover the config-first decoder layer inventory. Exact per-layer "
                "formulas and assumptions remain in `metrics.json`.",
            ]
        )
    else:
        lines.append("No Scenario cost analysis is present.")

    lines.extend(["", "## Theoretical hotspots", ""])
    hotspot_rows = []
    unknown_hotspots = []
    for summary in bundle.hotspot_summaries:
        aggregate = summary["aggregate"]
        if not summary["hotspots"]:
            unknown_hotspots.append(
                (
                    summary["phase"],
                    summary["metric_name"],
                    _value(aggregate["value"]),
                    aggregate["origin"],
                    f"{float(aggregate['coverage']):.1%}",
                )
            )
        for entry in summary["hotspots"]:
            hotspot_rows.append(
                (
                    summary["phase"],
                    summary["metric_name"],
                    entry["rank"],
                    entry["module_path"],
                    _value(entry["value"]),
                    entry["unit"],
                    entry["origin"],
                    f"{float(entry['coverage']):.1%}",
                )
            )
    if hotspot_rows:
        lines.append(
            _table(
                (
                    "Scenario",
                    "Metric",
                    "Rank",
                    "Module",
                    "Value",
                    "Unit",
                    "Origin",
                    "Coverage",
                ),
                hotspot_rows,
            )
        )
    if unknown_hotspots:
        lines.extend(["", "Unknown aggregates are not ranked:", ""])
        lines.append(
            _table(
                ("Scenario", "Metric", "Value", "Origin", "Coverage"),
                unknown_hotspots,
            )
        )
    if not hotspot_rows and not unknown_hotspots:
        lines.append("No hotspot analysis is present.")

    lines.extend(["", "## Roofline lower bounds", ""])
    if bundle.roofline_summaries:
        lines.append(
            _table(
                (
                    "Scenario",
                    "AI",
                    "Compute lower bound (s)",
                    "Bandwidth lower bound (s)",
                    "Max lower bound (s)",
                    "Class",
                ),
                (
                    (
                        summary["phase"],
                        _value(summary["arithmetic_intensity_flops_per_byte"]),
                        _value(summary["compute_lower_bound_seconds"]),
                        _value(summary["bandwidth_lower_bound_seconds"]),
                        _value(summary["max_lower_bound_seconds"]),
                        summary["bottleneck_class"],
                    )
                    for summary in bundle.roofline_summaries
                ),
            )
        )
        lines.extend(["", "**Theoretical lower bounds — not a latency estimate.**"])
    else:
        lines.append(
            "No explicit HardwareProfile was supplied; roofline bounds remain unavailable."
        )

    lines.extend(["", "## Workload diff", ""])
    diff_rows = []
    for result in workload_diff_summaries(model_map):
        for entry in result["entries"]:
            if entry["subject_id"] != model_map.model.id:
                continue
            left = entry.get("left") or {}
            right = entry.get("right") or {}
            diff_rows.append(
                (
                    result["left_scenario_id"],
                    result["right_scenario_id"],
                    entry["name"],
                    entry["status"],
                    _value(left.get("value")),
                    _value(right.get("value")),
                    _value(entry.get("value_delta")),
                )
            )
    if diff_rows:
        lines.append(
            _table(
                ("Left", "Right", "Metric", "Status", "Left value", "Right value", "Delta"),
                diff_rows,
            )
        )
    elif len(model_map.scenarios) < 2:
        lines.append("At least two Scenarios are required for same-artifact workload diff.")
    else:
        lines.append("No changed model-level metrics were found.")

    lines.extend(["", "## Runtime (external trace only)", ""])
    runtime_metrics = [metric for metric in model_metrics if metric.name.startswith("runtime.")]
    lines.append(
        _table(
            ("Scenario", "Metric", "Value", "Unit", "Origin", "Coverage"),
            (
                (
                    metric.scenario_id or "none",
                    metric.name,
                    _value(metric.value),
                    metric.unit,
                    metric.origin.value,
                    f"{metric.coverage:.1%}",
                )
                for metric in runtime_metrics
            ),
        )
    )
    lines.extend(
        [
            "",
            "No external runtime trace was imported. Unknown runtime values are not replaced "
            "with formula estimates or zero.",
            "",
            "## Diagnostics",
            "",
            _table(
                ("Severity", "Code", "Message"),
                ((item.severity.value, item.code, item.message) for item in model_map.diagnostics),
            ),
            "",
            "## Artifact inventory",
            "",
            "The complete IR, per-layer Metric formulas, captures, hotspots, roofline bounds, "
            "workload diffs and Model Explorer-compatible graphs are stored in the sibling "
            "JSON files listed by `manifest.json`.",
        ]
    )
    return "\n".join(lines) + "\n"
