"""Config-first analysis orchestration for M0-M3.6.

The inspection path is deliberately incapable of loading weights or invoking a
model forward pass. Optional representative-block capture is a separate M2 API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from llm_vis import __version__
from llm_vis.adapters import AdapterResult, build_from_config
from llm_vis.cost import CostAnalysis, ScenarioCost, analyze_costs
from llm_vis.ir import (
    CoverageStatus,
    Diagnostic,
    DiagnosticSeverity,
    Metric,
    MetricOrigin,
    ModelMap,
    Scenario,
    SourceArtifact,
    SourceArtifactKind,
    deterministic_id,
)
from llm_vis.performance import HardwareProfile, roofline_from_metrics
from llm_vis.resolver import ResolvedConfig, resolve_config

if TYPE_CHECKING:
    from llm_vis.graph_view import GraphViewDocument

_COST_MODEL_VERSION = "config-first-m3-v0.2"
_GRAPH_VIEW_VERSION = "config-first-graph-view-v0.1"
_HOTSPOT_METRICS = ("flops", "logical_bytes")
_HOTSPOT_LIMIT = 10


def _remote_code_declarations(config: Mapping[str, Any], prefix: str = "config") -> Tuple[str, ...]:
    declarations = []
    auto_map = config.get("auto_map")
    if isinstance(auto_map, Mapping) and auto_map:
        declarations.append(f"{prefix}.auto_map")
    for key in ("text_config", "vision_config", "audio_config"):
        nested = config.get(key)
        if isinstance(nested, Mapping):
            declarations.extend(_remote_code_declarations(nested, f"{prefix}.{key}"))
    return tuple(declarations)


def _diagnostic(
    code: str,
    message: str,
    *,
    subject_id: Optional[str] = None,
    severity: DiagnosticSeverity = DiagnosticSeverity.INFO,
    evidence: Iterable[str] = (),
) -> Diagnostic:
    evidence_list = list(evidence)
    return Diagnostic(
        id=deterministic_id(
            "diag",
            {
                "code": code,
                "message": message,
                "subject_id": subject_id,
                "evidence": evidence_list,
            },
        ),
        severity=severity,
        code=code,
        subject_id=subject_id,
        message=message,
        evidence=evidence_list,
    )


@dataclass(frozen=True)
class AnalysisBundle:
    """One reproducible config-first analysis and its presentation metadata."""

    resolved: ResolvedConfig
    adapter_result: AdapterResult
    model_map: ModelMap
    captures: Tuple[Mapping[str, Any], ...] = ()
    cost_analysis: Optional[CostAnalysis] = None
    hotspot_summaries: Tuple[Mapping[str, Any], ...] = ()
    hardware_profile: Optional[HardwareProfile] = None
    roofline_summaries: Tuple[Mapping[str, Any], ...] = ()

    @property
    def graph_view(self) -> GraphViewDocument:
        """Return the config-evidenced recursive DAG without constructing the target model."""

        from llm_vis.graph_view import build_graph_view_document

        return build_graph_view_document(self.adapter_result, self.model_map)

    @property
    def analysis_id(self) -> str:
        return deterministic_id(
            "analysis",
            {
                "adapter": self.adapter_result.adapter_name,
                "config_sha256": self.resolved.sha256,
                "captures": [dict(item) for item in self.captures],
                "model_revision": self.resolved.revision,
                "cost_capability": {
                    "enabled": self.cost_analysis is not None,
                    "model_version": _COST_MODEL_VERSION,
                    "hotspot_metrics": list(_HOTSPOT_METRICS),
                },
                "hardware_profile": (
                    self.hardware_profile.model_dump(mode="json")
                    if self.hardware_profile is not None
                    else None
                ),
                "graph_view_version": _GRAPH_VIEW_VERSION,
                "roofline_enabled": bool(self.roofline_summaries),
                "scenario_ids": [scenario.id for scenario in self.model_map.scenarios],
                "schema_version": self.model_map.schema_version,
                "tool_version": __version__,
            },
        )

    @property
    def layer_strip(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(asdict(entry) for entry in self.adapter_result.layer_strip)

    def manifest(self) -> Dict[str, Any]:
        """Return a deterministic manifest; wall-clock timestamps are excluded."""

        has_capture = bool(self.model_map.logical_ops)
        capability_level = "C2" if has_capture else "C1"
        graph_view = self.graph_view
        graph_nodes = sum(len(view.nodes) for view in graph_view.views)
        graph_edges = sum(len(view.edges) for view in graph_view.views)
        has_recursive_operator_decomposition = any(
            view.decomposes_node_id is not None for view in graph_view.views
        )
        return {
            "schema_version": "0.1",
            "analysis_id": self.analysis_id,
            "tool": {"name": "llm-vis", "version": __version__},
            "capability": {
                "level": capability_level,
                "config_resolved": True,
                "macro_structure": True,
                "representative_block_captured": has_capture,
                "cost_analyzed": self.cost_analysis is not None,
                "interactive_semantic_dag": True,
                "semantic_zoom_levels": ["L0", "L1"],
                "recursive_operator_decomposition": has_recursive_operator_decomposition,
                "roofline_lower_bounds": bool(self.roofline_summaries),
                "runtime_trace_mapped": False,
            },
            "safety": {
                "weights_loaded": False,
                "full_model_constructed": False,
                "full_forward_executed": False,
                "remote_code_executed": False,
                "full_meta_tree_enabled": False,
            },
            "model": {
                "id": self.model_map.model.id,
                "name": self.model_map.model.name,
                "family": self.model_map.model.family,
                "revision": self.model_map.model.revision,
                "config_sha256": self.resolved.sha256,
            },
            "model_map": {
                "artifact_id": graph_view.source_model_map_id,
                "schema_version": self.model_map.schema_version,
            },
            "source": {
                "uri": self.resolved.source_uri,
                "cache_hit": self.resolved.cache_hit,
                "is_remote": self.resolved.is_remote,
                "license": self.resolved.license,
            },
            "adapter": {
                "name": self.adapter_result.adapter_name,
                "metadata": dict(self.adapter_result.metadata),
            },
            "cost": {
                "enabled": self.cost_analysis is not None,
                "model_version": _COST_MODEL_VERSION,
                "scenario_count": (
                    len(self.cost_analysis.scenario_costs) if self.cost_analysis is not None else 0
                ),
                "metric_count": (
                    len(self.cost_analysis.metrics) if self.cost_analysis is not None else 0
                ),
                "hotspot_metrics": list(_HOTSPOT_METRICS),
                "hotspot_limit_per_metric": _HOTSPOT_LIMIT,
                "unknown_values_preserved": True,
                "aggregate_scopes": (
                    [
                        _aggregate_scope_summary(scenario_cost)
                        for scenario_cost in self.cost_analysis.scenario_costs
                    ]
                    if self.cost_analysis is not None
                    else []
                ),
                "hotspot_summaries": [dict(item) for item in self.hotspot_summaries],
            },
            "hardware_profile": (
                self.hardware_profile.model_dump(mode="json")
                if self.hardware_profile is not None
                else None
            ),
            "roofline": {
                "enabled": bool(self.roofline_summaries),
                "is_latency_estimate": False,
                "interpretation": "theoretical_lower_bounds_not_latency_estimate",
                "summaries": [dict(item) for item in self.roofline_summaries],
            },
            "captures": [dict(item) for item in self.captures],
            "counts": {
                "definitions": len(self.model_map.definitions),
                "instances": len(self.model_map.instances),
                "semantic_nodes": len(self.model_map.semantic_nodes),
                "logical_ops": len(self.model_map.logical_ops),
                "metrics": len(self.model_map.metrics),
                "diagnostics": len(self.model_map.diagnostics),
                "graph_views": len(graph_view.views),
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
            },
            "artifacts": [
                "manifest.json",
                "model-map.json",
                "scenarios.json",
                "metrics.json",
                "diagnostics.json",
                "layer-strip.json",
                "captures.json",
                "hardware-profile.json",
                "hotspots.json",
                "roofline.json",
                "workload-diffs.json",
                "model-explorer.json",
                "graph-view.json",
                "reports/report.md",
                "reports/report.html",
            ],
        }


def _with_provenance(
    model_map: ModelMap,
    resolved: ResolvedConfig,
    scenarios: Sequence[Scenario],
) -> ModelMap:
    original_source = model_map.source_artifacts[0]
    if resolved.is_remote:
        source = SourceArtifact(
            id=original_source.id,
            kind=SourceArtifactKind.HUGGINGFACE_CONFIG,
            uri=resolved.source_uri,
            revision=resolved.revision,
            sha256=resolved.sha256,
            license=resolved.license,
            trusted=False,
        )
    else:
        source = SourceArtifact(
            id=original_source.id,
            kind=SourceArtifactKind.LOCAL_CONFIG,
            path=resolved.source_uri,
            revision=resolved.revision,
            sha256=resolved.sha256,
            license=resolved.license,
            trusted=True,
        )

    diagnostics = list(model_map.diagnostics)
    diagnostics.append(
        _diagnostic(
            "CONFIG_FIRST_NO_EXECUTION",
            "Structure was generated from configuration without weights or a model forward pass.",
            subject_id=model_map.model.id,
            evidence=["weights_loaded=false", "full_forward_executed=false"],
        )
    )
    declarations = _remote_code_declarations(resolved.config)
    if declarations:
        diagnostics.append(
            _diagnostic(
                "REMOTE_CODE_EXECUTION_DISABLED",
                "The configuration declares remote code; execution is disabled for M0-M3.6.",
                subject_id=model_map.model.id,
                severity=DiagnosticSeverity.WARNING,
                evidence=declarations,
            )
        )
    diagnostics.append(
        _diagnostic(
            "FULL_META_TREE_DISABLED",
            "Full meta-model construction is disabled; config-first templates remain "
            "authoritative.",
            subject_id=model_map.model.id,
        )
    )
    diagnostics.append(
        _diagnostic(
            "LOGICAL_OPS_NOT_CAPTURED",
            "Config-first inspection provides semantic contracts. Logical ops are absent "
            "unless bounded representative-block capture is requested; full-model ops "
            "remain intentionally uncaptured.",
            subject_id=model_map.model.id,
        )
    )
    diagnostics.append(
        _diagnostic(
            "RUNTIME_TRACE_MISSING",
            "No external runtime trace was imported; every Runtime metric remains Unknown.",
            subject_id=model_map.model.id,
            evidence=["runtime_trace_mapped=false", "runtime_metrics=value:null"],
        )
    )

    runtime_metrics = []
    scenario_ids = [scenario.id for scenario in scenarios] or [None]
    for scenario_id in scenario_ids:
        for name, unit in (
            ("runtime.measured_latency", "seconds"),
            ("runtime.kernel_dispatch_count", "count"),
            ("runtime.measured_hbm_traffic", "bytes"),
        ):
            runtime_metrics.append(
                Metric(
                    id=deterministic_id(
                        "metric",
                        {
                            "name": name,
                            "scenario_id": scenario_id,
                            "subject_id": model_map.model.id,
                        },
                    ),
                    subject_id=model_map.model.id,
                    scenario_id=scenario_id,
                    name=name,
                    value=None,
                    unit=unit,
                    origin=MetricOrigin.UNKNOWN,
                    assumptions=["No external runtime trace was imported."],
                    confidence=None,
                    coverage=0.0,
                    coverage_status=CoverageStatus.UNMAPPED,
                )
            )

    payload = model_map.model_dump(mode="json")
    payload["source_artifacts"] = [source.model_dump(mode="json")]
    payload["scenarios"] = [scenario.model_dump(mode="json") for scenario in scenarios]
    payload["diagnostics"] = [item.model_dump(mode="json") for item in diagnostics]
    payload["metrics"] = [
        *payload["metrics"],
        *(item.model_dump(mode="json") for item in runtime_metrics),
    ]
    return ModelMap.model_validate(payload)


def _metric_summary(metric: Metric) -> Dict[str, Any]:
    """Return a compact JSON-safe representation without changing Unknown values."""

    return {
        "name": metric.name,
        "value": metric.value,
        "unit": metric.unit,
        "origin": metric.origin.value,
        "coverage": metric.coverage,
        "coverage_status": metric.coverage_status.value,
    }


def _aggregate_scope_summary(scenario_cost: ScenarioCost) -> Dict[str, Any]:
    scope = scenario_cost.aggregate_scope
    return {
        "scenario_id": scenario_cost.scenario.id,
        "name": scope.name,
        "subject_id": scope.subject_id,
        "included_instance_count": scope.included_instance_count,
        "excluded_instance_count": scope.excluded_instance_count,
        "excluded_instance_paths": list(scope.excluded_instance_paths),
        "structural_coverage": scope.structural_coverage,
        "coverage_basis": "structural_instances_not_cost_fraction",
    }


def _hotspot_summaries(cost_analysis: CostAnalysis) -> Tuple[Mapping[str, Any], ...]:
    summaries = []
    for scenario_cost in cost_analysis.scenario_costs:
        for metric_name in _HOTSPOT_METRICS:
            aggregate = scenario_cost.aggregate_metric(metric_name)
            entries = scenario_cost.hotspots(metric_name, limit=_HOTSPOT_LIMIT)
            summaries.append(
                {
                    "scenario_id": scenario_cost.scenario.id,
                    "phase": scenario_cost.scenario.phase.value,
                    "metric_name": metric_name,
                    "aggregate_scope": _aggregate_scope_summary(scenario_cost),
                    "aggregate": _metric_summary(aggregate),
                    "hotspots": [
                        {
                            "rank": entry.rank,
                            "subject_id": entry.subject_id,
                            "module_path": entry.module_path,
                            "layer_index": entry.layer_index,
                            "value": entry.value,
                            "unit": entry.unit,
                            "origin": entry.origin.value,
                            "coverage": entry.coverage,
                        }
                        for entry in entries
                    ],
                }
            )
    return tuple(summaries)


def _roofline_summaries(
    cost_analysis: CostAnalysis,
    hardware_profile: HardwareProfile,
) -> Tuple[Mapping[str, Any], ...]:
    """Build scenario bounds from aggregate model-level work metrics only."""

    summaries = []
    for scenario_cost in cost_analysis.scenario_costs:
        flops_metric = scenario_cost.aggregate_metric("flops")
        logical_bytes_metric = scenario_cost.aggregate_metric("logical_bytes")
        result = roofline_from_metrics(
            flops_metric=flops_metric,
            logical_bytes_metric=logical_bytes_metric,
            hardware_profile=hardware_profile,
            workload_dtype=scenario_cost.scenario.activation_dtype,
        )
        summaries.append(
            {
                "scenario_id": scenario_cost.scenario.id,
                "phase": scenario_cost.scenario.phase.value,
                "source_subject_id": scenario_cost.aggregate_scope.subject_id,
                "source_scope": _aggregate_scope_summary(scenario_cost),
                "source_metrics": {
                    "flops": _metric_summary(flops_metric),
                    "logical_bytes": _metric_summary(logical_bytes_metric),
                },
                **result.model_dump(mode="json"),
            }
        )
    return tuple(summaries)


def inspect_model(
    source: str,
    *,
    revision: Optional[str] = None,
    scenarios: Sequence[Scenario] = (),
    local_files_only: bool = False,
    cache_dir: Optional[Path] = None,
    timeout_seconds: float = 20.0,
    hardware_profile: Optional[HardwareProfile] = None,
) -> AnalysisBundle:
    """Resolve and inspect one model without loading weights or executing model code.

    Cost metrics are generated when at least one Scenario is supplied. Roofline
    lower bounds are generated only when the caller also supplies a hardware
    profile; they are theoretical lower bounds, never latency estimates.
    """

    resolved = resolve_config(
        source,
        revision=revision,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    adapter_result = build_from_config(
        resolved.config,
        model_id=resolved.identifier,
        revision=resolved.revision,
    )
    model_map = _with_provenance(adapter_result.to_model_map(), resolved, scenarios)
    if not scenarios:
        return AnalysisBundle(
            resolved=resolved,
            adapter_result=adapter_result,
            model_map=model_map,
            hardware_profile=hardware_profile,
        )

    cost_analysis = analyze_costs(
        resolved.config,
        adapter_result,
        model_map,
        scenarios,
    )
    model_map = cost_analysis.attach_to_model_map(model_map)
    hotspot_summaries = _hotspot_summaries(cost_analysis)
    roofline_summaries = (
        _roofline_summaries(cost_analysis, hardware_profile) if hardware_profile is not None else ()
    )
    return AnalysisBundle(
        resolved=resolved,
        adapter_result=adapter_result,
        model_map=model_map,
        cost_analysis=cost_analysis,
        hotspot_summaries=hotspot_summaries,
        hardware_profile=hardware_profile,
        roofline_summaries=roofline_summaries,
    )
