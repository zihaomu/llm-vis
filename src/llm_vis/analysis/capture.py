"""Merge bounded Tiny representative captures into a config-first analysis."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_vis.analysis.service import AnalysisBundle
from llm_vis.capture import (
    ExportCaptureResult,
    RepresentativeCandidate,
    RepresentativeKind,
    RepresentativeSelection,
    capture_tiny_representative,
    select_representatives,
)
from llm_vis.ir import (
    Diagnostic,
    DiagnosticSeverity,
    Lowering,
    LoweringKind,
    Model,
    ModelMap,
    SemanticKind,
    SemanticNode,
    SourceArtifact,
    SourceArtifactKind,
    TensorOrigin,
    canonical_json,
    deterministic_id,
)

CaptureFunction = Callable[[RepresentativeKind], ExportCaptureResult]


def _dense_target_id(bundle: AnalysisBundle, instance_path: str) -> str:
    """Return the stable config-semantic identity for one dense FFN component."""

    return deterministic_id(
        "art",
        {
            "adapter": bundle.adapter_result.adapter_name,
            "kind": SemanticKind.FFN.value,
            "model_revision": bundle.model_map.model.revision,
            "semantic_path": f"{instance_path}.mlp",
            "source": "config",
        },
    )


def _target_semantic_ids(bundle: AnalysisBundle) -> Dict[str, str]:
    paths_by_instance = {item.id: item.module_path for item in bundle.model_map.instances}
    targets: Dict[str, str] = {}
    block_kinds = {
        SemanticKind.FULL_ATTENTION,
        SemanticKind.LINEAR_ATTENTION,
        SemanticKind.DSA,
    }
    for node in bundle.model_map.semantic_nodes:
        if node.kind not in block_kinds:
            continue
        for instance_id in node.instance_ids:
            module_path = paths_by_instance.get(instance_id)
            if module_path is not None:
                targets[module_path] = node.id
    return targets


def representative_candidates(bundle: AnalysisBundle) -> Tuple[RepresentativeCandidate, ...]:
    """Derive safe representative choices from the config-first layer strip."""

    if bundle.adapter_result.adapter_name == "glm-moe-dsa":
        # GLM DSA/MoE dynamic capture is explicitly deferred to M5.
        return ()
    targets = _target_semantic_ids(bundle)
    definitions_by_instance = {item.id: item.definition_id for item in bundle.model_map.instances}
    instance_ids_by_path = {item.module_path: item.id for item in bundle.model_map.instances}
    candidates: List[RepresentativeCandidate] = []
    for entry in bundle.adapter_result.layer_strip:
        target_id = targets.get(entry.instance_path)
        instance_id = instance_ids_by_path.get(entry.instance_path)
        if target_id is None or instance_id is None:
            continue
        definition_id = definitions_by_instance[instance_id]
        if entry.mlp_kind == "dense":
            candidates.append(
                RepresentativeCandidate(
                    kind=RepresentativeKind.DENSE,
                    instance_id=_dense_target_id(bundle, entry.instance_path),
                    module_path=f"{entry.instance_path}.mlp",
                    definition_id=definition_id,
                    layer_index=entry.layer_index,
                )
            )
        if entry.attention_kind == "full_attention":
            candidates.append(
                RepresentativeCandidate(
                    kind=RepresentativeKind.FULL_ATTENTION,
                    instance_id=target_id,
                    module_path=f"{entry.instance_path}.self_attn",
                    definition_id=definition_id,
                    layer_index=entry.layer_index,
                )
            )
        elif entry.attention_kind == "linear_attention":
            candidates.append(
                RepresentativeCandidate(
                    kind=RepresentativeKind.LINEAR_STATE,
                    instance_id=target_id,
                    module_path=f"{entry.instance_path}.linear_attn",
                    definition_id=definition_id,
                    layer_index=entry.layer_index,
                )
            )
    return tuple(candidates)


def _capture_source(result: ExportCaptureResult) -> SourceArtifact:
    contract = {
        "backend": "torch.export.strict",
        "capture_id": result.capture_id,
        "coverage": result.coverage,
        "fixture_revision": "tiny-fixture-v0.1",
        "kind": result.kind.value,
        "logical_ops": [item.model_dump(mode="json") for item in result.logical_ops],
        "semantic_nodes": [item.model_dump(mode="json") for item in result.semantic_nodes],
        "status": result.status.value,
        "tensors": [item.model_dump(mode="json") for item in result.tensors],
        "tensor_mode": "meta-parameters-and-inputs/faketensor-trace",
    }
    return SourceArtifact(
        id=deterministic_id(
            "source",
            {"capture_id": result.capture_id, "kind": result.kind.value},
        ),
        kind=SourceArtifactKind.GENERATED,
        uri=f"generated://llm-vis/tiny-representative/{result.kind.value}",
        revision="tiny-fixture-v0.1",
        sha256=hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest(),
        license="Apache-2.0",
        trusted=True,
    )


def _torch_version() -> str:
    try:
        return version("torch")
    except PackageNotFoundError:
        return "unavailable"


def _not_supported_diagnostic(model_id: str) -> Diagnostic:
    return Diagnostic(
        id=deterministic_id("diag", {"code": "CAPTURE_DEFERRED_TO_M5", "subject_id": model_id}),
        severity=DiagnosticSeverity.INFO,
        code="CAPTURE_DEFERRED_TO_M5",
        subject_id=model_id,
        message=(
            "This adapter has no M2 Tiny representative capture; GLM DSA/MoE is deferred to M5."
        ),
        evidence=["bounded-capture-policy", "no-full-model-fallback"],
    )


def _kind_not_applicable_diagnostic(
    model_id: str,
    requested: Iterable[RepresentativeKind],
    available: Iterable[RepresentativeKind],
) -> Diagnostic:
    requested_values = sorted({item.value for item in requested})
    available_values = sorted({item.value for item in available})
    return Diagnostic(
        id=deterministic_id(
            "diag",
            {
                "available": available_values,
                "code": "CAPTURE_KIND_NOT_APPLICABLE",
                "requested": requested_values,
                "subject_id": model_id,
            },
        ),
        severity=DiagnosticSeverity.INFO,
        code="CAPTURE_KIND_NOT_APPLICABLE",
        subject_id=model_id,
        message=(
            "The requested Tiny representative kind is not applicable to this adapter; "
            "no capture was executed."
        ),
        evidence=[
            f"requested:{','.join(requested_values) or 'none'}",
            f"available:{','.join(available_values) or 'none'}",
            "no-full-model-fallback",
        ],
    )


def _bounded_scope_diagnostic(model_id: str) -> Diagnostic:
    return Diagnostic(
        id=deterministic_id(
            "diag", {"code": "CAPTURE_SCOPE_REPRESENTATIVE_ONLY", "subject_id": model_id}
        ),
        severity=DiagnosticSeverity.INFO,
        code="CAPTURE_SCOPE_REPRESENTATIVE_ONLY",
        subject_id=model_id,
        message=(
            "Logical ops come from project-owned Tiny semantic representatives only; "
            "the full model was not constructed or executed."
        ),
        evidence=[
            "capture_fixture:tiny-meta-faketensor",
            "architecture_equivalence:semantic-kind-only",
            "valid_for_full_model_performance:false",
        ],
    )


def _uncaptured_region_diagnostics(
    bundle: AnalysisBundle,
    captured_semantic_ids: Iterable[str],
) -> Tuple[Diagnostic, ...]:
    captured = set(captured_semantic_ids)
    block_kinds = {SemanticKind.FULL_ATTENTION, SemanticKind.LINEAR_ATTENTION}
    paths_by_instance = {item.id: item.module_path for item in bundle.model_map.instances}
    diagnostics = []
    for node in bundle.model_map.semantic_nodes:
        if node.kind not in block_kinds or node.id in captured:
            continue
        module_paths = sorted(
            paths_by_instance[instance_id]
            for instance_id in node.instance_ids
            if instance_id in paths_by_instance
        )
        evidence = [
            "coverage=0",
            "coverage_scope=decoder_attention_semantic_nodes",
            "logical_ops=config-only",
        ]
        evidence.extend(f"module_path:{path}" for path in module_paths)
        diagnostics.append(
            Diagnostic(
                id=deterministic_id(
                    "diag",
                    {
                        "code": "CAPTURE_REGION_NOT_SELECTED",
                        "subject_id": node.id,
                    },
                ),
                severity=DiagnosticSeverity.INFO,
                code="CAPTURE_REGION_NOT_SELECTED",
                subject_id=node.id,
                message=(
                    f"{node.label} remains config-only; no representative LogicalOp graph "
                    "was attached to this region."
                ),
                evidence=evidence,
            )
        )
    return tuple(diagnostics)


def _merge_results(
    bundle: AnalysisBundle,
    selections: Sequence[RepresentativeSelection],
    results: Sequence[ExportCaptureResult],
) -> AnalysisBundle:
    payload = bundle.model_map.model_dump(mode="json")
    semantic_by_id = {item["id"]: item for item in payload["semantic_nodes"]}
    sources = list(payload["source_artifacts"])
    source_ids = list(payload["model"]["source_artifact_ids"])
    summaries = []
    captured_semantic_ids = []
    instances_by_path = {item["module_path"]: item for item in payload["instances"]}

    for selection, result in zip(selections, results):
        source = _capture_source(result)
        sources.append(source.model_dump(mode="json"))
        source_ids.append(source.id)
        capture_evidence = [
            f"capture_id:{result.capture_id}",
            "capture_backend:torch.export.strict",
            "capture_fixture:tiny-meta-faketensor",
            "tensor_mode:meta-parameters-and-inputs/faketensor-trace",
        ]
        payload["tensors"].extend(
            {
                **item.model_dump(mode="json"),
                "origin": TensorOrigin.CAPTURE.value,
                "materialized": False,
                "source_artifact_ids": [source.id],
                "evidence": capture_evidence,
            }
            for item in result.tensors
        )
        scope_path = selection.module_path.rsplit(".", 1)[0]
        scope_instance = instances_by_path.get(scope_path)
        payload["logical_ops"].extend(
            {
                **item.model_dump(mode="json"),
                "attrs": {
                    **item.attrs,
                    "capture_id": result.capture_id,
                    "capture_backend": "torch.export.strict",
                    "source_artifact_id": source.id,
                    "tensor_mode": "meta-parameters-and-inputs/faketensor-trace",
                },
                "scope_instance_id": (scope_instance["id"] if scope_instance is not None else None),
            }
            for item in result.logical_ops
        )
        payload["semantic_nodes"].extend(
            item.model_dump(mode="json") for item in result.semantic_nodes
        )
        payload["diagnostics"].extend(item.model_dump(mode="json") for item in result.diagnostics)

        target = semantic_by_id.get(selection.instance_id)
        if target is None:
            if selection.kind != RepresentativeKind.DENSE:
                raise ValueError(
                    f"capture target semantic node does not exist: {selection.instance_id}"
                )
            layer_path = selection.module_path.removesuffix(".mlp")
            instance = instances_by_path.get(layer_path)
            if instance is None:
                raise ValueError(f"dense capture layer instance does not exist: {layer_path}")
            target_node = SemanticNode(
                id=selection.instance_id,
                kind=SemanticKind.FFN,
                label=f"Dense FFN layer {selection.layer_index}",
                definition_id=selection.definition_id,
                instance_ids=[instance["id"]],
                confidence=1.0,
                evidence=[
                    f"adapter:{bundle.adapter_result.adapter_name}",
                    "source:config",
                    "component:mlp",
                    "mlp_kind:dense",
                ],
            )
            target = target_node.model_dump(mode="json")
            payload["semantic_nodes"].append(target)
            semantic_by_id[target_node.id] = target
        capture_node_ids = [item.id for item in result.semantic_nodes]
        target["child_ids"] = list(dict.fromkeys([*target["child_ids"], *capture_node_ids]))
        target["evidence"] = list(
            dict.fromkeys(
                [
                    *target["evidence"],
                    f"bounded_capture:{result.capture_id}",
                    f"capture_coverage:{result.coverage:.6f}",
                    "capture_fixture:tiny-meta-faketensor",
                ]
            )
        )

        if result.logical_ops:
            captured_semantic_ids.append(selection.instance_id)
            lowering = Lowering(
                id=deterministic_id(
                    "lowering",
                    {
                        "capture_id": result.capture_id,
                        "source": selection.instance_id,
                        "targets": [item.id for item in result.logical_ops],
                    },
                ),
                source_ids=[selection.instance_id, *capture_node_ids],
                target_ids=[item.id for item in result.logical_ops],
                kind=LoweringKind.EXPAND,
            )
            payload["lowerings"].append(lowering.model_dump(mode="json"))

        summaries.append(
            {
                "kind": result.kind.value,
                "status": result.status.value,
                "capture_id": result.capture_id,
                "coverage": result.coverage,
                "representative_module_path": selection.module_path,
                "representative_layer_index": selection.layer_index,
                "semantic_kind": (
                    result.semantic_nodes[0].kind.value if result.semantic_nodes else "unknown"
                ),
                "logical_op_count": len(result.logical_ops),
                "tensor_count": len(result.tensors),
                "fixture": "tiny-meta-faketensor",
                "capture_source": "project-owned-tiny-fixture",
                "source_artifact_id": source.id,
                "capture_backend": "torch.export.strict",
                "tensor_mode": "meta-parameters-and-inputs/faketensor-trace",
                "torch_version": _torch_version(),
                "architecture_equivalence": "semantic-kind-only",
                "valid_for_full_model_performance": False,
                "weights_loaded": False,
                "full_forward_executed": False,
            }
        )

    payload["diagnostics"].append(
        _bounded_scope_diagnostic(bundle.model_map.model.id).model_dump(mode="json")
    )
    payload["diagnostics"].extend(
        diagnostic.model_dump(mode="json")
        for diagnostic in _uncaptured_region_diagnostics(bundle, captured_semantic_ids)
    )
    if payload["logical_ops"]:
        payload["diagnostics"] = [
            item for item in payload["diagnostics"] if item["code"] != "LOGICAL_OPS_NOT_CAPTURED"
        ]
    payload["source_artifacts"] = sources
    payload["model"] = Model.model_validate(
        {**payload["model"], "source_artifact_ids": source_ids}
    ).model_dump(mode="json")
    merged = ModelMap.model_validate(payload)
    return replace(
        bundle,
        model_map=merged,
        captures=tuple([*bundle.captures, *summaries]),
    )


def capture_representatives(
    bundle: AnalysisBundle,
    *,
    kinds: Optional[Iterable[RepresentativeKind]] = None,
    capture_fn: CaptureFunction = capture_tiny_representative,
) -> AnalysisBundle:
    """Capture the minimum safe representative set and merge it into Model Map IR."""

    candidates = representative_candidates(bundle)
    available_kinds = {item.kind for item in candidates}
    requested_kinds: Optional[set[RepresentativeKind]] = None
    if kinds is not None:
        requested_kinds = {RepresentativeKind(kind) for kind in kinds}
        candidates = tuple(item for item in candidates if item.kind in requested_kinds)
    selections = select_representatives(candidates)
    if not selections:
        payload = bundle.model_map.model_dump(mode="json")
        diagnostic = (
            _kind_not_applicable_diagnostic(
                bundle.model_map.model.id,
                requested_kinds,
                available_kinds,
            )
            if requested_kinds is not None and available_kinds
            else _not_supported_diagnostic(bundle.model_map.model.id)
        )
        payload["diagnostics"].append(diagnostic.model_dump(mode="json"))
        return replace(bundle, model_map=ModelMap.model_validate(payload))
    results = tuple(capture_fn(selection.kind) for selection in selections)
    return _merge_results(bundle, selections, results)


__all__ = ["capture_representatives", "representative_candidates"]
