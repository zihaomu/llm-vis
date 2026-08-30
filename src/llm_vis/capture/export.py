"""Safe ``torch.export`` orchestration for project-owned representative blocks."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from llm_vis.ir import (
    Diagnostic,
    SemanticNode,
    artifact_local_id,
    deterministic_id,
)

from . import fixtures
from .converter import IRConversion, convert_exported_program
from .patterns import classify_semantic_pattern, detect_semantic_components
from .types import (
    CaptureStatus,
    ExportCaptureResult,
    RepresentativeKind,
    TinyRepresentativeSpec,
    UnsupportedCaptureError,
)

ExportFunction = Callable[[Any, Tuple[Any, ...]], Any]


def _capture_id(spec: TinyRepresentativeSpec) -> str:
    return artifact_local_id(
        model_revision="tiny-fixture-v0.1",
        framework_domain="torch.export.capture",
        module_path=spec.module_path,
        capture_variant=spec.kind.value,
    )


def _semantic_id(spec: TinyRepresentativeSpec, *, opaque: bool) -> str:
    suffix = "opaque" if opaque else "semantic"
    return artifact_local_id(
        model_revision="tiny-fixture-v0.1",
        framework_domain="semantic",
        module_path=f"{spec.module_path}.{suffix}",
        capture_variant=spec.kind.value,
    )


def _diagnostic(
    *,
    capture_id: str,
    code: str,
    message: str,
    subject_id: str,
    severity: str = "warning",
) -> Diagnostic:
    return Diagnostic(
        id=deterministic_id(
            "diag",
            {
                "capture_id": capture_id,
                "code": code,
                "message": message,
                "subject_id": subject_id,
            },
        ),
        severity=severity,
        code=code,
        subject_id=subject_id,
        message=message,
    )


def _opaque_result(
    *,
    spec: TinyRepresentativeSpec,
    status: CaptureStatus,
    code: str,
    message: str,
) -> ExportCaptureResult:
    capture_id = _capture_id(spec)
    node_id = _semantic_id(spec, opaque=True)
    node = SemanticNode(
        id=node_id,
        kind="opaque",
        label=f"{spec.name} (opaque)",
        evidence=[f"capture_diagnostic={code}"],
        confidence=1.0,
        opaque=True,
    )
    diagnostic = _diagnostic(
        capture_id=capture_id,
        code=code,
        subject_id=node_id,
        message=message,
    )
    return ExportCaptureResult(
        kind=spec.kind,
        status=status,
        capture_id=capture_id,
        coverage=0.0,
        semantic_nodes=(node,),
        diagnostics=(diagnostic,),
    )


def _default_export(torch: Any, module: Any, args: Tuple[Any, ...]) -> Any:
    export_api = getattr(getattr(torch, "export", None), "export", None)
    if export_api is None:
        raise UnsupportedCaptureError("installed torch does not provide torch.export.export")
    # No eager fallback is permitted. torch.export performs its own FakeTensor
    # tracing; module parameters and example inputs remain on the meta device.
    return export_api(module, args, strict=True)


def _build_semantic_nodes(
    spec: TinyRepresentativeSpec,
    conversion: IRConversion,
) -> Tuple[SemanticNode, ...]:
    semantic_kind, evidence = classify_semantic_pattern(
        conversion.logical_ops,
        hint=spec.kind,
    )
    component_specs = [
        (kind, component_evidence)
        for kind, component_evidence in detect_semantic_components(conversion.logical_ops)
        if kind != semantic_kind
    ]
    component_nodes = tuple(
        SemanticNode(
            id=artifact_local_id(
                model_revision="tiny-fixture-v0.1",
                framework_domain="semantic.component",
                module_path=f"{spec.module_path}.{kind.value}.{ordinal}",
                capture_variant=spec.kind.value,
            ),
            kind=kind,
            label=f"{spec.name} / {kind.value}",
            confidence=1.0,
            evidence=list(component_evidence),
            opaque=False,
        )
        for ordinal, (kind, component_evidence) in enumerate(component_specs)
    )
    root = SemanticNode(
        id=_semantic_id(spec, opaque=False),
        kind=semantic_kind,
        label=spec.name,
        input_tensor_ids=list(conversion.graph_input_ids),
        output_tensor_ids=list(conversion.graph_output_ids),
        child_ids=[node.id for node in component_nodes],
        confidence=1.0,
        evidence=list(evidence),
        opaque=False,
    )
    return (root, *component_nodes)


def capture_tiny_representative(
    kind: RepresentativeKind,
    *,
    torch_module: Any = None,
    export_fn: Optional[ExportFunction] = None,
) -> ExportCaptureResult:
    """Capture one project-owned tiny block with no full-model fallback.

    ``torch_module`` and ``export_fn`` are dependency-injection seams used by
    capability tests and fault injection. They do not broaden capture to
    arbitrary model code.
    """

    resolved_kind = RepresentativeKind(kind)
    spec = fixtures.TINY_REPRESENTATIVE_SPECS[resolved_kind]
    try:
        torch = torch_module if torch_module is not None else fixtures._load_torch()
    except Exception as error:
        return _opaque_result(
            spec=spec,
            status=CaptureStatus.UNAVAILABLE,
            code="CAPTURE_TORCH_UNAVAILABLE",
            message=f"PyTorch capture is unavailable: {error}",
        )

    try:
        built = fixtures.build_tiny_representative(resolved_kind, torch_module=torch)
    except Exception as error:
        return _opaque_result(
            spec=spec,
            status=CaptureStatus.OPAQUE,
            code="CAPTURE_FIXTURE_BUILD_FAILED",
            message=f"Tiny representative construction failed: {type(error).__name__}: {error}",
        )

    try:
        if export_fn is None:
            exported_program = _default_export(torch, built.module, built.args)
        else:
            exported_program = export_fn(built.module, built.args)
    except UnsupportedCaptureError as error:
        return _opaque_result(
            spec=spec,
            status=CaptureStatus.OPAQUE,
            code="CAPTURE_UNSUPPORTED",
            message=f"Representative export is unsupported: {error}",
        )
    except Exception as error:
        return _opaque_result(
            spec=spec,
            status=CaptureStatus.OPAQUE,
            code="CAPTURE_EXPORT_FAILED",
            message=f"Representative export failed: {type(error).__name__}: {error}",
        )

    capture_id = _capture_id(spec)
    try:
        conversion = convert_exported_program(
            exported_program,
            spec=spec,
            capture_id=capture_id,
        )
    except Exception as error:
        return _opaque_result(
            spec=spec,
            status=CaptureStatus.OPAQUE,
            code="CAPTURE_IR_CONVERSION_FAILED",
            message=f"ExportedProgram conversion failed: {type(error).__name__}: {error}",
        )

    if conversion.total_ops == 0 or not conversion.logical_ops:
        return _opaque_result(
            spec=spec,
            status=CaptureStatus.OPAQUE,
            code="CAPTURE_EMPTY_GRAPH",
            message="Export completed but produced no convertible logical operators",
        )

    semantic_nodes = _build_semantic_nodes(spec, conversion)
    semantic_node = semantic_nodes[0]
    diagnostics = list(conversion.diagnostics)
    status = CaptureStatus.CAPTURED
    if conversion.coverage < 1.0:
        status = CaptureStatus.PARTIAL
        diagnostics.append(
            _diagnostic(
                capture_id=capture_id,
                code="CAPTURE_PARTIAL_COVERAGE",
                subject_id=semantic_node.id,
                message=(
                    f"Converted {conversion.converted_ops} of "
                    f"{conversion.total_ops} exported operators"
                ),
            )
        )

    return ExportCaptureResult(
        kind=resolved_kind,
        status=status,
        capture_id=capture_id,
        coverage=conversion.coverage,
        tensors=conversion.tensors,
        logical_ops=conversion.logical_ops,
        semantic_nodes=semantic_nodes,
        diagnostics=tuple(diagnostics),
        exported_program=exported_program,
    )
