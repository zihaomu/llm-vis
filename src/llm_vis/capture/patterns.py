"""Small, deterministic semantic classifier for representative exports."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from llm_vis.ir import LogicalOp, SemanticKind

from .types import RepresentativeKind

SemanticPattern = Tuple[SemanticKind, Tuple[str, ...]]


def _op_inventory(logical_ops: Iterable[LogicalOp]) -> Tuple[Tuple[LogicalOp, ...], str]:
    operations = tuple(logical_ops)
    return operations, "\n".join(op.op_type.lower() for op in operations)


def detect_semantic_components(logical_ops: Iterable[LogicalOp]) -> Tuple[SemanticPattern, ...]:
    """Return every deterministic semantic component evidenced by an op graph.

    This catalog deliberately recognizes components in addition to the enclosing
    representative kind, so a Dense fixture does not hide its Norm/residual and
    a stateful linear-attention fixture does not hide its recurrent state update.
    """

    operations, joined = _op_inventory(logical_ops)
    matches = []
    linear_count = sum("linear" in op.op_type.lower() for op in operations)
    matmul_count = sum("matmul" in op.op_type.lower() for op in operations)
    einsum_count = sum("einsum" in op.op_type.lower() for op in operations)

    if "layer_norm" in joined or "rms_norm" in joined:
        matches.append((SemanticKind.NORM, ("op_pattern=layer_norm|rms_norm",)))
    if "softmax" in joined and matmul_count >= 2:
        matches.append((SemanticKind.FULL_ATTENTION, ("op_pattern=softmax+matmul>=2",)))
    if einsum_count >= 2 and "sigmoid" in joined:
        matches.append((SemanticKind.LINEAR_ATTENTION, ("op_pattern=sigmoid+einsum>=2",)))
        matches.append((SemanticKind.RECURRENT_STATE, ("op_pattern=einsum_state_update",)))
    if any(activation in joined for activation in ("silu", "gelu")) and linear_count >= 2:
        matches.append((SemanticKind.FFN, ("op_pattern=linear>=2+activation",)))
    if ("add." in joined or "add.tensor" in joined) and (
        "layer_norm" in joined or "rms_norm" in joined
    ):
        matches.append((SemanticKind.RESIDUAL, ("op_pattern=residual_add",)))
    return tuple(matches)


def classify_semantic_pattern(
    logical_ops: Iterable[LogicalOp],
    *,
    hint: Optional[RepresentativeKind] = None,
) -> Tuple[SemanticKind, Tuple[str, ...]]:
    """Classify a normalized op list and return evidence strings.

    The fixture/adapter hint is deterministic evidence and takes precedence.
    Op-pattern rules remain useful when converting an externally supplied
    ``ExportedProgram`` later, without invoking an LLM or guessing from labels.
    """

    operations, joined = _op_inventory(logical_ops)
    components = detect_semantic_components(operations)
    component_kinds = {kind for kind, _ in components}

    if hint == RepresentativeKind.FULL_ATTENTION:
        evidence = ["representative_kind=full_attention"]
        if SemanticKind.FULL_ATTENTION in component_kinds:
            evidence.append("op_pattern=softmax+matmul>=2")
        return SemanticKind.FULL_ATTENTION, tuple(evidence)
    if hint == RepresentativeKind.LINEAR_STATE:
        evidence = ["representative_kind=linear_state"]
        if SemanticKind.LINEAR_ATTENTION in component_kinds:
            evidence.append("op_pattern=sigmoid+einsum>=2")
        return SemanticKind.LINEAR_ATTENTION, tuple(evidence)
    if hint == RepresentativeKind.DENSE:
        evidence = ["representative_kind=dense"]
        if SemanticKind.FFN in component_kinds:
            evidence.append("op_pattern=linear>=2+activation")
        return SemanticKind.FFN, tuple(evidence)

    for preferred in (
        SemanticKind.FULL_ATTENTION,
        SemanticKind.LINEAR_ATTENTION,
        SemanticKind.FFN,
        SemanticKind.NORM,
    ):
        for kind, evidence in components:
            if kind == preferred:
                return kind, evidence
    return SemanticKind.GENERIC, ("op_pattern=unclassified",)
