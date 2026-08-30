"""Public data types for representative-block capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from llm_vis.ir import Diagnostic, LogicalOp, SemanticNode, TensorSpec


class RepresentativeKind(str, Enum):
    DENSE = "dense"
    FULL_ATTENTION = "full_attention"
    LINEAR_STATE = "linear_state"


class CaptureStatus(str, Enum):
    CAPTURED = "captured"
    PARTIAL = "partial"
    OPAQUE = "opaque"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RepresentativeCandidate:
    """A block instance from which one representative may be selected."""

    kind: RepresentativeKind
    instance_id: str
    module_path: str
    definition_id: Optional[str] = None
    layer_index: Optional[int] = None
    supported: bool = True


@dataclass(frozen=True)
class RepresentativeSelection:
    kind: RepresentativeKind
    instance_id: str
    module_path: str
    definition_id: Optional[str] = None
    layer_index: Optional[int] = None


@dataclass(frozen=True)
class InputContract:
    name: str
    shape: Tuple[int, ...]
    role: str = "input"


@dataclass(frozen=True)
class TinyRepresentativeSpec:
    """Static contract for a project-owned, deliberately tiny block."""

    kind: RepresentativeKind
    name: str
    module_path: str
    hidden_size: int
    inputs: Tuple[InputContract, ...]


@dataclass(frozen=True)
class BuiltTinyRepresentative:
    """A meta-only module and inputs ready for ``torch.export``."""

    spec: TinyRepresentativeSpec
    module: Any = field(repr=False)
    args: Tuple[Any, ...] = field(repr=False)


@dataclass(frozen=True)
class ExportCaptureResult:
    """IR produced from one representative ``ExportedProgram``.

    ``exported_program`` is intentionally excluded from serialization concerns;
    callers may use it during the current process, while durable artifacts use
    the normalized IR fields.
    """

    kind: RepresentativeKind
    status: CaptureStatus
    capture_id: str
    coverage: float
    tensors: Tuple[TensorSpec, ...] = ()
    logical_ops: Tuple[LogicalOp, ...] = ()
    semantic_nodes: Tuple[SemanticNode, ...] = ()
    diagnostics: Tuple[Diagnostic, ...] = ()
    exported_program: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be between 0 and 1")
        if self.status in {CaptureStatus.OPAQUE, CaptureStatus.UNAVAILABLE}:
            if self.coverage != 0.0:
                raise ValueError("opaque/unavailable capture must have zero coverage")
            if not any(node.opaque for node in self.semantic_nodes):
                raise ValueError("opaque/unavailable capture requires an opaque semantic node")


class UnsupportedCaptureError(RuntimeError):
    """Raised by a controlled fixture when export support is intentionally absent."""
