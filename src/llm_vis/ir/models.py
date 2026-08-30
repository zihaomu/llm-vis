# ruff: noqa: UP045
"""Pydantic v2 models for the version 0.1 Model Map IR."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

from .ids import scenario_id as make_scenario_id

Number = Union[StrictInt, StrictFloat]


class IRBaseModel(BaseModel):
    """Base class shared by every structured IR object."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"
    TORCH_EXPORT = "torch_export"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class SourceArtifactKind(str, Enum):
    HUGGINGFACE_CONFIG = "huggingface_config"
    PYTHON_SOURCE = "python_source"
    LOCAL_CONFIG = "local_config"
    PARAMETER_INVENTORY = "parameter_inventory"
    EXPORTED_PROGRAM = "exported_program"
    PYTORCH_PROFILER = "pytorch_profiler"
    ROCPROFV3 = "rocprofv3"
    ROCM_COMPUTE_PROFILER = "rocm_compute_profiler"
    BACKEND_IR = "backend_ir"
    GENERATED = "generated"
    OTHER = "other"


class ScenarioPhase(str, Enum):
    PREFILL = "prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    DECODE = "decode"
    TRAIN = "train"


class DType(str, Enum):
    BOOL = "bool"
    UINT8 = "uint8"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT8 = "float8"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    UNKNOWN = "unknown"


class WeightFormat(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT8 = "float8"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    UNKNOWN = "unknown"


class ExpressionOp(str, Enum):
    LITERAL = "literal"
    SYMBOL = "symbol"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    FLOOR_DIVIDE = "floor_divide"
    CEIL_DIVIDE = "ceil_divide"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    NEGATE = "negate"


class DefinitionKind(str, Enum):
    MODEL = "model"
    TOWER = "tower"
    STAGE = "stage"
    BLOCK = "block"
    MODULE = "module"
    EXPERT_POOL = "expert_pool"
    EXPERT = "expert"
    OPERATOR = "operator"
    CONDITIONAL = "conditional"
    OPAQUE = "opaque"


class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class TensorLayout(str, Enum):
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    CHANNELS_LAST = "channels_last"
    PACKED = "packed"
    PAGED = "paged"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class TensorRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    ACTIVATION = "activation"
    CACHE = "cache"
    STATE = "state"
    ROUTE = "route"
    CONTROL = "control"
    UNKNOWN = "unknown"


class TensorOrigin(str, Enum):
    """Evidence boundary for tensor metadata.

    ``CONFIG`` tensors are virtual inventory contracts derived without allocating
    model storage.  ``CAPTURE``/``TRACE`` are reserved for later evidence sources;
    leaving the field unset keeps older IR producers backwards compatible.
    """

    CONFIG = "config"
    CAPTURE = "capture"
    TRACE = "trace"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EdgeKind(str, Enum):
    DATA = "data"
    WEIGHT = "weight"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    CONTROL = "control"
    ROUTE = "route"


class SemanticKind(str, Enum):
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    NORM = "norm"
    ATTENTION = "attention"
    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW_ATTENTION = "sliding_window_attention"
    LINEAR_ATTENTION = "linear_attention"
    MLA = "mla"
    DSA = "dsa"
    ROPE = "rope"
    FFN = "ffn"
    MOE_ROUTER = "moe_router"
    EXPERT_POOL = "expert_pool"
    SHARED_EXPERT = "shared_expert"
    SSM = "ssm"
    RECURRENT_STATE = "recurrent_state"
    KV_CACHE = "kv_cache"
    VISION_TOWER = "vision_tower"
    PROJECTOR = "projector"
    MTP = "mtp"
    RESIDUAL = "residual"
    GENERIC = "generic"
    OPAQUE = "opaque"


class LogicalOpDomain(str, Enum):
    ATEN = "aten"
    PRIMS = "prims"
    TORCH = "torch"
    CUSTOM = "custom"
    PYTHON = "python"
    BACKEND = "backend"
    OPAQUE = "opaque"


class LoweringKind(str, Enum):
    EXPAND = "expand"
    DECOMPOSE = "decompose"
    FUSE = "fuse"
    SPLIT = "split"
    COPY = "copy"
    FALLBACK = "fallback"


class MetricOrigin(str, Enum):
    EXACT = "exact"
    FORMULA = "formula"
    ESTIMATED = "estimated"
    MEASURED = "measured"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class CoverageStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    OPAQUE = "opaque"
    UNMAPPED = "unmapped"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TraceCategory(str, Enum):
    MODULE_RANGE = "module_range"
    OPERATOR = "operator"
    FUSION_GROUP = "fusion_group"
    KERNEL = "kernel"
    MEMCPY = "memcpy"
    SYNCHRONIZATION = "synchronization"
    ALLOCATOR = "allocator"
    OTHER = "other"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SymbolicExpression(IRBaseModel):
    """A validated AST node used for symbolic shapes and formulas."""

    op: ExpressionOp
    value: Optional[Number] = None
    symbol: Optional[str] = Field(default=None, min_length=1)
    args: List[SymbolicExpression] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expression(self) -> SymbolicExpression:
        if self.op == ExpressionOp.LITERAL:
            if self.value is None or self.symbol is not None or self.args:
                raise ValueError("literal requires value and forbids symbol/args")
            return self
        if self.op == ExpressionOp.SYMBOL:
            if self.symbol is None or self.value is not None or self.args:
                raise ValueError("symbol requires symbol and forbids value/args")
            return self
        if self.value is not None or self.symbol is not None:
            raise ValueError("operator expressions forbid value and symbol")
        if self.op == ExpressionOp.NEGATE and len(self.args) != 1:
            raise ValueError("negate requires exactly one argument")
        if (
            self.op
            in {
                ExpressionOp.SUBTRACT,
                ExpressionOp.DIVIDE,
                ExpressionOp.FLOOR_DIVIDE,
                ExpressionOp.CEIL_DIVIDE,
            }
            and len(self.args) != 2
        ):
            raise ValueError(f"{self.op.value} requires exactly two arguments")
        if (
            self.op
            in {
                ExpressionOp.ADD,
                ExpressionOp.MULTIPLY,
                ExpressionOp.MINIMUM,
                ExpressionOp.MAXIMUM,
            }
            and len(self.args) < 2
        ):
            raise ValueError(f"{self.op.value} requires at least two arguments")
        return self

    @classmethod
    def literal(cls, value: Number) -> SymbolicExpression:
        return cls(op=ExpressionOp.LITERAL, value=value)

    @classmethod
    def symbol_ref(cls, symbol: str) -> SymbolicExpression:
        return cls(op=ExpressionOp.SYMBOL, symbol=symbol)


class Model(IRBaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    framework: Framework
    source_artifact_ids: List[str] = Field(default_factory=list)


class SourceArtifact(IRBaseModel):
    id: str = Field(min_length=1)
    kind: SourceArtifactKind
    uri: Optional[str] = Field(default=None, min_length=1)
    path: Optional[str] = Field(default=None, min_length=1)
    revision: Optional[str] = Field(default=None, min_length=1)
    sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: Optional[str] = None
    trusted: bool = False

    @model_validator(mode="after")
    def require_location(self) -> SourceArtifact:
        if self.uri is None and self.path is None:
            raise ValueError("source artifact requires uri or path")
        return self


class Symbol(IRBaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: Optional[str] = None
    unit: Optional[str] = None
    lower_bound: Optional[Number] = None
    upper_bound: Optional[Number] = None
    default_value: Optional[Number] = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Symbol:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound cannot exceed upper_bound")
        if self.default_value is not None:
            if self.lower_bound is not None and self.default_value < self.lower_bound:
                raise ValueError("default_value is below lower_bound")
            if self.upper_bound is not None and self.default_value > self.upper_bound:
                raise ValueError("default_value is above upper_bound")
        return self


class Scenario(IRBaseModel):
    id: str = ""
    phase: ScenarioPhase
    batch: int = Field(ge=1)
    new_tokens: int = Field(ge=1)
    past_tokens: int = Field(ge=0)
    total_visible_length: int = Field(default=0, ge=0)
    output_length: int = Field(default=0, ge=0)
    activation_dtype: DType
    weight_format: WeightFormat
    kv_dtype: DType
    backend: str = Field(default="unknown", min_length=1)
    hardware: str = Field(default="unknown", min_length=1)

    @model_validator(mode="after")
    def validate_and_set_identity(self) -> Scenario:
        expected_visible = self.past_tokens + self.new_tokens
        if self.total_visible_length == 0:
            object.__setattr__(self, "total_visible_length", expected_visible)
        elif self.total_visible_length != expected_visible:
            raise ValueError("total_visible_length must equal past_tokens + new_tokens")

        if self.phase == ScenarioPhase.PREFILL:
            if self.past_tokens != 0 or self.new_tokens <= 1:
                raise ValueError("prefill requires past_tokens=0 and new_tokens>1")
        elif self.phase == ScenarioPhase.CHUNKED_PREFILL:
            if self.past_tokens <= 0 or self.new_tokens <= 1:
                raise ValueError("chunked_prefill requires past_tokens>0 and new_tokens>1")
        elif self.phase == ScenarioPhase.DECODE and self.new_tokens != 1:
            raise ValueError("decode requires new_tokens=1")

        expected_id = make_scenario_id(self.model_dump(mode="json", exclude={"id"}))
        if self.id and self.id != expected_id:
            raise ValueError("scenario id does not match its canonical payload")
        object.__setattr__(self, "id", expected_id)
        return self


class Port(IRBaseModel):
    name: str = Field(min_length=1)
    direction: PortDirection
    tensor_role: TensorRole = TensorRole.UNKNOWN


class Definition(IRBaseModel):
    id: str = Field(min_length=1)
    kind: DefinitionKind
    label: str = Field(min_length=1)
    class_name: Optional[str] = None
    ports: List[Port] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    semantic_pattern: Optional[str] = None


class Instance(IRBaseModel):
    id: str = Field(min_length=1)
    definition_id: str = Field(min_length=1)
    parent_id: Optional[str] = Field(default=None, min_length=1)
    module_path: str = Field(min_length=1)
    layer_index: Optional[int] = Field(default=None, ge=0)
    overrides: Dict[str, Any] = Field(default_factory=dict)


class TensorSpec(IRBaseModel):
    id: str = Field(min_length=1)
    semantic_name: Optional[str] = Field(default=None, min_length=1)
    owner_id: Optional[str] = Field(default=None, min_length=1)
    symbolic_shape: List[SymbolicExpression]
    concrete_shape: Optional[List[int]] = None
    shape_known: bool = True
    dtype: DType
    storage_dtype: DType
    layout: TensorLayout = TensorLayout.UNKNOWN
    stride: Optional[List[int]] = None
    device: Optional[str] = None
    role: TensorRole = TensorRole.UNKNOWN
    alias_group: Optional[str] = None
    origin: Optional[TensorOrigin] = None
    materialized: Optional[bool] = None
    source_artifact_ids: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape_metadata(self) -> TensorSpec:
        rank = len(self.symbolic_shape)
        if not self.shape_known:
            if self.symbolic_shape or self.concrete_shape is not None or self.stride is not None:
                raise ValueError(
                    "shape_known=false requires empty symbolic_shape and forbids "
                    "concrete_shape/stride"
                )
        if self.concrete_shape is not None:
            if any(dimension < 0 for dimension in self.concrete_shape):
                raise ValueError("concrete shape dimensions must be non-negative")
            if len(self.concrete_shape) != rank:
                raise ValueError("concrete_shape rank must match symbolic_shape")
        if self.stride is not None and len(self.stride) != rank:
            raise ValueError("stride rank must match symbolic_shape")
        if self.origin == TensorOrigin.CONFIG:
            if self.semantic_name is None or self.owner_id is None:
                raise ValueError("config-origin tensor requires semantic_name and owner_id")
            if self.materialized is not False:
                raise ValueError("config-origin tensor must declare materialized=false")
            if not self.source_artifact_ids or not self.evidence:
                raise ValueError("config-origin tensor requires source artifacts and evidence")
        if self.origin == TensorOrigin.CAPTURE:
            if self.materialized is not False:
                raise ValueError("capture-origin tensor must declare materialized=false")
            if len(self.source_artifact_ids) != 1 or not self.evidence:
                raise ValueError(
                    "capture-origin tensor requires exactly one source artifact and evidence"
                )
        return self


class Edge(IRBaseModel):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    tensor_id: Optional[str] = Field(default=None, min_length=1)
    kind: EdgeKind


class SemanticNode(IRBaseModel):
    id: str = Field(min_length=1)
    kind: SemanticKind
    label: str = Field(min_length=1)
    definition_id: Optional[str] = Field(default=None, min_length=1)
    instance_ids: List[str] = Field(default_factory=list)
    input_tensor_ids: List[str] = Field(default_factory=list)
    output_tensor_ids: List[str] = Field(default_factory=list)
    child_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    opaque: bool = False


class LogicalOp(IRBaseModel):
    id: str = Field(min_length=1)
    domain: LogicalOpDomain
    op_type: str = Field(min_length=1)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    scope_instance_id: Optional[str] = Field(default=None, min_length=1)


class Lowering(IRBaseModel):
    id: str = Field(min_length=1)
    source_ids: List[str] = Field(min_length=1)
    target_ids: List[str] = Field(min_length=1)
    kind: LoweringKind


class Metric(IRBaseModel):
    id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    scenario_id: Optional[str] = Field(default=None, min_length=1)
    name: str = Field(min_length=1)
    value: Optional[Number] = None
    unit: str = Field(min_length=1)
    origin: MetricOrigin
    formula: Optional[SymbolicExpression] = None
    assumptions: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    coverage_status: CoverageStatus = CoverageStatus.COMPLETE

    @model_validator(mode="after")
    def validate_metric_value(self) -> Metric:
        if self.origin == MetricOrigin.UNKNOWN:
            if self.value is not None:
                raise ValueError("unknown metrics must have value=None")
        elif self.value is None:
            raise ValueError("non-unknown metrics require a numeric value")
        return self


class TraceEvent(IRBaseModel):
    id: str = Field(min_length=1)
    parent_id: Optional[str] = Field(default=None, min_length=1)
    correlation_id: Optional[str] = None
    category: TraceCategory
    start: float = Field(ge=0.0)
    duration: float = Field(ge=0.0)
    device: Optional[str] = None
    stream: Optional[str] = None
    kernel_name: Optional[str] = None
    counters: Dict[str, Number] = Field(default_factory=dict)


class Diagnostic(IRBaseModel):
    id: str = Field(min_length=1)
    severity: DiagnosticSeverity
    code: str = Field(min_length=1)
    subject_id: Optional[str] = Field(default=None, min_length=1)
    message: str = Field(min_length=1)
    evidence: List[str] = Field(default_factory=list)


class TraceDocument(IRBaseModel):
    """Standalone trace artifact represented by ``trace.schema.json``."""

    schema_version: Literal["0.1"] = "0.1"
    events: List[TraceEvent] = Field(default_factory=list)
    diagnostics: List[Diagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_event_tree(self) -> TraceDocument:
        event_ids = {event.id for event in self.events}
        if len(event_ids) != len(self.events):
            raise ValueError("trace event ids must be unique")
        for event in self.events:
            if event.parent_id is not None and event.parent_id not in event_ids:
                raise ValueError(f"trace parent {event.parent_id!r} does not exist")
            if event.parent_id == event.id:
                raise ValueError("trace event cannot be its own parent")
        return self


class ModelMap(IRBaseModel):
    schema_version: Literal["0.1"] = "0.1"
    model: Model
    source_artifacts: List[SourceArtifact] = Field(default_factory=list)
    symbols: List[Symbol] = Field(default_factory=list)
    scenarios: List[Scenario] = Field(default_factory=list)
    definitions: List[Definition] = Field(default_factory=list)
    instances: List[Instance] = Field(default_factory=list)
    tensors: List[TensorSpec] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    semantic_nodes: List[SemanticNode] = Field(default_factory=list)
    logical_ops: List[LogicalOp] = Field(default_factory=list)
    lowerings: List[Lowering] = Field(default_factory=list)
    metrics: List[Metric] = Field(default_factory=list)
    trace_events: List[TraceEvent] = Field(default_factory=list)
    diagnostics: List[Diagnostic] = Field(default_factory=list)

    @staticmethod
    def _ids(items: List[Any], collection_name: str) -> Set[str]:
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{collection_name} ids must be unique")
        return set(ids)

    @model_validator(mode="after")
    def validate_references(self) -> ModelMap:
        collections: List[Tuple[str, List[Any]]] = [
            ("source_artifacts", self.source_artifacts),
            ("symbols", self.symbols),
            ("scenarios", self.scenarios),
            ("definitions", self.definitions),
            ("instances", self.instances),
            ("tensors", self.tensors),
            ("edges", self.edges),
            ("semantic_nodes", self.semantic_nodes),
            ("logical_ops", self.logical_ops),
            ("lowerings", self.lowerings),
            ("metrics", self.metrics),
            ("trace_events", self.trace_events),
            ("diagnostics", self.diagnostics),
        ]
        ids_by_collection = {name: self._ids(items, name) for name, items in collections}

        global_ids: Set[str] = {self.model.id}
        for name, item_ids in ids_by_collection.items():
            overlap = global_ids.intersection(item_ids)
            if overlap:
                duplicate = sorted(overlap)[0]
                raise ValueError(f"id {duplicate!r} is reused in {name}")
            global_ids.update(item_ids)

        artifact_ids = ids_by_collection["source_artifacts"]
        for artifact_id in self.model.source_artifact_ids:
            if artifact_id not in artifact_ids:
                raise ValueError(f"model references missing source artifact {artifact_id!r}")

        definition_ids = ids_by_collection["definitions"]
        for definition in self.definitions:
            for child_id in definition.children:
                if child_id not in definition_ids:
                    raise ValueError(f"definition child {child_id!r} does not exist")

        instance_ids = ids_by_collection["instances"]
        for instance in self.instances:
            if instance.definition_id not in definition_ids:
                raise ValueError(f"instance definition {instance.definition_id!r} does not exist")
            if instance.parent_id is not None and instance.parent_id not in instance_ids:
                raise ValueError(f"instance parent {instance.parent_id!r} does not exist")
            if instance.parent_id == instance.id:
                raise ValueError("instance cannot be its own parent")

        tensor_ids = ids_by_collection["tensors"]
        semantic_ids = ids_by_collection["semantic_nodes"]
        logical_ids = ids_by_collection["logical_ops"]
        graph_subject_ids = definition_ids | instance_ids | semantic_ids | logical_ids

        tensor_owner_ids = graph_subject_ids | {self.model.id}
        for tensor in self.tensors:
            if tensor.owner_id is not None and tensor.owner_id not in tensor_owner_ids:
                raise ValueError(f"tensor owner {tensor.owner_id!r} does not exist")
            missing_artifacts = set(tensor.source_artifact_ids) - artifact_ids
            if missing_artifacts:
                missing_artifact = sorted(missing_artifacts)[0]
                raise ValueError(f"tensor source artifact {missing_artifact!r} does not exist")

        for edge in self.edges:
            if edge.source_id not in graph_subject_ids:
                raise ValueError(f"edge source {edge.source_id!r} does not exist")
            if edge.target_id not in graph_subject_ids:
                raise ValueError(f"edge target {edge.target_id!r} does not exist")
            if edge.tensor_id is not None and edge.tensor_id not in tensor_ids:
                raise ValueError(f"edge tensor {edge.tensor_id!r} does not exist")

        for node in self.semantic_nodes:
            if node.definition_id is not None and node.definition_id not in definition_ids:
                raise ValueError(f"semantic definition {node.definition_id!r} does not exist")
            missing_instances = set(node.instance_ids) - instance_ids
            missing_tensors = (
                set(node.input_tensor_ids) | set(node.output_tensor_ids)
            ) - tensor_ids
            missing_children = set(node.child_ids) - semantic_ids
            if missing_instances:
                missing_instance = sorted(missing_instances)[0]
                raise ValueError(f"semantic instance {missing_instance!r} does not exist")
            if missing_tensors:
                raise ValueError(f"semantic tensor {sorted(missing_tensors)[0]!r} does not exist")
            if missing_children:
                raise ValueError(f"semantic child {sorted(missing_children)[0]!r} does not exist")

        for op in self.logical_ops:
            missing_tensors = (set(op.inputs) | set(op.outputs)) - tensor_ids
            if missing_tensors:
                raise ValueError(f"logical tensor {sorted(missing_tensors)[0]!r} does not exist")
            if op.scope_instance_id is not None and op.scope_instance_id not in instance_ids:
                raise ValueError(f"logical scope {op.scope_instance_id!r} does not exist")

        lowering_subject_ids = semantic_ids | logical_ids
        for lowering in self.lowerings:
            missing = (set(lowering.source_ids) | set(lowering.target_ids)) - lowering_subject_ids
            if missing:
                raise ValueError(f"lowering subject {sorted(missing)[0]!r} does not exist")

        scenario_ids = ids_by_collection["scenarios"]
        metric_subject_ids = global_ids - ids_by_collection["metrics"]
        for metric in self.metrics:
            if metric.subject_id not in metric_subject_ids:
                raise ValueError(f"metric subject {metric.subject_id!r} does not exist")
            if metric.scenario_id is not None and metric.scenario_id not in scenario_ids:
                raise ValueError(f"metric scenario {metric.scenario_id!r} does not exist")

        trace_ids = ids_by_collection["trace_events"]
        for event in self.trace_events:
            if event.parent_id is not None and event.parent_id not in trace_ids:
                raise ValueError(f"trace parent {event.parent_id!r} does not exist")
            if event.parent_id == event.id:
                raise ValueError("trace event cannot be its own parent")

        diagnostic_subjects = global_ids - ids_by_collection["diagnostics"]
        for diagnostic in self.diagnostics:
            if (
                diagnostic.subject_id is not None
                and diagnostic.subject_id not in diagnostic_subjects
            ):
                raise ValueError(f"diagnostic subject {diagnostic.subject_id!r} does not exist")
        return self


SymbolicExpression.model_rebuild()
