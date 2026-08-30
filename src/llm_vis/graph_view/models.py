"""Renderer-neutral graph-view contracts for the interactive DAG canvas.

The graph view is a projection of Model Map evidence.  It deliberately does not
claim to be an executed framework graph: config-origin activation interfaces are
synthetic display contracts, while state/cache ports may bind existing TensorSpec
records when config inventory proves them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_vis.ir.models import DType, PortDirection, TensorOrigin, TensorRole

ShapeDimension = Union[int, str]


class GraphViewBaseModel(BaseModel):
    """Strict base class shared by GraphView artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class GraphViewLevel(str, Enum):
    """Evidence views; operator decomposition is depth-based, not an audience mode."""

    MODEL = "L0"
    BLOCK = "L1"
    OPERATOR = "operator"


class GraphDecompositionStatus(str, Enum):
    """Whether a semantic node can be recursively disclosed."""

    NONE = "none"
    AVAILABLE = "available"
    PRIMITIVE = "primitive"
    OPAQUE = "opaque"


class GraphPrimitiveKind(str, Enum):
    """Frozen v1 semantic primitive ontology (not a GPU/kernel ontology)."""

    GEMM = "gemm"
    MATMUL = "matmul"
    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"
    SOFTMAX = "softmax"
    TOP_K = "top_k"
    SILU = "silu"
    GELU = "gelu"
    ADD = "add"
    MULTIPLY = "multiply"
    SCALE = "scale"
    MASK = "mask"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    BROADCAST = "broadcast"
    SPLIT = "split"
    CONCAT = "concat"
    ROPE = "rope"
    GATHER = "gather"
    SCATTER = "scatter"
    REDUCE = "reduce"
    CONVOLUTION = "convolution"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    OPAQUE = "opaque"


class GraphBoundaryKind(str, Enum):
    """How one compound-node port maps onto its decomposition boundary."""

    INPUT = "input"
    OUTPUT = "output"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"


class GraphMetricBindingStatus(str, Enum):
    """Whether a primitive owns a metric, is excluded, or remains unknown."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    EXCLUDED = "excluded"


class GraphCostReconciliationStatus(str, Enum):
    """Static reconciliation status for one parent metric decomposition."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class GraphEdgeKind(str, Enum):
    """Renderer-visible edge semantics."""

    DATA = "data"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    ROUTE = "route"
    CONTROL = "control"


class GraphPort(GraphViewBaseModel):
    """One stable input or output handle displayed inside a graph node."""

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    direction: PortDirection
    role: TensorRole
    shape: Optional[List[ShapeDimension]] = None
    shape_known: bool = True
    dtype: DType = DType.UNKNOWN
    tensor_spec_id: Optional[str] = Field(default=None, min_length=1)
    evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape_and_direction(self) -> GraphPort:
        if self.direction == PortDirection.INOUT:
            raise ValueError("graph-view ports must be distinct input or output handles")
        if self.shape_known and self.shape is None:
            raise ValueError("shape_known=true requires shape")
        if not self.shape_known and self.shape is not None:
            raise ValueError("shape_known=false forbids shape")
        if self.shape is not None:
            for dimension in self.shape:
                if isinstance(dimension, int) and dimension < 0:
                    raise ValueError("shape dimensions must be non-negative")
                if isinstance(dimension, str) and not dimension:
                    raise ValueError("symbolic shape dimensions must be non-empty")
        return self


class GraphNode(GraphViewBaseModel):
    """A semantic node with renderer-ready Inspector and navigation evidence."""

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    group_id: Optional[str] = Field(default=None, min_length=1)
    input_port_ids: List[str] = Field(default_factory=list)
    output_port_ids: List[str] = Field(default_factory=list)
    subject_ids: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    opaque: bool = False
    drilldown_view_id: Optional[str] = Field(default=None, min_length=1)
    decomposition_status: GraphDecompositionStatus = GraphDecompositionStatus.NONE
    primitive_kind: Optional[GraphPrimitiveKind] = None
    metric_bindings: List[GraphMetricBinding] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decomposition_state(self) -> GraphNode:
        dimensions = [binding.dimension for binding in self.metric_bindings]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("node metric bindings must use unique dimensions")
        if self.decomposition_status == GraphDecompositionStatus.AVAILABLE:
            if self.drilldown_view_id is None:
                raise ValueError("available decomposition requires drilldown_view_id")
            if self.opaque:
                raise ValueError("available decomposition requires opaque=false")
            if self.primitive_kind is not None:
                raise ValueError("expandable compound node cannot declare primitive_kind")
        elif self.decomposition_status == GraphDecompositionStatus.PRIMITIVE:
            if self.primitive_kind is None:
                raise ValueError("primitive decomposition status requires primitive_kind")
            if self.opaque:
                raise ValueError("primitive decomposition status requires opaque=false")
            if self.drilldown_view_id is not None:
                raise ValueError("primitive node cannot have a drilldown view")
        elif self.decomposition_status == GraphDecompositionStatus.OPAQUE:
            if not self.opaque:
                raise ValueError("opaque decomposition status requires opaque=true")
            if self.primitive_kind is not None:
                raise ValueError("opaque node cannot declare primitive_kind")
            if self.drilldown_view_id is not None:
                raise ValueError("opaque node cannot have a drilldown view")
        else:
            if self.opaque:
                raise ValueError("none decomposition status requires opaque=false")
            if self.primitive_kind is not None:
                raise ValueError("none decomposition status cannot declare primitive_kind")
            if self.drilldown_view_id is not None:
                raise ValueError("none decomposition status cannot have a drilldown view")
        return self


class GraphMetricBinding(GraphViewBaseModel):
    """Exact node-to-Metric name binding used by cost Inspector and heat overlays."""

    dimension: Literal["flops", "logical_bytes"]
    parent_metric_name: str = Field(min_length=1)
    status: GraphMetricBindingStatus
    metric_name: Optional[str] = Field(default=None, min_length=1)
    reason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_status_payload(self) -> GraphMetricBinding:
        if self.status == GraphMetricBindingStatus.KNOWN and self.metric_name is None:
            raise ValueError("known metric binding requires metric_name")
        if self.status != GraphMetricBindingStatus.KNOWN and self.reason is None:
            raise ValueError("unknown/excluded metric binding requires a reason")
        if self.status != GraphMetricBindingStatus.KNOWN and self.metric_name is not None:
            raise ValueError("unknown/excluded metric binding cannot name a Metric")
        return self


class GraphBoundaryBinding(GraphViewBaseModel):
    """One lossless parent-port to child-boundary-port mapping."""

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    parent_port_id: str = Field(min_length=1)
    child_port_id: str = Field(min_length=1)
    kind: GraphBoundaryKind
    evidence: List[str] = Field(min_length=1)


class GraphCostReconciliation(GraphViewBaseModel):
    """Static formula-scope accounting for one decomposed parent metric."""

    dimension: Literal["flops", "logical_bytes"]
    parent_metric_name: str = Field(min_length=1)
    known_child_metric_names: List[str] = Field(default_factory=list)
    unattributed_node_ids: List[str] = Field(default_factory=list)
    excluded_node_ids: List[str] = Field(default_factory=list)
    status: GraphCostReconciliationStatus
    reason: str = Field(min_length=1)
    evidence: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_accounting_sets(self) -> GraphCostReconciliation:
        for label, values in (
            ("known_child_metric_names", self.known_child_metric_names),
            ("unattributed_node_ids", self.unattributed_node_ids),
            ("excluded_node_ids", self.excluded_node_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if set(self.unattributed_node_ids) & set(self.excluded_node_ids):
            raise ValueError("a reconciliation node cannot be both unattributed and excluded")
        if self.status == GraphCostReconciliationStatus.COMPLETE and self.unattributed_node_ids:
            raise ValueError("complete reconciliation cannot retain unattributed nodes")
        if (
            self.status == GraphCostReconciliationStatus.COMPLETE
            and not self.known_child_metric_names
        ):
            raise ValueError("complete reconciliation requires known child metrics")
        if self.status == GraphCostReconciliationStatus.PARTIAL and not self.unattributed_node_ids:
            raise ValueError("partial reconciliation requires unattributed nodes")
        if self.status == GraphCostReconciliationStatus.UNKNOWN:
            if self.known_child_metric_names:
                raise ValueError("unknown reconciliation cannot retain known child metrics")
            if not self.unattributed_node_ids:
                raise ValueError("unknown reconciliation requires unattributed nodes")
        return self


class GraphEdge(GraphViewBaseModel):
    """A directed connection between two GraphPort handles."""

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    source_port_id: str = Field(min_length=1)
    target_port_id: str = Field(min_length=1)
    kind: GraphEdgeKind
    tensor_spec_id: Optional[str] = Field(default=None, min_length=1)
    origin: TensorOrigin
    evidence: List[str] = Field(min_length=1)
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphGroup(GraphViewBaseModel):
    """A renderer-neutral compound-node or layout group."""

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    member_node_ids: List[str] = Field(default_factory=list)
    parent_group_id: Optional[str] = Field(default=None, min_length=1)
    collapsed: bool = False
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphView(GraphViewBaseModel):
    """One independently laid-out semantic zoom view."""

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    level: GraphViewLevel
    label: str = Field(min_length=1)
    parent_view_id: Optional[str] = Field(default=None, min_length=1)
    decomposes_node_id: Optional[str] = Field(default=None, min_length=1)
    breadcrumb: List[str] = Field(default_factory=list)
    layer_index: Optional[int] = Field(default=None, ge=0)
    root_group_id: str = Field(min_length=1)
    nodes: List[GraphNode] = Field(default_factory=list)
    ports: List[GraphPort] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    groups: List[GraphGroup] = Field(default_factory=list)
    boundary_bindings: List[GraphBoundaryBinding] = Field(default_factory=list)
    cost_frontier_node_ids: List[str] = Field(default_factory=list)
    cost_reconciliations: List[GraphCostReconciliation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def _unique_ids(items: List[Any], collection_name: str) -> Set[str]:
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{collection_name} ids must be unique")
        return set(ids)

    @model_validator(mode="after")
    def validate_graph(self) -> GraphView:
        node_ids = self._unique_ids(self.nodes, "graph nodes")
        port_ids = self._unique_ids(self.ports, "graph ports")
        self._unique_ids(self.edges, "graph edges")
        group_ids = self._unique_ids(self.groups, "graph groups")
        self._unique_ids(self.boundary_bindings, "graph boundary bindings")

        for collection_name, items in (
            ("graph nodes", self.nodes),
            ("graph ports", self.ports),
            ("graph edges", self.edges),
            ("graph groups", self.groups),
        ):
            keys = [item.key for item in items]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{collection_name} keys must be unique")

        if self.root_group_id not in group_ids:
            raise ValueError("root_group_id must reference a graph group")
        if self.decomposes_node_id is not None and self.parent_view_id is None:
            raise ValueError("decomposition view requires parent_view_id")
        if len(self.cost_frontier_node_ids) != len(set(self.cost_frontier_node_ids)):
            raise ValueError("cost_frontier_node_ids must be unique")
        if set(self.cost_frontier_node_ids) - node_ids:
            raise ValueError("cost_frontier_node_ids must reference nodes in this view")
        reconciliation_dimensions = [
            reconciliation.dimension for reconciliation in self.cost_reconciliations
        ]
        if len(reconciliation_dimensions) != len(set(reconciliation_dimensions)):
            raise ValueError("cost reconciliations must use unique dimensions")
        if self.cost_reconciliations and not self.cost_frontier_node_ids:
            raise ValueError("cost reconciliation requires a non-empty cost frontier")

        ports_by_id = {port.id: port for port in self.ports}
        nodes_by_id = {node.id: node for node in self.nodes}
        if any(nodes_by_id[node_id].kind == "boundary" for node_id in self.cost_frontier_node_ids):
            raise ValueError("boundary nodes cannot belong to the cost frontier")
        groups_by_id = {group.id: group for group in self.groups}
        claimed_ports: Set[str] = set()
        for node in self.nodes:
            declared = node.input_port_ids + node.output_port_ids
            if len(declared) != len(set(declared)):
                raise ValueError(f"node {node.key!r} repeats a port reference")
            missing = set(declared) - port_ids
            if missing:
                raise ValueError(f"node {node.key!r} references missing ports")
            for port_id in node.input_port_ids:
                port = ports_by_id[port_id]
                if port.node_id != node.id or port.direction != PortDirection.INPUT:
                    raise ValueError("node input_port_ids must reference its input ports")
            for port_id in node.output_port_ids:
                port = ports_by_id[port_id]
                if port.node_id != node.id or port.direction != PortDirection.OUTPUT:
                    raise ValueError("node output_port_ids must reference its output ports")
            claimed_ports.update(declared)
            if node.group_id is not None:
                if node.group_id not in group_ids:
                    raise ValueError(f"node {node.key!r} references a missing group")
                if node.id not in groups_by_id[node.group_id].member_node_ids:
                    raise ValueError("node group_id and group membership disagree")
        if claimed_ports != port_ids:
            raise ValueError("every graph port must be owned by exactly one graph node")

        claimed_nodes: Set[str] = set()
        for group in self.groups:
            if len(group.member_node_ids) != len(set(group.member_node_ids)):
                raise ValueError(f"group {group.key!r} repeats a node reference")
            if set(group.member_node_ids) - node_ids:
                raise ValueError(f"group {group.key!r} references missing nodes")
            if claimed_nodes.intersection(group.member_node_ids):
                raise ValueError("a graph node cannot belong to multiple groups")
            claimed_nodes.update(group.member_node_ids)
            if group.parent_group_id is not None:
                if group.parent_group_id not in group_ids:
                    raise ValueError(f"group {group.key!r} references a missing parent")
                if group.parent_group_id == group.id:
                    raise ValueError("graph group cannot be its own parent")

        primary_adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in node_ids}
        producer_by_input: Dict[str, str] = {}
        for edge in self.edges:
            if edge.source_port_id not in port_ids or edge.target_port_id not in port_ids:
                raise ValueError(f"edge {edge.key!r} references a missing port")
            source = ports_by_id[edge.source_port_id]
            target = ports_by_id[edge.target_port_id]
            if source.direction != PortDirection.OUTPUT:
                raise ValueError("edge source must be an output port")
            if target.direction != PortDirection.INPUT:
                raise ValueError("edge target must be an input port")
            if source.shape_known != target.shape_known or source.shape != target.shape:
                raise ValueError("edge endpoints must preserve Tensor shape")
            if source.dtype != target.dtype:
                raise ValueError("edge endpoints must preserve dtype")
            previous_producer = producer_by_input.get(edge.target_port_id)
            if previous_producer is not None:
                raise ValueError(
                    "graph input ports must have at most one producer; "
                    f"edges {previous_producer!r} and {edge.key!r} share a target"
                )
            producer_by_input[edge.target_port_id] = edge.key
            if edge.tensor_spec_id is not None and (
                source.tensor_spec_id != edge.tensor_spec_id
                or target.tensor_spec_id != edge.tensor_spec_id
            ):
                raise ValueError("bound edge and endpoint tensor_spec_id values must agree")
            if edge.kind in {
                GraphEdgeKind.DATA,
                GraphEdgeKind.ROUTE,
                GraphEdgeKind.CONTROL,
            }:
                if source.node_id == target.node_id:
                    raise ValueError("primary-flow edges cannot be self loops")
                primary_adjacency[source.node_id].add(target.node_id)

        self._validate_primary_flow_is_acyclic(primary_adjacency, nodes_by_id)

        parent_port_ids = [binding.parent_port_id for binding in self.boundary_bindings]
        child_port_ids = [binding.child_port_id for binding in self.boundary_bindings]
        if len(parent_port_ids) != len(set(parent_port_ids)):
            raise ValueError("decomposition parent ports must be bound exactly once")
        if len(child_port_ids) != len(set(child_port_ids)):
            raise ValueError("decomposition child boundary ports must be bound exactly once")
        if set(child_port_ids) - port_ids:
            raise ValueError("boundary binding child_port_id must belong to this view")
        for reconciliation in self.cost_reconciliations:
            referenced = set(reconciliation.unattributed_node_ids) | set(
                reconciliation.excluded_node_ids
            )
            if referenced - node_ids:
                raise ValueError("cost reconciliation must reference nodes in this view")
            frontier_nodes = [nodes_by_id[node_id] for node_id in self.cost_frontier_node_ids]
            bindings = {
                node.id: next(
                    (
                        binding
                        for binding in node.metric_bindings
                        if binding.dimension == reconciliation.dimension
                    ),
                    None,
                )
                for node in frontier_nodes
            }
            if any(binding is None for binding in bindings.values()):
                raise ValueError("every reconciled frontier node requires a metric binding")
            if any(
                binding is not None
                and binding.parent_metric_name != reconciliation.parent_metric_name
                for binding in bindings.values()
            ):
                raise ValueError("frontier binding parent metric must match cost reconciliation")
            known_names = [
                binding.metric_name
                for binding in bindings.values()
                if binding is not None and binding.status == GraphMetricBindingStatus.KNOWN
            ]
            if len(known_names) != len(set(known_names)):
                raise ValueError("known child metric names must have one frontier owner")
            if set(known_names) != set(reconciliation.known_child_metric_names):
                raise ValueError("reconciliation known metrics must match frontier bindings")
            unknown_ids = {
                node_id
                for node_id, binding in bindings.items()
                if binding is not None and binding.status == GraphMetricBindingStatus.UNKNOWN
            }
            if unknown_ids != set(reconciliation.unattributed_node_ids):
                raise ValueError("reconciliation unattributed nodes must match unknown bindings")
            excluded_ids = {
                node_id
                for node_id, binding in bindings.items()
                if binding is not None and binding.status == GraphMetricBindingStatus.EXCLUDED
            }
            if excluded_ids != set(reconciliation.excluded_node_ids):
                raise ValueError("reconciliation excluded nodes must match excluded bindings")
        return self

    @staticmethod
    def _validate_primary_flow_is_acyclic(
        adjacency: Dict[str, Set[str]], nodes_by_id: Dict[str, GraphNode]
    ) -> None:
        indegree = {node_id: 0 for node_id in adjacency}
        for targets in adjacency.values():
            for target_id in targets:
                indegree[target_id] += 1
        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            node_id = ready.pop(0)
            visited += 1
            for target_id in sorted(adjacency[node_id]):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    ready.append(target_id)
                    ready.sort()
        if visited != len(nodes_by_id):
            raise ValueError("data/route/control graph must be acyclic")


class GraphViewProvenance(GraphViewBaseModel):
    """Explicit zero-execution boundary carried with every GraphView document."""

    projection: Literal["config-first-graph-view"] = "config-first-graph-view"
    projection_version: Literal["0.1"] = "0.1"
    source_artifact_ids: List[str] = Field(default_factory=list)
    weights_loaded: Literal[False] = False
    target_model_constructed: Literal[False] = False
    target_model_forward: Literal[False] = False
    remote_code_executed: Literal[False] = False
    evidence: List[str] = Field(min_length=1)


class GraphViewDocument(GraphViewBaseModel):
    """Top-level renderer-neutral artifact containing L0/L1 and operator views."""

    schema_version: Literal["0.1"] = "0.1"
    id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    source_model_map_id: str = Field(min_length=1)
    provenance: GraphViewProvenance
    views: List[GraphView] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_view_navigation(self) -> GraphViewDocument:
        view_ids = [view.id for view in self.views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("graph view ids must be unique")
        view_keys = [view.key for view in self.views]
        if len(view_keys) != len(set(view_keys)):
            raise ValueError("graph view keys must be unique")
        known = set(view_ids)
        views_by_id = {view.id: view for view in self.views}
        for view in self.views:
            if view.level == GraphViewLevel.OPERATOR and view.decomposes_node_id is None:
                raise ValueError("operator view must identify the decomposed source node")
            if view.parent_view_id is not None:
                if view.parent_view_id not in known:
                    raise ValueError(f"view {view.key!r} references a missing parent")
                if view.parent_view_id == view.id:
                    raise ValueError("graph view cannot be its own parent")
            for node in view.nodes:
                if node.drilldown_view_id is not None:
                    if node.drilldown_view_id not in known:
                        raise ValueError(f"node {node.key!r} references a missing drilldown view")
                    if node.drilldown_view_id == view.id:
                        raise ValueError("graph node cannot drill down to its containing view")
                    target = views_by_id[node.drilldown_view_id]
                    if target.parent_view_id != view.id:
                        raise ValueError("drilldown target must be a direct child view")
                    if (
                        target.level == GraphViewLevel.OPERATOR
                        and target.decomposes_node_id != node.id
                    ):
                        raise ValueError(
                            "operator decomposition target must identify its source node"
                        )
                    if (
                        target.decomposes_node_id is not None
                        and target.decomposes_node_id != node.id
                    ):
                        raise ValueError("decomposition target must identify its source node")
            if view.decomposes_node_id is not None:
                parent = views_by_id[view.parent_view_id or ""]
                parent_nodes = {node.id: node for node in parent.nodes}
                parent_node = parent_nodes.get(view.decomposes_node_id)
                if parent_node is None:
                    raise ValueError("decomposes_node_id must belong to parent view")
                if parent_node.drilldown_view_id != view.id:
                    raise ValueError("parent node drilldown must target its decomposition view")
                self._validate_boundary_contract(parent, parent_node, view)
        self._validate_view_hierarchy_is_acyclic(views_by_id)
        return self

    @staticmethod
    def _validate_boundary_contract(
        parent: GraphView, parent_node: GraphNode, child: GraphView
    ) -> None:
        parent_ports = {port.id: port for port in parent.ports}
        child_ports = {port.id: port for port in child.ports}
        child_nodes = {node.id: node for node in child.nodes}
        expected_parent_ports = set(parent_node.input_port_ids + parent_node.output_port_ids)
        mapped_parent_ports = {binding.parent_port_id for binding in child.boundary_bindings}
        if mapped_parent_ports != expected_parent_ports:
            raise ValueError("decomposition must bind every parent node port exactly once")
        state_reads_by_tensor: Dict[str, GraphBoundaryBinding] = {}
        state_writes_by_tensor: Dict[str, GraphBoundaryBinding] = {}
        for binding in child.boundary_bindings:
            parent_port = parent_ports.get(binding.parent_port_id)
            child_port = child_ports.get(binding.child_port_id)
            if parent_port is None or child_port is None:
                raise ValueError("boundary binding references a missing parent or child port")
            if child_nodes[child_port.node_id].kind != "boundary":
                raise ValueError("boundary binding child port must belong to a boundary node")
            if parent_port.role != child_port.role:
                raise ValueError("boundary binding must preserve Tensor role")
            if parent_port.dtype != child_port.dtype:
                raise ValueError("boundary binding must preserve dtype")
            if (
                parent_port.shape_known != child_port.shape_known
                or parent_port.shape != child_port.shape
            ):
                raise ValueError("boundary binding must preserve Tensor shape")
            if parent_port.tensor_spec_id != child_port.tensor_spec_id:
                raise ValueError("boundary binding must preserve tensor_spec_id")
            expected_parent_direction = (
                PortDirection.INPUT
                if binding.kind in {GraphBoundaryKind.INPUT, GraphBoundaryKind.STATE_READ}
                else PortDirection.OUTPUT
            )
            expected_child_direction = (
                PortDirection.OUTPUT
                if binding.kind in {GraphBoundaryKind.INPUT, GraphBoundaryKind.STATE_READ}
                else PortDirection.INPUT
            )
            if parent_port.direction != expected_parent_direction:
                raise ValueError("boundary kind disagrees with parent port direction")
            if child_port.direction != expected_child_direction:
                raise ValueError("boundary kind disagrees with child boundary direction")

            state_kind = binding.kind in {
                GraphBoundaryKind.STATE_READ,
                GraphBoundaryKind.STATE_WRITE,
            }
            state_role = parent_port.role in {TensorRole.CACHE, TensorRole.STATE}
            if state_kind and not state_role:
                raise ValueError("STATE_READ/STATE_WRITE boundary requires cache or state role")
            if not state_kind and state_role:
                raise ValueError("INPUT/OUTPUT boundary cannot bind cache or state role")
            if state_kind:
                tensor_spec_id = child_port.tensor_spec_id
                if tensor_spec_id is None:
                    raise ValueError("state boundary bindings require tensor_spec_id pairing")
                state_bindings = (
                    state_reads_by_tensor
                    if binding.kind == GraphBoundaryKind.STATE_READ
                    else state_writes_by_tensor
                )
                if tensor_spec_id in state_bindings:
                    raise ValueError(
                        "state boundary tensor_spec_id must have exactly one read and write"
                    )
                state_bindings[tensor_spec_id] = binding

        if set(state_reads_by_tensor) != set(state_writes_by_tensor):
            raise ValueError("state read/write boundaries must pair by tensor_spec_id")

        data_inputs = [
            binding.child_port_id
            for binding in child.boundary_bindings
            if binding.kind == GraphBoundaryKind.INPUT
        ]
        data_outputs = [
            binding.child_port_id
            for binding in child.boundary_bindings
            if binding.kind == GraphBoundaryKind.OUTPUT
        ]
        transitions: Dict[str, List[Tuple[str, bool, Optional[GraphEdgeKind]]]] = {
            port.id: [] for port in child.ports
        }
        for edge in child.edges:
            transitions[edge.source_port_id].append((edge.target_port_id, False, edge.kind))
        for node in child.nodes:
            if node.kind == "boundary":
                continue
            for input_port_id in node.input_port_ids:
                for output_port_id in node.output_port_ids:
                    transitions[input_port_id].append((output_port_id, True, None))

        def reaches_through_compute(
            source_port_id: str,
            target_port_id: str,
            allowed_edge_kinds: Set[GraphEdgeKind],
        ) -> bool:
            pending = [(source_port_id, False)]
            seen = {(source_port_id, False)}
            while pending:
                current_port_id, traversed_compute = pending.pop()
                for next_port_id, is_compute_step, edge_kind in transitions[current_port_id]:
                    if edge_kind is not None and edge_kind not in allowed_edge_kinds:
                        continue
                    next_state = (
                        next_port_id,
                        traversed_compute or is_compute_step,
                    )
                    if next_state[0] == target_port_id and next_state[1]:
                        return True
                    if next_state not in seen:
                        seen.add(next_state)
                        pending.append(next_state)
            return False

        primary_edge_kinds = {
            GraphEdgeKind.DATA,
            GraphEdgeKind.ROUTE,
            GraphEdgeKind.CONTROL,
        }
        for source in data_inputs:
            if any(
                not reaches_through_compute(source, target, primary_edge_kinds)
                for target in data_outputs
            ):
                raise ValueError("decomposition must preserve data boundary reachability")

        all_edge_kinds = set(GraphEdgeKind)
        for tensor_spec_id, read_binding in state_reads_by_tensor.items():
            write_binding = state_writes_by_tensor[tensor_spec_id]
            if not reaches_through_compute(
                read_binding.child_port_id,
                write_binding.child_port_id,
                all_edge_kinds,
            ):
                raise ValueError(
                    "state boundary read/write path must traverse a non-boundary compute node"
                )

    @staticmethod
    def _validate_view_hierarchy_is_acyclic(views_by_id: Dict[str, GraphView]) -> None:
        for view in views_by_id.values():
            seen: Set[str] = set()
            cursor: Optional[GraphView] = view
            while cursor is not None:
                if cursor.id in seen:
                    raise ValueError("graph view hierarchy must be acyclic")
                seen.add(cursor.id)
                cursor = views_by_id.get(cursor.parent_view_id or "")


__all__ = [
    "GraphEdge",
    "GraphEdgeKind",
    "GraphBoundaryBinding",
    "GraphBoundaryKind",
    "GraphCostReconciliation",
    "GraphCostReconciliationStatus",
    "GraphDecompositionStatus",
    "GraphGroup",
    "GraphMetricBinding",
    "GraphMetricBindingStatus",
    "GraphNode",
    "GraphPort",
    "GraphPrimitiveKind",
    "GraphView",
    "GraphViewDocument",
    "GraphViewLevel",
    "GraphViewProvenance",
    "ShapeDimension",
]
