"""Renderer-neutral graph-view contracts for the interactive DAG canvas.

The graph view is a projection of Model Map evidence.  It deliberately does not
claim to be an executed framework graph: config-origin activation interfaces are
synthetic display contracts, while state/cache ports may bind existing TensorSpec
records when config inventory proves them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Union

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
    """Supported semantic zoom levels in the first DAG canvas release."""

    MODEL = "L0"
    BLOCK = "L1"


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
    attributes: Dict[str, Any] = Field(default_factory=dict)


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
    breadcrumb: List[str] = Field(default_factory=list)
    layer_index: Optional[int] = Field(default=None, ge=0)
    root_group_id: str = Field(min_length=1)
    nodes: List[GraphNode] = Field(default_factory=list)
    ports: List[GraphPort] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    groups: List[GraphGroup] = Field(default_factory=list)
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

        ports_by_id = {port.id: port for port in self.ports}
        nodes_by_id = {node.id: node for node in self.nodes}
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
    """Top-level renderer-neutral artifact containing related L0/L1 views."""

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
        for view in self.views:
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
        return self


__all__ = [
    "GraphEdge",
    "GraphEdgeKind",
    "GraphGroup",
    "GraphNode",
    "GraphPort",
    "GraphView",
    "GraphViewDocument",
    "GraphViewLevel",
    "GraphViewProvenance",
    "ShapeDimension",
]
