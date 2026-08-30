"""Config-first projection from AdapterResult/ModelMap into GraphView DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from llm_vis.adapters.base import AdapterResult, LayerStripEntry
from llm_vis.ir.ids import canonical_json, deterministic_id
from llm_vis.ir.models import (
    DType,
    ExpressionOp,
    ModelMap,
    PortDirection,
    TensorOrigin,
    TensorRole,
    TensorSpec,
)

from .models import (
    GraphBoundaryBinding,
    GraphBoundaryKind,
    GraphCostReconciliation,
    GraphCostReconciliationStatus,
    GraphDecompositionStatus,
    GraphEdge,
    GraphEdgeKind,
    GraphGroup,
    GraphMetricBinding,
    GraphMetricBindingStatus,
    GraphNode,
    GraphPort,
    GraphPrimitiveKind,
    GraphView,
    GraphViewDocument,
    GraphViewLevel,
    GraphViewProvenance,
    ShapeDimension,
)

_PROJECTION_VERSION = "0.1"
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
}


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _known_metric_binding(
    dimension: str, metric_name: str, parent_metric_name: str
) -> GraphMetricBinding:
    return GraphMetricBinding(
        dimension=dimension,
        metric_name=metric_name,
        parent_metric_name=parent_metric_name,
        status=GraphMetricBindingStatus.KNOWN,
    )


def _unknown_metric_binding(
    dimension: str, parent_metric_name: str, reason: str
) -> GraphMetricBinding:
    return GraphMetricBinding(
        dimension=dimension,
        parent_metric_name=parent_metric_name,
        status=GraphMetricBindingStatus.UNKNOWN,
        reason=reason,
    )


def _excluded_metric_binding(
    dimension: str, parent_metric_name: str, reason: str
) -> GraphMetricBinding:
    return GraphMetricBinding(
        dimension=dimension,
        parent_metric_name=parent_metric_name,
        status=GraphMetricBindingStatus.EXCLUDED,
        reason=reason,
    )


def _config_dtype(value: Any) -> DType:
    if not isinstance(value, str):
        return DType.UNKNOWN
    normalized = _DTYPE_ALIASES.get(value.lower(), value.lower())
    try:
        return DType(normalized)
    except ValueError:
        return DType.UNKNOWN


def _tensor_shape(tensor: TensorSpec) -> Optional[List[ShapeDimension]]:
    if not tensor.shape_known:
        return None
    shape: List[ShapeDimension] = []
    for expression in tensor.symbolic_shape:
        if expression.op == ExpressionOp.LITERAL and isinstance(expression.value, int):
            shape.append(expression.value)
        elif expression.op == ExpressionOp.SYMBOL and expression.symbol is not None:
            shape.append(expression.symbol)
        else:
            shape.append(canonical_json(expression))
    return shape


@dataclass(frozen=True)
class _EvidenceIndex:
    result: AdapterResult
    model_map: ModelMap
    model_map_id: str
    instance_ids: Mapping[str, str]
    definition_ids: Mapping[str, str]
    semantic_ids_by_instance: Mapping[str, Tuple[str, ...]]
    semantic_kinds_by_id: Mapping[str, str]
    tensors_by_instance: Mapping[str, Tuple[TensorSpec, ...]]

    @classmethod
    def build(cls, result: AdapterResult, model_map: ModelMap) -> _EvidenceIndex:
        instance_ids = {instance.module_path: instance.id for instance in model_map.instances}
        missing_paths = {instance.path for instance in result.instances} - set(instance_ids)
        if missing_paths:
            raise ValueError(
                f"Model Map is missing adapter instances: {sorted(missing_paths)[:3]!r}"
            )
        definition_ids: Dict[str, str] = {}
        for instance in result.instances:
            model_instance = next(
                candidate
                for candidate in model_map.instances
                if candidate.module_path == instance.path
            )
            definition_ids.setdefault(instance.definition_key, model_instance.definition_id)

        semantic_ids_by_instance: Dict[str, List[str]] = {path: [] for path in instance_ids}
        path_by_instance_id = {value: key for key, value in instance_ids.items()}
        for semantic_node in model_map.semantic_nodes:
            for instance_id in semantic_node.instance_ids:
                path = path_by_instance_id.get(instance_id)
                if path is not None:
                    semantic_ids_by_instance[path].append(semantic_node.id)

        tensors_by_instance: Dict[str, List[TensorSpec]] = {path: [] for path in instance_ids}
        for tensor in model_map.tensors:
            path = path_by_instance_id.get(tensor.owner_id or "")
            if path is not None:
                tensors_by_instance[path].append(tensor)
        return cls(
            result=result,
            model_map=model_map,
            model_map_id=deterministic_id("mmap", model_map),
            instance_ids=instance_ids,
            definition_ids=definition_ids,
            semantic_ids_by_instance={
                key: tuple(sorted(value)) for key, value in semantic_ids_by_instance.items()
            },
            semantic_kinds_by_id={node.id: node.kind.value for node in model_map.semantic_nodes},
            tensors_by_instance={
                key: tuple(sorted(value, key=lambda item: (item.semantic_name or "", item.id)))
                for key, value in tensors_by_instance.items()
            },
        )

    @property
    def common_evidence(self) -> List[str]:
        return [
            f"adapter:{self.result.adapter_name}",
            "source:config",
            f"model:{self.model_map.model.id}",
            f"model_map:{self.model_map_id}",
            "weights_loaded:false",
            "target_model_constructed:false",
            "target_model_forward:false",
        ]

    def subjects(
        self,
        *,
        paths: Sequence[str] = (),
        definition_keys: Sequence[str] = (),
        include_model: bool = False,
        include_semantics: bool = True,
        semantic_kinds: Sequence[str] = (),
    ) -> List[str]:
        values: List[str] = [self.model_map.model.id] if include_model else []
        for key in definition_keys:
            definition_id = self.definition_ids.get(key)
            if definition_id is not None:
                values.append(definition_id)
        for path in paths:
            instance_id = self.instance_ids.get(path)
            if instance_id is not None:
                values.append(instance_id)
            if include_semantics:
                semantic_ids = self.semantic_ids_by_instance.get(path, ())
                if semantic_kinds:
                    allowed = set(semantic_kinds)
                    semantic_ids = tuple(
                        semantic_id
                        for semantic_id in semantic_ids
                        if self.semantic_kinds_by_id.get(semantic_id) in allowed
                    )
                values.extend(semantic_ids)
        return _unique(values)

    def state_tensors(self, instance_path: str) -> Tuple[TensorSpec, ...]:
        return tuple(
            tensor
            for tensor in self.tensors_by_instance.get(instance_path, ())
            if tensor.role in {TensorRole.CACHE, TensorRole.STATE}
        )


class _ViewBuilder:
    """Small deterministic builder that keeps GraphView object creation readable."""

    def __init__(
        self,
        *,
        view_id: str,
        key: str,
        level: GraphViewLevel,
        label: str,
        parent_view_id: Optional[str],
        breadcrumb: Sequence[str],
        layer_index: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.view_id = view_id
        self.key = key
        self.level = level
        self.label = label
        self.parent_view_id = parent_view_id
        self.breadcrumb = list(breadcrumb)
        self.layer_index = layer_index
        self.metadata = dict(metadata or {})
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._ports: Dict[str, GraphPort] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._groups: Dict[str, Dict[str, Any]] = {}

    def _id(self, namespace: str, key: str) -> str:
        return deterministic_id(namespace, {"view_id": self.view_id, "key": key})

    def add_group(
        self,
        key: str,
        label: str,
        kind: str,
        *,
        parent_group_key: Optional[str] = None,
        collapsed: bool = False,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if key in self._groups:
            raise ValueError(f"duplicate graph group key {key!r}")
        group_id = self._id("ggrp", key)
        parent_group_id = (
            self._groups[parent_group_key]["id"] if parent_group_key is not None else None
        )
        self._groups[key] = {
            "id": group_id,
            "key": key,
            "label": label,
            "kind": kind,
            "member_node_ids": [],
            "parent_group_id": parent_group_id,
            "collapsed": collapsed,
            "attributes": dict(attributes or {}),
        }
        return group_id

    def add_node(
        self,
        key: str,
        label: str,
        kind: str,
        *,
        group_key: str,
        subject_ids: Sequence[str] = (),
        evidence: Sequence[str] = (),
        coverage: float = 1.0,
        opaque: bool = False,
        drilldown_view_id: Optional[str] = None,
        decomposition_status: GraphDecompositionStatus = GraphDecompositionStatus.NONE,
        primitive_kind: Optional[GraphPrimitiveKind] = None,
        metric_bindings: Sequence[GraphMetricBinding] = (),
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if key in self._nodes:
            raise ValueError(f"duplicate graph node key {key!r}")
        node_id = self._id("gnode", key)
        self._nodes[key] = {
            "id": node_id,
            "key": key,
            "label": label,
            "kind": kind,
            "group_id": self._groups[group_key]["id"],
            "input_port_ids": [],
            "output_port_ids": [],
            "subject_ids": list(subject_ids),
            "evidence": list(evidence),
            "coverage": coverage,
            "opaque": opaque,
            "drilldown_view_id": drilldown_view_id,
            "decomposition_status": decomposition_status,
            "primitive_kind": primitive_kind,
            "metric_bindings": list(metric_bindings),
            "attributes": dict(attributes or {}),
        }
        self._groups[group_key]["member_node_ids"].append(node_id)
        return node_id

    def add_port(
        self,
        node_key: str,
        key: str,
        name: str,
        direction: PortDirection,
        role: TensorRole,
        *,
        shape: Optional[Sequence[ShapeDimension]],
        dtype: DType,
        tensor_spec_id: Optional[str] = None,
        evidence: Sequence[str] = (),
    ) -> str:
        canonical_key = f"{node_key}.{key}"
        if canonical_key in self._ports:
            raise ValueError(f"duplicate graph port key {canonical_key!r}")
        port_id = self._id("gport", canonical_key)
        port = GraphPort(
            id=port_id,
            key=canonical_key,
            node_id=self._nodes[node_key]["id"],
            name=name,
            direction=direction,
            role=role,
            shape=list(shape) if shape is not None else None,
            shape_known=shape is not None,
            dtype=dtype,
            tensor_spec_id=tensor_spec_id,
            evidence=list(evidence),
        )
        self._ports[canonical_key] = port
        collection = "input_port_ids" if direction == PortDirection.INPUT else "output_port_ids"
        self._nodes[node_key][collection].append(port_id)
        return port_id

    def port_id(self, node_key: str, port_key: str) -> str:
        return self._ports[f"{node_key}.{port_key}"].id

    def add_edge(
        self,
        key: str,
        source_node_key: str,
        source_port_key: str,
        target_node_key: str,
        target_port_key: str,
        kind: GraphEdgeKind,
        *,
        evidence: Sequence[str],
        coverage: float = 1.0,
        tensor_spec_id: Optional[str] = None,
    ) -> str:
        if key in self._edges:
            raise ValueError(f"duplicate graph edge key {key!r}")
        edge = GraphEdge(
            id=self._id("gedge", key),
            key=key,
            source_port_id=self.port_id(source_node_key, source_port_key),
            target_port_id=self.port_id(target_node_key, target_port_key),
            kind=kind,
            tensor_spec_id=tensor_spec_id,
            origin=TensorOrigin.CONFIG,
            evidence=list(evidence),
            coverage=coverage,
        )
        self._edges[key] = edge
        return edge.id

    def finish(
        self,
        *,
        root_group_key: str = "root",
        decomposes_node_id: Optional[str] = None,
        boundary_bindings: Sequence[GraphBoundaryBinding] = (),
        cost_frontier_node_keys: Sequence[str] = (),
        cost_reconciliations: Sequence[GraphCostReconciliation] = (),
    ) -> GraphView:
        return GraphView(
            id=self.view_id,
            key=self.key,
            level=self.level,
            label=self.label,
            parent_view_id=self.parent_view_id,
            decomposes_node_id=decomposes_node_id,
            breadcrumb=self.breadcrumb,
            layer_index=self.layer_index,
            root_group_id=self._groups[root_group_key]["id"],
            nodes=[GraphNode(**payload) for payload in self._nodes.values()],
            ports=list(self._ports.values()),
            edges=list(self._edges.values()),
            groups=[GraphGroup(**payload) for payload in self._groups.values()],
            boundary_bindings=list(boundary_bindings),
            cost_frontier_node_ids=[self._nodes[key]["id"] for key in cost_frontier_node_keys],
            cost_reconciliations=list(cost_reconciliations),
            metadata=self.metadata,
        )


def _finish_decomposition_view(
    builder: _ViewBuilder,
    *,
    parent_view: GraphView,
    parent_node_key: str,
    boundary_specs: Sequence[Tuple[str, str, GraphBoundaryKind]],
    cost_frontier_node_keys: Sequence[str],
    reconciliation_specs: Sequence[Mapping[str, Any]] = (),
) -> GraphView:
    """Finish a child view and attach a lossless cross-view boundary contract."""

    base = builder.finish(cost_frontier_node_keys=cost_frontier_node_keys)
    parent_node = next(node for node in parent_view.nodes if node.key == parent_node_key)
    parent_ports = {port.key: port for port in parent_view.ports}
    child_ports = {port.key: port for port in base.ports}
    child_nodes = {node.key: node for node in base.nodes}
    bindings = []
    for parent_port_key, child_port_key, kind in boundary_specs:
        key = f"{parent_port_key}->{child_port_key}"
        bindings.append(
            GraphBoundaryBinding(
                id=deterministic_id("gbind", {"view_id": base.id, "key": key}),
                key=key,
                parent_port_id=parent_ports[parent_port_key].id,
                child_port_id=child_ports[child_port_key].id,
                kind=kind,
                evidence=(
                    "source:config",
                    "contract:decomposition-boundary",
                    "shape-preserved:true",
                    "target_model_forward:false",
                ),
            )
        )
    reconciliations = []
    for spec in reconciliation_specs:
        reconciliations.append(
            GraphCostReconciliation(
                dimension=spec["dimension"],
                parent_metric_name=spec["parent_metric_name"],
                known_child_metric_names=list(spec.get("known_child_metric_names", ())),
                unattributed_node_ids=[
                    child_nodes[key].id for key in spec.get("unattributed_node_keys", ())
                ],
                excluded_node_ids=[
                    child_nodes[key].id for key in spec.get("excluded_node_keys", ())
                ],
                status=spec["status"],
                reason=spec["reason"],
                evidence=list(
                    spec.get(
                        "evidence",
                        (
                            "source:config",
                            "cost:formula-scope",
                            "unknown-is-not-zero",
                        ),
                    )
                ),
            )
        )
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "decomposes_node_id": parent_node.id,
            "boundary_bindings": [item.model_dump(mode="json") for item in bindings],
            "cost_reconciliations": [item.model_dump(mode="json") for item in reconciliations],
        }
    )
    return GraphView.model_validate(payload)


def _view_id(result: AdapterResult, view_key: str) -> str:
    return deterministic_id(
        "gvw",
        {
            "adapter_name": result.adapter_name,
            "model_id": result.model_id,
            "projection_version": _PROJECTION_VERSION,
            "revision": result.revision,
            "view_key": view_key,
        },
    )


def _view_ids(result: AdapterResult) -> Dict[str, str]:
    keys = ["l0"]
    if result.adapter_name == "qwen3-5":
        if any(entry.attention_kind == "linear_attention" for entry in result.layer_strip):
            keys.extend(("l1_linear_attention", "op_linear_attention", "op_linear_ffn"))
        if any(entry.attention_kind == "full_attention" for entry in result.layer_strip):
            keys.extend(("l1_full_attention", "op_full_attention", "op_full_ffn"))
    elif result.adapter_name == "glm-moe-dsa":
        if any(entry.mlp_kind == "dense" for entry in result.layer_strip):
            keys.extend(("l1_dense_dsa", "op_glm_dense_ffn"))
        if any(entry.mlp_kind == "sparse" for entry in result.layer_strip):
            keys.extend(
                (
                    "l1_sparse_dsa_moe",
                    "op_glm_routed_expert_ffn",
                    "op_glm_shared_expert_ffn",
                )
            )
    elif result.adapter_name == "tiny-dense":
        keys.append("l1_dense")
    else:
        raise ValueError(f"GraphView projection does not support adapter {result.adapter_name!r}")
    return {key: _view_id(result, key) for key in keys}


def _hidden_shape(result: AdapterResult) -> List[ShapeDimension]:
    hidden_size = result.inventory_config.get("hidden_size") or result.metadata.get("hidden_size")
    return ["B", "T", int(hidden_size)]


def _logits_shape(result: AdapterResult) -> List[ShapeDimension]:
    vocab_size = result.inventory_config.get("vocab_size") or result.metadata.get("vocab_size")
    return ["B", "T", int(vocab_size)]


def _activation_dtype(result: AdapterResult) -> DType:
    return _config_dtype(result.inventory_config.get("dtype"))


def _add_activation_port(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    node_key: str,
    port_key: str,
    name: str,
    direction: PortDirection,
    *,
    shape: Optional[Sequence[ShapeDimension]] = None,
    dtype: Optional[DType] = None,
    role: TensorRole = TensorRole.ACTIVATION,
) -> str:
    return builder.add_port(
        node_key,
        port_key,
        name,
        direction,
        role,
        shape=shape if shape is not None else _hidden_shape(index.result),
        dtype=dtype or _activation_dtype(index.result),
        evidence=(*index.common_evidence, "interface:config-projected"),
    )


def _add_data_edge(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    key: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    *,
    kind: GraphEdgeKind = GraphEdgeKind.DATA,
    coverage: float = 1.0,
) -> None:
    builder.add_edge(
        key,
        source_node,
        source_port,
        target_node,
        target_port,
        kind,
        evidence=(*index.common_evidence, f"edge:{kind.value}"),
        coverage=coverage,
    )


def _add_state_rails(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    *,
    group_key: str,
    layer_path: str,
    attention_node_key: str,
) -> None:
    tensors = index.state_tensors(layer_path)
    state_kind = "state" if any(tensor.role == TensorRole.STATE for tensor in tensors) else "cache"
    shape_coverage = 1.0 if tensors and all(tensor.shape_known for tensor in tensors) else 0.0
    semantic_kind = "recurrent_state" if state_kind == "state" else "kv_cache"
    subjects = index.subjects(paths=(layer_path,), semantic_kinds=(semantic_kind,))
    builder.add_node(
        "state_in",
        "State In" if state_kind == "state" else "KV Cache In",
        "state_boundary_in",
        group_key=group_key,
        subject_ids=subjects,
        evidence=(*index.common_evidence, "rail:state_in"),
        coverage=shape_coverage,
        attributes={"rail": "input", "state_kind": state_kind},
    )
    builder.add_node(
        "state_out",
        "State Out" if state_kind == "state" else "KV Cache Out",
        "state_boundary_out",
        group_key=group_key,
        subject_ids=subjects,
        evidence=(*index.common_evidence, "rail:state_out"),
        coverage=shape_coverage,
        attributes={"rail": "output", "state_kind": state_kind},
    )
    if not tensors:
        raise ValueError(f"layer {layer_path!r} has no config state/cache TensorSpec")
    for ordinal, tensor in enumerate(tensors):
        semantic_name = tensor.semantic_name or f"state_{ordinal}"
        port_key = semantic_name.replace(".", "_")
        role = tensor.role
        shape = _tensor_shape(tensor)
        evidence = _unique((*index.common_evidence, *tensor.evidence))
        builder.add_port(
            "state_in",
            port_key,
            semantic_name,
            PortDirection.OUTPUT,
            role,
            shape=shape,
            dtype=tensor.dtype,
            tensor_spec_id=tensor.id,
            evidence=evidence,
        )
        builder.add_port(
            attention_node_key,
            f"{port_key}_in",
            f"{semantic_name} in",
            PortDirection.INPUT,
            role,
            shape=shape,
            dtype=tensor.dtype,
            tensor_spec_id=tensor.id,
            evidence=evidence,
        )
        builder.add_port(
            attention_node_key,
            f"{port_key}_out",
            f"{semantic_name} out",
            PortDirection.OUTPUT,
            role,
            shape=shape,
            dtype=tensor.dtype,
            tensor_spec_id=tensor.id,
            evidence=evidence,
        )
        builder.add_port(
            "state_out",
            port_key,
            semantic_name,
            PortDirection.INPUT,
            role,
            shape=shape,
            dtype=tensor.dtype,
            tensor_spec_id=tensor.id,
            evidence=evidence,
        )
        coverage = 1.0 if tensor.shape_known else 0.0
        builder.add_edge(
            f"state_read_{ordinal}",
            "state_in",
            port_key,
            attention_node_key,
            f"{port_key}_in",
            GraphEdgeKind.STATE_READ,
            tensor_spec_id=tensor.id,
            evidence=(*evidence, "edge:state_read"),
            coverage=coverage,
        )
        builder.add_edge(
            f"state_write_{ordinal}",
            attention_node_key,
            f"{port_key}_out",
            "state_out",
            port_key,
            GraphEdgeKind.STATE_WRITE,
            tensor_spec_id=tensor.id,
            evidence=(*evidence, "edge:state_write"),
            coverage=coverage,
        )


def _add_l0_common_start(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    *,
    embedding_path: str,
    embedding_definition: str,
) -> None:
    evidence = index.common_evidence
    builder.add_node(
        "tokens",
        "Token IDs",
        "input",
        group_key="root",
        subject_ids=index.subjects(include_model=True),
        evidence=(*evidence, "boundary:model_input"),
        attributes={"boundary": "input"},
    )
    builder.add_port(
        "tokens",
        "token_ids",
        "token_ids",
        PortDirection.OUTPUT,
        TensorRole.INPUT,
        shape=["B", "T"],
        dtype=DType.INT64,
        evidence=(*evidence, "interface:token_ids"),
    )
    builder.add_node(
        "embedding",
        "Token Embedding",
        "embedding",
        group_key="root",
        subject_ids=index.subjects(
            paths=(embedding_path,), definition_keys=(embedding_definition,)
        ),
        evidence=(*evidence, "source:config_tensor_inventory"),
        attributes={"vocab_size": index.result.inventory_config.get("vocab_size")},
    )
    builder.add_port(
        "embedding",
        "token_ids",
        "token_ids",
        PortDirection.INPUT,
        TensorRole.INPUT,
        shape=["B", "T"],
        dtype=DType.INT64,
        evidence=(*evidence, "interface:token_ids"),
    )
    _add_activation_port(
        builder, index, "embedding", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    _add_data_edge(
        builder, index, "tokens_to_embedding", "tokens", "token_ids", "embedding", "token_ids"
    )


def _add_l0_common_end(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    *,
    upstream_node: str,
    norm_path: str,
) -> None:
    evidence = index.common_evidence
    builder.add_node(
        "final_norm",
        "Final Norm",
        "norm",
        group_key="root",
        subject_ids=index.subjects(paths=(norm_path,), definition_keys=("final_norm",)),
        evidence=evidence,
    )
    _add_activation_port(builder, index, "final_norm", "in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(
        builder, index, "final_norm", "out", "normalized_hidden", PortDirection.OUTPUT
    )
    builder.add_node(
        "lm_head",
        "LM Head",
        "lm_head",
        group_key="root",
        subject_ids=index.subjects(paths=("lm_head",), definition_keys=("lm_head",)),
        evidence=(*evidence, "source:config_tensor_inventory"),
        attributes={"vocab_size": index.result.inventory_config.get("vocab_size")},
    )
    _add_activation_port(builder, index, "lm_head", "hidden", "hidden_states", PortDirection.INPUT)
    _add_activation_port(
        builder,
        index,
        "lm_head",
        "logits",
        "logits",
        PortDirection.OUTPUT,
        shape=_logits_shape(index.result),
    )
    builder.add_node(
        "logits",
        "Logits",
        "output",
        group_key="root",
        subject_ids=index.subjects(include_model=True),
        evidence=(*evidence, "boundary:model_output"),
        attributes={"boundary": "output"},
    )
    _add_activation_port(
        builder,
        index,
        "logits",
        "logits",
        "logits",
        PortDirection.INPUT,
        shape=_logits_shape(index.result),
    )
    _add_data_edge(builder, index, "decoder_to_norm", upstream_node, "hidden", "final_norm", "in")
    _add_data_edge(builder, index, "norm_to_head", "final_norm", "out", "lm_head", "hidden")
    _add_data_edge(builder, index, "head_to_logits", "lm_head", "logits", "logits", "logits")


def _add_mtp_branch(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    *,
    upstream_node: str,
    mtp_paths: Sequence[str],
) -> None:
    if not mtp_paths:
        return
    evidence = index.common_evidence
    builder.add_node(
        "mtp",
        f"MTP × {len(mtp_paths)} (conditional)",
        "mtp",
        group_key="root",
        subject_ids=index.subjects(paths=mtp_paths, definition_keys=("mtp_block",)),
        evidence=(*evidence, "config:conditional_mtp"),
        attributes={"conditional": True, "instance_count": len(mtp_paths)},
    )
    _add_activation_port(
        builder,
        index,
        "mtp",
        "hidden_in",
        "hidden_states",
        PortDirection.INPUT,
    )
    _add_activation_port(
        builder,
        index,
        "mtp",
        "logits",
        "mtp_logits",
        PortDirection.OUTPUT,
        shape=_logits_shape(index.result),
    )
    builder.add_node(
        "mtp_logits",
        "MTP Draft Logits",
        "output",
        group_key="root",
        subject_ids=index.subjects(paths=mtp_paths, definition_keys=("mtp_block",)),
        evidence=(*evidence, "boundary:conditional_mtp_output"),
        attributes={"boundary": "conditional_output", "conditional": True},
    )
    _add_activation_port(
        builder,
        index,
        "mtp_logits",
        "logits",
        "mtp_logits",
        PortDirection.INPUT,
        shape=_logits_shape(index.result),
    )
    _add_data_edge(
        builder,
        index,
        "decoder_to_mtp",
        upstream_node,
        "hidden",
        "mtp",
        "hidden_in",
    )
    _add_data_edge(
        builder,
        index,
        "mtp_to_logits",
        "mtp",
        "logits",
        "mtp_logits",
        "logits",
    )


def _build_qwen_l0(index: _EvidenceIndex, view_ids: Mapping[str, str]) -> GraphView:
    result = index.result
    builder = _ViewBuilder(
        view_id=view_ids["l0"],
        key="l0",
        level=GraphViewLevel.MODEL,
        label="Qwen model DAG",
        parent_view_id=None,
        breadcrumb=[result.model_id],
        metadata={
            "layout_direction": "RIGHT",
            "primary_flow": "acyclic",
            "semantic_zoom": "model",
        },
    )
    builder.add_group("root", "Qwen L0", "model", attributes={"layout": "layered"})
    _add_l0_common_start(
        builder,
        index,
        embedding_path="model.text.embed_tokens",
        embedding_definition="token_embedding",
    )
    layer_paths = tuple(entry.instance_path for entry in result.layer_strip)
    linear_count = sum(entry.attention_kind == "linear_attention" for entry in result.layer_strip)
    full_count = len(result.layer_strip) - linear_count
    variants = {
        key: value
        for key, value in (
            ("linear_attention", view_ids.get("l1_linear_attention")),
            ("full_attention", view_ids.get("l1_full_attention")),
        )
        if value is not None
    }
    default_drilldown = variants.get("linear_attention") or variants.get("full_attention")
    builder.add_node(
        "decoder_pattern",
        f"Hybrid Decoder [L×3 → A] × {len(result.layer_strip) // 4}",
        "decoder_pattern",
        group_key="root",
        subject_ids=index.subjects(
            paths=layer_paths,
            definition_keys=("text_decoder", "linear_decoder_block", "full_decoder_block"),
        ),
        evidence=(
            *index.common_evidence,
            "layer_strip:linear_attention/full_attention",
            f"pattern:{result.metadata.get('layer_pattern', 'explicit')}",
        ),
        drilldown_view_id=default_drilldown,
        decomposition_status=GraphDecompositionStatus.AVAILABLE,
        attributes={
            "layer_count": len(result.layer_strip),
            "linear_attention_layers": linear_count,
            "full_attention_layers": full_count,
            "pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "repeat_count": len(result.layer_strip) // 4,
            "layer_indices": [entry.layer_index for entry in result.layer_strip],
            "drilldown_view_ids": variants,
        },
    )
    _add_activation_port(
        builder, index, "decoder_pattern", "hidden_in", "text_hidden", PortDirection.INPUT
    )
    _add_activation_port(
        builder, index, "decoder_pattern", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    _add_data_edge(
        builder,
        index,
        "embedding_to_decoder",
        "embedding",
        "hidden",
        "decoder_pattern",
        "hidden_in",
    )

    if bool(result.metadata.get("is_multimodal")):
        builder.add_node(
            "vision_input",
            "Image / Video Input (conditional)",
            "input",
            group_key="root",
            subject_ids=index.subjects(include_model=True),
            evidence=(*index.common_evidence, "config:vision_config"),
            attributes={"boundary": "conditional_input", "conditional": True},
        )
        builder.add_port(
            "vision_input",
            "pixels",
            "pixel_values",
            PortDirection.OUTPUT,
            TensorRole.INPUT,
            shape=None,
            dtype=_activation_dtype(result),
            evidence=(*index.common_evidence, "config:vision_shape_unknown"),
        )
        builder.add_node(
            "vision_tower",
            "Vision Tower (conditional)",
            "vision_tower",
            group_key="root",
            subject_ids=index.subjects(paths=("model.vision",), definition_keys=("vision_tower",)),
            evidence=(*index.common_evidence, "config:vision_component_present"),
            attributes={"conditional": True},
        )
        builder.add_port(
            "vision_tower",
            "pixels",
            "pixel_values",
            PortDirection.INPUT,
            TensorRole.INPUT,
            shape=None,
            dtype=_activation_dtype(result),
            evidence=(*index.common_evidence, "config:vision_shape_unknown"),
        )
        builder.add_port(
            "vision_tower",
            "features",
            "vision_features",
            PortDirection.OUTPUT,
            TensorRole.ACTIVATION,
            shape=None,
            dtype=_activation_dtype(result),
            evidence=(*index.common_evidence, "config:vision_shape_unknown"),
        )
        builder.add_node(
            "projector",
            "Vision Projector",
            "projector",
            group_key="root",
            subject_ids=index.subjects(
                paths=("model.projector",), definition_keys=("multimodal_projector",)
            ),
            evidence=(*index.common_evidence, "config:projector_component_present"),
            attributes={"conditional": True},
        )
        builder.add_port(
            "projector",
            "features",
            "vision_features",
            PortDirection.INPUT,
            TensorRole.ACTIVATION,
            shape=None,
            dtype=_activation_dtype(result),
            evidence=(*index.common_evidence, "config:vision_shape_unknown"),
        )
        _add_activation_port(
            builder,
            index,
            "projector",
            "hidden",
            "projected_hidden",
            PortDirection.OUTPUT,
            shape=["B", "T_vision", _hidden_shape(result)[-1]],
        )
        _add_activation_port(
            builder,
            index,
            "decoder_pattern",
            "visual_hidden",
            "visual_hidden",
            PortDirection.INPUT,
            shape=["B", "T_vision", _hidden_shape(result)[-1]],
        )
        _add_data_edge(
            builder,
            index,
            "vision_input_to_tower",
            "vision_input",
            "pixels",
            "vision_tower",
            "pixels",
        )
        _add_data_edge(
            builder,
            index,
            "vision_tower_to_projector",
            "vision_tower",
            "features",
            "projector",
            "features",
        )
        _add_data_edge(
            builder,
            index,
            "projector_to_decoder",
            "projector",
            "hidden",
            "decoder_pattern",
            "visual_hidden",
        )

    _add_l0_common_end(
        builder,
        index,
        upstream_node="decoder_pattern",
        norm_path="model.text.norm",
    )
    mtp_paths = tuple(
        instance.path for instance in result.instances if instance.path.startswith("model.mtp.")
    )
    _add_mtp_branch(builder, index, upstream_node="decoder_pattern", mtp_paths=mtp_paths)
    return builder.finish(cost_frontier_node_keys=("decoder_pattern",))


def _build_glm_l0(index: _EvidenceIndex, view_ids: Mapping[str, str]) -> GraphView:
    result = index.result
    builder = _ViewBuilder(
        view_id=view_ids["l0"],
        key="l0",
        level=GraphViewLevel.MODEL,
        label="GLM model DAG",
        parent_view_id=None,
        breadcrumb=[result.model_id],
        metadata={
            "layout_direction": "RIGHT",
            "primary_flow": "acyclic",
            "semantic_zoom": "model",
        },
    )
    builder.add_group("root", "GLM L0", "model", attributes={"layout": "layered"})
    _add_l0_common_start(
        builder,
        index,
        embedding_path="model.embed_tokens",
        embedding_definition="token_embedding",
    )
    dense_entries = tuple(entry for entry in result.layer_strip if entry.mlp_kind == "dense")
    sparse_entries = tuple(entry for entry in result.layer_strip if entry.mlp_kind == "sparse")
    previous_node = "embedding"
    previous_port = "hidden"
    if dense_entries:
        builder.add_node(
            "dense_decoder_stage",
            f"Dense DSA × {len(dense_entries)}",
            "decoder_pattern",
            group_key="root",
            subject_ids=index.subjects(
                paths=tuple(entry.instance_path for entry in dense_entries),
                definition_keys=("decoder", "dense_dsa_block"),
            ),
            evidence=(*index.common_evidence, "layer_strip:mlp_kind=dense"),
            drilldown_view_id=view_ids.get("l1_dense_dsa"),
            decomposition_status=GraphDecompositionStatus.AVAILABLE,
            attributes={
                "layer_count": len(dense_entries),
                "layer_indices": [entry.layer_index for entry in dense_entries],
                "attention_kind": "dsa",
                "mlp_kind": "dense",
            },
        )
        _add_activation_port(
            builder, index, "dense_decoder_stage", "hidden_in", "hidden_states", PortDirection.INPUT
        )
        _add_activation_port(
            builder, index, "dense_decoder_stage", "hidden", "hidden_states", PortDirection.OUTPUT
        )
        _add_data_edge(
            builder,
            index,
            "embedding_to_dense_stage",
            previous_node,
            previous_port,
            "dense_decoder_stage",
            "hidden_in",
        )
        previous_node = "dense_decoder_stage"
        previous_port = "hidden"
    if sparse_entries:
        builder.add_node(
            "sparse_decoder_stage",
            f"Sparse DSA + MoE × {len(sparse_entries)}",
            "decoder_pattern",
            group_key="root",
            subject_ids=index.subjects(
                paths=tuple(entry.instance_path for entry in sparse_entries),
                definition_keys=("decoder", "sparse_dsa_moe_block", "expert_pool"),
            ),
            evidence=(*index.common_evidence, "layer_strip:mlp_kind=sparse"),
            drilldown_view_id=view_ids.get("l1_sparse_dsa_moe"),
            decomposition_status=GraphDecompositionStatus.AVAILABLE,
            attributes={
                "layer_count": len(sparse_entries),
                "layer_indices": [entry.layer_index for entry in sparse_entries],
                "attention_kind": "dsa",
                "mlp_kind": "sparse_moe",
                "routed_experts": result.metadata.get("routed_experts"),
                "top_k": result.metadata.get("top_k"),
                "experts_materialized": 0,
            },
        )
        _add_activation_port(
            builder,
            index,
            "sparse_decoder_stage",
            "hidden_in",
            "hidden_states",
            PortDirection.INPUT,
        )
        _add_activation_port(
            builder, index, "sparse_decoder_stage", "hidden", "hidden_states", PortDirection.OUTPUT
        )
        _add_data_edge(
            builder,
            index,
            "dense_to_sparse_stage",
            previous_node,
            previous_port,
            "sparse_decoder_stage",
            "hidden_in",
        )
        previous_node = "sparse_decoder_stage"
    _add_l0_common_end(builder, index, upstream_node=previous_node, norm_path="model.norm")
    mtp_paths = tuple(
        instance.path for instance in result.instances if instance.path.startswith("model.mtp.")
    )
    _add_mtp_branch(builder, index, upstream_node=previous_node, mtp_paths=mtp_paths)
    frontier = tuple(
        key for key in ("dense_decoder_stage", "sparse_decoder_stage") if key in builder._nodes
    )
    return builder.finish(cost_frontier_node_keys=frontier)


def _build_tiny_l0(index: _EvidenceIndex, view_ids: Mapping[str, str]) -> GraphView:
    result = index.result
    builder = _ViewBuilder(
        view_id=view_ids["l0"],
        key="l0",
        level=GraphViewLevel.MODEL,
        label="Dense model DAG",
        parent_view_id=None,
        breadcrumb=[result.model_id],
        metadata={
            "layout_direction": "RIGHT",
            "primary_flow": "acyclic",
            "semantic_zoom": "model",
        },
    )
    builder.add_group("root", "Dense L0", "model", attributes={"layout": "layered"})
    _add_l0_common_start(
        builder,
        index,
        embedding_path="model.embed_tokens",
        embedding_definition="token_embedding",
    )
    layer_paths = tuple(entry.instance_path for entry in result.layer_strip)
    builder.add_node(
        "decoder_pattern",
        f"Dense Decoder × {len(result.layer_strip)}",
        "decoder_pattern",
        group_key="root",
        subject_ids=index.subjects(
            paths=layer_paths,
            definition_keys=("decoder", "dense_decoder_block"),
        ),
        evidence=(*index.common_evidence, "layer_strip:pattern=dense"),
        drilldown_view_id=view_ids["l1_dense"],
        decomposition_status=GraphDecompositionStatus.AVAILABLE,
        attributes={
            "layer_count": len(result.layer_strip),
            "layer_indices": [entry.layer_index for entry in result.layer_strip],
            "attention_kind": "full_attention",
            "mlp_kind": "dense",
        },
    )
    _add_activation_port(
        builder, index, "decoder_pattern", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    _add_activation_port(
        builder, index, "decoder_pattern", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    _add_data_edge(
        builder,
        index,
        "embedding_to_decoder",
        "embedding",
        "hidden",
        "decoder_pattern",
        "hidden_in",
    )
    _add_l0_common_end(builder, index, upstream_node="decoder_pattern", norm_path="model.norm")
    return builder.finish(cost_frontier_node_keys=("decoder_pattern",))


def _representative(
    result: AdapterResult, *, attention: str = "", mlp: str = ""
) -> LayerStripEntry:
    for entry in result.layer_strip:
        if attention and entry.attention_kind != attention:
            continue
        if mlp and entry.mlp_kind != mlp:
            continue
        return entry
    raise ValueError(f"No representative layer for attention={attention!r}, mlp={mlp!r}")


def _add_standard_l1_nodes(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    *,
    entry: LayerStripEntry,
    attention_label: str,
    attention_kind: str,
    attention_opaque: bool,
    attention_coverage: float,
    ffn_label: str,
    attention_drilldown_view_id: Optional[str] = None,
    ffn_drilldown_view_id: Optional[str] = None,
) -> None:
    path = entry.instance_path
    block_subjects = index.subjects(
        paths=(path,),
        definition_keys=(entry.definition_key,),
        include_semantics=False,
    )
    attention_subjects = index.subjects(
        paths=(path,),
        definition_keys=(entry.definition_key,),
        semantic_kinds=(entry.attention_kind,),
    )
    ffn_subjects = index.subjects(
        paths=(path,),
        definition_keys=(entry.definition_key,),
        semantic_kinds=("ffn",),
    )
    evidence = (
        *index.common_evidence,
        f"layer_strip:layer_index={entry.layer_index}",
        f"layer_strip:attention_kind={entry.attention_kind}",
        f"layer_strip:mlp_kind={entry.mlp_kind}",
    )
    builder.add_node(
        "hidden_in",
        "Hidden In",
        "input",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "boundary:block_input"),
    )
    _add_activation_port(
        builder, index, "hidden_in", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    builder.add_node(
        "input_norm",
        "Input RMSNorm",
        "norm",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.RMS_NORM,
    )
    _add_activation_port(builder, index, "input_norm", "in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(builder, index, "input_norm", "out", "normalized", PortDirection.OUTPUT)
    builder.add_node(
        "attention",
        attention_label,
        attention_kind,
        group_key="root",
        subject_ids=attention_subjects,
        evidence=(
            *evidence,
            *(("coverage:dsa_internals_unknown",) if attention_opaque else ()),
        ),
        coverage=attention_coverage,
        opaque=attention_opaque,
        drilldown_view_id=attention_drilldown_view_id,
        decomposition_status=(
            GraphDecompositionStatus.OPAQUE
            if attention_opaque
            else (
                GraphDecompositionStatus.AVAILABLE
                if attention_drilldown_view_id is not None
                else GraphDecompositionStatus.NONE
            )
        ),
        metric_bindings=(
            _known_metric_binding("flops", "flops.attention", "flops.attention"),
            _known_metric_binding(
                "logical_bytes", "logical_bytes.attention", "logical_bytes.attention"
            ),
        )
        if not attention_opaque
        else (
            _unknown_metric_binding(
                "flops", "flops.attention", "Opaque attention internals are not attributable."
            ),
            _unknown_metric_binding(
                "logical_bytes",
                "logical_bytes.attention",
                "Opaque attention internals are not attributable.",
            ),
        ),
        attributes={
            "attention_kind": entry.attention_kind,
            "state_kind": entry.state_kind,
            "internals": "unknown" if attention_opaque else "collapsed_semantic",
        },
    )
    _add_activation_port(
        builder, index, "attention", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    _add_activation_port(builder, index, "attention", "context", "context", PortDirection.OUTPUT)
    builder.add_node(
        "residual_1",
        "Residual Add",
        "residual",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.ADD,
    )
    _add_activation_port(builder, index, "residual_1", "skip", "residual", PortDirection.INPUT)
    _add_activation_port(
        builder, index, "residual_1", "branch", "attention_out", PortDirection.INPUT
    )
    _add_activation_port(
        builder, index, "residual_1", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    builder.add_node(
        "post_norm",
        "Post-Attention RMSNorm",
        "norm",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.RMS_NORM,
    )
    _add_activation_port(builder, index, "post_norm", "in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(builder, index, "post_norm", "out", "normalized", PortDirection.OUTPUT)
    builder.add_node(
        "ffn",
        ffn_label,
        "ffn",
        group_key="root",
        subject_ids=ffn_subjects,
        evidence=evidence,
        drilldown_view_id=ffn_drilldown_view_id,
        decomposition_status=(
            GraphDecompositionStatus.AVAILABLE
            if ffn_drilldown_view_id is not None
            else GraphDecompositionStatus.NONE
        ),
        metric_bindings=(
            (
                _unknown_metric_binding("flops", "flops", "GLM executed FFN/MoE cost is Unknown."),
                _unknown_metric_binding(
                    "logical_bytes", "logical_bytes", "GLM executed FFN/MoE traffic is Unknown."
                ),
            )
            if index.result.adapter_name == "glm-moe-dsa"
            else (
                _known_metric_binding("flops", "flops.ffn", "flops.ffn"),
                _known_metric_binding("logical_bytes", "logical_bytes.ffn", "logical_bytes.ffn"),
            )
        ),
        attributes={
            "mlp_kind": entry.mlp_kind,
            "intermediate_size": index.result.inventory_config.get("intermediate_size"),
        },
    )
    _add_activation_port(builder, index, "ffn", "hidden_in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(builder, index, "ffn", "hidden", "ffn_output", PortDirection.OUTPUT)
    builder.add_node(
        "residual_2",
        "Residual Add",
        "residual",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.ADD,
    )
    _add_activation_port(builder, index, "residual_2", "skip", "residual", PortDirection.INPUT)
    _add_activation_port(builder, index, "residual_2", "branch", "ffn_output", PortDirection.INPUT)
    _add_activation_port(
        builder, index, "residual_2", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    builder.add_node(
        "hidden_out",
        "Hidden Out",
        "output",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "boundary:block_output"),
    )
    _add_activation_port(
        builder, index, "hidden_out", "hidden", "hidden_states", PortDirection.INPUT
    )
    for args in (
        ("input_to_norm", "hidden_in", "hidden", "input_norm", "in"),
        ("norm_to_attention", "input_norm", "out", "attention", "hidden_in"),
        ("input_skip", "hidden_in", "hidden", "residual_1", "skip"),
        ("attention_to_residual", "attention", "context", "residual_1", "branch"),
        ("residual_to_post_norm", "residual_1", "hidden", "post_norm", "in"),
        ("post_norm_to_ffn", "post_norm", "out", "ffn", "hidden_in"),
        ("residual_skip_2", "residual_1", "hidden", "residual_2", "skip"),
        ("ffn_to_residual", "ffn", "hidden", "residual_2", "branch"),
        ("residual_to_output", "residual_2", "hidden", "hidden_out", "hidden"),
    ):
        _add_data_edge(builder, index, *args)
    _add_state_rails(
        builder,
        index,
        group_key="root",
        layer_path=path,
        attention_node_key="attention",
    )


def _build_qwen_or_tiny_l1(
    index: _EvidenceIndex,
    view_ids: Mapping[str, str],
    *,
    view_key: str,
    entry: LayerStripEntry,
) -> GraphView:
    is_linear = entry.attention_kind == "linear_attention"
    attention_label = "Gated DeltaNet Linear Attention" if is_linear else "Full Attention (GQA)"
    builder = _ViewBuilder(
        view_id=view_ids[view_key],
        key=view_key,
        level=GraphViewLevel.BLOCK,
        label=f"{attention_label} representative",
        parent_view_id=view_ids["l0"],
        breadcrumb=[index.result.model_id, "Decoder", entry.label + f"{entry.layer_index}"],
        layer_index=entry.layer_index,
        metadata={
            "layout_direction": "RIGHT",
            "primary_flow": "acyclic",
            "representative_instance_path": entry.instance_path,
            "attention_kind": entry.attention_kind,
            "mlp_kind": entry.mlp_kind,
            "layer_strip_target": entry.layer_index,
        },
    )
    builder.add_group(
        "root",
        f"Layer {entry.layer_index}",
        "decoder_block",
        attributes={"layout": "layered", "state_rails": True},
    )
    _add_standard_l1_nodes(
        builder,
        index,
        entry=entry,
        attention_label=attention_label,
        attention_kind=entry.attention_kind,
        attention_opaque=False,
        attention_coverage=1.0,
        ffn_label="Dense Gated FFN",
        attention_drilldown_view_id=(
            view_ids.get("op_linear_attention") if is_linear else view_ids.get("op_full_attention")
        ),
        ffn_drilldown_view_id=(
            view_ids.get("op_linear_ffn") if is_linear else view_ids.get("op_full_ffn")
        ),
    )
    return builder.finish(
        cost_frontier_node_keys=(
            "input_norm",
            "attention",
            "residual_1",
            "post_norm",
            "ffn",
            "residual_2",
        )
    )


def _build_glm_l1(
    index: _EvidenceIndex,
    view_ids: Mapping[str, str],
    *,
    view_key: str,
    entry: LayerStripEntry,
) -> GraphView:
    sparse = entry.mlp_kind == "sparse"
    builder = _ViewBuilder(
        view_id=view_ids[view_key],
        key=view_key,
        level=GraphViewLevel.BLOCK,
        label=f"GLM {'Sparse DSA + MoE' if sparse else 'Dense DSA'} representative",
        parent_view_id=view_ids["l0"],
        breadcrumb=[index.result.model_id, "DSA Decoder", entry.label + f"{entry.layer_index}"],
        layer_index=entry.layer_index,
        metadata={
            "layout_direction": "RIGHT",
            "primary_flow": "acyclic",
            "representative_instance_path": entry.instance_path,
            "attention_kind": "dsa",
            "mlp_kind": entry.mlp_kind,
            "layer_strip_target": entry.layer_index,
            "dsa_internals": "opaque",
        },
    )
    builder.add_group(
        "root",
        f"Layer {entry.layer_index}",
        "decoder_block",
        attributes={"layout": "layered", "state_rails": True},
    )
    if not sparse:
        _add_standard_l1_nodes(
            builder,
            index,
            entry=entry,
            attention_label="DSA (internals Unknown)",
            attention_kind="dsa",
            attention_opaque=True,
            attention_coverage=0.0,
            ffn_label="Dense Gated FFN",
            ffn_drilldown_view_id=view_ids.get("op_glm_dense_ffn"),
        )
        return builder.finish(
            cost_frontier_node_keys=(
                "input_norm",
                "attention",
                "residual_1",
                "post_norm",
                "ffn",
                "residual_2",
            )
        )

    path = entry.instance_path
    pool_path = f"{path}.mlp.expert_pool"
    block_subjects = index.subjects(
        paths=(path,),
        definition_keys=(entry.definition_key,),
        include_semantics=False,
    )
    attention_subjects = index.subjects(
        paths=(path,),
        definition_keys=(entry.definition_key,),
        semantic_kinds=("dsa",),
    )
    evidence = (
        *index.common_evidence,
        f"layer_strip:layer_index={entry.layer_index}",
        "layer_strip:attention_kind=dsa",
        "layer_strip:mlp_kind=sparse",
    )
    router_dtype = _config_dtype(index.result.inventory_config.get("moe_router_dtype"))
    builder.add_node(
        "hidden_in",
        "Hidden In",
        "input",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "boundary:block_input"),
    )
    _add_activation_port(
        builder, index, "hidden_in", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    builder.add_node(
        "input_norm",
        "Input RMSNorm",
        "norm",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.RMS_NORM,
    )
    _add_activation_port(builder, index, "input_norm", "in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(builder, index, "input_norm", "out", "normalized", PortDirection.OUTPUT)
    builder.add_node(
        "attention",
        "DSA (internals Unknown)",
        "dsa",
        group_key="root",
        subject_ids=attention_subjects,
        evidence=(*evidence, "coverage:dsa_internals_unknown"),
        coverage=0.0,
        opaque=True,
        decomposition_status=GraphDecompositionStatus.OPAQUE,
        attributes={"attention_kind": "dsa", "state_kind": "kv_cache", "internals": "unknown"},
    )
    _add_activation_port(
        builder, index, "attention", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    _add_activation_port(builder, index, "attention", "context", "context", PortDirection.OUTPUT)
    builder.add_node(
        "residual_1",
        "Residual Add",
        "residual",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.ADD,
    )
    _add_activation_port(builder, index, "residual_1", "skip", "residual", PortDirection.INPUT)
    _add_activation_port(
        builder, index, "residual_1", "branch", "attention_out", PortDirection.INPUT
    )
    _add_activation_port(
        builder, index, "residual_1", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    builder.add_node(
        "post_norm",
        "Post-Attention RMSNorm",
        "norm",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.RMS_NORM,
    )
    _add_activation_port(builder, index, "post_norm", "in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(builder, index, "post_norm", "out", "normalized", PortDirection.OUTPUT)
    builder.add_node(
        "router",
        "Router GEMM",
        "gemm",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "config:num_experts_per_tok"),
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.GEMM,
        metric_bindings=(
            _unknown_metric_binding(
                "flops", "flops", "GLM executed MoE cost is Unknown without route evidence."
            ),
            _unknown_metric_binding(
                "logical_bytes",
                "logical_bytes",
                "GLM executed MoE traffic is Unknown without route evidence.",
            ),
        ),
        attributes={
            "routed_experts": index.result.metadata.get("routed_experts"),
            "top_k": index.result.metadata.get("top_k"),
            "runtime_route_known": False,
        },
    )
    _add_activation_port(
        builder, index, "router", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    builder.add_port(
        "router",
        "scores",
        "router_scores",
        PortDirection.OUTPUT,
        TensorRole.ACTIVATION,
        shape=["B", "T", int(index.result.metadata["routed_experts"])],
        dtype=router_dtype,
        evidence=(*evidence, "config:n_routed_experts", "config:moe_router_dtype"),
    )
    builder.add_node(
        "topk",
        f"TopK (k={index.result.metadata['top_k']})",
        "top_k",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "config:num_experts_per_tok", "runtime_route_known:false"),
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.TOP_K,
        metric_bindings=(
            _unknown_metric_binding(
                "flops", "flops", "TopK execution cost is Unknown at config-only capability."
            ),
            _unknown_metric_binding(
                "logical_bytes",
                "logical_bytes",
                "TopK traffic is Unknown at config-only capability.",
            ),
        ),
        attributes={
            "routed_experts": index.result.metadata.get("routed_experts"),
            "top_k": index.result.metadata.get("top_k"),
            "runtime_route_known": False,
            "selected_expert_ids": None,
            "routing_weight_values_known": False,
        },
    )
    builder.add_port(
        "topk",
        "scores",
        "router_scores",
        PortDirection.INPUT,
        TensorRole.ACTIVATION,
        shape=["B", "T", int(index.result.metadata["routed_experts"])],
        dtype=router_dtype,
        evidence=(*evidence, "config:n_routed_experts", "config:moe_router_dtype"),
    )
    builder.add_port(
        "topk",
        "routes",
        "expert_indices",
        PortDirection.OUTPUT,
        TensorRole.ROUTE,
        shape=["B", "T", int(index.result.metadata["top_k"])],
        dtype=DType.INT64,
        evidence=(*evidence, "config:num_experts_per_tok", "runtime_route_known:false"),
    )
    builder.add_port(
        "topk",
        "dispatch",
        "dispatch_control",
        PortDirection.OUTPUT,
        TensorRole.CONTROL,
        shape=None,
        dtype=DType.BOOL,
        evidence=(*evidence, "config:virtual_expert_pool"),
    )
    builder.add_port(
        "topk",
        "weights",
        "routing_weights",
        PortDirection.OUTPUT,
        TensorRole.ROUTE,
        shape=["B", "T", int(index.result.metadata["top_k"])],
        dtype=router_dtype,
        evidence=(
            *evidence,
            "config:num_experts_per_tok",
            "config:moe_router_dtype",
            "runtime_route_known:false",
        ),
    )
    pool_subjects = index.subjects(
        paths=(pool_path,),
        definition_keys=("expert_pool",),
        semantic_kinds=("expert_pool",),
    )
    builder.add_node(
        "expert_pool",
        (
            f"Expert Pool [{index.result.metadata['routed_experts']} total, "
            f"top-{index.result.metadata['top_k']} active]"
        ),
        "expert_pool",
        group_key="root",
        subject_ids=pool_subjects,
        evidence=(*evidence, "config:virtual_expert_pool", "experts_expanded:false"),
        drilldown_view_id=view_ids.get("op_glm_routed_expert_ffn"),
        decomposition_status=GraphDecompositionStatus.AVAILABLE,
        attributes={
            "virtual": True,
            "experts_expanded": False,
            "experts_materialized": 0,
            "routed_experts": index.result.metadata.get("routed_experts"),
            "top_k": index.result.metadata.get("top_k"),
            "runtime_route_known": False,
        },
    )
    _add_activation_port(
        builder, index, "expert_pool", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    builder.add_port(
        "expert_pool",
        "routes",
        "expert_indices",
        PortDirection.INPUT,
        TensorRole.ROUTE,
        shape=["B", "T", int(index.result.metadata["top_k"])],
        dtype=DType.INT64,
        evidence=(*evidence, "config:num_experts_per_tok"),
    )
    _add_activation_port(
        builder, index, "expert_pool", "hidden", "routed_output", PortDirection.OUTPUT
    )
    builder.add_node(
        "shared_expert",
        f"Shared Expert × {index.result.metadata.get('shared_experts', 0)}",
        "shared_expert",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "config:n_shared_experts"),
        drilldown_view_id=view_ids.get("op_glm_shared_expert_ffn"),
        decomposition_status=GraphDecompositionStatus.AVAILABLE,
        attributes={
            "shared_experts": index.result.metadata.get("shared_experts", 0),
            "runtime_route_known": False,
        },
    )
    _add_activation_port(
        builder, index, "shared_expert", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    _add_activation_port(
        builder, index, "shared_expert", "hidden", "shared_output", PortDirection.OUTPUT
    )
    builder.add_node(
        "combine",
        "MoE Combine",
        "moe_combine",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.REDUCE,
    )
    _add_activation_port(builder, index, "combine", "routed", "routed_output", PortDirection.INPUT)
    _add_activation_port(builder, index, "combine", "shared", "shared_output", PortDirection.INPUT)
    builder.add_port(
        "combine",
        "dispatch",
        "dispatch_control",
        PortDirection.INPUT,
        TensorRole.CONTROL,
        shape=None,
        dtype=DType.BOOL,
        evidence=(*evidence, "config:virtual_expert_pool"),
    )
    builder.add_port(
        "combine",
        "routing_weights",
        "routing_weights",
        PortDirection.INPUT,
        TensorRole.ROUTE,
        shape=["B", "T", int(index.result.metadata["top_k"])],
        dtype=router_dtype,
        evidence=(
            *evidence,
            "config:num_experts_per_tok",
            "config:moe_router_dtype",
            "runtime_route_known:false",
        ),
    )
    _add_activation_port(builder, index, "combine", "hidden", "moe_output", PortDirection.OUTPUT)
    builder.add_node(
        "residual_2",
        "Residual Add",
        "residual",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=GraphPrimitiveKind.ADD,
    )
    _add_activation_port(builder, index, "residual_2", "skip", "residual", PortDirection.INPUT)
    _add_activation_port(builder, index, "residual_2", "branch", "moe_output", PortDirection.INPUT)
    _add_activation_port(
        builder, index, "residual_2", "hidden", "hidden_states", PortDirection.OUTPUT
    )
    builder.add_node(
        "hidden_out",
        "Hidden Out",
        "output",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "boundary:block_output"),
    )
    _add_activation_port(
        builder, index, "hidden_out", "hidden", "hidden_states", PortDirection.INPUT
    )
    for args in (
        ("input_to_norm", "hidden_in", "hidden", "input_norm", "in", GraphEdgeKind.DATA),
        ("norm_to_attention", "input_norm", "out", "attention", "hidden_in", GraphEdgeKind.DATA),
        ("input_skip", "hidden_in", "hidden", "residual_1", "skip", GraphEdgeKind.DATA),
        (
            "attention_to_residual",
            "attention",
            "context",
            "residual_1",
            "branch",
            GraphEdgeKind.DATA,
        ),
        ("residual_to_post_norm", "residual_1", "hidden", "post_norm", "in", GraphEdgeKind.DATA),
        ("post_norm_to_router", "post_norm", "out", "router", "hidden_in", GraphEdgeKind.DATA),
        ("post_norm_to_pool", "post_norm", "out", "expert_pool", "hidden_in", GraphEdgeKind.DATA),
        (
            "post_norm_to_shared",
            "post_norm",
            "out",
            "shared_expert",
            "hidden_in",
            GraphEdgeKind.DATA,
        ),
        ("router_to_topk", "router", "scores", "topk", "scores", GraphEdgeKind.DATA),
        ("topk_to_pool", "topk", "routes", "expert_pool", "routes", GraphEdgeKind.ROUTE),
        ("topk_to_combine", "topk", "dispatch", "combine", "dispatch", GraphEdgeKind.CONTROL),
        (
            "topk_weights_to_combine",
            "topk",
            "weights",
            "combine",
            "routing_weights",
            GraphEdgeKind.ROUTE,
        ),
        ("pool_to_combine", "expert_pool", "hidden", "combine", "routed", GraphEdgeKind.DATA),
        ("shared_to_combine", "shared_expert", "hidden", "combine", "shared", GraphEdgeKind.DATA),
        ("residual_skip_2", "residual_1", "hidden", "residual_2", "skip", GraphEdgeKind.DATA),
        ("combine_to_residual", "combine", "hidden", "residual_2", "branch", GraphEdgeKind.DATA),
        ("residual_to_output", "residual_2", "hidden", "hidden_out", "hidden", GraphEdgeKind.DATA),
    ):
        _add_data_edge(builder, index, *args[:5], kind=args[5])
    _add_state_rails(
        builder,
        index,
        group_key="root",
        layer_path=path,
        attention_node_key="attention",
    )
    return builder.finish(
        cost_frontier_node_keys=(
            "input_norm",
            "attention",
            "residual_1",
            "post_norm",
            "router",
            "topk",
            "expert_pool",
            "shared_expert",
            "combine",
            "residual_2",
        )
    )


def _parent_node_and_ports(
    parent_view: GraphView, node_key: str
) -> Tuple[GraphNode, Dict[str, GraphPort]]:
    node = next(item for item in parent_view.nodes if item.key == node_key)
    port_ids = set(node.input_port_ids + node.output_port_ids)
    return node, {port.key: port for port in parent_view.ports if port.id in port_ids}


def _copy_parent_port(
    builder: _ViewBuilder,
    parent_port: GraphPort,
    *,
    child_node_key: str,
    child_port_key: str,
    direction: PortDirection,
) -> str:
    return builder.add_port(
        child_node_key,
        child_port_key,
        parent_port.name,
        direction,
        parent_port.role,
        shape=parent_port.shape,
        dtype=parent_port.dtype,
        tensor_spec_id=parent_port.tensor_spec_id,
        evidence=(*parent_port.evidence, "contract:decomposition-boundary"),
    )


def _add_operator_node(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    *,
    key: str,
    label: str,
    primitive: GraphPrimitiveKind,
    subject_ids: Sequence[str],
    metric_bindings: Sequence[GraphMetricBinding] = (),
    attributes: Optional[Mapping[str, Any]] = None,
) -> None:
    builder.add_node(
        key,
        label,
        primitive.value,
        group_key="root",
        subject_ids=subject_ids,
        evidence=(
            *index.common_evidence,
            "projection:semantic-primitive-v1",
            f"primitive:{primitive.value}",
        ),
        decomposition_status=GraphDecompositionStatus.PRIMITIVE,
        primitive_kind=primitive,
        metric_bindings=metric_bindings,
        attributes={"operator_family": primitive.value, **dict(attributes or {})},
    )


def _add_op_port(
    builder: _ViewBuilder,
    index: _EvidenceIndex,
    node_key: str,
    port_key: str,
    direction: PortDirection,
    shape: Optional[Sequence[ShapeDimension]],
    *,
    name: Optional[str] = None,
    role: TensorRole = TensorRole.ACTIVATION,
    dtype: Optional[DType] = None,
) -> str:
    return builder.add_port(
        node_key,
        port_key,
        name or port_key,
        direction,
        role,
        shape=shape,
        dtype=dtype or _activation_dtype(index.result),
        evidence=(*index.common_evidence, "interface:config-projected"),
    )


def _operator_builder(
    index: _EvidenceIndex,
    *,
    view_id: str,
    view_key: str,
    label: str,
    parent_view: GraphView,
    parent_node_key: str,
) -> _ViewBuilder:
    parent_node = next(node for node in parent_view.nodes if node.key == parent_node_key)
    return _ViewBuilder(
        view_id=view_id,
        key=view_key,
        level=GraphViewLevel.OPERATOR,
        label=label,
        parent_view_id=parent_view.id,
        breadcrumb=[*parent_view.breadcrumb, parent_node.label, "Operators"],
        layer_index=parent_view.layer_index,
        metadata={
            "layout_direction": "RIGHT",
            "primary_flow": "acyclic",
            "view_kind": "operator_decomposition",
            "selector_visible": False,
            "semantic_primitive_ontology": "v1",
            "target_model_forward": False,
        },
    )


def _qwen_component_bindings(component: str) -> Tuple[GraphMetricBinding, ...]:
    return (
        _known_metric_binding("flops", f"flops.attention.{component}", "flops.attention"),
        (
            _known_metric_binding(
                "logical_bytes",
                f"logical_bytes.attention.{component}",
                "logical_bytes.attention",
            )
            if component in {"q_proj", "k_proj", "v_proj", "o_proj"}
            else _unknown_metric_binding(
                "logical_bytes",
                "logical_bytes.attention",
                "Attention-core logical traffic is not split by primitive; "
                "it remains unattributed.",
            )
        ),
    )


def _excluded_attention_bindings(reason: str) -> Tuple[GraphMetricBinding, ...]:
    return (
        _excluded_metric_binding("flops", "flops.attention", reason),
        _excluded_metric_binding("logical_bytes", "logical_bytes.attention", reason),
    )


def _state_append_attention_bindings() -> Tuple[GraphMetricBinding, ...]:
    return (
        _excluded_metric_binding(
            "flops",
            "flops.attention",
            "State append FLOPs are outside the parent formula scope.",
        ),
        _unknown_metric_binding(
            "logical_bytes",
            "logical_bytes.attention",
            "KV state traffic remains in the parent logical metric but is not split "
            "between cache append primitives.",
        ),
    )


def _build_qwen_full_attention_ops(
    index: _EvidenceIndex,
    view_ids: Mapping[str, str],
    parent_view: GraphView,
) -> GraphView:
    parent_node, parent_ports = _parent_node_and_ports(parent_view, "attention")
    builder = _operator_builder(
        index,
        view_id=view_ids["op_full_attention"],
        view_key="op_full_attention",
        label="Full Attention · primitive operators",
        parent_view=parent_view,
        parent_node_key="attention",
    )
    builder.add_group("root", "Full Attention operators", "operator_graph")
    subjects = parent_node.subject_ids
    hidden = _hidden_shape(index.result)
    config = index.result.inventory_config
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    query_width = query_heads * head_dim
    output_gate = config.get("attn_output_gate")
    if not isinstance(output_gate, bool):
        raise ValueError("Qwen full-attention projection requires boolean attn_output_gate")
    q_multiplier = 2 if output_gate else 1

    for key, label in (("hidden_in", "Attention In"), ("state_in", "KV Cache In")):
        builder.add_node(
            key,
            label,
            "boundary",
            group_key="root",
            subject_ids=subjects,
            evidence=(*index.common_evidence, "contract:decomposition-boundary"),
        )
    for key, label in (("hidden_out", "Attention Out"), ("state_out", "KV Cache Out")):
        builder.add_node(
            key,
            label,
            "boundary",
            group_key="root",
            subject_ids=subjects,
            evidence=(*index.common_evidence, "contract:decomposition-boundary"),
        )
    _copy_parent_port(
        builder,
        parent_ports["attention.hidden_in"],
        child_node_key="hidden_in",
        child_port_key="hidden",
        direction=PortDirection.OUTPUT,
    )
    _copy_parent_port(
        builder,
        parent_ports["attention.context"],
        child_node_key="hidden_out",
        child_port_key="context",
        direction=PortDirection.INPUT,
    )
    state_pairs = (
        ("kv_cache_key", "key_cache"),
        ("kv_cache_value", "value_cache"),
    )
    for parent_prefix, child_key in state_pairs:
        _copy_parent_port(
            builder,
            parent_ports[f"attention.{parent_prefix}_in"],
            child_node_key="state_in",
            child_port_key=child_key,
            direction=PortDirection.OUTPUT,
        )
        _copy_parent_port(
            builder,
            parent_ports[f"attention.{parent_prefix}_out"],
            child_node_key="state_out",
            child_port_key=child_key,
            direction=PortDirection.INPUT,
        )

    primitive_specs = [
        (
            "q_proj",
            "Q + gate projection" if output_gate else "Q projection",
            GraphPrimitiveKind.GEMM,
            _qwen_component_bindings("q_proj"),
        ),
        ("k_proj", "K projection", GraphPrimitiveKind.GEMM, _qwen_component_bindings("k_proj")),
        ("v_proj", "V projection", GraphPrimitiveKind.GEMM, _qwen_component_bindings("v_proj")),
    ]
    if output_gate:
        primitive_specs.append(
            (
                "q_split",
                "Split Q / output gate",
                GraphPrimitiveKind.SPLIT,
                _excluded_attention_bindings("Split is excluded from the parent FLOPs formula."),
            )
        )
    primitive_specs.extend(
        [
            (
                "q_reshape",
                "Reshape Q heads",
                GraphPrimitiveKind.RESHAPE,
                _excluded_attention_bindings("Q head reshape is excluded from the parent formula."),
            ),
            (
                "q_transpose",
                "Transpose Q to heads-first",
                GraphPrimitiveKind.TRANSPOSE,
                _excluded_attention_bindings(
                    "Q layout transpose is excluded from the parent formula."
                ),
            ),
            (
                "k_reshape",
                "Reshape K heads",
                GraphPrimitiveKind.RESHAPE,
                _excluded_attention_bindings("K head reshape is excluded from the parent formula."),
            ),
            (
                "k_transpose",
                "Transpose K to heads-first",
                GraphPrimitiveKind.TRANSPOSE,
                _excluded_attention_bindings(
                    "K layout transpose is excluded from the parent formula."
                ),
            ),
            (
                "v_reshape",
                "Reshape V heads",
                GraphPrimitiveKind.RESHAPE,
                _excluded_attention_bindings("V head reshape is excluded from the parent formula."),
            ),
            (
                "v_transpose",
                "Transpose V to heads-first",
                GraphPrimitiveKind.TRANSPOSE,
                _excluded_attention_bindings(
                    "V layout transpose is excluded from the parent formula."
                ),
            ),
            (
                "q_norm",
                "Q RMSNorm",
                GraphPrimitiveKind.RMS_NORM,
                _excluded_attention_bindings(
                    "Q/K normalization is excluded from the parent formula."
                ),
            ),
            (
                "k_norm",
                "K RMSNorm",
                GraphPrimitiveKind.RMS_NORM,
                _excluded_attention_bindings(
                    "Q/K normalization is excluded from the parent formula."
                ),
            ),
            (
                "q_rope",
                "Q RoPE",
                GraphPrimitiveKind.ROPE,
                _excluded_attention_bindings("RoPE is excluded from the parent formula."),
            ),
            (
                "k_rope",
                "K RoPE",
                GraphPrimitiveKind.ROPE,
                _excluded_attention_bindings("RoPE is excluded from the parent formula."),
            ),
            (
                "k_append",
                "Append K cache",
                GraphPrimitiveKind.CONCAT,
                _state_append_attention_bindings(),
            ),
            (
                "v_append",
                "Append V cache",
                GraphPrimitiveKind.CONCAT,
                _state_append_attention_bindings(),
            ),
            (
                "k_gqa_expand",
                f"Broadcast K heads {kv_heads}→{query_heads}",
                GraphPrimitiveKind.BROADCAST,
                _excluded_attention_bindings(
                    "GQA head broadcast is a logical view operation outside the parent formula."
                ),
            ),
            (
                "v_gqa_expand",
                f"Broadcast V heads {kv_heads}→{query_heads}",
                GraphPrimitiveKind.BROADCAST,
                _excluded_attention_bindings(
                    "GQA head broadcast is a logical view operation outside the parent formula."
                ),
            ),
            (
                "qk_matmul",
                "Q × Kᵀ",
                GraphPrimitiveKind.MATMUL,
                _qwen_component_bindings("qk_matmul"),
            ),
            (
                "scale",
                "Attention scale",
                GraphPrimitiveKind.SCALE,
                _excluded_attention_bindings("Scale FLOPs are excluded from the parent formula."),
            ),
            (
                "causal_mask",
                "Causal mask",
                GraphPrimitiveKind.MASK,
                _excluded_attention_bindings("Mask FLOPs are excluded from the parent formula."),
            ),
            (
                "softmax",
                "Softmax",
                GraphPrimitiveKind.SOFTMAX,
                _excluded_attention_bindings("Softmax FLOPs are excluded from the parent formula."),
            ),
            (
                "pv_matmul",
                "P × V",
                GraphPrimitiveKind.MATMUL,
                _qwen_component_bindings("pv_matmul"),
            ),
            (
                "context_transpose",
                "Transpose context to tokens-first",
                GraphPrimitiveKind.TRANSPOSE,
                _excluded_attention_bindings(
                    "Context layout transpose is excluded from the parent formula."
                ),
            ),
            (
                "merge_heads",
                "Merge heads",
                GraphPrimitiveKind.RESHAPE,
                _excluded_attention_bindings("Head reshape is excluded from the parent formula."),
            ),
        ]
    )
    if output_gate:
        primitive_specs.extend(
            [
                (
                    "gate_silu",
                    "Output gate SiLU",
                    GraphPrimitiveKind.SILU,
                    _excluded_attention_bindings(
                        "Output-gate activation is excluded from the parent formula."
                    ),
                ),
                (
                    "gate_multiply",
                    "Apply output gate",
                    GraphPrimitiveKind.MULTIPLY,
                    _excluded_attention_bindings(
                        "Output-gate Multiply is excluded from the parent formula."
                    ),
                ),
            ]
        )
    primitive_specs.append(
        ("o_proj", "O projection", GraphPrimitiveKind.GEMM, _qwen_component_bindings("o_proj"))
    )
    for key, label, primitive, bindings in primitive_specs:
        _add_operator_node(
            builder,
            index,
            key=key,
            label=label,
            primitive=primitive,
            subject_ids=subjects,
            metric_bindings=bindings,
        )

    shapes = {
        "q_packed": ["B", "T", q_multiplier * query_width],
        "q_flat": ["B", "T", query_width],
        "q_tokens_first": ["B", "T", query_heads, head_dim],
        "q": ["B", query_heads, "T", head_dim],
        "gate": ["B", "T", query_width],
        "kv_flat": ["B", "T", kv_heads * head_dim],
        "kv_tokens_first": ["B", "T", kv_heads, head_dim],
        "kv": ["B", kv_heads, "T", head_dim],
        "gqa_cache": ["B", query_heads, "S_KV", head_dim],
        "scores": ["B", query_heads, "T", "S_KV"],
        "context": ["B", query_heads, "T", head_dim],
        "context_tokens_first": ["B", "T", query_heads, head_dim],
        "merged": ["B", "T", query_width],
    }
    for key in ("q_proj", "k_proj", "v_proj"):
        _add_op_port(builder, index, key, "hidden", PortDirection.INPUT, hidden)
    _add_op_port(builder, index, "q_proj", "projected", PortDirection.OUTPUT, shapes["q_packed"])
    _add_op_port(builder, index, "k_proj", "flat", PortDirection.OUTPUT, shapes["kv_flat"])
    _add_op_port(builder, index, "v_proj", "flat", PortDirection.OUTPUT, shapes["kv_flat"])
    if output_gate:
        _add_op_port(
            builder, index, "q_split", "projected", PortDirection.INPUT, shapes["q_packed"]
        )
        _add_op_port(builder, index, "q_split", "q", PortDirection.OUTPUT, shapes["q_flat"])
        _add_op_port(builder, index, "q_split", "gate", PortDirection.OUTPUT, shapes["gate"])
    _add_op_port(builder, index, "q_reshape", "flat", PortDirection.INPUT, shapes["q_flat"])
    _add_op_port(
        builder,
        index,
        "q_reshape",
        "heads",
        PortDirection.OUTPUT,
        shapes["q_tokens_first"],
    )
    _add_op_port(
        builder,
        index,
        "q_transpose",
        "tokens_first",
        PortDirection.INPUT,
        shapes["q_tokens_first"],
    )
    _add_op_port(builder, index, "q_transpose", "heads_first", PortDirection.OUTPUT, shapes["q"])
    for key in ("k", "v"):
        _add_op_port(
            builder, index, f"{key}_reshape", "flat", PortDirection.INPUT, shapes["kv_flat"]
        )
        _add_op_port(
            builder,
            index,
            f"{key}_reshape",
            "heads",
            PortDirection.OUTPUT,
            shapes["kv_tokens_first"],
        )
        _add_op_port(
            builder,
            index,
            f"{key}_transpose",
            "tokens_first",
            PortDirection.INPUT,
            shapes["kv_tokens_first"],
        )
        _add_op_port(
            builder, index, f"{key}_transpose", "heads_first", PortDirection.OUTPUT, shapes["kv"]
        )
    for key, port_name, shape in (
        ("q_norm", "q", shapes["q"]),
        ("k_norm", "k", shapes["kv"]),
        ("q_rope", "q", shapes["q"]),
        ("k_rope", "k", shapes["kv"]),
    ):
        _add_op_port(builder, index, key, "in", PortDirection.INPUT, shape, name=port_name)
        _add_op_port(builder, index, key, "out", PortDirection.OUTPUT, shape, name=port_name)
    for key, tensor_name in (("k_append", "key"), ("v_append", "value")):
        parent_state = parent_ports[f"attention.kv_cache_{tensor_name}_in"]
        _add_op_port(builder, index, key, "new", PortDirection.INPUT, shapes["kv"])
        builder.add_port(
            key,
            "cache_in",
            f"{tensor_name}_cache",
            PortDirection.INPUT,
            TensorRole.CACHE,
            shape=parent_state.shape,
            dtype=parent_state.dtype,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=parent_state.evidence,
        )
        builder.add_port(
            key,
            "cache_out",
            f"{tensor_name}_cache",
            PortDirection.OUTPUT,
            TensorRole.CACHE,
            shape=parent_state.shape,
            dtype=parent_state.dtype,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=parent_state.evidence,
        )
    for key, tensor_name in (("k_gqa_expand", "key"), ("v_gqa_expand", "value")):
        parent_state = parent_ports[f"attention.kv_cache_{tensor_name}_in"]
        builder.add_port(
            key,
            "kv_heads",
            f"{tensor_name}_cache_kv_heads",
            PortDirection.INPUT,
            TensorRole.CACHE,
            shape=parent_state.shape,
            dtype=parent_state.dtype,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=parent_state.evidence,
        )
        _add_op_port(
            builder,
            index,
            key,
            "query_heads",
            PortDirection.OUTPUT,
            shapes["gqa_cache"],
            name=f"{tensor_name}_cache_query_heads",
        )
    _add_op_port(builder, index, "qk_matmul", "q", PortDirection.INPUT, shapes["q"])
    _add_op_port(builder, index, "qk_matmul", "k", PortDirection.INPUT, shapes["gqa_cache"])
    _add_op_port(builder, index, "qk_matmul", "scores", PortDirection.OUTPUT, shapes["scores"])
    for key in ("scale", "causal_mask", "softmax"):
        _add_op_port(builder, index, key, "in", PortDirection.INPUT, shapes["scores"])
        _add_op_port(builder, index, key, "out", PortDirection.OUTPUT, shapes["scores"])
    _add_op_port(
        builder, index, "pv_matmul", "probabilities", PortDirection.INPUT, shapes["scores"]
    )
    _add_op_port(builder, index, "pv_matmul", "v", PortDirection.INPUT, shapes["gqa_cache"])
    _add_op_port(builder, index, "pv_matmul", "context", PortDirection.OUTPUT, shapes["context"])
    _add_op_port(
        builder, index, "context_transpose", "heads_first", PortDirection.INPUT, shapes["context"]
    )
    _add_op_port(
        builder,
        index,
        "context_transpose",
        "tokens_first",
        PortDirection.OUTPUT,
        shapes["context_tokens_first"],
    )
    _add_op_port(
        builder,
        index,
        "merge_heads",
        "context",
        PortDirection.INPUT,
        shapes["context_tokens_first"],
    )
    _add_op_port(builder, index, "merge_heads", "merged", PortDirection.OUTPUT, shapes["merged"])
    if output_gate:
        _add_op_port(builder, index, "gate_silu", "gate", PortDirection.INPUT, shapes["gate"])
        _add_op_port(builder, index, "gate_silu", "activated", PortDirection.OUTPUT, shapes["gate"])
        _add_op_port(
            builder, index, "gate_multiply", "context", PortDirection.INPUT, shapes["merged"]
        )
        _add_op_port(builder, index, "gate_multiply", "gate", PortDirection.INPUT, shapes["gate"])
        _add_op_port(builder, index, "gate_multiply", "out", PortDirection.OUTPUT, shapes["merged"])
    _add_op_port(builder, index, "o_proj", "in", PortDirection.INPUT, shapes["merged"])
    _add_op_port(builder, index, "o_proj", "out", PortDirection.OUTPUT, hidden)

    edges = [
        ("input_q", "hidden_in", "hidden", "q_proj", "hidden", GraphEdgeKind.DATA),
        ("input_k", "hidden_in", "hidden", "k_proj", "hidden", GraphEdgeKind.DATA),
        ("input_v", "hidden_in", "hidden", "v_proj", "hidden", GraphEdgeKind.DATA),
    ]
    if output_gate:
        edges.extend(
            [
                (
                    "q_split",
                    "q_proj",
                    "projected",
                    "q_split",
                    "projected",
                    GraphEdgeKind.DATA,
                ),
                ("q_reshape", "q_split", "q", "q_reshape", "flat", GraphEdgeKind.DATA),
            ]
        )
    else:
        edges.append(("q_reshape", "q_proj", "projected", "q_reshape", "flat", GraphEdgeKind.DATA))
    edges.extend(
        [
            (
                "q_transpose",
                "q_reshape",
                "heads",
                "q_transpose",
                "tokens_first",
                GraphEdgeKind.DATA,
            ),
            ("q_norm", "q_transpose", "heads_first", "q_norm", "in", GraphEdgeKind.DATA),
            ("q_rope", "q_norm", "out", "q_rope", "in", GraphEdgeKind.DATA),
            ("k_reshape", "k_proj", "flat", "k_reshape", "flat", GraphEdgeKind.DATA),
            (
                "k_transpose",
                "k_reshape",
                "heads",
                "k_transpose",
                "tokens_first",
                GraphEdgeKind.DATA,
            ),
            ("k_norm", "k_transpose", "heads_first", "k_norm", "in", GraphEdgeKind.DATA),
            ("k_rope", "k_norm", "out", "k_rope", "in", GraphEdgeKind.DATA),
            ("v_reshape", "v_proj", "flat", "v_reshape", "flat", GraphEdgeKind.DATA),
            (
                "v_transpose",
                "v_reshape",
                "heads",
                "v_transpose",
                "tokens_first",
                GraphEdgeKind.DATA,
            ),
            ("new_k", "k_rope", "out", "k_append", "new", GraphEdgeKind.DATA),
            ("new_v", "v_transpose", "heads_first", "v_append", "new", GraphEdgeKind.DATA),
            ("q_scores", "q_rope", "out", "qk_matmul", "q", GraphEdgeKind.DATA),
            ("scores_scale", "qk_matmul", "scores", "scale", "in", GraphEdgeKind.DATA),
            ("scale_mask", "scale", "out", "causal_mask", "in", GraphEdgeKind.DATA),
            ("mask_softmax", "causal_mask", "out", "softmax", "in", GraphEdgeKind.DATA),
            ("softmax_pv", "softmax", "out", "pv_matmul", "probabilities", GraphEdgeKind.DATA),
            (
                "pv_context_transpose",
                "pv_matmul",
                "context",
                "context_transpose",
                "heads_first",
                GraphEdgeKind.DATA,
            ),
            (
                "context_merge",
                "context_transpose",
                "tokens_first",
                "merge_heads",
                "context",
                GraphEdgeKind.DATA,
            ),
        ]
    )
    if output_gate:
        edges.extend(
            [
                ("gate_activation", "q_split", "gate", "gate_silu", "gate", GraphEdgeKind.DATA),
                (
                    "merge_gate",
                    "merge_heads",
                    "merged",
                    "gate_multiply",
                    "context",
                    GraphEdgeKind.DATA,
                ),
                (
                    "gate_value",
                    "gate_silu",
                    "activated",
                    "gate_multiply",
                    "gate",
                    GraphEdgeKind.DATA,
                ),
                ("gate_o", "gate_multiply", "out", "o_proj", "in", GraphEdgeKind.DATA),
            ]
        )
    else:
        edges.append(("merge_o", "merge_heads", "merged", "o_proj", "in", GraphEdgeKind.DATA))
    edges.append(("output", "o_proj", "out", "hidden_out", "context", GraphEdgeKind.DATA))
    for edge in edges:
        _add_data_edge(builder, index, *edge[:5], kind=edge[5])
    for tensor_name, append_key in (("key", "k_append"), ("value", "v_append")):
        parent_state = parent_ports[f"attention.kv_cache_{tensor_name}_in"]
        builder.add_edge(
            f"{tensor_name}_state_read",
            "state_in",
            f"{tensor_name}_cache",
            append_key,
            "cache_in",
            GraphEdgeKind.STATE_READ,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=(*parent_state.evidence, "edge:state_read"),
        )
        builder.add_edge(
            f"{tensor_name}_state_write",
            append_key,
            "cache_out",
            "state_out",
            f"{tensor_name}_cache",
            GraphEdgeKind.STATE_WRITE,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=(*parent_state.evidence, "edge:state_write"),
        )
    builder.add_edge(
        "key_cache_to_gqa",
        "k_append",
        "cache_out",
        "k_gqa_expand",
        "kv_heads",
        GraphEdgeKind.STATE_READ,
        tensor_spec_id=parent_ports["attention.kv_cache_key_in"].tensor_spec_id,
        evidence=(*parent_ports["attention.kv_cache_key_in"].evidence, "edge:state_read"),
    )
    builder.add_edge(
        "value_cache_to_gqa",
        "v_append",
        "cache_out",
        "v_gqa_expand",
        "kv_heads",
        GraphEdgeKind.STATE_READ,
        tensor_spec_id=parent_ports["attention.kv_cache_value_in"].tensor_spec_id,
        evidence=(*parent_ports["attention.kv_cache_value_in"].evidence, "edge:state_read"),
    )
    _add_data_edge(
        builder,
        index,
        "gqa_key_to_qk",
        "k_gqa_expand",
        "query_heads",
        "qk_matmul",
        "k",
    )
    _add_data_edge(
        builder,
        index,
        "gqa_value_to_pv",
        "v_gqa_expand",
        "query_heads",
        "pv_matmul",
        "v",
    )

    frontier = tuple(key for key, _, _, _ in primitive_specs)
    known_flops = [
        f"flops.attention.{name}"
        for name in ("q_proj", "k_proj", "v_proj", "qk_matmul", "pv_matmul", "o_proj")
    ]
    known_logical = [
        f"logical_bytes.attention.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    excluded = tuple(
        key
        for key in frontier
        if key not in {"q_proj", "k_proj", "v_proj", "qk_matmul", "pv_matmul", "o_proj"}
    )
    return _finish_decomposition_view(
        builder,
        parent_view=parent_view,
        parent_node_key="attention",
        boundary_specs=(
            ("attention.hidden_in", "hidden_in.hidden", GraphBoundaryKind.INPUT),
            ("attention.context", "hidden_out.context", GraphBoundaryKind.OUTPUT),
            ("attention.kv_cache_key_in", "state_in.key_cache", GraphBoundaryKind.STATE_READ),
            ("attention.kv_cache_key_out", "state_out.key_cache", GraphBoundaryKind.STATE_WRITE),
            ("attention.kv_cache_value_in", "state_in.value_cache", GraphBoundaryKind.STATE_READ),
            (
                "attention.kv_cache_value_out",
                "state_out.value_cache",
                GraphBoundaryKind.STATE_WRITE,
            ),
        ),
        cost_frontier_node_keys=frontier,
        reconciliation_specs=(
            {
                "dimension": "flops",
                "parent_metric_name": "flops.attention",
                "known_child_metric_names": known_flops,
                "excluded_node_keys": excluded,
                "status": GraphCostReconciliationStatus.COMPLETE,
                "reason": (
                    "Six GEMM/MatMul child formulas sum exactly to the parent formula; "
                    "excluded elementwise/normalization operators are outside that "
                    "formula scope."
                ),
            },
            {
                "dimension": "logical_bytes",
                "parent_metric_name": "logical_bytes.attention",
                "known_child_metric_names": known_logical,
                "unattributed_node_keys": (
                    "k_append",
                    "v_append",
                    "qk_matmul",
                    "pv_matmul",
                ),
                "excluded_node_keys": tuple(
                    key for key in excluded if key not in {"k_append", "v_append"}
                ),
                "status": GraphCostReconciliationStatus.PARTIAL,
                "reason": (
                    "Projection traffic is attributable; attention-core and state "
                    "traffic remain an explicit parent-minus-known-child remainder, "
                    "never zero-filled."
                ),
            },
        ),
    )


def _build_ffn_ops(
    index: _EvidenceIndex,
    *,
    view_id: str,
    view_key: str,
    label: str,
    parent_view: GraphView,
    parent_node_key: str,
    intermediate_size: int,
    routed: bool = False,
) -> GraphView:
    parent_node, parent_ports = _parent_node_and_ports(parent_view, parent_node_key)
    builder = _operator_builder(
        index,
        view_id=view_id,
        view_key=view_key,
        label=label,
        parent_view=parent_view,
        parent_node_key=parent_node_key,
    )
    builder.add_group("root", label, "operator_graph")
    subjects = parent_node.subject_ids
    is_qwen = index.result.adapter_name == "qwen3-5"
    parent_flops = "flops.ffn" if is_qwen else "flops"
    parent_logical = "logical_bytes.ffn" if is_qwen else "logical_bytes"

    builder.add_node(
        "hidden_in",
        "FFN In",
        "boundary",
        group_key="root",
        subject_ids=subjects,
        evidence=(*index.common_evidence, "contract:decomposition-boundary"),
    )
    builder.add_node(
        "hidden_out",
        "FFN Out",
        "boundary",
        group_key="root",
        subject_ids=subjects,
        evidence=(*index.common_evidence, "contract:decomposition-boundary"),
    )
    input_key = f"{parent_node_key}.hidden_in"
    output_key = f"{parent_node_key}.hidden"
    _copy_parent_port(
        builder,
        parent_ports[input_key],
        child_node_key="hidden_in",
        child_port_key="hidden",
        direction=PortDirection.OUTPUT,
    )
    _copy_parent_port(
        builder,
        parent_ports[output_key],
        child_node_key="hidden_out",
        child_port_key="hidden",
        direction=PortDirection.INPUT,
    )
    boundary_specs: List[Tuple[str, str, GraphBoundaryKind]] = [
        (input_key, "hidden_in.hidden", GraphBoundaryKind.INPUT),
        (output_key, "hidden_out.hidden", GraphBoundaryKind.OUTPUT),
    ]
    source_node, source_port = "hidden_in", "hidden"
    compute_keys: List[str] = []
    if routed:
        builder.add_node(
            "route_in",
            "Static TopK route",
            "boundary",
            group_key="root",
            subject_ids=subjects,
            evidence=(
                *index.common_evidence,
                "contract:decomposition-boundary",
                "runtime_route_known:false",
            ),
        )
        route_key = f"{parent_node_key}.routes"
        _copy_parent_port(
            builder,
            parent_ports[route_key],
            child_node_key="route_in",
            child_port_key="routes",
            direction=PortDirection.OUTPUT,
        )
        boundary_specs.append((route_key, "route_in.routes", GraphBoundaryKind.INPUT))
        unknown_route = (
            _unknown_metric_binding(
                "flops", parent_flops, "Runtime token-to-expert assignment is Unknown."
            ),
            _unknown_metric_binding(
                "logical_bytes",
                parent_logical,
                "Runtime token-to-expert dispatch traffic is Unknown.",
            ),
        )
        _add_operator_node(
            builder,
            index,
            key="gather",
            label="Gather selected expert inputs",
            primitive=GraphPrimitiveKind.GATHER,
            subject_ids=subjects,
            metric_bindings=unknown_route,
            attributes={"runtime_route_known": False},
        )
        _add_operator_node(
            builder,
            index,
            key="scatter",
            label="Scatter expert outputs",
            primitive=GraphPrimitiveKind.SCATTER,
            subject_ids=subjects,
            metric_bindings=unknown_route,
            attributes={"runtime_route_known": False},
        )
        hidden_shape = parent_ports[input_key].shape
        selected_shape = ["B", "T_selected", hidden_shape[-1] if hidden_shape else "H"]
        _add_op_port(builder, index, "gather", "hidden", PortDirection.INPUT, hidden_shape)
        route_parent = parent_ports[route_key]
        builder.add_port(
            "gather",
            "routes",
            route_parent.name,
            PortDirection.INPUT,
            TensorRole.ROUTE,
            shape=route_parent.shape,
            dtype=route_parent.dtype,
            tensor_spec_id=route_parent.tensor_spec_id,
            evidence=route_parent.evidence,
        )
        _add_op_port(builder, index, "gather", "selected", PortDirection.OUTPUT, selected_shape)
        _add_op_port(builder, index, "scatter", "selected", PortDirection.INPUT, selected_shape)
        _add_op_port(builder, index, "scatter", "hidden", PortDirection.OUTPUT, hidden_shape)
        _add_data_edge(builder, index, "route_hidden", "hidden_in", "hidden", "gather", "hidden")
        _add_data_edge(
            builder,
            index,
            "route_indices",
            "route_in",
            "routes",
            "gather",
            "routes",
            kind=GraphEdgeKind.ROUTE,
        )
        source_node, source_port = "gather", "selected"
        compute_keys.extend(("gather", "scatter"))

    for key, label_value, primitive in (
        ("gate_proj", "Gate GEMM", GraphPrimitiveKind.GEMM),
        ("up_proj", "Up GEMM", GraphPrimitiveKind.GEMM),
        ("activation", "SiLU", GraphPrimitiveKind.SILU),
        ("multiply", "Gate × Up", GraphPrimitiveKind.MULTIPLY),
        ("down_proj", "Down GEMM", GraphPrimitiveKind.GEMM),
    ):
        if is_qwen and key in {"gate_proj", "up_proj", "down_proj"}:
            bindings = (
                _known_metric_binding("flops", f"flops.ffn.{key}", "flops.ffn"),
                _known_metric_binding(
                    "logical_bytes", f"logical_bytes.ffn.{key}", "logical_bytes.ffn"
                ),
            )
        elif is_qwen:
            bindings = (
                _excluded_metric_binding(
                    "flops",
                    "flops.ffn",
                    "Activation/Multiply FLOPs are outside the parent formula.",
                ),
                _excluded_metric_binding(
                    "logical_bytes",
                    "logical_bytes.ffn",
                    "Activation/Multiply traffic is outside the parent logical formula.",
                ),
            )
        else:
            bindings = (
                _unknown_metric_binding(
                    "flops", parent_flops, "GLM executed FFN/MoE cost is Unknown."
                ),
                _unknown_metric_binding(
                    "logical_bytes",
                    parent_logical,
                    "GLM executed FFN/MoE traffic is Unknown.",
                ),
            )
        _add_operator_node(
            builder,
            index,
            key=key,
            label=label_value,
            primitive=primitive,
            subject_ids=subjects,
            metric_bindings=bindings,
            attributes={"runtime_route_known": False} if routed else None,
        )
        compute_keys.append(key)

    input_shape = (
        ["B", "T_selected", parent_ports[input_key].shape[-1]]
        if routed and parent_ports[input_key].shape
        else parent_ports[input_key].shape
    )
    intermediate_shape = [*(input_shape[:-1] if input_shape else ["B", "T"]), intermediate_size]
    for key in ("gate_proj", "up_proj"):
        _add_op_port(builder, index, key, "in", PortDirection.INPUT, input_shape)
        _add_op_port(builder, index, key, "out", PortDirection.OUTPUT, intermediate_shape)
    _add_op_port(builder, index, "activation", "in", PortDirection.INPUT, intermediate_shape)
    _add_op_port(builder, index, "activation", "out", PortDirection.OUTPUT, intermediate_shape)
    _add_op_port(builder, index, "multiply", "gate", PortDirection.INPUT, intermediate_shape)
    _add_op_port(builder, index, "multiply", "up", PortDirection.INPUT, intermediate_shape)
    _add_op_port(builder, index, "multiply", "out", PortDirection.OUTPUT, intermediate_shape)
    _add_op_port(builder, index, "down_proj", "in", PortDirection.INPUT, intermediate_shape)
    _add_op_port(builder, index, "down_proj", "out", PortDirection.OUTPUT, input_shape)
    for key, source, source_key, target, target_key in (
        ("input_gate", source_node, source_port, "gate_proj", "in"),
        ("input_up", source_node, source_port, "up_proj", "in"),
        ("gate_activation", "gate_proj", "out", "activation", "in"),
        ("activation_multiply", "activation", "out", "multiply", "gate"),
        ("up_multiply", "up_proj", "out", "multiply", "up"),
        ("multiply_down", "multiply", "out", "down_proj", "in"),
    ):
        _add_data_edge(builder, index, key, source, source_key, target, target_key)
    if routed:
        _add_data_edge(builder, index, "down_scatter", "down_proj", "out", "scatter", "selected")
        _add_data_edge(
            builder, index, "scatter_output", "scatter", "hidden", "hidden_out", "hidden"
        )
    else:
        _add_data_edge(builder, index, "ffn_output", "down_proj", "out", "hidden_out", "hidden")

    if is_qwen:
        reconciliation_specs = (
            {
                "dimension": "flops",
                "parent_metric_name": parent_flops,
                "known_child_metric_names": [
                    "flops.ffn.gate_proj",
                    "flops.ffn.up_proj",
                    "flops.ffn.down_proj",
                ],
                "excluded_node_keys": ("activation", "multiply"),
                "status": GraphCostReconciliationStatus.COMPLETE,
                "reason": (
                    "The three GEMM child formulas sum exactly to the parent FFN FLOPs "
                    "formula; activation and Multiply are explicitly outside its scope."
                ),
            },
            {
                "dimension": "logical_bytes",
                "parent_metric_name": parent_logical,
                "known_child_metric_names": [
                    "logical_bytes.ffn.gate_proj",
                    "logical_bytes.ffn.up_proj",
                    "logical_bytes.ffn.down_proj",
                ],
                "excluded_node_keys": ("activation", "multiply"),
                "status": GraphCostReconciliationStatus.COMPLETE,
                "reason": (
                    "Three matrix logical-traffic metrics sum to the parent FFN logical "
                    "metric; elementwise traffic is explicitly outside that formula scope."
                ),
            },
        )
    else:
        reconciliation_specs = (
            {
                "dimension": "flops",
                "parent_metric_name": parent_flops,
                "unattributed_node_keys": tuple(compute_keys),
                "status": GraphCostReconciliationStatus.UNKNOWN,
                "reason": (
                    "GLM executed FFN/MoE FLOPs remain Unknown without runtime route evidence."
                ),
            },
            {
                "dimension": "logical_bytes",
                "parent_metric_name": parent_logical,
                "unattributed_node_keys": tuple(compute_keys),
                "status": GraphCostReconciliationStatus.UNKNOWN,
                "reason": (
                    "GLM executed FFN/MoE traffic remains Unknown without runtime route evidence."
                ),
            },
        )
    return _finish_decomposition_view(
        builder,
        parent_view=parent_view,
        parent_node_key=parent_node_key,
        boundary_specs=boundary_specs,
        cost_frontier_node_keys=tuple(compute_keys),
        reconciliation_specs=reconciliation_specs,
    )


def _build_qwen_linear_attention_ops(
    index: _EvidenceIndex,
    view_ids: Mapping[str, str],
    parent_view: GraphView,
) -> GraphView:
    parent_node, parent_ports = _parent_node_and_ports(parent_view, "attention")
    builder = _operator_builder(
        index,
        view_id=view_ids["op_linear_attention"],
        view_key="op_linear_attention",
        label="Gated DeltaNet · state-aware operators",
        parent_view=parent_view,
        parent_node_key="attention",
    )
    builder.add_group("root", "Linear Attention operators", "operator_graph")
    subjects = parent_node.subject_ids
    hidden = _hidden_shape(index.result)
    config = index.result.inventory_config
    key_width = int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
    value_width = int(config["linear_num_value_heads"]) * int(config["linear_value_head_dim"])
    conv_width = 2 * key_width + value_width
    unknown_bindings = (
        _unknown_metric_binding(
            "flops",
            "flops.attention",
            "Linear-attention aggregate cost is not yet split into exact primitive metrics.",
        ),
        _unknown_metric_binding(
            "logical_bytes",
            "logical_bytes.attention",
            "Linear-attention aggregate traffic is not yet split into exact primitive metrics.",
        ),
    )
    for key, label in (
        ("hidden_in", "Attention In"),
        ("state_in", "Recurrent State In"),
        ("hidden_out", "Attention Out"),
        ("state_out", "Recurrent State Out"),
    ):
        builder.add_node(
            key,
            label,
            "boundary",
            group_key="root",
            subject_ids=subjects,
            evidence=(*index.common_evidence, "contract:decomposition-boundary"),
        )
    _copy_parent_port(
        builder,
        parent_ports["attention.hidden_in"],
        child_node_key="hidden_in",
        child_port_key="hidden",
        direction=PortDirection.OUTPUT,
    )
    _copy_parent_port(
        builder,
        parent_ports["attention.context"],
        child_node_key="hidden_out",
        child_port_key="context",
        direction=PortDirection.INPUT,
    )
    for state_name, child_key in (
        ("recurrent_state_conv", "conv_state"),
        ("recurrent_state_delta", "delta_state"),
    ):
        _copy_parent_port(
            builder,
            parent_ports[f"attention.{state_name}_in"],
            child_node_key="state_in",
            child_port_key=child_key,
            direction=PortDirection.OUTPUT,
        )
        _copy_parent_port(
            builder,
            parent_ports[f"attention.{state_name}_out"],
            child_node_key="state_out",
            child_port_key=child_key,
            direction=PortDirection.INPUT,
        )
    for key, label, primitive in (
        ("qkv_proj", "Q/K/V projection GEMM", GraphPrimitiveKind.GEMM),
        ("z_proj", "Output gate GEMM", GraphPrimitiveKind.GEMM),
        ("b_proj", "Beta projection GEMM", GraphPrimitiveKind.GEMM),
        ("a_proj", "Alpha projection GEMM", GraphPrimitiveKind.GEMM),
        ("depthwise_conv", "Depthwise Conv1D", GraphPrimitiveKind.CONVOLUTION),
        ("gate_silu", "Output gate SiLU", GraphPrimitiveKind.SILU),
        ("gate_multiply", "Apply output gate", GraphPrimitiveKind.MULTIPLY),
        ("out_proj", "Output projection GEMM", GraphPrimitiveKind.GEMM),
    ):
        _add_operator_node(
            builder,
            index,
            key=key,
            label=label,
            primitive=primitive,
            subject_ids=subjects,
            metric_bindings=unknown_bindings,
        )
    builder.add_node(
        "delta_core",
        "Gated Delta rule core (opaque)",
        "opaque",
        group_key="root",
        subject_ids=subjects,
        evidence=(
            *index.common_evidence,
            "coverage:linear_attention_core_not_decomposed",
            "unknown-is-not-zero",
        ),
        coverage=0.0,
        opaque=True,
        decomposition_status=GraphDecompositionStatus.OPAQUE,
        metric_bindings=unknown_bindings,
        attributes={"internals": "unknown", "state_kind": "recurrent_state"},
    )
    for key in ("qkv_proj", "z_proj", "b_proj", "a_proj"):
        _add_op_port(builder, index, key, "hidden", PortDirection.INPUT, hidden)
    conv_shape = ["B", "T", conv_width]
    gate_shape = ["B", "T", value_width]
    scalar_heads = ["B", "T", int(config["linear_num_value_heads"])]
    _add_op_port(builder, index, "qkv_proj", "qkv", PortDirection.OUTPUT, conv_shape)
    _add_op_port(builder, index, "z_proj", "gate", PortDirection.OUTPUT, gate_shape)
    _add_op_port(builder, index, "b_proj", "beta", PortDirection.OUTPUT, scalar_heads)
    _add_op_port(builder, index, "a_proj", "alpha", PortDirection.OUTPUT, scalar_heads)
    _add_op_port(builder, index, "depthwise_conv", "qkv", PortDirection.INPUT, conv_shape)
    _add_op_port(builder, index, "depthwise_conv", "filtered", PortDirection.OUTPUT, conv_shape)
    conv_parent = parent_ports["attention.recurrent_state_conv_in"]
    for direction, port_key in (
        (PortDirection.INPUT, "state_in"),
        (PortDirection.OUTPUT, "state_out"),
    ):
        builder.add_port(
            "depthwise_conv",
            port_key,
            "conv_state",
            direction,
            TensorRole.STATE,
            shape=conv_parent.shape,
            dtype=conv_parent.dtype,
            tensor_spec_id=conv_parent.tensor_spec_id,
            evidence=conv_parent.evidence,
        )
    for port_key, shape in (
        ("filtered", conv_shape),
        ("beta", scalar_heads),
        ("alpha", scalar_heads),
    ):
        _add_op_port(builder, index, "delta_core", port_key, PortDirection.INPUT, shape)
    delta_parent = parent_ports["attention.recurrent_state_delta_in"]
    builder.add_port(
        "delta_core",
        "state_in",
        "delta_state",
        PortDirection.INPUT,
        TensorRole.STATE,
        shape=delta_parent.shape,
        dtype=delta_parent.dtype,
        tensor_spec_id=delta_parent.tensor_spec_id,
        evidence=delta_parent.evidence,
    )
    _add_op_port(builder, index, "delta_core", "context", PortDirection.OUTPUT, gate_shape)
    builder.add_port(
        "delta_core",
        "state_out",
        "delta_state",
        PortDirection.OUTPUT,
        TensorRole.STATE,
        shape=delta_parent.shape,
        dtype=delta_parent.dtype,
        tensor_spec_id=delta_parent.tensor_spec_id,
        evidence=delta_parent.evidence,
    )
    _add_op_port(builder, index, "gate_silu", "gate", PortDirection.INPUT, gate_shape)
    _add_op_port(builder, index, "gate_silu", "activated", PortDirection.OUTPUT, gate_shape)
    _add_op_port(builder, index, "gate_multiply", "context", PortDirection.INPUT, gate_shape)
    _add_op_port(builder, index, "gate_multiply", "gate", PortDirection.INPUT, gate_shape)
    _add_op_port(builder, index, "gate_multiply", "out", PortDirection.OUTPUT, gate_shape)
    _add_op_port(builder, index, "out_proj", "in", PortDirection.INPUT, gate_shape)
    _add_op_port(builder, index, "out_proj", "out", PortDirection.OUTPUT, hidden)
    for key in ("qkv_proj", "z_proj", "b_proj", "a_proj"):
        _add_data_edge(builder, index, f"input_{key}", "hidden_in", "hidden", key, "hidden")
    for edge in (
        ("qkv_conv", "qkv_proj", "qkv", "depthwise_conv", "qkv"),
        ("conv_core", "depthwise_conv", "filtered", "delta_core", "filtered"),
        ("beta_core", "b_proj", "beta", "delta_core", "beta"),
        ("alpha_core", "a_proj", "alpha", "delta_core", "alpha"),
        ("z_gate", "z_proj", "gate", "gate_silu", "gate"),
        ("core_gate", "delta_core", "context", "gate_multiply", "context"),
        ("gate_value", "gate_silu", "activated", "gate_multiply", "gate"),
        ("gate_out", "gate_multiply", "out", "out_proj", "in"),
        ("output", "out_proj", "out", "hidden_out", "context"),
    ):
        _add_data_edge(builder, index, *edge)
    for state_name, op_key, parent_key, child_key in (
        ("conv", "depthwise_conv", "recurrent_state_conv", "conv_state"),
        ("delta", "delta_core", "recurrent_state_delta", "delta_state"),
    ):
        parent_state = parent_ports[f"attention.{parent_key}_in"]
        builder.add_edge(
            f"{state_name}_state_read",
            "state_in",
            child_key,
            op_key,
            "state_in",
            GraphEdgeKind.STATE_READ,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=(*parent_state.evidence, "edge:state_read"),
        )
        builder.add_edge(
            f"{state_name}_state_write",
            op_key,
            "state_out",
            "state_out",
            child_key,
            GraphEdgeKind.STATE_WRITE,
            tensor_spec_id=parent_state.tensor_spec_id,
            evidence=(*parent_state.evidence, "edge:state_write"),
        )
    frontier = (
        "qkv_proj",
        "z_proj",
        "b_proj",
        "a_proj",
        "depthwise_conv",
        "delta_core",
        "gate_silu",
        "gate_multiply",
        "out_proj",
    )
    return _finish_decomposition_view(
        builder,
        parent_view=parent_view,
        parent_node_key="attention",
        boundary_specs=(
            ("attention.hidden_in", "hidden_in.hidden", GraphBoundaryKind.INPUT),
            ("attention.context", "hidden_out.context", GraphBoundaryKind.OUTPUT),
            (
                "attention.recurrent_state_conv_in",
                "state_in.conv_state",
                GraphBoundaryKind.STATE_READ,
            ),
            (
                "attention.recurrent_state_conv_out",
                "state_out.conv_state",
                GraphBoundaryKind.STATE_WRITE,
            ),
            (
                "attention.recurrent_state_delta_in",
                "state_in.delta_state",
                GraphBoundaryKind.STATE_READ,
            ),
            (
                "attention.recurrent_state_delta_out",
                "state_out.delta_state",
                GraphBoundaryKind.STATE_WRITE,
            ),
        ),
        cost_frontier_node_keys=frontier,
        reconciliation_specs=(
            {
                "dimension": "flops",
                "parent_metric_name": "flops.attention",
                "unattributed_node_keys": frontier,
                "status": GraphCostReconciliationStatus.PARTIAL,
                "reason": (
                    "The aggregate linear-attention estimate is preserved, but exact "
                    "primitive attribution is not yet frozen; no child is zero-filled."
                ),
            },
            {
                "dimension": "logical_bytes",
                "parent_metric_name": "logical_bytes.attention",
                "unattributed_node_keys": frontier,
                "status": GraphCostReconciliationStatus.PARTIAL,
                "reason": (
                    "Aggregate logical traffic is preserved while primitive traffic "
                    "remains unattributed."
                ),
            },
        ),
    )


def build_graph_view_document(
    adapter_result: AdapterResult,
    model_map: Optional[ModelMap] = None,
) -> GraphViewDocument:
    """Project config evidence into deterministic L0 and representative L1 DAGs.

    This function performs no model import, construction, weight loading, export, or
    forward execution.  When ``model_map`` is omitted it builds only the existing
    config-first Model Map inventory from ``adapter_result``.
    """

    if adapter_result.metadata.get("weights_loaded") is not False:
        raise ValueError("GraphView requires explicit weights_loaded=false provenance")
    if adapter_result.metadata.get("remote_code_executed") is not False:
        raise ValueError("GraphView requires explicit remote_code_executed=false provenance")
    source_map = model_map or adapter_result.to_model_map()
    if source_map.model.name != adapter_result.model_id:
        raise ValueError("Model Map model name does not match AdapterResult")
    if source_map.model.revision != adapter_result.revision:
        raise ValueError("Model Map revision does not match AdapterResult")

    index = _EvidenceIndex.build(adapter_result, source_map)
    view_ids = _view_ids(adapter_result)
    views: List[GraphView] = []
    if adapter_result.adapter_name == "qwen3-5":
        views.append(_build_qwen_l0(index, view_ids))
        if "l1_linear_attention" in view_ids:
            linear_entry = _representative(adapter_result, attention="linear_attention")
            linear_view = _build_qwen_or_tiny_l1(
                index,
                view_ids,
                view_key="l1_linear_attention",
                entry=linear_entry,
            )
            views.extend(
                (
                    linear_view,
                    _build_qwen_linear_attention_ops(index, view_ids, linear_view),
                    _build_ffn_ops(
                        index,
                        view_id=view_ids["op_linear_ffn"],
                        view_key="op_linear_ffn",
                        label="Dense Gated FFN · primitive operators",
                        parent_view=linear_view,
                        parent_node_key="ffn",
                        intermediate_size=int(adapter_result.inventory_config["intermediate_size"]),
                    ),
                )
            )
        if "l1_full_attention" in view_ids:
            full_entry = _representative(adapter_result, attention="full_attention")
            full_view = _build_qwen_or_tiny_l1(
                index,
                view_ids,
                view_key="l1_full_attention",
                entry=full_entry,
            )
            views.extend(
                (
                    full_view,
                    _build_qwen_full_attention_ops(index, view_ids, full_view),
                    _build_ffn_ops(
                        index,
                        view_id=view_ids["op_full_ffn"],
                        view_key="op_full_ffn",
                        label="Dense Gated FFN · primitive operators",
                        parent_view=full_view,
                        parent_node_key="ffn",
                        intermediate_size=int(adapter_result.inventory_config["intermediate_size"]),
                    ),
                )
            )
    elif adapter_result.adapter_name == "glm-moe-dsa":
        views.append(_build_glm_l0(index, view_ids))
        if "l1_dense_dsa" in view_ids:
            dense_view = _build_glm_l1(
                index,
                view_ids,
                view_key="l1_dense_dsa",
                entry=_representative(adapter_result, mlp="dense"),
            )
            views.extend(
                (
                    dense_view,
                    _build_ffn_ops(
                        index,
                        view_id=view_ids["op_glm_dense_ffn"],
                        view_key="op_glm_dense_ffn",
                        label="GLM Dense FFN · primitive operators",
                        parent_view=dense_view,
                        parent_node_key="ffn",
                        intermediate_size=int(adapter_result.inventory_config["intermediate_size"]),
                    ),
                )
            )
        if "l1_sparse_dsa_moe" in view_ids:
            sparse_view = _build_glm_l1(
                index,
                view_ids,
                view_key="l1_sparse_dsa_moe",
                entry=_representative(adapter_result, mlp="sparse"),
            )
            views.extend(
                (
                    sparse_view,
                    _build_ffn_ops(
                        index,
                        view_id=view_ids["op_glm_routed_expert_ffn"],
                        view_key="op_glm_routed_expert_ffn",
                        label="Symbolic routed Expert FFN · primitive operators",
                        parent_view=sparse_view,
                        parent_node_key="expert_pool",
                        intermediate_size=int(
                            adapter_result.inventory_config["moe_intermediate_size"]
                        ),
                        routed=True,
                    ),
                    _build_ffn_ops(
                        index,
                        view_id=view_ids["op_glm_shared_expert_ffn"],
                        view_key="op_glm_shared_expert_ffn",
                        label="Shared Expert FFN · primitive operators",
                        parent_view=sparse_view,
                        parent_node_key="shared_expert",
                        intermediate_size=int(
                            adapter_result.inventory_config["moe_intermediate_size"]
                        ),
                    ),
                )
            )
    elif adapter_result.adapter_name == "tiny-dense":
        views.append(_build_tiny_l0(index, view_ids))
        views.append(
            _build_qwen_or_tiny_l1(
                index,
                view_ids,
                view_key="l1_dense",
                entry=_representative(adapter_result, attention="full_attention", mlp="dense"),
            )
        )
    else:
        raise ValueError(
            f"GraphView projection does not support adapter {adapter_result.adapter_name!r}"
        )

    provenance = GraphViewProvenance(
        source_artifact_ids=list(source_map.model.source_artifact_ids),
        evidence=index.common_evidence,
    )
    source_model_map_id = index.model_map_id
    document_payload = {
        "adapter_name": adapter_result.adapter_name,
        "model_id": adapter_result.model_id,
        "projection_version": _PROJECTION_VERSION,
        "revision": adapter_result.revision,
        "source_model_map_id": source_model_map_id,
        "views": [view.model_dump(mode="json") for view in views],
    }
    return GraphViewDocument(
        id=deterministic_id("gdoc", document_payload),
        model_id=adapter_result.model_id,
        revision=adapter_result.revision,
        adapter_name=adapter_result.adapter_name,
        source_model_map_id=source_model_map_id,
        provenance=provenance,
        views=views,
    )


project_graph_view = build_graph_view_document


__all__ = ["build_graph_view_document", "project_graph_view"]
