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
    GraphEdge,
    GraphEdgeKind,
    GraphGroup,
    GraphNode,
    GraphPort,
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

    def finish(self, *, root_group_key: str = "root") -> GraphView:
        return GraphView(
            id=self.view_id,
            key=self.key,
            level=self.level,
            label=self.label,
            parent_view_id=self.parent_view_id,
            breadcrumb=self.breadcrumb,
            layer_index=self.layer_index,
            root_group_id=self._groups[root_group_key]["id"],
            nodes=[GraphNode(**payload) for payload in self._nodes.values()],
            ports=list(self._ports.values()),
            edges=list(self._edges.values()),
            groups=[GraphGroup(**payload) for payload in self._groups.values()],
            metadata=self.metadata,
        )


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
            keys.append("l1_linear_attention")
        if any(entry.attention_kind == "full_attention" for entry in result.layer_strip):
            keys.append("l1_full_attention")
    elif result.adapter_name == "glm-moe-dsa":
        if any(entry.mlp_kind == "dense" for entry in result.layer_strip):
            keys.append("l1_dense_dsa")
        if any(entry.mlp_kind == "sparse" for entry in result.layer_strip):
            keys.append("l1_sparse_dsa_moe")
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
    return builder.finish()


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
    return builder.finish()


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
    return builder.finish()


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
    )
    return builder.finish()


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
        )
        return builder.finish()

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
    )
    _add_activation_port(builder, index, "post_norm", "in", "hidden_states", PortDirection.INPUT)
    _add_activation_port(builder, index, "post_norm", "out", "normalized", PortDirection.OUTPUT)
    builder.add_node(
        "router",
        "MoE Router",
        "moe_router",
        group_key="root",
        subject_ids=block_subjects,
        evidence=(*evidence, "config:num_experts_per_tok"),
        attributes={
            "routed_experts": index.result.metadata.get("routed_experts"),
            "top_k": index.result.metadata.get("top_k"),
        },
    )
    _add_activation_port(
        builder, index, "router", "hidden_in", "hidden_states", PortDirection.INPUT
    )
    builder.add_port(
        "router",
        "routes",
        "expert_indices",
        PortDirection.OUTPUT,
        TensorRole.ROUTE,
        shape=["B", "T", int(index.result.metadata["top_k"])],
        dtype=DType.INT64,
        evidence=(*evidence, "config:num_experts_per_tok"),
    )
    builder.add_port(
        "router",
        "dispatch",
        "dispatch_control",
        PortDirection.OUTPUT,
        TensorRole.CONTROL,
        shape=None,
        dtype=DType.BOOL,
        evidence=(*evidence, "config:virtual_expert_pool"),
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
        attributes={
            "virtual": True,
            "experts_expanded": False,
            "experts_materialized": 0,
            "routed_experts": index.result.metadata.get("routed_experts"),
            "top_k": index.result.metadata.get("top_k"),
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
        attributes={"shared_experts": index.result.metadata.get("shared_experts", 0)},
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
    _add_activation_port(builder, index, "combine", "hidden", "moe_output", PortDirection.OUTPUT)
    builder.add_node(
        "residual_2",
        "Residual Add",
        "residual",
        group_key="root",
        subject_ids=block_subjects,
        evidence=evidence,
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
        ("router_to_pool", "router", "routes", "expert_pool", "routes", GraphEdgeKind.ROUTE),
        ("router_to_combine", "router", "dispatch", "combine", "dispatch", GraphEdgeKind.CONTROL),
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
    return builder.finish()


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
            views.append(
                _build_qwen_or_tiny_l1(
                    index,
                    view_ids,
                    view_key="l1_linear_attention",
                    entry=_representative(adapter_result, attention="linear_attention"),
                )
            )
        if "l1_full_attention" in view_ids:
            views.append(
                _build_qwen_or_tiny_l1(
                    index,
                    view_ids,
                    view_key="l1_full_attention",
                    entry=_representative(adapter_result, attention="full_attention"),
                )
            )
    elif adapter_result.adapter_name == "glm-moe-dsa":
        views.append(_build_glm_l0(index, view_ids))
        if "l1_dense_dsa" in view_ids:
            views.append(
                _build_glm_l1(
                    index,
                    view_ids,
                    view_key="l1_dense_dsa",
                    entry=_representative(adapter_result, mlp="dense"),
                )
            )
        if "l1_sparse_dsa_moe" in view_ids:
            views.append(
                _build_glm_l1(
                    index,
                    view_ids,
                    view_key="l1_sparse_dsa_moe",
                    entry=_representative(adapter_result, mlp="sparse"),
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
