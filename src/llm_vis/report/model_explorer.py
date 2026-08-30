"""Pure-JSON Model Explorer adapter spike.

The emitted shape follows Model Explorer's processed graph format and can be consumed
by its JSON adapter without importing the optional UI package.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from llm_vis.ir import Definition, ModelMap

MAX_DEFINITION_NODES = 50
MAX_GRAPH_NODES = 200


def _attrs(values: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [{"key": key, "value": str(value)} for key, value in values.items() if value is not None]


def _definition_namespaces(definitions: Iterable[Definition]) -> Dict[str, str]:
    items = list(definitions)
    parents: Dict[str, str] = {}
    labels = {item.id: item.label for item in items}
    for parent in items:
        for child_id in parent.children:
            parents.setdefault(child_id, parent.id)

    def namespace(item_id: str) -> str:
        path = []
        current = parents.get(item_id)
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            path.append(labels[current])
            current = parents.get(current)
        return "/".join(reversed(path))

    return {item.id: namespace(item.id) for item in items}


def _definition_graph(model_map: ModelMap) -> Dict[str, Any]:
    namespaces = _definition_namespaces(model_map.definitions)
    nodes = []
    for definition in model_map.definitions[:MAX_DEFINITION_NODES]:
        nodes.append(
            {
                "id": definition.id,
                "label": definition.label,
                "namespace": namespaces[definition.id],
                "attrs": _attrs(
                    {
                        "kind": definition.kind.value,
                        "class": definition.class_name,
                        "semantic_pattern": definition.semantic_pattern,
                    }
                ),
                "incomingEdges": [],
            }
        )
    return {"id": "architecture-definitions", "nodes": nodes}


def _semantic_graph(model_map: ModelMap) -> Dict[str, Any]:
    instances = {item.id: item for item in model_map.instances}
    groups: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for node in model_map.semantic_nodes:
        groups[(node.kind.value, node.definition_id or "")].append(node)
    group_id_by_source = {
        node.id: sorted(item.id for item in group)[0] for group in groups.values() for node in group
    }
    ordered_group_ids = [sorted(item.id for item in group)[0] for group in groups.values()][
        :MAX_GRAPH_NODES
    ]
    group_ids = set(ordered_group_ids)
    incoming: Dict[str, List[Dict[str, str]]] = {item_id: [] for item_id in group_ids}
    for edge in model_map.edges:
        source_group = group_id_by_source.get(edge.source_id)
        target_group = group_id_by_source.get(edge.target_id)
        if source_group in group_ids and target_group in group_ids and source_group != target_group:
            incoming[target_group].append(
                {
                    "sourceNodeId": source_group,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            )
    for node in model_map.semantic_nodes:
        for child_id in node.child_ids:
            source_group = group_id_by_source[node.id]
            target_group = group_id_by_source.get(child_id)
            if (
                source_group in group_ids
                and target_group in group_ids
                and source_group != target_group
            ):
                incoming[target_group].append(
                    {
                        "sourceNodeId": source_group,
                        "sourceNodeOutputId": "state",
                        "targetNodeInputId": "0",
                    }
                )

    nodes = []
    for group in groups.values():
        group = sorted(group, key=lambda item: item.id)
        node = group[0]
        group_id = group_id_by_source[node.id]
        if group_id not in group_ids:
            continue
        module_paths = sorted(
            {
                instances[instance_id].module_path
                for item in group
                for instance_id in item.instance_ids
            }
        )
        module_path = module_paths[0] if module_paths else ""
        namespace = "/".join(module_path.split(".")[:-1])
        group_incoming = {
            (
                edge["sourceNodeId"],
                edge["sourceNodeOutputId"],
                edge["targetNodeInputId"],
            ): edge
            for edge in incoming[group_id]
        }
        nodes.append(
            {
                "id": group_id,
                "label": node.label if len(group) == 1 else f"{node.kind.value} × {len(group)}",
                "namespace": namespace,
                "attrs": _attrs(
                    {
                        "kind": node.kind.value,
                        "representative_module_path": module_path,
                        "instance_count": len(module_paths),
                        "confidence": min(item.confidence for item in group),
                        "opaque": any(item.opaque for item in group),
                        "evidence": "; ".join(
                            sorted({evidence for item in group for evidence in item.evidence})
                        ),
                        "source_semantic_ids": ",".join(item.id for item in group),
                    }
                ),
                "incomingEdges": list(group_incoming.values()),
            }
        )
    return {"id": "semantic-structure", "nodes": nodes}


def _logical_op_graph(model_map: ModelMap) -> Dict[str, Any]:
    selected = tuple(model_map.logical_ops[:MAX_GRAPH_NODES])
    selected_ids = {op.id for op in selected}
    instances = {item.id: item for item in model_map.instances}
    producer_by_tensor: Dict[str, str] = {}
    for op in selected:
        for tensor_id in op.outputs:
            producer_by_tensor.setdefault(tensor_id, op.id)

    nodes = []
    for op in selected:
        incoming_by_tensor: Dict[Tuple[str, str], Dict[str, str]] = {}
        for tensor_id in op.inputs:
            producer_id = producer_by_tensor.get(tensor_id)
            if producer_id is None or producer_id == op.id or producer_id not in selected_ids:
                continue
            incoming_by_tensor[(producer_id, tensor_id)] = {
                "sourceNodeId": producer_id,
                "sourceNodeOutputId": tensor_id,
                "targetNodeInputId": tensor_id,
            }
        scope = instances.get(op.scope_instance_id or "")
        namespace = scope.module_path if scope is not None else op.domain.value
        nodes.append(
            {
                "id": op.id,
                "label": op.op_type,
                "namespace": namespace,
                "attrs": _attrs(
                    {
                        "domain": op.domain.value,
                        "op_type": op.op_type,
                        "scope_instance_id": op.scope_instance_id,
                        "input_count": len(op.inputs),
                        "output_count": len(op.outputs),
                    }
                ),
                "incomingEdges": list(incoming_by_tensor.values()),
            }
        )
    return {"id": "logical-ops", "nodes": nodes}


def model_explorer_graphs(model_map: ModelMap) -> Dict[str, Any]:
    """Convert one Model Map into deterministic Model Explorer JSON graphs."""

    graphs = [_definition_graph(model_map), _semantic_graph(model_map)]
    if model_map.logical_ops:
        graphs.append(_logical_op_graph(model_map))
    return {"graphs": graphs}
