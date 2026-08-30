"""Convert a torch ExportedProgram into framework-neutral Model Map IR."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_vis.ir import (
    Diagnostic,
    LogicalOp,
    LogicalOpDomain,
    SymbolicExpression,
    TensorSpec,
    artifact_local_id,
    deterministic_id,
)

from .types import TinyRepresentativeSpec

_DTYPE_MAP = {
    "torch.bool": "bool",
    "torch.uint8": "uint8",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.complex64": "complex64",
    "torch.complex128": "complex128",
}


@dataclass(frozen=True)
class IRConversion:
    tensors: Tuple[TensorSpec, ...]
    logical_ops: Tuple[LogicalOp, ...]
    diagnostics: Tuple[Diagnostic, ...]
    graph_input_ids: Tuple[str, ...]
    graph_output_ids: Tuple[str, ...]
    converted_ops: int
    total_ops: int

    @property
    def coverage(self) -> float:
        if self.total_ops == 0:
            return 0.0
        return self.converted_ops / self.total_ops


def _expression_from_ast(node: ast.AST) -> SymbolicExpression:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return SymbolicExpression.literal(node.value)
    if isinstance(node, ast.Name):
        return SymbolicExpression.symbol_ref(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return SymbolicExpression(op="negate", args=[_expression_from_ast(node.operand)])
    if isinstance(node, ast.BinOp):
        operations = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.Div: "divide",
            ast.FloorDiv: "floor_divide",
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported symbolic binary operator")
        return SymbolicExpression(
            op=operation,
            args=[_expression_from_ast(node.left), _expression_from_ast(node.right)],
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = node.func.id.lower()
        if function in {"min", "minimum"}:
            operation = "minimum"
        elif function in {"max", "maximum"}:
            operation = "maximum"
        elif function in {"ceildiv", "ceil_div"} and len(node.args) == 2:
            operation = "ceil_divide"
        else:
            raise ValueError("unsupported symbolic function")
        return SymbolicExpression(
            op=operation,
            args=[_expression_from_ast(argument) for argument in node.args],
        )
    raise ValueError("unsupported symbolic expression")


def symbolic_expression_from_dimension(dimension: Any) -> SymbolicExpression:
    """Convert an integer/SymInt-like dimension to the structured IR AST."""

    if isinstance(dimension, int):
        return SymbolicExpression.literal(dimension)
    text = str(dimension)
    try:
        parsed = ast.parse(text, mode="eval")
        return _expression_from_ast(parsed.body)
    except (SyntaxError, ValueError):
        # Still structured as a symbolic leaf; the original contract is never
        # silently converted to a guessed numeric value.
        return SymbolicExpression.symbol_ref(text)


def _flatten_tensor_values(value: Any) -> Tuple[Any, ...]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return (value,)
    if isinstance(value, (tuple, list)):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(_flatten_tensor_values(item))
        return tuple(flattened)
    if isinstance(value, dict):
        flattened = []
        for key in sorted(value):
            flattened.extend(_flatten_tensor_values(value[key]))
        return tuple(flattened)
    return ()


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    module = getattr(target, "__module__", None)
    if name and module:
        return f"{module}.{name}"
    return str(target)


def _op_domain(op_type: str) -> LogicalOpDomain:
    lowered = op_type.lower()
    if "aten." in lowered or "torch._ops.aten" in lowered:
        return LogicalOpDomain.ATEN
    if "prims." in lowered:
        return LogicalOpDomain.PRIMS
    if lowered.startswith("torch."):
        return LogicalOpDomain.TORCH
    return LogicalOpDomain.CUSTOM


def _node_references(value: Any) -> Iterable[Any]:
    if all(hasattr(value, attribute) for attribute in ("op", "name", "target")):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _node_references(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _node_references(value[key])


def _tensor_role(
    node: Any,
    user_inputs: Sequence[str],
    input_roles: Dict[str, str],
) -> str:
    if node.op == "placeholder":
        target = str(node.target)
        if target in user_inputs or node.name in user_inputs:
            return input_roles.get(target, input_roles.get(node.name, "input"))
        return "parameter"
    if node.op == "get_attr":
        return "parameter"
    return "activation"


def _safe_stride(tensor: Any) -> Optional[List[int]]:
    try:
        stride = tuple(tensor.stride())
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not all(isinstance(item, int) for item in stride):
        return None
    return list(stride)


def _layout(tensor: Any) -> str:
    try:
        return "contiguous" if tensor.is_contiguous() else "strided"
    except (AttributeError, RuntimeError, TypeError):
        return "unknown"


def _user_input_names(exported_program: Any) -> Tuple[str, ...]:
    signature = getattr(exported_program, "graph_signature", None)
    user_inputs = getattr(signature, "user_inputs", ())
    return tuple(str(item) for item in user_inputs)


def _diagnostic(
    *,
    capture_id: str,
    code: str,
    message: str,
    subject_id: Optional[str] = None,
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
        severity="warning",
        code=code,
        subject_id=subject_id,
        message=message,
    )


def convert_exported_program(
    exported_program: Any,
    *,
    spec: TinyRepresentativeSpec,
    capture_id: str,
) -> IRConversion:
    """Normalize FX nodes and FakeTensor metadata without running the module."""

    graph_module = getattr(exported_program, "graph_module", None)
    graph = getattr(graph_module, "graph", None)
    if graph is None:
        raise TypeError("exported_program does not expose graph_module.graph")

    nodes = tuple(graph.nodes)
    user_inputs = _user_input_names(exported_program)
    input_roles = {contract.name: contract.role for contract in spec.inputs}
    tensors: List[TensorSpec] = []
    logical_ops: List[LogicalOp] = []
    diagnostics: List[Diagnostic] = []
    tensor_ids_by_node: Dict[Any, Tuple[str, ...]] = {}
    computational_nodes = tuple(
        node for node in nodes if node.op in {"call_function", "call_method", "call_module"}
    )
    converted_ops = 0

    for ordinal, node in enumerate(nodes):
        values = _flatten_tensor_values(getattr(node, "meta", {}).get("val"))
        output_ids: List[str] = []
        for output_ordinal, tensor in enumerate(values):
            tensor_id = artifact_local_id(
                model_revision="tiny-fixture-v0.1",
                framework_domain="torch.export.tensor",
                module_path=f"{spec.module_path}.{node.name}.output.{output_ordinal}",
                op_ordinal=ordinal,
                capture_variant=spec.kind.value,
            )
            dtype = _DTYPE_MAP.get(str(tensor.dtype), "unknown")
            shape = [symbolic_expression_from_dimension(item) for item in tensor.shape]
            tensors.append(
                TensorSpec(
                    id=tensor_id,
                    symbolic_shape=shape,
                    dtype=dtype,
                    storage_dtype=dtype,
                    layout=_layout(tensor),
                    stride=_safe_stride(tensor),
                    device=str(getattr(tensor, "device", "unknown")),
                    role=_tensor_role(node, user_inputs, input_roles),
                )
            )
            output_ids.append(tensor_id)
        tensor_ids_by_node[node] = tuple(output_ids)

        if node not in computational_nodes:
            continue
        op_type = _target_name(node.target)
        op_id = artifact_local_id(
            model_revision="tiny-fixture-v0.1",
            framework_domain="torch.export.op",
            module_path=f"{spec.module_path}.{node.name}",
            op_ordinal=ordinal,
            capture_variant=spec.kind.value,
        )
        input_ids: List[str] = []
        for reference in _node_references((node.args, node.kwargs)):
            input_ids.extend(tensor_ids_by_node.get(reference, ()))
        logical_ops.append(
            LogicalOp(
                id=op_id,
                domain=_op_domain(op_type),
                op_type=op_type,
                inputs=list(dict.fromkeys(input_ids)),
                outputs=output_ids,
                attrs={"fx_name": node.name, "fx_op": node.op},
            )
        )
        if output_ids:
            converted_ops += 1
        else:
            diagnostics.append(
                _diagnostic(
                    capture_id=capture_id,
                    code="CAPTURE_TENSOR_METADATA_MISSING",
                    subject_id=op_id,
                    message=f"No tensor metadata was available for exported node {node.name!r}",
                )
            )

    graph_input_ids: List[str] = []
    for node in nodes:
        if node.op == "placeholder" and _tensor_role(node, user_inputs, input_roles) in {
            "input",
            "state",
        }:
            graph_input_ids.extend(tensor_ids_by_node.get(node, ()))

    graph_output_ids: List[str] = []
    for node in nodes:
        if node.op != "output":
            continue
        for reference in _node_references((node.args, node.kwargs)):
            graph_output_ids.extend(tensor_ids_by_node.get(reference, ()))
    graph_output_ids = list(dict.fromkeys(graph_output_ids))
    output_id_set = set(graph_output_ids)
    normalized_tensors = tuple(
        TensorSpec.model_validate({**tensor.model_dump(mode="python"), "role": "output"})
        if tensor.id in output_id_set
        else tensor
        for tensor in tensors
    )

    return IRConversion(
        tensors=normalized_tensors,
        logical_ops=tuple(logical_ops),
        diagnostics=tuple(diagnostics),
        graph_input_ids=tuple(dict.fromkeys(graph_input_ids)),
        graph_output_ids=tuple(graph_output_ids),
        converted_ops=converted_ops,
        total_ops=len(computational_nodes),
    )
