"""Config-first M3 cost analysis for supported decoder layer templates.

The engine reads configuration and Model Map metadata only. It never imports model
implementations, constructs weights, or invokes a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from llm_vis.adapters import AdapterResult
from llm_vis.ir.ids import deterministic_id
from llm_vis.ir.models import (
    CoverageStatus,
    DType,
    ExpressionOp,
    Metric,
    MetricOrigin,
    ModelMap,
    Number,
    Scenario,
    ScenarioPhase,
    Symbol,
    SymbolicExpression,
    WeightFormat,
)

from .expressions import evaluate_expression, required_symbols
from .formulas import attention_core_logical_bytes
from .storage import INT4_GROUP_SIZE, WeightStorage, dtype_nbytes, weight_storage


class CostModelError(ValueError):
    """Raised when a scenario or adapter/config combination is unsupported."""


@dataclass(frozen=True)
class MatrixSpec:
    """One repeated 2-D weight tensor shape."""

    name: str
    input_width: int
    output_width: int
    count: int = 1

    @property
    def parameters(self) -> int:
        return self.input_width * self.output_width * self.count


@dataclass(frozen=True)
class StorageBreakdown:
    packed_weight_bytes: int
    scale_metadata_bytes: int
    zero_point_metadata_bytes: int
    non_quantized_bytes: int
    padding_values: int
    format_names: Tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return (
            self.packed_weight_bytes
            + self.scale_metadata_bytes
            + self.zero_point_metadata_bytes
            + self.non_quantized_bytes
        )


@dataclass(frozen=True)
class SubjectCost:
    """All metrics and expression bindings for one layer under one scenario."""

    scenario_id: str
    subject_id: str
    module_path: str
    layer_index: int
    metrics: Tuple[Metric, ...]
    bindings: Mapping[str, Number]

    def metric(self, name: str) -> Metric:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


@dataclass(frozen=True)
class HotspotEntry:
    rank: int
    scenario_id: str
    metric_name: str
    subject_id: str
    module_path: str
    layer_index: int
    value: Number
    unit: str
    origin: MetricOrigin
    coverage: float


@dataclass(frozen=True)
class AggregateScope:
    """Structural scope represented by model-subject aggregate cost metrics.

    Aggregate values remain attached to the model for backwards-compatible
    reporting, but they are explicitly decoder-only partial sums.  The numeric
    coverage is a structural instance fraction, not a parameter/FLOP fraction.
    """

    name: str
    subject_id: str
    included_instance_ids: Tuple[str, ...]
    excluded_instance_paths: Tuple[str, ...]
    structural_coverage: float

    @property
    def included_instance_count(self) -> int:
        return len(self.included_instance_ids)

    @property
    def excluded_instance_count(self) -> int:
        return len(self.excluded_instance_paths)


@dataclass(frozen=True)
class ScenarioCost:
    scenario: Scenario
    subjects: Tuple[SubjectCost, ...]
    aggregate_metrics: Tuple[Metric, ...]
    aggregate_scope: AggregateScope

    def aggregate_metric(self, name: str) -> Metric:
        for metric in self.aggregate_metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def hotspots(
        self, metric_name: str = "flops", *, limit: int | None = None
    ) -> Tuple[HotspotEntry, ...]:
        candidates = []
        for subject in self.subjects:
            try:
                metric = subject.metric(metric_name)
            except KeyError:
                continue
            if metric.value is None:
                continue
            candidates.append((subject, metric))
        candidates.sort(
            key=lambda item: (-float(item[1].value), item[0].layer_index, item[0].module_path)
        )
        if limit is not None:
            if limit < 0:
                raise ValueError("hotspot limit must be non-negative")
            candidates = candidates[:limit]
        return tuple(
            HotspotEntry(
                rank=rank,
                scenario_id=self.scenario.id,
                metric_name=metric_name,
                subject_id=subject.subject_id,
                module_path=subject.module_path,
                layer_index=subject.layer_index,
                value=metric.value,
                unit=metric.unit,
                origin=metric.origin,
                coverage=metric.coverage,
            )
            for rank, (subject, metric) in enumerate(candidates, start=1)
            if metric.value is not None
        )


@dataclass(frozen=True)
class CostAnalysis:
    """Cost output grouped by scenario, ready for analysis/report pipelines."""

    model_id: str
    model_revision: str
    scenario_costs: Tuple[ScenarioCost, ...]

    @property
    def metrics(self) -> Tuple[Metric, ...]:
        metrics = []
        for scenario_cost in self.scenario_costs:
            for subject in scenario_cost.subjects:
                metrics.extend(subject.metrics)
            metrics.extend(scenario_cost.aggregate_metrics)
        return tuple(metrics)

    def for_scenario(self, scenario_id: str) -> ScenarioCost:
        for scenario_cost in self.scenario_costs:
            if scenario_cost.scenario.id == scenario_id:
                return scenario_cost
        raise KeyError(scenario_id)

    def hotspots(
        self,
        scenario_id: str,
        metric_name: str = "flops",
        *,
        limit: int | None = None,
    ) -> Tuple[HotspotEntry, ...]:
        return self.for_scenario(scenario_id).hotspots(metric_name, limit=limit)

    def attach_to_model_map(self, model_map: ModelMap) -> ModelMap:
        """Return a Model Map with self-contained formula bindings.

        Subject-local formulas use short names such as ``B``, ``T``, ``H`` and
        ``N_Q`` while they are being built.  A Model Map can contain several
        scenarios and heterogeneous layer kinds, so those names are qualified
        before serialization.  Every qualified name is backed by a ``Symbol``
        whose ``default_value`` is the exact Scenario/config binding.  This lets
        an offline consumer recompute every known persisted formula using only
        ``model-map.json``.
        """

        if model_map.model.id != self.model_id:
            raise CostModelError("CostAnalysis belongs to a different Model Map")
        payload = model_map.model_dump(mode="json")
        scenario_by_id = {scenario["id"]: scenario for scenario in payload["scenarios"]}
        for scenario_cost in self.scenario_costs:
            scenario_by_id[scenario_cost.scenario.id] = scenario_cost.scenario.model_dump(
                mode="json"
            )
        symbols_by_name = {symbol["name"]: symbol for symbol in payload["symbols"]}
        symbols_by_id = {symbol["id"]: symbol for symbol in payload["symbols"]}
        persisted_subject_metrics: Dict[Tuple[str, str, str], Metric] = {}

        for scenario_cost in self.scenario_costs:
            for subject in scenario_cost.subjects:
                name_map: Dict[str, str] = {}
                for binding_name, binding_value in subject.bindings.items():
                    binding_symbol = _binding_symbol(
                        scenario=scenario_cost.scenario,
                        subject_id=subject.subject_id,
                        binding_name=binding_name,
                        value=binding_value,
                    )
                    name_map[binding_name] = binding_symbol.name
                    existing_name = symbols_by_name.get(binding_symbol.name)
                    existing_id = symbols_by_id.get(binding_symbol.id)
                    dumped = binding_symbol.model_dump(mode="json")
                    if existing_name is not None and existing_name != dumped:
                        raise CostModelError(f"Conflicting formula binding {binding_symbol.name!r}")
                    if existing_id is not None and existing_id != dumped:
                        raise CostModelError(
                            f"Conflicting formula binding id {binding_symbol.id!r}"
                        )
                    symbols_by_name[binding_symbol.name] = dumped
                    symbols_by_id[binding_symbol.id] = dumped

                for metric in subject.metrics:
                    formula = metric.formula
                    assumptions = list(metric.assumptions)
                    if formula is not None:
                        missing = required_symbols(formula) - set(name_map)
                        if missing:
                            raise CostModelError(
                                f"Metric {metric.id!r} lacks bindings for {sorted(missing)!r}"
                            )
                        formula = _rename_symbols(formula, name_map)
                        assumptions.extend(
                            (
                                "formula_bindings:model_map.symbols.default_value",
                                "formula_binding_scope:scenario_and_decoder_instance",
                            )
                        )
                    persisted = Metric.model_validate(
                        {
                            **metric.model_dump(mode="json"),
                            "formula": (
                                formula.model_dump(mode="json") if formula is not None else None
                            ),
                            "assumptions": assumptions,
                        }
                    )
                    persisted_subject_metrics[
                        (scenario_cost.scenario.id, subject.subject_id, metric.name)
                    ] = persisted

        persisted_aggregate_metrics: Dict[Tuple[str, str], Metric] = {}
        for scenario_cost in self.scenario_costs:
            for metric in scenario_cost.aggregate_metrics:
                formula = _persisted_aggregate_formula(
                    metric=metric,
                    scenario_cost=scenario_cost,
                    subject_metrics=persisted_subject_metrics,
                    aggregate_metrics=persisted_aggregate_metrics,
                )
                assumptions = list(metric.assumptions)
                if formula is not None:
                    assumptions.extend(
                        (
                            "formula_provenance:composed_decoder_layer_formulas",
                            "formula_bindings:model_map.symbols.default_value",
                        )
                    )
                persisted_aggregate_metrics[(scenario_cost.scenario.id, metric.name)] = (
                    Metric.model_validate(
                        {
                            **metric.model_dump(mode="json"),
                            "formula": (
                                formula.model_dump(mode="json") if formula is not None else None
                            ),
                            "assumptions": assumptions,
                        }
                    )
                )

        metric_by_id = {metric["id"]: metric for metric in payload["metrics"]}
        for metric in persisted_subject_metrics.values():
            metric_by_id[metric.id] = metric.model_dump(mode="json")
        for metric in persisted_aggregate_metrics.values():
            metric_by_id[metric.id] = metric.model_dump(mode="json")
        payload["scenarios"] = list(scenario_by_id.values())
        payload["symbols"] = list(symbols_by_id.values())
        payload["metrics"] = list(metric_by_id.values())
        return ModelMap.model_validate(payload)


_SCENARIO_BINDING_SOURCES = {
    "B": "Scenario.batch",
    "T": "Scenario.new_tokens",
    "L": "Scenario.past_tokens",
    "S_KV": "Scenario.total_visible_length",
    "B_A": "Scenario.activation_dtype->bytes_per_element",
    "B_KV": "Scenario.kv_dtype->bytes_per_element",
}

_BINDING_UNITS = {
    "B": "sequences",
    "T": "tokens",
    "L": "tokens",
    "S_KV": "tokens",
    "B_A": "bytes/element",
    "B_KV": "bytes/element",
    "B_STATE": "bytes/element",
    "P_ATTN": "parameters",
    "P_FFN": "parameters",
    "P_TOTAL": "parameters",
    "P_RESIDENT": "parameters",
    "P_ACTIVE": "parameters",
}


def _binding_symbol(
    *,
    scenario: Scenario,
    subject_id: str,
    binding_name: str,
    value: Number,
) -> Symbol:
    scenario_source = _SCENARIO_BINDING_SOURCES.get(binding_name)
    if scenario_source is not None:
        scope_kind = "scenario"
        scope_id = scenario.id
        source = scenario_source
    else:
        scope_kind = "decoder_instance"
        scope_id = subject_id
        source = "config-derived decoder dimension"
    qualified_name = f"{scope_kind}.{scope_id}.{binding_name}"
    return Symbol(
        id=deterministic_id(
            "symbol",
            {
                "binding_name": binding_name,
                "scope_id": scope_id,
                "scope_kind": scope_kind,
            },
        ),
        name=qualified_name,
        description=(
            f"M3 formula binding; original_name={binding_name}; source={source}; "
            f"{scope_kind}_id={scope_id}"
        ),
        unit=_BINDING_UNITS.get(binding_name),
        lower_bound=0,
        default_value=value,
    )


def _rename_symbols(
    expression: SymbolicExpression,
    name_map: Mapping[str, str],
) -> SymbolicExpression:
    if expression.op == ExpressionOp.LITERAL:
        assert expression.value is not None
        return _literal(expression.value)
    if expression.op == ExpressionOp.SYMBOL:
        assert expression.symbol is not None
        return _symbol(name_map[expression.symbol])
    return SymbolicExpression(
        op=expression.op,
        args=[_rename_symbols(argument, name_map) for argument in expression.args],
    )


def _persisted_aggregate_formula(
    *,
    metric: Metric,
    scenario_cost: ScenarioCost,
    subject_metrics: Mapping[Tuple[str, str, str], Metric],
    aggregate_metrics: Mapping[Tuple[str, str], Metric],
) -> Optional[SymbolicExpression]:
    if metric.value is None:
        return None
    if metric.name == "arithmetic_intensity":
        flops = aggregate_metrics.get((scenario_cost.scenario.id, "flops"))
        logical_bytes = aggregate_metrics.get((scenario_cost.scenario.id, "logical_bytes"))
        if (
            flops is None
            or logical_bytes is None
            or flops.formula is None
            or logical_bytes.formula is None
        ):
            return _literal(metric.value)
        return _divide(flops.formula, logical_bytes.formula)

    formulas = []
    for subject in scenario_cost.subjects:
        member = subject_metrics.get((scenario_cost.scenario.id, subject.subject_id, metric.name))
        if member is None or member.value is None:
            continue
        formulas.append(member.formula or _literal(member.value))
    return _add(*formulas) if formulas else _literal(metric.value)


def _literal(value: Number) -> SymbolicExpression:
    return SymbolicExpression.literal(value)


def _symbol(name: str) -> SymbolicExpression:
    return SymbolicExpression.symbol_ref(name)


def _operation(op: ExpressionOp, *args: SymbolicExpression) -> SymbolicExpression:
    return SymbolicExpression(op=op, args=list(args))


def _add(*args: SymbolicExpression) -> SymbolicExpression:
    if not args:
        return _literal(0)
    if len(args) == 1:
        return args[0]
    return _operation(ExpressionOp.ADD, *args)


def _multiply(*args: SymbolicExpression) -> SymbolicExpression:
    if not args:
        return _literal(1)
    if len(args) == 1:
        return args[0]
    return _operation(ExpressionOp.MULTIPLY, *args)


def _divide(left: SymbolicExpression, right: SymbolicExpression) -> SymbolicExpression:
    return _operation(ExpressionOp.DIVIDE, left, right)


def _floor_divide(left: SymbolicExpression, right: SymbolicExpression) -> SymbolicExpression:
    return _operation(ExpressionOp.FLOOR_DIVIDE, left, right)


def _coverage_status(coverage: float, *, unknown: bool = False) -> CoverageStatus:
    if unknown or coverage == 0:
        return CoverageStatus.UNKNOWN
    if coverage < 1:
        return CoverageStatus.PARTIAL
    return CoverageStatus.COMPLETE


def _metric_id(subject_id: str, scenario_id: str, name: str) -> str:
    return deterministic_id(
        "metric",
        {"name": name, "scenario_id": scenario_id, "subject_id": subject_id},
    )


def _known_metric(
    *,
    subject_id: str,
    scenario: Scenario,
    name: str,
    unit: str,
    formula: SymbolicExpression,
    bindings: Mapping[str, Number],
    origin: MetricOrigin,
    assumptions: Iterable[str],
    coverage: float = 1.0,
) -> Metric:
    value = evaluate_expression(formula, bindings)
    return Metric(
        id=_metric_id(subject_id, scenario.id, name),
        subject_id=subject_id,
        scenario_id=scenario.id,
        name=name,
        value=value,
        unit=unit,
        origin=origin,
        formula=formula,
        assumptions=list(assumptions),
        confidence=0.8 if origin == MetricOrigin.ESTIMATED else 1.0,
        coverage=coverage,
        coverage_status=_coverage_status(coverage),
    )


def _unknown_metric(
    *,
    subject_id: str,
    scenario: Scenario,
    name: str,
    unit: str,
    assumptions: Iterable[str],
) -> Metric:
    return Metric(
        id=_metric_id(subject_id, scenario.id, name),
        subject_id=subject_id,
        scenario_id=scenario.id,
        name=name,
        value=None,
        unit=unit,
        origin=MetricOrigin.UNKNOWN,
        assumptions=list(assumptions),
        confidence=None,
        coverage=0.0,
        coverage_status=CoverageStatus.UNKNOWN,
    )


def _positive_config_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CostModelError(f"Config field {key!r} must be a positive integer")
    return value


def _storage_for_matrices(
    matrices: Sequence[MatrixSpec],
    *,
    non_quantized_parameters: int,
    scenario: Scenario,
    int4_group_size: int,
) -> StorageBreakdown:
    packed = 0
    scales = 0
    zero_points = 0
    padding = 0
    format_names = set()
    for matrix in matrices:
        for _ in range(matrix.count):
            storage = weight_storage(
                matrix.input_width * matrix.output_width,
                scenario.weight_format,
                group_size=int4_group_size,
            )
            packed += storage.packed_weight_bytes
            scales += storage.scale_metadata_bytes
            zero_points += storage.zero_point_metadata_bytes
            padding += storage.padding_values
            format_names.add(storage.format_name)
    side_bytes = non_quantized_parameters * dtype_nbytes(scenario.activation_dtype)
    return StorageBreakdown(
        packed_weight_bytes=packed,
        scale_metadata_bytes=scales,
        zero_point_metadata_bytes=zero_points,
        non_quantized_bytes=side_bytes,
        padding_values=padding,
        format_names=tuple(sorted(format_names)),
    )


def _storage_assumptions(storage: StorageBreakdown) -> Tuple[str, ...]:
    return (
        f"weight_format:{','.join(storage.format_names)}",
        f"packed_weight_bytes:{storage.packed_weight_bytes}",
        f"scale_metadata_bytes:{storage.scale_metadata_bytes}",
        f"zero_point_metadata_bytes:{storage.zero_point_metadata_bytes}",
        f"non_quantized_side_bytes:{storage.non_quantized_bytes}",
        f"padding_values:{storage.padding_values}",
    )


def _matrix_logical_bytes(
    matrix: MatrixSpec,
    *,
    tokens: int,
    activation_bytes: int,
    scenario: Scenario,
    int4_group_size: int,
) -> int:
    if matrix.count != 1:
        raise CostModelError("Logical op accounting expects one concrete matrix")
    storage: WeightStorage = weight_storage(
        matrix.input_width * matrix.output_width,
        scenario.weight_format,
        group_size=int4_group_size,
    )
    return (
        tokens * matrix.input_width * activation_bytes
        + storage.total_bytes
        + tokens * matrix.output_width * activation_bytes
    )


def _scenario_bindings(scenario: Scenario, **dimensions: Number) -> Dict[str, Number]:
    return {
        "B": scenario.batch,
        "T": scenario.new_tokens,
        "L": scenario.past_tokens,
        "S_KV": scenario.total_visible_length,
        "B_A": dtype_nbytes(scenario.activation_dtype),
        "B_KV": dtype_nbytes(scenario.kv_dtype),
        **dimensions,
    }


def _base_assumptions(adapter_result: AdapterResult) -> Tuple[str, ...]:
    return (
        f"provenance:config-first:{adapter_result.adapter_name}",
        "weights_loaded:false",
        "full_forward_executed:false",
        "batch sequences are equal length and unpadded",
    )


def _standard_attention_subject(
    *,
    scenario: Scenario,
    subject_id: str,
    module_path: str,
    layer_index: int,
    adapter_result: AdapterResult,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    q_projection_multiplier: int,
    attention_bias: bool,
    include_qk_norm: bool,
    int4_group_size: int,
) -> SubjectCost:
    query_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    q_projection_width = q_projection_multiplier * query_width
    attention_matrices = (
        MatrixSpec("q_proj", hidden_size, q_projection_width),
        MatrixSpec("k_proj", hidden_size, kv_width),
        MatrixSpec("v_proj", hidden_size, kv_width),
        MatrixSpec("o_proj", query_width, hidden_size),
    )
    ffn_matrices = (
        MatrixSpec("gate_proj", hidden_size, intermediate_size),
        MatrixSpec("up_proj", hidden_size, intermediate_size),
        MatrixSpec("down_proj", intermediate_size, hidden_size),
    )
    attention_bias_params = q_projection_width + 2 * kv_width + hidden_size if attention_bias else 0
    attention_side_params = hidden_size + attention_bias_params
    if include_qk_norm:
        attention_side_params += 2 * head_dim
    ffn_side_params = hidden_size
    all_matrices = (*attention_matrices, *ffn_matrices)
    resident_parameters = sum(matrix.parameters for matrix in all_matrices)
    resident_parameters += attention_side_params + ffn_side_params
    attention_parameters = sum(matrix.parameters for matrix in attention_matrices)
    attention_parameters += attention_side_params
    ffn_parameters = sum(matrix.parameters for matrix in ffn_matrices) + ffn_side_params

    bindings = _scenario_bindings(
        scenario,
        H=hidden_size,
        I=intermediate_size,
        N_Q=query_heads,
        N_KV=kv_heads,
        D_H=head_dim,
        Q_PROJ_MULT=q_projection_multiplier,
        P_ATTN=attention_parameters,
        P_FFN=ffn_parameters,
        P_TOTAL=resident_parameters,
    )
    base = _base_assumptions(adapter_result)
    parameter_formula = _symbol("P_TOTAL")
    attention_parameter_formula = _symbol("P_ATTN")
    ffn_parameter_formula = _symbol("P_FFN")

    storage = _storage_for_matrices(
        all_matrices,
        non_quantized_parameters=attention_side_params + ffn_side_params,
        scenario=scenario,
        int4_group_size=int4_group_size,
    )
    tokens = scenario.batch * scenario.new_tokens
    attention_matrix_params_formula = _add(
        _multiply(
            _symbol("H"),
            _symbol("Q_PROJ_MULT"),
            _symbol("N_Q"),
            _symbol("D_H"),
        ),
        _multiply(_literal(2), _symbol("H"), _symbol("N_KV"), _symbol("D_H")),
        _multiply(_symbol("N_Q"), _symbol("D_H"), _symbol("H")),
    )
    ffn_matrix_params_formula = _multiply(_literal(3), _symbol("H"), _symbol("I"))
    causal_pairs_formula = _add(
        _multiply(_symbol("T"), _symbol("L")),
        _floor_divide(
            _multiply(_symbol("T"), _add(_symbol("T"), _literal(1))),
            _literal(2),
        ),
    )
    projection_flops_formula = _multiply(
        _literal(2),
        _symbol("B"),
        _symbol("T"),
        attention_matrix_params_formula,
    )
    attention_core_formula = _multiply(
        _literal(4),
        _symbol("B"),
        _symbol("N_Q"),
        _symbol("D_H"),
        causal_pairs_formula,
    )
    attention_flops_formula = _add(projection_flops_formula, attention_core_formula)
    ffn_flops_formula = _multiply(
        _literal(2),
        _symbol("B"),
        _symbol("T"),
        ffn_matrix_params_formula,
    )
    total_flops_formula = _add(attention_flops_formula, ffn_flops_formula)

    activation_bytes = dtype_nbytes(scenario.activation_dtype)
    attention_logical = sum(
        _matrix_logical_bytes(
            matrix,
            tokens=tokens,
            activation_bytes=activation_bytes,
            scenario=scenario,
            int4_group_size=int4_group_size,
        )
        for matrix in attention_matrices
    )
    attention_logical += attention_core_logical_bytes(
        batch=scenario.batch,
        new_tokens=scenario.new_tokens,
        total_visible_tokens=scenario.total_visible_length,
        query_width=query_width,
        key_width=kv_width,
        value_width=kv_width,
        activation_bytes=activation_bytes,
        kv_bytes=dtype_nbytes(scenario.kv_dtype),
    )
    kv_write_bytes = (
        scenario.batch * scenario.new_tokens * 2 * kv_width * dtype_nbytes(scenario.kv_dtype)
    )
    attention_logical += kv_write_bytes
    ffn_logical = sum(
        _matrix_logical_bytes(
            matrix,
            tokens=tokens,
            activation_bytes=activation_bytes,
            scenario=scenario,
            int4_group_size=int4_group_size,
        )
        for matrix in ffn_matrices
    )
    total_logical = attention_logical + ffn_logical
    kv_formula = _multiply(
        _symbol("B"),
        _symbol("S_KV"),
        _literal(2),
        _symbol("N_KV"),
        _symbol("D_H"),
        _symbol("B_KV"),
    )
    kv_per_token_formula = _multiply(_literal(2), _symbol("N_KV"), _symbol("D_H"), _symbol("B_KV"))

    formulas = {
        "parameters.total": (parameter_formula, "parameters", MetricOrigin.EXACT),
        "parameters.resident": (parameter_formula, "parameters", MetricOrigin.EXACT),
        "parameters.active": (parameter_formula, "parameters", MetricOrigin.EXACT),
        "parameters.attention": (
            attention_parameter_formula,
            "parameters",
            MetricOrigin.EXACT,
        ),
        "parameters.ffn": (ffn_parameter_formula, "parameters", MetricOrigin.EXACT),
        "weights.storage_bytes": (
            _literal(storage.total_bytes),
            "bytes",
            MetricOrigin.EXACT,
        ),
        "flops.attention": (attention_flops_formula, "FLOPs", MetricOrigin.FORMULA),
        "flops.ffn": (ffn_flops_formula, "FLOPs", MetricOrigin.FORMULA),
        "flops": (total_flops_formula, "FLOPs", MetricOrigin.FORMULA),
        "logical_bytes.attention": (
            _literal(attention_logical),
            "bytes",
            MetricOrigin.FORMULA,
        ),
        "logical_bytes.ffn": (_literal(ffn_logical), "bytes", MetricOrigin.FORMULA),
        "logical_bytes": (_literal(total_logical), "bytes", MetricOrigin.FORMULA),
        "kv_cache.bytes": (kv_formula, "bytes", MetricOrigin.FORMULA),
        "kv_cache.bytes_per_sequence_token": (
            kv_per_token_formula,
            "bytes/token",
            MetricOrigin.FORMULA,
        ),
        "state.bytes": (_literal(0), "bytes", MetricOrigin.EXACT),
        "arithmetic_intensity": (
            _divide(total_flops_formula, _literal(total_logical)),
            "FLOP/byte",
            MetricOrigin.FORMULA,
        ),
    }
    storage_notes = _storage_assumptions(storage)
    attention_shape = (
        f"attention_shape:N_Q={query_heads},N_KV={kv_heads},D_H={head_dim}; "
        f"q_projection_multiplier={q_projection_multiplier}"
    )
    metrics = []
    for name, (formula, unit, origin) in formulas.items():
        assumptions = list(base)
        assumptions.append(attention_shape)
        if name == "weights.storage_bytes":
            assumptions.extend(storage_notes)
        if name.startswith("logical_bytes") or name == "arithmetic_intensity":
            assumptions.append("logical minimum traffic; not measured HBM traffic")
            assumptions.append("linear weights are read once per layer invocation")
        if name.startswith("flops"):
            assumptions.append("normalization, RoPE, softmax, and elementwise FLOPs excluded")
        metrics.append(
            _known_metric(
                subject_id=subject_id,
                scenario=scenario,
                name=name,
                unit=unit,
                formula=formula,
                bindings=bindings,
                origin=origin,
                assumptions=assumptions,
            )
        )
    return SubjectCost(
        scenario_id=scenario.id,
        subject_id=subject_id,
        module_path=module_path,
        layer_index=layer_index,
        metrics=tuple(metrics),
        bindings=bindings,
    )


def _qwen_linear_subject(
    *,
    scenario: Scenario,
    subject_id: str,
    module_path: str,
    layer_index: int,
    adapter_result: AdapterResult,
    text_config: Mapping[str, Any],
    int4_group_size: int,
) -> SubjectCost:
    hidden_size = _positive_config_int(text_config, "hidden_size")
    intermediate_size = _positive_config_int(text_config, "intermediate_size")
    key_heads = _positive_config_int(text_config, "linear_num_key_heads")
    value_heads = _positive_config_int(text_config, "linear_num_value_heads")
    key_head_dim = _positive_config_int(text_config, "linear_key_head_dim")
    value_head_dim = _positive_config_int(text_config, "linear_value_head_dim")
    conv_kernel = _positive_config_int(text_config, "linear_conv_kernel_dim")
    key_width = key_heads * key_head_dim
    value_width = value_heads * value_head_dim
    conv_width = 2 * key_width + value_width

    attention_dense_matrices = (
        MatrixSpec("in_proj_qkv", hidden_size, conv_width),
        MatrixSpec("in_proj_z", hidden_size, value_width),
        MatrixSpec("in_proj_b", hidden_size, value_heads),
        MatrixSpec("in_proj_a", hidden_size, value_heads),
        MatrixSpec("out_proj", value_width, hidden_size),
    )
    conv_matrix = MatrixSpec("depthwise_conv1d", conv_width, conv_kernel)
    ffn_matrices = (
        MatrixSpec("gate_proj", hidden_size, intermediate_size),
        MatrixSpec("up_proj", hidden_size, intermediate_size),
        MatrixSpec("down_proj", intermediate_size, hidden_size),
    )
    attention_side_params = hidden_size + 2 * value_heads + value_head_dim
    ffn_side_params = hidden_size
    attention_parameters = sum(matrix.parameters for matrix in attention_dense_matrices)
    attention_parameters += conv_matrix.parameters + attention_side_params
    ffn_parameters = sum(matrix.parameters for matrix in ffn_matrices) + ffn_side_params
    resident_parameters = attention_parameters + ffn_parameters
    all_matrices = (*attention_dense_matrices, conv_matrix, *ffn_matrices)
    storage = _storage_for_matrices(
        all_matrices,
        non_quantized_parameters=attention_side_params + ffn_side_params,
        scenario=scenario,
        int4_group_size=int4_group_size,
    )

    state_dtype = DType(str(text_config.get("mamba_ssm_dtype", "float32")))
    state_bytes_per_value = dtype_nbytes(state_dtype)
    bindings = _scenario_bindings(
        scenario,
        H=hidden_size,
        I=intermediate_size,
        N_K=key_heads,
        N_V=value_heads,
        D_K=key_head_dim,
        D_V=value_head_dim,
        C=conv_kernel,
        CONV_W=conv_width,
        P_ATTN=attention_parameters,
        P_FFN=ffn_parameters,
        P_TOTAL=resident_parameters,
        B_STATE=state_bytes_per_value,
    )
    recurrent_state_formula = _multiply(
        _symbol("B"),
        _symbol("N_V"),
        _symbol("D_K"),
        _symbol("D_V"),
        _symbol("B_STATE"),
    )
    conv_state_formula = _multiply(
        _symbol("B"),
        _symbol("CONV_W"),
        _symbol("C"),
        _symbol("B_STATE"),
    )
    state_formula = _add(recurrent_state_formula, conv_state_formula)

    attention_dense_params_formula = _add(
        _multiply(_symbol("H"), _symbol("CONV_W")),
        _multiply(_literal(2), _symbol("H"), _symbol("N_V"), _symbol("D_V")),
        _multiply(_literal(2), _symbol("H"), _symbol("N_V")),
    )
    projection_flops = _multiply(
        _literal(2),
        _symbol("B"),
        _symbol("T"),
        attention_dense_params_formula,
    )
    convolution_flops = _multiply(
        _literal(2),
        _symbol("B"),
        _symbol("T"),
        _symbol("CONV_W"),
        _symbol("C"),
    )
    delta_rule_flops = _multiply(
        _literal(4),
        _symbol("B"),
        _symbol("T"),
        _symbol("N_V"),
        _symbol("D_K"),
        _symbol("D_V"),
    )
    attention_flops = _add(projection_flops, convolution_flops, delta_rule_flops)
    ffn_matrix_params_formula = _multiply(_literal(3), _symbol("H"), _symbol("I"))
    ffn_flops = _multiply(
        _literal(2),
        _symbol("B"),
        _symbol("T"),
        ffn_matrix_params_formula,
    )
    total_flops = _add(attention_flops, ffn_flops)

    tokens = scenario.batch * scenario.new_tokens
    activation_bytes = dtype_nbytes(scenario.activation_dtype)
    attention_logical = sum(
        _matrix_logical_bytes(
            matrix,
            tokens=tokens,
            activation_bytes=activation_bytes,
            scenario=scenario,
            int4_group_size=int4_group_size,
        )
        for matrix in attention_dense_matrices
    )
    conv_storage = weight_storage(
        conv_matrix.parameters,
        scenario.weight_format,
        group_size=int4_group_size,
    )
    attention_logical += 2 * tokens * conv_width * activation_bytes + conv_storage.total_bytes
    state_bytes = evaluate_expression(state_formula, bindings)
    attention_logical += 2 * int(state_bytes)
    ffn_logical = sum(
        _matrix_logical_bytes(
            matrix,
            tokens=tokens,
            activation_bytes=activation_bytes,
            scenario=scenario,
            int4_group_size=int4_group_size,
        )
        for matrix in ffn_matrices
    )
    total_logical = attention_logical + ffn_logical

    base = _base_assumptions(adapter_result)
    estimated = (
        "Gated DeltaNet core FLOPs use a useful-work estimate of 4*B*T*N_V*D_K*D_V",
        "state traffic assumes one ideal read and one write per invocation",
        "backend chunking, fusion, padding, and temporary workspaces are excluded",
    )
    formulas = {
        "parameters.total": (_symbol("P_TOTAL"), "parameters", MetricOrigin.EXACT),
        "parameters.resident": (_symbol("P_TOTAL"), "parameters", MetricOrigin.EXACT),
        "parameters.active": (_symbol("P_TOTAL"), "parameters", MetricOrigin.EXACT),
        "parameters.attention": (_symbol("P_ATTN"), "parameters", MetricOrigin.EXACT),
        "parameters.ffn": (_symbol("P_FFN"), "parameters", MetricOrigin.EXACT),
        "weights.storage_bytes": (
            _literal(storage.total_bytes),
            "bytes",
            MetricOrigin.EXACT,
        ),
        "flops.attention": (attention_flops, "FLOPs", MetricOrigin.ESTIMATED),
        "flops.ffn": (ffn_flops, "FLOPs", MetricOrigin.FORMULA),
        "flops": (total_flops, "FLOPs", MetricOrigin.ESTIMATED),
        "logical_bytes.attention": (
            _literal(attention_logical),
            "bytes",
            MetricOrigin.ESTIMATED,
        ),
        "logical_bytes.ffn": (_literal(ffn_logical), "bytes", MetricOrigin.FORMULA),
        "logical_bytes": (_literal(total_logical), "bytes", MetricOrigin.ESTIMATED),
        "kv_cache.bytes": (_literal(0), "bytes", MetricOrigin.EXACT),
        "kv_cache.bytes_per_sequence_token": (
            _literal(0),
            "bytes/token",
            MetricOrigin.EXACT,
        ),
        "recurrent_state.bytes": (
            recurrent_state_formula,
            "bytes",
            MetricOrigin.FORMULA,
        ),
        "conv_state.bytes": (conv_state_formula, "bytes", MetricOrigin.FORMULA),
        "state.bytes": (state_formula, "bytes", MetricOrigin.FORMULA),
        "arithmetic_intensity": (
            _divide(total_flops, _literal(total_logical)),
            "FLOP/byte",
            MetricOrigin.ESTIMATED,
        ),
    }
    metrics = []
    for name, (formula, unit, origin) in formulas.items():
        assumptions = list(base)
        if name == "weights.storage_bytes":
            assumptions.extend(_storage_assumptions(storage))
        if origin == MetricOrigin.ESTIMATED:
            assumptions.extend(estimated)
        if name in {"state.bytes", "recurrent_state.bytes", "conv_state.bytes"}:
            assumptions.append(f"state_dtype:{state_dtype.value}")
        if name.startswith("logical_bytes") or name == "arithmetic_intensity":
            assumptions.append("logical minimum traffic; not measured HBM traffic")
        metrics.append(
            _known_metric(
                subject_id=subject_id,
                scenario=scenario,
                name=name,
                unit=unit,
                formula=formula,
                bindings=bindings,
                origin=origin,
                assumptions=assumptions,
            )
        )
    return SubjectCost(
        scenario_id=scenario.id,
        subject_id=subject_id,
        module_path=module_path,
        layer_index=layer_index,
        metrics=tuple(metrics),
        bindings=bindings,
    )


def _glm_subject(
    *,
    scenario: Scenario,
    subject_id: str,
    module_path: str,
    layer_index: int,
    mlp_kind: str,
    adapter_result: AdapterResult,
    config: Mapping[str, Any],
    int4_group_size: int,
) -> SubjectCost:
    hidden_size = _positive_config_int(config, "hidden_size")
    dense_intermediate = _positive_config_int(config, "intermediate_size")
    expert_intermediate = _positive_config_int(config, "moe_intermediate_size")
    routed_experts = _positive_config_int(config, "n_routed_experts")
    top_k = _positive_config_int(config, "num_experts_per_tok")
    shared_experts = int(config.get("n_shared_experts", 0))
    if shared_experts < 0:
        raise CostModelError("n_shared_experts cannot be negative")

    if mlp_kind == "dense":
        matrices = (
            MatrixSpec("gate_proj", hidden_size, dense_intermediate),
            MatrixSpec("up_proj", hidden_size, dense_intermediate),
            MatrixSpec("down_proj", dense_intermediate, hidden_size),
        )
        non_quantized = 2 * hidden_size
        resident = sum(matrix.parameters for matrix in matrices) + non_quantized
        active = resident
    elif mlp_kind == "sparse":
        matrices = (
            MatrixSpec("router", hidden_size, routed_experts),
            MatrixSpec("routed_gate_up", hidden_size, expert_intermediate, 2 * routed_experts),
            MatrixSpec("routed_down", expert_intermediate, hidden_size, routed_experts),
            MatrixSpec("shared_gate_up", hidden_size, expert_intermediate, 2 * shared_experts),
            MatrixSpec("shared_down", expert_intermediate, hidden_size, shared_experts),
        )
        non_quantized = 2 * hidden_size
        resident = sum(matrix.parameters for matrix in matrices) + non_quantized
        active = (
            hidden_size * routed_experts
            + 3 * hidden_size * expert_intermediate * (top_k + shared_experts)
            + non_quantized
        )
    else:
        raise CostModelError(f"Unsupported GLM MLP layer kind {mlp_kind!r}")

    storage = _storage_for_matrices(
        matrices,
        non_quantized_parameters=non_quantized,
        scenario=scenario,
        int4_group_size=int4_group_size,
    )
    bindings = _scenario_bindings(
        scenario,
        H=hidden_size,
        I=dense_intermediate,
        I_E=expert_intermediate,
        E=routed_experts,
        K=top_k,
        E_SHARED=shared_experts,
        P_RESIDENT=resident,
        P_ACTIVE=active,
    )
    base = _base_assumptions(adapter_result)
    partial = (
        "coverage:config-proven MLP/MoE and layer norms only",
        "DSA/indexer attention parameters require a parameter inventory and are excluded",
    )
    metrics = [
        _known_metric(
            subject_id=subject_id,
            scenario=scenario,
            name="parameters.total",
            unit="parameters",
            formula=_symbol("P_RESIDENT"),
            bindings=bindings,
            origin=MetricOrigin.EXACT,
            assumptions=(*base, *partial),
            coverage=0.5,
        ),
        _known_metric(
            subject_id=subject_id,
            scenario=scenario,
            name="parameters.resident",
            unit="parameters",
            formula=_symbol("P_RESIDENT"),
            bindings=bindings,
            origin=MetricOrigin.EXACT,
            assumptions=(*base, *partial),
            coverage=0.5,
        ),
        _known_metric(
            subject_id=subject_id,
            scenario=scenario,
            name="parameters.active",
            unit="parameters",
            formula=_symbol("P_ACTIVE"),
            bindings=bindings,
            origin=MetricOrigin.FORMULA,
            assumptions=(
                *base,
                *partial,
                "active routed parameters use static top-k; no route histogram is available",
            ),
            coverage=0.5,
        ),
        _known_metric(
            subject_id=subject_id,
            scenario=scenario,
            name="weights.storage_bytes",
            unit="bytes",
            formula=_literal(storage.total_bytes),
            bindings=bindings,
            origin=MetricOrigin.EXACT,
            assumptions=(*base, *partial, *_storage_assumptions(storage)),
            coverage=0.5,
        ),
    ]
    unknown_reason = (
        *base,
        "GLM DSA and executed MoE costs are intentionally Unknown at config-only capability",
        "a representative-block capture or runtime route distribution is required",
    )
    for name, unit in (
        ("flops", "FLOPs"),
        ("logical_bytes", "bytes"),
        ("kv_cache.bytes", "bytes"),
        ("state.bytes", "bytes"),
        ("arithmetic_intensity", "FLOP/byte"),
    ):
        metrics.append(
            _unknown_metric(
                subject_id=subject_id,
                scenario=scenario,
                name=name,
                unit=unit,
                assumptions=unknown_reason,
            )
        )
    return SubjectCost(
        scenario_id=scenario.id,
        subject_id=subject_id,
        module_path=module_path,
        layer_index=layer_index,
        metrics=tuple(metrics),
        bindings=bindings,
    )


_AGGREGATE_NAMES = (
    "parameters.total",
    "parameters.resident",
    "parameters.active",
    "weights.storage_bytes",
    "flops",
    "logical_bytes",
    "kv_cache.bytes",
    "state.bytes",
    "arithmetic_intensity",
)


def _decoder_aggregate_scope(
    model_map: ModelMap,
    subjects: Sequence[SubjectCost],
) -> AggregateScope:
    """Describe decoder coverage without pretending it is an end-to-end cost ratio."""

    included_ids = tuple(subject.subject_id for subject in subjects)
    included_set = set(included_ids)
    parent_by_id = {instance.id: instance.parent_id for instance in model_map.instances}
    definition_kinds = {
        definition.id: definition.kind.value for definition in model_map.definitions
    }

    def is_inside_decoder_instance(instance_id: str) -> bool:
        parent_id = parent_by_id.get(instance_id)
        while parent_id is not None:
            if parent_id in included_set:
                return True
            parent_id = parent_by_id.get(parent_id)
        return False

    excluded_paths = tuple(
        sorted(
            instance.module_path
            for instance in model_map.instances
            if instance.id not in included_set
            and not is_inside_decoder_instance(instance.id)
            and definition_kinds.get(instance.definition_id) not in {"model", "stage"}
        )
    )
    denominator = len(included_ids) + len(excluded_paths)
    structural_coverage = len(included_ids) / denominator if denominator else 0.0
    return AggregateScope(
        name="decoder_layers",
        subject_id=model_map.model.id,
        included_instance_ids=included_ids,
        excluded_instance_paths=excluded_paths,
        structural_coverage=structural_coverage,
    )


def _aggregate_scope_assumptions(scope: AggregateScope) -> Tuple[str, ...]:
    excluded = ",".join(scope.excluded_instance_paths) or "none"
    total = scope.included_instance_count + scope.excluded_instance_count
    return (
        f"aggregate_scope:{scope.name}",
        "aggregate_value_is_decoder_only_not_whole_model",
        f"included_decoder_instances:{scope.included_instance_count}",
        f"excluded_model_instance_paths:{excluded}",
        (
            "coverage_basis:mean_decoder_metric_coverage*"
            f"structural_instance_fraction({scope.included_instance_count}/{total})"
        ),
        "coverage_is_structural_not_parameter_flop_byte_or_latency_fraction",
    )


def _aggregate_scenario_metrics(
    *,
    aggregate_scope: AggregateScope,
    scenario: Scenario,
    subjects: Sequence[SubjectCost],
) -> Tuple[Metric, ...]:
    aggregates = []
    scope_assumptions = _aggregate_scope_assumptions(aggregate_scope)
    for name in _AGGREGATE_NAMES:
        member_metrics = []
        for subject in subjects:
            try:
                member_metrics.append(subject.metric(name))
            except KeyError:
                pass
        unit = member_metrics[0].unit if member_metrics else "unknown"
        known = [metric for metric in member_metrics if metric.value is not None]
        if not known:
            aggregates.append(
                _unknown_metric(
                    subject_id=aggregate_scope.subject_id,
                    scenario=scenario,
                    name=name,
                    unit=unit,
                    assumptions=(
                        "provenance:aggregate:config-first-layer-metrics",
                        *scope_assumptions,
                        "aggregate contains no known values",
                        "Unknown values are not substituted with zero",
                    ),
                )
            )
            continue
        decoder_metric_coverage = (
            sum(metric.coverage for metric in member_metrics) / len(member_metrics)
            if member_metrics
            else 0.0
        )
        coverage = decoder_metric_coverage * aggregate_scope.structural_coverage
        if name == "arithmetic_intensity":
            try:
                aggregate_flops = next(metric for metric in aggregates if metric.name == "flops")
                aggregate_bytes = next(
                    metric for metric in aggregates if metric.name == "logical_bytes"
                )
            except StopIteration:
                value = None
            else:
                value = (
                    aggregate_flops.value / aggregate_bytes.value
                    if aggregate_flops.value is not None and aggregate_bytes.value not in (None, 0)
                    else None
                )
            if value is None:
                aggregates.append(
                    _unknown_metric(
                        subject_id=aggregate_scope.subject_id,
                        scenario=scenario,
                        name=name,
                        unit=unit,
                        assumptions=(
                            "provenance:aggregate:config-first-layer-metrics",
                            *scope_assumptions,
                            "aggregate FLOPs or logical bytes are Unknown",
                            "Unknown values are not substituted with zero",
                        ),
                    )
                )
                continue
        else:
            value = sum(metric.value for metric in known if metric.value is not None)

        origins = {metric.origin for metric in known}
        if MetricOrigin.ESTIMATED in origins:
            origin = MetricOrigin.ESTIMATED
        elif MetricOrigin.FORMULA in origins:
            origin = MetricOrigin.FORMULA
        else:
            origin = MetricOrigin.EXACT
        aggregates.append(
            _known_metric(
                subject_id=aggregate_scope.subject_id,
                scenario=scenario,
                name=name,
                unit=unit,
                formula=_literal(value),
                bindings={},
                origin=origin,
                assumptions=(
                    "provenance:aggregate:config-first-layer-metrics",
                    *scope_assumptions,
                    f"aggregate of {len(member_metrics)} decoder layer metrics",
                    "Unknown values are omitted, never substituted with zero",
                ),
                coverage=coverage,
            )
        )
    return tuple(aggregates)


def _validate_scenarios(scenarios: Sequence[Scenario]) -> None:
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise CostModelError("Scenario IDs must be unique")
    for scenario in scenarios:
        if scenario.phase not in {
            ScenarioPhase.PREFILL,
            ScenarioPhase.CHUNKED_PREFILL,
            ScenarioPhase.DECODE,
        }:
            raise CostModelError(f"Unsupported scenario phase: {scenario.phase.value}")
        if scenario.activation_dtype not in {DType.BFLOAT16, DType.FLOAT16}:
            raise CostModelError("M3 supports BF16/FP16 activations only")
        if scenario.kv_dtype not in {DType.BFLOAT16, DType.FLOAT16}:
            raise CostModelError("M3 supports BF16/FP16 KV cache only")
        if scenario.weight_format not in {
            WeightFormat.BFLOAT16,
            WeightFormat.FLOAT16,
            WeightFormat.INT4,
        }:
            raise CostModelError("M3 supports BF16, FP16, and groupwise INT4 weights")


def analyze_costs(
    config: Mapping[str, Any],
    adapter_result: AdapterResult,
    model_map: ModelMap,
    scenarios: Sequence[Scenario],
    *,
    int4_group_size: int = INT4_GROUP_SIZE,
) -> CostAnalysis:
    """Analyze supported config-first decoder layers for one or more workloads."""

    _validate_scenarios(scenarios)
    if int4_group_size <= 0 or int4_group_size % 2:
        raise CostModelError("int4_group_size must be a positive even integer")
    if adapter_result.revision != model_map.model.revision:
        raise CostModelError("Adapter result and Model Map revisions do not match")
    instance_ids = {instance.module_path: instance.id for instance in model_map.instances}
    missing_paths = {entry.instance_path for entry in adapter_result.layer_strip} - set(
        instance_ids
    )
    if missing_paths:
        raise CostModelError(
            f"Model Map is missing adapter layer instances: {sorted(missing_paths)!r}"
        )

    model_type = config.get("model_type")
    text_config: Mapping[str, Any]
    if model_type == "qwen3_5":
        nested = config.get("text_config")
        if not isinstance(nested, Mapping):
            raise CostModelError("Qwen config requires text_config")
        text_config = nested
    else:
        text_config = config

    scenario_costs = []
    for scenario in scenarios:
        subjects = []
        for entry in adapter_result.layer_strip:
            subject_id = instance_ids[entry.instance_path]
            if adapter_result.adapter_name == "tiny-dense":
                hidden_size = _positive_config_int(text_config, "hidden_size")
                intermediate_size = _positive_config_int(text_config, "intermediate_size")
                query_heads = _positive_config_int(text_config, "num_attention_heads")
                kv_heads = int(text_config.get("num_key_value_heads", query_heads))
                head_dim = int(text_config.get("head_dim", hidden_size // query_heads))
                if kv_heads <= 0 or head_dim <= 0:
                    raise CostModelError("Invalid Tiny attention dimensions")
                subjects.append(
                    _standard_attention_subject(
                        scenario=scenario,
                        subject_id=subject_id,
                        module_path=entry.instance_path,
                        layer_index=entry.layer_index,
                        adapter_result=adapter_result,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        query_heads=query_heads,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        q_projection_multiplier=1,
                        attention_bias=bool(text_config.get("attention_bias", False)),
                        include_qk_norm=False,
                        int4_group_size=int4_group_size,
                    )
                )
            elif adapter_result.adapter_name == "qwen3-5":
                if entry.attention_kind == "linear_attention":
                    subjects.append(
                        _qwen_linear_subject(
                            scenario=scenario,
                            subject_id=subject_id,
                            module_path=entry.instance_path,
                            layer_index=entry.layer_index,
                            adapter_result=adapter_result,
                            text_config=text_config,
                            int4_group_size=int4_group_size,
                        )
                    )
                elif entry.attention_kind == "full_attention":
                    subjects.append(
                        _standard_attention_subject(
                            scenario=scenario,
                            subject_id=subject_id,
                            module_path=entry.instance_path,
                            layer_index=entry.layer_index,
                            adapter_result=adapter_result,
                            hidden_size=_positive_config_int(text_config, "hidden_size"),
                            intermediate_size=_positive_config_int(
                                text_config, "intermediate_size"
                            ),
                            query_heads=_positive_config_int(text_config, "num_attention_heads"),
                            kv_heads=_positive_config_int(text_config, "num_key_value_heads"),
                            head_dim=_positive_config_int(text_config, "head_dim"),
                            q_projection_multiplier=2,
                            attention_bias=bool(text_config.get("attention_bias", False)),
                            include_qk_norm=True,
                            int4_group_size=int4_group_size,
                        )
                    )
                else:
                    raise CostModelError(
                        f"Unsupported Qwen attention kind {entry.attention_kind!r}"
                    )
            elif adapter_result.adapter_name == "glm-moe-dsa":
                subjects.append(
                    _glm_subject(
                        scenario=scenario,
                        subject_id=subject_id,
                        module_path=entry.instance_path,
                        layer_index=entry.layer_index,
                        mlp_kind=entry.mlp_kind,
                        adapter_result=adapter_result,
                        config=text_config,
                        int4_group_size=int4_group_size,
                    )
                )
            else:
                raise CostModelError(f"No M3 cost adapter for {adapter_result.adapter_name!r}")
        aggregate_scope = _decoder_aggregate_scope(model_map, subjects)
        aggregate_metrics = _aggregate_scenario_metrics(
            aggregate_scope=aggregate_scope,
            scenario=scenario,
            subjects=subjects,
        )
        scenario_costs.append(
            ScenarioCost(
                scenario=scenario,
                subjects=tuple(subjects),
                aggregate_metrics=aggregate_metrics,
                aggregate_scope=aggregate_scope,
            )
        )
    return CostAnalysis(
        model_id=model_map.model.id,
        model_revision=model_map.model.revision,
        scenario_costs=tuple(scenario_costs),
    )
