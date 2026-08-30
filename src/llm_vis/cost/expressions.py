"""Safe evaluation and binding validation for Model Map symbolic expressions."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Mapping

from llm_vis.ir.models import ExpressionOp, Number, Symbol, SymbolicExpression


class SymbolBindingError(ValueError):
    """Raised when expression bindings are missing, invalid, or out of range."""


class ExpressionEvaluationError(ValueError):
    """Raised when a valid expression cannot be evaluated mathematically."""


def required_symbols(expression: SymbolicExpression) -> frozenset[str]:
    """Return every symbol referenced by ``expression``."""

    names = set()
    stack = [expression]
    while stack:
        current = stack.pop()
        if current.op == ExpressionOp.SYMBOL:
            assert current.symbol is not None
            names.add(current.symbol)
        stack.extend(current.args)
    return frozenset(names)


def _validate_number(name: str, value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SymbolBindingError(f"Binding {name!r} must be a finite real number")
    if not math.isfinite(value):
        raise SymbolBindingError(f"Binding {name!r} must be finite")
    return value


def validate_symbol_bindings(
    expression: SymbolicExpression,
    bindings: Mapping[str, Number],
    *,
    symbols: Iterable[Symbol] = (),
    allow_extra: bool = True,
) -> None:
    """Validate required names, numeric values, and optional symbol bounds."""

    required = required_symbols(expression)
    missing = required - set(bindings)
    if missing:
        raise SymbolBindingError(f"Missing symbol bindings: {sorted(missing)!r}")

    catalog = {}
    for symbol in symbols:
        if symbol.name in catalog:
            raise SymbolBindingError(f"Duplicate symbol definition {symbol.name!r}")
        catalog[symbol.name] = symbol
    if catalog:
        undefined = required - set(catalog)
        if undefined:
            raise SymbolBindingError(
                f"Expression references symbols absent from the catalog: {sorted(undefined)!r}"
            )

    allowed = set(catalog) if catalog else required
    if not allow_extra:
        extra = set(bindings) - allowed
        if extra:
            raise SymbolBindingError(f"Unexpected symbol bindings: {sorted(extra)!r}")

    for name, raw_value in bindings.items():
        value = _validate_number(name, raw_value)
        definition = catalog.get(name)
        if definition is None:
            continue
        if definition.lower_bound is not None and value < definition.lower_bound:
            raise SymbolBindingError(
                f"Binding {name!r}={value} is below lower bound {definition.lower_bound}"
            )
        if definition.upper_bound is not None and value > definition.upper_bound:
            raise SymbolBindingError(
                f"Binding {name!r}={value} is above upper bound {definition.upper_bound}"
            )


def evaluate_expression(
    expression: SymbolicExpression,
    bindings: Mapping[str, Number],
    *,
    symbols: Iterable[Symbol] = (),
    allow_extra: bool = True,
) -> Number:
    """Evaluate all IR expression operators without using ``eval`` or Python code."""

    validate_symbol_bindings(
        expression,
        bindings,
        symbols=symbols,
        allow_extra=allow_extra,
    )

    def visit(node: SymbolicExpression) -> Number:
        if node.op == ExpressionOp.LITERAL:
            assert node.value is not None
            return node.value
        if node.op == ExpressionOp.SYMBOL:
            assert node.symbol is not None
            return bindings[node.symbol]

        values = [visit(argument) for argument in node.args]
        try:
            if node.op == ExpressionOp.ADD:
                result = sum(values)
            elif node.op == ExpressionOp.SUBTRACT:
                result = values[0] - values[1]
            elif node.op == ExpressionOp.MULTIPLY:
                result = 1
                for value in values:
                    result *= value
            elif node.op == ExpressionOp.DIVIDE:
                result = values[0] / values[1]
            elif node.op == ExpressionOp.FLOOR_DIVIDE:
                result = values[0] // values[1]
            elif node.op == ExpressionOp.CEIL_DIVIDE:
                numerator, denominator = values
                if isinstance(numerator, int) and isinstance(denominator, int):
                    quotient, remainder = divmod(numerator, denominator)
                    result = quotient + int(remainder != 0)
                else:
                    result = math.ceil(numerator / denominator)
            elif node.op == ExpressionOp.MINIMUM:
                result = min(values)
            elif node.op == ExpressionOp.MAXIMUM:
                result = max(values)
            elif node.op == ExpressionOp.NEGATE:
                result = -values[0]
            else:  # pragma: no cover - enum exhaustiveness guard
                raise ExpressionEvaluationError(f"Unsupported expression op {node.op.value!r}")
        except ZeroDivisionError as exc:
            raise ExpressionEvaluationError(
                f"Division by zero while evaluating {node.op.value}"
            ) from exc
        if isinstance(result, bool) or not isinstance(result, Real) or not math.isfinite(result):
            raise ExpressionEvaluationError(
                f"Expression {node.op.value!r} produced a non-finite result"
            )
        return result

    return visit(expression)
