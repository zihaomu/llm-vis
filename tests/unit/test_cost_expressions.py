from __future__ import annotations

import math

import pytest

from llm_vis.cost import (
    ExpressionEvaluationError,
    SymbolBindingError,
    evaluate_expression,
    required_symbols,
    validate_symbol_bindings,
)
from llm_vis.ir import ExpressionOp, Symbol, SymbolicExpression


def _literal(value: int | float) -> SymbolicExpression:
    return SymbolicExpression.literal(value)


def _binary(
    op: ExpressionOp, left: SymbolicExpression, right: SymbolicExpression
) -> SymbolicExpression:
    return SymbolicExpression(op=op, args=[left, right])


def test_required_symbols_and_catalog_bounds() -> None:
    expression = _binary(
        ExpressionOp.MULTIPLY,
        SymbolicExpression.symbol_ref("B"),
        SymbolicExpression.symbol_ref("T"),
    )
    symbols = [
        Symbol(id="b", name="B", lower_bound=1, upper_bound=32),
        Symbol(id="t", name="T", lower_bound=1, upper_bound=4096),
    ]

    assert required_symbols(expression) == frozenset({"B", "T"})
    validate_symbol_bindings(expression, {"B": 2, "T": 8}, symbols=symbols)
    assert evaluate_expression(expression, {"B": 2, "T": 8}, symbols=symbols) == 16

    with pytest.raises(SymbolBindingError, match="Missing"):
        validate_symbol_bindings(expression, {"B": 2}, symbols=symbols)
    with pytest.raises(SymbolBindingError, match="above upper bound"):
        validate_symbol_bindings(expression, {"B": 2, "T": 4097}, symbols=symbols)
    with pytest.raises(SymbolBindingError, match="Unexpected"):
        validate_symbol_bindings(
            expression,
            {"B": 2, "T": 8, "unused": 1},
            symbols=symbols,
            allow_extra=False,
        )


@pytest.mark.parametrize("invalid", [True, float("inf"), float("-inf"), float("nan")])
def test_rejects_non_numeric_or_non_finite_bindings(invalid: object) -> None:
    expression = SymbolicExpression.symbol_ref("X")
    with pytest.raises(SymbolBindingError):
        evaluate_expression(expression, {"X": invalid})  # type: ignore[dict-item]


def test_all_binary_operators_match_integer_reference_exhaustively() -> None:
    for left in range(-7, 8):
        for right in range(-7, 8):
            values = {"X": left, "Y": right}
            x = SymbolicExpression.symbol_ref("X")
            y = SymbolicExpression.symbol_ref("Y")
            assert evaluate_expression(_binary(ExpressionOp.ADD, x, y), values) == left + right
            assert evaluate_expression(_binary(ExpressionOp.SUBTRACT, x, y), values) == left - right
            assert evaluate_expression(_binary(ExpressionOp.MULTIPLY, x, y), values) == left * right
            assert evaluate_expression(_binary(ExpressionOp.MINIMUM, x, y), values) == min(
                left, right
            )
            assert evaluate_expression(_binary(ExpressionOp.MAXIMUM, x, y), values) == max(
                left, right
            )
            if right:
                assert (
                    evaluate_expression(_binary(ExpressionOp.DIVIDE, x, y), values) == left / right
                )
                assert (
                    evaluate_expression(_binary(ExpressionOp.FLOOR_DIVIDE, x, y), values)
                    == left // right
                )
                assert evaluate_expression(
                    _binary(ExpressionOp.CEIL_DIVIDE, x, y), values
                ) == math.ceil(left / right)


def test_variadic_and_negate_operators() -> None:
    add = SymbolicExpression(
        op=ExpressionOp.ADD,
        args=[_literal(1), _literal(2), _literal(3)],
    )
    multiply = SymbolicExpression(
        op=ExpressionOp.MULTIPLY,
        args=[_literal(2), _literal(3), _literal(4)],
    )
    negate = SymbolicExpression(op=ExpressionOp.NEGATE, args=[_literal(5)])
    assert evaluate_expression(add, {}) == 6
    assert evaluate_expression(multiply, {}) == 24
    assert evaluate_expression(negate, {}) == -5


def test_division_by_zero_is_a_structured_error() -> None:
    expression = _binary(ExpressionOp.CEIL_DIVIDE, _literal(1), _literal(0))
    with pytest.raises(ExpressionEvaluationError, match="Division by zero"):
        evaluate_expression(expression, {})
