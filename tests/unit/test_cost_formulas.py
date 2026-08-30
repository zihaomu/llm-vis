from __future__ import annotations

import pytest

from llm_vis.cost import (
    aggregate_known,
    attention_core_flops,
    causal_attention_pairs,
    dense_linear_cost,
    kv_cache_cost,
    sliding_window_attention_pairs,
    swiglu_flops,
)


def test_dense_linear_reference_formula() -> None:
    cost = dense_linear_cost(2, 3, 5, activation_bytes=2, weight_bytes=1)
    assert cost.macs == 30
    assert cost.flops == 60
    assert cost.logical_bytes == (2 * 3 * 2) + (3 * 5) + (2 * 5 * 2)


@pytest.mark.parametrize(
    ("new_tokens", "past_tokens", "expected"),
    [(1, 0, 1), (1, 127, 128), (4, 0, 10), (4, 8, 42)],
)
def test_causal_attention_pairs(new_tokens: int, past_tokens: int, expected: int) -> None:
    assert causal_attention_pairs(new_tokens, past_tokens) == expected


def test_sliding_window_formula_matches_direct_reference() -> None:
    for new_tokens in range(0, 12):
        for past_tokens in range(0, 12):
            for window in range(1, 9):
                expected = sum(min(past_tokens + index + 1, window) for index in range(new_tokens))
                assert sliding_window_attention_pairs(new_tokens, past_tokens, window) == expected


def test_attention_and_swiglu_flops() -> None:
    pairs = causal_attention_pairs(1, 127)
    assert (
        attention_core_flops(batch=2, query_heads=8, head_dim=64, pairs_per_sequence=pairs)
        == 4 * 2 * 8 * 64 * 128
    )
    assert swiglu_flops(batch=2, new_tokens=4, hidden_size=32, intermediate_size=64) == (
        6 * 2 * 4 * 32 * 64
    )


def test_kv_cache_uses_only_kv_layers_and_heads() -> None:
    cost = kv_cache_cost(
        batch=2,
        new_tokens=1,
        past_tokens=7,
        kv_layers=3,
        kv_heads=2,
        key_head_dim=4,
        value_head_dim=6,
        key_bytes=2,
    )
    per_sequence_token = 3 * 2 * ((4 * 2) + (6 * 2))
    assert cost.bytes_per_sequence_token == per_sequence_token
    assert cost.total_bytes == 2 * per_sequence_token * 8
    assert cost.write_logical_bytes == 2 * per_sequence_token
    assert cost.decode_read_logical_bytes == 2 * 8 * per_sequence_token


def test_unknown_is_not_aggregated_as_zero() -> None:
    partial = aggregate_known([4, None, 6])
    assert partial.value == 10
    assert partial.coverage == pytest.approx(2 / 3)
    assert partial.complete is False

    unknown = aggregate_known([None, None])
    assert unknown.value is None
    assert unknown.coverage == 0


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        dense_linear_cost(1, 0, 1, activation_bytes=2, weight_bytes=2)
