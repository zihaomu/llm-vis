from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from llm_vis.cost import (
    attention_core_flops,
    causal_attention_pairs,
    sliding_window_attention_pairs,
    weight_storage,
)


@given(
    new_tokens=st.integers(min_value=0, max_value=4096),
    past_tokens=st.integers(min_value=0, max_value=262_144),
)
def test_causal_pairs_match_closed_form_and_monotonicity(new_tokens: int, past_tokens: int) -> None:
    expected = new_tokens * past_tokens + new_tokens * (new_tokens + 1) // 2

    assert causal_attention_pairs(new_tokens, past_tokens) == expected
    assert causal_attention_pairs(new_tokens, past_tokens + 1) >= expected


@given(
    new_tokens=st.integers(min_value=0, max_value=256),
    past_tokens=st.integers(min_value=0, max_value=4096),
    window=st.integers(min_value=1, max_value=4096),
)
def test_sliding_window_pairs_match_direct_definition(
    new_tokens: int, past_tokens: int, window: int
) -> None:
    expected = sum(min(past_tokens + token_index + 1, window) for token_index in range(new_tokens))

    assert sliding_window_attention_pairs(new_tokens, past_tokens, window) == expected
    assert expected <= causal_attention_pairs(new_tokens, past_tokens)


@given(
    logical_values=st.integers(min_value=0, max_value=10_000_000),
    group_size=st.integers(min_value=1, max_value=256).map(lambda value: value * 2),
)
def test_groupwise_int4_storage_never_undercounts_packing_or_metadata(
    logical_values: int, group_size: int
) -> None:
    storage = weight_storage(logical_values, "int4", group_size=group_size)
    groups = (logical_values + group_size - 1) // group_size

    assert storage.packed_weight_bytes == groups * group_size // 2
    assert storage.scale_metadata_bytes == groups * 2
    assert storage.total_bytes >= (logical_values + 1) // 2
    assert storage.padding_values == groups * group_size - logical_values


@given(
    batch=st.integers(min_value=1, max_value=64),
    query_heads=st.integers(min_value=1, max_value=256),
    head_dim=st.integers(min_value=1, max_value=512),
    new_tokens=st.integers(min_value=0, max_value=256),
    past_tokens=st.integers(min_value=0, max_value=4096),
)
def test_attention_core_flops_is_exactly_four_flops_per_attended_scalar(
    batch: int,
    query_heads: int,
    head_dim: int,
    new_tokens: int,
    past_tokens: int,
) -> None:
    pairs = causal_attention_pairs(new_tokens, past_tokens)

    assert (
        attention_core_flops(
            batch=batch,
            query_heads=query_heads,
            head_dim=head_dim,
            pairs_per_sequence=pairs,
        )
        == 4 * batch * query_heads * head_dim * pairs
    )
