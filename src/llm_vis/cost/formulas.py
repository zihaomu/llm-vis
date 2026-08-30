"""Integer reference formulas from the LLM-Vis implementation plan.

These functions describe useful algorithm work and logical minimum traffic. They do
not predict latency or physical HBM traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Union

Number = Union[int, float]


def _non_negative(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive(name: str, value: int) -> int:
    _non_negative(name, value)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class DenseLinearCost:
    macs: int
    flops: int
    logical_bytes: int


@dataclass(frozen=True)
class KvCacheCost:
    bytes_per_sequence_token: int
    bytes_per_decode_step_batch: int
    total_bytes: int
    write_logical_bytes: int
    decode_read_logical_bytes: int


@dataclass(frozen=True)
class Aggregate:
    """A partial aggregation that never silently substitutes zero for unknown data."""

    value: Number | None
    known_count: int
    total_count: int

    @property
    def coverage(self) -> float:
        return 1.0 if self.total_count == 0 else self.known_count / self.total_count

    @property
    def complete(self) -> bool:
        return self.known_count == self.total_count


def dense_linear_cost(
    m: int,
    k: int,
    n: int,
    *,
    activation_bytes: int,
    weight_bytes: int,
    output_bytes: int | None = None,
) -> DenseLinearCost:
    """Return MACs, FLOPs, and logical minimum bytes for ``[M,K] @ [K,N]``."""

    m = _non_negative("m", m)
    k = _positive("k", k)
    n = _positive("n", n)
    activation_bytes = _positive("activation_bytes", activation_bytes)
    weight_bytes = _positive("weight_bytes", weight_bytes)
    output_bytes = (
        activation_bytes if output_bytes is None else _positive("output_bytes", output_bytes)
    )
    macs = m * k * n
    return DenseLinearCost(
        macs=macs,
        flops=2 * macs,
        logical_bytes=(m * k * activation_bytes) + (k * n * weight_bytes) + (m * n * output_bytes),
    )


def causal_attention_pairs(new_tokens: int, past_tokens: int) -> int:
    """Count valid causal query-key pairs for one sequence in one invocation."""

    new_tokens = _non_negative("new_tokens", new_tokens)
    past_tokens = _non_negative("past_tokens", past_tokens)
    return new_tokens * past_tokens + new_tokens * (new_tokens + 1) // 2


def sliding_window_attention_pairs(new_tokens: int, past_tokens: int, window: int) -> int:
    """Count pairs for ``sum(min(L+i+1, W))`` without iterating over tokens."""

    new_tokens = _non_negative("new_tokens", new_tokens)
    past_tokens = _non_negative("past_tokens", past_tokens)
    window = _positive("window", window)
    unsaturated = min(new_tokens, max(0, window - past_tokens))
    arithmetic_prefix = unsaturated * (past_tokens + 1) + unsaturated * (unsaturated - 1) // 2
    return arithmetic_prefix + (new_tokens - unsaturated) * window


def attention_core_flops(
    *, batch: int, query_heads: int, head_dim: int, pairs_per_sequence: int
) -> int:
    """Return useful QK^T + PV FLOPs for standard attention."""

    batch = _positive("batch", batch)
    query_heads = _positive("query_heads", query_heads)
    head_dim = _positive("head_dim", head_dim)
    pairs_per_sequence = _non_negative("pairs_per_sequence", pairs_per_sequence)
    return 4 * batch * query_heads * head_dim * pairs_per_sequence


def attention_core_logical_bytes(
    *,
    batch: int,
    new_tokens: int,
    total_visible_tokens: int,
    query_width: int,
    key_width: int,
    value_width: int,
    activation_bytes: int,
    kv_bytes: int,
) -> int:
    """Return ideal input/output traffic for a standard attention core.

    Scores and backend workspaces are deliberately excluded. Q and output use the
    activation dtype; K/V use the configured cache dtype.
    """

    batch = _positive("batch", batch)
    new_tokens = _non_negative("new_tokens", new_tokens)
    total_visible_tokens = _non_negative("total_visible_tokens", total_visible_tokens)
    query_width = _positive("query_width", query_width)
    key_width = _positive("key_width", key_width)
    value_width = _positive("value_width", value_width)
    activation_bytes = _positive("activation_bytes", activation_bytes)
    kv_bytes = _positive("kv_bytes", kv_bytes)
    query_and_output = 2 * batch * new_tokens * query_width * activation_bytes
    key_and_value = batch * total_visible_tokens * (key_width + value_width) * kv_bytes
    return query_and_output + key_and_value


def swiglu_flops(*, batch: int, new_tokens: int, hidden_size: int, intermediate_size: int) -> int:
    """Return projection FLOPs for gate/up/down SwiGLU, excluding elementwise work."""

    batch = _positive("batch", batch)
    new_tokens = _non_negative("new_tokens", new_tokens)
    hidden_size = _positive("hidden_size", hidden_size)
    intermediate_size = _positive("intermediate_size", intermediate_size)
    return 6 * batch * new_tokens * hidden_size * intermediate_size


def kv_cache_cost(
    *,
    batch: int,
    new_tokens: int,
    past_tokens: int,
    kv_layers: int,
    kv_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    key_bytes: int,
    value_bytes: int | None = None,
) -> KvCacheCost:
    """Return logical KV capacity/read/write values for homogeneous KV layers."""

    batch = _positive("batch", batch)
    new_tokens = _non_negative("new_tokens", new_tokens)
    past_tokens = _non_negative("past_tokens", past_tokens)
    kv_layers = _non_negative("kv_layers", kv_layers)
    kv_heads = _positive("kv_heads", kv_heads)
    key_head_dim = _positive("key_head_dim", key_head_dim)
    value_head_dim = _positive("value_head_dim", value_head_dim)
    key_bytes = _positive("key_bytes", key_bytes)
    value_bytes = key_bytes if value_bytes is None else _positive("value_bytes", value_bytes)

    per_layer_token = kv_heads * (key_head_dim * key_bytes + value_head_dim * value_bytes)
    per_sequence_token = kv_layers * per_layer_token
    return KvCacheCost(
        bytes_per_sequence_token=per_sequence_token,
        bytes_per_decode_step_batch=batch * per_sequence_token,
        total_bytes=batch * per_sequence_token * (past_tokens + new_tokens),
        write_logical_bytes=batch * new_tokens * per_sequence_token,
        decode_read_logical_bytes=batch * (past_tokens + 1) * per_sequence_token,
    )


def aggregate_known(values: Iterable[Number | None]) -> Aggregate:
    """Sum known values while returning explicit coverage for missing values."""

    known = []
    total = 0
    for value in values:
        total += 1
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("metric values must be numeric or None")
        known.append(value)
    return Aggregate(
        value=sum(known) if known else None,
        known_count=len(known),
        total_count=total,
    )
