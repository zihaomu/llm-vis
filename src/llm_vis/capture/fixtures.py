"""Project-owned tiny blocks used for safe representative capture.

The fixtures have meta-device parameters and inputs.  They never read a model
repository, call ``torch.load``, or materialize parameter storage.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, Tuple

from .types import (
    BuiltTinyRepresentative,
    InputContract,
    RepresentativeKind,
    TinyRepresentativeSpec,
)

_HIDDEN_SIZE = 16
_SEQUENCE_LENGTH = 4
_BATCH_SIZE = 1


TINY_REPRESENTATIVE_SPECS: Dict[RepresentativeKind, TinyRepresentativeSpec] = {
    RepresentativeKind.DENSE: TinyRepresentativeSpec(
        kind=RepresentativeKind.DENSE,
        name="TinyDenseBlock",
        module_path="llm_vis.fixtures.dense",
        hidden_size=_HIDDEN_SIZE,
        inputs=(InputContract("hidden_states", (_BATCH_SIZE, _SEQUENCE_LENGTH, _HIDDEN_SIZE)),),
    ),
    RepresentativeKind.FULL_ATTENTION: TinyRepresentativeSpec(
        kind=RepresentativeKind.FULL_ATTENTION,
        name="TinyFullAttentionBlock",
        module_path="llm_vis.fixtures.full_attention",
        hidden_size=_HIDDEN_SIZE,
        inputs=(InputContract("hidden_states", (_BATCH_SIZE, _SEQUENCE_LENGTH, _HIDDEN_SIZE)),),
    ),
    RepresentativeKind.LINEAR_STATE: TinyRepresentativeSpec(
        kind=RepresentativeKind.LINEAR_STATE,
        name="TinyLinearStateBlock",
        module_path="llm_vis.fixtures.linear_state",
        hidden_size=_HIDDEN_SIZE,
        inputs=(
            InputContract("hidden_states", (_BATCH_SIZE, _SEQUENCE_LENGTH, _HIDDEN_SIZE)),
            InputContract("recurrent_state", (_BATCH_SIZE, _HIDDEN_SIZE, _HIDDEN_SIZE), "state"),
        ),
    ),
}


def available_tiny_representatives() -> Tuple[TinyRepresentativeSpec, ...]:
    return tuple(TINY_REPRESENTATIVE_SPECS[kind] for kind in RepresentativeKind)


def _load_torch() -> Any:
    """Import torch lazily so importing ``llm_vis.capture`` remains optional."""

    return importlib.import_module("torch")


def _build_dense(torch: Any, spec: TinyRepresentativeSpec) -> Any:
    class TinyDenseBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = spec.hidden_size
            self.norm = torch.nn.LayerNorm(hidden, device="meta")
            self.gate = torch.nn.Linear(hidden, hidden * 2, bias=False, device="meta")
            self.up = torch.nn.Linear(hidden, hidden * 2, bias=False, device="meta")
            self.down = torch.nn.Linear(hidden * 2, hidden, bias=False, device="meta")

        def forward(self, hidden_states: Any) -> Any:
            normalized = self.norm(hidden_states)
            activated = torch.nn.functional.silu(self.gate(normalized))
            return hidden_states + self.down(activated * self.up(normalized))

    return TinyDenseBlock()


def _build_full_attention(torch: Any, spec: TinyRepresentativeSpec) -> Any:
    class TinyFullAttentionBlock(torch.nn.Module):
        num_heads = 4

        def __init__(self) -> None:
            super().__init__()
            hidden = spec.hidden_size
            self.head_dim = hidden // self.num_heads
            self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")
            self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")
            self.v_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")
            self.o_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")

        def _heads(self, tensor: Any) -> Any:
            batch, tokens, hidden = tensor.shape
            return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        def forward(self, hidden_states: Any) -> Any:
            query = self._heads(self.q_proj(hidden_states))
            key = self._heads(self.k_proj(hidden_states))
            value = self._heads(self.v_proj(hidden_states))
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            probabilities = torch.softmax(scores, dim=-1)
            context = torch.matmul(probabilities, value)
            batch, _, tokens, _ = context.shape
            merged = context.transpose(1, 2).reshape(batch, tokens, spec.hidden_size)
            return self.o_proj(merged)

    return TinyFullAttentionBlock()


def _build_linear_state(torch: Any, spec: TinyRepresentativeSpec) -> Any:
    class TinyLinearStateBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = spec.hidden_size
            self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")
            self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")
            self.v_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")
            self.out_proj = torch.nn.Linear(hidden, hidden, bias=False, device="meta")

        def forward(self, hidden_states: Any, recurrent_state: Any) -> Any:
            query = torch.sigmoid(self.q_proj(hidden_states))
            key = torch.sigmoid(self.k_proj(hidden_states))
            value = self.v_proj(hidden_states)
            update = torch.einsum("bth,btk->bhk", key, value)
            next_state = recurrent_state * 0.9375 + update
            output = torch.einsum("bth,bhk->btk", query, next_state)
            return self.out_proj(output), next_state

    return TinyLinearStateBlock()


_BUILDERS = {
    RepresentativeKind.DENSE: _build_dense,
    RepresentativeKind.FULL_ATTENTION: _build_full_attention,
    RepresentativeKind.LINEAR_STATE: _build_linear_state,
}


def build_tiny_representative(
    kind: RepresentativeKind,
    *,
    torch_module: Any = None,
) -> BuiltTinyRepresentative:
    """Build one fixture entirely on the meta device."""

    resolved_kind = RepresentativeKind(kind)
    torch = torch_module if torch_module is not None else _load_torch()
    spec = TINY_REPRESENTATIVE_SPECS[resolved_kind]
    module = _BUILDERS[resolved_kind](torch, spec)
    module.eval()
    module._llm_vis_tiny_fixture = True
    module._llm_vis_representative_kind = resolved_kind.value
    args = tuple(
        torch.empty(contract.shape, dtype=torch.float32, device="meta") for contract in spec.inputs
    )

    parameters = tuple(module.parameters())
    if any(not parameter.is_meta for parameter in parameters):
        raise RuntimeError("tiny capture fixture unexpectedly materialized parameter storage")
    if any(getattr(argument, "device", None).type != "meta" for argument in args):
        raise RuntimeError("tiny capture fixture inputs must remain on the meta device")
    return BuiltTinyRepresentative(spec=spec, module=module, args=args)
