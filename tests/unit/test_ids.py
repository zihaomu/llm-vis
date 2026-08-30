from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_vis.ir import (  # noqa: E402
    artifact_local_id,
    canonical_json,
    canonical_semantic_key,
    deterministic_id,
    scenario_id,
)


def test_canonical_json_is_order_and_whitespace_independent() -> None:
    left = {"z": [1, {"beta": "模型", "alpha": True}], "a": None}
    right = {"a": None, "z": [1, {"alpha": True, "beta": "模型"}]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":null,"z":[1,{"alpha":true,"beta":"模型"}]}'
    assert deterministic_id("test", left) == deterministic_id("test", right)


def test_canonical_json_rejects_non_json_and_non_finite_values() -> None:
    with pytest.raises(TypeError, match="not JSON-compatible"):
        canonical_json({"bad": object()})
    with pytest.raises(ValueError):
        canonical_json({"bad": math.nan})


def test_artifact_local_id_is_stable_and_sensitive_to_capture_identity() -> None:
    common = {
        "model_revision": "abc123",
        "framework_domain": "aten",
        "module_path": "model.layers.0.self_attn",
        "capture_variant": "decode",
    }
    first = artifact_local_id(**common, op_ordinal=3)
    repeated = artifact_local_id(**common, op_ordinal=3)
    next_op = artifact_local_id(**common, op_ordinal=4)

    assert first == repeated
    assert first != next_op
    assert first.startswith("art_")
    assert len(first.split("_", 1)[1]) == 24


def test_canonical_semantic_key_has_no_revision_input() -> None:
    key = canonical_semantic_key(
        semantic_path="decoder.block.attention",
        definition_signature={"kind": "attention", "heads": 32},
        symbolic_contract={"input": ["B", "T", "H"]},
        structural_signature=["norm", "qkv", "rope", "sdpa", "out"],
    )
    reordered = canonical_semantic_key(
        semantic_path="decoder.block.attention",
        definition_signature={"heads": 32, "kind": "attention"},
        symbolic_contract={"input": ["B", "T", "H"]},
        structural_signature=["norm", "qkv", "rope", "sdpa", "out"],
    )

    assert key == reordered
    assert key.startswith("sem_")


def test_scenario_id_ignores_existing_id_and_mapping_order() -> None:
    fields = {
        "phase": "decode",
        "batch": 8,
        "new_tokens": 1,
        "past_tokens": 1024,
    }
    with_id = {"id": "old-id", **fields}
    reordered = dict(reversed(list(fields.items())))

    assert scenario_id(fields) == scenario_id(with_id)
    assert scenario_id(fields) == scenario_id(reordered)


def test_deterministic_id_validates_namespace_and_digest_length() -> None:
    with pytest.raises(ValueError, match="namespace"):
        deterministic_id("bad namespace", {})
    with pytest.raises(ValueError, match="digest_length"):
        deterministic_id("ok", {}, digest_length=7)
