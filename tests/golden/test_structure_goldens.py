from __future__ import annotations

import json
from pathlib import Path

from llm_vis.adapters import build_from_config
from llm_vis.ir import ModelMap

ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "tests" / "fixtures" / "configs"
GOLDEN_DIR = ROOT / "tests" / "golden" / "structure"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_structure_goldens_parse_and_match_current_adapters() -> None:
    for name in ("tiny_dense", "qwen3_8_27b", "glm_5_3_bf16"):
        golden_payload = _load(GOLDEN_DIR / name / "model-map.json")
        golden = ModelMap.model_validate(golden_payload)
        config = _load(CONFIG_DIR / f"{name}.json")
        provenance = _load(CONFIG_DIR / f"{name}.provenance.json")
        current = build_from_config(
            config,
            model_id=provenance["model_id"],
            revision=provenance["revision"],
        ).to_model_map()

        assert golden.model_dump(mode="json") == current.model_dump(mode="json")


def test_acceptance_structure_summaries() -> None:
    qwen = _load(GOLDEN_DIR / "qwen3_8_27b" / "summary.json")
    assert qwen["layer_count"] == 64
    assert qwen["layer_labels"] == "LLLA" * 16
    assert qwen["semantic_kinds"]["linear_attention"] == 48
    assert qwen["semantic_kinds"]["kv_cache"] == 16
    assert qwen["semantic_kinds"]["recurrent_state"] == 48
    qwen_strip = _load(GOLDEN_DIR / "qwen3_8_27b" / "layer-strip.json")
    assert all(item["expected_pattern"] == item["attention_kind"] for item in qwen_strip)
    assert not any(item["anomaly"] for item in qwen_strip)
    assert all(item["reason"] is None for item in qwen_strip)

    glm = _load(GOLDEN_DIR / "glm_5_3_bf16" / "summary.json")
    assert glm["layer_count"] == 78
    assert glm["layer_labels"] == "DDD" + "M" * 75
    assert glm["metadata"]["routed_experts"] == 256
    assert glm["metadata"]["top_k"] == 8
    assert glm["metadata"]["experts_materialized"] == 0
    glm_strip = _load(GOLDEN_DIR / "glm_5_3_bf16" / "layer-strip.json")
    assert all(item["expected_pattern"] == item["mlp_kind"] for item in glm_strip)
    assert not any(item["anomaly"] for item in glm_strip)
    assert all(item["reason"] is None for item in glm_strip)
