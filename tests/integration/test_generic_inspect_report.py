from __future__ import annotations

import json
from pathlib import Path

from llm_vis.analysis import inspect_model
from llm_vis.cli import main
from llm_vis.ir import Scenario
from llm_vis.report import write_analysis


def test_inspect_unknown_config_with_scenario_writes_renderable_bundle(tmp_path: Path) -> None:
    config = {
        "model_type": "future_wrapper",
        "text_config": {
            "model_type": "future_text",
            "num_hidden_layers": 3,
            "hidden_size": 96,
            "vocab_size": 1024,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    scenario = Scenario(
        phase="decode",
        batch=1,
        new_tokens=1,
        past_tokens=8,
        activation_dtype="bfloat16",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
        backend="formula-only",
        hardware="unknown",
    )
    bundle = inspect_model(str(config_path), scenarios=[scenario])

    assert bundle.adapter_result.adapter_name == "generic-config"
    assert bundle.cost_analysis is None
    assert bundle.model_map.scenarios == [scenario]
    capability = bundle.manifest()["capability"]
    assert capability["level"] == "C0"
    assert capability["family_adapter_available"] is False
    assert capability["architecture_internals"] == "opaque"
    assert capability["semantic_zoom_levels"] == ["L0"]
    assert "THEORETICAL_COST_UNAVAILABLE_FOR_GENERIC_CONFIG" in {
        item.code for item in bundle.model_map.diagnostics
    }

    written = write_analysis(bundle, tmp_path / "artifact")
    assert written["graph-view.json"].is_file()
    assert written["reports/report.html"].is_file()
    assert "generic-config" in written["reports/report.html"].read_text(encoding="utf-8")


def test_view_unknown_config_keeps_default_presets_but_does_not_invent_costs(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "future_wrapper",
                "text_config": {
                    "model_type": "future_text",
                    "num_hidden_layers": 3,
                    "hidden_size": 96,
                    "vocab_size": 1024,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "generic-view"

    assert main(["view", str(config_path), "--output", str(output), "--no-open"]) == 0

    scenarios = json.loads((output / "scenarios.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert [item["phase"] for item in scenarios] == ["prefill", "decode"]
    assert not any(item["name"] in {"flops", "logical_bytes"} for item in metrics)
    assert all(item["name"].startswith("runtime.") for item in metrics)
    assert all(
        item["value"] is None and item["origin"] == "unknown" and item["coverage"] == 0
        for item in metrics
    )
    assert manifest["adapter"]["name"] == "generic-config"
    assert manifest["capability"]["cost_analyzed"] is False
    assert manifest["cost"]["scenario_count"] == 0
    assert manifest["hardware_profile"] is None
    assert manifest["roofline"]["enabled"] is False
