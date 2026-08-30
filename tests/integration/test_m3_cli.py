from __future__ import annotations

import json
from pathlib import Path

from llm_vis.cli import main

PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def test_cli_inspect_with_explicit_hardware_and_same_artifact_diff(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "phase": "prefill",
                    "batch": 1,
                    "new_tokens": 8,
                    "past_tokens": 0,
                    "activation_dtype": "bfloat16",
                    "weight_format": "bfloat16",
                    "kv_dtype": "bfloat16",
                },
                {
                    "phase": "decode",
                    "batch": 1,
                    "new_tokens": 1,
                    "past_tokens": 8,
                    "activation_dtype": "bfloat16",
                    "weight_format": "int4",
                    "kv_dtype": "bfloat16",
                },
            ]
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact"
    assert (
        main(
            [
                "inspect",
                "--model",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--scenario",
                str(scenario_path),
                "--hardware-profile",
                str(PROJECT_ROOT / "examples" / "hardware" / "synthetic-bf16.json"),
                "--output",
                str(artifact),
            ]
        )
        == 0
    )

    scenarios = json.loads((artifact / "scenarios.json").read_text(encoding="utf-8"))
    roofline = json.loads((artifact / "roofline.json").read_text(encoding="utf-8"))
    assert len(roofline) == 2
    assert all(item["is_latency_estimate"] is False for item in roofline)

    output = tmp_path / "diff.json"
    assert (
        main(
            [
                "diff",
                str(artifact),
                "--left-scenario",
                scenarios[0]["id"],
                "--right-scenario",
                scenarios[1]["id"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["left_scenario_id"] == scenarios[0]["id"]
    assert result["right_scenario_id"] == scenarios[1]["id"]
    assert {entry["status"] for entry in result["entries"]} <= {
        "added",
        "removed",
        "changed",
        "unknown",
    }


def test_cli_diff_rejects_missing_scenario_and_protects_existing_output(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    assert (
        main(
            [
                "inspect",
                "--model",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--output",
                str(artifact),
            ]
        )
        == 0
    )
    output = tmp_path / "diff.json"
    output.write_text("keep", encoding="utf-8")

    assert (
        main(
            [
                "diff",
                str(artifact),
                "--left-scenario",
                "missing-left",
                "--right-scenario",
                "missing-right",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_text(encoding="utf-8") == "keep"
