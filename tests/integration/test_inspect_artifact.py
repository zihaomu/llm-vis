from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_vis.analysis import inspect_model
from llm_vis.cli import main
from llm_vis.graph_view import GraphViewDocument
from llm_vis.ir import ModelMap, Scenario
from llm_vis.report import ArtifactWriteError, write_analysis

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def _scenario() -> Scenario:
    return Scenario(
        phase="decode",
        batch=2,
        new_tokens=1,
        past_tokens=127,
        activation_dtype="bfloat16",
        weight_format="bfloat16",
        kv_dtype="bfloat16",
        backend="formula-only",
        hardware="unknown",
    )


def test_inspect_writes_reproducible_offline_artifact(tmp_path: Path) -> None:
    bundle = inspect_model(str(FIXTURE_DIR / "tiny_dense.json"), scenarios=[_scenario()])
    written = write_analysis(bundle, tmp_path / "artifact")

    assert set(written) == {
        "manifest.json",
        "model-map.json",
        "graph-view.json",
        "scenarios.json",
        "metrics.json",
        "diagnostics.json",
        "layer-strip.json",
        "captures.json",
        "hardware-profile.json",
        "hotspots.json",
        "roofline.json",
        "workload-diffs.json",
        "model-explorer.json",
        "reports/report.md",
        "reports/report.html",
    }
    model_map = ModelMap.model_validate_json(written["model-map.json"].read_text())
    graph_view = GraphViewDocument.model_validate_json(written["graph-view.json"].read_text())
    manifest = json.loads(written["manifest.json"].read_text())
    layer_strip = json.loads(written["layer-strip.json"].read_text())
    assert manifest["safety"] == {
        "weights_loaded": False,
        "full_model_constructed": False,
        "full_forward_executed": False,
        "remote_code_executed": False,
        "full_meta_tree_enabled": False,
    }
    assert manifest["capability"]["level"] == "C1"
    assert manifest["capability"]["interactive_semantic_dag"] is True
    assert manifest["capability"]["semantic_zoom_levels"] == ["L0", "L1"]
    assert manifest["counts"]["graph_views"] == len(graph_view.views)
    assert manifest["model_map"]["artifact_id"] == graph_view.source_model_map_id
    assert manifest["counts"]["graph_nodes"] == sum(len(view.nodes) for view in graph_view.views)
    assert graph_view.views[0].key == "l0"
    assert all(
        set(("expected_pattern", "anomaly", "reason")).issubset(entry) for entry in layer_strip
    )
    assert not any(entry["anomaly"] for entry in layer_strip)
    assert len(model_map.scenarios) == 1
    assert "no weights were loaded" in written["reports/report.md"].read_text()
    html = written["reports/report.html"].read_text()
    assert "application/json" in html
    html_without_svg_namespace = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in html_without_svg_namespace
    assert "https://" not in html_without_svg_namespace


def test_writer_requires_explicit_force_but_preserves_unknown_files(tmp_path: Path) -> None:
    bundle = inspect_model(str(FIXTURE_DIR / "tiny_dense.json"))
    output = tmp_path / "artifact"
    write_analysis(bundle, output)
    unknown = output / "user-note.txt"
    unknown.write_text("keep", encoding="utf-8")

    with pytest.raises(ArtifactWriteError):
        write_analysis(bundle, output)
    write_analysis(bundle, output, force=True)
    assert unknown.read_text(encoding="utf-8") == "keep"


def test_identical_inputs_produce_byte_identical_artifacts(tmp_path: Path) -> None:
    first = inspect_model(str(FIXTURE_DIR / "tiny_dense.json"), scenarios=[_scenario()])
    second = inspect_model(str(FIXTURE_DIR / "tiny_dense.json"), scenarios=[_scenario()])
    first_paths = write_analysis(first, tmp_path / "first")
    second_paths = write_analysis(second, tmp_path / "second")

    assert first.analysis_id == second.analysis_id
    assert set(first_paths) == set(second_paths)
    for relative in first_paths:
        assert first_paths[relative].read_bytes() == second_paths[relative].read_bytes()


def test_remote_code_declaration_is_visible_but_never_executed(tmp_path: Path) -> None:
    config = json.loads((FIXTURE_DIR / "tiny_dense.json").read_text())
    config["auto_map"] = {"AutoModel": "modeling_untrusted.UntrustedModel"}
    model_dir = tmp_path / "untrusted"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "modeling_untrusted.py").write_text(
        "raise RuntimeError('this file must never execute')\n", encoding="utf-8"
    )

    bundle = inspect_model(str(model_dir))
    codes = {item.code for item in bundle.model_map.diagnostics}
    assert "REMOTE_CODE_EXECUTION_DISABLED" in codes
    assert "RUNTIME_TRACE_MISSING" in codes
    assert bundle.manifest()["safety"]["remote_code_executed"] is False


def test_cli_inspect_and_validate(tmp_path: Path) -> None:
    output = tmp_path / "cli-artifact"
    assert (
        main(
            [
                "inspect",
                "--model",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert main(["validate", str(output / "model-map.json")]) == 0
