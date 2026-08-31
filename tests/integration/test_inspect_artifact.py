from __future__ import annotations

import io
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
    assert manifest["source"]["reads_config_json_only"] is True
    assert manifest["source"]["config_sha256"] == bundle.resolved.sha256
    assert manifest["source"]["resolved_revision"] == bundle.resolved.revision
    assert manifest["source"]["input_kind"] == "local_file"
    assert manifest["capability"]["level"] == "C1"
    assert manifest["capability"]["interactive_semantic_dag"] is True
    assert manifest["capability"]["semantic_zoom_levels"] == ["L0", "L1"]
    assert manifest["capability"]["recursive_operator_decomposition"] is False
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

    qwen = inspect_model(str(FIXTURE_DIR / "qwen3_8_27b.json"))
    assert qwen.manifest()["capability"]["recursive_operator_decomposition"] is True


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


def test_cli_uses_temporary_output_and_opens_report_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "default-artifact"
    opened: list[str] = []
    monkeypatch.setattr("llm_vis.cli._default_output_dir", lambda bundle: output)
    monkeypatch.setattr(
        "llm_vis.cli.webbrowser.open",
        lambda uri: opened.append(uri) or True,
    )

    assert main(["inspect", str(FIXTURE_DIR / "tiny_dense.json")]) == 0

    report = (output / "reports" / "report.html").resolve()
    assert report.is_file()
    assert opened == [report.as_uri()]


def test_cli_explicit_output_only_opens_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        "llm_vis.cli.webbrowser.open",
        lambda uri: opened.append(uri) or True,
    )

    first = tmp_path / "explicit"
    assert (
        main(
            [
                "inspect",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--output",
                str(first),
            ]
        )
        == 0
    )
    assert opened == []

    second = tmp_path / "explicit-open"
    assert (
        main(
            [
                "inspect",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--output",
                str(second),
                "--open",
            ]
        )
        == 0
    )
    assert opened == [(second / "reports" / "report.html").resolve().as_uri()]


def test_cli_view_reads_json_from_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "stdin-view"
    config_text = (FIXTURE_DIR / "tiny_dense.json").read_text(encoding="utf-8")
    monkeypatch.setattr("llm_vis.cli.sys.stdin", io.StringIO(config_text))

    assert (
        main(
            [
                "view",
                "-",
                "--output",
                str(output),
                "--no-open",
            ]
        )
        == 0
    )
    assert (output / "reports" / "report.html").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["uri"].startswith("inline://sha256/")
    assert manifest["source"]["input_kind"] == "inline_json"


def test_cli_view_adds_labeled_prefill_and_decode_presets_without_pressure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "default-view-presets"

    assert (
        main(
            [
                "view",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--output",
                str(output),
                "--no-open",
            ]
        )
        == 0
    )

    scenarios = json.loads((output / "scenarios.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    html = (output / "reports" / "report.html").read_text(encoding="utf-8")

    assert [(item["phase"], item["new_tokens"], item["past_tokens"]) for item in scenarios] == [
        ("prefill", 512, 0),
        ("decode", 1, 512),
    ]
    assert {item["backend"] for item in scenarios} == {"llm-vis-default-preset"}
    assert {item["hardware"] for item in scenarios} == {"unprofiled"}
    assert manifest["cost"]["scenario_count"] == 2
    assert manifest["hardware_profile"] is None
    assert manifest["roofline"]["enabled"] is False
    cost_scenario_ids = {
        item["scenario_id"] for item in metrics if item["name"] in {"flops", "logical_bytes"}
    }
    assert cost_scenario_ids == {
        item["id"] for item in scenarios
    }
    assert "Default preset · " in html
    assert (
        '<option value="pressure" disabled>Pressure (requires HardwareProfile)</option>' in html
    )
    assert '<option value="compute" selected>Compute</option>' in html
    assert "Pressure (requires HardwareProfile)" in html
    assert "pressure.disabled=true" in html
    assert "select.value='compute'" in html
    assert "Pressure requires an explicit HardwareProfile." in html
    assert manifest["safety"] == {
        "weights_loaded": False,
        "full_model_constructed": False,
        "full_forward_executed": False,
        "remote_code_executed": False,
        "full_meta_tree_enabled": False,
    }


def test_cli_view_opens_by_default_and_keeps_artifact_when_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "view-open-failed"
    opened: list[str] = []
    monkeypatch.setattr(
        "llm_vis.cli.webbrowser.open",
        lambda uri: opened.append(uri) or False,
    )

    assert (
        main(
            [
                "view",
                str(FIXTURE_DIR / "tiny_dense.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = (output / "reports" / "report.html").resolve()
    assert report.is_file()
    assert opened == [report.as_uri()]
    assert "report was generated" in capsys.readouterr().err


def test_cli_rejects_positional_and_option_model_inputs_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = str(FIXTURE_DIR / "tiny_dense.json")

    assert main(["view", config, "--model", config, "--output", str(tmp_path)]) == 2
    assert "accepts one model input" in capsys.readouterr().err
