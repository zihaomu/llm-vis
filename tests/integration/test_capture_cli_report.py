from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from llm_vis.analysis import capture_representatives, inspect_model
from llm_vis.capture import RepresentativeKind, capture_tiny_representative
from llm_vis.cli import main
from llm_vis.report import write_analysis

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is an optional capture dependency",
)


@requires_torch
def test_cli_capture_filters_kind_and_writes_dynamic_reports(tmp_path: Path) -> None:
    output = tmp_path / "capture-artifact"
    result = main(
        [
            "capture",
            "--model",
            str(FIXTURE_DIR / "tiny_dense.json"),
            "--kind",
            "full_attention",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    captures = json.loads((output / "captures.json").read_text(encoding="utf-8"))
    explorer = json.loads((output / "model-explorer.json").read_text(encoding="utf-8"))
    markdown = (output / "reports" / "report.md").read_text(encoding="utf-8")
    html = (output / "reports" / "report.html").read_text(encoding="utf-8")

    assert manifest["capability"]["level"] == "C2"
    assert manifest["capability"]["representative_block_captured"] is True
    assert manifest["safety"]["weights_loaded"] is False
    assert [capture["kind"] for capture in captures] == ["full_attention"]
    assert captures[0]["coverage"] == 1.0
    assert captures[0]["logical_op_count"] > 0
    assert "## Capture summary" in markdown
    assert "| full_attention | captured | 100.0% |" in markdown
    assert "| Capability | C2 |" in markdown
    assert "Capture summary" in html
    assert "bounded Tiny capture" in html
    assert "allCaptureSourceIds" in html
    assert "filter(id=>!allCaptureSourceIds.has(id))" in html
    assert "item.attrs?.capture_id" in html
    assert "captureIds.has(item.capture_id)" in html
    assert [graph["id"] for graph in explorer["graphs"]] == [
        "architecture-definitions",
        "semantic-structure",
        "logical-ops",
    ]
    logical = explorer["graphs"][2]
    assert logical["nodes"]
    assert len(logical["nodes"]) <= 200
    assert any(node["incomingEdges"] for node in logical["nodes"])


@requires_torch
def test_opaque_capture_is_visible_in_offline_reports(tmp_path: Path) -> None:
    bundle = inspect_model(str(FIXTURE_DIR / "tiny_dense.json"))

    def opaque(kind: RepresentativeKind):
        def fail(module: object, args: tuple[object, ...]) -> object:
            raise RuntimeError(f"fault injection for {kind.value}")

        return capture_tiny_representative(kind, export_fn=fail)

    captured = capture_representatives(
        bundle,
        kinds=[RepresentativeKind.FULL_ATTENTION],
        capture_fn=opaque,
    )
    output = tmp_path / "opaque-artifact"
    write_analysis(captured, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    markdown = (output / "reports" / "report.md").read_text(encoding="utf-8")
    html = (output / "reports" / "report.html").read_text(encoding="utf-8")
    explorer = json.loads((output / "model-explorer.json").read_text(encoding="utf-8"))

    assert manifest["capability"]["level"] == "C1"
    assert manifest["capability"]["representative_block_captured"] is False
    assert captured.captures[0]["status"] == "opaque"
    assert "| full_attention | opaque | 0.0% | 0 | 0 |" in markdown
    assert "| Capability | C1 |" in markdown
    assert "opaque" in html
    assert "0.0% coverage" in html
    assert [graph["id"] for graph in explorer["graphs"]] == [
        "architecture-definitions",
        "semantic-structure",
    ]
