"""Atomic artifact directory writer."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from llm_vis.analysis.service import AnalysisBundle
from llm_vis.report.html import render_html
from llm_vis.report.markdown import render_markdown
from llm_vis.report.model_explorer import model_explorer_graphs
from llm_vis.report.summaries import workload_diff_summaries


class ArtifactWriteError(RuntimeError):
    """Raised when an output directory is unsafe to overwrite."""


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_analysis(
    bundle: AnalysisBundle, output_dir: Path, *, force: bool = False
) -> Dict[str, Path]:
    """Write one analysis without deleting unrelated user files.

    If known artifact files already exist, callers must pass ``force=True``. Even in
    force mode only the known files are replaced atomically; the directory is never
    recursively cleared.
    """

    output_dir = output_dir.resolve()
    graph_view = bundle.graph_view
    known = {
        "manifest.json": bundle.manifest(),
        "model-map.json": bundle.model_map.model_dump(mode="json"),
        "graph-view.json": graph_view.model_dump(mode="json"),
        "scenarios.json": [item.model_dump(mode="json") for item in bundle.model_map.scenarios],
        "metrics.json": [item.model_dump(mode="json") for item in bundle.model_map.metrics],
        "diagnostics.json": [item.model_dump(mode="json") for item in bundle.model_map.diagnostics],
        "layer-strip.json": list(bundle.layer_strip),
        "captures.json": [dict(item) for item in bundle.captures],
        "hardware-profile.json": (
            bundle.hardware_profile.model_dump(mode="json")
            if bundle.hardware_profile is not None
            else None
        ),
        "hotspots.json": [dict(item) for item in bundle.hotspot_summaries],
        "roofline.json": [dict(item) for item in bundle.roofline_summaries],
        "workload-diffs.json": list(workload_diff_summaries(bundle.model_map)),
        "model-explorer.json": model_explorer_graphs(bundle.model_map),
    }
    target_paths = [output_dir / relative for relative in known]
    target_paths.extend(
        [
            output_dir / "reports" / "report.md",
            output_dir / "reports" / "report.html",
        ]
    )
    existing = [path for path in target_paths if path.exists()]
    if existing and not force:
        raise ArtifactWriteError(
            "Known artifact files already exist; pass force=True to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for relative, value in known.items():
        path = output_dir / relative
        _atomic_text(path, _json_text(value))
        written[relative] = path
    markdown_path = output_dir / "reports" / "report.md"
    html_path = output_dir / "reports" / "report.html"
    _atomic_text(markdown_path, render_markdown(bundle))
    _atomic_text(html_path, render_html(bundle, graph_view=graph_view))
    written["reports/report.md"] = markdown_path
    written["reports/report.html"] = html_path
    return written
