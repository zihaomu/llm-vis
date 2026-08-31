"""LLM-Vis command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from llm_vis.adapters import AdapterConfigError
from llm_vis.analysis import AnalysisBundle, capture_representatives, inspect_model
from llm_vis.capture import RepresentativeKind
from llm_vis.diff import diff_scenario_metrics
from llm_vis.ir import ModelMap, Scenario, export_schemas
from llm_vis.performance import HardwareProfile
from llm_vis.report import ArtifactWriteError, write_analysis
from llm_vis.resolver import ResolutionError

_DEFAULT_VIEW_PRESET_BACKEND = "llm-vis-default-preset"


def _scenario(path: Optional[str]) -> Sequence[Scenario]:
    if path is None:
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload if isinstance(payload, list) else [payload]
    return tuple(Scenario.model_validate(value) for value in values)


def _default_view_scenarios() -> tuple[Scenario, Scenario]:
    """Return useful formula-only workloads for the zero-argument ``view`` path.

    These are deliberately ordinary Scenario records, so every assumption is
    persisted in ``scenarios.json`` and visible in the report.  They do not imply
    a HardwareProfile and therefore cannot enable theoretical pressure.
    """

    common = {
        "batch": 1,
        "activation_dtype": "bfloat16",
        "weight_format": "bfloat16",
        "kv_dtype": "bfloat16",
        "backend": _DEFAULT_VIEW_PRESET_BACKEND,
        "hardware": "unprofiled",
    }
    return (
        Scenario(
            phase="prefill",
            new_tokens=512,
            past_tokens=0,
            **common,
        ),
        Scenario(
            phase="decode",
            new_tokens=1,
            past_tokens=512,
            **common,
        ),
    )


def _hardware_profile(path: Optional[str]) -> Optional[HardwareProfile]:
    if path is None:
        return None
    return HardwareProfile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _model_map(path: Path) -> ModelMap:
    target = path / "model-map.json" if path.is_dir() else path
    return ModelMap.model_validate_json(target.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, *, force: bool) -> None:
    if path.exists() and not force:
        raise ValueError(f"output already exists; pass --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _default_output_dir(bundle: AnalysisBundle) -> Path:
    """Return a unique temporary root with a recognizable artifact directory."""

    staging_root = Path(tempfile.mkdtemp(prefix="llm-vis-"))
    identifier = bundle.resolved.identifier
    if identifier.startswith(("config:", "inline", "local")):
        identifier = str(bundle.resolved.config.get("model_type") or "model")
    slug = re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")[:64] or "model"
    return staging_root / f"{slug}-{bundle.resolved.sha256[:8]}"


def _open_report(path: Path) -> bool:
    """Open one generated local report without turning headless failure into analysis failure."""

    try:
        return bool(webbrowser.open(path.resolve().as_uri()))
    except (OSError, webbrowser.Error):
        return False


def _add_analysis_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "model_positional",
        nargs="?",
        help="HF ID/URL, local config, inline JSON, or '-' to read JSON from stdin",
    )
    command_parser.add_argument(
        "--model",
        dest="model_option",
        help="HF ID/URL, local config, inline JSON, or '-' for stdin",
    )
    command_parser.add_argument("--revision")
    command_parser.add_argument("--scenario", help="Scenario JSON object or list")
    command_parser.add_argument(
        "--hardware-profile",
        help="Explicit HardwareProfile JSON for theoretical roofline lower bounds",
    )
    command_parser.add_argument(
        "--output",
        type=Path,
        help="Artifact directory; omitted uses a unique temporary user directory",
    )
    command_parser.add_argument("--local-files-only", action="store_true")
    command_parser.add_argument("--force", action="store_true")
    open_group = command_parser.add_mutually_exclusive_group()
    open_group.add_argument(
        "--open",
        dest="open_report",
        action="store_true",
        help="Open the generated offline HTML report",
    )
    open_group.add_argument(
        "--no-open",
        dest="open_report",
        action="store_false",
        help="Do not open the generated offline HTML report",
    )
    command_parser.set_defaults(open_report=None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-vis", description="Config-first LLM visualizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Generate structure without weights")
    _add_analysis_arguments(inspect_parser)

    view_parser = subparsers.add_parser(
        "view",
        help="Resolve one config, generate its offline graph, and open it by default",
    )
    _add_analysis_arguments(view_parser)

    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture bounded Tiny representative blocks without model weights",
    )
    _add_analysis_arguments(capture_parser)
    capture_parser.add_argument(
        "--kind",
        action="append",
        choices=[kind.value for kind in RepresentativeKind],
        help="Representative kind to capture; repeat to select multiple kinds",
    )

    schema_parser = subparsers.add_parser("schema", help="Export JSON Schemas")
    schema_parser.add_argument("--output", type=Path, default=Path("schemas"))

    validate_parser = subparsers.add_parser("validate", help="Validate a model-map artifact")
    validate_parser.add_argument("path", type=Path)

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two Scenarios in the same Model Map artifact",
    )
    diff_parser.add_argument("artifact", type=Path, help="Artifact directory or model-map.json")
    diff_parser.add_argument("--left-scenario", required=True)
    diff_parser.add_argument("--right-scenario", required=True)
    diff_parser.add_argument("--output", required=True, type=Path)
    diff_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "schema":
            paths = export_schemas(args.output)
            for path in paths.values():
                print(path)
            return 0
        if args.command == "validate":
            ModelMap.model_validate_json(args.path.read_text(encoding="utf-8"))
            print(f"valid: {args.path}")
            return 0
        if args.command == "diff":
            model_map = _model_map(args.artifact)
            result = diff_scenario_metrics(
                model_map,
                args.left_scenario,
                args.right_scenario,
            )
            _write_json(args.output, result.model_dump(mode="json"), force=args.force)
            print(f"diff: {result.id}")
            print(f"output: {args.output.resolve()}")
            return 0
        if args.command in {"view", "inspect", "capture"}:
            if args.model_option is not None and args.model_positional is not None:
                raise ValueError(
                    f"{args.command} accepts one model input; use either MODEL or --model"
                )
            source = args.model_option or args.model_positional
            if not source:
                raise ValueError(f"{args.command} requires MODEL or --model")
            if source == "-":
                source = sys.stdin.read()
                if not source.strip():
                    raise ValueError("stdin did not contain a JSON config")
            scenarios = _scenario(args.scenario)
            if args.command == "view" and args.scenario is None:
                scenarios = _default_view_scenarios()
            bundle = inspect_model(
                source,
                revision=args.revision,
                scenarios=scenarios,
                local_files_only=args.local_files_only,
                hardware_profile=_hardware_profile(args.hardware_profile),
            )
            if args.command == "capture":
                kinds = (
                    tuple(RepresentativeKind(kind) for kind in args.kind)
                    if args.kind is not None
                    else None
                )
                bundle = capture_representatives(bundle, kinds=kinds)
            output = args.output or _default_output_dir(bundle)
            written = write_analysis(bundle, output, force=args.force)
            print(f"analysis: {bundle.analysis_id}")
            print(f"output: {output.resolve()}")
            report_path = written["reports/report.html"].resolve()
            print(f"report: {report_path}")
            default_open = args.command == "view" or args.output is None
            should_open = args.open_report if args.open_report is not None else default_open
            if should_open:
                if _open_report(report_path):
                    print(f"opened: {report_path.as_uri()}")
                else:
                    print(
                        "warning: report was generated but no browser accepted the open request",
                        file=sys.stderr,
                    )
            return 0
    except (
        AdapterConfigError,
        ArtifactWriteError,
        json.JSONDecodeError,
        OSError,
        ResolutionError,
        ValidationError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


__all__ = ["main"]
