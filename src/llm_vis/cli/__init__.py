"""LLM-Vis command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from llm_vis.adapters import AdapterConfigError
from llm_vis.analysis import capture_representatives, inspect_model
from llm_vis.capture import RepresentativeKind
from llm_vis.diff import diff_scenario_metrics
from llm_vis.ir import ModelMap, Scenario, export_schemas
from llm_vis.performance import HardwareProfile
from llm_vis.report import ArtifactWriteError, write_analysis
from llm_vis.resolver import ResolutionError


def _scenario(path: Optional[str]) -> Sequence[Scenario]:
    if path is None:
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload if isinstance(payload, list) else [payload]
    return tuple(Scenario.model_validate(value) for value in values)


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


def _add_analysis_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "model_positional", nargs="?", help="Local config/model directory or HF ID"
    )
    command_parser.add_argument(
        "--model", dest="model_option", help="Local config/model directory or HF ID"
    )
    command_parser.add_argument("--revision")
    command_parser.add_argument("--scenario", help="Scenario JSON object or list")
    command_parser.add_argument(
        "--hardware-profile",
        help="Explicit HardwareProfile JSON for theoretical roofline lower bounds",
    )
    command_parser.add_argument("--output", required=True, type=Path)
    command_parser.add_argument("--local-files-only", action="store_true")
    command_parser.add_argument("--force", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-vis", description="Config-first LLM visualizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Generate structure without weights")
    _add_analysis_arguments(inspect_parser)

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
        if args.command in {"inspect", "capture"}:
            source = args.model_option or args.model_positional
            if not source:
                raise ValueError(f"{args.command} requires MODEL or --model")
            bundle = inspect_model(
                source,
                revision=args.revision,
                scenarios=_scenario(args.scenario),
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
            write_analysis(bundle, args.output, force=args.force)
            print(f"analysis: {bundle.analysis_id}")
            print(f"output: {args.output.resolve()}")
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
