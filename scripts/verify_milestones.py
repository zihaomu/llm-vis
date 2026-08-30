"""Offline, machine-runnable static exit checks for LLM-Vis M0-M3.5.

This verifier only reads checked-in files.  It never resolves a remote model,
imports torch, loads weights, constructs a model, or invokes a forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = DEFAULT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from llm_vis.graph_view import GraphEdgeKind, GraphViewDocument  # noqa: E402
from llm_vis.ir import ModelMap, Scenario, deterministic_id  # noqa: E402

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
except ImportError:  # pragma: no cover - exercised only outside the dev environment
    Draft202012Validator = None  # type: ignore[assignment,misc]
    SchemaError = ValueError  # type: ignore[assignment,misc]


MILESTONES: Tuple[str, ...] = ("M0", "M1", "M2", "M3", "M3.5")
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
STRUCTURE_GOLDENS = ("tiny_dense", "qwen3_8_27b", "glm_5_3_bf16")
COST_GOLDENS = ("tiny_prefill_bf16.json", "qwen_decode_int4.json")
CAPTURE_GOLDEN = "qwen_representatives.json"
GRAPH_VIEW_GOLDENS = ("qwen3_8_27b", "glm_5_3_bf16")
REQUIRED_COST_METRICS = {
    "parameters.resident",
    "parameters.active",
    "weights.storage_bytes",
    "flops",
    "logical_bytes",
    "kv_cache.bytes",
    "state.bytes",
    "arithmetic_intensity",
}


class VerificationError(ValueError):
    """A checked-in milestone asset is absent or invalid."""


@dataclass(frozen=True)
class AssetCheck:
    milestone: str
    name: str
    path_label: str
    validator: Callable[[Path], str]


@dataclass(frozen=True)
class CheckResult:
    milestone: str
    name: str
    path: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    requested: str
    results: Tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def passed(self) -> int:
        return sum(result.ok for result in self.results)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed


def _require_file(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise VerificationError(f"missing file: {relative_path}")
    return path


def _json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise VerificationError("top-level JSON value must be an object")
    return payload


def _schema_validator(relative_path: str) -> Callable[[Path], str]:
    def validate(root: Path) -> str:
        path = _require_file(root, relative_path)
        payload = _json_object(path)
        if payload.get("$schema") != SCHEMA_DIALECT:
            raise VerificationError(
                f"$schema must be {SCHEMA_DIALECT!r}, got {payload.get('$schema')!r}"
            )
        if Draft202012Validator is None:
            raise VerificationError("jsonschema is required to validate Draft 2020-12 schemas")
        try:
            Draft202012Validator.check_schema(payload)
        except SchemaError as error:
            raise VerificationError(f"invalid Draft 2020-12 schema: {error.message}") from error
        return f"Draft 2020-12 schema valid ({len(payload.get('$defs', {}))} definitions)"

    return validate


def _structure_validator(relative_path: str) -> Callable[[Path], str]:
    def validate(root: Path) -> str:
        path = _require_file(root, relative_path)
        payload = _json_object(path)
        try:
            model_map = ModelMap.model_validate(payload)
        except Exception as error:
            raise VerificationError(f"invalid Model Map IR: {error}") from error
        if model_map.schema_version != "0.1":
            raise VerificationError(f"unexpected schema version: {model_map.schema_version}")
        return (
            f"Model Map valid ({len(model_map.definitions)} definitions, "
            f"{len(model_map.instances)} instances)"
        )

    return validate


def _validate_metric_map(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise VerificationError(f"expected.{name} must be an object")
    missing = REQUIRED_COST_METRICS - set(value)
    if missing:
        raise VerificationError(f"expected.{name} is missing metrics: {sorted(missing)}")
    for metric_name in REQUIRED_COST_METRICS:
        metric_value = value[metric_name]
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise VerificationError(f"{name}.{metric_name} must be numeric")
        if metric_value < 0:
            raise VerificationError(f"{name}.{metric_name} must be non-negative")


def _cost_validator(relative_path: str) -> Callable[[Path], str]:
    def validate(root: Path) -> str:
        path = _require_file(root, relative_path)
        payload = _json_object(path)
        if not isinstance(payload.get("fixture"), str) or not payload["fixture"]:
            raise VerificationError("fixture must be a non-empty string")
        try:
            scenario = Scenario.model_validate(payload.get("scenario"))
        except Exception as error:
            raise VerificationError(f"invalid Scenario: {error}") from error
        expected = payload.get("expected")
        if not isinstance(expected, dict):
            raise VerificationError("expected must be an object")
        if expected.get("scenario_id") != scenario.id:
            raise VerificationError("expected.scenario_id does not match the canonical Scenario id")
        layer_count = expected.get("layer_count")
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
            raise VerificationError("expected.layer_count must be a positive integer")
        _validate_metric_map(expected.get("first_layer_metrics"), "first_layer_metrics")
        _validate_metric_map(expected.get("aggregate_metrics"), "aggregate_metrics")
        return f"cost golden valid ({payload['fixture']}, scenario={scenario.id})"

    return validate


def _capture_validator(relative_path: str) -> Callable[[Path], str]:
    def validate(root: Path) -> str:
        path = _require_file(root, relative_path)
        payload = _json_object(path)
        if payload.get("fixture") != "qwen3_8_27b":
            raise VerificationError("capture golden must target qwen3_8_27b")
        revision = payload.get("revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise VerificationError("capture golden revision must be a full hexadecimal SHA")

        policy = payload.get("policy")
        required_policy = {
            "architecture_equivalence": "semantic-kind-only",
            "capture_backend": "torch.export.strict",
            "capture_source": "project-owned-tiny-fixture",
            "full_forward_executed": False,
            "tensor_mode": "meta-parameters-and-inputs/faketensor-trace",
            "valid_for_full_model_performance": False,
            "weights_loaded": False,
        }
        if policy != required_policy:
            raise VerificationError("capture golden policy does not match the bounded M2 contract")
        if payload.get("component_patterns") != [
            "ffn",
            "full_attention",
            "linear_attention",
            "norm",
            "recurrent_state",
            "residual",
        ]:
            raise VerificationError("capture golden semantic component catalog is incomplete")
        if payload.get("coverage_contract") != {
            "model_scope_diagnostic": "CAPTURE_SCOPE_REPRESENTATIVE_ONLY",
            "scope": "decoder_attention_semantic_nodes",
            "scope_evidence": "coverage_scope=decoder_attention_semantic_nodes",
            "unselected_region_diagnostic": "CAPTURE_REGION_NOT_SELECTED",
            "unselected_region_evidence": "coverage=0",
        }:
            raise VerificationError(
                "capture golden decoder-attention per-region coverage contract is incomplete"
            )

        representatives = payload.get("representatives")
        if not isinstance(representatives, list):
            raise VerificationError("capture golden representatives must be a list")
        expected_kinds = {"dense", "full_attention", "linear_state"}
        actual_kinds = {item.get("kind") for item in representatives if isinstance(item, dict)}
        if len(representatives) != 3 or actual_kinds != expected_kinds:
            raise VerificationError("capture golden must contain the three M2 representative kinds")
        for item in representatives:
            if not isinstance(item, dict):
                raise VerificationError("capture representative must be an object")
            if not isinstance(item.get("module_path"), str) or not item["module_path"]:
                raise VerificationError("capture representative requires module_path")
            if not isinstance(item.get("semantic_kind"), str) or not item["semantic_kind"]:
                raise VerificationError("capture representative requires semantic_kind")
            layer_index = item.get("representative_layer_index")
            if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
                raise VerificationError("capture representative layer index must be non-negative")
            for count_name in ("minimum_logical_ops", "minimum_tensors"):
                count = item.get(count_name)
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise VerificationError(f"capture representative {count_name} must be positive")

        opaque = payload.get("opaque_fallback")
        if opaque != {
            "diagnostic_code": "CAPTURE_EXPORT_FAILED",
            "full_model_fallback": False,
            "status": "opaque",
        }:
            raise VerificationError("capture golden must freeze opaque/no-full-model fallback")
        return (
            "bounded Qwen capture contract valid "
            "(3 kinds, 6 semantic patterns, decoder-attention per-region coverage, "
            "opaque fallback)"
        )

    return validate


def _graph_view_validator(relative_path: str, fixture: str) -> Callable[[Path], str]:
    def validate(root: Path) -> str:
        path = _require_file(root, relative_path)
        payload = _json_object(path)
        if Draft202012Validator is None:
            raise VerificationError("jsonschema is required to validate GraphView goldens")
        graph_schema = _json_object(_require_file(root, "schemas/graph-view.schema.json"))
        try:
            Draft202012Validator(graph_schema).validate(payload)
        except Exception as error:
            raise VerificationError(
                f"GraphView golden does not match graph-view.schema.json: {error}"
            ) from error
        try:
            document = GraphViewDocument.model_validate(payload)
        except Exception as error:
            raise VerificationError(f"invalid GraphView artifact: {error}") from error
        source_model_map = _json_object(
            _require_file(root, f"tests/golden/structure/{fixture}/model-map.json")
        )
        try:
            model_map = ModelMap.model_validate(source_model_map)
        except Exception as error:
            raise VerificationError(f"invalid source Model Map: {error}") from error
        expected_model_map_id = deterministic_id("mmap", source_model_map)
        if document.source_model_map_id != expected_model_map_id:
            raise VerificationError(
                "GraphView source_model_map_id does not identify its structure golden"
            )
        subject_ids = {
            model_map.model.id,
            *(item.id for item in model_map.definitions),
            *(item.id for item in model_map.instances),
            *(item.id for item in model_map.semantic_nodes),
            *(item.id for item in model_map.logical_ops),
        }
        tensor_ids = {item.id for item in model_map.tensors}
        for view in document.views:
            for node in view.nodes:
                if set(node.subject_ids) - subject_ids:
                    raise VerificationError(
                        f"GraphView node {node.key!r} references a missing Model Map subject"
                    )
            for port in view.ports:
                if port.tensor_spec_id is not None and port.tensor_spec_id not in tensor_ids:
                    raise VerificationError(
                        f"GraphView port {port.key!r} references a missing TensorSpec"
                    )
        if any(
            (
                document.provenance.weights_loaded,
                document.provenance.target_model_constructed,
                document.provenance.target_model_forward,
                document.provenance.remote_code_executed,
            )
        ):
            raise VerificationError("GraphView violates the zero-execution provenance boundary")
        if document.views[0].key != "l0":
            raise VerificationError("GraphView must begin with the L0 model view")
        if any(len(view.nodes) > 200 for view in document.views):
            raise VerificationError("GraphView exceeds the 200 visible-node budget")

        views = {view.key: view for view in document.views}
        if fixture == "qwen3_8_27b":
            required_views = {"l0", "l1_linear_attention", "l1_full_attention"}
        else:
            required_views = {"l0", "l1_dense_dsa", "l1_sparse_dsa_moe"}
        if set(views) != required_views:
            raise VerificationError(f"unexpected {fixture} GraphView set: {sorted(views)!r}")

        if fixture == "glm_5_3_bf16":
            sparse = views["l1_sparse_dsa_moe"]
            kinds = {edge.kind for edge in sparse.edges}
            if GraphEdgeKind.ROUTE not in kinds or GraphEdgeKind.CONTROL not in kinds:
                raise VerificationError("GLM sparse L1 requires route and control edges")
            dsa = next((node for node in sparse.nodes if node.kind == "dsa"), None)
            if dsa is None or not dsa.opaque or dsa.coverage != 0.0:
                raise VerificationError("GLM DSA internals must remain opaque with zero coverage")
        return (
            f"GraphView valid ({fixture}, {len(document.views)} views, "
            f"{sum(len(view.nodes) for view in document.views)} nodes)"
        )

    return validate


def _decision_validator(number: int) -> Callable[[Path], str]:
    pattern = f"DR-{number:04d}-*.md"

    def validate(root: Path) -> str:
        decision_dir = root / "doc" / "decisions"
        matches = sorted(decision_dir.glob(pattern)) if decision_dir.is_dir() else []
        if len(matches) != 1:
            raise VerificationError(
                f"expected exactly one doc/decisions/{pattern}, found {len(matches)}"
            )
        try:
            text = matches[0].read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise VerificationError(f"unreadable Decision Record: {error}") from error
        heading = f"# DR-{number:04d}"
        if not text.startswith(heading):
            raise VerificationError(f"Decision Record must start with {heading!r}")
        if len(text.strip()) < 120:
            raise VerificationError("Decision Record is unexpectedly empty")
        expected_status = "Blocked" if number == 6 else "Accepted"
        if f"状态：{expected_status}" not in text:
            raise VerificationError(f"Decision Record must declare status {expected_status!r}")
        if number == 3 and "退出审计：Partial" not in text:
            raise VerificationError("DR-0003 must preserve its Partial consumer exit audit")
        detail = f"status={expected_status}"
        if number == 3:
            detail += ", consumer-exit=Partial"
        if number == 6:
            detail += ", blocks=M4a"
        return f"Decision Record valid ({matches[0].name}; {detail})"

    return validate


def _markdown_validator(relative_path: str, required_phrase: str) -> Callable[[Path], str]:
    def validate(root: Path) -> str:
        path = _require_file(root, relative_path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise VerificationError(f"unreadable Markdown: {error}") from error
        if not text.lstrip().startswith("#"):
            raise VerificationError("Markdown must start with a heading")
        if required_phrase.lower() not in text.lower():
            raise VerificationError(f"Markdown does not mention {required_phrase!r}")
        if len(text.strip()) < 200:
            raise VerificationError("Markdown asset is unexpectedly empty")
        return f"Markdown asset present ({len(text.splitlines())} lines)"

    return validate


def _asset_checks() -> Tuple[AssetCheck, ...]:
    checks: List[AssetCheck] = []
    for filename in ("model-map.schema.json", "scenario.schema.json", "trace.schema.json"):
        relative_path = f"schemas/{filename}"
        checks.append(
            AssetCheck("M0", f"schema:{filename}", relative_path, _schema_validator(relative_path))
        )
    for number in range(1, 10):
        checks.append(
            AssetCheck(
                "M0",
                f"decision:DR-{number:04d}",
                f"doc/decisions/DR-{number:04d}-*.md",
                _decision_validator(number),
            )
        )
    checks.extend(
        [
            AssetCheck(
                "M0",
                "document:ui-wireframe",
                "doc/ui-wireframe.md",
                _markdown_validator("doc/ui-wireframe.md", "wireframe"),
            ),
            AssetCheck(
                "M0",
                "document:measurement-protocol",
                "doc/measurement-protocol.md",
                _markdown_validator("doc/measurement-protocol.md", "measurement"),
            ),
        ]
    )
    for fixture in STRUCTURE_GOLDENS:
        relative_path = f"tests/golden/structure/{fixture}/model-map.json"
        checks.append(
            AssetCheck(
                "M1",
                f"structure:{fixture}",
                relative_path,
                _structure_validator(relative_path),
            )
        )
    capture_relative_path = f"tests/golden/capture/{CAPTURE_GOLDEN}"
    checks.append(
        AssetCheck(
            "M2",
            f"capture:{CAPTURE_GOLDEN}",
            capture_relative_path,
            _capture_validator(capture_relative_path),
        )
    )
    for filename in COST_GOLDENS:
        relative_path = f"tests/golden/cost/{filename}"
        checks.append(
            AssetCheck(
                "M3",
                f"cost:{filename}",
                relative_path,
                _cost_validator(relative_path),
            )
        )
    graph_schema_path = "schemas/graph-view.schema.json"
    checks.append(
        AssetCheck(
            "M3.5",
            "schema:graph-view.schema.json",
            graph_schema_path,
            _schema_validator(graph_schema_path),
        )
    )
    checks.append(
        AssetCheck(
            "M3.5",
            "decision:DR-0010",
            "doc/decisions/DR-0010-*.md",
            _decision_validator(10),
        )
    )
    for fixture in GRAPH_VIEW_GOLDENS:
        relative_path = f"tests/golden/graph-view/{fixture}/graph-view.json"
        checks.append(
            AssetCheck(
                "M3.5",
                f"graph-view:{fixture}",
                relative_path,
                _graph_view_validator(relative_path, fixture),
            )
        )
    return tuple(checks)


def _included_milestones(requested: str) -> Tuple[str, ...]:
    if requested == "all":
        return MILESTONES
    if requested not in MILESTONES:
        raise ValueError(f"unknown milestone {requested!r}")
    index = MILESTONES.index(requested)
    return MILESTONES[: index + 1]


def verify_milestones(root: Path = DEFAULT_ROOT, milestone: str = "all") -> VerificationReport:
    """Verify a milestone and all of its static prerequisites."""

    root = root.resolve()
    included = set(_included_milestones(milestone))
    results: List[CheckResult] = []
    for check in _asset_checks():
        if check.milestone not in included:
            continue
        try:
            detail = check.validator(root)
            results.append(CheckResult(check.milestone, check.name, check.path_label, True, detail))
        except (OSError, VerificationError) as error:
            results.append(
                CheckResult(check.milestone, check.name, check.path_label, False, str(error))
            )
    return VerificationReport(requested=milestone, results=tuple(results))


def print_report(report: VerificationReport) -> None:
    for milestone in _included_milestones(report.requested):
        print(f"[{milestone}]")
        milestone_results = tuple(
            result for result in report.results if result.milestone == milestone
        )
        if not milestone_results:
            print("  PASS no additional static assets — prerequisites verified above")
        for result in milestone_results:
            marker = "PASS" if result.ok else "FAIL"
            print(f"  {marker} {result.name} — {result.path} — {result.detail}")
    final = "PASSED" if report.ok else "FAILED"
    print(
        f"SUMMARY {final}: requested={report.requested} "
        f"passed={report.passed} failed={report.failed} total={len(report.results)}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify checked-in, offline M0-M3.5 milestone exit assets."
    )
    parser.add_argument(
        "--milestone",
        choices=[*MILESTONES, "all"],
        default="all",
        help="Milestone to verify; M1-M3.5 include all earlier prerequisites.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root (primarily useful for isolated verification tests).",
    )
    args = parser.parse_args(argv)
    report = verify_milestones(args.root, args.milestone)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
