from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_milestones.py"
SPEC = importlib.util.spec_from_file_location("llm_vis_verify_milestones", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY_MODULE
SPEC.loader.exec_module(VERIFY_MODULE)
main = VERIFY_MODULE.main
verify_milestones = VERIFY_MODULE.verify_milestones


def _copy_m0_assets(destination: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "schemas", destination / "schemas")
    shutil.copytree(PROJECT_ROOT / "doc", destination / "doc")


def test_all_checked_in_milestone_assets_pass_without_importing_torch() -> None:
    torch_modules_before = {
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    }

    report = verify_milestones(PROJECT_ROOT, "all")

    torch_modules_after = {
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    }
    assert report.ok
    assert report.failed == 0
    assert {result.milestone for result in report.results} == {
        "M0",
        "M1",
        "M2",
        "M3",
        "M3.5",
    }
    assert any(result.name == "decision:DR-0009" for result in report.results)
    assert any(result.name == "decision:DR-0010" for result in report.results)
    assert sum(result.name.startswith("schema:") for result in report.results) == 4
    assert sum(result.name.startswith("structure:") for result in report.results) == 3
    assert sum(result.name.startswith("capture:") for result in report.results) == 1
    assert sum(result.name.startswith("cost:") for result in report.results) == 2
    assert sum(result.name.startswith("graph-view:") for result in report.results) == 2
    assert torch_modules_after == torch_modules_before


def test_selected_milestone_includes_prerequisites_but_not_later_assets(capsys) -> None:
    report = verify_milestones(PROJECT_ROOT, "M2")
    assert report.ok
    assert {result.milestone for result in report.results} == {"M0", "M1", "M2"}
    assert any(result.name.startswith("capture:") for result in report.results)
    assert not any(result.name.startswith("cost:") for result in report.results)

    assert main(["--milestone", "M0", "--root", str(PROJECT_ROOT)]) == 0
    output = capsys.readouterr().out
    assert "[M0]" in output
    assert "PASS schema:model-map.schema.json" in output
    assert "SUMMARY PASSED: requested=M0" in output


def test_missing_required_asset_fails_with_clear_path(tmp_path: Path, capsys) -> None:
    isolated_root = tmp_path / "repository"
    _copy_m0_assets(isolated_root)
    missing = isolated_root / "schemas" / "trace.schema.json"
    missing.unlink()

    report = verify_milestones(isolated_root, "M0")
    failures = [result for result in report.results if not result.ok]
    assert not report.ok
    assert len(failures) == 1
    assert failures[0].path == "schemas/trace.schema.json"
    assert failures[0].detail == "missing file: schemas/trace.schema.json"

    assert main(["--milestone", "M0", "--root", str(isolated_root)]) == 1
    output = capsys.readouterr().out
    assert "FAIL schema:trace.schema.json" in output
    assert "SUMMARY FAILED: requested=M0" in output


def test_decision_status_is_validated_not_only_file_presence(tmp_path: Path) -> None:
    isolated_root = tmp_path / "repository"
    _copy_m0_assets(isolated_root)
    decision = isolated_root / "doc" / "decisions" / "DR-0001-local-tool-and-offline-reports.md"
    decision.write_text(
        decision.read_text(encoding="utf-8").replace("状态：Accepted", "状态：Proposed", 1),
        encoding="utf-8",
    )

    report = verify_milestones(isolated_root, "M0")
    failure = next(result for result in report.results if result.name == "decision:DR-0001")
    assert failure.ok is False
    assert "status 'Accepted'" in failure.detail
