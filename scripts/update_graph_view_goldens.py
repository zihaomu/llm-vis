"""Generate or verify deterministic M3.5 GraphView golden artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from llm_vis.adapters import build_from_config
from llm_vis.graph_view import build_graph_view_document

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "tests" / "fixtures" / "configs"
GOLDEN_DIR = ROOT / "tests" / "golden" / "graph-view"
FIXTURES = ("qwen3_8_27b", "glm_5_3_bf16")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(name: str) -> str:
    config = _load(CONFIG_DIR / f"{name}.json")
    provenance = _load(CONFIG_DIR / f"{name}.provenance.json")
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    document = build_graph_view_document(result, result.to_model_map())
    return (
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def update(*, check: bool) -> int:
    changed = []
    for name in FIXTURES:
        path = GOLDEN_DIR / name / "graph-view.json"
        content = _render(name)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            changed.append(path)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
    if changed:
        action = "out of date" if check else "updated"
        for path in changed:
            print(f"{action}: {path.relative_to(ROOT)}")
        return 1 if check else 0
    print("graph-view goldens are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return update(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
