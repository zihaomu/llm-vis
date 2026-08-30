"""Generate or verify deterministic M0/M1 structure golden artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

from llm_vis.adapters import build_from_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "tests" / "fixtures" / "configs"
GOLDEN_DIR = ROOT / "tests" / "golden" / "structure"
FIXTURES = ("tiny_dense", "qwen3_8_27b", "glm_5_3_bf16")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(name: str) -> Tuple[str, str, str]:
    config = _load(CONFIG_DIR / f"{name}.json")
    provenance = _load(CONFIG_DIR / f"{name}.provenance.json")
    result = build_from_config(
        config,
        model_id=provenance["model_id"],
        revision=provenance["revision"],
    )
    model_map = result.to_model_map()
    summary = {
        "model_id": provenance["model_id"],
        "revision": provenance["revision"],
        "adapter": result.adapter_name,
        "definition_kinds": dict(
            sorted(Counter(item.kind.value for item in model_map.definitions).items())
        ),
        "semantic_kinds": dict(
            sorted(Counter(item.kind.value for item in model_map.semantic_nodes).items())
        ),
        "layer_count": len(result.layer_strip),
        "layer_labels": "".join(item.label for item in result.layer_strip),
        "metadata": dict(result.metadata),
    }
    return (
        json.dumps(model_map.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        json.dumps([item.__dict__ for item in result.layer_strip], indent=2, sort_keys=True) + "\n",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )


def update(*, check: bool) -> int:
    changed = []
    for name in FIXTURES:
        contents = _render(name)
        paths = (
            GOLDEN_DIR / name / "model-map.json",
            GOLDEN_DIR / name / "layer-strip.json",
            GOLDEN_DIR / name / "summary.json",
        )
        for path, content in zip(paths, contents):
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
    print("structure goldens are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return update(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
