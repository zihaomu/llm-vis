from __future__ import annotations

import json
from pathlib import Path

from llm_vis.resolver import resolve_config

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "configs"


def test_config_fixtures_have_pinned_revision_hash_and_no_weights() -> None:
    for name in ("tiny_dense", "qwen3_8_27b", "glm_5_3_bf16"):
        config_path = FIXTURE_DIR / f"{name}.json"
        provenance = json.loads(
            (FIXTURE_DIR / f"{name}.provenance.json").read_text(encoding="utf-8")
        )
        revision = (
            provenance["revision"]
            if provenance["kind"] == "huggingface_config_subset"
            else None
        )
        resolved = resolve_config(str(config_path), revision=revision)

        assert provenance["weights_included"] is False
        assert provenance["revision"] not in {"", "main", "master"}
        assert provenance["fixture_canonical_sha256"] == resolved.sha256
        if provenance["kind"] == "huggingface_config_subset":
            assert len(provenance["upstream_config_raw_sha256"]) == 64
            assert provenance["source_uri"].startswith("https://huggingface.co/")
            assert provenance["revision"] in provenance["source_uri"]
