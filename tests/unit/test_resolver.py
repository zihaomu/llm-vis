from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_vis.resolver import ResolutionError, resolve_config


def test_resolves_local_directory_without_importing_code(tmp_path: Path) -> None:
    config = {
        "model_type": "fixture",
        "architectures": ["UntrustedModel"],
        "auto_map": {"AutoModel": "modeling_untrusted.UntrustedModel"},
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "modeling_untrusted.py").write_text(
        "raise RuntimeError('must never be imported')\n", encoding="utf-8"
    )

    resolved = resolve_config(str(model_dir))

    assert resolved.config == config
    assert resolved.revision == "local"
    assert resolved.cache_hit is True
    assert resolved.is_remote is False
    assert len(resolved.sha256) == 64


def test_rejects_unknown_non_path_identifier() -> None:
    with pytest.raises(ResolutionError, match="neither an existing local path"):
        resolve_config("not-a-model-id", local_files_only=True)


def test_local_files_only_reports_cache_miss(tmp_path: Path) -> None:
    with pytest.raises(ResolutionError, match="OFFLINE_CACHE_MISS:.*not present"):
        resolve_config(
            "owner/model",
            revision="missing",
            local_files_only=True,
            cache_dir=tmp_path,
        )


def test_resolves_hugging_face_cache_ref(tmp_path: Path) -> None:
    repo = tmp_path / "models--owner--model"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snapshot / "config.json").write_text(json.dumps({"model_type": "cached"}), encoding="utf-8")

    resolved = resolve_config("owner/model", local_files_only=True, cache_dir=tmp_path)

    assert resolved.revision == "abc123"
    assert resolved.config["model_type"] == "cached"
    assert resolved.cache_hit is True
    assert resolved.source_uri == "hf://owner/model@abc123/config.json"
